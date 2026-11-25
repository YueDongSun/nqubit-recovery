"""Offline-trained feedback model interface.

The experiment loop only calls inference. Training is an explicit offline step
over saved FeedbackSample JSON files.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml_recon.quantum_representation.tomography_features import (
    RHO_PAIR_DISTANCE_FEATURE_NAMES,
    entanglement_signature_feature_names,
    entanglement_signature_features,
    geometric_phase_feature_names,
    geometric_phase_features,
    rho_pair_distance_features,
)

try:
    from ml_recon.quantum_representation.torch_model import (
        _select_training_device,
        _training_accelerator,
        _training_device_name,
    )
except Exception:
    _select_training_device = None
    _training_accelerator = None
    _training_device_name = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from system_control.experiment_control.models import (
    CalibrationDelta,
    FeedbackPrediction,
    FeedbackSample,
)
from system_control.experiment_control.model_paths import display_feedback_model_path


if TORCH_AVAILABLE:

    class NeuralFeedbackGenerator(nn.Module):
        """Small generator mapping experiment features to normalized delta vectors."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, latent_dim: int = 48):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.08),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, features):
            return self.net(features)


    class NeuralFeedbackInverse(nn.Module):
        """Inverse consistency head: proposed delta should explain the input residual features."""

        def __init__(self, output_dim: int, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, deltas):
            return self.net(deltas)


    class NeuralFeedbackDiscriminator(nn.Module):
        """Lightweight adversarial critic over normalized delta vectors."""

        def __init__(self, output_dim: int, hidden_dim: int = 96):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, deltas):
            return self.net(deltas).squeeze(-1)


    class FeedbackResidualAttentionBlock(nn.Module):
        """Residual spatial block with a lightweight channel gate."""

        def __init__(self, channels: int):
            super().__init__()
            hidden = max(4, int(channels) // 4)
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.gate = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            y = F.silu(self.conv1(x))
            y = self.conv2(y)
            pooled = y.mean(dim=(-2, -1), keepdim=True)
            return F.silu(x + y * self.gate(pooled))


    class FeedbackPhaseMapEncoder(nn.Module):
        """Conv encoder over 15x15/16x16 W_CH1 phase correction maps."""

        def __init__(self, latent_dim: int = 48, base_channels: int = 16, num_blocks: int = 3):
            super().__init__()
            channels = int(base_channels)
            self.stem = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.blocks = nn.Sequential(*[
                FeedbackResidualAttentionBlock(channels)
                for _ in range(max(1, int(num_blocks)))
            ])
            self.proj = nn.Sequential(
                nn.Linear(channels * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )

        def forward(self, phase_map):
            z = self.blocks(self.stem(phase_map))
            mean_pool = z.mean(dim=(-2, -1))
            rms_pool = torch.sqrt(torch.clamp((z ** 2).mean(dim=(-2, -1)), min=1e-12))
            return self.proj(torch.cat([mean_pool, rms_pool], dim=-1))


    class HybridFeedbackPolicy(nn.Module):
        """Image + physics fusion policy for normalized feedback delta vectors."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            phase_map_size: int,
            hidden_dim: int = 128,
            latent_dim: int = 48,
            image_channels: int = 16,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.phase_map_size = int(phase_map_size)
            self.physics_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )
            self.image_branch = FeedbackPhaseMapEncoder(latent_dim, image_channels)
            self.fusion_head = nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, physics_features, phase_map):
            if phase_map.ndim == 3:
                phase_map = phase_map.unsqueeze(1)
            physics_latent = self.physics_branch(physics_features)
            image_latent = self.image_branch(phase_map)
            return self.fusion_head(torch.cat([physics_latent, image_latent], dim=-1))


    class HybridFeedbackInverse(nn.Module):
        """Inverse consistency head for the hybrid feedback policy."""

        def __init__(self, output_dim: int, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, deltas):
            return self.net(deltas)

else:

    class NeuralFeedbackGenerator:  # type: ignore[no-redef]
        pass

    class NeuralFeedbackInverse:  # type: ignore[no-redef]
        pass

    class NeuralFeedbackDiscriminator:  # type: ignore[no-redef]
        pass

    class HybridFeedbackPolicy:  # type: ignore[no-redef]
        pass

    class HybridFeedbackInverse:  # type: ignore[no-redef]
        pass


class TrainedFeedbackModel:
    """Small-sample feedback model with physics-aware feature plumbing."""

    def __init__(self, system_type: str = "4qubit"):
        if system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        self.system_type = system_type
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.neural_checkpoint: Optional[Dict[str, Any]] = None
        self.neural_generator = None
        self.neural_inverse = None
        self.hybrid_checkpoint: Optional[Dict[str, Any]] = None
        self.hybrid_model = None
        self.hybrid_inverse = None
        self.physics_transfer_checkpoint: Optional[Dict[str, Any]] = None
        self.physics_transfer_model = None  # restored PhysicsTransferGNN instance
        self.phase_map_feature_mode = "pyramid"
        # v1 keeps old checkpoints byte-for-byte feature compatible.  v2 is
        # selected explicitly for operation-point/noise-aware training and has
        # no rho-derived g_si or trace-derived double-excitation proxy.
        self.feature_schema_version = 1
        self.phase_map_pca_mean: Optional[np.ndarray] = None
        self.phase_map_pca_components: Optional[np.ndarray] = None

    def train(self, samples_dir: str, config: Optional[Dict[str, Any]] = None) -> str:
        samples = self._load_samples(samples_dir)
        config = config or {}
        model_path = config.get("model_path") or display_feedback_model_path(self.system_type)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model_kind = str(config.get("model_kind", "auto") or "auto").lower()
        if model_kind == "auto":
            neural_mode = str(config.get("neural_mode", "auto") or "auto").lower()
            model_kind = "linear" if neural_mode in {"off", "false", "none", "linear"} else "neural_mlp"
        if model_kind not in {"linear", "neural_mlp", "hybrid_image_physics", "physics_transfer_gnn"}:
            raise ValueError("model_kind must be linear, neural_mlp, hybrid_image_physics, or physics_transfer_gnn")
        self.phase_map_feature_mode = str(config.get("phase_map_feature_mode", "pyramid") or "pyramid").lower()
        self.feature_schema_version = int(config.get("feature_schema_version", 1))
        if self.feature_schema_version not in {1, 2}:
            raise ValueError("feature_schema_version must be 1 or 2")
        if self.phase_map_feature_mode not in {"raw", "pyramid", "pca"}:
            raise ValueError("phase_map_feature_mode must be raw, pyramid, or pca")

        labelled = [sample for sample in samples if sample.accepted_delta]
        if labelled:
            self._fit_phase_map_pca(labelled, config)
            x = np.vstack([self._feature_vector(sample.rho_target, sample.rho_measured, {
                "system_type": sample.system_type,
                "target_state_params": sample.target_state_params,
                "retrieval_metrics": sample.retrieval_metrics,
                "noise_context": sample.noise_context,
                "feature_schema_version": sample.feature_schema_version,
            }) for sample in labelled])
            y = np.vstack([self._label_vector(sample) for sample in labelled])
            self.feature_mean = np.mean(x, axis=0)
            self.feature_scale = np.std(x, axis=0) + 1e-9
            x_norm = self._normalize_features(x)
            x_aug = np.column_stack([x_norm, np.ones(x_norm.shape[0])])
            ridge_alpha = float(config.get("ridge_alpha", 1.0))
            reg = ridge_alpha * np.eye(x_aug.shape[1], dtype=float)
            reg[-1, -1] = 0.0
            coeff = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
            self.weights = coeff[:-1].T
            self.bias = coeff[-1]
            self.hybrid_checkpoint = None
            self.hybrid_model = None
            self.hybrid_inverse = None
            self.physics_transfer_checkpoint = None
            self.physics_transfer_model = None
            if model_kind == "hybrid_image_physics":
                self._train_hybrid_head(labelled, x_norm, y, config)
            elif model_kind == "physics_transfer_gnn":
                self._train_physics_transfer_head(samples, labelled, config)
            if model_kind == "neural_mlp" or str(config.get("neural_mode", "off") or "off").lower() in {
                "auto",
                "neural",
                "adversarial",
                "force",
            }:
                self._train_neural_head(x_norm, y, config)
            else:
                self.neural_checkpoint = None
                self.neural_generator = None
                self.neural_inverse = None
        else:
            self.weights = None
            self.bias = None
            self.feature_mean = None
            self.feature_scale = None
            self.neural_checkpoint = None
            self.neural_generator = None
            self.neural_inverse = None
            self.hybrid_checkpoint = None
            self.hybrid_model = None
            self.hybrid_inverse = None
            self.physics_transfer_checkpoint = None
            self.physics_transfer_model = None
            self.phase_map_pca_mean = None
            self.phase_map_pca_components = None

        self.metadata = {
            "system_type": self.system_type,
            "model_kind": model_kind,
            "num_samples": len(samples),
            "num_labelled_samples": len(labelled),
            "trained_linear_head": bool(labelled),
            "trained_neural_head": bool(self.neural_checkpoint),
            "trained_hybrid_head": bool(self.hybrid_checkpoint),
            "trained_physics_transfer_head": bool(self.physics_transfer_checkpoint),
            "neural_feedback": self._neural_metadata(),
            "hybrid_feedback": self._hybrid_metadata(),
            "physics_transfer_feedback": self._physics_transfer_metadata(),
            "linear_head_regularization": float(config.get("ridge_alpha", 1.0)),
            "feature_normalization": bool(labelled),
            "phase_map_feature_mode": self.phase_map_feature_mode,
            "phase_map_pca_components": 0 if self.phase_map_pca_components is None else int(self.phase_map_pca_components.shape[0]),
            "physics_prior_blend": float(config.get("physics_prior_blend", 0.6 if labelled else 1.0)),
            "inference_fallback": None if labelled else "physics_constrained_heuristic",
            "feature_schema": self._feature_schema(self.system_type),
            "feature_schema_version": self.feature_schema_version,
            "output_schema": self._output_schema(self.system_type),
            "physics_constraint_settings": self._physics_constraint_settings(self.system_type),
            "physics_priors": [
                "multi_frequency_rf_power_compression",
                "retrieval_map_weighting",
                *( ["gsi_proxy", "double_excitation_proxy"] if self.feature_schema_version == 1 else ["operation_point_noise_context_v2"] ),
                "coupled_wave_precompute_proxy",
                "pointwise_wch1_phase_map",
                "bounded_phase_amp_projection",
            ],
        }
        self.save(model_path)
        return model_path

    def load(self, model_path: str) -> "TrainedFeedbackModel":
        with open(model_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.system_type = payload["system_type"]
        self.metadata = payload.get("metadata", {})
        self.feature_schema_version = int(self.metadata.get("feature_schema_version", 1))
        self.weights = None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float)
        self.bias = None if payload.get("bias") is None else np.asarray(payload["bias"], dtype=float)
        self.feature_mean = None if payload.get("feature_mean") is None else np.asarray(payload["feature_mean"], dtype=float)
        self.feature_scale = None if payload.get("feature_scale") is None else np.asarray(payload["feature_scale"], dtype=float)
        self.phase_map_feature_mode = str(
            payload.get("phase_map_feature_mode")
            or self.metadata.get("phase_map_feature_mode")
            or "pyramid"
        )
        self.phase_map_pca_mean = None if payload.get("phase_map_pca_mean") is None else np.asarray(payload["phase_map_pca_mean"], dtype=float)
        self.phase_map_pca_components = None if payload.get("phase_map_pca_components") is None else np.asarray(payload["phase_map_pca_components"], dtype=float)
        self.neural_checkpoint = payload.get("neural_checkpoint")
        self.hybrid_checkpoint = payload.get("hybrid_checkpoint")
        self.physics_transfer_checkpoint = payload.get("physics_transfer_checkpoint")
        self._restore_neural_head()
        self._restore_hybrid_head()
        self._restore_physics_transfer_head()
        return self

    def save(self, model_path: str) -> None:
        payload = {
            "system_type": self.system_type,
            "metadata": self.metadata,
            "weights": None if self.weights is None else self.weights.tolist(),
            "bias": None if self.bias is None else self.bias.tolist(),
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_scale": None if self.feature_scale is None else self.feature_scale.tolist(),
            "phase_map_feature_mode": self.phase_map_feature_mode,
            "feature_schema_version": self.feature_schema_version,
            "phase_map_pca_mean": None if self.phase_map_pca_mean is None else self.phase_map_pca_mean.tolist(),
            "phase_map_pca_components": None if self.phase_map_pca_components is None else self.phase_map_pca_components.tolist(),
            "neural_checkpoint": self.neural_checkpoint,
            "hybrid_checkpoint": self.hybrid_checkpoint,
            "physics_transfer_checkpoint": self.physics_transfer_checkpoint,
        }
        with open(model_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def predict(self, target_rho: np.ndarray, measured_rho: np.ndarray, context: Dict[str, Any]) -> FeedbackPrediction:
        system_type = context.get("system_type", self.system_type)
        n_modes = 3 if system_type == "3qubit" else 4
        size = 15 if system_type == "3qubit" else 16
        requested_phase_strategy = str(
            context.get("phase_feedback_strategy", "w_subspace_phase") or "w_subspace_phase"
        )
        phase_strategy = self._normalise_phase_feedback_strategy(requested_phase_strategy)
        features = self._feature_vector(target_rho, measured_rho, context)

        if self.physics_transfer_model is not None and self.physics_transfer_checkpoint is not None:
            try:
                raw_model = self._predict_physics_transfer_raw(target_rho, measured_rho, context, n_modes)
                output_mode = str(self.physics_transfer_checkpoint.get("output_mode", "legacy_vector_v1"))
                prior_blend = float(self.metadata.get("physics_prior_blend", 0.0) or 0.0)
                if prior_blend > 0.0 and output_mode != "wch1_phase_map_v1":
                    raw_prior = self._align_raw_output(
                        self._heuristic_output(target_rho, measured_rho, context, n_modes),
                        raw_model.size,
                    )
                    raw = (1.0 - prior_blend) * raw_model + prior_blend * raw_prior
                else:
                    raw = raw_model
                calibrated_confidence = self.physics_transfer_checkpoint.get("confidence")
                confidence = 0.0 if calibrated_confidence is None else float(calibrated_confidence)
                source = "physics_transfer_gnn"
            except Exception:  # noqa: BLE001
                # Fall through to hybrid/neural/linear chain on any inference error.
                raw = None
                confidence = None
                source = None
        else:
            raw = None
            confidence = None
            source = None

        if raw is None:
            if self.hybrid_model is not None and self.hybrid_checkpoint is not None:
                raw_model = self._predict_hybrid_raw(features, context, size)
                prior_blend = float(self.metadata.get("physics_prior_blend", 0.0) or 0.0)
                if prior_blend > 0.0:
                    raw_prior = self._align_raw_output(
                        self._heuristic_output(target_rho, measured_rho, context, n_modes),
                        raw_model.size,
                    )
                    raw = (1.0 - prior_blend) * raw_model + prior_blend * raw_prior
                else:
                    raw = raw_model
                confidence = float(self.hybrid_checkpoint.get("confidence", 0.82))
                source = self.hybrid_checkpoint.get("training_mode", "hybrid_image_physics_feedback")
            elif self.neural_generator is not None and self.neural_checkpoint is not None:
                raw_model = self._predict_neural_raw(features)
                prior_blend = float(self.metadata.get("physics_prior_blend", 0.0) or 0.0)
                if prior_blend > 0.0:
                    raw_prior = self._align_raw_output(
                        self._heuristic_output(target_rho, measured_rho, context, n_modes),
                        raw_model.size,
                    )
                    raw = (1.0 - prior_blend) * raw_model + prior_blend * raw_prior
                else:
                    raw = raw_model
                confidence = float(self.neural_checkpoint.get("confidence", 0.80))
                source = self.neural_checkpoint.get("training_mode", "neural_bidirectional_feedback")
            elif self.weights is not None and self.bias is not None:
                raw_linear = self.weights @ self._align_features_for_weights(self._normalize_features(features)) + self.bias
                prior_blend = float(self.metadata.get("physics_prior_blend", 0.0) or 0.0)
                if prior_blend > 0.0:
                    raw_prior = self._align_raw_output(
                        self._heuristic_output(target_rho, measured_rho, context, n_modes),
                        raw_linear.size,
                    )
                    raw = (1.0 - prior_blend) * raw_linear + prior_blend * raw_prior
                else:
                    raw = raw_linear
                confidence = 0.75
                source = "trained_linear_head"
            else:
                raw = self._heuristic_output(target_rho, measured_rho, context, n_modes)
                confidence = 0.35
                source = "physics_constrained_heuristic"

        physics_output_mode = (
            str(self.physics_transfer_checkpoint.get("output_mode", "legacy_vector_v1"))
            if source == "physics_transfer_gnn" and self.physics_transfer_checkpoint is not None
            else "legacy_vector_v1"
        )
        if physics_output_mode == "wch1_phase_map_v1":
            expected = size * size
            if np.asarray(raw).size != expected:
                raise ValueError(f"physics-transfer phase map width {np.asarray(raw).size} != {expected}")
            phase_limit = float(self.physics_transfer_checkpoint.get("phase_limit", 0.20))
            deltas = [CalibrationDelta(
                target_channel="W_CH1",
                matrix_type="phase",
                delta=np.asarray(raw, dtype=float).reshape(size, size) * phase_limit,
                limit=phase_limit,
                reason="structured physics-transfer residual phase map",
            )]
        else:
            deltas = self._raw_to_deltas(raw, system_type, size, n_modes, context.get("target_state_params"))
        phase_resolver = None
        if context.get("enable_wch1_phase_resolver", True):
            phase_resolver = self._wch1_phase_resolver_delta(
                target_rho,
                measured_rho,
                system_type,
                size,
                n_modes,
                context,
            )
            if phase_resolver is not None:
                deltas = self._merge_wch1_phase_resolver(deltas, phase_resolver)
        safety = self._feedback_safety_diagnostics(
            target_rho,
            measured_rho,
            system_type,
            n_modes,
            context,
            confidence,
            source,
            phase_resolver,
            phase_strategy,
        )
        confidence = float(min(confidence, safety["confidence"]))
        safe_deltas = deltas
        if not safety["safe"]:
            safe_deltas = self._unsafe_deltas(deltas, safety["unsafe_reason"])
            deltas = safe_deltas
        feature_diagnostics = self._entanglement_phase_feature_diagnostics(
            target_rho,
            measured_rho,
            n_modes,
            phase_strategy,
            context,
        )
        heuristic_gain_proxy = float(np.clip(features[0] * 0.05, -0.02, 0.05))
        return FeedbackPrediction(
            deltas=deltas,
            safe_deltas=safe_deltas,
            confidence=confidence,
            heuristic_gain_proxy=heuristic_gain_proxy,
            prediction_source=source,
            physics_diagnostics={
                "source": source,
                "heuristic_gain_proxy": heuristic_gain_proxy,
                "phase_feedback_strategy": phase_strategy,
                "requested_phase_feedback_strategy": requested_phase_strategy,
                "rho_diff_role": "evidence_prior_only",
                "feature_schema": feature_diagnostics["feature_schema"],
                "entanglement_signature": feature_diagnostics["entanglement_signature"],
                "geometric_phase_features": feature_diagnostics["geometric_phase_features"],
                "classifier_enhanced": feature_diagnostics["classifier_enhanced"],
                "feature_vector": features.tolist(),
                "model_metadata": self.metadata,
                "neural_feedback": self._neural_prediction_diagnostics(features),
                "hybrid_feedback": self._hybrid_prediction_diagnostics(features, context, size),
                "wch1_phase_resolver": None if phase_resolver is None else phase_resolver["diagnostics"],
                "feedback_safety": safety,
                "physicality_ok": safety["physicality_ok"],
                "trace_ok": safety["trace_ok"],
                "psd_ok": safety["psd_ok"],
                "correctable_fraction": safety["correctable_fraction"],
                "coherence_score": safety["coherence_score"],
                "confidence": safety["confidence"],
                "unsafe_reason": safety["unsafe_reason"],
            },
        )

    def _normalise_phase_feedback_strategy(self, requested: str) -> str:
        """Keep old config names compatible without making rho-diff a decision path."""

        strategy = str(requested or "w_subspace_phase").lower()
        if strategy == "legacy_diff_guarded":
            return "w_subspace_phase"
        return strategy

    def _select_torch_device(self, requested: str = "auto"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for neural feedback training")
        if _select_training_device is not None:
            return _select_training_device(requested)
        requested = str(requested or "auto").lower()
        if requested not in {"auto", "cuda", "directml", "cpu"}:
            raise ValueError("device must be auto, cuda, directml, or cpu")
        if requested in {"auto", "cuda"} and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if requested == "directml":
            raise RuntimeError("DirectML was requested but the DirectML helper is unavailable")
        return torch.device("cpu")

    def _device_metadata(self, device) -> Dict[str, Any]:
        accelerator = None
        device_name = None
        if _training_accelerator is not None:
            accelerator = _training_accelerator(device)
        if _training_device_name is not None:
            device_name = _training_device_name(device)
        return {
            "device": str(device),
            "accelerator": accelerator or ("cuda" if str(device).startswith("cuda") else "cpu"),
            "device_name": device_name,
            "torch_version": getattr(torch, "__version__", None) if TORCH_AVAILABLE else None,
        }

    def _train_hybrid_head(
        self,
        samples: List[FeedbackSample],
        x_norm: np.ndarray,
        y: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        """Train the hybrid phase-map image + physics feedback policy."""

        self.hybrid_checkpoint = None
        self.hybrid_model = None
        self.hybrid_inverse = None
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for hybrid_image_physics feedback training")

        seed = int(config.get("seed", 1234))
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        device = self._select_torch_device(str(config.get("device", "auto") or "auto"))
        input_dim = int(x_norm.shape[1])
        output_dim = int(y.shape[1])
        size = 15 if self.system_type == "3qubit" else 16
        n_modes = 3 if self.system_type == "3qubit" else 4
        base_size = n_modes * 4 + 2
        hidden_dim = int(config.get("hybrid_hidden_dim", config.get("neural_hidden_dim", 128)))
        latent_dim = int(config.get("hybrid_latent_dim", 48))
        image_channels = int(config.get("hybrid_image_channels", 16))
        epochs = int(config.get("hybrid_epochs", 12))
        learning_rate = float(config.get("hybrid_learning_rate", 1e-3))
        cycle_weight = float(config.get("cycle_consistency_weight", 0.10))
        phase_weight = float(config.get("phase_map_loss_weight", 0.45))
        vector_weight = float(config.get("vector_delta_loss_weight", 0.35))
        bounds_weight = float(config.get("bounds_loss_weight", 0.02))
        # V5 upgrade: circular phase loss (1 - cos(pred - target)) is smooth at
        # the +/-pi seam, unlike Smooth-L1 which gives a ~2pi error there.
        # Default "circular"; set phase_loss_kind="smooth_l1" for the legacy path.
        phase_loss_kind = str(config.get("phase_loss_kind", "circular")).lower()
        val_fraction = float(config.get("validation_fraction", 0.2))
        lr_patience = int(config.get("lr_scheduler_patience", 3))
        early_patience = int(config.get("early_stopping_patience", max(4, epochs // 3)))
        batch_size = max(1, min(int(config.get("hybrid_batch_size", 128)), int(x_norm.shape[0])))

        phase_maps = np.stack([
            self._hybrid_phase_map_input_from_sample(sample, size)
            for sample in samples
        ]).astype(np.float32)
        x_tensor = torch.tensor(x_norm.astype(np.float32), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(np.clip(y, -1.0, 1.0).astype(np.float32), dtype=torch.float32, device=device)
        map_tensor = torch.tensor(phase_maps[:, None, :, :], dtype=torch.float32, device=device)

        model = HybridFeedbackPolicy(input_dim, output_dim, size, hidden_dim, latent_dim, image_channels).to(device)
        inverse = HybridFeedbackInverse(output_dim, input_dim, hidden_dim).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(inverse.parameters()),
            lr=learning_rate,
            weight_decay=float(config.get("hybrid_weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, lr_patience),
        )

        n = int(x_norm.shape[0])
        indices = rng.permutation(n)
        val_count = int(round(n * val_fraction)) if n >= 8 else 0
        val_count = min(max(val_count, 0), max(n - 1, 0))
        val_idx_np = indices[:val_count]
        train_idx_np = indices[val_count:] if val_count else indices
        train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=device) if val_count else None
        phase_slice = slice(base_size, base_size + size * size)

        history: List[Dict[str, float]] = []
        best_state = None
        best_loss = None
        best_epoch = 0
        stale_epochs = 0
        epochs_completed = 0

        for epoch in range(max(1, epochs)):
            model.train()
            inverse.train()
            order_np = rng.permutation(train_idx_np)
            order = torch.tensor(order_np, dtype=torch.long, device=device)
            epoch_loss = 0.0
            epoch_supervised = 0.0
            epoch_phase = 0.0
            epoch_vector = 0.0
            epoch_cycle = 0.0
            epoch_bounds = 0.0
            seen = 0
            for start in range(0, int(order.shape[0]), batch_size):
                idx = order[start:start + batch_size]
                xb = x_tensor[idx]
                yb = y_tensor[idx]
                mb = map_tensor[idx]
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb, mb)
                reconstructed = inverse(pred)
                supervised = F.smooth_l1_loss(pred, yb)
                vector_loss = F.smooth_l1_loss(pred[:, :base_size], yb[:, :base_size])
                if phase_loss_kind == "smooth_l1":
                    phase_loss = F.smooth_l1_loss(pred[:, phase_slice], yb[:, phase_slice])
                else:
                    phase_delta = pred[:, phase_slice] - yb[:, phase_slice]
                    phase_loss = torch.mean(1.0 - torch.cos(phase_delta))
                cycle_loss = F.mse_loss(reconstructed, xb)
                bounds_loss = torch.mean(torch.relu(torch.abs(pred) - 1.0) ** 2)
                loss = (
                    supervised
                    + vector_weight * vector_loss
                    + phase_weight * phase_loss
                    + cycle_weight * cycle_loss
                    + bounds_weight * bounds_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(inverse.parameters()), 1.0)
                optimizer.step()
                count = int(xb.shape[0])
                epoch_loss += float(loss.item()) * count
                epoch_supervised += float(supervised.item()) * count
                epoch_phase += float(phase_loss.item()) * count
                epoch_vector += float(vector_loss.item()) * count
                epoch_cycle += float(cycle_loss.item()) * count
                epoch_bounds += float(bounds_loss.item()) * count
                seen += count

            model.eval()
            inverse.eval()
            eval_idx = val_idx if val_idx is not None else train_idx
            with torch.no_grad():
                pred_eval = model(x_tensor[eval_idx], map_tensor[eval_idx])
                inv_eval = inverse(pred_eval)
                y_eval = y_tensor[eval_idx]
                val_supervised = F.smooth_l1_loss(pred_eval, y_eval)
                val_vector = F.smooth_l1_loss(pred_eval[:, :base_size], y_eval[:, :base_size])
                if phase_loss_kind == "smooth_l1":
                    val_phase = F.smooth_l1_loss(pred_eval[:, phase_slice], y_eval[:, phase_slice])
                else:
                    val_phase = torch.mean(
                        1.0 - torch.cos(pred_eval[:, phase_slice] - y_eval[:, phase_slice])
                    )
                val_cycle = F.mse_loss(inv_eval, x_tensor[eval_idx])
                val_bounds = torch.mean(torch.relu(torch.abs(pred_eval) - 1.0) ** 2)
                val_score_tensor = (
                    val_supervised
                    + vector_weight * val_vector
                    + phase_weight * val_phase
                    + cycle_weight * val_cycle
                    + bounds_weight * val_bounds
                )
                val_score = float(val_score_tensor.item())
            scheduler.step(val_score)
            epochs_completed = epoch + 1
            row = {
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss / max(seen, 1),
                "train_supervised_loss": epoch_supervised / max(seen, 1),
                "train_vector_loss": epoch_vector / max(seen, 1),
                "train_phase_map_loss": epoch_phase / max(seen, 1),
                "train_cycle_loss": epoch_cycle / max(seen, 1),
                "train_bounds_loss": epoch_bounds / max(seen, 1),
                "validation_loss": val_score,
                "validation_supervised_loss": float(val_supervised.item()),
                "validation_vector_loss": float(val_vector.item()),
                "validation_phase_map_loss": float(val_phase.item()),
                "validation_cycle_loss": float(val_cycle.item()),
                "validation_bounds_loss": float(val_bounds.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if best_loss is None or val_score < best_loss - 1e-7:
                best_loss = val_score
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = {
                    "policy_state_dict": _state_dict_to_json(model.state_dict()),
                    "inverse_state_dict": _state_dict_to_json(inverse.state_dict()),
                }
            else:
                stale_epochs += 1
                if early_patience > 0 and stale_epochs >= early_patience:
                    break

        if best_state is not None:
            model.load_state_dict(_state_dict_from_json(best_state["policy_state_dict"], device=device))
            inverse.load_state_dict(_state_dict_from_json(best_state["inverse_state_dict"], device=device))

        confidence = float(np.clip(0.88 - 0.22 * np.sqrt(max(best_loss or 0.0, 0.0)), 0.48, 0.92))
        self.hybrid_checkpoint = {
            **(best_state or {
                "policy_state_dict": _state_dict_to_json(model.state_dict()),
                "inverse_state_dict": _state_dict_to_json(inverse.state_dict()),
            }),
            "architecture": "phase_map_conv_residual_attention_plus_physics_mlp",
            "model_kind": "hybrid_image_physics",
            "training_mode": "hybrid_image_physics_feedback",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "phase_map_size": size,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "image_channels": image_channels,
            "epochs": epochs,
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_loss": float(best_loss or 0.0),
            "converged": bool(best_epoch > 0 and (stale_epochs < early_patience or epochs_completed < epochs)),
            "confidence": confidence,
            "loss_weights": {
                "phase_map": phase_weight,
                "vector_delta": vector_weight,
                "inverse_cycle": cycle_weight,
                "bounds": bounds_weight,
            },
            "physics_constraint_settings": self._physics_constraint_settings(self.system_type),
            "history": history,
            **self._device_metadata(device),
        }
        model.eval()
        inverse.eval()
        self.hybrid_model = model
        self.hybrid_inverse = inverse

    def _train_physics_transfer_head(
        self,
        all_samples: List[FeedbackSample],
        labelled_samples: List[FeedbackSample],
        config: Dict[str, Any],
    ) -> None:
        """Two-stage A+C training driver.

        Pretrain on synthetic samples (auxiliary physics_aux loss), finetune on
        labelled real samples (delta-only loss, encoder frozen).

        Synthetic samples provide full-map oracle actions. Real NPY snapshots with
        no transition label are deliberately excluded: they belong to forward
        system identification, not inverse-control supervision. Only explicitly
        labelled real transitions may fine-tune the residual controller.
        """
        try:
            from ml_recon.physics_transfer import (
                PhysicsTransferTrainConfig,
                TORCH_GEOMETRIC_AVAILABLE,
                train_physics_transfer_model,
            )
        except ImportError:
            self.physics_transfer_checkpoint = None
            self.physics_transfer_model = None
            return
        if not TORCH_GEOMETRIC_AVAILABLE:
            self.physics_transfer_checkpoint = None
            self.physics_transfer_model = None
            return

        # Build per-sample shim mirrors of NoisySample dataclass shape.
        from dataclasses import dataclass

        @dataclass
        class _Shim:
            target_rho: np.ndarray
            measured_rho: np.ndarray
            scalar_delta: np.ndarray
            classifier_physics_aux: np.ndarray
            physics_aux_gt: np.ndarray
            classifier_aux_missing: bool

        n_modes = 3 if self.system_type == "3qubit" else 4
        phase_grid_size = 15 if self.system_type == "3qubit" else 16
        scalar_dim = phase_grid_size * phase_grid_size
        phase_limit = float(config.get("wch1_phase_limit", 0.20))
        classifier_feat_dim = int(config.get("classifier_feat_dim", 26))

        def _label_vector(sample: FeedbackSample) -> np.ndarray:
            matches = [
                delta for delta in (sample.accepted_delta or [])
                if delta.accepted and delta.target_channel == "W_CH1" and delta.matrix_type == "phase"
            ]
            if not matches:
                raise ValueError(f"sample {sample.sample_id} has no accepted W_CH1 phase transition label")
            values = np.asarray(matches[0].clipped_delta(), dtype=np.float32)
            if values.shape != (phase_grid_size, phase_grid_size):
                raise ValueError(
                    f"sample {sample.sample_id} W_CH1 phase shape {values.shape} "
                    f"!= {(phase_grid_size, phase_grid_size)}"
                )
            return np.clip(values.reshape(-1) / phase_limit, -1.0, 1.0)

        def _phys_aux_from_sample(sample: FeedbackSample) -> tuple[np.ndarray, bool]:
            diag = (sample.retrieval_metrics or {}).get("diagnostics", {}) if hasattr(sample, "retrieval_metrics") else {}
            clf = diag.get("classifier_diagnostics") if isinstance(diag, dict) else None
            phys = None
            if isinstance(clf, dict):
                phys = clf.get("physics_aux") or clf.get("physics_targets")
            if phys is None:
                return np.zeros(classifier_feat_dim, dtype=np.float32), True
            arr = np.asarray(phys, dtype=np.float32).reshape(-1)
            if arr.size != classifier_feat_dim:
                raise ValueError(
                    f"sample {sample.sample_id} classifier physics_aux width {arr.size} "
                    f"!= checkpoint schema {classifier_feat_dim}; truncation is forbidden"
                )
            return arr, False

        def _make_shim(sample: FeedbackSample) -> "_Shim":
            physics_aux, missing = _phys_aux_from_sample(sample)
            return _Shim(
                target_rho=np.asarray(sample.rho_target, dtype=complex),
                measured_rho=np.asarray(sample.rho_measured, dtype=complex),
                scalar_delta=_label_vector(sample),
                classifier_physics_aux=physics_aux,
                physics_aux_gt=physics_aux,
                classifier_aux_missing=missing,
            )

        labelled_ids = {sample.sample_id for sample in labelled_samples}
        supervised = [sample for sample in all_samples if sample.sample_id in labelled_ids]
        synthetic = [sample for sample in supervised if sample.data_provenance == "synthetic"]
        real = [sample for sample in supervised if sample.data_provenance == "real"]
        unknown = [sample for sample in supervised if sample.data_provenance == "unknown"]
        if bool(config.get("allow_legacy_unknown_provenance", False)):
            real.extend(unknown)

        pretrain_shims = [_make_shim(s) for s in synthetic]
        finetune_shims = [_make_shim(s) for s in real]
        if not pretrain_shims and not finetune_shims:
            raise ValueError(
                "physics-transfer training requires provenance-tagged synthetic oracle maps "
                "or real FeedbackTransition labels"
            )

        cfg = PhysicsTransferTrainConfig(
            pretrain_epochs=int(config.get("pretrain_epochs", 15)),
            finetune_epochs=int(config.get("finetune_epochs", 20)),
            batch_size=int(config.get("batch_size", 32)),
            head_lr=float(config.get("learning_rate", 1e-3)),
            encoder_lr=float(config.get("encoder_lr", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            physics_aux_weight=float(config.get("physics_aux_weight", 0.2)),
            classifier_feat_dim=classifier_feat_dim,
            output_dim=scalar_dim,
            phase_grid_size=phase_grid_size,
            dct_rank=int(config.get("phase_dct_rank", 4)),
            phase_tv_weight=float(config.get("phase_tv_weight", 1e-3)),
            phase_sparse_weight=float(config.get("phase_sparse_weight", 1e-4)),
        )
        device = self._select_torch_device(str(config.get("device", "auto")))

        ckpt = train_physics_transfer_model(
            pretrain_samples=pretrain_shims,
            finetune_samples=finetune_shims,
            num_qubits=n_modes,
            config=cfg,
            device=device,
        )
        ckpt["phase_limit"] = phase_limit
        ckpt["provenance_audit"] = {
            "synthetic_supervised": len(synthetic),
            "real_transition_supervised": len(real),
            "real_unlabelled_system_id_only": sum(
                sample.data_provenance == "real" and not sample.accepted_delta for sample in all_samples
            ),
            "unknown_excluded": 0 if bool(config.get("allow_legacy_unknown_provenance", False)) else len(unknown),
            "legacy_unknown_override": bool(config.get("allow_legacy_unknown_provenance", False)),
            "classifier_aux_missing": sum(shim.classifier_aux_missing for shim in pretrain_shims + finetune_shims),
        }
        minimum_pretrain = int(config.get("minimum_synthetic_pretrain_samples", 100_000))
        ckpt["minimum_synthetic_pretrain_samples"] = minimum_pretrain
        ckpt["pretrain_requirement_met"] = len(pretrain_shims) >= minimum_pretrain
        ckpt["promotion_ready"] = False
        self.physics_transfer_checkpoint = ckpt
        self._restore_physics_transfer_head()

    def _train_neural_head(self, x_norm: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> None:
        """Train a bidirectional generator when enough labelled samples exist."""

        self.neural_checkpoint = None
        self.neural_generator = None
        self.neural_inverse = None
        mode = str(config.get("neural_mode", config.get("model_kind", "auto")) or "auto").lower()
        if mode in {"off", "false", "none", "linear"}:
            return
        min_samples = int(config.get("min_neural_samples", 64))
        if x_norm.shape[0] < min_samples and mode != "force":
            return
        if not TORCH_AVAILABLE:
            if mode in {"force", "neural", "adversarial"}:
                raise RuntimeError("PyTorch is required for neural feedback training")
            return

        seed = int(config.get("seed", 1234))
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        device = self._select_torch_device(str(config.get("device", "auto") or "auto"))
        input_dim = int(x_norm.shape[1])
        output_dim = int(y.shape[1])
        hidden_dim = int(config.get("neural_hidden_dim", 128))
        latent_dim = int(config.get("neural_latent_dim", 48))
        epochs = int(config.get("neural_epochs", 160 if x_norm.shape[0] < 64 else 80))
        learning_rate = float(config.get("neural_learning_rate", 1e-3))
        cycle_weight = float(config.get("cycle_consistency_weight", 0.15))
        adversarial_weight = float(config.get("adversarial_weight", 0.03 if mode == "adversarial" else 0.0))
        if adversarial_weight > 0.0 and _training_accelerator is not None and _training_accelerator(device) == "directml":
            device = torch.device("cpu")
        batch_size = max(1, min(int(config.get("neural_batch_size", 32)), int(x_norm.shape[0])))

        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(np.clip(y, -1.0, 1.0), dtype=torch.float32, device=device)
        generator = NeuralFeedbackGenerator(input_dim, output_dim, hidden_dim, latent_dim).to(device)
        inverse = NeuralFeedbackInverse(output_dim, input_dim, hidden_dim).to(device)
        discriminator = NeuralFeedbackDiscriminator(output_dim, max(32, hidden_dim // 2)).to(device)
        optimizer = torch.optim.AdamW(
            list(generator.parameters()) + list(inverse.parameters()),
            lr=learning_rate,
            weight_decay=float(config.get("neural_weight_decay", 1e-4)),
        )
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=learning_rate,
            weight_decay=float(config.get("neural_weight_decay", 1e-4)),
        )

        history: List[Dict[str, float]] = []
        best_state = None
        best_loss = None
        n = int(x_tensor.shape[0])
        for epoch in range(epochs):
            order = torch.tensor(rng.permutation(n), dtype=torch.long, device=device)
            epoch_supervised = 0.0
            epoch_cycle = 0.0
            epoch_adv = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb = x_tensor[idx]
                yb = y_tensor[idx]
                if adversarial_weight > 0.0:
                    with torch.no_grad():
                        fake_for_disc = generator(xb)
                    disc_optimizer.zero_grad(set_to_none=True)
                    real_logits = discriminator(yb)
                    fake_logits = discriminator(fake_for_disc.detach())
                    disc_loss = (
                        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                    )
                    disc_loss.backward()
                    disc_optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                pred = generator(xb)
                inv = inverse(pred)
                supervised = F.smooth_l1_loss(pred, yb)
                cycle = F.mse_loss(inv, xb)
                adv = torch.tensor(0.0, device=device)
                if adversarial_weight > 0.0:
                    fake_logits = discriminator(pred)
                    adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                loss = supervised + cycle_weight * cycle + adversarial_weight * adv
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(inverse.parameters()), 1.0)
                optimizer.step()
                count = int(xb.shape[0])
                epoch_supervised += float(supervised.item()) * count
                epoch_cycle += float(cycle.item()) * count
                epoch_adv += float(adv.item()) * count
                seen += count

            with torch.no_grad():
                pred_all = generator(x_tensor)
                inv_all = inverse(pred_all)
                supervised_all = float(F.smooth_l1_loss(pred_all, y_tensor).item())
                cycle_all = float(F.mse_loss(inv_all, x_tensor).item())
                score = supervised_all + cycle_weight * cycle_all
            row = {
                "epoch": float(epoch + 1),
                "supervised_loss": supervised_all,
                "cycle_loss": cycle_all,
                "batch_supervised_loss": epoch_supervised / max(seen, 1),
                "batch_cycle_loss": epoch_cycle / max(seen, 1),
                "batch_adversarial_loss": epoch_adv / max(seen, 1),
            }
            history.append(row)
            if best_loss is None or score < best_loss:
                best_loss = score
                best_state = {
                    "generator_state_dict": _state_dict_to_json(generator.state_dict()),
                    "inverse_state_dict": _state_dict_to_json(inverse.state_dict()),
                }

        if best_state is not None:
            generator.load_state_dict(_state_dict_from_json(best_state["generator_state_dict"], device=device))
            inverse.load_state_dict(_state_dict_from_json(best_state["inverse_state_dict"], device=device))

        confidence = float(np.clip(0.85 - 0.25 * np.sqrt(max(best_loss or 0.0, 0.0)), 0.45, 0.90))
        self.neural_checkpoint = {
            **(best_state or {
                "generator_state_dict": _state_dict_to_json(generator.state_dict()),
                "inverse_state_dict": _state_dict_to_json(inverse.state_dict()),
            }),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "epochs": epochs,
            **self._device_metadata(device),
            "training_mode": "adversarial_bidirectional_feedback" if adversarial_weight > 0.0 else "neural_bidirectional_feedback",
            "cycle_consistency_weight": cycle_weight,
            "adversarial_weight": adversarial_weight,
            "best_loss": float(best_loss or 0.0),
            "confidence": confidence,
            "history": history,
        }
        generator.eval()
        inverse.eval()
        self.neural_generator = generator
        self.neural_inverse = inverse

    def _restore_neural_head(self) -> None:
        self.neural_generator = None
        self.neural_inverse = None
        if not self.neural_checkpoint or not TORCH_AVAILABLE:
            return
        device = self._select_torch_device("auto")
        ckpt = self.neural_checkpoint
        generator = NeuralFeedbackGenerator(
            int(ckpt["input_dim"]),
            int(ckpt["output_dim"]),
            int(ckpt.get("hidden_dim", 128)),
            int(ckpt.get("latent_dim", 48)),
        ).to(device)
        inverse = NeuralFeedbackInverse(
            int(ckpt["output_dim"]),
            int(ckpt["input_dim"]),
            int(ckpt.get("hidden_dim", 128)),
        ).to(device)
        generator.load_state_dict(_state_dict_from_json(ckpt["generator_state_dict"], device=device))
        inverse.load_state_dict(_state_dict_from_json(ckpt["inverse_state_dict"], device=device))
        generator.eval()
        inverse.eval()
        self.neural_generator = generator
        self.neural_inverse = inverse

    def _restore_hybrid_head(self) -> None:
        self.hybrid_model = None
        self.hybrid_inverse = None
        if not self.hybrid_checkpoint or not TORCH_AVAILABLE:
            return
        device = self._select_torch_device("auto")
        ckpt = self.hybrid_checkpoint
        model = HybridFeedbackPolicy(
            int(ckpt["input_dim"]),
            int(ckpt["output_dim"]),
            int(ckpt.get("phase_map_size", 16)),
            int(ckpt.get("hidden_dim", 128)),
            int(ckpt.get("latent_dim", 48)),
            int(ckpt.get("image_channels", 16)),
        ).to(device)
        inverse = HybridFeedbackInverse(
            int(ckpt["output_dim"]),
            int(ckpt["input_dim"]),
            int(ckpt.get("hidden_dim", 128)),
        ).to(device)
        model.load_state_dict(_state_dict_from_json(ckpt["policy_state_dict"], device=device))
        inverse.load_state_dict(_state_dict_from_json(ckpt["inverse_state_dict"], device=device))
        model.eval()
        inverse.eval()
        self.hybrid_model = model
        self.hybrid_inverse = inverse

    def _restore_physics_transfer_head(self) -> None:
        """Restore PhysicsTransferGNN from saved checkpoint payload."""
        self.physics_transfer_model = None
        if not self.physics_transfer_checkpoint:
            return
        try:
            from ml_recon.physics_transfer import (
                TORCH_GEOMETRIC_AVAILABLE,
                restore_physics_transfer_model,
            )
        except ImportError:
            return
        if not TORCH_GEOMETRIC_AVAILABLE:
            return
        try:
            self.physics_transfer_model = restore_physics_transfer_model(
                self.physics_transfer_checkpoint, device="cpu",
            )
        except Exception:  # noqa: BLE001
            self.physics_transfer_model = None

    def _predict_hybrid_raw(self, features: np.ndarray, context: Dict[str, Any], size: int) -> np.ndarray:
        if self.hybrid_model is None or self.hybrid_checkpoint is None:
            raise RuntimeError("Hybrid feedback head is not loaded")
        device = next(self.hybrid_model.parameters()).device
        values = self._normalize_features(features)
        expected = int(self.hybrid_checkpoint["input_dim"])
        if values.size > expected:
            values = values[:expected]
        elif values.size < expected:
            values = np.pad(values, (0, expected - values.size))
        phase_map = self._hybrid_phase_map_input_from_context(context, size)
        x = torch.tensor(values.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.tensor(phase_map[None, None, :, :].astype(np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            raw = self.hybrid_model(x, m).detach().cpu()[0].tolist()
        return np.asarray(raw, dtype=float)

    def _hybrid_prediction_diagnostics(
        self,
        features: np.ndarray,
        context: Dict[str, Any],
        size: int,
    ) -> Dict[str, Any]:
        if self.hybrid_model is None or self.hybrid_inverse is None or self.hybrid_checkpoint is None:
            return {"available": False}
        try:
            device = next(self.hybrid_model.parameters()).device
            values = self._normalize_features(features)
            expected = int(self.hybrid_checkpoint["input_dim"])
            values = values[:expected] if values.size >= expected else np.pad(values, (0, expected - values.size))
            phase_map = self._hybrid_phase_map_input_from_context(context, size)
            x = torch.tensor(values.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(phase_map[None, None, :, :].astype(np.float32), dtype=torch.float32, device=device)
            with torch.no_grad():
                raw = self.hybrid_model(x, m)
                reconstructed = self.hybrid_inverse(raw)
                cycle_error = float(torch.mean((reconstructed - x) ** 2).cpu().item())
                raw_norm = float(torch.linalg.vector_norm(raw).cpu().item())
            return {
                "available": True,
                "training_mode": self.hybrid_checkpoint.get("training_mode"),
                "cycle_error": cycle_error,
                "raw_norm": raw_norm,
                "phase_map_input_rms": float(np.sqrt(np.mean(phase_map ** 2))),
                "best_loss": self.hybrid_checkpoint.get("best_loss"),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _predict_neural_raw(self, features: np.ndarray) -> np.ndarray:
        if self.neural_generator is None or self.neural_checkpoint is None:
            raise RuntimeError("Neural feedback head is not loaded")
        device = next(self.neural_generator.parameters()).device
        values = self._normalize_features(features)
        expected = int(self.neural_checkpoint["input_dim"])
        if values.size > expected:
            values = values[:expected]
        elif values.size < expected:
            values = np.pad(values, (0, expected - values.size))
        x = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            raw = self.neural_generator(x).detach().cpu()[0].tolist()
        return np.asarray(raw, dtype=float)

    def _neural_prediction_diagnostics(self, features: np.ndarray) -> Dict[str, Any]:
        if self.neural_generator is None or self.neural_inverse is None or self.neural_checkpoint is None:
            return {"available": False}
        try:
            device = next(self.neural_generator.parameters()).device
            values = self._normalize_features(features)
            expected = int(self.neural_checkpoint["input_dim"])
            values = values[:expected] if values.size >= expected else np.pad(values, (0, expected - values.size))
            x = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                raw = self.neural_generator(x)
                reconstructed = self.neural_inverse(raw)
                cycle_error = float(torch.mean((reconstructed - x) ** 2).cpu().item())
                raw_norm = float(torch.linalg.vector_norm(raw).cpu().item())
            return {
                "available": True,
                "training_mode": self.neural_checkpoint.get("training_mode"),
                "cycle_error": cycle_error,
                "raw_norm": raw_norm,
                "best_loss": self.neural_checkpoint.get("best_loss"),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _neural_metadata(self) -> Dict[str, Any]:
        if not self.neural_checkpoint:
            return {
                "available": False,
                "torch_available": bool(TORCH_AVAILABLE),
            }
        return {
            "available": True,
            "torch_available": bool(TORCH_AVAILABLE),
            "training_mode": self.neural_checkpoint.get("training_mode"),
            "input_dim": self.neural_checkpoint.get("input_dim"),
            "output_dim": self.neural_checkpoint.get("output_dim"),
            "epochs": self.neural_checkpoint.get("epochs"),
            "best_loss": self.neural_checkpoint.get("best_loss"),
            "confidence": self.neural_checkpoint.get("confidence"),
            "cycle_consistency_weight": self.neural_checkpoint.get("cycle_consistency_weight"),
            "adversarial_weight": self.neural_checkpoint.get("adversarial_weight"),
        }

    def _hybrid_metadata(self) -> Dict[str, Any]:
        if not self.hybrid_checkpoint:
            return {
                "available": False,
                "torch_available": bool(TORCH_AVAILABLE),
            }
        return {
            "available": True,
            "torch_available": bool(TORCH_AVAILABLE),
            "architecture": self.hybrid_checkpoint.get("architecture"),
            "training_mode": self.hybrid_checkpoint.get("training_mode"),
            "input_dim": self.hybrid_checkpoint.get("input_dim"),
            "output_dim": self.hybrid_checkpoint.get("output_dim"),
            "phase_map_size": self.hybrid_checkpoint.get("phase_map_size"),
            "epochs": self.hybrid_checkpoint.get("epochs"),
            "epochs_completed": self.hybrid_checkpoint.get("epochs_completed"),
            "best_epoch": self.hybrid_checkpoint.get("best_epoch"),
            "best_loss": self.hybrid_checkpoint.get("best_loss"),
            "converged": self.hybrid_checkpoint.get("converged"),
            "confidence": self.hybrid_checkpoint.get("confidence"),
            "device": self.hybrid_checkpoint.get("device"),
            "accelerator": self.hybrid_checkpoint.get("accelerator"),
            "device_name": self.hybrid_checkpoint.get("device_name"),
        }

    def _predict_physics_transfer_raw(
        self,
        target_rho: np.ndarray,
        measured_rho: np.ndarray,
        context: Dict[str, Any],
        n_modes: int,
    ) -> np.ndarray:
        """Forward pass through PhysicsTransferGNN. Pulls classifier_physics_aux
        from context (populated by runner.py F.3 hookup, or fallback zeros)."""
        if self.physics_transfer_model is None:
            raise RuntimeError("Physics-transfer model is not loaded")
        from ml_recon.physics_transfer import predict_physics_transfer_delta  # noqa: PLC0415

        clf_diag = context.get("classifier_diagnostics") or {}
        # Try several likely fields (deep classifier outputs physics_aux directly)
        physics_aux = (
            clf_diag.get("physics_aux")
            or clf_diag.get("physics_targets")
            or context.get("classifier_physics_aux")
        )
        if clf_diag.get("deep_classifier_deployment_eligible") is False:
            # Legacy target-conditioned checkpoints may be loaded for UI
            # diagnostics, but their auxiliary vector is never a controller
            # input.  The explicit zero vector is covered by the missing-mask
            # training audit and cannot create target leakage.
            physics_aux = None
        if physics_aux is None:
            cfg = self.physics_transfer_checkpoint.get("config", {})
            feat_dim = int(cfg.get("classifier_feat_dim", 26))
            physics_aux = np.zeros(feat_dim, dtype=np.float32)
        else:
            physics_aux = np.asarray(physics_aux, dtype=np.float32).reshape(-1)
            cfg = self.physics_transfer_checkpoint.get("config", {})
            feat_dim = int(cfg.get("classifier_feat_dim", 26))
            if physics_aux.size != feat_dim:
                raise ValueError(
                    f"classifier physics_aux width {physics_aux.size} != checkpoint schema {feat_dim}; "
                    "silent truncation is forbidden"
                )

        # MC Dropout: enable train mode if context asks for it (used by runner ensemble)
        train_mode = bool(context.get("ensemble_train_mode", False))
        delta = predict_physics_transfer_delta(
            self.physics_transfer_model,
            target_rho=np.asarray(target_rho, dtype=complex),
            measured_rho=np.asarray(measured_rho, dtype=complex),
            classifier_physics_aux=physics_aux,
            num_qubits=int(n_modes),
            device="cpu",
            train_mode_for_dropout=train_mode,
        )
        return np.asarray(delta, dtype=float)

    def _physics_transfer_metadata(self) -> Dict[str, Any]:
        try:
            from ml_recon.physics_transfer import TORCH_GEOMETRIC_AVAILABLE  # noqa: PLC0415
        except ImportError:
            TORCH_GEOMETRIC_AVAILABLE = False  # noqa: N806
        if not self.physics_transfer_checkpoint:
            return {
                "available": False,
                "torch_geometric_available": bool(TORCH_GEOMETRIC_AVAILABLE),
            }
        cfg = self.physics_transfer_checkpoint.get("config", {})
        return {
            "available": True,
            "torch_geometric_available": bool(TORCH_GEOMETRIC_AVAILABLE),
            "training_mode": self.physics_transfer_checkpoint.get("training_mode"),
            "n_pretrain": self.physics_transfer_checkpoint.get("n_pretrain"),
            "n_finetune": self.physics_transfer_checkpoint.get("n_finetune"),
            "confidence": self.physics_transfer_checkpoint.get("confidence"),
            "calibration_status": self.physics_transfer_checkpoint.get("calibration_status", "legacy_unknown"),
            "output_mode": self.physics_transfer_checkpoint.get("output_mode", "legacy_vector_v1"),
            "pretrain_requirement_met": self.physics_transfer_checkpoint.get("pretrain_requirement_met", False),
            "promotion_ready": self.physics_transfer_checkpoint.get("promotion_ready", False),
            "num_qubits": self.physics_transfer_checkpoint.get("num_qubits"),
            "classifier_feat_dim": int(cfg.get("classifier_feat_dim", 26)),
            "output_dim": int(cfg.get("output_dim", 8)),
            "pretrain_epochs": int(cfg.get("pretrain_epochs", 0)),
            "finetune_epochs": int(cfg.get("finetune_epochs", 0)),
        }

    def _load_samples(self, samples_dir: str) -> List[FeedbackSample]:
        samples: List[FeedbackSample] = []
        root = Path(samples_dir)
        print(f"[feedback] loading samples from {root} ...", flush=True)
        for path in sorted(root.rglob("*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or "sample_id" not in payload:
                continue
            sample = FeedbackSample.from_json_dict(payload)
            if sample.system_type == self.system_type:
                samples.append(sample)
        _shards_done = 0
        for pattern, compressed in (("*.jsonl", False), ("*.jsonl.gz", True)):
            for path in sorted(root.rglob(pattern)):
                opener = gzip.open if compressed else open
                with opener(path, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                            sample = FeedbackSample.from_json_dict(payload)
                        except Exception as exc:
                            raise ValueError(f"invalid feedback JSONL row {path}:{line_number}: {exc}") from exc
                        if sample.system_type == self.system_type:
                            samples.append(sample)
                _shards_done += 1
                if _shards_done % 20 == 0:
                    print(f"[feedback] loaded {len(samples)} samples ({_shards_done} shards done)", flush=True)
        print(f"[feedback] loaded {len(samples)} samples total from {_shards_done} shards", flush=True)
        return samples

    def _feature_vector(self, target_rho: np.ndarray, measured_rho: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        target = np.asarray(target_rho, dtype=complex)
        measured = np.asarray(measured_rho, dtype=complex)
        system_type = context.get("system_type", self.system_type)
        size = 15 if system_type == "3qubit" else 16
        distance = self._rho_pair_distance_summary(target, measured)
        state = context.get("target_state_params", {})
        amps = np.asarray(state.get("amplitudes", []), dtype=float)
        retrieval = context.get("retrieval_metrics", {})
        retrieval_rate = float(retrieval.get("retrieval_rate", context.get("retrieval_rate", 0.0)) or 0.0)
        total_power = float(np.sum(amps ** 2)) if amps.size else 0.0
        noise_context = context.get("noise_context", {}) or {}
        if self.feature_schema_version >= 2:
            multi = noise_context.get("multi_click_fractions", {}) or {}
            pair = noise_context.get("pair_model", {}) or {}
            proxy_a = float(multi.get("multi2") or 0.0)
            proxy_b = float(pair.get("pair_dark_fraction") or 0.0)
        else:
            proxy_a = float(distance["coherence_magnitude_l1"])
            proxy_b = float(max(0.0, 1.0 - np.real(np.trace(measured))))
        coupled_wave_proxy = float(total_power * (np.std(amps) if amps.size else 0.0))
        diagnosis_features = self._diagnosis_feature_vector(context)
        wch1_phase_map = self._context_wch1_phase_map(context, size)
        phase_map_features = self._phase_map_feature_vector(wch1_phase_map, size)
        base_features = np.asarray([
            distance["state_infidelity"],
            distance["trace_distance"],
            distance["bures_distance"],
            distance["quantum_jsd"],
            distance["population_l1"],
            distance["coherence_magnitude_l1"],
            distance["coherence_phase_rms"],
            np.mean(np.abs(target - np.diag(np.diag(target)))) if target.size else 0.0,
            np.mean(np.abs(measured - np.diag(np.diag(measured)))) if measured.size else 0.0,
            retrieval_rate,
            total_power,
            proxy_a,
            proxy_b,
            coupled_wave_proxy,
            *diagnosis_features.tolist(),
        ], dtype=float)
        return np.concatenate([base_features, phase_map_features])

    def _rho_pair_distance_summary(self, target_rho: np.ndarray, measured_rho: np.ndarray) -> Dict[str, float]:
        values = rho_pair_distance_features(target_rho, measured_rho)
        mapping = {
            name: float(value)
            for name, value in zip(RHO_PAIR_DISTANCE_FEATURE_NAMES, values)
        }
        return {
            "state_infidelity": mapping.get("state_infidelity", 0.0),
            "trace_distance": mapping.get("trace_distance", 0.0),
            "bures_distance": mapping.get("bures_distance", 0.0),
            "quantum_jsd": mapping.get("quantum_jsd", 0.0),
            "population_l1": mapping.get("population_l1", 0.0),
            "coherence_magnitude_l1": mapping.get("coherence_magnitude_l1", 0.0),
            "coherence_phase_rms": mapping.get("coherence_phase_rms", 0.0),
        }

    def _fit_phase_map_pca(self, samples: List[FeedbackSample], config: Dict[str, Any]) -> None:
        self.phase_map_pca_mean = None
        self.phase_map_pca_components = None
        if self.phase_map_feature_mode != "pca":
            return
        size = 15 if self.system_type == "3qubit" else 16
        rows = []
        for sample in samples:
            context = {
                "system_type": sample.system_type,
                "target_state_params": sample.target_state_params,
                "retrieval_metrics": sample.retrieval_metrics,
            }
            phase_map = self._context_wch1_phase_map(context, size)
            rows.append(self._raw_phase_map_vector(phase_map, size))
        matrix = np.vstack(rows) if rows else np.zeros((0, size * size), dtype=float)
        if matrix.shape[0] < 2:
            self.phase_map_pca_mean = np.zeros(size * size, dtype=float)
            self.phase_map_pca_components = np.zeros((0, size * size), dtype=float)
            return
        self.phase_map_pca_mean = np.mean(matrix, axis=0)
        centered = matrix - self.phase_map_pca_mean
        max_components = min(centered.shape[0] - 1, centered.shape[1])
        requested = int(config.get("phase_map_pca_components", min(32, max_components)))
        n_components = max(1, min(requested, max_components))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.phase_map_pca_components = vt[:n_components]

    def _hybrid_phase_map_input_from_sample(self, sample: FeedbackSample, size: int) -> np.ndarray:
        context = {
            "system_type": sample.system_type,
            "target_state_params": sample.target_state_params,
            "retrieval_metrics": sample.retrieval_metrics,
        }
        return self._hybrid_phase_map_input_from_context(context, size)

    def _hybrid_phase_map_input_from_context(self, context: Dict[str, Any], size: int) -> np.ndarray:
        phase_map = self._context_wch1_phase_map(context, size)
        if phase_map is None:
            return np.zeros((size, size), dtype=np.float32)
        arr = np.asarray(phase_map, dtype=float)
        if arr.shape != (size, size):
            fixed = np.zeros((size, size), dtype=float)
            rows = min(size, arr.shape[0]) if arr.ndim >= 1 else 0
            cols = min(size, arr.shape[1]) if arr.ndim >= 2 else 0
            if rows and cols:
                fixed[:rows, :cols] = arr[:rows, :cols]
            arr = fixed
        return np.clip(arr / 0.20, -1.0, 1.0).astype(np.float32)

    def _phase_map_feature_vector(self, phase_map: Optional[np.ndarray], size: int) -> np.ndarray:
        mode = str(self.phase_map_feature_mode or "pyramid").lower()
        if mode == "raw":
            return self._raw_phase_map_vector(phase_map, size)
        if mode == "pca":
            raw = self._raw_phase_map_vector(phase_map, size)
            if self.phase_map_pca_mean is None or self.phase_map_pca_components is None:
                return np.zeros(int(self.metadata.get("phase_map_pca_components", 0) or 0), dtype=float)
            if self.phase_map_pca_components.size == 0:
                return np.zeros(0, dtype=float)
            return (raw - self.phase_map_pca_mean) @ self.phase_map_pca_components.T
        return self._pyramid_phase_map_features(phase_map, size)

    def _raw_phase_map_vector(self, phase_map: Optional[np.ndarray], size: int) -> np.ndarray:
        if phase_map is None:
            return np.zeros(size * size, dtype=float)
        arr = np.asarray(phase_map, dtype=float)
        if arr.shape != (size, size):
            return np.zeros(size * size, dtype=float)
        return np.clip(arr.reshape(-1), -0.20, 0.20) / 0.20

    def _pyramid_phase_map_features(self, phase_map: Optional[np.ndarray], size: int) -> np.ndarray:
        if phase_map is None:
            arr = np.zeros((size, size), dtype=float)
        else:
            arr = np.asarray(phase_map, dtype=float)
            if arr.shape != (size, size):
                arr = np.zeros((size, size), dtype=float)
        arr = np.clip(arr, -0.20, 0.20) / 0.20
        features: List[float] = []
        for scale in (1, 2, 4):
            for row_block in np.array_split(arr, scale, axis=0):
                for block in np.array_split(row_block, scale, axis=1):
                    features.extend([
                        float(np.mean(block)),
                        float(np.std(block)),
                        float(np.sqrt(np.mean(block * block))),
                    ])
        return np.asarray(features, dtype=float)

    def _phase_map_feature_names(self, size: int) -> List[str]:
        mode = str(self.phase_map_feature_mode or "pyramid").lower()
        if mode == "raw":
            return [
                f"wch1_phase_map_r{row:02d}_c{col:02d}"
                for row in range(size)
                for col in range(size)
            ]
        if mode == "pca":
            count = 0 if self.phase_map_pca_components is None else int(self.phase_map_pca_components.shape[0])
            count = int(self.metadata.get("phase_map_pca_components", count) or count)
            return [f"wch1_phase_pca_{idx:02d}" for idx in range(count)]
        names = []
        for scale in (1, 2, 4):
            for row in range(scale):
                for col in range(scale):
                    for stat in ("mean", "std", "rms"):
                        names.append(f"wch1_phase_pyramid_s{scale}_r{row}_c{col}_{stat}")
        return names

    def _feature_schema(self, system_type: str) -> List[str]:
        size = 15 if system_type == "3qubit" else 16
        return [
            "rho_infidelity",
            "trace_distance",
            "bures_distance",
            "quantum_jsd",
            "population_l1",
            "coherence_magnitude_l1",
            "coherence_phase_rms",
            "target_coherence",
            "measured_coherence",
            "retrieval_rate",
            "total_rf_power",
            *( ["gsi_proxy", "double_excitation_proxy"] if self.feature_schema_version == 1 else ["noise_multi2_fraction", "noise_pair_dark_fraction"] ),
            "coupled_wave_proxy",
            "diagnosis_mode_confidence",
            "diagnosis_mode_entropy",
            "diagnosis_non_target_mode_mass",
            "diagnosis_phase_drift_prob",
            "diagnosis_amp_imbalance_prob",
            "diagnosis_dephasing_prob",
            "diagnosis_retrieval_imbalance_prob",
            "diagnosis_double_excitation_prob",
            "diagnosis_unknown_probability",
            "diagnosis_latent_norm",
            *self._phase_map_feature_names(size),
        ]

    def _output_schema(self, system_type: str) -> List[str]:
        n_modes = 3 if system_type == "3qubit" else 4
        size = 15 if system_type == "3qubit" else 16
        return [
            *[f"W_CH3_phase_mode_{idx + 1}" for idx in range(n_modes)],
            *[f"R_CH3_phase_mode_{idx + 1}" for idx in range(n_modes)],
            *[f"W_CH3_amp_mode_{idx + 1}" for idx in range(n_modes)],
            *[f"R_CH3_amp_mode_{idx + 1}" for idx in range(n_modes)],
            "W_CH1_phase_common",
            "R_CH1_phase_common",
            *[
                f"W_CH1_phase_r{row:02d}_c{col:02d}"
                for row in range(size)
                for col in range(size)
            ],
        ]

    def _diagnosis_feature_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode first-stage classifier diagnostics when they are available."""

        retrieval = context.get("retrieval_metrics", {}) or {}
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
        )
        if not diagnosis and isinstance(retrieval.get("diagnostics"), dict):
            nested = retrieval["diagnostics"]
            diagnosis = (
                nested.get("classifier_diagnostics")
                or nested.get("diagnosis")
                or nested.get("quantum_diagnosis")
            )
        diagnosis = diagnosis or {}
        mode_probs = diagnosis.get("mode_probabilities") or {}
        error_probs = diagnosis.get("error_source_probabilities") or {}
        mode_values = np.asarray(list(mode_probs.values()), dtype=float)
        if mode_values.size and np.sum(mode_values) > 0:
            mode_values = mode_values / np.sum(mode_values)
            mode_confidence = float(np.max(mode_values))
            mode_entropy = float(
                -np.sum(mode_values * np.log(mode_values + 1e-12))
                / np.log(mode_values.size + 1e-12)
            )
        else:
            mode_confidence = 0.0
            mode_entropy = 1.0

        state = context.get("target_state_params", {}) or {}
        target_mode = int(
            context.get("excitation_mode", context.get("target_excitation_mode", state.get("excitation_mode", 0)))
            or 0
        )
        target_label = f"mode_{target_mode}" if target_mode else None
        if mode_probs:
            target_mass = float(mode_probs.get(target_label, 0.0)) if target_label else max(mode_probs.values())
            non_target_mass = float(np.clip(1.0 - target_mass, 0.0, 1.0))
        else:
            non_target_mass = 0.0

        latent = np.asarray(diagnosis.get("latent_vector", []), dtype=float)
        covariance_features = np.asarray(diagnosis.get("covariance_feature_vector", []), dtype=float)
        if latent.size:
            latent_norm = float(np.linalg.norm(latent))
        elif covariance_features.size:
            latent_norm = float(np.linalg.norm(covariance_features) / np.sqrt(covariance_features.size))
        else:
            latent_norm = 0.0

        def error_prob(label: str) -> float:
            return float(error_probs.get(label, 0.0) or 0.0)

        return np.asarray([
            mode_confidence,
            mode_entropy,
            non_target_mass,
            error_prob("phase_drift"),
            error_prob("amp_imbalance"),
            error_prob("dephasing"),
            error_prob("retrieval_imbalance"),
            error_prob("double_excitation"),
            float(diagnosis.get("unknown_probability", 0.0) or 0.0),
            latent_norm,
        ], dtype=float)

    def _align_features_for_weights(self, features: np.ndarray) -> np.ndarray:
        """Keep older checkpoints usable when the feature schema grows."""

        if self.weights is None:
            return features
        expected = int(self.weights.shape[1])
        if features.size == expected:
            return features
        if features.size > expected:
            return features[:expected]
        return np.pad(features, (0, expected - features.size))

    def _align_raw_output(self, values: np.ndarray, expected: int) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == expected:
            return values
        if values.size > expected:
            return values[:expected]
        return np.pad(values, (0, expected - values.size))

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return np.asarray(features, dtype=float)
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            if self.feature_mean.size != values.size:
                return values
            return (values - self.feature_mean) / self.feature_scale
        if values.shape[1] != self.feature_mean.size:
            return values
        return (values - self.feature_mean) / self.feature_scale

    def _entanglement_phase_feature_diagnostics(
        self,
        target_rho: np.ndarray,
        measured_rho: np.ndarray,
        num_qubits: int,
        phase_strategy: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        signature_values = entanglement_signature_features(
            target_rho,
            measured_rho,
            int(num_qubits),
            include_geometric=True,
            history=context.get("rho_history"),
        )
        signature_names = entanglement_signature_feature_names(int(num_qubits), include_geometric=True)
        geometric_values = geometric_phase_features(
            target_rho,
            measured_rho,
            int(num_qubits),
            history=context.get("rho_history"),
        )
        geometric_names = geometric_phase_feature_names(int(num_qubits))
        classifier = self._classifier_enhanced_status(context)
        return {
            "feature_schema": "physics_v2",
            "entanglement_signature": {
                "schema": "entanglement_signature_v1",
                "feature_count": int(signature_values.size),
                "names": signature_names,
                "values": [float(value) for value in signature_values],
            },
            "geometric_phase_features": {
                "schema": "geometric_phase_experimental_v1",
                "active": phase_strategy == "geometric_phase_experimental",
                "diagnostic_only": phase_strategy == "geometric_phase_experimental"
                and not bool(context.get("allow_geometric_phase_feedback", False)),
                "names": geometric_names,
                "values": [float(value) for value in geometric_values],
            },
            "classifier_enhanced": classifier,
        }

    def _classifier_enhanced_status(self, context: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = self._extract_classifier_diagnosis(context)
        schema = (
            diagnosis.get("physics_feature_schema")
            or diagnosis.get("feature_schema")
            or (diagnosis.get("feature_config") or {}).get("physics_feature_schema")
            or (diagnosis.get("feature_config") or {}).get("feature_schema")
        )
        schema = None if schema is None else str(schema)
        enhanced = schema in {"physics_v2", "entanglement_signature_v1"}
        return {
            "requested": str(context.get("phase_feedback_strategy", "")) == "classifier_enhanced",
            "available": bool(enhanced),
            "feature_schema": schema,
            "fallback": not bool(enhanced),
            "fallback_reason": None if enhanced else "no physics_v2 classifier diagnostics were supplied",
        }

    def _extract_classifier_diagnosis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
        )
        retrieval = context.get("retrieval_metrics", {}) or {}
        if not diagnosis and isinstance(retrieval.get("diagnostics"), dict):
            nested = retrieval["diagnostics"]
            diagnosis = (
                nested.get("classifier_diagnostics")
                or nested.get("diagnosis")
                or nested.get("quantum_diagnosis")
            )
        return diagnosis or {}

    def _feedback_safety_diagnostics(
        self,
        target_rho: np.ndarray,
        measured_rho: np.ndarray,
        system_type: str,
        n_modes: int,
        context: Dict[str, Any],
        base_confidence: float,
        source: str,
        phase_resolver: Optional[Dict[str, Any]],
        phase_strategy: str,
    ) -> Dict[str, Any]:
        target_physicality = self._density_physicality(target_rho)
        measured_physicality = self._density_physicality(measured_rho)
        single = [2 ** (n_modes - 1 - i) for i in range(n_modes)]
        target = np.asarray(target_rho, dtype=complex)
        measured = np.asarray(measured_rho, dtype=complex)
        target_sub = target[np.ix_(single, single)] if target.shape[0] > max(single) else np.zeros((n_modes, n_modes))
        measured_sub = measured[np.ix_(single, single)] if measured.shape[0] > max(single) else np.zeros((n_modes, n_modes))
        target_coherence = self._subspace_coherence(target_sub)
        measured_coherence = self._subspace_coherence(measured_sub)
        coherence_score = float(np.clip(measured_coherence / max(target_coherence, 1e-9), 0.0, 1.0))
        measured_trace = float(np.real(np.trace(measured))) if measured.size else 0.0
        w_population = float(np.clip(np.real(np.trace(measured_sub)), 0.0, None))
        w_population_fraction = float(np.clip(w_population / max(abs(measured_trace), 1e-9), 0.0, 1.0))
        correctable_fraction = float(np.clip(w_population_fraction * coherence_score, 0.0, 1.0))
        classifier = self._classifier_safety(context)

        min_correctable = float(context.get("min_correctable_fraction", 0.05) or 0.05)
        max_unknown = float(context.get("max_unknown_probability_for_feedback", 0.85) or 0.85)
        max_noncorrectable = float(context.get("max_noncorrectable_probability_for_feedback", 0.75) or 0.75)
        reasons: List[str] = []
        if not measured_physicality["physicality_ok"]:
            reasons.append("measured density matrix is not trace-one positive semidefinite")
        if correctable_fraction < min_correctable:
            reasons.append(
                f"correctable_fraction {correctable_fraction:.3f} below {min_correctable:.3f}"
            )
        if classifier["unknown_probability"] > max_unknown:
            reasons.append(
                f"classifier unknown probability {classifier['unknown_probability']:.3f} above {max_unknown:.3f}"
            )
        if classifier["noncorrectable_probability"] > max_noncorrectable:
            reasons.append(
                f"dominant classifier mass is non-correctable ({classifier['noncorrectable_probability']:.3f})"
            )
        auto_apply_allowed = True
        if phase_strategy == "geometric_phase_experimental" and not bool(
            context.get("allow_geometric_phase_feedback", False)
        ):
            auto_apply_allowed = False
            reasons.append("geometric_phase_experimental is diagnostic-only by default")

        safe = not reasons
        safety_factor = 0.5 + 0.5 * correctable_fraction
        if not measured_physicality["physicality_ok"]:
            safety_factor = min(safety_factor, 0.1)
        if classifier["unknown_probability"] > 0.0:
            safety_factor *= float(np.clip(1.0 - 0.5 * classifier["unknown_probability"], 0.25, 1.0))
        confidence = float(np.clip(float(base_confidence) * safety_factor, 0.0, 1.0))
        if not safe:
            confidence = min(confidence, 0.05)

        return {
            "safe": safe,
            "unsafe_reason": "; ".join(reasons),
            "source": source,
            "phase_feedback_strategy": phase_strategy,
            "auto_apply_allowed": bool(auto_apply_allowed),
            "strict_hardware_mode": bool(context.get("strict_hardware_mode", False)),
            "phase_resolver_active": bool(
                phase_resolver is not None and phase_resolver.get("accepted", True)
            ),
            "phase_resolver_blocked": bool(
                phase_resolver is not None and not phase_resolver.get("accepted", True)
            ),
            "physicality_ok": bool(target_physicality["physicality_ok"] and measured_physicality["physicality_ok"]),
            "trace_ok": bool(target_physicality["trace_ok"] and measured_physicality["trace_ok"]),
            "psd_ok": bool(target_physicality["psd_ok"] and measured_physicality["psd_ok"]),
            "target_physicality": target_physicality,
            "measured_physicality": measured_physicality,
            "w_population_fraction": w_population_fraction,
            "correctable_fraction": correctable_fraction,
            "coherence_score": coherence_score,
            "target_coherence": target_coherence,
            "measured_coherence": measured_coherence,
            "classifier_unknown_probability": classifier["unknown_probability"],
            "classifier_noncorrectable_probability": classifier["noncorrectable_probability"],
            "classifier_top_error": classifier["top_error"],
            "confidence": confidence,
        }

    def _density_physicality(self, rho: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(rho, dtype=complex)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
            return {
                "physicality_ok": False,
                "trace_ok": False,
                "psd_ok": False,
                "hermitian_ok": False,
                "trace_real": 0.0,
                "trace_imag": 0.0,
                "min_eigenvalue": float("-inf"),
                "hermitian_error": float("inf"),
            }
        herm = 0.5 * (arr + arr.conj().T)
        norm = max(float(np.linalg.norm(arr)), 1e-12)
        hermitian_error = float(np.linalg.norm(arr - arr.conj().T) / norm)
        trace = np.trace(arr)
        try:
            eigvals = np.linalg.eigvalsh(herm)
            min_eigenvalue = float(np.min(np.real(eigvals)))
        except np.linalg.LinAlgError:
            min_eigenvalue = float("-inf")
        trace_ok = abs(float(np.real(trace)) - 1.0) <= 0.10 and abs(float(np.imag(trace))) <= 1e-5
        hermitian_ok = hermitian_error <= 1e-5
        psd_ok = min_eigenvalue >= -0.05
        return {
            "physicality_ok": bool(trace_ok and hermitian_ok and psd_ok),
            "trace_ok": bool(trace_ok),
            "psd_ok": bool(psd_ok),
            "hermitian_ok": bool(hermitian_ok),
            "trace_real": float(np.real(trace)),
            "trace_imag": float(np.imag(trace)),
            "min_eigenvalue": min_eigenvalue,
            "hermitian_error": hermitian_error,
        }

    def _classifier_safety(self, context: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
            or {}
        )
        if not isinstance(diagnosis, dict):
            return {"unknown_probability": 0.0, "noncorrectable_probability": 0.0, "top_error": None}
        unknown = float(diagnosis.get("unknown_probability", 0.0) or 0.0)
        probs = (
            diagnosis.get("error_source_probabilities")
            or diagnosis.get("error_probabilities")
            or diagnosis.get("error_probs")
            or {}
        )
        if not isinstance(probs, dict):
            return {"unknown_probability": unknown, "noncorrectable_probability": 0.0, "top_error": None}
        top_error = max(probs, key=probs.get) if probs else None
        noncorrectable = 0.0
        tokens = ("dephas", "decoher", "leak", "double", "loss", "mixed", "thermal")
        for label, value in probs.items():
            lower = str(label).lower()
            if "phase_drift" in lower or "phase-map" in lower:
                continue
            if any(token in lower for token in tokens):
                noncorrectable += float(value or 0.0)
        return {
            "unknown_probability": float(np.clip(unknown, 0.0, 1.0)),
            "noncorrectable_probability": float(np.clip(noncorrectable, 0.0, 1.0)),
            "top_error": top_error,
        }

    def _unsafe_deltas(self, deltas: List[CalibrationDelta], reason: str) -> List[CalibrationDelta]:
        return [
            CalibrationDelta(
                target_channel=delta.target_channel,
                matrix_type=delta.matrix_type,
                delta=np.zeros_like(delta.delta),
                limit=delta.limit,
                accepted=False,
                reason=reason or "feedback safety guard rejected the prediction",
                path=delta.path,
            )
            for delta in deltas
        ]

    def _heuristic_output(self, target_rho: np.ndarray, measured_rho: np.ndarray, context: Dict[str, Any], n_modes: int) -> np.ndarray:
        target = np.asarray(target_rho, dtype=complex)
        measured = np.asarray(measured_rho, dtype=complex)
        diff = target - measured
        diag = np.real(np.diag(diff))
        single_indices = [2 ** (n_modes - 1 - i) for i in range(n_modes)]
        local = np.asarray([diag[idx] if idx < diag.size else 0.0 for idx in single_indices], dtype=float)
        local = np.tanh(local)
        phases = np.angle(target[np.ix_(single_indices, single_indices)] + 1e-12)
        measured_phases = np.angle(measured[np.ix_(single_indices, single_indices)] + 1e-12)
        phase_error = np.tanh(np.mean(phases - measured_phases, axis=1))
        common = float(np.tanh(np.mean(phase_error))) if phase_error.size else 0.0
        return np.concatenate([
            phase_error,
            -phase_error,
            local,
            local,
            np.array([common, 0.5 * common], dtype=float),
        ])

    def _wch1_phase_resolver_delta(
        self,
        target_rho: np.ndarray,
        measured_rho: np.ndarray,
        system_type: str,
        size: int,
        n_modes: int,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Map single-excitation phase residuals onto the W_CH1 phase matrix."""

        strict_hardware_mode = bool(context.get("strict_hardware_mode", False))
        direct_map = self._context_wch1_phase_map(context, size)
        if direct_map is not None:
            matrix = np.clip(direct_map, -0.20, 0.20)
            return {
                "matrix": matrix,
                "merge_strategy": "replace_wch1_phase",
                "accepted": True,
                "diagnostics": {
                    "mapping": "direct W_CH1 per-point phase correction map",
                    "strict_hardware_mode": strict_hardware_mode,
                    "evidence": "direct_phase_correction_map",
                    "max_abs_correction": float(np.max(np.abs(matrix))) if matrix.size else 0.0,
                    "rms_correction": float(np.sqrt(np.mean(matrix * matrix))) if matrix.size else 0.0,
                },
            }

        sensitivity = self._context_wch1_sensitivity(context, size, n_modes)
        if strict_hardware_mode and sensitivity is None:
            reason = (
                "strict_hardware_mode requires a direct W_CH1 phase correction map "
                "or W_CH1 phase sensitivity tensor"
            )
            return {
                "matrix": np.zeros((size, size), dtype=float),
                "merge_strategy": "replace_wch1_phase",
                "accepted": False,
                "reason": reason,
                "diagnostics": {
                    "mapping": "blocked",
                    "strict_hardware_mode": True,
                    "strict_hardware_blocked": True,
                    "reason": reason,
                    "required_evidence": [
                        "wch1_phase_correction_map",
                        "phase_point_correction_map",
                        "wch1_phase_sensitivity",
                        "phase_point_sensitivity",
                    ],
                },
            }

        target = np.asarray(target_rho, dtype=complex)
        measured = np.asarray(measured_rho, dtype=complex)
        single = [2 ** (n_modes - 1 - i) for i in range(n_modes)]
        if target.shape[0] <= max(single) or measured.shape[0] <= max(single):
            return None

        target_sub = target[np.ix_(single, single)]
        measured_sub = measured[np.ix_(single, single)]
        target_vec = self._dominant_subspace_vector(target_sub)
        measured_vec = self._dominant_subspace_vector(measured_sub)
        if target_vec is None or measured_vec is None:
            return None

        phase_error = np.angle(target_vec * np.conjugate(measured_vec))
        phase_error = self._wrap_phase(np.unwrap(phase_error))
        weights = np.abs(target_vec) * np.abs(measured_vec)
        if float(np.sum(weights)) > 1e-9:
            phase_error = phase_error - float(np.average(phase_error, weights=weights))
        else:
            phase_error = phase_error - float(np.mean(phase_error))
        phase_error = self._wrap_phase(phase_error)

        target_coherence = self._subspace_coherence(target_sub)
        measured_coherence = self._subspace_coherence(measured_sub)
        coherence_gate = float(np.clip(measured_coherence / max(target_coherence, 1e-9), 0.0, 1.0))
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
            or {}
        )
        unknown = float(diagnosis.get("unknown_probability", 0.0) or 0.0) if isinstance(diagnosis, dict) else 0.0
        gate = coherence_gate * float(np.clip(1.0 - 0.5 * unknown, 0.25, 1.0))

        correction = np.clip(gate * phase_error, -0.20, 0.20)
        matrix = self._wch1_phase_matrix_from_sensitivity(
            correction,
            system_type,
            size,
            n_modes,
            context,
            sensitivity=sensitivity,
        )
        mapping = (
            "W_CH1 phase sensitivity tensor"
            if sensitivity is not None
            else "W_CH1 phase group/offset pattern"
        )
        return {
            "matrix": matrix,
            "merge_strategy": "additive_wch1_phase",
            "accepted": True,
            "diagnostics": {
                "phase_error_rad": phase_error.tolist(),
                "applied_correction_rad": correction.tolist(),
                "coherence_gate": gate,
                "target_coherence": target_coherence,
                "measured_coherence": measured_coherence,
                "mapping": mapping,
                "strict_hardware_mode": strict_hardware_mode,
                "evidence": "sensitivity_tensor" if sensitivity is not None else "pattern_fallback",
            },
        }

    def _context_wch1_phase_map(self, context: Dict[str, Any], size: int) -> Optional[np.ndarray]:
        candidates = [
            context.get("wch1_phase_correction_map"),
            context.get("phase_point_correction_map"),
        ]
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
            or {}
        )
        if isinstance(diagnosis, dict):
            candidates.extend([
                diagnosis.get("wch1_phase_correction_map"),
                diagnosis.get("phase_point_correction_map"),
            ])
        retrieval = context.get("retrieval_metrics", {}) or {}
        nested = retrieval.get("diagnostics") if isinstance(retrieval, dict) else None
        if isinstance(nested, dict):
            nested_diag = (
                nested.get("classifier_diagnostics")
                or nested.get("diagnosis")
                or nested.get("quantum_diagnosis")
                or {}
            )
            if isinstance(nested_diag, dict):
                candidates.extend([
                    nested_diag.get("wch1_phase_correction_map"),
                    nested_diag.get("phase_point_correction_map"),
                ])

        for candidate in candidates:
            if candidate is None:
                continue
            arr = np.asarray(candidate, dtype=float)
            if arr.shape == (size, size):
                return arr
        return None

    def _wch1_phase_matrix_from_sensitivity(
        self,
        mode_correction: np.ndarray,
        system_type: str,
        size: int,
        n_modes: int,
        context: Dict[str, Any],
        sensitivity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if sensitivity is None:
            sensitivity = self._context_wch1_sensitivity(context, size, n_modes)
        if sensitivity is not None:
            matrix = np.tensordot(sensitivity, np.asarray(mode_correction, dtype=float)[:n_modes], axes=([2], [0]))
            return np.clip(matrix, -0.20, 0.20)
        return self._wch1_phase_pattern_matrix(mode_correction, system_type, size, n_modes)

    def _context_wch1_sensitivity(self, context: Dict[str, Any], size: int, n_modes: int) -> Optional[np.ndarray]:
        candidates = [
            context.get("wch1_phase_sensitivity"),
            context.get("phase_point_sensitivity"),
        ]
        diagnosis = (
            context.get("classifier_diagnostics")
            or context.get("diagnosis")
            or context.get("quantum_diagnosis")
            or {}
        )
        if isinstance(diagnosis, dict):
            candidates.extend([
                diagnosis.get("wch1_phase_sensitivity"),
                diagnosis.get("phase_point_sensitivity"),
            ])
        for candidate in candidates:
            if candidate is None:
                continue
            arr = np.asarray(candidate, dtype=float)
            if arr.shape == (size, size, n_modes):
                return arr
        return None

    def _merge_wch1_phase_resolver(
        self,
        deltas: List[CalibrationDelta],
        phase_resolver: Dict[str, Any],
    ) -> List[CalibrationDelta]:
        matrix = np.asarray(phase_resolver["matrix"], dtype=float)
        replace = phase_resolver.get("merge_strategy") == "replace_wch1_phase"
        resolver_accepted = bool(phase_resolver.get("accepted", True))
        resolver_reason = str(phase_resolver.get("reason", "") or "")
        merged: List[CalibrationDelta] = []
        used = False
        for delta in deltas:
            if delta.target_channel == "W_CH1" and delta.matrix_type == "phase":
                if replace:
                    update = np.clip(matrix, -delta.limit, delta.limit)
                else:
                    update = np.clip(np.asarray(delta.delta, dtype=float) + matrix, -delta.limit, delta.limit)
                merged.append(CalibrationDelta(
                    target_channel=delta.target_channel,
                    matrix_type=delta.matrix_type,
                    delta=update,
                    limit=delta.limit,
                    accepted=bool(delta.accepted and resolver_accepted),
                    reason=resolver_reason or delta.reason,
                    path=delta.path,
                ))
                used = True
            else:
                merged.append(delta)
        if not used:
            merged.append(CalibrationDelta(
                "W_CH1",
                "phase",
                matrix,
                0.20,
                accepted=resolver_accepted,
                reason=resolver_reason,
            ))
        return merged

    def _dominant_subspace_vector(self, subspace_rho: np.ndarray) -> Optional[np.ndarray]:
        herm = 0.5 * (subspace_rho + subspace_rho.conj().T)
        try:
            eigvals, eigvecs = np.linalg.eigh(herm)
        except np.linalg.LinAlgError:
            return None
        idx = int(np.argmax(np.real(eigvals)))
        if float(np.real(eigvals[idx])) <= 1e-9:
            return None
        vec = np.asarray(eigvecs[:, idx], dtype=complex)
        ref = int(np.argmax(np.abs(vec)))
        if abs(vec[ref]) > 1e-12:
            vec = vec * np.exp(-1j * np.angle(vec[ref]))
        return vec

    def _subspace_coherence(self, subspace_rho: np.ndarray) -> float:
        sub = np.asarray(subspace_rho, dtype=complex)
        offdiag = sub - np.diag(np.diag(sub))
        return float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0

    def _wch1_phase_pattern_matrix(self, values: np.ndarray, system_type: str, size: int, n_modes: int) -> np.ndarray:
        group_size = 3 if system_type == "3qubit" else 4
        groups_per_row = 5 if system_type == "3qubit" else 4
        matrix = np.zeros((size, size), dtype=float)
        clipped = np.clip(np.asarray(values, dtype=float)[:n_modes], -0.20, 0.20)
        for row in range(size):
            for group in range(groups_per_row):
                col_start = group * group_size
                for offset in range(min(group_size, n_modes)):
                    col = col_start + offset
                    if col < size:
                        matrix[row, col] = clipped[offset]
        return matrix

    def _wrap_phase(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    def _raw_to_deltas(
        self,
        raw: np.ndarray,
        system_type: str,
        size: int,
        n_modes: int,
        target_state: Optional[Dict[str, Any]] = None,
    ) -> List[CalibrationDelta]:
        raw = np.asarray(raw, dtype=float).reshape(-1)
        base_size = n_modes * 4 + 2
        if raw.size < base_size:
            raw = np.pad(raw, (0, base_size - raw.size))
        phase_cell_limit = 0.20
        active_mask = np.ones(n_modes, dtype=bool)
        if isinstance(target_state, dict):
            phase_cell_limit = float(np.deg2rad(float(target_state.get("phase_cell_limit_deg", np.rad2deg(0.20)))))
            amps = np.asarray(
                target_state.get("amplitudes", target_state.get("basis_amp", [])),
                dtype=float,
            ).reshape(-1)
            if amps.size:
                active_mask = np.ones(n_modes, dtype=bool)
                active_mask[:min(n_modes, amps.size)] = np.abs(amps[:n_modes]) > 1e-9

        def square(values: np.ndarray, limit: float, zero_inactive: bool = False) -> np.ndarray:
            mat = np.zeros((size, size), dtype=float)
            vals = np.clip(values[:n_modes], -1.0, 1.0)
            if zero_inactive:
                vals = vals.copy()
                vals[~active_mask] = 0.0
            mat[:n_modes, :n_modes] = np.diag(vals * limit)
            return mat

        w_phase = raw[0:n_modes]
        r_phase = raw[n_modes:2 * n_modes]
        w_amp = raw[2 * n_modes:3 * n_modes]
        r_amp = raw[3 * n_modes:4 * n_modes]
        common_w = float(np.clip(raw[4 * n_modes], -1.0, 1.0) * phase_cell_limit)
        common_r = float(np.clip(raw[4 * n_modes + 1], -1.0, 1.0) * phase_cell_limit)
        wch1_pointwise = None
        if raw.size >= base_size + size * size:
            point_values = np.asarray(raw[base_size:base_size + size * size], dtype=float)
            if float(np.linalg.norm(point_values)) > 1e-12:
                wch1_pointwise = np.clip(point_values.reshape(size, size), -1.0, 1.0) * phase_cell_limit

        return [
            CalibrationDelta("W_CH3", "phase", square(w_phase, 0.08), 0.08),
            CalibrationDelta("R_CH3", "phase", square(r_phase, 0.08), 0.08),
            CalibrationDelta("W_CH3", "amp", square(w_amp, 0.03, zero_inactive=True), 0.03),
            CalibrationDelta("R_CH3", "amp", square(r_amp, 0.03, zero_inactive=True), 0.03),
            CalibrationDelta(
                "W_CH1",
                "phase",
                wch1_pointwise if wch1_pointwise is not None else np.full((size, size), common_w),
                phase_cell_limit,
            ),
            CalibrationDelta("R_CH1", "phase", np.full((size, size), common_r), phase_cell_limit),
        ]

    def _physics_constraint_settings(self, system_type: str) -> Dict[str, Any]:
        size = 15 if system_type == "3qubit" else 16
        return {
            "phase_cell_limit_rad": 0.20,
            "phase_cell_limit_deg": float(np.rad2deg(0.20)),
            "ch3_phase_limit_rad": 0.08,
            "amp_relative_limit": 0.03,
            "amp_square_bounds": [0.7, 1.3],
            "inactive_amp_cells_zero": True,
            "delta_clipping": "tanh_normalized_then_limit_scaled",
            "phase_map_shape": [size, size],
        }

    def _label_vector(self, sample: FeedbackSample) -> np.ndarray:
        n_modes = 3 if sample.system_type == "3qubit" else 4
        size = 15 if sample.system_type == "3qubit" else 16
        base_size = n_modes * 4 + 2
        values = np.zeros(base_size + size * size, dtype=float)
        if not sample.accepted_delta:
            return values
        offsets = {
            ("W_CH3", "phase"): 0,
            ("R_CH3", "phase"): n_modes,
            ("W_CH3", "amp"): 2 * n_modes,
            ("R_CH3", "amp"): 3 * n_modes,
        }
        for delta in sample.accepted_delta:
            key = (delta.target_channel, delta.matrix_type)
            if key in offsets:
                values[offsets[key]:offsets[key] + n_modes] = np.diag(delta.delta)[:n_modes] / max(delta.limit, 1e-9)
            elif key == ("W_CH1", "phase"):
                delta_matrix = np.asarray(delta.delta, dtype=float)
                values[4 * n_modes] = np.mean(delta_matrix) / max(delta.limit, 1e-9)
                if delta_matrix.shape == (size, size):
                    values[base_size:base_size + size * size] = delta_matrix.reshape(-1) / max(delta.limit, 1e-9)
            elif key == ("R_CH1", "phase"):
                values[4 * n_modes + 1] = np.mean(delta.delta) / max(delta.limit, 1e-9)
        return values


def _state_dict_to_json(state_dict) -> Dict[str, Any]:
    return {
        key: value.detach().cpu().tolist()
        for key, value in state_dict.items()
    }


def _state_dict_from_json(payload: Dict[str, Any], device=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to restore neural feedback state")
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in payload.items()
    }


def train_feedback_model(
    system_type: str,
    samples_dir: str = "ml_recon/data/feedback_samples",
    model_path: str = "",
    physics_prior_blend: Optional[float] = None,
    neural_mode: str = "auto",
    model_kind: str = "auto",
    device: str = "auto",
    hybrid_epochs: Optional[int] = None,
    hybrid_batch_size: Optional[int] = None,
    hybrid_hidden_dim: Optional[int] = None,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    encoder_lr: Optional[float] = None,
    minimum_synthetic_pretrain_samples: int = 100_000,
    allow_legacy_unknown_provenance: bool = False,
) -> Dict[str, Any]:
    """Train or refresh a feedback model from saved FeedbackSample files."""

    model = TrainedFeedbackModel(system_type)
    config: Dict[str, Any] = {"model_path": model_path or display_feedback_model_path(system_type)}
    if physics_prior_blend is not None:
        config["physics_prior_blend"] = float(physics_prior_blend)
    if neural_mode is not None:
        config["neural_mode"] = neural_mode
    if model_kind is not None:
        config["model_kind"] = model_kind
    if device is not None:
        config["device"] = device
    if hybrid_epochs is not None:
        config["hybrid_epochs"] = int(hybrid_epochs)
    if hybrid_batch_size is not None:
        config["hybrid_batch_size"] = int(hybrid_batch_size)
    if hybrid_hidden_dim is not None:
        config["hybrid_hidden_dim"] = int(hybrid_hidden_dim)
    if pretrain_epochs is not None:
        config["pretrain_epochs"] = int(pretrain_epochs)
    if finetune_epochs is not None:
        config["finetune_epochs"] = int(finetune_epochs)
    if batch_size is not None:
        config["batch_size"] = int(batch_size)
    if encoder_lr is not None:
        config["encoder_lr"] = float(encoder_lr)
    config["minimum_synthetic_pretrain_samples"] = int(minimum_synthetic_pretrain_samples)
    config["allow_legacy_unknown_provenance"] = bool(allow_legacy_unknown_provenance)
    saved_path = model.train(samples_dir, config)
    metadata = dict(model.metadata)
    if metadata.get("trained_physics_transfer_head"):
        message = "trained_full_wch1_physics_transfer_candidate"
    elif metadata.get("trained_hybrid_head"):
        message = "trained_hybrid_image_physics_head"
    elif metadata.get("trained_neural_head"):
        message = "trained_neural_head"
    else:
        message = "trained_linear_head"
    if not metadata.get("trained_linear_head"):
        message = (
            "No accepted_delta labels found; model metadata was saved and "
            "feedback inference will use physics_constrained_heuristic."
        )
    return {
        "model_path": saved_path,
        "system_type": model.system_type,
        "metadata": metadata,
        "message": message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline feedback delta model")
    parser.add_argument("--train", action="store_true", help="Train from saved FeedbackSample JSON files")
    parser.add_argument("--system-type", choices=["3qubit", "4qubit"], default="4qubit")
    parser.add_argument("--samples-dir", default="ml_recon/data/feedback_samples")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--physics-prior-blend", type=float, default=None)
    parser.add_argument(
        "--model-kind",
        choices=["auto", "linear", "neural_mlp", "hybrid_image_physics", "physics_transfer_gnn"],
        default="auto",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "directml", "cpu"], default="auto")
    parser.add_argument("--hybrid-epochs", type=int, default=None)
    parser.add_argument("--hybrid-batch-size", type=int, default=None)
    parser.add_argument("--hybrid-hidden-dim", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--minimum-synthetic-pretrain-samples", type=int, default=100000)
    parser.add_argument(
        "--allow-legacy-unknown-provenance", action="store_true",
        help="Explicitly treat labelled pre-v3 unknown-provenance samples as real transitions",
    )
    parser.add_argument(
        "--neural-mode",
        choices=["auto", "off", "neural", "adversarial", "force"],
        default="auto",
        help="Train a bidirectional/adversarial neural feedback head when labelled samples permit.",
    )
    args = parser.parse_args()

    if not args.train:
        parser.error("Use --train to train or refresh the feedback model")
    result = train_feedback_model(
        args.system_type,
        args.samples_dir,
        args.model_path,
        physics_prior_blend=args.physics_prior_blend,
        neural_mode=args.neural_mode,
        model_kind=args.model_kind,
        device=args.device,
        hybrid_epochs=args.hybrid_epochs,
        hybrid_batch_size=args.hybrid_batch_size,
        hybrid_hidden_dim=args.hybrid_hidden_dim,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        minimum_synthetic_pretrain_samples=args.minimum_synthetic_pretrain_samples,
        allow_legacy_unknown_provenance=args.allow_legacy_unknown_provenance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
