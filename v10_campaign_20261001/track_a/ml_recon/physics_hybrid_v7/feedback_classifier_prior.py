"""Frozen classifier priors for the V7 feedback models.

The bridge consumes only a reconstructed density matrix and the requested
target state, both available before a feedback decision.  It never consumes a
future state, applied action, or target-derived training label.  Missing real
rho rows produce a zero latent with high uncertainty rather than a fabricated
density observation.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


CLASSIFIER_PRIOR_SOURCES = (
    "none",
    "quantum_rho_frozen",
    "state_family_frozen",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def embed_core_density(core_density: np.ndarray, system_type: str) -> np.ndarray:
    """Embed an N x N W-core rho into the repository's full Hilbert basis."""

    core = np.asarray(core_density, dtype=np.complex64)
    qubit_count = 3 if system_type == "3qubit" else 4 if system_type == "4qubit" else 0
    if qubit_count == 0 or core.shape[-2:] != (qubit_count, qubit_count):
        raise ValueError("core density and system_type disagree")
    full_shape = core.shape[:-2] + (2 ** qubit_count, 2 ** qubit_count)
    full = np.zeros(full_shape, dtype=np.complex64)
    indices = [2 ** (qubit_count - 1 - index) for index in range(qubit_count)]
    for row, full_row in enumerate(indices):
        for column, full_column in enumerate(indices):
            full[..., full_row, full_column] = core[..., row, column]
    return full


def target_full_density_from_features(
    target_features: "torch.Tensor",
    system_type: str,
) -> np.ndarray:
    qubit_count = 3 if system_type == "3qubit" else 4 if system_type == "4qubit" else 0
    if qubit_count == 0 or target_features.ndim != 2 or target_features.shape[1] != 16:
        raise ValueError("target feature and system contract changed")
    value = target_features.detach().float().cpu().numpy()
    amplitude = value[:, :4][:, :qubit_count]
    phase = np.arctan2(value[:, 4:8][:, :qubit_count], value[:, 8:12][:, :qubit_count])
    state = amplitude * np.exp(1j * phase)
    state /= np.maximum(np.linalg.norm(state, axis=1, keepdims=True), 1e-12)
    core = state[:, :, None] * state.conj()[:, None, :]
    return embed_core_density(core, system_type)


if nn is not None:

    class _QuantumRhoPrior(nn.Module):
        def __init__(self, system_type: str, path: Path) -> None:
            super().__init__()
            from ml_recon.quantum_representation.torch_model import (
                RhoPairTorchDiagnosisModel,
            )

            diagnosis = RhoPairTorchDiagnosisModel.load(str(path))
            self.inner = diagnosis.model
            self.feature_config = dict(diagnosis.feature_config)
            self.use_physics = bool(diagnosis.use_physics_features)
            self.physics_feature_schema = str(diagnosis.physics_feature_schema)
            self.physics_feature_dim = int(self.feature_config.get("physics_feature_dim", 0))
            self.latent_dim = int(self.feature_config.get("latent_dim", 32))
            self.mode_count = int(len(diagnosis.mode_labels))
            self.error_count = int(len(diagnosis.error_labels))
            self.system_type = system_type
            self.qubit_count = 3 if system_type == "3qubit" else 4
            self.register_buffer(
                "physics_mean",
                torch.as_tensor(np.asarray(diagnosis.physics_mean, dtype=np.float32)),
                persistent=False,
            )
            self.register_buffer(
                "physics_std",
                torch.as_tensor(np.asarray(diagnosis.physics_std, dtype=np.float32)),
                persistent=False,
            )

        def encode_numpy(
            self,
            current_full: np.ndarray,
            target_full: np.ndarray,
            device: "torch.device",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            from ml_recon.quantum_representation.torch_model import (
                rho_pair_physics_features,
                rho_pair_tensor,
            )

            image = torch.as_tensor(np.stack([
                rho_pair_tensor(target, current)
                for target, current in zip(target_full, current_full)
            ]), dtype=torch.float32, device=device)
            physics: Optional[torch.Tensor] = None
            if self.use_physics:
                raw = np.stack([
                    rho_pair_physics_features(
                        target,
                        current,
                        self.qubit_count,
                        schema=self.physics_feature_schema,
                    )
                    for target, current in zip(target_full, current_full)
                ])
                if raw.shape[1] != self.physics_feature_dim:
                    raise ValueError("quantum rho checkpoint physics schema changed")
                physics = torch.as_tensor(raw, dtype=torch.float32, device=device)
                physics = (
                    physics - self.physics_mean.to(device)
                ) / self.physics_std.to(device).clamp_min(1e-6)
            output = self.inner(image, physics)
            uncertainty = 0.5 * (
                _normalized_entropy(output["mode_logits"])
                + _normalized_entropy(output["error_logits"])
            )
            return output["latent"], uncertainty


    class _StateFamilyPrior(nn.Module):
        def __init__(self, system_type: str, path: Path) -> None:
            super().__init__()
            from ml_recon.quantum_representation.state_family import (
                QuantumStateFamilyTorchClassifier,
            )

            classifier = QuantumStateFamilyTorchClassifier.load(str(path))
            self.inner = classifier.model
            self.latent_dim = int(classifier.feature_config.get("hidden_dim", 160))
            self.family_count = int(len(classifier.family_labels))
            self.error_count = int(len(classifier.error_labels))
            self.system_type = system_type
            self.qubit_count = int(classifier.num_qubits)

        def encode_numpy(
            self,
            current_full: np.ndarray,
            target_full: np.ndarray,
            device: "torch.device",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            del target_full
            from ml_recon.quantum_representation.state_family import (
                rho_state_tensor,
                state_covariance_features,
            )

            image = torch.as_tensor(np.stack([
                rho_state_tensor(current) for current in current_full
            ]), dtype=torch.float32, device=device)
            covariance = torch.as_tensor(np.stack([
                state_covariance_features(current, self.qubit_count)
                for current in current_full
            ]), dtype=torch.float32, device=device)
            output = self.inner(image, covariance)
            uncertainty = 0.5 * (
                _normalized_entropy(output["family_logits"])
                + _normalized_entropy(output["error_logits"])
            )
            return output["latent"], uncertainty


    class FrozenFeedbackClassifierPrior(nn.Module):
        """System-specific frozen pretrained encoders with uncertainty output."""

        def __init__(
            self,
            *,
            source: str,
            model_dir: str | Path = "ml_recon/models",
            expected_latent_dim: int,
            device: "torch.device",
        ) -> None:
            super().__init__()
            if source not in CLASSIFIER_PRIOR_SOURCES or source == "none":
                raise ValueError(f"unsupported active classifier prior source: {source}")
            self.source = source
            self.expected_latent_dim = int(expected_latent_dim)
            self.model_dir = Path(model_dir)
            encoders: Dict[str, nn.Module] = {}
            paths: Dict[str, Path] = {}
            for system_type, suffix in (("3qubit", "3q"), ("4qubit", "4q")):
                if source == "quantum_rho_frozen":
                    path = self.model_dir / f"quantum_rho_encoder_{suffix}.pt"
                    encoder = _QuantumRhoPrior(system_type, path)
                else:
                    path = self.model_dir / f"state_family_classifier_{suffix}.pt"
                    encoder = _StateFamilyPrior(system_type, path)
                if encoder.latent_dim != self.expected_latent_dim:
                    raise ValueError(
                        f"{system_type} classifier latent {encoder.latent_dim} does not match "
                        f"configured {self.expected_latent_dim}"
                    )
                encoders[system_type] = encoder
                paths[system_type] = path
            self.encoders = nn.ModuleDict(encoders)
            self.paths = paths
            self.to(device)
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        @torch.no_grad()
        def encode(
            self,
            *,
            current_full_rho: "torch.Tensor",
            target_features: "torch.Tensor",
            rho_available_mask: "torch.Tensor",
            system_type: str,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            device = current_full_rho.device
            current = current_full_rho.detach().cpu().numpy()
            target = target_full_density_from_features(target_features, system_type)
            latent, entropy = self.encoders[system_type].encode_numpy(
                current, target, device,
            )
            mask = rho_available_mask.to(device=device, dtype=latent.dtype).reshape(-1)
            latent = latent * mask[:, None]
            uncertainty = entropy.to(dtype=latent.dtype) + (1.0 - mask) * 4.0
            return latent.detach(), uncertainty.detach()

        def audit_dict(self) -> Dict[str, Any]:
            return {
                "source": self.source,
                "frozen": True,
                "latent_dim": self.expected_latent_dim,
                "target_is_known_command_not_label": True,
                "missing_rho_latent_is_zero": True,
                "missing_rho_uncertainty": 4.0,
                "checkpoints": {
                    system_type: {
                        "path": str(path),
                        "sha256": _file_sha256(path),
                    }
                    for system_type, path in self.paths.items()
                },
            }


    def _normalized_entropy(logits: "torch.Tensor") -> "torch.Tensor":
        probability = torch.softmax(logits.float(), dim=-1).clamp_min(1e-8)
        entropy = -(probability * torch.log(probability)).sum(dim=-1)
        return entropy / max(math.log(float(logits.shape[-1])), 1e-8)


else:  # pragma: no cover

    class FrozenFeedbackClassifierPrior:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("classifier prior requires torch")


__all__ = [
    "CLASSIFIER_PRIOR_SOURCES",
    "FrozenFeedbackClassifierPrior",
    "embed_core_density",
    "target_full_density_from_features",
]
