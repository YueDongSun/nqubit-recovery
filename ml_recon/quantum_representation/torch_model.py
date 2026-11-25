"""PyTorch density-matrix pair encoder for excitation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .features import density_fidelity_pure_target
from .synthetic import ERROR_SOURCES, SyntheticRhoSample
from .tomography_features import (
    DEFAULT_PHYSICS_FEATURE_SCHEMA,
    PHYSICS_FEATURE_SCHEMA_VERSION,
    PHYSICS_FEATURE_SCHEMAS,
    RHO_PAIR_TENSOR_CHANNELS,
    entanglement_signature_features,
    rho_pair_distance_features,
    rho_pair_physics_feature_names_by_schema,
    rho_pair_physics_features_by_schema,
    rho_pair_legacy_diff_tensor,
    rho_pair_tomography_tensor,
    validate_density_pair,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    random_split = None
    TORCH_AVAILABLE = False


DEFAULT_ERROR_LABELS = list(ERROR_SOURCES)
IMAGE_RECOGNITION_ARCHITECTURES = {"legacy_cnn", "residual_attention", "error_aware_attention"}


def default_mode_labels(num_qubits: int) -> List[str]:
    return [f"mode_{idx}" for idx in range(1, int(num_qubits) + 1)]


def default_w_target(num_qubits: int) -> np.ndarray:
    dim = 2 ** int(num_qubits)
    psi = np.zeros(dim, dtype=complex)
    for i in range(int(num_qubits)):
        psi[2 ** (int(num_qubits) - 1 - i)] = 1.0 / np.sqrt(num_qubits)
    return np.outer(psi, psi.conj())


def rho_pair_tensor(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Build a 12-channel tensor from target/measured and tomographic comparison maps."""

    validate_density_pair(rho_target, rho_measured)
    return rho_pair_tomography_tensor(rho_target, rho_measured)


def rho_pair_legacy_tensor(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Build the historical raw-diff tensor for legacy checkpoints."""

    return rho_pair_legacy_diff_tensor(rho_target, rho_measured)


def rho_pair_physics_features(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int,
    schema: str = DEFAULT_PHYSICS_FEATURE_SCHEMA,
) -> np.ndarray:
    """Physics-guided scalar features for the rho-pair diagnosis network."""

    return rho_pair_physics_features_by_schema(
        rho_target,
        rho_measured,
        int(num_qubits),
        schema=schema,
    ).astype(np.float32)


def rho_pair_physics_feature_names(
    num_qubits: int,
    schema: str = DEFAULT_PHYSICS_FEATURE_SCHEMA,
) -> List[str]:
    return rho_pair_physics_feature_names_by_schema(int(num_qubits), schema=schema)


if TORCH_AVAILABLE:

    def _norm_groups(channels: int) -> int:
        channels = int(channels)
        for groups in (8, 4, 2):
            if channels % groups == 0:
                return groups
        return 1


    class ConvNormAct(nn.Module):
        """Image-recognition style Conv2D block for density-matrix maps."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
        ):
            super().__init__()
            if padding is None:
                padding = kernel_size // 2
            self.block = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    stride=int(stride),
                    padding=int(padding),
                    bias=False,
                ),
                nn.GroupNorm(_norm_groups(out_channels), int(out_channels)),
                nn.SiLU(),
            )

        def forward(self, x):
            return self.block(x)


    class SqueezeExcitation2d(nn.Module):
        """Channel-attention block used by the residual image encoder."""

        def __init__(self, channels: int, reduction: int = 4):
            super().__init__()
            hidden = max(4, int(channels) // int(reduction))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.gate = nn.Sequential(
                nn.Conv2d(int(channels), hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden, int(channels), kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.gate(self.pool(x))


    class SpatialAttention2d(nn.Module):
        """Small spatial-attention gate over density-matrix image locations."""

        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            mean_map = torch.mean(x, dim=1, keepdim=True)
            max_map = torch.amax(x, dim=1, keepdim=True)
            return x * self.gate(torch.cat([mean_map, max_map], dim=1))


    class MultiScaleConvStem(nn.Module):
        """Parallel 1x1/3x3/5x5 filters before residual processing."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            branch_channels = max(4, int(out_channels) // 2)
            self.branches = nn.ModuleList([
                ConvNormAct(in_channels, branch_channels, kernel_size=1, padding=0),
                ConvNormAct(in_channels, branch_channels, kernel_size=3),
                ConvNormAct(in_channels, branch_channels, kernel_size=5),
            ])
            self.project = ConvNormAct(branch_channels * 3, out_channels, kernel_size=1, padding=0)

        def forward(self, x):
            return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


    class ResidualAttentionBlock(nn.Module):
        """Residual Conv2D block with channel and spatial attention."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.08):
            super().__init__()
            self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
            self.conv2 = nn.Sequential(
                nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_norm_groups(out_channels), int(out_channels)),
            )
            self.channel_attention = SqueezeExcitation2d(out_channels)
            self.spatial_attention = SpatialAttention2d()
            self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
            if int(stride) != 1 or int(in_channels) != int(out_channels):
                self.shortcut = nn.Sequential(
                    nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=int(stride), bias=False),
                    nn.GroupNorm(_norm_groups(out_channels), int(out_channels)),
                )
            else:
                self.shortcut = nn.Identity()
            self.activation = nn.SiLU()

        def forward(self, x):
            residual = self.shortcut(x)
            out = self.conv2(self.conv1(x))
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
            out = self.dropout(out)
            return self.activation(out + residual)


    class ImageResidualEncoder(nn.Module):
        """Deeper image-style encoder for tomographic density-matrix maps."""

        def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.08):
            super().__init__()
            base = max(8, int(base_channels))
            self.base_channels = base
            self.output_dim = base * 8
            self.stem = MultiScaleConvStem(in_channels, base)
            self.blocks = nn.Sequential(
                ResidualAttentionBlock(base, base, stride=1, dropout=dropout),
                ResidualAttentionBlock(base, base * 2, stride=2, dropout=dropout),
                ResidualAttentionBlock(base * 2, base * 2, stride=1, dropout=dropout),
                ResidualAttentionBlock(base * 2, base * 4, stride=2, dropout=dropout),
                ResidualAttentionBlock(base * 4, base * 4, stride=1, dropout=dropout),
            )
            self.final_norm = nn.GroupNorm(_norm_groups(base * 4), base * 4)
            self.activation = nn.SiLU()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x):
            y = self.activation(self.final_norm(self.blocks(self.stem(x))))
            pooled = torch.cat([self.avg_pool(y), self.max_pool(y)], dim=1)
            return torch.flatten(pooled, 1)


    class RhoPairEncoder(nn.Module):
        """Small convolutional multi-task encoder for complex rho pairs."""

        def __init__(
            self,
            num_qubits: int = 4,
            num_modes: Optional[int] = None,
            num_errors: int = len(DEFAULT_ERROR_LABELS),
            latent_dim: int = 32,
            hidden_dim: int = 128,
            architecture: str = "residual_attention",
            base_channels: int = 32,
            dropout: float = 0.08,
            physics_feature_dim: int = 0,
            physics_dim: int = 64,
        ):
            super().__init__()
            if num_qubits not in {3, 4}:
                raise ValueError("num_qubits must be 3 or 4")
            architecture = str(architecture or "residual_attention").lower()
            if architecture not in IMAGE_RECOGNITION_ARCHITECTURES:
                raise ValueError(f"architecture must be one of {sorted(IMAGE_RECOGNITION_ARCHITECTURES)}")
            self.num_qubits = int(num_qubits)
            self.num_modes = int(num_modes or num_qubits)
            self.num_errors = int(num_errors)
            self.latent_dim = int(latent_dim)
            self.architecture = architecture
            self.base_channels = int(base_channels)
            self.dropout = float(dropout)
            self.physics_feature_dim = int(physics_feature_dim)
            self.physics_dim = int(physics_dim) if int(physics_feature_dim) > 0 else 0

            if architecture == "legacy_cnn":
                self.encoder = nn.Sequential(
                    nn.Conv2d(12, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(4, 32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SiLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                encoder_dim = 128
            else:
                self.encoder = ImageResidualEncoder(12, base_channels=base_channels, dropout=dropout)
                encoder_dim = int(self.encoder.output_dim)

            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )
            if self.physics_feature_dim > 0:
                self.physics_proj = nn.Sequential(
                    nn.Linear(self.physics_feature_dim, self.physics_dim),
                    nn.LayerNorm(self.physics_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.physics_dim, self.physics_dim),
                    nn.LayerNorm(self.physics_dim),
                    nn.SiLU(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(latent_dim + self.physics_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.SiLU(),
                )
            else:
                self.physics_proj = None
                self.fusion = None
            self.mode_head = nn.Linear(latent_dim, self.num_modes)
            self.error_head = nn.Linear(latent_dim, self.num_errors)
            self.fidelity_head = nn.Linear(latent_dim, 1)
            self.purity_head = nn.Linear(latent_dim, 1)

        def forward(self, x, physics=None):
            latent = self.proj(self.encoder(x))
            if self.physics_proj is not None:
                if physics is None:
                    physics = torch.zeros(
                        (latent.shape[0], self.physics_feature_dim),
                        dtype=latent.dtype,
                        device=latent.device,
                    )
                physics_latent = self.physics_proj(physics.float())
                latent = self.fusion(torch.cat([latent, physics_latent], dim=1))
            return {
                "latent": latent,
                "mode_logits": self.mode_head(latent),
                "error_logits": self.error_head(latent),
                "fidelity": torch.sigmoid(self.fidelity_head(latent)).squeeze(-1),
                "purity": torch.sigmoid(self.purity_head(latent)).squeeze(-1),
            }

else:

    class ImageResidualEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available")

    class RhoPairEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available")


class RhoPairDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SyntheticRhoSample],
        error_labels: Sequence[str],
        num_qubits: int = 4,
        use_physics_features: bool = True,
        physics_feature_schema: str = DEFAULT_PHYSICS_FEATURE_SCHEMA,
        physics_mean: Optional[np.ndarray] = None,
        physics_std: Optional[np.ndarray] = None,
        cache_tensors: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        self.samples = list(samples)
        self.error_labels = list(error_labels)
        self.num_qubits = int(num_qubits)
        self.use_physics_features = bool(use_physics_features)
        self.physics_feature_schema = str(physics_feature_schema or DEFAULT_PHYSICS_FEATURE_SCHEMA)
        self.cache_tensors = bool(cache_tensors)
        if self.use_physics_features:
            raw_physics = np.asarray([
                rho_pair_physics_features(
                    sample.rho_target,
                    sample.rho_measured,
                    self.num_qubits,
                    schema=self.physics_feature_schema,
                )
                for sample in self.samples
            ], dtype=np.float32)
            if physics_mean is None:
                physics_mean = np.mean(raw_physics, axis=0)
            if physics_std is None:
                physics_std = np.std(raw_physics, axis=0)
            self.physics_mean = np.asarray(physics_mean, dtype=np.float32)
            self.physics_std = np.maximum(np.asarray(physics_std, dtype=np.float32), 1e-6)
            self._physics_features = ((raw_physics - self.physics_mean) / self.physics_std).astype(np.float32)
        else:
            self.physics_mean = np.zeros(0, dtype=np.float32)
            self.physics_std = np.ones(0, dtype=np.float32)
            self._physics_features = np.zeros((len(self.samples), 0), dtype=np.float32)
        self.physics_feature_dim = int(self._physics_features.shape[1])
        self._cached_items = [self._build_item(idx) for idx in range(len(self.samples))] if self.cache_tensors else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._cached_items is not None:
            return self._cached_items[idx]
        return self._build_item(idx)

    def _build_item(self, idx):
        sample = self.samples[idx]
        x = rho_pair_tensor(sample.rho_target, sample.rho_measured)
        mode = int(sample.excitation_mode) - 1
        error = self.error_labels.index(sample.error_source)
        metrics = np.asarray([sample.fidelity, sample.purity], dtype=np.float32)
        return (
            torch.tensor(x.tolist(), dtype=torch.float32),
            torch.tensor(self._physics_features[idx].tolist(), dtype=torch.float32),
            torch.tensor(mode, dtype=torch.long),
            torch.tensor(error, dtype=torch.long),
            torch.tensor(metrics.tolist(), dtype=torch.float32),
        )


def _select_training_device(requested: str = "auto"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    requested = str(requested or "auto").lower()
    env_requested = (
        os.environ.get("QUANTUM_REPRESENTATION_DEVICE")
        or os.environ.get("QUBIT_RN_TORCH_DEVICE")
    )
    if requested == "auto" and env_requested:
        requested = str(env_requested).lower()
    if requested not in {"auto", "cuda", "directml", "cpu"}:
        raise ValueError("device must be auto, cuda, directml, or cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "directml":
        try:
            import torch_directml

            if torch_directml.is_available():
                return torch_directml.device()
        except Exception as exc:
            raise RuntimeError(f"DirectML was requested but is not available: {exc}") from exc
    if requested == "directml":
        raise RuntimeError("DirectML was requested but torch_directml reports no available device")
    return torch.device("cpu")


def _training_device_name(device) -> Optional[str]:
    label = str(device)
    if label.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except (AssertionError, RuntimeError):
            # Test collection or an external launcher may change
            # CUDA_VISIBLE_DEVICES after PyTorch cached is_available().  Device
            # metadata is optional and must not invalidate a completed train.
            return None
    if label.startswith("privateuseone"):
        try:
            import torch_directml

            name = torch_directml.device_name(0)
            return str(name).replace("\x00", "")
        except Exception:
            return "DirectML"
    return None


def _training_accelerator(device) -> str:
    label = str(device)
    if label.startswith("cuda"):
        return "cuda"
    if label.startswith("privateuseone"):
        return "directml"
    return "cpu"


@dataclass
class TorchTrainingConfig:
    num_qubits: int = 4
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    latent_dim: int = 32
    hidden_dim: int = 128
    seed: int = 1234
    validation_fraction: float = 0.2
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.10
    use_ema: bool = True
    ema_decay: float = 0.995
    architecture: str = "residual_attention"
    base_channels: int = 32
    dropout: float = 0.08
    use_physics_features: bool = True
    physics_feature_schema: str = DEFAULT_PHYSICS_FEATURE_SCHEMA
    physics_dim: int = 64
    enable_siamese_pretraining: bool = False
    siamese_epochs: int = 1
    siamese_embedding_dim: int = 8
    cache_tensors: bool = True
    device: str = "auto"
    early_stopping_patience: int = 0
    min_delta: float = 0.0
    lr_scheduler_patience: int = 0
    lr_scheduler_factor: float = 0.5


def train_torch_diagnosis_model(
    samples: Sequence[SyntheticRhoSample],
    output: str,
    config: Optional[TorchTrainingConfig] = None,
    mode_labels: Optional[Sequence[str]] = None,
    error_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train the PyTorch rho pair encoder and save a .pt checkpoint."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    if not samples:
        raise ValueError("At least one sample is required")

    cfg = config or TorchTrainingConfig()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = _select_training_device(cfg.device)

    mode_labels = list(mode_labels or default_mode_labels(cfg.num_qubits))
    error_labels = list(error_labels or DEFAULT_ERROR_LABELS)
    dataset = RhoPairDataset(
        samples,
        error_labels,
        num_qubits=cfg.num_qubits,
        use_physics_features=cfg.use_physics_features,
        physics_feature_schema=cfg.physics_feature_schema,
        cache_tensors=cfg.cache_tensors,
    )
    siamese_metrics = (
        _train_siamese_entanglement_embeddings(samples, error_labels, cfg, device)
        if bool(cfg.enable_siamese_pretraining)
        else {"enabled": False}
    )

    val_count = max(1, int(round(len(dataset) * float(cfg.validation_fraction)))) if len(dataset) > 4 else 1
    train_count = max(1, len(dataset) - val_count)
    if train_count + val_count > len(dataset):
        val_count = len(dataset) - train_count
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False)

    model = RhoPairEncoder(
        num_qubits=cfg.num_qubits,
        num_modes=len(mode_labels),
        num_errors=len(error_labels),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        architecture=cfg.architecture,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
        physics_feature_dim=dataset.physics_feature_dim,
        physics_dim=cfg.physics_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.lr_scheduler_factor),
            patience=int(cfg.lr_scheduler_patience),
        )
        if int(cfg.lr_scheduler_patience) > 0
        else None
    )
    ema_state = _initial_ema_state(model) if bool(cfg.use_ema) else None

    history = []
    best_score = None
    patience_score = None
    stale_epochs = 0
    stopped_early = False
    best_state_dict = None
    best_metrics = None
    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        seen = 0
        for x, physics, mode, error, metrics in train_loader:
            x = x.to(device, non_blocking=True)
            physics = physics.to(device, non_blocking=True)
            mode = mode.to(device, non_blocking=True)
            error = error.to(device, non_blocking=True)
            metrics = metrics.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_train, physics_train, mode_a, mode_b, error_a, error_b, metrics_train, mix_lam = _mixup_rho_batch(
                x.float(),
                physics.float(),
                mode,
                error,
                metrics,
                float(cfg.mixup_alpha),
            )
            out = model(x_train, physics_train)
            loss = (
                _mixed_ce(out["mode_logits"], mode_a, mode_b, mix_lam, float(cfg.label_smoothing))
                + _mixed_ce(out["error_logits"], error_a, error_b, mix_lam, float(cfg.label_smoothing))
                + F.mse_loss(out["fidelity"], metrics_train[:, 0])
                + F.mse_loss(out["purity"], metrics_train[:, 1])
            )
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                _update_ema_state(ema_state, model, float(cfg.ema_decay))
            running += float(loss.item()) * x.shape[0]
            seen += int(x.shape[0])
        metrics_out = _evaluate_with_optional_ema(model, val_loader, device, ema_state)
        metrics_out["epoch"] = epoch + 1
        metrics_out["train_loss"] = running / max(seen, 1)
        metrics_out["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics_out)
        score = _selection_score(metrics_out)
        if scheduler is not None:
            scheduler.step(score)
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = dict(metrics_out)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in _merged_ema_state(model, ema_state).items()
            }
        if patience_score is None or score > patience_score + float(cfg.min_delta):
            patience_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        if int(cfg.early_stopping_patience) > 0 and stale_epochs >= int(cfg.early_stopping_patience):
            stopped_early = True
            break

    training_metrics = {
        "num_samples": len(dataset),
        "train_samples": train_count,
        "validation_samples": val_count,
        "epochs": int(cfg.epochs),
        "epochs_completed": len(history),
        "device": str(device),
        "requested_device": str(cfg.device),
        "accelerator": _training_accelerator(device),
        "accelerator_device_name": _training_device_name(device),
        "cuda_device_name": _training_device_name(torch.device("cuda")) if torch.cuda.is_available() else None,
        "final": history[-1] if history else {},
        "best": best_metrics or (history[-1] if history else {}),
        "checkpoint_selection": "best_validation_score",
        "regularization": {
            "label_smoothing": float(cfg.label_smoothing),
            "mixup_alpha": float(cfg.mixup_alpha),
            "use_ema": bool(cfg.use_ema),
            "ema_decay": float(cfg.ema_decay),
        },
        "architecture": {
            "name": str(cfg.architecture),
            "base_channels": int(cfg.base_channels),
            "hidden_dim": int(cfg.hidden_dim),
            "latent_dim": int(cfg.latent_dim),
            "dropout": float(cfg.dropout),
            "use_physics_features": bool(cfg.use_physics_features),
            "physics_feature_schema": str(dataset.physics_feature_schema),
            "physics_feature_schema_version": int(
                PHYSICS_FEATURE_SCHEMA_VERSION.get(str(dataset.physics_feature_schema), 0)
            ),
            "physics_feature_dim": int(dataset.physics_feature_dim),
            "physics_dim": int(cfg.physics_dim),
            "image_blocks": "multiscale_stem+5_residual_attention_blocks",
            "siamese_entanglement": siamese_metrics,
        },
        "convergence": {
            "stopped_early": bool(stopped_early),
            "stop_epoch": int(history[-1]["epoch"]) if history else 0,
            "early_stopping_patience": int(cfg.early_stopping_patience),
            "min_delta": float(cfg.min_delta),
            "lr_scheduler_patience": int(cfg.lr_scheduler_patience),
            "lr_scheduler_factor": float(cfg.lr_scheduler_factor),
            "best_score": float(best_score) if best_score is not None else None,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        "val_std": _history_metric_std(history),
        "history": history,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_state_dict = best_state_dict or {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    checkpoint = {
        "model_state_dict": final_state_dict,
        "num_qubits": int(cfg.num_qubits),
        "mode_labels": mode_labels,
        "error_labels": error_labels,
        "feature_config": {
            "input": "rho_target/rho_measured/tomographic_pair_maps",
            "representation": "tomographic_pair_tensor_v2",
            "tensor_channels": list(RHO_PAIR_TENSOR_CHANNELS),
            "num_input_channels": 12,
            "uses_raw_rho_diff": False,
            "latent_dim": int(cfg.latent_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "architecture": str(cfg.architecture),
            "base_channels": int(cfg.base_channels),
            "dropout": float(cfg.dropout),
            "use_physics_features": bool(cfg.use_physics_features),
            "physics_feature_schema": str(dataset.physics_feature_schema),
            "physics_feature_schema_version": int(
                PHYSICS_FEATURE_SCHEMA_VERSION.get(str(dataset.physics_feature_schema), 0)
            ),
            "physics_feature_dim": int(dataset.physics_feature_dim),
            "physics_dim": int(cfg.physics_dim),
            "physics_feature_names": (
                rho_pair_physics_feature_names(cfg.num_qubits, schema=dataset.physics_feature_schema)
                if dataset.physics_feature_dim
                else []
            ),
            "physics_mean": dataset.physics_mean.astype(float).tolist(),
            "physics_std": dataset.physics_std.astype(float).tolist(),
            "image_blocks": "multiscale_stem+5_residual_attention_blocks",
            "siamese_entanglement": siamese_metrics,
        },
        "training_metrics": training_metrics,
    }
    torch.save(checkpoint, output_path)
    return {
        "output": str(output_path),
        "backend": "torch",
        "num_qubits": int(cfg.num_qubits),
        **training_metrics,
    }


def evaluate_torch_model(model, data_loader, device=None) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()
    total = 0
    mode_correct = 0
    error_correct = 0
    fidelity_abs = 0.0
    purity_abs = 0.0
    with torch.no_grad():
        for x, physics, mode, error, metrics in data_loader:
            x = x.to(device, non_blocking=True)
            physics = physics.to(device, non_blocking=True)
            mode = mode.to(device, non_blocking=True)
            error = error.to(device, non_blocking=True)
            metrics = metrics.to(device, non_blocking=True)
            out = model(x.float(), physics.float())
            mode_pred = torch.argmax(out["mode_logits"], dim=1)
            error_pred = torch.argmax(out["error_logits"], dim=1)
            total += int(x.shape[0])
            mode_correct += int(torch.sum(mode_pred == mode).item())
            error_correct += int(torch.sum(error_pred == error).item())
            fidelity_abs += float(torch.sum(torch.abs(out["fidelity"] - metrics[:, 0])).item())
            purity_abs += float(torch.sum(torch.abs(out["purity"] - metrics[:, 1])).item())
    return {
        "mode_accuracy": mode_correct / max(total, 1),
        "error_source_accuracy": error_correct / max(total, 1),
        "mean_abs_fidelity_error": fidelity_abs / max(total, 1),
        "mean_abs_purity_error": purity_abs / max(total, 1),
    }


def _mixup_rho_batch(x, physics, mode, error, metrics, alpha: float):
    if alpha <= 0.0 or int(x.shape[0]) < 2:
        return x, physics, mode, mode, error, error, metrics, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(int(x.shape[0]), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_physics = lam * physics + (1.0 - lam) * physics[perm]
    mixed_metrics = lam * metrics + (1.0 - lam) * metrics[perm]
    return mixed_x, mixed_physics, mode, mode[perm], error, error[perm], mixed_metrics, lam


def _mixed_ce(logits, target_a, target_b, lam: float, smoothing: float):
    loss_a = F.cross_entropy(logits, target_a, label_smoothing=float(smoothing))
    if lam >= 0.999:
        return loss_a
    loss_b = F.cross_entropy(logits, target_b, label_smoothing=float(smoothing))
    return float(lam) * loss_a + (1.0 - float(lam)) * loss_b


def _train_siamese_entanglement_embeddings(
    samples: Sequence[SyntheticRhoSample],
    error_labels: Sequence[str],
    cfg: TorchTrainingConfig,
    device,
) -> Dict[str, Any]:
    """Optional contrastive pretraining over deterministic entanglement signatures."""

    if not samples or not TORCH_AVAILABLE:
        return {"enabled": False, "reason": "torch unavailable or no samples"}
    raw = np.asarray([
        entanglement_signature_features(
            sample.rho_target,
            sample.rho_measured,
            int(cfg.num_qubits),
            include_geometric=True,
        )
        for sample in samples
    ], dtype=np.float32)
    labels = np.asarray([list(error_labels).index(sample.error_source) for sample in samples], dtype=np.int64)
    if raw.shape[0] < 2:
        return {"enabled": False, "reason": "at least two samples are required"}
    mean = np.mean(raw, axis=0)
    std = np.maximum(np.std(raw, axis=0), 1e-6)
    x = torch.tensor(((raw - mean) / std).tolist(), dtype=torch.float32, device=device)
    y = torch.tensor(labels.tolist(), dtype=torch.long, device=device)
    hidden = max(16, int(cfg.siamese_embedding_dim) * 4)
    embedder = nn.Sequential(
        nn.Linear(raw.shape[1], hidden),
        nn.SiLU(),
        nn.Linear(hidden, int(cfg.siamese_embedding_dim)),
    ).to(device)
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)
    pair_count = max(8, min(256, raw.shape[0] * 4))

    def pair_loss(seed_offset: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + int(seed_offset))
        left = torch.randint(0, raw.shape[0], (pair_count,), generator=generator).to(device)
        right = torch.randint(0, raw.shape[0], (pair_count,), generator=generator).to(device)
        same = (y[left] == y[right]).float()
        z_left = F.normalize(embedder(x[left]), dim=-1)
        z_right = F.normalize(embedder(x[right]), dim=-1)
        distance = torch.linalg.norm(z_left - z_right, dim=-1)
        margin = 1.0
        return torch.mean(same * distance * distance + (1.0 - same) * F.relu(margin - distance) ** 2)

    with torch.no_grad():
        initial_loss = float(pair_loss(17).detach().cpu().item())
    history = []
    for epoch in range(max(1, int(cfg.siamese_epochs))):
        embedder.train()
        optimizer.zero_grad(set_to_none=True)
        loss = pair_loss(1000 + epoch)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))
    final_loss = history[-1] if history else initial_loss
    return {
        "enabled": True,
        "training_mode": "siamese_entanglement",
        "feature_schema": "entanglement_signature_v1",
        "embedding_dim": int(cfg.siamese_embedding_dim),
        "epochs": max(1, int(cfg.siamese_epochs)),
        "initial_loss": initial_loss,
        "final_loss": float(final_loss),
        "loss_delta": float(initial_loss - final_loss),
        "history": history,
    }


def _initial_ema_state(model) -> Dict[str, Any]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


def _update_ema_state(ema_state: Dict[str, Any], model, decay: float) -> None:
    state = model.state_dict()
    for key, value in state.items():
        if key not in ema_state or not torch.is_floating_point(value):
            continue
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _evaluate_with_optional_ema(model, data_loader, device, ema_state):
    if ema_state is None:
        return evaluate_torch_model(model, data_loader, device=device)
    current = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    merged = dict(current)
    merged.update(ema_state)
    model.load_state_dict(merged)
    try:
        return evaluate_torch_model(model, data_loader, device=device)
    finally:
        model.load_state_dict(current)


def _merged_ema_state(model, ema_state):
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    if ema_state:
        state.update(ema_state)
    return state


def _history_metric_std(history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "mode_accuracy",
        "error_source_accuracy",
        "mean_abs_fidelity_error",
        "mean_abs_purity_error",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in history], dtype=float)
        values = values[np.isfinite(values)]
        out[key] = float(np.std(values)) if values.size else float("nan")
    return out


def _selection_score(metrics: Dict[str, float]) -> float:
    """Prefer reliable mode classification while keeping regression heads sane."""

    return (
        float(metrics.get("mode_accuracy", 0.0))
        + 0.25 * float(metrics.get("error_source_accuracy", 0.0))
        - 0.10 * float(metrics.get("mean_abs_fidelity_error", 1.0))
        - 0.10 * float(metrics.get("mean_abs_purity_error", 1.0))
    )


class RhoPairTorchDiagnosisModel:
    """Loaded .pt checkpoint wrapper with the same predict API as the JSON fallback."""

    def __init__(self, model, checkpoint: Dict[str, Any]):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        self.model = model
        self.checkpoint = checkpoint
        self.device = next(model.parameters()).device
        self.num_qubits = int(checkpoint["num_qubits"])
        self.mode_labels = list(checkpoint.get("mode_labels") or default_mode_labels(self.num_qubits))
        self.error_labels = list(checkpoint.get("error_labels") or DEFAULT_ERROR_LABELS)
        self.feature_config = dict(checkpoint.get("feature_config") or {})
        self.training_metrics = dict(checkpoint.get("training_metrics") or {})
        self.use_physics_features = bool(self.feature_config.get("use_physics_features", False))
        self.physics_feature_schema = _checkpoint_physics_feature_schema(
            self.feature_config,
            self.num_qubits,
        )
        self.physics_mean = np.asarray(self.feature_config.get("physics_mean") or [], dtype=np.float32)
        self.physics_std = np.maximum(
            np.asarray(self.feature_config.get("physics_std") or [], dtype=np.float32),
            1e-6,
        )
        self.backend = "torch"

    @classmethod
    def load(cls, path: str) -> "RhoPairTorchDiagnosisModel":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        checkpoint = _torch_load_checkpoint(path)
        num_qubits = int(checkpoint["num_qubits"])
        mode_labels = list(checkpoint.get("mode_labels") or default_mode_labels(num_qubits))
        error_labels = list(checkpoint.get("error_labels") or DEFAULT_ERROR_LABELS)
        feature_config = dict(checkpoint.get("feature_config") or {})
        model = RhoPairEncoder(
            num_qubits=num_qubits,
            num_modes=len(mode_labels),
            num_errors=len(error_labels),
            latent_dim=int(feature_config.get("latent_dim", 32)),
            hidden_dim=int(feature_config.get("hidden_dim", 128)),
            architecture=str(feature_config.get("architecture", "legacy_cnn")),
            base_channels=int(feature_config.get("base_channels", 32)),
            dropout=float(feature_config.get("dropout", 0.05)),
            physics_feature_dim=int(feature_config.get("physics_feature_dim", 0)),
            physics_dim=int(feature_config.get("physics_dim", 64)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return cls(model, checkpoint)

    def predict(self, rho_measured: np.ndarray, rho_target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if rho_target is None:
            rho_target = default_w_target(self.num_qubits)
        tensor_builder = rho_pair_legacy_tensor if _uses_legacy_rho_diff_tensor(self.feature_config) else rho_pair_tensor
        x = torch.tensor(
            tensor_builder(rho_target, rho_measured).tolist(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        physics = None
        raw_physics = np.zeros(0, dtype=np.float32)
        physics_schema = self.physics_feature_schema
        if self.use_physics_features:
            raw_physics = rho_pair_physics_features(
                rho_target,
                rho_measured,
                self.num_qubits,
                schema=physics_schema,
            )
            if self.physics_mean.size != raw_physics.size or self.physics_std.size != raw_physics.size:
                fallback = _matching_physics_feature_schema(
                    self.physics_mean.size or self.physics_std.size,
                    self.num_qubits,
                )
                if fallback is not None and fallback != physics_schema:
                    physics_schema = fallback
                    raw_physics = rho_pair_physics_features(
                        rho_target,
                        rho_measured,
                        self.num_qubits,
                        schema=physics_schema,
                    )
            if self.physics_mean.size != raw_physics.size or self.physics_std.size != raw_physics.size:
                raise ValueError("Checkpoint physics normalization shape does not match the feature vector")
            physics_values = (raw_physics - self.physics_mean) / self.physics_std
            physics = torch.tensor(
                physics_values.tolist(),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x, physics)
            mode_probs = torch.softmax(out["mode_logits"], dim=1).cpu()[0].tolist()
            error_probs = torch.softmax(out["error_logits"], dim=1).cpu()[0].tolist()
            latent = out["latent"].cpu()[0].tolist()
            fidelity = float(out["fidelity"].cpu()[0].item())
            purity = float(out["purity"].cpu()[0].item())

        return {
            "available": True,
            "backend": "torch",
            "device": str(self.device),
            "fidelity_estimate": fidelity,
            "purity_estimate": purity,
            "mode_probabilities": {
                label: float(prob)
                for label, prob in zip(self.mode_labels, mode_probs)
            },
            "error_source_probabilities": {
                label: float(prob)
                for label, prob in zip(self.error_labels, error_probs)
            },
            "unknown_probability": _unknown_probability(mode_probs, error_probs),
            "latent_vector": [float(value) for value in latent],
            "physics_feature_schema": physics_schema if self.use_physics_features else None,
            "physics_feature_names": (
                rho_pair_physics_feature_names(self.num_qubits, schema=physics_schema)
                if self.use_physics_features
                else []
            ),
            "physics_feature_vector": raw_physics.astype(float).tolist() if self.use_physics_features else [],
            "target_fidelity_proxy": density_fidelity_pure_target(rho_target, rho_measured),
            "tomographic_distance_features": [
                float(value)
                for value in rho_pair_distance_features(rho_target, rho_measured)
            ],
        }


def _unknown_probability(mode_probs: Iterable[float], error_probs: Iterable[float]) -> float:
    mode_probs = np.asarray(list(mode_probs), dtype=float)
    error_probs = np.asarray(list(error_probs), dtype=float)
    max_conf = float(min(np.max(mode_probs), np.max(error_probs)))
    return float(np.clip(1.0 - max_conf, 0.0, 1.0))


def _checkpoint_physics_feature_schema(feature_config: Dict[str, Any], num_qubits: int) -> str:
    schema = (
        feature_config.get("physics_feature_schema")
        or feature_config.get("feature_schema")
        or feature_config.get("physics_schema")
    )
    if schema and str(schema).lower() in PHYSICS_FEATURE_SCHEMAS:
        return str(schema).lower()
    expected = int(
        feature_config.get("physics_feature_dim")
        or len(feature_config.get("physics_mean") or [])
        or 0
    )
    matched = _matching_physics_feature_schema(expected, int(num_qubits))
    return matched or "physics_v1"


def _matching_physics_feature_schema(expected_dim: int, num_qubits: int) -> Optional[str]:
    expected = int(expected_dim)
    if expected <= 0:
        return None
    for schema in ("physics_v1", "physics_v2", "entanglement_signature_v1"):
        try:
            if len(rho_pair_physics_feature_names(int(num_qubits), schema=schema)) == expected:
                return schema
        except Exception:
            continue
    return None


def _uses_legacy_rho_diff_tensor(feature_config: Dict[str, Any]) -> bool:
    if bool(feature_config.get("uses_raw_rho_diff")):
        return True
    representation = str(feature_config.get("representation", "")).lower()
    if "legacy" in representation or "rho_diff" in representation:
        return True
    return str(feature_config.get("input", "")).lower() == "rho_target/rho_measured/rho_diff"


def _torch_load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
