"""Configuration objects for the self-contained deep training stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PACKAGE_ROOT / "runs"


@dataclass(frozen=True)
class SystemPhysicsConfig:
    system: str
    num_qubits: int
    hilbert_dim: int
    freq_grid_shape: Tuple[int, int]
    phase_map_shape: Tuple[int, int]
    crosstalk_shape: Tuple[int, int]
    freq_min_mhz: float = 85.0
    freq_max_mhz: float = 122.5
    max_freq_delta_mhz: float = 5.0
    phase_map_clamp_rad: float = 0.20
    rho_image_size: int = 16
    # v2 uses the complete 3Q/4Q vector (67/77 values) and pads to a shared
    # width.  The previous width 64 silently truncated physically named fields.
    physics_feature_dim: int = 80
    scalar_delta_dim: int = 8

    @property
    def freq_step_mhz(self) -> float:
        rows = max(self.freq_grid_shape[0] - 1, 1)
        return (self.freq_max_mhz - self.freq_min_mhz) / rows

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["freq_step_mhz"] = self.freq_step_mhz
        return data


SYSTEM_CONFIGS: Dict[str, SystemPhysicsConfig] = {
    "3qubit": SystemPhysicsConfig(
        system="3qubit",
        num_qubits=3,
        hilbert_dim=8,
        freq_grid_shape=(8, 8),
        phase_map_shape=(15, 15),
        crosstalk_shape=(6, 6),
    ),
    "4qubit": SystemPhysicsConfig(
        system="4qubit",
        num_qubits=4,
        hilbert_dim=16,
        freq_grid_shape=(16, 16),
        phase_map_shape=(16, 16),
        crosstalk_shape=(8, 8),
    ),
}


def normalize_system(system: str) -> str:
    value = str(system).strip().lower().replace("_", "").replace("-", "")
    if value in {"3q", "3", "w3", "3qubit", "threequbit"}:
        return "3qubit"
    if value in {"4q", "4", "w4", "4qubit", "fourqubit"}:
        return "4qubit"
    if value == "both":
        return "both"
    raise ValueError(f"Unknown system {system!r}; expected 3qubit, 4qubit, or both")


def get_system_config(system: str) -> SystemPhysicsConfig:
    key = normalize_system(system)
    if key == "both":
        raise ValueError("get_system_config requires a concrete system, not 'both'")
    return SYSTEM_CONFIGS[key]


@dataclass
class DeepTrainingConfig:
    task: str = "classifier"
    system: str = "both"
    preset: str = "smoke"
    num_samples: int = 512
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20260613
    hidden_dim: int = 192
    base_channels: int = 32
    dropout: float = 0.08
    physics_loss_weight: float = 0.25
    feedback_error_weight: float = 1.4
    phase_map_smoothness_weight: float = 0.05
    synthetic_noise_tier: str = "medium"
    grad_accum_steps: int = 1
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.995
    log_interval: int = 25
    run_id: str | None = None
    output_root: str = str(DEFAULT_RUN_ROOT)
    resume: bool = False
    num_workers: int = 0
    device: str = "auto"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "num_samples": 384,
        "epochs": 2,
        "batch_size": 64,
        "hidden_dim": 96,
        "base_channels": 16,
        "learning_rate": 5e-4,
        "log_interval": 10,
    },
    "basic": {
        "num_samples": 2048,
        "epochs": 6,
        "batch_size": 128,
        "hidden_dim": 160,
        "base_channels": 24,
        "learning_rate": 4e-4,
    },
    "4070ti_single_night": {
        "num_samples": 200000,
        "epochs": 24,
        "batch_size": 256,
        "hidden_dim": 384,
        "base_channels": 48,
        "learning_rate": 2.5e-4,
        "grad_accum_steps": 2,
        "dropout": 0.08,
        "synthetic_noise_tier": "high",
    },
}


def apply_preset(config: DeepTrainingConfig) -> DeepTrainingConfig:
    preset = PRESETS.get(config.preset)
    if not preset:
        raise ValueError(f"Unknown preset {config.preset!r}; available: {sorted(PRESETS)}")
    explicit = config.to_dict()
    default = DeepTrainingConfig().to_dict()
    merged = dict(explicit)
    for key, value in preset.items():
        if explicit.get(key) == default.get(key):
            merged[key] = value
    return DeepTrainingConfig(**merged)


def make_run_id(task: str, system: str, seed: int) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_system = normalize_system(system)
    return f"{stamp}_{task}_{safe_system}_seed{int(seed)}"
