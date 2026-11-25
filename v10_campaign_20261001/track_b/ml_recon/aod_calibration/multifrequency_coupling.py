"""Multi-frequency AOD coupling model for feedback delta prediction.

This module models the high-RF-power regime where several AOD frequencies are
enabled at once. It exposes a bounded delta interface that can be used before
real training data exists: an untrained model falls back to conservative,
physics-inspired heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from system_control.qubit_recon_files.system_specs import SYSTEM_SPECS

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


@dataclass
class MultiFrequencyFeatures:
    frequencies_mhz: np.ndarray
    amplitudes: np.ndarray
    phases_rad: np.ndarray
    active_mask: np.ndarray
    previous_phase_delta: Optional[np.ndarray] = None
    previous_amp_delta: Optional[np.ndarray] = None

    def vector(self) -> np.ndarray:
        freqs = np.asarray(self.frequencies_mhz, dtype=float)
        amps = np.asarray(self.amplitudes, dtype=float)
        phases = np.asarray(self.phases_rad, dtype=float)
        mask = np.asarray(self.active_mask, dtype=float)
        prev_phase = np.zeros_like(freqs) if self.previous_phase_delta is None else np.asarray(self.previous_phase_delta, dtype=float)
        prev_amp = np.zeros_like(freqs) if self.previous_amp_delta is None else np.asarray(self.previous_amp_delta, dtype=float)

        freq_norm = (freqs - np.min(freqs)) / (np.ptp(freqs) + 1e-9)
        total_power = np.sum((amps * mask) ** 2, keepdims=True)
        spacing = np.diff(freqs, prepend=freqs[0])
        spacing_norm = spacing / (np.max(np.abs(spacing)) + 1e-9)
        phase_sin = np.sin(phases)
        phase_cos = np.cos(phases)

        return np.concatenate([
            freq_norm,
            amps,
            phase_sin,
            phase_cos,
            mask,
            spacing_norm,
            np.repeat(total_power, len(freqs)),
            prev_phase,
            prev_amp,
        ]).astype(np.float32)


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
            self.activation = nn.SiLU()

        def forward(self, x):
            return self.activation(x + self.net(x))


    class MultiFrequencyCouplingNet(nn.Module):
        """Residual MLP for multi-frequency coupling and RF power compression."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, n_blocks: int = 3):
            super().__init__()
            blocks = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
            blocks.extend(ResidualBlock(hidden_dim) for _ in range(n_blocks))
            blocks.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return torch.tanh(self.net(x))
else:
    class MultiFrequencyCouplingNet:  # type: ignore
        pass


class MultiFrequencyCouplingModel:
    """Bounded correction model for W_CH3/R_CH3 local deltas and CH1 common phase."""

    def __init__(
        self,
        num_modes: int,
        phase_limit_rad: float = 0.08,
        amp_limit: float = 0.03,
        common_phase_limit_rad: float = 0.20,
        model_path: Optional[str] = None,
    ):
        self.num_modes = int(num_modes)
        self.phase_limit_rad = float(phase_limit_rad)
        self.amp_limit = float(amp_limit)
        self.common_phase_limit_rad = float(common_phase_limit_rad)
        self.model = None
        self.input_dim = self.num_modes * 9
        self.output_dim = self.num_modes * 4 + 2

        if model_path:
            self.load(model_path)

    def predict(self, features: MultiFrequencyFeatures) -> Dict[str, np.ndarray | float]:
        x = features.vector()
        if x.size != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {x.size}")

        source = "neural"
        if self.model is not None and TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                y = self.model(torch.tensor(x.reshape(1, -1), dtype=torch.float32)).numpy().reshape(-1)
        else:
            y = self._heuristic_raw(features)
            source = "heuristic"

        n = self.num_modes
        w_phase = y[0:n] * self.phase_limit_rad
        r_phase = y[n:2 * n] * self.phase_limit_rad
        w_amp = y[2 * n:3 * n] * self.amp_limit
        r_amp = y[3 * n:4 * n] * self.amp_limit
        common_w = float(y[4 * n] * self.common_phase_limit_rad)
        common_r = float(y[4 * n + 1] * self.common_phase_limit_rad)

        return {
            "W_CH3_phase": w_phase,
            "R_CH3_phase": r_phase,
            "W_CH3_amp": w_amp,
            "R_CH3_amp": r_amp,
            "W_CH1_global_phase": common_w,
            "R_CH1_global_phase": common_r,
            "prediction_source": source,
        }

    def save(self, model_path: str) -> None:
        if self.model is None or not TORCH_AVAILABLE:
            raise RuntimeError("No torch model is available to save")
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_modes": self.num_modes,
            "phase_limit_rad": self.phase_limit_rad,
            "amp_limit": self.amp_limit,
            "common_phase_limit_rad": self.common_phase_limit_rad,
        }, model_path)

    def load(self, model_path: str) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load MultiFrequencyCouplingModel")
        data = torch.load(model_path, map_location="cpu")
        self.num_modes = int(data["num_modes"])
        self.phase_limit_rad = float(data["phase_limit_rad"])
        self.amp_limit = float(data["amp_limit"])
        self.common_phase_limit_rad = float(data["common_phase_limit_rad"])
        self.input_dim = self.num_modes * 9
        self.output_dim = self.num_modes * 4 + 2
        self.model = MultiFrequencyCouplingNet(self.input_dim, self.output_dim)
        self.model.load_state_dict(data["state_dict"])

    def _heuristic_raw(self, features: MultiFrequencyFeatures) -> np.ndarray:
        freqs = np.asarray(features.frequencies_mhz, dtype=float)
        amps = np.asarray(features.amplitudes, dtype=float)
        mask = np.asarray(features.active_mask, dtype=float)
        active_power = (amps * mask) ** 2
        total_power = float(np.sum(active_power))
        centered_freq = (freqs - np.mean(freqs)) / (np.ptp(freqs) + 1e-9)
        compression = np.tanh((total_power - self.num_modes * 0.65) / max(self.num_modes, 1))
        local = -compression * centered_freq * mask
        amp_balance = (np.mean(amps[mask > 0]) if np.any(mask > 0) else np.mean(amps)) - amps
        amp_delta = np.tanh(amp_balance) * mask
        common = np.tanh(np.mean(np.sin(features.phases_rad)) + 0.25 * compression)
        return np.concatenate([
            local,
            -local,
            amp_delta,
            amp_delta,
            np.array([common, 0.5 * common], dtype=float),
        ]).astype(np.float32)


def build_default_features(system_type: str, state: Dict) -> MultiFrequencyFeatures:
    spec = SYSTEM_SPECS[system_type]
    n = int(spec["core_dim"])
    frequencies = np.linspace(
        float(spec["freq_start_mhz"]),
        float(spec["freq_end_mhz"]),
        int(spec["freq_steps"]),
    )[:n]
    amplitudes = np.asarray(state.get("amplitudes", [1.0] * n), dtype=float)
    if "phases_rad" in state:
        phases = np.asarray(state["phases_rad"], dtype=float)
    else:
        phases = np.deg2rad(np.asarray(state.get("phases_deg", [0.0] * n), dtype=float))
    return MultiFrequencyFeatures(
        frequencies_mhz=frequencies,
        amplitudes=amplitudes,
        phases_rad=phases,
        active_mask=np.ones(n, dtype=float),
    )
