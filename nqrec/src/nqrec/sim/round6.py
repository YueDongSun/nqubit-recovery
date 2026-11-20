"""Round-6 synthetic dataset generator (DECISIONS J1-J17).

One row is one dataset: a commanded target state, one mechanism (or the nominal
device), and k rounds of four-cell counts on every setting of a design. The
trial count per setting is fixed by the budget (J13) and does not depend on the
row. Random parameters are drawn with numpy from the caller's generator; the
forward map and the count draws run in torch float64 on the requested device.

Designs (J12):
  hardware  N = 3 or 4, the V2.4 basis, operation i on its own cell group (H1).
  fixed     any N >= 2, N fixed cells read with spanning tree + trine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nqrec.physics.dlcz_forward import four_cell_probabilities, matched_read, retrieval_overlap, analyzer_on_support
from nqrec.physics.efficiency import (
    EfficiencyChain,
    chain_from_config,
    idler_efficiency_map,
    load_histogram_row,
    log_weight_pattern,
)
from nqrec.physics.excitation import background_for_g_si, chi_from_signal_rate, g_si_closed_form
from nqrec.physics.hardware_basis import hardware_basis
from nqrec.physics.layout import fixed_cell_block, hardware_groups
from nqrec.physics.mechanisms import ALL_CLASSES, coherence_on_support, draw_mechanisms
from nqrec.physics.schedule import spanning_tree_trine

ROOT = Path(__file__).resolve().parents[3]
HIGH_SIGNAL_RATE_CHECK = 0.04  # J0 check: s = 4% at the nominal background gives g_si about 4.9


@dataclass(frozen=True)
class Device:
    chain: EfficiencyChain
    signal_rate: float
    background: float
    background_idler: float
    g_si_nominal: float
    storage_time_us: float
    n_max: int
    phase_sign: int
    points_per_round: int
    idler_raw: np.ndarray
    signal_raw: np.ndarray
    priors: dict
    uniform_idler_map: bool

    @property
    def chi(self) -> float:
        return chi_from_signal_rate(self.signal_rate, self.chain.eta_signal)

    def eta_map(self, system: str) -> np.ndarray:
        if self.uniform_idler_map:
            return np.full(self.idler_raw.size, self.chain.measured_mean_idler)
        return idler_efficiency_map(self.chain, self.idler_raw, system).reshape(-1)

    def log_weight(self, system: str) -> np.ndarray:
        return log_weight_pattern(self.signal_raw, system).reshape(-1)

    def nominal(self) -> dict:
        return {"signal_rate": self.signal_rate, "background": self.background, "background_idler": self.background_idler,
                "storage_time_us": self.storage_time_us}


def load_device(path: str | Path, overrides: dict | None = None) -> Device:
    with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".json":
            cfg = json.load(handle)
        else:
            import yaml
            cfg = yaml.safe_load(handle)
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override {key} is not a config key")
        node[parts[-1]] = value
    chain = chain_from_config(cfg["efficiency"])
    signal_rate = float(cfg["signal_rate"])
    chi = chi_from_signal_rate(signal_rate, chain.eta_signal)
    g_si = float(cfg["g_si_nominal"])
    background = cfg["background"]
    if background is None:
        background = background_for_g_si(g_si, chi, chain.eta_signal, chain.measured_mean_idler)
    # Idler-arm background; null keeps the J0 assumption of equal arms.
    background_idler = background if cfg["background_idler"] is None else cfg["background_idler"]
    csv_path = ROOT / cfg["calibration_csv"]
    return Device(
        chain=chain,
        signal_rate=signal_rate,
        background=float(background),
        background_idler=float(background_idler),
        g_si_nominal=g_si,
        storage_time_us=float(cfg["storage_time_us"]),
        n_max=int(cfg["n_max"]),
        phase_sign=int(cfg["phase_sign"]),
        points_per_round=int(cfg["points_per_round"]),
        idler_raw=load_histogram_row(csv_path, "Idler Absolute"),
        signal_raw=load_histogram_row(csv_path, "Signal Absolute"),
        priors=dict(cfg["priors"]),
        uniform_idler_map=bool(cfg["uniform_idler_map"]),
    )


@dataclass(frozen=True)
class Design:
    kind: str
    n_modes: int
    beta: np.ndarray
    theta: np.ndarray
    support_modes: np.ndarray
    groups: np.ndarray | None
    system: str

    @property
    def n_settings(self) -> int:
        return int(self.beta.shape[0])

    @property
    def support(self) -> int:
        return int(self.beta.shape[1])


def make_design(kind: str, n_modes: int) -> Design:
    if kind == "hardware":
        basis = hardware_basis(n_modes)
        n_settings = basis["n_settings"]
        support_modes = np.broadcast_to(np.arange(n_modes), (n_settings, n_modes)).copy()
        system = "3qubit" if n_modes == 3 else "4qubit"
        return Design("hardware", n_modes, basis["beta"], basis["theta"], support_modes, hardware_groups(n_modes), system)
    if kind == "fixed":
        schedule = spanning_tree_trine(n_modes)
        beta_full, theta_full = schedule["beta"], schedule["theta"]
        n_settings = beta_full.shape[0]
        support_modes = np.zeros((n_settings, 2), dtype=np.int64)
        beta = np.zeros((n_settings, 2))
        theta = np.zeros((n_settings, 2))
        for s in range(n_settings):
            active = np.flatnonzero(beta_full[s] > 0)
            # Population settings repeat their cell in the padding slot with beta = 0.
            modes = np.array([active[0], active[-1]])
            support_modes[s] = modes
            beta[s, : active.size] = beta_full[s, active]
            theta[s, : active.size] = theta_full[s, active]
        return Design("fixed", n_modes, beta, theta, support_modes, None, "grid")
    raise ValueError(f"unknown design {kind!r}")


def trials_per_setting(device: Device, design: Design, shots_per_point: float) -> int:
    """Trials per setting per round (J13): T_point = shots / P_C(nominal matched read)."""
    matched = matched_read(device.chi, device.chain.eta_signal, device.chain.measured_mean_idler,
                           device.background, device.background_idler, device.n_max)
    t_point = shots_per_point / matched["coincidence"]
    return max(1, int(round(t_point * device.points_per_round / design.n_settings)))


def sample_targets(n_modes: int, n_rows: int, rng: np.random.Generator, log_a_std: float) -> np.ndarray:
    """General states for N <= 4, the equal-amplitude zero-phase W state above (J12)."""
    if n_modes > 4:
        return np.full((n_rows, n_modes), n_modes**-0.5, dtype=np.complex128)
    log_a = log_a_std * rng.standard_normal((n_rows, n_modes))
    weight = np.exp(2.0 * (log_a - log_a.max(axis=1, keepdims=True)))
    weight /= weight.sum(axis=1, keepdims=True)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_rows, n_modes))
    return np.sqrt(weight) * np.exp(1j * phase)
