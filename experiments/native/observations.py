"""Reconstruct public native analyzer transfers and origin-bound V2.4 observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .v24 import load_index, load_shard
from nqrec.physics.dlcz_forward import analyzer_on_support
import yaml
from nqrec.physics.efficiency import (
    chain_from_config,
    idler_efficiency_map,
    load_histogram_row,
)
from nqrec.physics.hardware_basis import hardware_basis
from nqrec.physics.layout import hardware_groups
from .basis_configs import build_3qubit_basis_config, build_4qubit_basis_config
from ml_recon.physics_hybrid_v10.effective_projector_observation import (
    make_public_observation,
)

SCHEMA = "v24_native_single_event_public_v1"
NOMINAL = {
    3: ("train", "validation"),
    4: ("train_target_family", "validation_target_family"),
}


@dataclass
class NativeSchedule:
    n: int
    labels: list
    cells: np.ndarray
    beta: np.ndarray
    theta_read: np.ndarray
    eta_cells: np.ndarray
    transfer: torch.Tensor
    eta_signal: float
    signal_background: float
    idler_background: float


@dataclass
class NativePanel:
    split: str
    counts: np.ndarray
    reads: np.ndarray
    command: np.ndarray
    trajectory: np.ndarray
    known_seconds: np.ndarray


def site_schedule(n, calibration_dir, histogram):
    """Build the signed transfer from the external TRAIN calibration and Idler Absolute map."""
    basis = hardware_basis(n)
    cells = hardware_groups(n)
    config = yaml.safe_load(
        (Path(calibration_dir) / f"round6_device_v24_{n}q.yaml").read_text()
    )
    chain = chain_from_config(config["efficiency"])
    raw = load_histogram_row(histogram, "Idler Absolute")
    system = f"{n}qubit"
    if config["uniform_idler_map"]:
        eta_map = np.full(raw.size, chain.measured_mean_idler)
    else:
        eta_map = idler_efficiency_map(chain, raw, system).reshape(-1)
    cfg = (build_3qubit_basis_config if n == 3 else build_4qubit_basis_config)()
    entries = (
        cfg["entries"]
        if n == 3
        else [e for cycle in cfg["cycles"] for e in cycle["entries"]]
    )
    amp = np.array([e["amp"] for e in entries])
    amp /= np.linalg.norm(amp, axis=1)[:, None]
    phases = np.array([e["phase"] for e in entries])
    np.testing.assert_allclose(amp, basis["beta"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        amp * np.exp(1j * phases),
        basis["beta"] * np.exp(1j * basis["theta"]),
        atol=1e-12,
        rtol=0,
    )
    if config["phase_sign"] != -1:
        raise ValueError("The native read-AOD phase sign must equal -1")
    eta = eta_map[cells]
    transfer = analyzer_on_support(
        torch.from_numpy(basis["beta"]),
        torch.from_numpy(basis["theta"]),
        torch.from_numpy(eta),
        config["phase_sign"],
    )
    labels = basis["labels"]
    return NativeSchedule(
        n,
        labels,
        cells,
        basis["beta"],
        basis["theta"],
        eta,
        transfer,
        chain.eta_signal,
        float(config["background"]),
        float(config["background_idler"]),
    )


def read_panel(data_root, schedule, split):
    n = schedule.n
    if split not in NOMINAL[n]:
        raise ValueError(
            "Only the registered nominal TRAIN and validation splits are admitted"
        )
    root = Path(data_root)
    path = root / f"observations/real/real_{n}qubit_{split}.h5"
    rows = load_shard(path, load_index(root / "index/index.sqlite"))
    with h5py.File(path, "r") as handle:
        if (
            not np.asarray(handle["operation_mask"]).all()
            or np.asarray(handle["real_action_supervision_mask"]).any()
        ):
            raise ValueError(
                "Unsupported missing operations or real action supervision"
            )
    if any(
        r["flat_order"] != schedule.labels or r["phase_sign"] != -1 or r["is_anchor"]
        for r in rows
    ):
        raise ValueError("Unsupported native operation order or sign")
    descriptors = rows[0]["operation_descriptors"]
    for i, desc in enumerate(descriptors):
        support = np.flatnonzero(schedule.beta[i] > 0)
        if desc["phase_unit"] != "rad" or list(support) != desc["cells"]:
            raise ValueError(
                "Per-operation support/phase units differ from native registry"
            )
        np.testing.assert_allclose(desc["amplitudes"], schedule.beta[i, support])
        np.testing.assert_allclose(desc["phases"], schedule.theta_read[i, support])
    amplitude = np.stack([r["target_amplitudes"] for r in rows])
    phase = np.stack([r["target_phases"] for r in rows])
    command = amplitude * np.exp(1j * phase)
    norm = np.linalg.norm(command, axis=1)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Invalid public preparation command")
    return NativePanel(
        split,
        np.stack([r["four_cell_counts"] for r in rows]),
        np.array([r["read_count"] for r in rows]),
        command / norm[:, None],
        np.array([r["trajectory_id"] for r in rows]),
        np.array([not r["exposure_unknown"] for r in rows]),
    )


def verify_packed(schedule, panel, bank):
    """Reject direction, registry, row, target or group mismatches against the shared panel."""
    for key, value in (
        ("cells", schedule.cells),
        ("beta", schedule.beta),
        ("theta_read", schedule.theta_read),
        ("eta_cells", schedule.eta_cells),
        ("counts", panel.counts),
        ("read_trials", panel.reads),
        ("target", panel.command),
    ):
        if not np.allclose(bank[key], value, atol=1e-12, rtol=0):
            raise ValueError(f"Packed native-field mismatch: {key}")
    grp = bank["trajectory_group"]
    if not np.array_equal(
        grp[:, None] == grp, panel.trajectory[:, None] == panel.trajectory
    ):
        raise ValueError(
            "Packed trajectory grouping differs from the linked repeat groups"
        )


def native_public_observation(
    panel, schedule, indices, *, signal_rate, idler_gain, idler_background, visible_mask
):
    """One native [B,S,4] window with per-operation calibrated efficiency and independent READ."""
    transfer = schedule.transfer * np.sqrt(idler_gain)
    efficiency = transfer.abs().square().sum(-1)
    if (efficiency <= 0).any() or (efficiency > 1).any():
        raise ValueError("Native efficiency exceeds physical support")
    analyzers = transfer / efficiency.sqrt()[:, None]
    idx = np.atleast_1d(indices)
    B, O = len(idx), len(efficiency)
    fields = [
        torch.full_like(efficiency, x)
        for x in (signal_rate, schedule.eta_signal, schedule.signal_background)
    ]
    fields += [efficiency, torch.full_like(efficiency, idler_background)]
    calib = torch.stack(fields, -1)[None].expand(B, -1, -1)
    return make_public_observation(
        counts=torch.from_numpy(panel.counts[idx]).long(),
        shots=torch.from_numpy(
            np.broadcast_to(panel.reads[idx, None], (B, O)).copy()
        ).long(),
        analyzers=analyzers,
        prepared=torch.from_numpy(panel.command[idx]),
        calibration=calib,
        visible_mask=visible_mask,
    )
