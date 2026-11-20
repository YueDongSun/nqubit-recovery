"""Public projector observations for the archived and effective single-event laws."""

from __future__ import annotations

import torch

from .training_ideas_data import (
    PUBLIC_PROJECTOR_KEYS,
    PROJECTOR_READOUT_SCOPE,
    _descriptors,
    _layout,
)
from .training_ideas_geometry import public_control_candidates

EFFECTIVE_LAW = "effective_single_event_v1"
PUBLIC_SCHEMA = "projector_observation_v2"
CALIBRATION_FIELDS = (
    "total_signal_rate",
    "signal_efficiency",
    "signal_background",
    "idler_efficiency",
    "idler_background",
)


def make_public_observation(
    *, counts, shots, analyzers, prepared, calibration, visible_mask, readout_law=EFFECTIVE_LAW
):
    """Preserve the actual acquisition schedule, independent R, and visible counts only."""
    counts, shots, visible = map(torch.as_tensor, (counts, shots, visible_mask))
    device = counts.device
    if shots.device != device or visible.device != device:
        raise ValueError("counts, exposure and masks must share a device")
    if counts.ndim != 3 or counts.shape[-1] != 4 or shots.shape != counts.shape[:2]:
        raise ValueError("expected native [B,S,4] counts and independent [B,S] exposure")
    if visible.shape != shots.shape or visible.dtype != torch.bool or not visible.any(-1).all():
        raise ValueError("a Boolean visible mask and at least one acquisition are required")
    if counts.dtype != torch.int64 or shots.dtype != torch.int64:
        raise TypeError("native counts and exposures must be int64")
    if (counts < 0).any() or (shots < 0).any() or (shots[visible] <= 0).any():
        raise ValueError("invalid counts or exposures")
    totals = counts.sum(-1)
    if (
        not torch.equal(totals[visible], shots[visible])
        or ((totals != 0) & (totals != shots)).any()
    ):
        raise ValueError("native counts fail independent READ conservation")
    batch, settings = shots.shape
    analyzer = torch.as_tensor(analyzers, device=device)
    command = torch.as_tensor(prepared, device=device)
    if analyzer.ndim == 2:
        analyzer = analyzer.expand(batch, -1, -1)
    if command.ndim != 2 or command.shape[0] != batch:
        raise ValueError("a normalized public command [B,N] is required")
    n = command.shape[-1]
    if n < 3 or analyzer.shape != (batch, settings, n):
        raise ValueError("analyzer and command dimensions are incompatible")
    for vector in (analyzer, command):
        norm = vector.abs().square().sum(-1)
        if not torch.isfinite(norm).all() or not torch.allclose(
            norm, torch.ones_like(norm), atol=1e-6
        ):
            raise ValueError("public analyzer and command must be unit vectors")
    calib = torch.as_tensor(calibration, dtype=torch.float32, device=device)
    width = {EFFECTIVE_LAW: 5, PROJECTOR_READOUT_SCOPE: 3}.get(readout_law)
    if (
        width is None
        or not (calib.shape == (batch, width) or (readout_law == EFFECTIVE_LAW and calib.shape == (batch, settings, width)))
        or not torch.isfinite(calib).all()
        or ((calib < 0) | (calib > 1)).any()
    ):
        raise ValueError("unknown law or invalid calibration coordinates")
    if readout_law == EFFECTIVE_LAW and ((calib[..., 1] <= 0) | (calib[..., 0] < calib[..., 2])).any():
        raise ValueError("total signal and signal background/efficiency are inconsistent")
    feature, adjacency = _layout(n)

    def shared(value):
        return torch.as_tensor(value, device=device).unsqueeze(0).expand(batch, -1, -1)

    # A hyperedge is an actually measured two-mode setting support; retain the
    # command-wide hyperedge as in the archived classifier.
    supports = analyzer.abs() > 1e-8
    pair = supports & (supports.sum(-1, keepdim=True) == 2)
    # Stable common dimensions even for heterogeneous schedules; repeated supports
    # have no extra weight. Build per-row union, followed by the command hyperedge.
    same = (pair.unsqueeze(1) == pair.unsqueeze(2)).all(-1)
    first = pair.any(-1) & ~torch.tril(same, diagonal=-1).any(-1)
    edge_count = int(first.sum(-1).max())
    incidence = torch.zeros(batch, n, edge_count + 1, device=device)
    for row in range(batch):
        unique = pair[row, first[row]]
        incidence[row, :, : len(unique)] = unique.T.float()
    incidence[:, :, -1] = 1
    actions = public_control_candidates(batch, n, device=device, dtype=torch.float32)
    result = {
        "counts": torch.where(visible[..., None], counts, 0),
        "basis_descriptors": torch.from_numpy(_descriptors(analyzer.detach().cpu().numpy())).to(
            device
        ),
        "shots": shots,
        "setting_mask": visible,
        "setting_uncertainty": torch.where(visible, 0.5 * shots.float().clamp_min(1).rsqrt(), 0),
        "mode_features": shared(feature),
        "mode_mask": torch.ones(batch, n, dtype=torch.bool, device=device),
        "physical_adjacency": shared(adjacency),
        "correlation_adjacency": torch.zeros(batch, n, n, device=device),
        "incidence": incidence,
        "certificate_features": torch.zeros(batch, n, 10, device=device),
        "certificate_mask": torch.zeros(batch, n, dtype=torch.bool, device=device),
        "analyzer_re": analyzer.real.float(),
        "analyzer_im": analyzer.to(torch.complex64).imag,
        "prepared_re": command.real.float(),
        "prepared_im": command.to(torch.complex64).imag,
        "public_control_target": torch.stack((command.abs(), command.angle()), -1).float(),
        "public_candidate_actions": actions,
        "public_candidate_valid_mask": torch.ones(
            batch, actions.shape[1], dtype=torch.bool, device=device
        ),
        "public_noop_candidate_index": torch.zeros(batch, dtype=torch.long, device=device),
    }
    result["effective_calibration" if readout_law == EFFECTIVE_LAW else "readout_parameters"] = (
        calib
    )
    return result


def corrected_bank_observation(bank, *, visible_mask, role="k0_k1_decoder"):
    """Convert the PR45 generator output; hidden sector and truth arrays never enter the ABI."""
    if "metadata" in bank:
        meta = bank["metadata"]
    else:
        import json

        meta = json.loads(str(bank["metadata_json"].item()))
    if meta["schema"] != EFFECTIVE_LAW or tuple(meta["cell_order"]) != ("00", "10", "01", "11"):
        raise ValueError("unrecognized observation law or cell order")
    if not (meta["write_equals_read"] and meta["signal_rate_includes_background"]):
        raise ValueError("unknown exposure or total signal convention")
    if role not in {"k0_k1_decoder", "frozen_k2_sensitivity"}:
        raise ValueError("unknown corrected-bank consumption role")
    if meta.get("conditional_state_hypothesis") != "same_declared_state_for_S0_and_S1":
        raise ValueError("unknown conditional-branch state family")
    weight = meta.get("k2_source_probability")
    if not isinstance(weight, (float, int)) or not 0 <= weight <= 1:
        raise ValueError("an explicit finite K2 source probability is required")
    if weight == 0 and meta.get("k2_role") != "absent_in_k1_reference":
        raise ValueError("K2 role disagrees with zero sector weight")
    if weight > 0:
        if role != "frozen_k2_sensitivity":
            raise ValueError("positive K2 is outside the K0/K1 decoder family")
        if meta.get("k2_role") != "explicit_esm_sensitivity":
            raise ValueError("unknown positive-K2 sensitivity family")
        for key in ("k2_signal_event_probability","k2_local_fraction","k2_two_photon_single_event_acceptance"):
            value=meta.get(key)
            if not isinstance(value, (int,float)) or not 0<=value<=1:
                raise ValueError(f"missing or invalid positive-K2 assumption: {key}")
    public = meta["public_calibration"]
    if public["read_phase_sign"] != -1:
        raise ValueError("unrecognized analyzer phase convention")
    counts = torch.as_tensor(bank["counts"])
    coords = [meta["requested_signal_rate"], *(public[key] for key in CALIBRATION_FIELDS[1:])]
    return make_public_observation(
        counts=counts,
        shots=bank["read_opportunities"],
        visible_mask=visible_mask,
        analyzers=bank["analyzer_real"] + 1j * bank["analyzer_imag"],
        prepared=bank["prepared_real"] + 1j * bank["prepared_imag"],
        calibration=torch.tensor(coords).expand(counts.shape[0], -1),
    )


def archived_projector_observation(batch):
    """Convert the archived PR44 tensors without changing its observation law."""
    result = make_public_observation(
        counts=batch["counts"],
        shots=batch["shots"],
        analyzers=torch.complex(batch["analyzer_re"], batch["analyzer_im"]),
        prepared=torch.complex(batch["prepared_re"], batch["prepared_im"]),
        calibration=batch["readout_parameters"],
        visible_mask=batch["setting_mask"],
        readout_law=PROJECTOR_READOUT_SCOPE,
    )
    # Retain the archive's public topology ordering and exact registered candidates.
    if "effective_calibration" in batch:
        raise ValueError("effective-law observations cannot be converted as archived")
    result.update(
        {key: batch[key] for key in PUBLIC_PROJECTOR_KEYS if key in batch and key != "counts"}
    )
    return result
