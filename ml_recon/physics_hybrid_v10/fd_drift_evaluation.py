"""One strong-readout FD decision on a drifting preparation at t=10, with frozen CAL risk gates."""

from __future__ import annotations

import numpy as np
import torch

from .effective_projector_law import effective_probabilities
from .fd_policy_evaluation import select_fd_actions
from .long_projector_data import bank_seed, build_bank
from .matrix_diagnostics import (
    CAL_SCHEMA,
    STRONG_CALIBRATION,
    TARGET_COVERAGES,
    _score_frozen_thresholds,
)
from .temporal_recovery_environment import VELOCITY_SD
from .training_ideas_data import _states, projector_schedule, public_projector_inputs
from .training_ideas_geometry import apply_diagonal_control, public_control_candidates

DRIFT_TIME = 10
DRIFT_VERSION = "strong_fixed_command_fd_snapshot_t10_v1"


def drift_truth(initial, velocity, time_index):
    """Change the underlying next-preparation amplitudes and phases with no intervening action."""
    vel = velocity - velocity.mean(1, keepdim=True)
    return apply_diagonal_control(initial, (time_index * vel).unsqueeze(1))[:, 0]


def true_gains(rho, target, actions):
    after = apply_diagonal_control(rho, actions)
    before = torch.einsum("bi,bij,bj->b", target.conj(), rho, target).real
    corrected = torch.einsum("bi,bkij,bj->bk", target.conj(), after, target).real
    zero = actions.abs().sum((-2, -1)) == 0
    return before, torch.where(
        zero, torch.zeros_like(corrected), corrected - before[:, None]
    )


def drift_inputs(n, size, split_namespace, *, role="confirmation"):
    """Generate a single 4,096-READ endpoint paired with the registered static initial panel."""
    if n not in (3, 4) or role not in ("confirmation", "fixture"):
        raise ValueError("strong drift endpoint is registered only for 3Q/4Q")
    seed = bank_seed(n, role, 0, split_namespace)
    static = build_bank(
        n,
        size,
        role,
        0,
        False,
        split_namespace=split_namespace,
        fixed_calibration=STRONG_CALIBRATION,
    )
    cmd_np, rho_np, _, _, _, _ = _states(
        n, size, np.random.Generator(np.random.Philox(seed))
    )
    cmd, initial = torch.from_numpy(cmd_np), torch.from_numpy(rho_np)
    if not torch.equal(initial.real.float(), static["target_R1_re"]) or not torch.equal(
        initial.imag.float(), static["target_R1_im"]
    ):
        raise ValueError(
            "drift starting state differs from the strong static confirmation pair"
        )
    drift_rng = np.random.default_rng(np.random.SeedSequence([seed, 10471, DRIFT_TIME]))
    draws = torch.from_numpy(
        drift_rng.normal(size=(size, n, 2)) * np.asarray(VELOCITY_SD)
    )
    vel = draws - draws.mean(1, keepdim=True)
    terminal = drift_truth(initial, draws, DRIFT_TIME)
    base = projector_schedule(n)[0]
    analyzers = torch.from_numpy(base[None] * (cmd_np / np.abs(cmd_np))[:, None, :])
    calibration = torch.tensor(STRONG_CALIBRATION, dtype=torch.float64).expand(size, -1)
    probabilities = effective_probabilities(terminal, analyzers, calibration).numpy()
    count_rng = np.random.default_rng(np.random.SeedSequence([seed, 10477, DRIFT_TIME]))
    counts = np.empty(probabilities.shape, np.int64)
    shots = static["shots"].numpy()
    for i in range(size):
        for j in range(shots.shape[1]):
            counts[i, j] = count_rng.multinomial(int(shots[i, j]), probabilities[i, j])
    public = public_projector_inputs(static).copy()
    public["counts"] = torch.where(
        static["setting_mask"][:, :, None],
        torch.from_numpy(counts),
        torch.zeros_like(static["counts"]),
    ).to(static["counts"].dtype)
    actions = public_control_candidates(
        size, n, device=torch.device("cpu"), dtype=torch.float64
    )
    public.update(
        public_candidate_actions=actions.float(),
        public_candidate_valid_mask=torch.ones(size, 7, dtype=torch.bool),
        public_noop_candidate_index=torch.zeros(size, dtype=torch.long),
    )
    before, gains = true_gains(terminal, cmd, actions)
    return public, {
        "initial": initial,
        "terminal": terminal,
        "velocity": vel,
        "command": cmd,
        "unadjusted_overlap": before,
        "gain": gains,
        "counts": counts,
        "setting_mask": static["setting_mask"],
    }


@torch.inference_mode()
def score_drift_endpoint(model, n, size, split_namespace, device, receipt):
    if n not in (3, 4) or not model.training_ideas_config.fd:
        return {"available": False, "eligible_sizes": [3, 4]}, {}
    if (
        model.training
        or receipt["schema"] != CAL_SCHEMA
        or receipt["n"] != n
        or receipt["split_namespace"] != split_namespace
        or not receipt["available"]
        or receipt["strong_calibration"] != list(STRONG_CALIBRATION)
        or [k["target"] for k in receipt["thresholds"]] != list(TARGET_COVERAGES)
    ):
        raise ValueError(
            "drift scoring requires the bound strong CAL receipt and frozen model"
        )
    public, truth = drift_inputs(n, size, split_namespace)
    public = {k: v.to(device) for k, v in public.items()}
    # Hidden terminal rho and counterfactual gains are used only after public ranking.
    ranked = select_fd_actions(model(public, N=n), public)
    report, states = _score_frozen_thresholds(ranked, truth["gain"].to(device), receipt)
    report.update(
        available=True,
        schema=DRIFT_VERSION,
        t=DRIFT_TIME,
        velocity_sd_before_gauge_projection=list(VELOCITY_SD),
        velocity_per_mode="constant per interval in relative natural log field-amplitude and phase radians",
        physical_endpoint="one allowed next-preparation action at t=10; no prior interventions or acquisitions",
        truth="D(t) rho_initial D(t)^H normalized by its trace, D(t)=diag(exp(t*(v_amp+i*v_phase)))",
        realized_reads_at_t=4096,
        preceding_paid_reads=0,
        strong_calibration=list(STRONG_CALIBRATION),
        target="the public preparation-command ket, held fixed within each paired state",
        states=size,
        mean_terminal_before_overlap=float(truth["unadjusted_overlap"].mean()),
    )
    states.update(
        state_index=list(range(size)),
        velocity=truth["velocity"].tolist(),
        terminal_rho_re=truth["terminal"].real.tolist(),
        terminal_rho_im=truth["terminal"].imag.tolist(),
        before_action_overlap=truth["unadjusted_overlap"].tolist(),
    )
    return report, states
