"""Exact-N corrected star readout, a frozen V10 estimator, and bounded diagonal next-preparation actions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from nqrec.physics.single_event_mc import sample_counts
from ml_recon.physics_hybrid_v10.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from ml_recon.physics_hybrid_v10.effective_projector_observation import (
    EFFECTIVE_LAW,
    PUBLIC_SCHEMA,
    make_public_observation,
)
from ml_recon.physics_hybrid_v10.training_ideas_data import (
    allocate_integer_budget,
    projector_schedule,
    public_projector_inputs,
)
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state
from ml_recon.physics_hybrid_v10.training_ideas_geometry import apply_diagonal_control
from ml_recon.physics_hybrid_v10.temporal_recovery_environment import feasible

SIZES = (3, 4, 8, 16, 32, 64, 128, 256)


def training_panel(checkpoint):
    """Identify the training split, operating point and updates held fixed across capacities."""
    ident = checkpoint["identity"]
    return {
        "source": ident["source"],
        "split_namespace": ident["split_namespace"],
        "training_states": ident["train_states_per_n"],
        "sizes": ident["sizes"],
        "batch": ident["batch"],
        "seed": ident["seed"],
        "learning_rate": ident["learning_rate"],
        "updates": checkpoint["updates"],
        "factor_rank": checkpoint["classifier_config"]["factor_rank"],
    }


def load_shared(core, device):
    path = Path(core)
    state = torch.load(path, map_location="cpu", weights_only=True)
    ident = state["identity"]
    if (
        ident["lane"] != "joint_cs_32768_s7301"
        or ident["seed"] != 7301
        or state["classifier_config"]["profile"] != "C-S"
        or ident["train_states_per_n"] != 32768
        or state["updates"] != 32768
        or ident["sizes"] != [3, 4, 8]
        or state["observation_schema"] != PUBLIC_SCHEMA
        or state["law"] != EFFECTIVE_LAW
    ):
        raise ValueError("unexpected shared-N fit, matched update or observation law")
    model = V10ClassifierNFull(**state["classifier_config"])
    model.load_state_dict(state["classifier_state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def measure(density, command, calibration, reads, seed, visible=None):
    """Sample the actual 4N−3 settings with one WRITE per READ. All supplied latents stay outside the ABI."""
    density = density.detach().to("cpu", torch.complex128)
    command = command.detach().to("cpu", torch.complex128)
    b, n = command.shape
    if density.shape != (b, n, n):
        raise ValueError("density and exact mode count differ")
    schedule, _, _ = projector_schedule(n)
    settings = len(schedule)
    analyzer = (
        torch.from_numpy(schedule)[None, :, :] * (command / command.abs())[:, None, :]
    )
    calib = torch.as_tensor(calibration, dtype=torch.float64)
    if calib.ndim == 1:
        calib = calib.expand(b, -1)
    allocations = allocate_integer_budget(reads, settings)
    shots = torch.from_numpy(allocations.astype(np.int64)).expand(b, -1).contiguous()
    probability = effective_probabilities(density, analyzer, calib)
    counts = np.zeros((b, settings, 4), dtype=np.int64)
    if isinstance(seed, (list, tuple, np.ndarray)):
        if len(seed) != b:
            raise ValueError("one independent count seed per trajectory")
        seeds = list(seed)
    else:
        seeds = [int(seed) + i for i in range(b)]
    for row, origin in enumerate(seeds):
        rng = np.random.Generator(np.random.Philox(int(origin)))
        for trials in np.unique(allocations):
            mask = allocations == trials
            counts[row, mask] = sample_counts(probability[row, mask], int(trials), rng)
    flags = (
        torch.ones((b, settings), dtype=torch.bool)
        if visible is None
        else torch.as_tensor(visible, dtype=torch.bool)
    )
    public = make_public_observation(
        counts=torch.from_numpy(counts),
        shots=shots,
        analyzers=analyzer,
        prepared=command,
        calibration=calib,
        visible_mask=flags,
        readout_law=EFFECTIVE_LAW,
    )
    if (
        public["counts"].shape != (b, 4 * n - 3, 4)
        or not torch.equal(
            public["counts"].sum(-1).long(), torch.where(flags, shots, 0)
        )
        or not torch.equal(shots.sum(-1), torch.full((b,), reads, dtype=torch.int64))
    ):
        raise ValueError("read budget or four-cell observation mismatch")
    return public


@torch.inference_mode()
def infer(model, public, device):
    """The frozen full classifier executes its measurement/graph/hypergraph/structure path at this exact N."""
    n = public["mode_mask"].shape[1]
    safe = public_projector_inputs(public)
    gpu = {k: v.to(device) for k, v in safe.items()}
    raw = model(gpu, N=n, fine_identifiable=False)
    rho = normalized_state(raw).to("cpu")
    if rho.shape != (len(public["counts"]), n, n) or not torch.isfinite(rho).all():
        raise ValueError("invalid true-N state estimate")
    return rho


def bounded_action(estimate, target, *, gain=1.0, component="joint"):
    """Relative population and star-reference phase inversion followed by the declared common radial limit."""
    if component not in ("joint", "amplitude", "phase"):
        raise ValueError("invalid action support")
    estimate = estimate.to("cpu", torch.complex128)
    target = target.to("cpu", torch.complex128)
    population = estimate.diagonal(dim1=-2, dim2=-1).real
    if (population <= 0).any() or (target.abs() == 0).any():
        raise ValueError(
            "population inversion requires positive fitted and target support"
        )
    log_amp = target.abs().log() - 0.5 * population.log()
    ref = estimate[:, :, 0]
    target_ref = target * target[:, 0, None].conj()
    defined = ref.abs() > 0
    # A zero inferred coherence carries no phase direction and receives zero phase command.
    raw_phase = torch.where(
        defined, torch.angle(target_ref * ref.conj()), torch.zeros_like(log_amp)
    )
    if component == "phase":
        log_amp = torch.zeros_like(log_amp)
    if component == "amplitude":
        raw_phase = torch.zeros_like(raw_phase)
    return feasible(torch.stack((log_amp, raw_phase), dim=-1) * gain)


def apply_action(density, action):
    return apply_diagonal_control(
        density.to(torch.complex128), action.to(torch.float64)[:, None, :, :]
    )[:, 0]


def overlap(density, target):
    return torch.einsum("bi,bij,bj->b", target.conj(), density, target).real
