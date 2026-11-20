"""Finite-budget synthetic projector counts for the opt-in V10 training pilot.

NEW single-trial multinomial law; not the native FPGA sticky-window law, R6
thermal law, or Pauli full histogram. Public command, actual state and weak
teacher are distinct objects. This module reads no files and touches no hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

FOUR_CELL_ORDER = ("00", "10", "01", "11")
PROJECTOR_READOUT_SCOPE = "synthetic_single_excitation_single_trial_multinomial_v1"
STATE_BOUNDARY = "normalized_single_excitation_conditional_on_herald"
NOISE_CLASS_NAMES = (
    "phase",
    "amplitude",
    "mixed_coherent",
    "decoherence_or_leakage",
    "count_background",
    "mechanism_shift",
)
SUPPORTED_NOISE_CLASSES = (0, 1, 2, 3, 5)
UNSUPPORTED_NOISE_CLASSES = (4,)
DEFAULT_READOUT_PARAMETERS = (0.35, 0.7, 0.015)
PUBLIC_PROJECTOR_KEYS = (
    "counts",
    "basis_descriptors",
    "shots",
    "setting_uncertainty",
    "setting_mask",
    "mode_features",
    "mode_mask",
    "physical_adjacency",
    "correlation_adjacency",
    "incidence",
    "certificate_features",
    "certificate_mask",
    "analyzer_re",
    "analyzer_im",
    "prepared_re",
    "prepared_im",
    "readout_parameters",
    "effective_calibration",
    "public_control_target",
    "public_candidate_actions",
    "public_candidate_valid_mask",
    "public_noop_candidate_index",
)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def allocate_integer_budget(total_budget: int, setting_count: int) -> np.ndarray:
    """Exact total in canonical setting order, including its integer remainder."""
    _integer(setting_count, "setting_count")
    _integer(total_budget, "total_budget", setting_count)
    base, remainder = divmod(total_budget, setting_count)
    result = np.full(setting_count, base, dtype=np.int64)
    result[:remainder] += 1
    return result


def _probability(value: float | torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 1 and result.shape[0] == reference.shape[0]:
        result = result[:, None]
    try:
        result = torch.broadcast_to(result, reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} cannot broadcast to [batch,settings]") from error
    if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
        raise ValueError(f"{name} must be a finite probability")
    return result


def readout_probability_from_rho(
    rho: torch.Tensor,
    analyzers: torch.Tensor,
    *,
    herald_probability: float | torch.Tensor = DEFAULT_READOUT_PARAMETERS[0],
    read_efficiency: float | torch.Tensor = DEFAULT_READOUT_PARAMETERS[1],
    background_probability: float | torch.Tensor = DEFAULT_READOUT_PARAMETERS[2],
) -> torch.Tensor:
    """Return [00,10,01,11], preserving gradients of tensor calibration parameters.

    rho: normalized conditional R1 [B,N,N]; analyzers: normalized complex [B,S,N].
    Public parameters accept scalar, [B], or [B,S]. With q=Tr(P rho),
    P(I|S)=b+(1-b)*eta*q and P(I|not S)=b. Efficiency enters once.
    """
    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho must have shape [batch,N,N]")
    if analyzers.ndim != 3 or analyzers.shape[0] != rho.shape[0]:
        raise ValueError("analyzers must have shape [batch,settings,N]")
    if analyzers.shape[-1] != rho.shape[-1]:
        raise ValueError("state and analyzer mode dimensions differ")
    if not rho.is_complex() or not analyzers.is_complex():
        raise TypeError("rho and analyzers must be complex tensors")
    dtype = torch.promote_types(rho.dtype, analyzers.dtype)
    state = rho.to(dtype)
    vector = analyzers.to(device=rho.device, dtype=dtype)
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(vector).all()):
        raise ValueError("state and analyzer must be finite")
    # Geometry teachers promote public float32 tensors to float64; promotion
    # does not restore the bits lost during serialization.
    tolerance = 2e-5 if dtype == torch.complex64 else 2e-6
    norms = vector.abs().square().sum(-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=tolerance, rtol=0):
        raise ValueError("analyzer vectors must be normalized")
    if not torch.allclose(state, state.mH, atol=tolerance, rtol=0):
        raise ValueError("rho must be Hermitian")
    trace = state.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not torch.allclose(trace, torch.ones_like(trace), atol=tolerance, rtol=0):
        raise ValueError("rho must be normalized conditional R1")
    born = torch.einsum("bsi,bij,bsj->bs", vector.conj(), state, vector).real
    if bool(((born < -tolerance) | (born > 1 + tolerance)).any()):
        raise ValueError("Born probability outside [0,1]")
    # Only roundoff at a physical boundary is removed.
    born = born.clamp(0.0, 1.0)
    herald = _probability(herald_probability, born, "herald_probability")
    efficiency = _probability(read_efficiency, born, "read_efficiency")
    background = _probability(background_probability, born, "background_probability")
    click = background + (1.0 - background) * efficiency * born
    return torch.stack(
        (
            (1.0 - herald) * (1.0 - background),
            herald * (1.0 - click),
            (1.0 - herald) * background,
            herald * click,
        ),
        dim=-1,
    )


def projector_schedule(n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N populations plus three trine probes on N-1 star edges: S=4N-3.

    This is a declared structured-state schedule, not arbitrary NxN tomography.
    Vectors are subsequently referenced to the public commanded phase.
    """
    _integer(n_modes, "n_modes", 3)
    if n_modes > 256:
        raise ValueError("V10 projector pilot supports at most 256 modes")
    vectors = list(np.eye(n_modes, dtype=np.complex128))
    edges = np.asarray([(0, j) for j in range(1, n_modes)], dtype=np.int64)
    for first, second in edges:
        for phase in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            vector = np.zeros(n_modes, dtype=np.complex128)
            vector[first] = 1 / np.sqrt(2.0)
            vector[second] = np.exp(1j * phase) / np.sqrt(2.0)
            vectors.append(vector)
    incidence = np.zeros((n_modes, n_modes), dtype=np.float32)
    for index, edge in enumerate(edges):
        incidence[edge, index] = 1.0
    incidence[:, -1] = 1.0
    return np.stack(vectors), edges, incidence


def _descriptors(analyzers: np.ndarray) -> np.ndarray:
    """Fifteen public geometry coordinates; no latent state or generated labels."""
    batch_size, settings, n_modes = analyzers.shape
    positions = np.arange(n_modes, dtype=np.float64) / (n_modes - 1)
    probability = np.abs(analyzers) ** 2
    support = probability > 1e-12
    result = np.zeros((batch_size, settings, 15), dtype=np.float32)
    result[..., 0] = n_modes / 256.0
    result[..., 1] = support.sum(-1) / n_modes
    result[..., 2] = support.sum(-1) == 1
    result[..., 3] = support.sum(-1) == 2
    result[..., 4] = (probability * positions).sum(-1)
    result[..., 5] = (probability * positions**2).sum(-1)
    result[..., 6] = (probability**2).sum(-1)
    result[..., 7] = np.abs(analyzers).sum(-1) / np.sqrt(n_modes)
    for row in range(batch_size):
        for setting in range(settings):
            active = np.flatnonzero(support[row, setting])
            result[row, setting, 8] = active[0] / (n_modes - 1)
            result[row, setting, 9] = active[-1] / (n_modes - 1)
            if len(active) == 2:
                relative = analyzers[row, setting, active[0]].conjugate()
                relative *= analyzers[row, setting, active[1]]
                relative /= abs(relative)
                result[row, setting, 10] = relative.real
                result[row, setting, 11] = relative.imag
    coordinate = np.arange(settings) / max(1, settings - 1)
    result[..., 12] = coordinate
    result[..., 13] = np.sin(2 * np.pi * coordinate)
    result[..., 14] = np.cos(2 * np.pi * coordinate)
    return result


def _layout(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    width = int(np.ceil(np.sqrt(n_modes)))
    x, y = np.arange(n_modes) % width, np.arange(n_modes) // width
    feature = np.zeros((n_modes, 12), dtype=np.float32)
    feature[:, 0] = np.arange(n_modes) / (n_modes - 1)
    feature[:, 1] = x / max(1, width - 1)
    feature[:, 2] = y / max(1, int(y.max()))
    feature[:, 3] = 1.0
    feature[:, 4] = np.sin(2 * np.pi * feature[:, 0])
    feature[:, 5] = np.cos(2 * np.pi * feature[:, 0])
    adjacency = (np.abs(x[:, None] - x) + np.abs(y[:, None] - y)) == 1
    return feature, adjacency.astype(np.float32)


def public_projector_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Explicitly exclude query outcomes, latent labels, and teacher diagnostics."""
    return {key: batch[key] for key in PUBLIC_PROJECTOR_KEYS if key in batch}


def _states(n_modes: int, batch_size: int, rng: np.random.Generator) -> tuple:
    prepared = np.exp(
        rng.normal(0, 0.22, (batch_size, n_modes)) + 1j * rng.normal(0, 0.35, (batch_size, n_modes))
    )
    prepared /= np.linalg.norm(prepared, axis=-1, keepdims=True)
    families = np.arange(batch_size, dtype=np.int64) % 6
    rng.shuffle(families)
    classes = np.asarray([-1, 0, 1, 2, 3, 5], dtype=np.int64)[families]
    state = np.empty((batch_size, n_modes, n_modes), dtype=np.complex128)
    partition = np.ones((batch_size, n_modes, n_modes), dtype=np.bool_)
    coherent_pure = np.ones(batch_size, dtype=np.bool_)
    severity = np.zeros(batch_size, dtype=np.float32)
    for row, family in enumerate(families):
        da, dp = np.zeros(n_modes), np.zeros(n_modes)
        if family in (1, 3):
            dp = rng.normal(0, rng.uniform(0.25, 0.8), n_modes)
            dp -= dp.mean()
        if family in (2, 3):
            da = rng.normal(0, rng.uniform(0.2, 0.65), n_modes)
            da -= da.mean()
        actual = prepared[row] * np.exp(da + 1j * dp)
        actual /= np.linalg.norm(actual)
        density = np.outer(actual, actual.conj())
        # Actual gauge-removed perturbation magnitude, not a label inferred
        # from count noise. r/(1+r) bounds the class-specific RMS in [0,1].
        rms = float(np.sqrt(np.mean(da**2 + dp**2)))
        severity[row] = rms / (1.0 + rms)
        if family == 4:
            visibility = rng.uniform(0.2, 0.75)
            density = visibility * density + (1 - visibility) * np.diag(abs(actual) ** 2)
            coherent_pure[row] = False
            severity[row] = 1.0 - visibility
        elif family == 5:
            shuffled = rng.permutation(n_modes)
            groups = np.zeros(n_modes, dtype=np.int64)
            split = int(rng.integers(1, n_modes))
            groups[shuffled[split:]] = 1
            partition[row] = groups[:, None] == groups[None, :]
            density *= partition[row]
            coherent_pure[row] = False
            severity[row] = (~partition[row]).sum() / (n_modes * (n_modes - 1))
        state[row] = density
    return prepared, state, classes, partition, coherent_pure, severity


def make_projector_training_batch(
    n_modes: int,
    batch_size: int,
    total_budget: int,
    seed: int,
    *,
    query_fraction: float = 0.25,
    device: str | torch.device = "cpu",
    include_teachers: bool = True,
    observation_law: str = PROJECTOR_READOUT_SCOPE,
    fixed_calibration: tuple[float, ...] | None = None,
    report_fixed_geometry: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate common-random-number data for matched opt-in training arms.

    All arrays have batch first, including phase_edges [B,N-1,2]. Counts and query_counts
    are complementary int64 tables. Shots are independently allocated, never
    reconstructed from hidden outcomes. Nominal rows have noise-label mask false.
    """
    _integer(batch_size, "batch_size")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if fixed_calibration is not None:
        fixed = np.asarray(fixed_calibration, dtype=np.float64)
        if (observation_law != "effective_single_event_v1" or fixed.shape != (5,)
                or not np.isfinite(fixed).all() or not ((fixed > 0) & (fixed < 1)).all()):
            raise ValueError("fixed calibration requires five interior probabilities under the effective law")
    if not np.isfinite(query_fraction) or not 0 < query_fraction < 1:
        raise ValueError("query_fraction must lie strictly between zero and one")
    base, edges, incidence = projector_schedule(n_modes)
    settings = len(base)
    allocation = allocate_integer_budget(total_budget, settings)
    rng = np.random.Generator(np.random.Philox(seed))
    prepared, rho, classes, partition, coherent_pure, severity = _states(n_modes, batch_size, rng)
    analyzers = base[None] * (prepared / np.abs(prepared))[:, None, :]
    params = np.tile(np.asarray(DEFAULT_READOUT_PARAMETERS), (batch_size, 1))
    if observation_law == PROJECTOR_READOUT_SCOPE:
        probability = readout_probability_from_rho(
            torch.from_numpy(rho), torch.from_numpy(analyzers),
            herald_probability=torch.from_numpy(params[:, 0]),
            read_efficiency=torch.from_numpy(params[:, 1]),
            background_probability=torch.from_numpy(params[:, 2]),
        ).numpy()
    elif observation_law == "effective_single_event_v1":
        from .effective_projector_law import effective_probabilities
        if fixed_calibration is None:
            apparatus_rng = np.random.default_rng(np.random.SeedSequence([seed, 811]))
            params = apparatus_rng.uniform((.01, .25, .0005, .45, .001),
                                           (.04, .55, .003, .8, .008), (batch_size, 5))
        else:
            params = np.broadcast_to(fixed, (batch_size, 5)).copy()
        probability = effective_probabilities(torch.from_numpy(rho), torch.from_numpy(analyzers),
                                              torch.from_numpy(params)).numpy()
    else:
        raise ValueError(f"unsupported observation law: {observation_law}")
    raw = np.empty((batch_size, settings, 4), dtype=np.int64)
    for row in range(batch_size):
        for setting in range(settings):
            raw[row, setting] = rng.multinomial(int(allocation[setting]), probability[row, setting])
    query = np.zeros((batch_size, settings), dtype=np.bool_)
    query_count = min(settings - 1, max(1, int(round(settings * query_fraction))))
    # With a fixed calibration, count draws consume a law-dependent amount of RNG.
    # Give the paired strong/weak panels a separate mask stream; preserve archive draws.
    mask_rng = rng if fixed_calibration is None else np.random.default_rng(
        np.random.SeedSequence([seed, 829])
    )
    for row in range(batch_size):
        query[row, mask_rng.choice(settings, query_count, replace=False)] = True
    visible, shots = ~query, np.broadcast_to(allocation, (batch_size, settings)).copy()
    feature, physical = _layout(n_modes)
    pair = partition.astype(np.float32)
    pair_mask = np.broadcast_to(~np.eye(n_modes, dtype=bool), pair.shape).copy()
    pair[:, np.arange(n_modes), np.arange(n_modes)] = 0.0
    hyperedge = np.zeros((batch_size, n_modes), dtype=np.float32)
    hyperedge[:, :-1] = partition[:, edges[:, 0], edges[:, 1]]
    hyperedge[:, -1] = coherent_pure
    hyperedge_mask = np.ones_like(hyperedge, dtype=np.bool_)
    # Nonzero pair coherences alone do not certify N-party GME in mixed states.
    hyperedge_mask[classes == 3, -1] = False
    noise_valid = classes >= 0
    values = {
        "counts": np.where(visible[..., None], raw, 0),
        "query_counts": np.where(query[..., None], raw, 0),
        "basis_descriptors": _descriptors(analyzers),
        "shots": shots,
        "setting_uncertainty": np.where(visible, 0.5 / np.sqrt(shots), 0).astype(np.float32),
        "setting_mask": visible,
        "measurement_query_mask": query,
        "measurement_available_mask": np.ones_like(visible),
        "mode_features": np.broadcast_to(feature, (batch_size, *feature.shape)).copy(),
        "mode_mask": np.ones((batch_size, n_modes), dtype=np.bool_),
        "physical_adjacency": np.broadcast_to(physical, (batch_size, *physical.shape)).copy(),
        "correlation_adjacency": np.zeros((batch_size, n_modes, n_modes), dtype=np.float32),
        "incidence": np.broadcast_to(incidence, (batch_size, *incidence.shape)).copy(),
        "certificate_features": np.zeros((batch_size, n_modes, 10), dtype=np.float32),
        "certificate_mask": np.zeros((batch_size, n_modes), dtype=np.bool_),
        "analyzer_re": analyzers.real.astype(np.float32),
        "analyzer_im": analyzers.imag.astype(np.float32),
        "prepared_re": prepared.real.astype(np.float32),
        "prepared_im": prepared.imag.astype(np.float32),
        ("readout_parameters" if observation_law == PROJECTOR_READOUT_SCOPE else "effective_calibration"): params.astype(np.float32),
        "public_control_target": np.stack((abs(prepared), np.angle(prepared)), -1).astype(
            np.float32
        ),
        "target_R1_re": rho.real.astype(np.float32),
        "target_R1_im": rho.imag.astype(np.float32),
        "target_w_mask": np.ones(batch_size, dtype=np.bool_),
        "target_sector": np.tile(np.asarray([0, 1, 0], dtype=np.float32), (batch_size, 1)),
        "target_pair": pair,
        "target_pair_mask": pair_mask,
        "target_hyperedge": hyperedge,
        "target_hyperedge_mask": hyperedge_mask,
        "target_partition": pair.copy(),
        "target_partition_mask": pair_mask.copy(),
        "target_operational_noise": np.where(noise_valid, classes, 0),
        "target_operational_noise_mask": noise_valid,
        "target_noise_severity": severity,
        "target_noise_severity_mask": noise_valid.copy(),
        "target_is_nominal": ~noise_valid,
        "target_fine_mask": np.zeros(batch_size, dtype=np.bool_),
        "target_fine_identifiable_mask": np.zeros(batch_size, dtype=np.bool_),
        "target_is_synthetic": np.ones(batch_size, dtype=np.bool_),
        "budget_read_requested": np.full(batch_size, total_budget, dtype=np.int64),
        "budget_read_realized": shots.sum(-1),
        "budget_mode_read_realized": (shots * (abs(analyzers) > 1e-12).sum(-1)).sum(-1),
        "dense_mode_budget_convention": n_modes * shots.sum(-1),
        "phase_edges": np.broadcast_to(edges, (batch_size, *edges.shape)).copy(),
    }
    batch = {key: torch.from_numpy(value).to(device) for key, value in values.items()}
    validate_projector_training_batch(batch)
    if include_teachers:
        from .training_ideas import build_visible_rho8_views
        from .training_ideas_geometry import build_projector_geometry_targets

        av = torch.complex(batch["analyzer_re"], batch["analyzer_im"])
        truth = torch.complex(batch["target_R1_re"], batch["target_R1_im"])
        command = torch.complex(batch["prepared_re"], batch["prepared_im"])
        if observation_law == "effective_single_event_v1":
            # The effective owner verifies PSD at the supplied dtype. Keep the
            # original double-precision teacher inputs instead of upcasting
            # independently rounded real and imaginary array components.
            truth = torch.from_numpy(rho).to(device)
            av = torch.from_numpy(analyzers).to(device)
            command = torch.from_numpy(prepared).to(device)
        batch.update(
            build_visible_rho8_views(
                batch["counts"],
                av,
                batch["setting_mask"],
                batch["shots"],
            )
        )
        batch.update(
            build_projector_geometry_targets(
                truth,
                av,
                batch["shots"],
                command,
                visible_mask=batch["setting_mask"],
                report_fixed_geometry=report_fixed_geometry,
                **({"effective_calibration": batch["effective_calibration"], "profile_nuisance": True}
                   if observation_law == "effective_single_event_v1" else
                   dict(herald_probability=batch["readout_parameters"][:, 0],
                        read_efficiency=batch["readout_parameters"][:, 1],
                        background_probability=batch["readout_parameters"][:, 2])),
            )
        )
        if observation_law == "effective_single_event_v1":
            batch = {name: value.float() if value.dtype == torch.float64 else value for name, value in batch.items()}
        # A nominal row has target == actual by construction. Its RSM fraction
        # is 0/0 (undefined), not a direction to learn. Independently rounded
        # float32 command/state tensors can create a tiny numerical residual.
        nominal = batch["target_is_nominal"]
        batch["target_rsm_mask"] = batch["target_rsm_mask"] & ~nominal
        batch["target_rsm_fraction"] = torch.where(
            nominal,
            torch.zeros_like(batch["target_rsm_fraction"]),
            batch["target_rsm_fraction"],
        )
    return batch


def validate_projector_training_batch(batch: dict[str, torch.Tensor]) -> None:
    """Fail on count-law, budget, and visible/query-boundary violations."""
    counts, target, shots = batch["counts"], batch["query_counts"], batch["shots"]
    visible, query = batch["setting_mask"], batch["measurement_query_mask"]
    if counts.dtype != torch.int64 or target.dtype != torch.int64:
        raise ValueError("four-cell observations must be native integer tensors")
    if counts.ndim != 3 or counts.shape[-1] != 4 or target.shape != counts.shape:
        raise ValueError("counts/query_counts must be [batch,settings,4]")
    if shots.dtype != torch.int64 or shots.shape != counts.shape[:2]:
        raise ValueError("shots must be independently allocated int64 [batch,settings]")
    if visible.dtype != torch.bool or query.dtype != torch.bool:
        raise ValueError("visible and query masks must be boolean")
    if visible.shape != shots.shape or query.shape != shots.shape:
        raise ValueError("observation masks and shots must align")
    if bool((visible & query).any()) or not bool((visible | query).all()):
        raise ValueError("visible and query must partition the available operations")
    if bool((visible.sum(-1) == 0).any()) or bool((query.sum(-1) == 0).any()):
        raise ValueError("every row needs visible and held-out query observations")
    if bool((counts < 0).any()) or bool((target < 0).any()) or bool((shots <= 0).any()):
        raise ValueError("counts/opportunities must be nonnegative")
    if bool((counts[query] != 0).any()) or bool((target[visible] != 0).any()):
        raise ValueError("hidden counts leaked across visible/query boundary")
    if not torch.equal(counts.sum(-1) + target.sum(-1), shots):
        raise ValueError("four-cell conservation differs from known independent shots")
    if not torch.equal(shots.sum(-1), batch["budget_read_realized"]):
        raise ValueError("realized READ budget differs from exposures")
    if not torch.equal(batch["budget_read_realized"], batch["budget_read_requested"]):
        raise ValueError("integer exposure must conserve the requested budget")
    if "global_outcome_histogram" in batch or "full_outcome_mask" in batch:
        raise ValueError("projector counts cannot masquerade as full Pauli histograms")
    if bool(batch["target_fine_identifiable_mask"].any()):
        raise ValueError("Pauli fine-family labels are unavailable for this readout")
