"""One W forward for synthetic joint counts, geometry and bounded decisions.

Outcomes are observable signal/idler bits ordered 00,01,10,11. Dark and true
signal photons are not separately observable labels. This declared synthetic
acquisition model is not a calibration of the real hardware plates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

import torch

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.control_tangent import helmert_matrix
from ml_recon.physics_hybrid_v10.csm_profile_contract import Profiled_csm_contract
from ml_recon.physics_hybrid_v10.four_cell_layout import TRACK_A_FOUR_CELL_SEMANTICS
from ml_recon.physics_hybrid_v10.operation_excitation_contract import (
    OPERATION_EXCITATION_MODE,
    normalized_cell_excitation_mean,
    validate_operation_excitation_mean,
)
from ml_recon.physics_hybrid_v10.excitation_count_model import poisson_sector_probabilities

FORWARD_SCHEMA = "V10SyntheticAodJointForwardV2"
FOUR_CELL_SEMANTICS = TRACK_A_FOUR_CELL_SEMANTICS
FISHER_PROBABILITY_FLOOR = 1e-10
RANK_RELATIVE_TOLERANCE = 1e-8
RANK_ABSOLUTE_TOLERANCE = 1e-10
GAIN_TOLERANCE = 1e-10
NUISANCE_FIELDS = (
    "state_visibility",
    "apparatus_efficiency",
    "detector_dark_probability",
    "per_cell_success_probability",
)


@dataclass(frozen=True)
class Torch_aod_noise:
    state_visibility: float | torch.Tensor = 1.0
    apparatus_efficiency: float | torch.Tensor = 1.0
    detector_dark_probability: float | torch.Tensor = 0.0
    per_cell_success_probability: float | torch.Tensor = 0.005


@dataclass(frozen=True)
class Aod_forward_result:
    probabilities: torch.Tensor
    state_coefficients: torch.Tensor
    synthetic_four_cell_semantics: str
    forward_recipe_sha: str
    herald_probability: torch.Tensor
    operation_excitation_mean: torch.Tensor | None = None
    cell_excitation_mean: torch.Tensor | None = None
    poisson_sector_probabilities: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class Bounded_action_targets:
    raw_gain: torch.Tensor
    recoverable_gain: torch.Tensor
    bounded_deployable_gain: torch.Tensor
    bounded_harm: torch.Tensor
    stop_mask: torch.Tensor
    projection_loss: torch.Tensor
    information_objective: torch.Tensor
    information_objective_mask: torch.Tensor
    oracle_action_mask: torch.Tensor
    before_cost: torch.Tensor
    bounded_cost: torch.Tensor


def _probability_scalar(value, reference, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be a boolean")
    result = torch.as_tensor(value, dtype=reference.real.dtype, device=reference.device)
    if (
        result.ndim != 0
        or not bool(torch.isfinite(result))
        or not bool((result >= 0) & (result <= 1))
    ):
        raise ValueError(f"{name} must be a finite scalar probability")
    return result


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(
        str((array.dtype.str, array.shape)).encode() + array.tobytes()
    ).hexdigest()


def torch_aod_forward(
    prepared_coefficients: torch.Tensor,
    mode_action: torch.Tensor,
    read_amplitudes: torch.Tensor,
    read_phases: torch.Tensor,
    mode_mask: torch.Tensor,
    noise: Torch_aod_noise,
    *,
    read_throughput: torch.Tensor | None = None,
    operation_excitation_mean: torch.Tensor | None = None,
    cell_gain: torch.Tensor | None = None,
) -> Aod_forward_result:
    """Apply a relative amplitude/phase action exactly once before readout."""
    if prepared_coefficients.ndim != 2 or not torch.is_complex(prepared_coefficients):
        raise ValueError("prepared coefficients must be complex [batch,mode]")
    batch_size, n_mode = prepared_coefficients.shape
    if (
        mode_action.shape != (batch_size, n_mode, 2)
        or not torch.is_floating_point(mode_action)
        or mode_mask.shape != (batch_size, n_mode)
        or mode_mask.dtype is not torch.bool
        or read_amplitudes.ndim != 3
        or read_amplitudes.shape != read_phases.shape
        or read_amplitudes.shape[0] != batch_size
        or read_amplitudes.shape[2] != n_mode
        or read_amplitudes.shape[1] < 1
    ):
        raise ValueError(
            "AOD action, mask and read descriptors have incompatible shapes"
        )
    for value in (prepared_coefficients, mode_action, read_amplitudes, read_phases):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("AOD forward inputs must be finite")
    if bool((read_amplitudes < 0).any()) or bool((mode_mask.sum(-1) < 2).any()):
        raise ValueError(
            "read amplitudes are nonnegative and at least two modes must be active"
        )
    if bool((prepared_coefficients.masked_select(~mode_mask).abs() > 0).any()):
        raise ValueError("inactive modes must not carry prepared-state amplitude")
    inactive_action = ~mode_mask.unsqueeze(-1).expand_as(mode_action)
    if bool((mode_action.masked_select(inactive_action).abs() > 0).any()):
        raise ValueError("inactive modes must not carry an action")
    mask = mode_mask.to(prepared_coefficients.real.dtype)
    # Subtract the common amplitude gauge before exp to avoid overflow.
    amplitude_shift = (
        mode_action[..., 0].masked_fill(~mode_mask, -torch.inf).amax(-1, keepdim=True)
    )
    state = prepared_coefficients * torch.exp(mode_action[..., 0] - amplitude_shift)
    state = state * torch.exp(1j * mode_action[..., 1]) * mask
    state_norm = state.abs().square().sum(-1, keepdim=True).sqrt()
    if bool((state_norm <= 0).any()) or not bool(torch.isfinite(state_norm).all()):
        raise ValueError("prepared action must produce a nonzero finite W state")
    state = state / state_norm
    read = read_amplitudes * torch.exp(-1j * read_phases) * mask[:, None, :]
    read_norm = read.abs().square().sum(-1, keepdim=True).sqrt()
    if bool((read_norm <= 0).any()):
        raise ValueError("read projector cannot vanish on active modes")
    read = read / read_norm
    visibility = _probability_scalar(noise.state_visibility, state, "state_visibility")
    efficiency = _probability_scalar(
        noise.apparatus_efficiency, state, "apparatus_efficiency"
    )
    dark = _probability_scalar(
        noise.detector_dark_probability, state, "detector_dark_probability"
    )
    per_cell = _probability_scalar(
        noise.per_cell_success_probability, state, "per_cell_success_probability"
    )
    coherent = (read.conj() * state[:, None, :]).sum(-1).abs().square()
    incoherent = (read.abs().square() * state.abs().square()[:, None, :]).sum(-1)
    projection = visibility * coherent + (1.0 - visibility) * incoherent
    if bool((projection < -1e-6).any()) or bool((projection > 1 + 1e-6).any()):
        raise ValueError("W projection left the probability simplex")
    projection = projection.clamp(0.0, 1.0)
    if read_throughput is None:
        read_throughput = torch.ones_like(projection)
    if (
        read_throughput.shape != projection.shape
        or not bool(torch.isfinite(read_throughput).all())
        or bool(((read_throughput < 0) | (read_throughput > 1)).any())
    ):
        raise ValueError("read throughput must be [batch,operation] probabilities")
    cell_mean = None
    sector_probabilities = None
    if operation_excitation_mean is None:
        # Legacy path deliberately remains byte-compatible with the prior herald law.
        cell_success = per_cell * mode_mask.sum(-1, keepdim=True) * state.abs().square()
        if bool((cell_success > 1).any()):
            raise ValueError("per-cell excitation exceeds one; invalid working point")
        herald = (1.0 - (1.0 - cell_success).prod(-1))[:, None].expand_as(projection)
    else:
        operation_excitation_mean = validate_operation_excitation_mean(
            operation_excitation_mean, batch_size=batch_size, operation_count=projection.shape[1]
        )
        cell_mean = normalized_cell_excitation_mean(
            operation_excitation_mean, state.abs().square(), cell_gain
        )
        sector_probabilities = poisson_sector_probabilities(operation_excitation_mean)
        herald = 1.0 - sector_probabilities[0]
    idler_given_signal = 1.0 - (1.0 - efficiency * read_throughput * projection) * (
        1.0 - dark
    )
    idler_without_signal = dark.expand_as(projection)
    probabilities = torch.stack(
        (
            (1.0 - herald) * (1.0 - idler_without_signal),
            (1.0 - herald) * idler_without_signal,
            herald * (1.0 - idler_given_signal),
            herald * idler_given_signal,
        ),
        dim=-1,
    )
    if (
        not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or not torch.allclose(
            probabilities.sum(-1), torch.ones_like(projection), atol=1e-7, rtol=1e-7
        )
    ):
        raise ValueError("joint signal/idler probabilities are not conserved")
    recipe = {
        "schema": FORWARD_SCHEMA,
        "cell_order": FOUR_CELL_SEMANTICS,
        "phase_convention": "read_aod_negative_order",
        "action_applications": 1,
        "read_amplitude_sha": _tensor_sha(read_amplitudes),
        "read_phase_sha": _tensor_sha(read_phases),
        "throughput_sha": _tensor_sha(read_throughput),
        "noise": {
            name: float(torch.as_tensor(getattr(noise, name)).detach().cpu())
            for name in NUISANCE_FIELDS
        },
    }
    if operation_excitation_mean is not None:
        recipe.update(
            schema="V10SyntheticAodJointForwardV3",
            operation_excitation_mode=OPERATION_EXCITATION_MODE,
            operation_excitation_mean_sha=_tensor_sha(operation_excitation_mean),
            cell_gain_sha=None if cell_gain is None else _tensor_sha(cell_gain),
        )
    return Aod_forward_result(
        probabilities, state, FOUR_CELL_SEMANTICS, sha256_of(recipe), herald,
        operation_excitation_mean, cell_mean, sector_probabilities,
    )


def _repeat_weights(repeats, probabilities):
    if (
        repeats.shape != probabilities.shape[:2]
        or repeats.dtype is torch.bool
        or not bool(torch.isfinite(repeats).all())
        or bool((repeats < 0).any())
        or not bool((repeats == repeats.floor()).all())
    ):
        raise ValueError("physical repeats must be aligned nonnegative integers")
    total = repeats.sum(-1)
    return repeats.to(probabilities.dtype) / total.clamp_min(1)[:, None], total > 0


def bounded_action_targets(
    *,
    raw_mode_action,
    recoverable_mode_action,
    bounded_mode_action,
    desired_target_coefficients,
    physical_repeats,
    before_forward,
    raw_forward,
    recoverable_forward,
    bounded_forward,
    target_forward,
    candidate_information_value=None,
    candidate_information_mask=None,
) -> Bounded_action_targets:
    """Hellinger-squared gain against desired target at the deployed action."""
    results = (
        before_forward,
        raw_forward,
        recoverable_forward,
        bounded_forward,
        target_forward,
    )
    shape = before_forward.probabilities.shape
    if (
        desired_target_coefficients.shape != target_forward.state_coefficients.shape
        or any(result.probabilities.shape != shape for result in results)
        or any(
            result.synthetic_four_cell_semantics != FOUR_CELL_SEMANTICS
            for result in results
        )
        or len({result.forward_recipe_sha for result in results}) != 1
    ):
        raise ValueError(
            "targets require one same-forward detector/schedule and target shape"
        )
    normalized_target = (
        desired_target_coefficients
        / desired_target_coefficients.abs().square().sum(-1, keepdim=True).sqrt()
    )
    overlap = (
        (normalized_target.conj() * target_forward.state_coefficients)
        .sum(-1)
        .abs()
        .square()
    )
    if not bool(torch.isfinite(overlap).all()) or not torch.allclose(
        overlap, torch.ones_like(overlap), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("target forward does not represent the desired state")
    weights, valid = _repeat_weights(physical_repeats, before_forward.probabilities)

    def cost(result):
        difference = (
            target_forward.probabilities.clamp_min(1e-15).sqrt()
            - result.probabilities.clamp_min(1e-15).sqrt()
        )
        return 0.5 * (weights * difference.square().sum(-1)).sum(-1)

    before_cost = cost(before_forward)
    raw_gain = before_cost - cost(raw_forward)
    recoverable_gain = before_cost - cost(recoverable_forward)
    bounded_cost = cost(bounded_forward)
    gain = before_cost - bounded_cost
    if (candidate_information_value is None) != (candidate_information_mask is None):
        raise ValueError(
            "candidate information values and validity masks must be supplied together"
        )
    if candidate_information_value is None:
        information = gain * 0.0
        information_mask = torch.zeros_like(valid)
    else:
        if (
            candidate_information_mask.dtype is not torch.bool
            or candidate_information_mask.shape != candidate_information_value.shape
            or candidate_information_value.shape[0] != gain.shape[0]
            or not bool(
                torch.isfinite(
                    candidate_information_value[candidate_information_mask]
                ).all()
            )
        ):
            raise ValueError(
                "candidate information targets require strict masks and finite valid values"
            )
        information_mask = candidate_information_mask
        information = torch.where(
            information_mask,
            candidate_information_value,
            torch.zeros_like(candidate_information_value),
        )
    return Bounded_action_targets(
        raw_gain,
        recoverable_gain,
        gain,
        (gain < -GAIN_TOLERANCE) & valid,
        (gain <= GAIN_TOLERANCE) & valid,
        (raw_mode_action - bounded_mode_action).square().sum((-1, -2)),
        information,
        information_mask,
        valid,
        before_cost,
        bounded_cost,
    )


def aod_control_geometry(
    prepared_coefficients,
    read_amplitudes,
    read_phases,
    physical_repeats,
    mode_mask,
    noise: Torch_aod_noise,
    *,
    geometry_mode: str = "RAW_CSM",
    nuisance_fields: Sequence[str] = (),
    profile_contract: Profiled_csm_contract | None = None,
    profile_calibration_root=None,
    read_throughput=None,
):
    """Target-free interleaved Helmert JVPs with explicit raw/profiled estimands.

    Computing one JVP per control coordinate avoids a B-by-B dense Jacobian.
    It also avoids Hilbert-space 2**N state construction at every N.
    """
    if geometry_mode not in {"RAW_CSM", "PROFILED_CSM"}:
        raise ValueError("geometry_mode must be RAW_CSM or PROFILED_CSM")
    nuisance_fields = tuple(nuisance_fields)
    if len(set(nuisance_fields)) != len(nuisance_fields) or any(
        name not in NUISANCE_FIELDS for name in nuisance_fields
    ):
        raise ValueError("unregistered or duplicated nuisance coordinate")
    if geometry_mode == "RAW_CSM":
        if (
            nuisance_fields
            or profile_contract is not None
            or profile_calibration_root is not None
        ):
            raise ValueError("RAW_CSM rejects nuisance profiling and profile contracts")
    else:
        if profile_contract is None or profile_calibration_root is None:
            raise ValueError(
                "PROFILED_CSM requires an immutable profile contract and calibration root"
            )
        profile_contract.validate()
        profile_contract.verify_calibration_source(profile_calibration_root)
        if profile_contract.nuisance_fields != nuisance_fields:
            raise ValueError("profile contract nuisance fields do not match geometry")
    batch_size, n_mode = prepared_coefficients.shape
    basis = torch.as_tensor(
        helmert_matrix(n_mode),
        dtype=prepared_coefficients.real.dtype,
        device=prepared_coefficients.device,
    )
    logical = torch.zeros(
        (batch_size, 2 * (n_mode - 1)), dtype=basis.dtype, device=basis.device
    )
    nuisance = torch.tensor(
        [
            float(torch.as_tensor(getattr(noise, name)).detach().cpu())
            for name in nuisance_fields
        ],
        dtype=basis.dtype,
        device=basis.device,
    )

    def forward(coordinates, nuisance_coordinates):
        action = torch.einsum(
            "kn,bkj->bnj", basis, coordinates.reshape(batch_size, n_mode - 1, 2)
        )
        action = action * mode_mask[..., None]
        local_noise = replace(
            noise,
            **{
                name: nuisance_coordinates[index]
                for index, name in enumerate(nuisance_fields)
            },
        )
        return torch_aod_forward(
            prepared_coefficients,
            action,
            read_amplitudes,
            read_phases,
            mode_mask,
            local_noise,
            read_throughput=read_throughput,
        ).probabilities

    probability = forward(logical, nuisance)
    if geometry_mode == "PROFILED_CSM":
        if profile_contract.forward_recipe_sha != torch_aod_forward(
            prepared_coefficients,
            torch.einsum(
                "kn,bkj->bnj", basis, logical.reshape(batch_size, n_mode - 1, 2)
            ) * mode_mask[..., None],
            read_amplitudes,
            read_phases,
            mode_mask,
            noise,
            read_throughput=read_throughput,
        ).forward_recipe_sha:
            raise ValueError("profile contract forward SHA does not match geometry forward")
        for name, (lower, upper) in zip(
            nuisance_fields, profile_contract.nuisance_domain_bounds
        ):
            value = float(torch.as_tensor(getattr(noise, name)).detach().cpu())
            tolerance = profile_contract.solver_tolerance
            if not lower + tolerance < value < upper - tolerance:
                raise ValueError(
                    "analytic profile projection requires nuisance values strictly interior to domain"
                )
    _repeat_weights(physical_repeats, probability)
    control_columns = []
    for coordinate in range(logical.shape[1]):
        direction = torch.zeros_like(logical)
        direction[:, coordinate] = 1
        _, column = torch.autograd.functional.jvp(
            lambda value: forward(value, nuisance), logical, direction
        )
        control_columns.append(column)
    control = torch.stack(control_columns, dim=-1)
    nuisance_columns = []
    for coordinate in range(len(nuisance_fields)):
        direction = torch.zeros_like(nuisance)
        direction[coordinate] = 1
        _, column = torch.autograd.functional.jvp(
            lambda value: forward(logical, value), nuisance, direction
        )
        nuisance_columns.append(column)
    nuisance_jacobian = (
        torch.stack(nuisance_columns, -1)
        if nuisance_columns
        else probability.new_zeros((*probability.shape, 0))
    )
    multiplier = (
        physical_repeats.to(probability.dtype)[..., None]
        / probability.clamp_min(FISHER_PROBABILITY_FLOOR)
    ).sqrt()
    control_white = (control * multiplier[..., None]).reshape(
        batch_size, -1, logical.shape[1]
    )
    profiled = []
    for index in range(batch_size):
        matrix = control_white[index]
        if geometry_mode == "PROFILED_CSM":
            nuisance_white = (
                nuisance_jacobian[index] * multiplier[index, ..., None]
            ).reshape(-1, len(nuisance_fields))
            left, singular, _ = torch.linalg.svd(nuisance_white, full_matrices=False)
            threshold = max(
                profile_contract.solver_tolerance,
                float(singular.max().detach()) * RANK_RELATIVE_TOLERANCE,
            )
            nuisance_basis = left[:, singular > threshold]
            matrix = matrix - nuisance_basis @ (
                nuisance_basis.transpose(-1, -2) @ matrix
            )
        profiled.append(matrix)
    profiled_jacobian = torch.stack(profiled)
    gram = profiled_jacobian.transpose(-1, -2) @ profiled_jacobian
    # Rank comes from J itself. Square-rooting tiny Gram eigenvalues can promote
    # numerical roundoff into spurious recoverable directions.
    _, singular_values, right_vectors = torch.linalg.svd(profiled_jacobian, full_matrices=False)
    threshold = torch.maximum(
        singular_values[:, :1] * RANK_RELATIVE_TOLERANCE,
        torch.full_like(singular_values[:, :1], RANK_ABSOLUTE_TOLERANCE),
    )
    return {
        "schema": (
            "V10AodRawCSMGeometryV1"
            if geometry_mode == "RAW_CSM"
            else "V10AodProfiledCSMGeometryV1"
        ),
        "estimand_identity": geometry_mode,
        "coordinate_layout": "interleaved_helmert_log_amplitude_phase",
        "control_jacobian": control,
        "nuisance_jacobian": nuisance_jacobian,
        "raw_gram": control_white.transpose(-1, -2) @ control_white,
        "profiled_gram": gram,
        "singular_values": singular_values,
        "recoverable_basis": right_vectors.transpose(-1, -2),
        "support_mask": singular_values > threshold,
        "rank": (singular_values > threshold).sum(-1),
        "target_independent": True,
        "profile_contract_sha": (
            None if profile_contract is None else profile_contract.sha256()
        ),
        "profile_algorithm": (
            "none_fixed_nuisance_forward"
            if profile_contract is None
            else profile_contract.solver_identity
        ),
        "profile_nuisance_domains": (
            () if profile_contract is None else profile_contract.nuisance_domain_bounds
        ),
        "profile_solver_tolerance": (
            None if profile_contract is None else profile_contract.solver_tolerance
        ),
    }
