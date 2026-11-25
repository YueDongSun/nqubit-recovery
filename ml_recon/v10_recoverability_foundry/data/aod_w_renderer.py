"""Keyed physical count streams using the same forward as control evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np
import torch

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.four_cell_layout import four_cell_layout_metadata
from ml_recon.physics_hybrid_v10.excitation_count_model import (
    opportunity_weighted_sector_probabilities,
    poisson_k2_mark_probabilities,
)

from ..contracts.physical_budget import Physical_budget
from ..physics.aod_w_state import Aod_w_state
from ..physics.aod_detector_forward import Aod_noise_model
from ..physics.aod_feedback_forward import (
    FOUR_CELL_SEMANTICS,
    Torch_aod_noise,
    torch_aod_forward,
)
from .aod_operation_schedule import Aod_operation_schedule


@dataclass(frozen=True)
class Aod_render_result:
    observation: dict
    truth: dict
    metadata: dict


def render_aod_w(
    *,
    state: Aod_w_state,
    schedule: Aod_operation_schedule,
    budget: Physical_budget,
    noise: Aod_noise_model,
    keyed_rng,
    recipe_id: str,
    group_id: str,
    visible_operation_mask: np.ndarray | None = None,
    operation_excitation_mean: np.ndarray | None = None,
    cell_gain: np.ndarray | None = None,
    topology_observable_mask: np.ndarray | None = None,
) -> Aod_render_result:
    """Render one sample without storing raw observations or privileged truth.

    Herald and conditional idler uniforms are keyed by operation identity. The
    same uniforms can therefore be replayed for paired action/budget panels.
    Multiplicity is measured from the same events, never independently resampled.
    """
    schedule.validate()
    budget.validate()
    budget.require_repeat_measurements()
    noise.validate()
    if (
        state.n_mode != schedule.n_mode
        or budget.n_mode != state.n_mode
        or tuple(schedule.operation_ids) != budget.registry.operation_ids
        or budget.registry.measurement_schedule_sha != schedule.sha256()
    ):
        raise ValueError("state, schedule, registry SHA and budget must bind exactly")
    if budget.provenance.source_kind != "synthetic_fixture":
        raise ValueError(
            "a synthetic renderer cannot generate real observation evidence"
        )
    if (
        not recipe_id
        or not group_id
        or not callable(getattr(keyed_rng, "stream", None))
    ):
        raise ValueError("counter-based RNG and stream identities are required")
    operation_count = len(schedule.operation_ids)
    if visible_operation_mask is None:
        visible_operation_mask = np.ones(operation_count, dtype=bool)
    visible = np.asarray(visible_operation_mask)
    if visible.dtype != np.bool_ or visible.shape != (operation_count,):
        raise ValueError("visible operation mask must be strict Boolean [operation]")
    repeats = np.asarray(budget.repeats_by_operation, dtype=np.int64)
    available = repeats > 0
    visible = visible & available
    throughput = np.asarray(
        getattr(schedule, "read_throughput", np.ones(operation_count)), dtype=np.float64
    )
    excitation_tensor = None
    if operation_excitation_mean is not None:
        excitation = np.asarray(operation_excitation_mean, dtype=np.float64)
        if excitation.shape != (operation_count,) or not np.isfinite(excitation).all() or np.any(excitation < 0):
            raise ValueError("operation_excitation_mean must be finite nonnegative [operation]")
        excitation_tensor = torch.as_tensor(excitation)[None]
    gain_tensor = None
    if cell_gain is not None:
        gain = np.asarray(cell_gain, dtype=np.float64)
        if gain.shape != (operation_count, state.n_mode) or not np.isfinite(gain).all() or np.any(gain < 0):
            raise ValueError("cell_gain must be finite nonnegative [operation,mode]")
        gain_tensor = torch.as_tensor(gain)[None]
    if topology_observable_mask is None:
        topology_observable_mask = np.zeros(operation_count, dtype=bool)
    topology_available = np.asarray(topology_observable_mask)
    if topology_available.dtype != np.bool_ or topology_available.shape != (operation_count,):
        raise ValueError("topology_observable_mask must be strict Boolean [operation]")
    if topology_available.any():
        raise ValueError("operation_poisson_v1 has no topology-resolving observable; mask must be all false")
    noise_tensor = Torch_aod_noise(**asdict(noise))
    forward = torch_aod_forward(
        torch.as_tensor(np.array(state.coefficients), dtype=torch.complex128)[None],
        torch.zeros((1, state.n_mode, 2), dtype=torch.float64),
        torch.as_tensor(np.array(schedule.amplitudes), dtype=torch.float64)[None],
        torch.as_tensor(np.array(schedule.phases), dtype=torch.float64)[None],
        torch.as_tensor(np.array(state.mode_mask), dtype=torch.bool)[None],
        noise_tensor,
        read_throughput=torch.as_tensor(throughput)[None],
        operation_excitation_mean=excitation_tensor,
        cell_gain=gain_tensor,
    )
    probabilities = forward.probabilities[0].detach().numpy()
    herald_probabilities = forward.herald_probability[0].detach().numpy()
    counts = np.zeros((operation_count, 4), dtype=np.int64)
    uniform_digest = hashlib.sha256()
    for operation_index, (operation_id, count) in enumerate(
        zip(schedule.operation_ids, repeats)
    ):
        herald_uniform = keyed_rng.stream("aod_herald_" + operation_id, 0).random(
            int(count)
        )
        idler_uniform = keyed_rng.stream("aod_idler_" + operation_id, 0).random(
            int(count)
        )
        uniform_digest.update(operation_id.encode())
        uniform_digest.update(herald_uniform.tobytes())
        uniform_digest.update(idler_uniform.tobytes())
        signal_probability = herald_probabilities[operation_index]
        signal = herald_uniform < signal_probability
        conditional = (
            probabilities[operation_index, 3] / signal_probability
            if signal_probability > 0
            else 0.0
        )
        idler = idler_uniform < np.where(
            signal, conditional, noise.detector_dark_probability
        )
        counts[operation_index] = np.bincount(
            2 * signal.astype(np.int64) + idler.astype(np.int64), minlength=4
        )
    if not np.array_equal(counts.sum(-1), repeats):
        raise RuntimeError("joint counts do not conserve physical trial allocation")
    success_count = int(counts[:, 2:].sum())
    poisson_truth = {}
    if excitation_tensor is not None:
        p0, p1, pge2 = forward.poisson_sector_probabilities
        mixture = opportunity_weighted_sector_probabilities(
            excitation_tensor, torch.as_tensor(repeats)[None]
        )
        local, cross, k2 = poisson_k2_mark_probabilities(forward.cell_excitation_mean)
        poisson_truth = {
            "operation_excitation_mode": "operation_poisson_v1",
            "operation_excitation_mean": excitation_tensor[0].detach().numpy(),
            "effective_excitation_mean": excitation_tensor[0].detach().numpy(),
            "read_throughput": throughput.copy(),
            "operation_exposure_per_opportunity": np.ones(operation_count, dtype=np.float64),
            "opportunity_count": repeats.copy(),
            "poisson_p0": p0[0].detach().numpy(),
            "poisson_p1": p1[0].detach().numpy(),
            "poisson_pge2": pge2[0].detach().numpy(),
            "opportunity_weighted_sector_probabilities": np.asarray([value[0].item() for value in mixture]),
            "local_k2_mark_probability": local[0].detach().numpy(),
            "cross_k2_mark_probability": cross[0].detach().numpy(),
            "k2_mark_probability": k2[0].detach().numpy(),
            "cross_k2_total_probability": torch.triu(cross[0], diagonal=1).sum((-1, -2)).detach().numpy(),
            "topology_availability_mask": topology_available.copy(),
            "topology_label": "NOT_AVAILABLE",
            "topology_evolution_status": (
                "DEFER_NOT_IDENTIFIABLE_NO_REGISTERED_PROPAGATOR_OR_OBSERVABLE"
            ),
            "observable_count_law_topology_dependence": "NONE_TOTAL_COUNT_MARGINAL_ONLY",
            "synthetic_only": True,
        }
    declared_sha = budget.canonical_sha256()
    completed_budget = replace(
        budget,
        success_probability_by_operation=tuple(
            float(value) for value in herald_probabilities
        ),
        success_probability_source="SYNTHETIC_FORWARD",
        m_succ_realized=success_count,
    )
    completed_budget.validate()
    public_counts = np.where(visible[:, None], counts, 0)
    recipe = {
        "schema": "V10AodPhysicalStreamRecipeV2",
        "recipe_id": recipe_id,
        "measurement_schedule_sha": schedule.sha256(),
        "declared_budget_sha": declared_sha,
        "forward_recipe_sha": forward.forward_recipe_sha,
        "noise": asdict(noise),
    }
    observation = {
        "operation_ids": schedule.operation_ids,
        "four_cell_counts": public_counts,
        "observation_mask": visible.copy(),
        "observed_operation_mask": visible.copy(),
        "hidden_count_mask": (available & ~visible).copy(),
        "visible_success_count": int(public_counts[:, 2:].sum()),
        "repeats_by_operation": repeats.copy(),
        **four_cell_layout_metadata(FOUR_CELL_SEMANTICS),
        "event_ownership": {
            "herald_generation": "SAME_TRIAL_GENERATION_EVENT",
            "idler_readout": "DESTRUCTIVE_READOUT_DIAGNOSTIC",
            "storable_memory_resource_available": False,
            "storable_memory_resource_inventory": None,
            "stored_memory_resource_ids": None,
            "reuse_as_stored_memory_forbidden": True,
        },
    }
    truth = {
        "four_cell_probabilities": probabilities.copy(),
        "four_cell_counts": counts.copy(),
        "state_coefficients": np.array(state.coefficients),
        "success_probability_by_operation": herald_probabilities.copy(),
        "m_succ_realized": success_count,
        "mu_succ_predicted": completed_budget.mu_succ_predicted,
        "available_operation_mask": available.copy(),
        **poisson_truth,
    }
    metadata = {
        "recipe_id": recipe_id,
        "recipe_sha": sha256_of(recipe),
        "group_id": group_id,
        "physical_budget_sha": completed_budget.canonical_sha256(),
        "declared_budget_sha": declared_sha,
        "physical_budget": completed_budget.to_dict(),
        "b_trial": budget.b_trial,
        "b_mode_shot": budget.b_mode_shot,
        "forward_recipe_sha": forward.forward_recipe_sha,
        "success_uniforms_sha": uniform_digest.hexdigest(),
        "operation_excitation_mode": "legacy" if excitation_tensor is None else "operation_poisson_v1",
        "synthetic_excitation_model": excitation_tensor is not None,
        "stream_only": True,
        "event_ownership_schema": "V10AodReadoutOwnershipV1",
        "idler_readout_classification": "DESTRUCTIVE_READOUT_DIAGNOSTIC",
        "storable_memory_resource_available": False,
        "storable_memory_resource_inventory": None,
        "stored_memory_resource_ids": None,
        "readout_trials_reusable_as_stored_memory": False,
        "research_only": True,
        "hardware_action_authorized": False,
    }
    return Aod_render_result(observation, truth, metadata)
