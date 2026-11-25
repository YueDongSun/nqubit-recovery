"""Bounded V2.4 smoke build: state -> noise -> measurement -> disk."""

from __future__ import annotations

import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ml_recon.physics_hybrid_v9.full_family_renderer import (
    coarse_groups_for_system,
)
from ml_recon.physics_hybrid_v10.family_renderer import V10FamilyRenderer
from ml_recon.physics_hybrid_v10.measurement_model import born_probabilities
from ml_recon.physics_hybrid_v10.measurement_scheduler import V10MeasurementScheduler
from ml_recon.physics_hybrid_v10.pauli_registry import descriptor_vector
from ml_recon.physics_hybrid_v10.ppt_mixture import negativity
from ml_recon.physics_hybrid_v10.state_backends import (
    DenseDensityStateBackend,
    MPSStateBackend,
    ProductBlockStateBackend,
)
from ml_recon.physics_hybrid_v10.structure_renderer import V10StructureRenderer

from ..contracts import freeze_contract, load_contract
from ..hashing import sha256_array, sha256_json, write_canonical_json
from ..index_sqlite import create_index
from ..manifests import V24BuildManifest, V24ReadyManifest
from ..sample_keys import make_sample_key
from ..schema_registry import schema_sha256
from ..storage_hdf5 import write_hdf5_shard
from ..version import GENERATOR_SHA
from ..validators import (
    validate_authority_flags,
    validate_global_multi,
    validate_partition_labels,
    validate_split_groups,
    validate_view_keys,
)
from .feedback_renderers import render_feedback_history, render_feedback_sample
from .global_multi_renderer import render_global_multi
from .large_n_samplers import sample_global_pauli_outcomes
from .measurement_renderers import (
    observed_correlation_adjacency,
    outcome_histogram,
    pair_marginal_from_global,
    subset_marginal_from_global,
)
from .noise_renderers import render_noise_truth
from .negative_control_renderers import (
    apply_persistent_negative_control,
    split_negative_control_family,
)
from .state_renderers import ExplicitFockMixtureBackend


V24_GENERATOR_SHA = GENERATOR_SHA
DEFAULT_SMOKE_PLAN = (
    {"cell_id": "classifier_n3", "dataset_family": "small_n_family", "N": 3,
     "system_type": "3qubit", "measurement_domain": "global_pauli_v24",
     "train": 8, "validation": 4},
    {"cell_id": "classifier_n4", "dataset_family": "small_n_family", "N": 4,
     "system_type": "4qubit", "measurement_domain": "global_pauli_v24",
     "train": 4, "validation": 2},
    {"cell_id": "structure_n24", "dataset_family": "general_n_structure", "N": 24,
     "system_type": "nqubit", "measurement_domain": "global_pauli_v24",
     "train": 4, "validation": 1},
    {"cell_id": "w_control_n4", "dataset_family": "w_control", "N": 4,
     "system_type": "4qubit", "measurement_domain": "global_pauli_v24",
     "train": 2, "validation": 1},
    {"cell_id": "feedback_n4", "dataset_family": "feedback_one_step", "N": 4,
     "system_type": "4qubit", "measurement_domain": "global_pauli_v24",
     "train": 8, "validation": 4},
)


@dataclass(frozen=True)
class V24CellRenderParameters:
    noise_family: str
    eta: float
    kappa: float
    shot_mode: str
    shot_budget: int
    measurement_schedule: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None,
                     *, default_shots: int) -> "V24CellRenderParameters":
        source = value or {}
        result = cls(
            noise_family=str(source.get("noise_family", "smoke_mixed")),
            eta=float(source.get("eta", 0.08)),
            kappa=float(source.get("kappa_inj", source.get("kappa", 0.5))),
            shot_mode=str(source.get("shot_mode", "S_setting")),
            shot_budget=int(source.get("shot_budget", default_shots)),
            measurement_schedule=str(source.get(
                "measurement_schedule", "smoke_default")),
        )
        if result.shot_mode not in ("S_total", "S_setting", "S_ideal"):
            raise ValueError(f"unknown V2.4 shot mode: {result.shot_mode}")
        if result.shot_budget <= 0 or not 0.0 <= result.eta <= 1.0 \
                or not 0.0 <= result.kappa <= 1.0:
            raise ValueError("invalid V2.4 renderer cell parameters")
        return result

    def shots_by_setting(self, setting_count: int) -> np.ndarray:
        if setting_count <= 0:
            raise ValueError("shot allocation requires at least one setting")
        if self.shot_mode == "S_total":
            if self.shot_budget < setting_count:
                raise ValueError("S_total shot budget must cover every registered setting")
            quotient, remainder = divmod(self.shot_budget, setting_count)
            return np.asarray([
                quotient + int(index < remainder) for index in range(setting_count)
            ], dtype=np.int64)
        return np.full(setting_count, self.shot_budget, dtype=np.int64)

    @property
    def ideal_probability(self) -> bool:
        return self.shot_mode == "S_ideal"

    def require_schedule(self, expected: str) -> None:
        if self.measurement_schedule not in ("smoke_default", expected):
            raise ValueError(
                "renderer measurement schedule does not match its registered cell")


def _deterministic_counts(probability: np.ndarray, shots: int) -> np.ndarray:
    """Largest-remainder counts used only for the registered S_ideal view."""
    law = np.asarray(probability, dtype=np.float64)
    law = np.maximum(law, 0.0)
    law /= max(float(law.sum()), 1e-15)
    expected = law * int(shots)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(shots) - int(counts.sum())
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _outcomes_from_counts(outcomes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(outcomes), np.asarray(counts, dtype=np.int64), axis=0)


def _git_identity() -> tuple[str, str]:
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), text=True).strip()
    return commit, tree


def _reduced_two_qubit(rho: np.ndarray, N: int, pair: tuple[int, int]) -> np.ndarray:
    keep = tuple(pair)
    trace = tuple(index for index in range(N) if index not in keep)
    tensor = rho.reshape((2,) * (2 * N))
    current_n = N
    for qubit in sorted(trace, reverse=True):
        tensor = np.trace(tensor, axis1=qubit, axis2=current_n + qubit)
        current_n -= 1
    return tensor.reshape(4, 4)


def _reduced_density(rho: np.ndarray, N: int, keep: tuple[int, ...]) -> np.ndarray:
    """Partial trace to an ordered qubit subset."""
    keep = tuple(keep)
    trace = tuple(index for index in range(N) if index not in keep)
    tensor = np.asarray(rho, dtype=np.complex128).reshape((2,) * (2 * N))
    current_n = N
    for qubit in sorted(trace, reverse=True):
        tensor = np.trace(tensor, axis1=qubit, axis2=current_n + qubit)
        current_n -= 1
    return tensor.reshape(2 ** len(keep), 2 ** len(keep))


def _partial_transpose_negativity(rho: np.ndarray, subset: tuple[int, ...]) -> float:
    k = int(round(np.log2(np.asarray(rho).shape[0])))
    if k < 2:
        return 0.0
    if np.asarray(rho).shape != (2 ** k, 2 ** k):
        raise ValueError("partial-transpose input is not a k-qubit density matrix")
    values = np.asarray(rho, dtype=np.complex128).reshape((2,) * (2 * k))
    transposed = values.copy()
    for axis in subset:
        transposed = np.swapaxes(transposed, axis, k + axis)
    eigenvalues = np.linalg.eigvalsh(
        transposed.reshape(2 ** k, 2 ** k).astype(np.complex128))
    return float(np.maximum(-eigenvalues, 0.0).sum())


def _gme_hyperedge_targets(
    rho: np.ndarray, N: int, registered_subsets: Sequence[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate conservative pure-state GME certificates, never pair labels."""
    targets = np.zeros(len(registered_subsets), dtype=np.float32)
    masks = np.zeros(len(registered_subsets), dtype=np.bool_)
    for index, subset in enumerate(registered_subsets):
        if len(subset) < 3:
            continue
        reduced = _reduced_density(rho, N, tuple(subset))
        purity = float(np.real(np.trace(reduced @ reduced)))
        # All-cuts NPT is decisive for pure states. For mixed states it does
        # not exclude a PPT-mixture/biseparable convex decomposition, so the
        # target must remain masked until an SDP certificate exists.
        if purity < 1.0 - 1e-7:
            continue
        all_cuts_npt = True
        # Anchor local qubit 0 in A. Exactly one of A and its complement then
        # represents every non-trivial bipartition: 2^(k-1)-1 unique cuts.
        local_qubits = tuple(range(len(subset)))
        cuts = tuple(
            cut for cut_size in range(1, len(subset))
            for cut in itertools.combinations(local_qubits, cut_size)
            if 0 in cut and len(cut) < len(subset)
        )
        for left in cuts:
            if _partial_transpose_negativity(reduced, left) <= 1e-8:
                all_cuts_npt = False
                break
        if all_cuts_npt:
            targets[index] = 1.0
            masks[index] = True
        else:
            masks[index] = True
    return targets, masks

def _family_partition(label: str, N: int) -> tuple[np.ndarray, bool]:
    if label == "fully_separable" or label.startswith("single_excitation"):
        return np.arange(N, dtype=np.int16), True
    if label.startswith("biseparable_"):
        left, right = label.removeprefix("biseparable_").split("_")
        blocks = (tuple(int(value) - 1 for value in left),
                  tuple(int(value) - 1 for value in right))
    elif label.startswith("bell_"):
        chunks = label.removeprefix("bell_").split("_")
        blocks = tuple(tuple(int(value) - 1 for value in chunk) for chunk in chunks)
    elif label.startswith("two_bell_"):
        chunks = label.removeprefix("two_bell_").split("_")
        blocks = tuple(tuple(int(value) - 1 for value in chunk) for chunk in chunks)
    elif label == "mixed":
        return np.zeros(N, dtype=np.int16), False
    else:
        blocks = (tuple(range(N)),)
    block_ids = np.empty(N, dtype=np.int16)
    for block_id, block in enumerate(blocks):
        block_ids[list(block)] = block_id
    return block_ids, True


def _operational_noise_class(mechanism_name: str, *,
                             kappa_component: float = 0.0,
                             control_mechanism: str = "",
                             irrecoverable_mechanism: str = ""
                             ) -> tuple[int, bool]:
    phase = {"finite_relative_phase_error", "aod_phase_response"}
    amplitude = {"finite_amplitude_error", "aod_nonlinear_amplitude_response"}
    mixed = {"amplitude_phase_coupling", "calibration_drift_control_subspace"}
    decoherence = {"correlated_dephasing", "w_subspace_leakage",
                   "explicit_multi_excitation"}
    count_background = {"detector_dark", "detector_background",
                        "detector_accidentals", "retrieval_efficiency_imbalance",
                        "brightness_drift"}
    if mechanism_name.startswith("composite:"):
        selected = (control_mechanism if kappa_component >= 0.67
                    else irrecoverable_mechanism)
        return _operational_noise_class(selected)
    if mechanism_name in phase:
        return 0, True
    if mechanism_name in amplitude:
        return 1, True
    if mechanism_name in mixed:
        return 2, True
    if mechanism_name in decoherence:
        return 3, True
    if mechanism_name in count_background:
        return 4, True
    if mechanism_name == "mechanism_shift":
        return 5, True
    return 5, False


def _block_pair_entangled(block_backend, block_size: int,
                          local_pair: tuple[int, int]) -> bool | None:
    """Exact within-block pair-entanglement label for every compact backend.

    Large W blocks render as MPS and large product blocks as nested product
    backends; both previously fell through the dense-only path and lost pair
    supervision entirely. Returns ``None`` for backends without an exact
    reduction so the pair stays masked (unsupervised) instead of guessed.
    """
    if isinstance(block_backend, DenseDensityStateBackend):
        reduced = _reduced_two_qubit(block_backend.rho, block_size, local_pair)
        return bool(negativity(reduced, 1, (2, 2)) > 1e-8)
    if isinstance(block_backend, MPSStateBackend):
        reduced = block_backend.two_qubit_reduced_density(*local_pair)
        return bool(negativity(reduced, 1, (2, 2)) > 1e-8)
    if isinstance(block_backend, ProductBlockStateBackend):
        left_site, right_site = local_pair
        for inner_qubits, inner_backend in block_backend.blocks:
            if left_site in inner_qubits and right_site in inner_qubits:
                return _block_pair_entangled(
                    inner_backend, len(inner_qubits),
                    (inner_qubits.index(left_site), inner_qubits.index(right_site)))
        # The pair spans distinct inner blocks: product across the cut, so
        # the exact separable label is available without any reduction.
        return False
    return None


def _structure_entanglement_targets(
    rendered: Mapping[str, Any],
    registered_subsets: Sequence[tuple[int, ...]],
    block_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate pair negativity evidence from partition and GME evidence."""
    N = len(block_ids)
    pair_target = np.zeros((N, N), dtype=np.float32)
    pair_mask = np.zeros((N, N), dtype=np.bool_)
    backend = rendered["state_backend"]
    observed_blocks = tuple(tuple(block) for block in rendered["observed_partition"])
    for left, right in itertools.combinations(range(N), 2):
        if block_ids[left] != block_ids[right]:
            pair_mask[left, right] = pair_mask[right, left] = True
            continue
        containing = next(
            (block for block in observed_blocks if left in block and right in block), None)
        source = next(
            ((qubits, block_backend) for qubits, block_backend in backend.blocks
             if left in qubits and right in qubits), None)
        if containing is None or source is None:
            continue
        qubits, block_backend = source
        entangled = _block_pair_entangled(
            block_backend, len(qubits), (qubits.index(left), qubits.index(right)))
        if entangled is None:
            continue
        pair_target[left, right] = pair_target[right, left] = float(entangled)
        pair_mask[left, right] = pair_mask[right, left] = True

    templates_by_generating_block = {
        frozenset(qubits): template
        for qubits, template in zip(
            rendered["generating_partition"], rendered["block_templates"])
    }
    gme_templates = {"W", "GHZ", "Dicke", "graph_cluster"}
    hyperedge_target = np.zeros(len(registered_subsets), dtype=np.float32)
    hyperedge_mask = np.zeros(len(registered_subsets), dtype=np.bool_)
    for index, subset in enumerate(registered_subsets):
        subset_set = frozenset(subset)
        blocks_touched = {int(block_ids[qubit]) for qubit in subset}
        if len(blocks_touched) > 1:
            hyperedge_mask[index] = True
            continue
        if len(subset) < 3:
            continue
        generating = next(
            (block for block in templates_by_generating_block if subset_set == block), None)
        if generating is not None:
            template = templates_by_generating_block[generating]
            if template in gme_templates:
                hyperedge_target[index] = 1.0
                hyperedge_mask[index] = True
            elif template == "product":
                hyperedge_mask[index] = True
    return pair_target, pair_mask, hyperedge_target, hyperedge_mask


def _base_mode_inputs(N: int, setting_count: int, *,
                      registered_subsets: Sequence[tuple[int, ...]] | None = None,
                      correlation_adjacency: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    mode_features = np.zeros((N, 12), dtype=np.float32)
    mode_features[:, 0] = np.arange(N, dtype=np.float32) / max(N - 1, 1)
    mode_features[:, 1] = np.sin(2.0 * np.pi * np.arange(N) / max(N, 1))
    mode_features[:, 2] = np.cos(2.0 * np.pi * np.arange(N) / max(N, 1))
    mode_features[:, 3] = 1.0
    physical = np.zeros((N, N), dtype=np.float32)
    for index in range(N - 1):
        physical[index, index + 1] = physical[index + 1, index] = 1.0
    subsets = tuple(registered_subsets or itertools.combinations(range(N), 2))
    incidence = np.zeros((N, len(subsets)), dtype=np.float32)
    for edge_index, edge in enumerate(subsets):
        incidence[list(edge), edge_index] = 1.0
    return {
        "mode_features": mode_features,
        "mode_mask": np.ones(N, dtype=np.bool_),
        "physical_adjacency": physical,
        "correlation_adjacency": (
            np.asarray(correlation_adjacency, dtype=np.float32)
            if correlation_adjacency is not None else np.zeros_like(physical)),
        "incidence": incidence,
        "certificate_features": np.zeros((len(subsets), 10), dtype=np.float32),
        "certificate_mask": np.zeros(len(subsets), dtype=np.bool_),
        "setting_mask": np.ones(setting_count, dtype=np.bool_),
    }


def _multinomial_cell_uncertainty(
    cell_probability: np.ndarray,
    shots: int,
    *,
    probability_tolerance: float = 1e-12,
) -> float:
    """Return finite cell uncertainty while rejecting non-roundoff invalid laws."""

    probability = np.asarray(cell_probability, dtype=np.float64)
    if shots <= 0 or probability.ndim != 1 or probability.size == 0:
        raise ValueError("cell uncertainty requires positive shots and a probability vector")
    if not np.all(np.isfinite(probability)):
        raise ValueError("cell probability contains a non-finite value")
    if float(probability.min()) < -float(probability_tolerance):
        raise ValueError("cell probability contains a physically negative value")
    probability = np.clip(probability, 0.0, None)
    total = float(probability.sum())
    if total <= 0.0:
        raise ValueError("cell probability has zero mass")
    probability = np.clip(probability / total, 0.0, 1.0)
    variance = np.maximum(probability * (1.0 - probability), 0.0)
    return float(np.sqrt(np.mean(variance / int(shots))))


def _render_classifier(key, *, cell: Mapping[str, Any] | None = None, locked_ood: bool = False) -> Dict[str, Any]:
    family = V10FamilyRenderer().render(key, type("Rng", (), {
        "stream": lambda _, name: key.rng(name)
    })())
    N = key.N
    rho_clean = np.asarray(family["rho_clean"], dtype=np.complex128)
    settings = V10MeasurementScheduler(N).schedule()
    render = V24CellRenderParameters.from_mapping(cell, default_shots=64)
    render.require_schedule("full_global_pauli")
    shots_by_setting = render.shots_by_setting(len(settings))
    setting_axes = tuple(setting.descriptor.pauli_axes for setting in settings)
    noise = render_noise_truth(
        key.rng("noise"), rho_clean=rho_clean, settings=setting_axes,
        noise_family=render.noise_family, eta_target=render.eta,
        correctable_fraction=render.kappa, composite=True, locked_ood=locked_ood)
    rho_noisy = noise.rho_noisy
    clean_backend = DenseDensityStateBackend(rho_clean)
    noisy_backend = DenseDensityStateBackend(rho_noisy)
    descriptors = np.asarray([descriptor_vector(item.descriptor) for item in settings],
                             dtype=np.float32)
    full_histograms = []
    marginal_counts = []
    clean_marginals = []
    noisy_marginals = []
    uncertainty = []
    outcome_panels = []
    shot_rng = key.rng("shot")
    for setting_index, setting in enumerate(settings):
        setting_shots = int(shots_by_setting[setting_index])
        outcomes, clean_probability = born_probabilities(
            clean_backend, setting.descriptor.pauli_axes)
        _, state_probability = born_probabilities(
            noisy_backend, setting.descriptor.pauli_axes)
        apparatus_probability = noise.intervention.apply_apparatus_channel(
            state_probability, outcomes, noise.physical_parameter)
        detected_law = noise.intervention.apply_detector_channel(
            apparatus_probability, noise.physical_parameter)
        if render.ideal_probability:
            sampled = _outcomes_from_counts(
                outcomes, _deterministic_counts(detected_law[:-1], setting_shots))
        else:
            indices = shot_rng.choice(
                len(detected_law), size=setting_shots, p=detected_law)
            sampled = outcomes[indices[indices < len(outcomes)]]
        noisy_probability = detected_law[:-1] / max(detected_law[:-1].sum(), 1e-15)
        outcome_panels.append(sampled)
        full_histograms.append(outcome_histogram(sampled, N))
        marginal_counts.append(pair_marginal_from_global(sampled, (0, 1)))
        clean_cells = np.zeros(4, dtype=np.float64)
        noisy_cells = np.zeros(4, dtype=np.float64)
        for outcome, clean_value, noisy_value in zip(outcomes, clean_probability, noisy_probability):
            cell = int(outcome[0] > 0) * 2 + int(outcome[1] > 0)
            clean_cells[cell] += clean_value
            noisy_cells[cell] += noisy_value
        clean_marginals.append(clean_cells)
        noisy_marginals.append(noisy_cells)
        uncertainty.append(
            0.0 if render.ideal_probability else
            _multinomial_cell_uncertainty(noisy_cells, setting_shots))
    block_ids, partition_exact = _family_partition(str(family["fine_family_label"]), N)
    _, sizes = np.unique(block_ids, return_counts=True)
    depth = int(sizes.max()) if partition_exact else 1
    intactness = int(len(sizes)) if partition_exact else N
    pair_target = np.zeros((N, N), dtype=np.float32)
    for left, right in itertools.combinations(range(N), 2):
        value = negativity(_reduced_two_qubit(rho_clean, N, (left, right)), 1, (2, 2))
        pair_target[left, right] = pair_target[right, left] = float(value > 1e-8)
    p0 = float(rho_clean[0, 0].real)
    single_indices = [1 << (N - 1 - qubit) for qubit in range(N)]
    R1 = rho_clean[np.ix_(single_indices, single_indices)]
    p1 = float(np.trace(R1).real)
    w_mask = np.asarray(p1 > 1e-8, dtype=np.bool_)
    registered_subsets = tuple(itertools.combinations(range(N), order)
                                for order in range(2, N + 1))
    registered_subsets = tuple(subset for group in registered_subsets for subset in group)
    hyperedge_target, hyperedge_mask = _gme_hyperedge_targets(
        rho_clean, N, registered_subsets)
    pair_statistics = _aggregate_pair_sufficient_statistics(
        settings, outcome_panels, shots_by_setting, N)
    pair_coverage = np.zeros((N, N), dtype=np.bool_)
    for left, right in pair_statistics["pair_stat_qubits"]:
        pair_coverage[left, right] = pair_coverage[right, left] = True
    base = _base_mode_inputs(
        N, len(settings), registered_subsets=registered_subsets,
        correlation_adjacency=observed_correlation_adjacency(outcome_panels, N))
    base.update({
        "counts": np.asarray(marginal_counts, dtype=np.float32),
        "global_outcome_histogram": np.asarray(full_histograms, dtype=np.uint32),
        "full_outcome_mask": np.ones(
            np.asarray(full_histograms).shape, dtype=np.bool_),
        "subset_parent_setting_id": np.empty(0, dtype=np.int32),
        "subset_qubits": np.empty((0, 4), dtype=np.int16),
        "subset_order": np.empty(0, dtype=np.int8),
        "subset_axes": np.empty((0, 4), dtype=np.int8),
        "subset_marginal_counts": np.empty((0, 16), dtype=np.uint32),
        "subset_marginal_mask": np.empty((0, 16), dtype=np.bool_),
        "subset_shots": np.empty(0, dtype=np.uint32),
        "common_shot_group_id": np.empty(0, dtype=np.int64),
        "pair_coverage_mask": pair_coverage,
        "hyperedge_coverage_mask": np.empty(0, dtype=np.bool_),
        **pair_statistics,
        "basis_descriptors": descriptors,
        "shots": shots_by_setting.astype(np.float32),
        "setting_uncertainty": np.asarray(uncertainty, dtype=np.float32),
        "global_multi": render_global_multi(key.rng("global_multi")),
        "global_multi_mask": np.asarray(True, dtype=np.bool_),
        "environment": np.zeros(7, dtype=np.float32),
        "environment_mask": np.zeros(7, dtype=np.bool_),
    })
    coarse_groups = coarse_groups_for_system(f"{N}qubit")
    operational_noise, operational_noise_mask = _operational_noise_class(
        noise.mechanism_name,
        kappa_component=noise.kappa_component_achieved,
        control_mechanism=noise.control_mechanism_name,
        irrecoverable_mechanism=noise.irrecoverable_mechanism_name)
    target = {
        "target_p_clean": np.asarray(clean_marginals, dtype=np.float32),
        "target_p_noisy": np.asarray(noisy_marginals, dtype=np.float32),
        "target_measurement_mask": np.ones(len(settings), dtype=np.bool_),
        "target_fine": np.asarray(int(family["fine_family_id"]), dtype=np.int64),
        "target_fine_mask": np.asarray(True, dtype=np.bool_),
        "target_fine_identifiable_mask": np.asarray(
            V10MeasurementScheduler(N).contract().full_family_identifiable,
            dtype=np.bool_),
        "target_coarse": np.asarray(coarse_groups.index(str(family["coarse_group"])), dtype=np.int64),
        "target_coarse_mask": np.asarray(True, dtype=np.bool_),
        "target_pair": pair_target,
        "target_pair_mask": np.ones((N, N), dtype=np.bool_) ^ np.eye(N, dtype=np.bool_),
        "target_partition": (block_ids[:, None] == block_ids[None, :]).astype(np.float32),
        "target_partition_mask": (np.ones((N, N), dtype=np.bool_)
                                  ^ np.eye(N, dtype=np.bool_)) & partition_exact,
        "target_depth": np.asarray(depth - 1, dtype=np.int64),
        "target_intactness": np.asarray(intactness - 1, dtype=np.int64),
        "target_stretchability": np.asarray(depth - intactness + (N - 1), dtype=np.int64),
        "target_structure_mask": np.asarray(partition_exact, dtype=np.bool_),
        "target_R1_re": R1.real.astype(np.float32),
        "target_R1_im": R1.imag.astype(np.float32),
        "target_w_mask": w_mask,
        "target_sector": np.asarray((p0, p1, max(0.0, 1.0 - p0 - p1)), dtype=np.float32),
        "target_hyperedge": hyperedge_target,
        "target_hyperedge_mask": hyperedge_mask,
        "target_operational_noise": np.asarray(
            operational_noise, dtype=np.int64),
        "target_operational_noise_mask": np.asarray(
            operational_noise_mask, dtype=np.bool_),
        "target_certificate_status": np.asarray(0, dtype=np.int64),
        "target_certificate_mask": np.asarray(False, dtype=np.bool_),
        "target_ood": np.asarray(float(locked_ood), dtype=np.float32),
        "partition_truth_available": np.asarray(
            partition_exact, dtype=np.bool_),
        "certificate_complete": np.asarray(False, dtype=np.bool_),
        "epistemic_ambiguity": np.asarray(
            not partition_exact, dtype=np.bool_),
        "distribution_ood": np.asarray(locked_ood, dtype=np.bool_),
        "rho_clean_re": rho_clean.real.astype(np.float64),
        "rho_clean_im": rho_clean.imag.astype(np.float64),
        "rho_clean_mask": np.asarray(True, dtype=np.bool_),
        "correctable_mechanism": noise.correctable_multihot,
        "irrecoverable_mechanism": noise.irrecoverable_multihot,
        "mechanism_severity": noise.severity,
        "mechanism_name": np.asarray(noise.mechanism_name.encode("ascii"), dtype="S64"),
        "epsilon_irr": np.asarray(noise.epsilon_irr, dtype=np.float64),
        "eta_target": np.asarray(noise.eta_target, dtype=np.float64),
        "eta_achieved": np.asarray(noise.eta_achieved, dtype=np.float64),
        "eta_matching_residual": np.asarray(noise.matching_residual, dtype=np.float64),
        "eta_matching_converged": np.asarray(noise.matching_converged, dtype=np.bool_),
        "physical_noise_parameter": np.asarray(noise.physical_parameter, dtype=np.float64),
        "state_infidelity": np.asarray(noise.state_infidelity, dtype=np.float64),
        "d3_train_fit_sha": np.asarray(noise.train_fit_sha.encode("ascii"), dtype="S64"),
        "d3_heldout_error": np.asarray(noise.heldout_error, dtype=np.float64),
        "intervention_forward_sha": np.asarray(
            noise.intervention_forward_sha.encode("ascii"), dtype="S64"),
        "eta_ctrl_target": np.asarray([noise.eta_ctrl_target], dtype=np.float64),
        "eta_irr_target": np.asarray([noise.eta_irr_target], dtype=np.float64),
        "eta_ctrl_achieved": np.asarray([noise.eta_ctrl_achieved], dtype=np.float64),
        "eta_irr_achieved": np.asarray([noise.eta_irr_achieved], dtype=np.float64),
        "kappa_inj": np.asarray([noise.kappa_inj], dtype=np.float64),
        "kappa_achieved": np.asarray([noise.kappa_achieved], dtype=np.float64),
        "kappa_forward_achieved": np.asarray([noise.kappa_forward_achieved], dtype=np.float64),
        "kappa_target": np.asarray([noise.kappa_target], dtype=np.float64),
        "kappa_component_achieved": np.asarray(
            [noise.kappa_component_achieved], dtype=np.float64),
        "kappa_component_mask": np.asarray(
            noise.kappa_component_mask, dtype=np.bool_),
        "kappa_operational": np.asarray(
            [noise.kappa_operational], dtype=np.float64),
        "kappa_operational_mask": np.asarray(
            noise.kappa_operational_mask, dtype=np.bool_),
        "control_parameter": np.asarray([noise.control_parameter], dtype=np.float64),
        "irrecoverable_parameter": np.asarray([noise.irrecoverable_parameter], dtype=np.float64),
        "control_mechanism_name": np.asarray([noise.control_mechanism_name.encode("ascii")], dtype="S64"),
        "irrecoverable_mechanism_name": np.asarray([noise.irrecoverable_mechanism_name.encode("ascii")], dtype="S64"),
        "partition_exact_mask": np.asarray(partition_exact, dtype=np.bool_),
    }
    validate_partition_labels(block_ids, depth, intactness, partition_exact)
    return {"observation": base, "truth": target,
            "identities": {"latent_state_id": family["state_sha"] if "state_sha" in family else
                             sha256_array(rho_clean.view(np.float64)),
                             "transition_id": "", "crn_group_id": ""}}


def _apply_sampled_cell_noise(outcomes: np.ndarray, *, rng: np.random.Generator,
                              eta: float, kappa_inj: float) -> np.ndarray:
    """Compact-backend apparatus channel controlled by the registered cell."""
    value = np.array(outcomes, copy=True)
    if eta <= 0.0 or value.size == 0:
        return value
    correctable = rng.random(len(value)) < eta * kappa_inj
    if np.any(correctable):
        modes = rng.integers(value.shape[1], size=int(correctable.sum()))
        rows = np.flatnonzero(correctable)
        value[rows, modes] *= -1
    background = rng.random(len(value)) < eta * (1.0 - kappa_inj)
    if np.any(background):
        value[background] = rng.choice((-1, 1), size=(int(background.sum()), value.shape[1]))
    return value


def _explicit_fock_pair_density(
    backend: ExplicitFockMixtureBackend,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Exact reduced two-mode density from compact Fock sectors."""
    left, right = pair
    if not 0 <= left < right < backend.N:
        raise ValueError("explicit Fock pair must be ordered and in range")
    rho_pair = np.zeros((4, 4), dtype=np.complex128)
    single_mass = float(np.trace(backend.R1).real)
    outside_single = single_mass - float(
        backend.R1[left, left].real + backend.R1[right, right].real)
    pair_double = float(backend.pair_probabilities[left, right])
    left_double = float(backend.pair_probabilities[left].sum() - pair_double)
    right_double = float(backend.pair_probabilities[right].sum() - pair_double)
    outside_double = max(0.0, 1.0 - pair_double - left_double - right_double)
    rho_pair[0, 0] = backend.p0 + outside_single + backend.p2 * outside_double
    rho_pair[1, 1] = backend.R1[right, right].real + backend.p2 * right_double
    rho_pair[2, 2] = backend.R1[left, left].real + backend.p2 * left_double
    rho_pair[3, 3] = backend.p2 * pair_double
    rho_pair[2, 1] = backend.R1[left, right]
    rho_pair[1, 2] = backend.R1[right, left]
    return rho_pair


def _aggregate_pair_sufficient_statistics(
    settings,
    outcome_panels: Sequence[np.ndarray | None],
    shots_by_setting: np.ndarray,
    N: int,
    *,
    exact_expectation=None,
) -> Dict[str, np.ndarray]:
    """Aggregate each unique pair/axis statistic across global settings."""
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    product_sum = np.zeros((3, 3, N, N), dtype=np.float64)
    shot_sum = np.zeros((3, 3, N, N), dtype=np.uint64)
    for setting_index, setting in enumerate(settings):
        axes = np.asarray([
            axis_index[axis] for axis in setting.descriptor.pauli_axes],
            dtype=np.int8)
        outcomes = outcome_panels[setting_index]
        setting_shots = int(shots_by_setting[setting_index])
        correlation_sum = None if outcomes is None else (
            np.asarray(outcomes, dtype=np.float64).T
            @ np.asarray(outcomes, dtype=np.float64))
        for left_axis in range(3):
            left_modes = np.flatnonzero(axes == left_axis)
            for right_axis in range(3):
                right_modes = np.flatnonzero(axes == right_axis)
                if not len(left_modes) or not len(right_modes):
                    continue
                left_grid, right_grid = np.meshgrid(
                    left_modes, right_modes, indexing="ij")
                valid = left_grid < right_grid
                left_values = left_grid[valid]
                right_values = right_grid[valid]
                if not len(left_values):
                    continue
                shot_sum[left_axis, right_axis, left_values, right_values] += setting_shots
                if correlation_sum is not None:
                    product_sum[
                        left_axis, right_axis, left_values, right_values
                    ] += correlation_sum[left_values, right_values]
    pair_qubits = []
    pair_axes = []
    expectations = []
    variances = []
    pair_shots = []
    for left, right in itertools.combinations(range(N), 2):
        for left_axis in range(3):
            for right_axis in range(3):
                shots = int(shot_sum[left_axis, right_axis, left, right])
                if shots <= 0:
                    continue
                if exact_expectation is None:
                    expectation = float(
                        product_sum[left_axis, right_axis, left, right] / shots)
                    variance = max(0.0, 1.0 - expectation * expectation) / shots
                else:
                    expectation = float(exact_expectation(
                        (left, right), (left_axis, right_axis)))
                    variance = 0.0
                pair_qubits.append((left, right))
                pair_axes.append((left_axis, right_axis))
                expectations.append(expectation)
                variances.append(variance)
                pair_shots.append(shots)
    return {
        "pair_stat_qubits": np.asarray(pair_qubits, dtype=np.int16),
        "pair_stat_axes": np.asarray(pair_axes, dtype=np.int8),
        "pair_stat_expectation": np.asarray(expectations, dtype=np.float32),
        "pair_stat_variance": np.asarray(variances, dtype=np.float32),
        "pair_stat_shots": np.asarray(pair_shots, dtype=np.uint32),
        "pair_stat_coverage_mask": np.ones(len(pair_shots), dtype=np.bool_),
    }


def _render_structure(key, *, cell: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    renderer_rng = type("Rng", (), {"stream": lambda _, name: key.rng(name)})()
    rendered = V10StructureRenderer().render(key, renderer_rng)
    backend = rendered["state_backend"]
    settings = V10MeasurementScheduler(key.N).schedule(max_hyperedges=16)
    render = V24CellRenderParameters.from_mapping(cell, default_shots=8)
    render.require_schedule("registered_subsets_v1")
    shots_by_setting = render.shots_by_setting(len(settings))
    count_rows = []
    outcome_panels = []
    for setting_index, setting in enumerate(settings):
        setting_shots = int(shots_by_setting[setting_index])
        outcomes = sample_global_pauli_outcomes(
            backend, setting.descriptor.pauli_axes, setting_shots,
            key.rng("shot", setting_index))
        outcomes = _apply_sampled_cell_noise(
            outcomes, rng=key.rng("compact_noise", setting_index),
            eta=render.eta, kappa_inj=render.kappa)
        outcome_panels.append(outcomes)
        count_rows.append(pair_marginal_from_global(outcomes, (0, 1)))
    block_ids = np.empty(key.N, dtype=np.int16)
    for block_id, block in enumerate(rendered["observed_partition"]):
        block_ids[list(block)] = block_id
    registered_subsets = tuple(sorted({
        subset for setting in settings for subset in setting.covered_subsets
        if 2 <= len(subset) <= 4
    }, key=lambda value: (len(value), value)))
    pair_statistics = _aggregate_pair_sufficient_statistics(
        settings, outcome_panels, shots_by_setting, key.N)
    pair_coverage = np.zeros((key.N, key.N), dtype=np.bool_)
    for left, right in pair_statistics["pair_stat_qubits"]:
        pair_coverage[left, right] = pair_coverage[right, left] = True
    base = _base_mode_inputs(
        key.N, len(settings), registered_subsets=registered_subsets,
        correlation_adjacency=observed_correlation_adjacency(outcome_panels, key.N))
    base.update({
        "counts": np.asarray(count_rows, dtype=np.float32),
        "basis_descriptors": np.asarray([
            descriptor_vector(item.descriptor) for item in settings], dtype=np.float32),
        "shots": shots_by_setting.astype(np.float32),
        "setting_uncertainty": (0.25 / np.sqrt(shots_by_setting)).astype(np.float32),
        "global_multi": render_global_multi(key.rng("global_multi")),
        "global_multi_mask": np.asarray(True, dtype=np.bool_),
        "environment": np.zeros(7, dtype=np.float32),
        "environment_mask": np.zeros(7, dtype=np.bool_),
        "subset_parent_setting_id": np.empty(0, dtype=np.int32),
        "subset_qubits": np.empty((0, 4), dtype=np.int16),
        "subset_order": np.empty(0, dtype=np.int8),
        "subset_axes": np.empty((0, 4), dtype=np.int8),
        "subset_marginal_counts": np.empty((0, 16), dtype=np.uint32),
        "subset_marginal_mask": np.empty((0, 16), dtype=np.bool_),
        "subset_shots": np.empty(0, dtype=np.uint32),
        "common_shot_group_id": np.empty(0, dtype=np.int64),
        "pair_coverage_mask": pair_coverage,
        "hyperedge_coverage_mask": np.ones(len(registered_subsets), dtype=np.bool_),
        **pair_statistics,
    })
    exact = bool(rendered["partition_exact"])
    partition_target = (block_ids[:, None] == block_ids[None, :]).astype(np.float32)
    off_diagonal = np.ones((key.N, key.N), dtype=np.bool_) ^ np.eye(
        key.N, dtype=np.bool_)
    empirical_probability = (np.asarray(count_rows, dtype=np.float32)
                             / shots_by_setting[:, None].astype(np.float32))
    zero_r1 = np.zeros((key.N, key.N), dtype=np.float32)
    pair_target, pair_target_mask, edge_target, edge_target_mask = (
        _structure_entanglement_targets(
            rendered, registered_subsets, block_ids))
    largest_block = int(rendered["entanglement_depth"])
    coarse_template = min(9, int(np.ceil(np.log2(max(largest_block, 1)))))
    truth = {
        # Compact large-N backends do not expose an exact dense clean law.  The
        # placeholder keeps shard shapes regular; the explicit masks make the
        # unsupported clean/fine/W objectives fail closed during training.
        "target_p_clean": empirical_probability,
        "target_p_noisy": empirical_probability,
        "target_measurement_mask": np.zeros(len(settings), dtype=np.bool_),
        "target_fine": np.asarray(0, dtype=np.int64),
        "target_fine_mask": np.asarray(False, dtype=np.bool_),
        "target_fine_identifiable_mask": np.asarray(False, dtype=np.bool_),
        "target_coarse": np.asarray(coarse_template, dtype=np.int64),
        "target_coarse_mask": np.asarray(exact, dtype=np.bool_),
        "target_pair": pair_target,
        "target_pair_mask": pair_target_mask & off_diagonal & exact,
        "target_partition": partition_target,
        "target_partition_mask": off_diagonal & exact,
        "partition_block_ids": block_ids,
        "partition_exact_mask": np.asarray(exact, dtype=np.bool_),
        "target_depth": np.asarray(int(rendered["entanglement_depth"]) - 1, dtype=np.int64),
        "target_intactness": np.asarray(int(rendered["entanglement_intactness"]) - 1, dtype=np.int64),
        "target_stretchability": np.asarray(int(rendered["stretchability"]) + key.N - 1,
                                             dtype=np.int64),
        "target_structure_mask": np.asarray(exact, dtype=np.bool_),
        "target_R1_re": zero_r1,
        "target_R1_im": zero_r1,
        "target_w_mask": np.asarray(False, dtype=np.bool_),
        "target_sector": np.zeros(3, dtype=np.float32),
        "target_hyperedge": edge_target,
        "target_hyperedge_mask": edge_target_mask & exact,
        "target_operational_noise": np.asarray(5, dtype=np.int64),
        "target_operational_noise_mask": np.asarray(False, dtype=np.bool_),
        "target_certificate_status": np.asarray(0, dtype=np.int64),
        "target_certificate_mask": np.asarray(False, dtype=np.bool_),
        "target_ood": np.asarray(0.0, dtype=np.float32),
        "partition_truth_available": np.asarray(exact, dtype=np.bool_),
        "certificate_complete": np.asarray(False, dtype=np.bool_),
        "epistemic_ambiguity": np.asarray(not exact, dtype=np.bool_),
        "distribution_ood": np.asarray(False, dtype=np.bool_),
    }
    validate_partition_labels(block_ids, int(rendered["entanglement_depth"]),
                              int(rendered["entanglement_intactness"]), exact)
    return {"observation": base, "truth": truth,
            "identities": {"latent_state_id": rendered["state_sha"],
                             "transition_id": "", "crn_group_id": ""}}


def _render_w_control(key, *, cell: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    N = key.N
    state_rng = key.rng("state")
    coefficients = state_rng.normal(size=N) + 1j * state_rng.normal(size=N)
    coefficients /= np.linalg.norm(coefficients)
    p0 = float(state_rng.uniform(0.0, 0.08))
    p2 = float(state_rng.uniform(0.02, 0.12))
    p1 = 1.0 - p0 - p2
    R1 = p1 * np.outer(coefficients, coefficients.conj())
    pair_weight = state_rng.random((N, N))
    pair_weight = np.triu(pair_weight, 1)
    pair_weight /= pair_weight.sum()
    pair_probability = pair_weight + pair_weight.T
    backend = ExplicitFockMixtureBackend(p0, R1, pair_probability, p2)
    settings = V10MeasurementScheduler(N).schedule(max_hyperedges=0)
    render = V24CellRenderParameters.from_mapping(cell, default_shots=32)
    render.require_schedule("registered_subsets_v1")
    shots_by_setting = render.shots_by_setting(len(settings))
    registered_pairs = tuple(itertools.combinations(range(N), 2))
    representative_rho = _explicit_fock_pair_density(backend, (0, 1))
    representative_settings = tuple(itertools.product(("X", "Y", "Z"), repeat=2))
    noise = render_noise_truth(
        key.rng("w_control_noise"), rho_clean=representative_rho,
        settings=representative_settings, noise_family=render.noise_family,
        eta_target=render.eta, correctable_fraction=render.kappa,
        composite=True)

    def pair_law(pair: tuple[int, int], pair_axes: tuple[int, int]
                 ) -> np.ndarray:
        axis_names = ("X", "Y", "Z")
        axes = (axis_names[pair_axes[0]], axis_names[pair_axes[1]])
        rho_pair = _explicit_fock_pair_density(backend, pair)
        noisy_rho = noise.intervention.apply_state_channel(
            rho_pair, noise.physical_parameter)
        outcomes, probability = born_probabilities(
            DenseDensityStateBackend(noisy_rho), axes)
        probability = noise.intervention.apply_apparatus_channel(
            probability, outcomes, noise.physical_parameter)
        detected_law = noise.intervention.apply_detector_channel(
            probability, noise.physical_parameter)
        conditional = detected_law[:-1]
        return conditional / max(float(conditional.sum()), 1e-15)

    count_rows = []
    probability_rows = []
    setting_rng = key.rng("w_control_setting_counts")
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    for setting_index, setting in enumerate(settings):
        setting_shots = int(shots_by_setting[setting_index])
        axes = setting.descriptor.pauli_axes
        probability = pair_law(
            (0, 1), (axis_index[axes[0]], axis_index[axes[1]]))
        probability_rows.append(probability)
        if render.ideal_probability:
            count_rows.append(_deterministic_counts(probability, setting_shots))
        else:
            count_rows.append(setting_rng.multinomial(setting_shots, probability))

    def exact_expectation(pair, pair_axes):
        probability = pair_law(pair, pair_axes)
        return float(probability @ np.asarray((1.0, -1.0, -1.0, 1.0)))

    pair_statistics = _aggregate_pair_sufficient_statistics(
        settings, [None] * len(settings), shots_by_setting, N,
        exact_expectation=exact_expectation)
    if not render.ideal_probability:
        statistic_rng = key.rng("w_control_pair_statistics")
        expectation = pair_statistics["pair_stat_expectation"].astype(np.float64)
        statistic_shots = pair_statistics["pair_stat_shots"].astype(np.int64)
        plus_probability = np.clip((1.0 + expectation) / 2.0, 0.0, 1.0)
        plus_counts = statistic_rng.binomial(statistic_shots, plus_probability)
        pair_statistics["pair_stat_expectation"] = (
            2.0 * plus_counts / np.maximum(statistic_shots, 1) - 1.0
        ).astype(np.float32)
        pair_statistics["pair_stat_variance"] = (
            np.maximum(1.0 - expectation * expectation, 0.0)
            / np.maximum(statistic_shots, 1)
        ).astype(np.float32)
    pair_coverage = np.ones((N, N), dtype=np.bool_) ^ np.eye(N, dtype=np.bool_)
    correlation = np.zeros((N, N), dtype=np.float32)
    correlation_weight = np.zeros((N, N), dtype=np.float32)
    for (left, right), value in zip(
            pair_statistics["pair_stat_qubits"],
            np.abs(pair_statistics["pair_stat_expectation"])):
        correlation[left, right] += value
        correlation[right, left] += value
        correlation_weight[left, right] += 1.0
        correlation_weight[right, left] += 1.0
    correlation /= np.maximum(correlation_weight, 1.0)
    base = _base_mode_inputs(
        N, len(settings), registered_subsets=registered_pairs,
        correlation_adjacency=correlation)
    base.update({
        "counts": np.asarray(count_rows, dtype=np.float32),
        "measurement_probability": np.asarray(probability_rows, dtype=np.float32),
        "basis_descriptors": np.asarray([
            descriptor_vector(item.descriptor) for item in settings], dtype=np.float32),
        "shots": shots_by_setting.astype(np.float32),
        "setting_uncertainty": (
            np.zeros(len(settings), dtype=np.float32)
            if render.ideal_probability else
            np.asarray([
                _multinomial_cell_uncertainty(probability, int(shots))
                for probability, shots in zip(probability_rows, shots_by_setting)
            ], dtype=np.float32)),
        "deterministic_expected_count_mask": np.asarray(
            render.ideal_probability, dtype=np.bool_),
        "global_multi": render_global_multi(key.rng("global_multi")),
        "global_multi_mask": np.asarray(True, dtype=np.bool_),
        "environment": np.zeros(7, dtype=np.float32),
        "environment_mask": np.zeros(7, dtype=np.bool_),
        "subset_parent_setting_id": np.empty(0, dtype=np.int32),
        "subset_qubits": np.empty((0, 4), dtype=np.int16),
        "subset_order": np.empty(0, dtype=np.int8),
        "subset_axes": np.empty((0, 4), dtype=np.int8),
        "subset_marginal_counts": np.empty((0, 16), dtype=np.uint32),
        "subset_marginal_mask": np.empty((0, 16), dtype=np.bool_),
        "subset_shots": np.empty(0, dtype=np.uint32),
        "common_shot_group_id": np.empty(0, dtype=np.int64),
        "pair_coverage_mask": pair_coverage,
        "hyperedge_coverage_mask": np.ones(len(registered_pairs), dtype=np.bool_),
        **pair_statistics,
    })
    truth = {
        "p0": np.asarray(p0, dtype=np.float32), "p1": np.asarray(p1, dtype=np.float32),
        "p2": np.asarray(p2, dtype=np.float32), "p3_plus": np.asarray(0.0, dtype=np.float32),
        "R1_re": R1.real.astype(np.float64), "R1_im": R1.imag.astype(np.float64),
        "pair_probability": pair_probability.astype(np.float64),
        "multi_excitation_backend_mask": np.asarray(True, dtype=np.bool_),
        "clean_coefficients_re": coefficients.real.astype(np.float64),
        "clean_coefficients_im": coefficients.imag.astype(np.float64),
        "eta_target": np.asarray(noise.eta_target, dtype=np.float64),
        "eta_achieved": np.asarray(noise.eta_achieved, dtype=np.float64),
        "eta_matching_residual": np.asarray(
            noise.matching_residual, dtype=np.float64),
        "eta_matching_converged": np.asarray(
            noise.matching_converged, dtype=np.bool_),
        "kappa_inj": np.asarray([noise.kappa_inj], dtype=np.float64),
        "kappa_achieved": np.asarray(
            [noise.kappa_forward_achieved], dtype=np.float64),
        "kappa_forward_achieved": np.asarray(
            [noise.kappa_forward_achieved], dtype=np.float64),
        "kappa_target": np.asarray([noise.kappa_target], dtype=np.float64),
        "kappa_component_achieved": np.asarray(
            [noise.kappa_component_achieved], dtype=np.float64),
        "kappa_component_mask": np.asarray(
            noise.kappa_component_mask, dtype=np.bool_),
        "kappa_operational": np.asarray(
            [noise.kappa_operational], dtype=np.float64),
        "kappa_operational_mask": np.asarray(
            noise.kappa_operational_mask, dtype=np.bool_),
        "control_mechanism_name": np.asarray(
            [noise.control_mechanism_name.encode("ascii")], dtype="S64"),
        "irrecoverable_mechanism_name": np.asarray(
            [noise.irrecoverable_mechanism_name.encode("ascii")], dtype="S64"),
        "mechanism_name": np.asarray(
            noise.mechanism_name.encode("ascii"), dtype="S64"),
    }
    return {"observation": base, "truth": truth,
            "identities": {"latent_state_id": backend.state_sha(),
                             "transition_id": "", "crn_group_id": ""}}


def _feedback_classifier_observation(
    key, rendered: Mapping[str, Any], *, cell: Mapping[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Add the frozen C_N measurement domain to one feedback observation."""
    N = key.N
    if N not in (3, 4):
        raise ValueError("feedback semantic cache currently supports N=3 or N=4")
    render = V24CellRenderParameters.from_mapping(cell, default_shots=256)
    settings = V10MeasurementScheduler(N).schedule()
    shots_by_setting = render.shots_by_setting(len(settings))
    internal = rendered["internal"]
    backend = DenseDensityStateBackend(np.asarray(
        internal["rho_noisy"], dtype=np.complex128))
    intervention = internal["intervention"]
    physical_parameter = float(internal["physical_parameter"])
    shot_rng = key.rng("classifier_feedback_shot")
    histograms = []
    marginal_counts = []
    outcome_panels = []
    uncertainty = []
    for setting_index, setting in enumerate(settings):
        setting_shots = int(shots_by_setting[setting_index])
        outcomes, state_probability = born_probabilities(
            backend, setting.descriptor.pauli_axes)
        apparatus_probability = intervention.apply_apparatus_channel(
            state_probability, outcomes, physical_parameter)
        detected_law = intervention.apply_detector_channel(
            apparatus_probability, physical_parameter)
        if render.ideal_probability:
            sampled = _outcomes_from_counts(
                outcomes, _deterministic_counts(
                    detected_law[:-1], setting_shots))
        else:
            indices = shot_rng.choice(
                len(detected_law), size=setting_shots, p=detected_law)
            sampled = outcomes[indices[indices < len(outcomes)]]
        outcome_panels.append(sampled)
        histograms.append(outcome_histogram(sampled, N))
        marginal_counts.append(pair_marginal_from_global(sampled, (0, 1)))
        pair_probability = np.asarray(marginal_counts[-1], dtype=np.float64)
        pair_probability /= max(float(pair_probability.sum()), 1.0)
        uncertainty.append(
            0.0 if render.ideal_probability else
            _multinomial_cell_uncertainty(pair_probability, setting_shots))
    base = _base_mode_inputs(
        N, len(settings),
        correlation_adjacency=observed_correlation_adjacency(outcome_panels, N))
    base.update({
        "counts": np.asarray(marginal_counts, dtype=np.float32),
        "global_outcome_histogram": np.asarray(histograms, dtype=np.uint32),
        "full_outcome_mask": np.ones(
            np.asarray(histograms).shape, dtype=np.bool_),
        "basis_descriptors": np.asarray([
            descriptor_vector(item.descriptor) for item in settings],
            dtype=np.float32),
        "shots": shots_by_setting.astype(np.float32),
        "setting_uncertainty": np.asarray(uncertainty, dtype=np.float32),
        "subset_parent_setting_id": np.empty(0, dtype=np.int32),
        "subset_qubits": np.empty((0, 4), dtype=np.int16),
        "subset_order": np.empty(0, dtype=np.int8),
        "subset_axes": np.empty((0, 4), dtype=np.int8),
        "subset_marginal_counts": np.empty((0, 16), dtype=np.uint32),
        "subset_marginal_mask": np.empty((0, 16), dtype=np.bool_),
        "subset_shots": np.empty(0, dtype=np.uint32),
        "common_shot_group_id": np.empty(0, dtype=np.int64),
        "pair_coverage_mask": np.zeros((N, N), dtype=np.bool_),
        "hyperedge_coverage_mask": np.empty(0, dtype=np.bool_),
        "global_multi": render_global_multi(
            key.rng("classifier_feedback_global_multi")),
        "global_multi_mask": np.asarray(True, dtype=np.bool_),
        "environment": np.zeros(7, dtype=np.float32),
        "environment_mask": np.zeros(7, dtype=np.bool_),
    })
    return base


def _render_dataset_family_base(key, family: str,
                                cell: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    if family == "small_n_family":
        rendered = _render_classifier(key, cell=cell)
    elif family == "general_n_structure":
        rendered = _render_structure(key, cell=cell)
    elif family == "w_control":
        rendered = _render_w_control(key, cell=cell)
    elif family in ("feedback_one_step", "feedback_history", "feedback_noop"):
        render = V24CellRenderParameters.from_mapping(cell, default_shots=256)
        render.require_schedule("phase_sensitive_w_v1")
        rendered = render_feedback_sample(
            key, shot_mode=render.shot_mode, shot_budget=render.shot_budget,
            noise_family=render.noise_family, eta=render.eta, kappa=render.kappa,
            composite_noise=True)
        rendered["observation"].update(
            _feedback_classifier_observation(key, rendered, cell=cell))
        if family == "feedback_history":
            rendered = render_feedback_history(
                key, rendered, shot_mode=render.shot_mode,
                shot_budget=render.shot_budget)
        if family == "feedback_noop":
            zero_action = np.zeros_like(rendered["truth"]["target_action"])
            rendered["truth"]["oracle_action"] = zero_action
            rendered["truth"]["target_action"] = zero_action
            rendered["truth"]["target_gain"] = np.asarray(0.0, dtype=np.float32)
            rendered["truth"]["target_state_fidelity_gain"] = np.asarray(0.0, dtype=np.float32)
            rendered["truth"]["target_next_measurement"] = rendered["truth"]["target_noop_next_measurement"]
            rendered["truth"]["true_next_measurement"] = rendered["truth"]["target_noop_next_measurement"]
            rendered["truth"]["target_harm"] = np.asarray(0.0, dtype=np.float32)
            rendered["truth"]["target_stop"] = np.asarray(1.0, dtype=np.float32)
            rendered["truth"]["target_history_gain"] = np.asarray(0.0, dtype=np.float32)
            rendered["truth"]["target_history_gain_mask"] = np.asarray(False, dtype=np.bool_)
            rendered["truth"]["history_round_count"] = np.asarray(0, dtype=np.int16)
            rendered["truth"]["oracle_solver_success"] = np.asarray(True, dtype=np.bool_)
            rendered["truth"]["oracle_solver_status"] = np.asarray(
                b"registered_noop", dtype="S128")
            rendered["truth"]["oracle_action_mask"] = np.asarray(True, dtype=np.bool_)
            rendered["truth"]["oracle_objective_after"] = np.asarray(
                rendered["truth"]["oracle_objective_before"], dtype=np.float64)
            rendered["truth"]["oracle_constraint_violation"] = np.asarray(
                0.0, dtype=np.float64)
    elif family in ("mechanism_shift", "ood_detector_drift", "locked_ood_composite",
                    "negative_control"):
        locked = family in ("mechanism_shift", "ood_detector_drift", "locked_ood_composite")
        rendered = _render_classifier(key, cell=cell, locked_ood=locked)
        rendered["truth"]["target_ood"] = np.asarray(float(locked), dtype=np.float32)
        rendered["truth"]["negative_control_mask"] = np.asarray(
            family == "negative_control", dtype=np.bool_)
    else:
        raise ValueError(f"V2.4 has no physical renderer for dataset family {family}")
    return rendered


def _render_dataset_family(key, *, cell: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    family, negative_control_kind = split_negative_control_family(key.dataset_family)
    evaluation_suffix = ""
    if family.endswith("_locked_ood"):
        family = family[:-len("_locked_ood")]
        evaluation_suffix = "locked_ood"
    if evaluation_suffix:
        if family != "small_n_family":
            raise ValueError(
                "locked OOD is registered only for the small-N dense classifier")
        paired_key = make_sample_key(
            global_seed=key.global_seed, split="validation",
            dataset_family=family, N=key.N,
            index=int(key.sample_id.rsplit(":", 1)[1]),
            generator_sha=key.generator_sha,
            cell_id=key.cell_id.removesuffix("_locked_ood"))
        rendered = _render_classifier(paired_key, cell=cell, locked_ood=True)
        rendered["truth"]["target_ood"] = np.asarray(1.0, dtype=np.float32)
        rendered["truth"]["locked_ood_pair_id"] = np.asarray(
            paired_key.sample_id.encode("utf-8"), dtype="S192")
    else:
        rendered = _render_dataset_family_base(key, family, cell)
    if negative_control_kind:
        partner_rendered = None
        if negative_control_kind == "target_shuffle":
            sample_index = int(key.sample_id.rsplit(":", 1)[1])
            partner_key = make_sample_key(
                global_seed=key.global_seed, split=key.split,
                dataset_family=key.dataset_family, N=key.N,
                index=sample_index ^ 1, generator_sha=key.generator_sha,
                cell_id=key.cell_id)
            partner_rendered = _render_dataset_family_base(partner_key, family, cell)
        rendered = apply_persistent_negative_control(
            rendered, kind=negative_control_kind, key=key,
            partner_rendered=partner_rendered)
    return rendered


def _stack(samples: Sequence[Mapping[str, np.ndarray]], sample_ids: Sequence[str]
           ) -> Dict[str, np.ndarray]:
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError("homogeneous cell rows have inconsistent fields")
    result = {key: np.stack([np.asarray(sample[key]) for sample in samples]) for key in sorted(keys)}
    maximum = max(len(value.encode("utf-8")) for value in sample_ids)
    result["sample_id"] = np.asarray(sample_ids, dtype=f"S{maximum}")
    result["row_valid_mask"] = np.ones(len(sample_ids), dtype=np.bool_)
    return result


def _row_sha(observation: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray], row: int) -> str:
    return sha256_json({
        "observation": {key: sha256_array(np.asarray(value[row])) for key, value in sorted(observation.items())},
        "truth": {key: sha256_array(np.asarray(value[row])) for key, value in sorted(truth.items())},
    })


def generate_smoke_build(dataset_root: str | Path, *, global_seed: int = 20261001,
                         plan: Sequence[Mapping[str, Any]] = DEFAULT_SMOKE_PLAN,
                         build_kind: str = "smoke") -> Dict[str, Any]:
    root = Path(dataset_root)
    if build_kind != "smoke":
        raise ValueError(
            "registered rungs must use the bounded shard producer, not the smoke builder")
    build_id = build_kind + "_" + sha256_json({"seed": global_seed, "plan": list(plan),
                                                "generator": V24_GENERATOR_SHA})[:16]
    if (root / "READY.json").exists():
        from ..loaders import validate_dataset_root
        from ..hashing import read_canonical_json
        identity = validate_dataset_root(root, build_id=build_id)
        validation = read_canonical_json(root / "reports" / f"{build_id}_validation.json")
        return {**validation, **identity, "dataset_root": str(root),
                "cells": list(plan), "shard_count": len(
                    read_canonical_json(root / "READY.json")["shard_hashes"])}
    freeze = freeze_contract(root)
    contract = load_contract(root)
    source_commit, source_tree = _git_identity()
    shard_results: Dict[str, str] = {}
    index_rows = []
    manifest_cells = []
    feedback_derived: Dict[str, list[tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]]] = {}
    for cell in plan:
        cell_id = str(cell["cell_id"])
        family = str(cell["dataset_family"])
        N = int(cell["N"])
        for split in ("train", "validation"):
            samples = []
            sample_ids = []
            keys = []
            for index in range(int(cell[split])):
                key = make_sample_key(global_seed=global_seed, split=split,
                                      dataset_family=family, N=N, index=index,
                                      generator_sha=V24_GENERATOR_SHA, cell_id=cell_id)
                rendered = _render_dataset_family(key, cell=cell)
                samples.append(rendered)
                sample_ids.append(key.sample_id)
                keys.append(key)
            observations = _stack([sample["observation"] for sample in samples], sample_ids)
            truths = _stack([sample["truth"] for sample in samples], sample_ids)
            validate_global_multi(
                observations["global_multi"], observations["global_multi_mask"],
                ["state"] * len(samples)) if "global_multi" in observations else None
            validate_view_keys("classifier_input" if family != "feedback_one_step"
                               else "feedback_student_base", observations)
            metadata = {
                "provenance": "synthetic", "dataset_family": family,
                "store": "observation",
                "cell_id": cell_id, "split": split, "system_type": cell["system_type"],
                "N": N, "measurement_domain": cell["measurement_domain"],
                "schedule_id": "full_pauli" if N <= 4 else "b1_prime_strength_two",
                "shot_mode": "fixed_per_setting", "storage_tier": "CORE",
                "contract_sha": contract.sha256(), "build_id": build_id,
            }
            observation_path = root / "observations" / "synthetic" / f"{cell_id}_{split}.h5"
            truth_path = root / "truth" / "synthetic" / f"{cell_id}_{split}.h5"
            observation_result = write_hdf5_shard(observation_path, observations, metadata)
            truth_metadata = {**metadata, "storage_tier": "EXTENDED", "store": "truth"}
            truth_result = write_hdf5_shard(truth_path, truths, truth_metadata)
            observation_relative = observation_path.relative_to(root).as_posix()
            truth_relative = truth_path.relative_to(root).as_posix()
            shard_results[observation_relative] = observation_result["sha256"]
            shard_results[truth_relative] = truth_result["sha256"]
            for row, (key, rendered) in enumerate(zip(keys, samples)):
                identities = rendered["identities"]
                index_rows.append({
                    "sample_id": key.sample_id,
                    "latent_state_id": identities["latent_state_id"],
                    "transition_id": identities["transition_id"],
                    "crn_group_id": identities["crn_group_id"],
                    "session_id": "", "acquisition_run_id": "",
                    "repeat_group_id": "", "trajectory_id": "",
                    "provenance": "synthetic", "dataset_family": family,
                    "cell_id": cell_id, "split": split, "system_type": cell["system_type"],
                    "N": N, "measurement_domain": cell["measurement_domain"],
                    "observation_shard": observation_relative, "observation_row": row,
                    "truth_shard": truth_relative, "truth_row": row,
                    "sample_key_sha": key.sha256(),
                    "row_content_sha": _row_sha(observations, truths, row),
                    "contract_sha": contract.sha256(), "build_id": build_id,
                    "derived_rho8": 0, "derived_count_mle": 0,
                    "derived_classifier": int("derived_classifier" in rendered),
                    "derived_control": int("derived_control" in rendered),
                })
            if family.startswith("feedback"):
                feedback_derived[f"{cell_id}:{split}"] = [
                    (sample_id, sample["derived_classifier"], sample["derived_control"])
                    for sample_id, sample in zip(sample_ids, samples)
                ]
        manifest_cells.append({**dict(cell), "schedule_id": (
            "full_pauli" if N <= 4 else "b1_prime_strength_two"),
            "shot_mode": "fixed_per_setting", "storage_tier": "CORE"})
    classifier_checkpoint_sha = sha256_json({"kind": "v2_4_smoke_classifier", "build_id": build_id})
    control_forward_sha = sha256_json({"forward": "global_pauli_w_v24", "build_id": build_id})
    for key, derived_rows in feedback_derived.items():
        cell_id, split = key.split(":")
        ids = [row[0] for row in derived_rows]
        classifier_arrays = _stack([row[1] for row in derived_rows], ids)
        control_arrays = _stack([row[2] for row in derived_rows], ids)
        classifier_path = root / "derived" / "classifier" / classifier_checkpoint_sha / f"{cell_id}_{split}.h5"
        control_path = root / "derived" / "control" / control_forward_sha / f"{cell_id}_{split}.h5"
        derived_metadata = {"build_id": build_id, "contract_sha": contract.sha256(),
                            "cell_id": cell_id, "split": split,
                            "classifier_checkpoint_sha": classifier_checkpoint_sha,
                            "generator_kind": "fixture_heuristic",
                            "fixture_heuristic": True,
                            "store": "derived", "provenance": "synthetic", "N": 4}
        result = write_hdf5_shard(classifier_path, classifier_arrays, derived_metadata)
        shard_results[classifier_path.relative_to(root).as_posix()] = result["sha256"]
        result = write_hdf5_shard(control_path, control_arrays, {
            **derived_metadata, "forward_sha": control_forward_sha})
        shard_results[control_path.relative_to(root).as_posix()] = result["sha256"]
    index_path = root / "index" / "index.sqlite"
    index_result = create_index(index_path, index_rows)
    manifest = V24BuildManifest(
        build_id=build_id, contract_sha=contract.sha256(), schema_sha=schema_sha256(),
        generator_sha=V24_GENERATOR_SHA, source_commit=source_commit,
        source_tree_sha=source_tree, global_seed=global_seed,
        cells=tuple(manifest_cells))
    manifest_path = root / "manifests" / f"{build_id}.json"
    manifest_sha = write_canonical_json(manifest_path, manifest.to_dict())
    split_report = validate_split_groups(index_rows)
    validation_report = {
        "build_id": build_id, "rows": len(index_rows), "split": split_report,
        "truth_visibility": True, "physics_validators": True,
        "classifier_checkpoint_sha": classifier_checkpoint_sha,
        "control_forward_sha": control_forward_sha,
        "research_only": True, "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False,
    }
    validate_authority_flags(validation_report)
    validation_path = root / "reports" / f"{build_id}_validation.json"
    validation_sha = write_canonical_json(validation_path, validation_report)
    split_sha = sha256_json({row["sample_id"]: row["split"] for row in index_rows})
    ready = V24ReadyManifest(
        build_id=build_id, build_manifest_sha=manifest_sha,
        contract_sha=contract.sha256(), schema_sha=schema_sha256(),
        index_sha=index_result["sha256"], split_sha=split_sha,
        shard_hashes=shard_results, validation_report_sha=validation_sha,
        readiness_checks=("shard_hashes", "sqlite_integrity", "split_disjoint",
                          "group_isolation", "truth_visibility", "physics_validators"))
    ready_path = root / "READY.json"
    ready_sha = write_canonical_json(ready_path, ready.to_dict())
    return {**validation_report, "contract_sha": freeze["contract_sha"],
            "schema_sha": schema_sha256(), "build_manifest_sha": manifest_sha,
            "index_sha": index_result["sha256"], "ready_sha": ready_sha,
            "dataset_root": str(root), "cells": manifest_cells,
            "shard_count": len(shard_results)}
