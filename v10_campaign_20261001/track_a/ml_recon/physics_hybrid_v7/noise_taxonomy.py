"""Synthetic noise taxonomy for low-count V7 representation pretraining.

The taxonomy labels simulator mechanisms, not asserted causes of the measured
campaign.  Real snapshots can receive posterior scores, but a score is never
treated as detector calibration or permission to subtract counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ml_recon.physics_hybrid_v5.full_synthetic_generator import sample_nb2

from .geometry import OperationFrequencyGeometryV1
from .physical_ceiling import retrieval_ceiling_features


NOISE_TYPE_NAMES = (
    "poisson_reference",
    "nb2_overdispersion",
    "accidental_dominant",
    "smooth_gain_drift",
    "sparse_burst",
    "cycle_memory_crosstalk",
    "uniform_background",
)
NOISE_COARSE_TYPE_NAMES = (
    "reference_or_weak",
    "independent_overdispersion",
    "mixture_background",
    "structured_correlated",
)
NOISE_FINE_TO_COARSE = np.asarray((0, 1, 2, 3, 3, 3, 2), dtype=np.int64)
OPERATION_INPUT_FEATURE_NAMES = (
    "observed_coin_log1p",
    "signal_log1p",
    "idler_log1p",
    "observed_coin_log_share",
    "signal_log_share",
    "idler_log_share",
    "observed_coin_mask",
    "x_frequency",
    "y_frequency",
    "x_y_frequency_interaction",
    "x_frequency_squared",
    "y_frequency_squared",
    "cycle_sin",
    "cycle_cos",
    "basis_amplitude_mean",
    "basis_amplitude_std",
    "basis_amplitude_power",
    "basis_phase_sin_mean",
    "basis_phase_cos_mean",
    "retrieval_mean",
    "retrieval_cv",
    "retrieval_min",
    "retrieval_max",
    "target_measurement_overlap",
    "program_measurement_overlap",
    "log_accidental_seed",
)
# V83: optional physics-enriched feature names (4 additional DLCZ features).
# When enabled via build_operation_input_features(..., physics_enriched=True),
# these are appended after the base 26 features, yielding 30-dim tokens.
PHYSICS_ENRICHED_FEATURE_NAMES = (
    "bose_thermal_excitation",      # sqrt(signal * idler): n_bar
    "signal_idler_log_ratio",       # log(signal+1) - log(idler+1)
    "poisson_noise_estimate",       # 1/sqrt(coin+1): relative uncertainty
    "accidental_fraction",          # accidental / (accidental + coin)
)
TARGET_CONTEXT_FEATURE_NAMES = tuple(
    [f"target_amplitude_{index + 1}" for index in range(4)]
    + [f"target_phase_sin_{index + 1}" for index in range(4)]
    + [f"target_phase_cos_{index + 1}" for index in range(4)]
    + [f"program_amplitude_{index + 1}" for index in range(4)]
    + [f"program_phase_sin_{index + 1}" for index in range(4)]
    + [f"program_phase_cos_{index + 1}" for index in range(4)]
    + ["program_command_available"]
)


@dataclass(frozen=True)
class V7SyntheticPhysicsSupport:
    system_type: str
    static_features: np.ndarray
    graph_modes: np.ndarray
    graph_kernel: np.ndarray
    retrieval_rate_vectors: np.ndarray
    geometry_hash: str


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values, axis=1, keepdims=True)
    probability = np.exp(np.clip(centered, -60.0, 60.0))
    return probability / np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)


def _log_share(values: np.ndarray, smoothing: float = 0.5) -> np.ndarray:
    counts = np.asarray(values, dtype=float)
    share = (counts + float(smoothing)) / (
        np.sum(counts, axis=1, keepdims=True) + float(smoothing) * counts.shape[1]
    )
    return np.log(np.maximum(share, 1e-12))


def encode_target_program_context(
    target_amplitudes: Any,
    target_phases_rad: Any,
    program_amplitudes: Any,
    program_phases_rad: Any,
    program_command_available: Any,
    *,
    qubit_count: int,
) -> np.ndarray:
    """Encode desired and snapshot-bound executed state without normalizing raw amp."""

    target_amplitude = np.asarray(target_amplitudes, dtype=float)
    target_phase = np.asarray(target_phases_rad, dtype=float)
    program_amplitude = np.asarray(program_amplitudes, dtype=float)
    program_phase = np.asarray(program_phases_rad, dtype=float)
    if any(array.ndim != 2 for array in (
        target_amplitude, target_phase, program_amplitude, program_phase,
    )):
        raise ValueError("target/program arrays must have shape [batch, qubit]")
    expected_shape = (target_amplitude.shape[0], int(qubit_count))
    if any(array.shape != expected_shape for array in (
        target_amplitude, target_phase, program_amplitude, program_phase,
    )):
        raise ValueError("target/program arrays disagree with qubit_count")
    available = np.asarray(program_command_available, dtype=float).reshape(-1, 1)
    if available.shape[0] != expected_shape[0]:
        raise ValueError("program command availability has the wrong length")
    padded = []
    for array in (target_amplitude, target_phase, program_amplitude, program_phase):
        values = np.zeros((expected_shape[0], 4), dtype=np.float64)
        values[:, :qubit_count] = array
        padded.append(values)
    return np.concatenate((
        padded[0], np.sin(padded[1]), np.cos(padded[1]),
        padded[2], np.sin(padded[3]), np.cos(padded[3]),
        np.clip(available, 0.0, 1.0),
    ), axis=1)


def build_operation_input_features(
    support: V7SyntheticPhysicsSupport,
    *,
    observed_coin_counts: Any,
    signal_counts: Any,
    idler_counts: Any,
    observed_mask: Any,
    target_overlap: Any,
    program_overlap: Any,
    physics_enriched: bool = False,
) -> np.ndarray:
    """Build the fixed operation-token schema for synthetic or measured rows.

    When ``physics_enriched=True``, appends 4 DLCZ-specific physics features
    (V83): bose_thermal_excitation, signal_idler_log_ratio,
    poisson_noise_estimate, accidental_fraction.  Total dim 26→30.
    """

    coin = np.asarray(observed_coin_counts, dtype=float)
    signal = np.asarray(signal_counts, dtype=float)
    idler = np.asarray(idler_counts, dtype=float)
    mask = np.asarray(observed_mask, dtype=float)
    target = np.asarray(target_overlap, dtype=float)
    program = np.asarray(program_overlap, dtype=float)
    expected_shape = (coin.shape[0], support.static_features.shape[0])
    if any(array.shape != expected_shape for array in (
        coin, signal, idler, mask, target, program,
    )):
        raise ValueError("operation input arrays disagree with synthetic support")
    accidental_seed = (signal + 0.5) * (idler + 0.5)
    # V83: optional physics-derived features.
    bose_thermal = np.sqrt(np.maximum(signal * idler, 0.0))
    signal_idler_ratio = np.log1p(signal) - np.log1p(idler)
    poisson_noise = 1.0 / np.sqrt(np.maximum(coin, 0.0) + 1.0)
    acc_total = accidental_seed + np.maximum(coin, 0.0) + 1.0
    acc_fraction = accidental_seed / acc_total
    normalized_rates = support.retrieval_rate_vectors / max(
        float(np.max(support.retrieval_rate_vectors)), 1e-12,
    )
    retrieval_summary = np.stack((
        np.mean(normalized_rates, axis=1),
        np.std(normalized_rates, axis=1)
        / np.maximum(np.mean(normalized_rates, axis=1), 1e-12),
        np.min(normalized_rates, axis=1),
        np.max(normalized_rates, axis=1),
    ), axis=1)
    static_batch = np.broadcast_to(
        support.static_features[None, :, :],
        (coin.shape[0],) + support.static_features.shape,
    )
    retrieval_batch = np.broadcast_to(
        retrieval_summary[None, :, :],
        (coin.shape[0],) + retrieval_summary.shape,
    )
    parts = [
        np.log1p(coin)[:, :, None],
        np.log1p(signal)[:, :, None],
        np.log1p(idler)[:, :, None],
        _log_share(coin)[:, :, None],
        _log_share(signal)[:, :, None],
        _log_share(idler)[:, :, None],
        mask[:, :, None],
        static_batch,
        retrieval_batch,
        target[:, :, None],
        program[:, :, None],
        np.log1p(accidental_seed)[:, :, None],
    ]
    if physics_enriched:
        parts.extend([
            bose_thermal[:, :, None],
            signal_idler_ratio[:, :, None],
            poisson_noise[:, :, None],
            acc_fraction[:, :, None],
        ])
    return np.concatenate(parts, axis=2).astype(np.float32)


def build_synthetic_physics_support(
    geometry: OperationFrequencyGeometryV1,
    retrieval_rate_vectors: Any,
    *,
    graph_mode_count: int = 12,
    graph_length_mhz: float = 8.0,
    sparse_k: int = 0,
) -> V7SyntheticPhysicsSupport:
    """Build synthetic physics support with optional sparse graph kernel.

    When ``sparse_k > 0``, uses k-NN sparse kernel (V86) instead of the
    default dense Gaussian kernel.  This reduces interior-operation
    over-smoothing that drives miscalibration (V85 finding).
    """
    rates = np.asarray(retrieval_rate_vectors, dtype=float)
    if rates.shape != (geometry.operation_count, geometry.qubit_count):
        raise ValueError("retrieval_rate_vectors disagree with geometry")
    if sparse_k > 0:
        kernel = geometry.sparse_knn_kernel(
            x_length_mhz=float(graph_length_mhz),
            y_length_mhz=float(graph_length_mhz),
            same_local_cross_cycle_weight=0.05,
            k=int(sparse_k),
        )
    else:
        kernel = geometry.coupling_kernel(
        x_length_mhz=float(graph_length_mhz),
        y_length_mhz=float(graph_length_mhz),
        same_local_cross_cycle_weight=0.05,
    )
    symmetric = 0.5 * (kernel + kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    mode_count = min(max(1, int(graph_mode_count)), geometry.operation_count)
    modes = eigenvectors[:, np.argsort(eigenvalues)[-mode_count:]].T
    for index in range(modes.shape[0]):
        pivot = int(np.argmax(np.abs(modes[index])))
        if modes[index, pivot] < 0.0:
            modes[index] *= -1.0
    static, _ = geometry.static_features()
    return V7SyntheticPhysicsSupport(
        system_type=geometry.system_type,
        static_features=static.astype(np.float64),
        graph_modes=modes.astype(np.float64),
        graph_kernel=kernel.astype(np.float64),
        retrieval_rate_vectors=rates.astype(np.float64),
        geometry_hash=geometry.geometry_hash,
    )


def _target_batch(
    geometry: OperationFrequencyGeometryV1,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    qubit_count = geometry.qubit_count
    population = rng.dirichlet(np.full(qubit_count, 1.3), size=int(batch_size))
    amplitude = np.sqrt(population)
    phase = rng.uniform(-np.pi, np.pi, size=(int(batch_size), qubit_count))
    phase -= phase[:, :1]
    state = amplitude * np.exp(1j * phase)
    basis = geometry.basis_amplitudes * np.exp(1j * geometry.basis_phases_rad)
    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    overlap = np.square(np.abs(state @ basis.conj().T))
    program_amplitude = amplitude * np.exp(rng.normal(0.0, 0.12, size=amplitude.shape))
    program_phase = phase + rng.normal(0.0, 0.18, size=phase.shape)
    program_phase -= program_phase[:, :1]
    program_state = program_amplitude * np.exp(1j * program_phase)
    program_state /= np.maximum(np.linalg.norm(program_state, axis=1, keepdims=True), 1e-12)
    program_overlap = np.square(np.abs(program_state @ basis.conj().T))
    context = encode_target_program_context(
        amplitude, phase, program_amplitude, program_phase,
        np.ones(int(batch_size), dtype=np.float64), qubit_count=qubit_count,
    )
    targets = [
        {"amplitudes": amplitude[index].tolist(), "phases_rad": phase[index].tolist()}
        for index in range(int(batch_size))
    ]
    return context, overlap, program_overlap, targets


def _observation_mask(
    batch_size: int,
    operation_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    index = np.arange(operation_count)
    masks = np.zeros((int(batch_size), operation_count), dtype=bool)
    for sample_index in range(int(batch_size)):
        kind = sample_index % 4
        if kind == 0:
            masks[sample_index] = index % 2 == 0
        elif kind == 1:
            masks[sample_index] = index % 2 == 1
        elif kind == 2:
            masks[sample_index] = np.isin(index % 4, (0, 3))
        else:
            masks[sample_index] = rng.random(operation_count) < 0.5
        if not np.any(masks[sample_index]) or np.all(masks[sample_index]):
            masks[sample_index, ::2] = True
            masks[sample_index, 1::2] = False
    return masks


def generate_noise_taxonomy_batch(
    geometry: OperationFrequencyGeometryV1,
    support: V7SyntheticPhysicsSupport,
    *,
    batch_size: int,
    seed: int,
    low_count_fraction: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Generate one labeled multi-mechanism, low-count synthetic batch."""

    if support.geometry_hash != geometry.geometry_hash:
        raise ValueError("synthetic support and geometry disagree")
    if not 0.0 <= float(low_count_fraction) <= 1.0:
        raise ValueError("low_count_fraction must lie in [0, 1]")
    rng = np.random.default_rng(int(seed))
    batch = int(batch_size)
    operation_count = geometry.operation_count
    target_context, target_overlap, program_overlap, targets = _target_batch(
        geometry, batch, rng,
    )
    static = support.static_features
    x = static[:, 0]
    y = static[:, 1]
    latent = rng.normal(0.0, 0.28, size=(batch, support.graph_modes.shape[0]))
    smooth_response = latent @ support.graph_modes
    signal_logits = (
        0.35 * x[None, :] - 0.18 * np.square(y)[None, :]
        + 0.35 * target_overlap + 0.35 * smooth_response
    )
    idler_logits = (
        -0.22 * y[None, :] + 0.14 * x[None, :] * y[None, :]
        + 0.28 * target_overlap - 0.28 * smooth_response
    )
    dominant_label = rng.integers(0, len(NOISE_TYPE_NAMES), size=batch)
    noise_strength = np.zeros((batch, len(NOISE_TYPE_NAMES)), dtype=np.float64)
    noise_strength[np.arange(batch), dominant_label] = rng.uniform(0.35, 1.0, size=batch)

    gain_strength = noise_strength[:, NOISE_TYPE_NAMES.index("smooth_gain_drift")]
    gain_mode = rng.normal(0.0, 1.0, size=(batch, support.graph_modes.shape[0])) @ support.graph_modes
    gain_mode /= np.maximum(np.std(gain_mode, axis=1, keepdims=True), 1e-6)
    signal_logits += 0.45 * gain_strength[:, None] * gain_mode
    idler_logits -= 0.32 * gain_strength[:, None] * gain_mode
    signal_probability = _softmax(signal_logits)
    idler_probability = _softmax(idler_logits)

    low_count = rng.random(batch) < float(low_count_fraction)
    signal_total = np.exp(rng.uniform(np.log(2500.0), np.log(30000.0), size=batch))
    idler_total = np.exp(rng.uniform(np.log(2000.0), np.log(26000.0), size=batch))
    coin_total = np.exp(rng.uniform(np.log(40.0), np.log(1200.0), size=batch))
    coin_total[~low_count] = np.exp(
        rng.uniform(np.log(1200.0), np.log(30000.0), size=int(np.sum(~low_count))),
    )
    signal_counts = rng.poisson(signal_total[:, None] * signal_probability).astype(np.float64)
    idler_counts = rng.poisson(idler_total[:, None] * idler_probability).astype(np.float64)
    accidental_seed = (signal_counts + 0.5) * (idler_counts + 0.5)
    accidental_probability = accidental_seed / np.maximum(
        np.sum(accidental_seed, axis=1, keepdims=True), 1e-12,
    )

    coherent_logits = (
        0.45 * target_overlap + 0.90 * program_overlap + 0.55 * smooth_response
        + 0.20 * x[None, :] * program_overlap - 0.12 * y[None, :] * program_overlap
    )
    coherent_logits += 0.55 * gain_strength[:, None] * gain_mode
    coherent_probability = _softmax(coherent_logits)
    memory_strength = noise_strength[:, NOISE_TYPE_NAMES.index("cycle_memory_crosstalk")]
    if geometry.system_type == "4qubit":
        previous = np.roll(coherent_probability, geometry.operation_count // 4, axis=1)
    else:
        previous = 0.5 * (
            np.roll(coherent_probability, 1, axis=1)
            + np.roll(coherent_probability, -1, axis=1)
        )
    coherent_probability = (
        (1.0 - 0.35 * memory_strength[:, None]) * coherent_probability
        + 0.35 * memory_strength[:, None] * previous
    )
    coherent_probability /= np.sum(coherent_probability, axis=1, keepdims=True)

    accidental_strength = noise_strength[:, NOISE_TYPE_NAMES.index("accidental_dominant")]
    coherent_fraction = rng.uniform(0.25, 0.65, size=batch)
    coherent_fraction *= 1.0 - 0.70 * accidental_strength
    coherent_fraction = np.clip(coherent_fraction, 0.08, 0.75)
    uniform_strength = noise_strength[:, NOISE_TYPE_NAMES.index("uniform_background")]
    uniform_fraction = 0.32 * uniform_strength
    probability = (
        (1.0 - uniform_fraction[:, None])
        * (
            coherent_fraction[:, None] * coherent_probability
            + (1.0 - coherent_fraction[:, None]) * accidental_probability
        )
        + uniform_fraction[:, None] / float(operation_count)
    )
    probability /= np.sum(probability, axis=1, keepdims=True)
    expected_counts = coin_total[:, None] * probability
    dispersion_strength = noise_strength[:, NOISE_TYPE_NAMES.index("nb2_overdispersion")]
    dispersion = 0.30 * np.square(dispersion_strength)
    coin_counts = np.stack([
        sample_nb2(expected_counts[index], dispersion[index], rng)
        for index in range(batch)
    ]).astype(np.float64)
    burst_strength = noise_strength[:, NOISE_TYPE_NAMES.index("sparse_burst")]
    for index in np.flatnonzero(burst_strength > 0.0):
        selected = rng.random(operation_count) < (0.01 + 0.06 * burst_strength[index])
        coin_counts[index, selected] += rng.gamma(
            shape=2.0,
            scale=max(1.0, burst_strength[index] * coin_total[index] / operation_count),
            size=int(np.sum(selected)),
        )
    coin_counts = np.rint(np.maximum(coin_counts, 0.0))

    observed_mask = _observation_mask(batch, operation_count, rng)
    observed_coin = coin_counts * observed_mask
    operation_inputs = build_operation_input_features(
        support,
        observed_coin_counts=observed_coin,
        signal_counts=signal_counts,
        idler_counts=idler_counts,
        observed_mask=observed_mask,
        target_overlap=target_overlap,
        program_overlap=program_overlap,
    )
    retrieval_ceiling = np.asarray([
        retrieval_ceiling_features(
            targets[index], support.retrieval_rate_vectors, geometry,
        )[0][0]
        for index in range(batch)
    ], dtype=np.float64)
    physics_ceiling = coherent_fraction * retrieval_ceiling + (
        1.0 - coherent_fraction
    ) / float(geometry.qubit_count)
    retention = np.clip(
        0.88
        - 0.16 * dispersion_strength
        - 0.22 * burst_strength
        - 0.12 * memory_strength
        - 0.18 * uniform_strength
        - 0.10 * gain_strength
        + rng.normal(0.0, 0.03, size=batch),
        0.20,
        0.95,
    )
    best_fidelity = physics_ceiling * retention
    mixture_weights = np.stack((
        (1.0 - uniform_fraction) * coherent_fraction,
        (1.0 - uniform_fraction) * (1.0 - coherent_fraction),
        uniform_fraction,
    ), axis=1)
    return {
        "operation_inputs": operation_inputs.astype(np.float32),
        "target_context": target_context.astype(np.float32),
        "signal_counts": signal_counts.astype(np.float32),
        "idler_counts": idler_counts.astype(np.float32),
        "coin_counts": coin_counts.astype(np.float32),
        "observed_mask": observed_mask.astype(np.float32),
        "expected_counts": expected_counts.astype(np.float32),
        "true_probability": probability.astype(np.float32),
        "noise_type_index": dominant_label.astype(np.int64),
        "noise_coarse_type_index": NOISE_FINE_TO_COARSE[dominant_label],
        "noise_strength": noise_strength.astype(np.float32),
        "mixture_weights": mixture_weights.astype(np.float32),
        "nb2_dispersion": dispersion.astype(np.float32),
        "best_fidelity": best_fidelity.astype(np.float32),
        "physics_ceiling": physics_ceiling.astype(np.float32),
        "retrieval_ceiling": retrieval_ceiling.astype(np.float32),
        "low_count_mask": low_count.astype(bool),
    }


def numpy_batch_to_torch(batch: Mapping[str, np.ndarray], device: Any) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("noise-taxonomy tensor conversion requires torch") from exc
    return {
        name: torch.as_tensor(value, device=device)
        for name, value in batch.items()
    }


__all__ = [
    "NOISE_COARSE_TYPE_NAMES",
    "NOISE_FINE_TO_COARSE",
    "NOISE_TYPE_NAMES",
    "OPERATION_INPUT_FEATURE_NAMES",
    "TARGET_CONTEXT_FEATURE_NAMES",
    "V7SyntheticPhysicsSupport",
    "build_synthetic_physics_support",
    "build_operation_input_features",
    "encode_target_program_context",
    "generate_noise_taxonomy_batch",
    "numpy_batch_to_torch",
]
