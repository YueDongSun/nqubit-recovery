"""Physics-aware tomographic features for density-matrix pairs."""

from __future__ import annotations

from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PAULI_AXES = ("x", "y", "z")
RHO_PAIR_TENSOR_CHANNELS = [
    "target_real",
    "target_imag",
    "target_abs",
    "target_phase",
    "measured_real",
    "measured_imag",
    "measured_abs",
    "measured_phase",
    "coherence_overlap",
    "coherence_log_magnitude_ratio",
    "phase_residual_sin_weighted",
    "phase_residual_cos_loss_weighted",
]
RHO_PAIR_DISTANCE_FEATURE_NAMES = [
    "state_fidelity",
    "state_infidelity",
    "bures_distance",
    "purified_distance",
    "trace_distance",
    "hilbert_schmidt_distance",
    "eigenvalue_l1",
    "eigenvalue_l2",
    "target_purity",
    "measured_purity",
    "purity_delta",
    "target_entropy",
    "measured_entropy",
    "entropy_delta",
    "quantum_jsd",
    "commutator_norm",
    "population_l1",
    "population_l2",
    "coherence_magnitude_l1",
    "coherence_phase_rms",
    "hilbert_schmidt_overlap",
]
PHYSICS_FEATURE_SCHEMAS = {
    "physics_v1",
    "entanglement_signature_v1",
    "physics_v2",
}
DEFAULT_PHYSICS_FEATURE_SCHEMA = "physics_v2"
PHYSICS_FEATURE_SCHEMA_VERSION = {
    "physics_v1": 1,
    "entanglement_signature_v1": 1,
    "physics_v2": 2,
}

_EPS = 1e-12
_PAULI = {
    "x": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_IDENTITY_2 = np.eye(2, dtype=complex)
_OPERATOR_CACHE: Dict[int, Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, str, int, int], np.ndarray]]] = {}
_PAULI_STRING_CACHE: Dict[Tuple[int, Tuple[Tuple[int, str], ...]], np.ndarray] = {}


def infer_num_qubits_from_dim(dim: int) -> Optional[int]:
    """Infer qubit count from a Hilbert-space dimension when possible."""

    dim = int(dim)
    if dim <= 0:
        return None
    num_qubits = int(round(np.log2(dim)))
    if 2 ** num_qubits != dim:
        return None
    return num_qubits


def validate_density_pair(rho_target: np.ndarray, rho_measured: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return complex arrays after validating that a rho pair has one shape."""

    target = np.asarray(rho_target, dtype=complex)
    measured = np.asarray(rho_measured, dtype=complex)
    if target.shape != measured.shape:
        raise ValueError("rho_target and rho_measured must have the same shape")
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("rho matrices must be square")
    return target, measured


def hermitian_part(rho: np.ndarray) -> np.ndarray:
    mat = np.asarray(rho, dtype=complex)
    return 0.5 * (mat + mat.conj().T)


def physicalize_density(rho: np.ndarray) -> np.ndarray:
    """Project a reconstructed rho onto the nearest simple PSD, trace-one form."""

    herm = hermitian_part(rho)
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    total = float(np.sum(eigvals))
    if total <= _EPS:
        return np.eye(herm.shape[0], dtype=complex) / max(herm.shape[0], 1)
    out = eigvecs @ np.diag(eigvals / total) @ eigvecs.conj().T
    return hermitian_part(out)


def density_fidelity(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Uhlmann fidelity for two mixed density matrices."""

    a = physicalize_density(rho_a)
    b = physicalize_density(rho_b)
    sqrt_a = _sqrt_psd(a)
    inner = hermitian_part(sqrt_a @ b @ sqrt_a)
    eigvals = np.linalg.eigvalsh(inner)
    root_trace = float(np.sum(np.sqrt(np.clip(np.real(eigvals), 0.0, None))))
    return float(np.clip(root_trace * root_trace, 0.0, 1.0))


def rho_pair_distance_features(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Return scalar quantum distances and tomographic mismatch summaries."""

    target_raw, measured_raw = validate_density_pair(rho_target, rho_measured)
    target = physicalize_density(target_raw)
    measured = physicalize_density(measured_raw)
    delta = hermitian_part(measured - target)
    dim = target.shape[0]

    fidelity = density_fidelity(target, measured)
    trace_distance = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))
    hs_distance = float(np.linalg.norm(delta, ord="fro"))
    eig_target = np.sort(np.linalg.eigvalsh(target))
    eig_measured = np.sort(np.linalg.eigvalsh(measured))
    eig_delta = eig_measured - eig_target
    purity_target = float(np.real(np.trace(target @ target)))
    purity_measured = float(np.real(np.trace(measured @ measured)))
    entropy_target = _normalized_entropy(target)
    entropy_measured = _normalized_entropy(measured)
    qjsd = _quantum_jensen_shannon_divergence(target, measured)
    commutator = target @ measured - measured @ target

    pop_delta = np.real(np.diag(measured) - np.diag(target))
    target_off, measured_off = _offdiag(target), _offdiag(measured)
    coherence_magnitude_l1 = float(np.mean(np.abs(np.abs(measured_off) - np.abs(target_off)))) if dim > 1 else 0.0
    phase_rms = _weighted_phase_rms(target_off, measured_off)
    hs_overlap = float(np.real(np.trace(target @ measured)))
    bures_distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(max(fidelity, 0.0)))))

    return np.asarray([
        fidelity,
        1.0 - fidelity,
        bures_distance,
        float(np.sqrt(max(0.0, 1.0 - fidelity))),
        trace_distance,
        hs_distance,
        float(np.sum(np.abs(eig_delta))),
        float(np.linalg.norm(eig_delta)),
        purity_target,
        purity_measured,
        purity_measured - purity_target,
        entropy_target,
        entropy_measured,
        entropy_measured - entropy_target,
        qjsd,
        float(np.linalg.norm(commutator, ord="fro")),
        0.5 * float(np.sum(np.abs(pop_delta))),
        float(np.linalg.norm(pop_delta)),
        coherence_magnitude_l1,
        phase_rms,
        hs_overlap,
    ], dtype=float)


def rho_pair_tomography_tensor(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Build a 12-channel tensor without a raw ``rho_target - rho_measured`` block."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    abs_target = np.abs(target)
    abs_measured = np.abs(measured)
    overlap = np.sqrt(abs_target * abs_measured)
    weight = overlap / max(float(np.max(overlap)), _EPS)
    phase_delta = np.angle(measured * np.conjugate(target))
    magnitude_ratio = np.tanh(np.log((abs_measured + _EPS) / (abs_target + _EPS)))

    channels = [
        np.real(target),
        np.imag(target),
        abs_target,
        np.angle(target),
        np.real(measured),
        np.imag(measured),
        abs_measured,
        np.angle(measured),
        overlap,
        magnitude_ratio,
        np.sin(phase_delta) * weight,
        (1.0 - np.cos(phase_delta)) * weight,
    ]
    return np.asarray(channels, dtype=np.float32)


def rho_pair_legacy_diff_tensor(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Build the historical target/measured/raw-diff tensor for old checkpoints."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    diff = target - measured
    channels = []
    for mat in (target, measured, diff):
        channels.extend([
            np.real(mat),
            np.imag(mat),
            np.abs(mat),
            np.angle(mat),
        ])
    return np.asarray(channels, dtype=np.float32)


def pauli_expectation_vector(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Return one- and two-body Pauli tomography expectations."""

    state = physicalize_density(rho)
    singles, pairs = _pauli_operators(num_qubits)
    values: List[float] = []
    for qubit in range(int(num_qubits)):
        for axis in PAULI_AXES:
            values.append(float(np.real(np.trace(state @ singles[(axis, qubit)]))))
    for i in range(int(num_qubits)):
        for j in range(i + 1, int(num_qubits)):
            for axis_a in PAULI_AXES:
                for axis_b in PAULI_AXES:
                    values.append(float(np.real(np.trace(state @ pairs[(axis_a, axis_b, i, j)]))))
    return np.asarray(values, dtype=np.float64)


def pauli_covariance_matrix(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Return covariance matrix of Pauli measurement outcomes."""

    state = physicalize_density(rho)
    singles, pairs = _pauli_operators(num_qubits)
    dim = 3 * int(num_qubits)
    cov = np.zeros((dim, dim), dtype=np.float64)
    single_values = {
        key: float(np.real(np.trace(state @ op)))
        for key, op in singles.items()
    }

    for i in range(int(num_qubits)):
        for a_idx, axis_a in enumerate(PAULI_AXES):
            row = 3 * i + a_idx
            mean_a = single_values[(axis_a, i)]
            cov[row, row] = max(0.0, 1.0 - mean_a * mean_a)
            for j in range(i + 1, int(num_qubits)):
                for b_idx, axis_b in enumerate(PAULI_AXES):
                    col = 3 * j + b_idx
                    mean_b = single_values[(axis_b, j)]
                    pair_value = float(np.real(np.trace(state @ pairs[(axis_a, axis_b, i, j)])))
                    value = pair_value - mean_a * mean_b
                    cov[row, col] = value
                    cov[col, row] = value
    return cov


def tomographic_residual_summary(rho_target: np.ndarray, rho_measured: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compact summaries of Pauli expectation and covariance residuals."""

    exp_target = pauli_expectation_vector(rho_target, num_qubits)
    exp_measured = pauli_expectation_vector(rho_measured, num_qubits)
    exp_residual = exp_measured - exp_target
    cov_residual = pauli_covariance_matrix(rho_measured, num_qubits) - pauli_covariance_matrix(rho_target, num_qubits)
    return np.asarray([
        _rms(exp_residual),
        float(np.mean(np.abs(exp_residual))) if exp_residual.size else 0.0,
        float(np.max(np.abs(exp_residual))) if exp_residual.size else 0.0,
        _rms(cov_residual),
        float(np.mean(np.abs(cov_residual))) if cov_residual.size else 0.0,
        float(np.max(np.abs(cov_residual))) if cov_residual.size else 0.0,
    ], dtype=np.float64)


def single_excitation_indices(num_qubits: int) -> List[int]:
    return [2 ** (int(num_qubits) - 1 - i) for i in range(int(num_qubits))]


def single_excitation_subspace_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Return tomography features restricted to the single-excitation subspace."""

    single = single_excitation_indices(num_qubits)
    sub = np.asarray(rho, dtype=complex)[np.ix_(single, single)]
    herm = hermitian_part(sub)
    pieces = [
        np.real(sub).reshape(-1),
        np.imag(sub).reshape(-1),
        np.abs(sub).reshape(-1),
        np.angle(sub + _EPS).reshape(-1),
        np.linalg.eigvalsh(herm),
    ]
    return np.concatenate(pieces).astype(np.float64)


def single_excitation_pair_features(rho_target: np.ndarray, rho_measured: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compare population, coherence magnitude, and phase inside the W subspace."""

    single = single_excitation_indices(num_qubits)
    target, measured = validate_density_pair(rho_target, rho_measured)
    target_sub = target[np.ix_(single, single)]
    measured_sub = measured[np.ix_(single, single)]
    abs_target = np.abs(target_sub)
    abs_measured = np.abs(measured_sub)
    overlap = np.sqrt(abs_target * abs_measured)
    weight = overlap / max(float(np.max(overlap)), _EPS)
    phase_delta = np.angle(measured_sub * np.conjugate(target_sub))
    magnitude_ratio = np.tanh(np.log((abs_measured + _EPS) / (abs_target + _EPS)))
    target_eig = np.linalg.eigvalsh(hermitian_part(target_sub))
    measured_eig = np.linalg.eigvalsh(hermitian_part(measured_sub))
    pop_delta = np.real(np.diag(measured_sub) - np.diag(target_sub))

    pieces = [
        np.real(np.diag(target_sub)),
        np.real(np.diag(measured_sub)),
        pop_delta,
        abs_target.reshape(-1),
        abs_measured.reshape(-1),
        magnitude_ratio.reshape(-1),
        (np.sin(phase_delta) * weight).reshape(-1),
        ((1.0 - np.cos(phase_delta)) * weight).reshape(-1),
        overlap.reshape(-1),
        target_eig,
        measured_eig,
        measured_eig - target_eig,
        np.asarray([
            0.5 * float(np.sum(np.abs(pop_delta))),
            float(np.linalg.norm(pop_delta)),
            float(np.mean(np.abs(abs_measured - abs_target))) if abs_target.size else 0.0,
            _weighted_phase_rms(target_sub, measured_sub),
            float(np.real(np.trace(target_sub.conj().T @ measured_sub))),
        ], dtype=np.float64),
    ]
    return np.concatenate(pieces).astype(np.float64)


def normalise_physics_feature_schema(schema: str | None) -> str:
    """Return a supported rho-pair physics feature schema name."""

    value = str(schema or DEFAULT_PHYSICS_FEATURE_SCHEMA).strip().lower()
    if value in {"default", "current"}:
        return DEFAULT_PHYSICS_FEATURE_SCHEMA
    if value not in PHYSICS_FEATURE_SCHEMAS:
        raise ValueError(
            "physics feature schema must be one of "
            + ", ".join(sorted(PHYSICS_FEATURE_SCHEMAS))
        )
    return value


def rho_pair_physics_v1_features(rho_target: np.ndarray, rho_measured: np.ndarray, num_qubits: int) -> np.ndarray:
    """Original physics scalar feature set kept stable for old checkpoints."""

    validate_density_pair(rho_target, rho_measured)
    pieces = [
        rho_pair_distance_features(rho_target, rho_measured),
        single_excitation_pair_features(rho_target, rho_measured, int(num_qubits)),
        tomographic_residual_summary(rho_target, rho_measured, int(num_qubits)),
    ]
    return np.concatenate([np.asarray(piece, dtype=np.float32).reshape(-1) for piece in pieces]).astype(np.float32)


def rho_pair_physics_features_by_schema(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int,
    schema: str | None = DEFAULT_PHYSICS_FEATURE_SCHEMA,
    history: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Build schema-versioned physics features for trainable rho-pair models."""

    selected = normalise_physics_feature_schema(schema)
    if selected == "physics_v1":
        return rho_pair_physics_v1_features(rho_target, rho_measured, int(num_qubits))
    if selected == "entanglement_signature_v1":
        return entanglement_signature_features(
            rho_target,
            rho_measured,
            int(num_qubits),
            include_geometric=True,
            history=history,
        ).astype(np.float32)
    return np.concatenate([
        rho_pair_physics_v1_features(rho_target, rho_measured, int(num_qubits)),
        entanglement_signature_features(
            rho_target,
            rho_measured,
            int(num_qubits),
            include_geometric=True,
            history=history,
        ),
    ]).astype(np.float32)


def rho_pair_physics_feature_names_by_schema(
    num_qubits: int,
    schema: str | None = DEFAULT_PHYSICS_FEATURE_SCHEMA,
) -> List[str]:
    """Return stable feature names for a schema and qubit count."""

    selected = normalise_physics_feature_schema(schema)
    if selected == "physics_v1":
        return _physics_v1_feature_names(int(num_qubits))
    if selected == "entanglement_signature_v1":
        return entanglement_signature_feature_names(int(num_qubits), include_geometric=True)
    return (
        _physics_v1_feature_names(int(num_qubits))
        + entanglement_signature_feature_names(int(num_qubits), include_geometric=True)
    )


def entanglement_signature_features(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int,
    include_geometric: bool = False,
    history: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Feature-fusion signature for entanglement, separability, and phase diagnostics."""

    validate_density_pair(rho_target, rho_measured)
    target_features = _entanglement_state_features(rho_target, int(num_qubits))
    measured_features = _entanglement_state_features(rho_measured, int(num_qubits))
    pieces = [
        target_features,
        measured_features,
        measured_features - target_features,
        np.abs(measured_features - target_features),
    ]
    if include_geometric:
        pieces.append(geometric_phase_features(rho_target, rho_measured, int(num_qubits), history=history))
    return np.concatenate([np.asarray(piece, dtype=np.float64).reshape(-1) for piece in pieces]).astype(np.float64)


def entanglement_signature_feature_names(num_qubits: int, include_geometric: bool = False) -> List[str]:
    state_names = _entanglement_state_feature_names(int(num_qubits))
    names = (
        [f"target_{name}" for name in state_names]
        + [f"measured_{name}" for name in state_names]
        + [f"delta_{name}" for name in state_names]
        + [f"abs_delta_{name}" for name in state_names]
    )
    if include_geometric:
        names.extend(geometric_phase_feature_names(int(num_qubits)))
    return names


def ppt_negativity_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Return PPT negativity diagnostics for every unique bipartition."""

    state = physicalize_density(rho)
    features: List[float] = []
    for mask in _unique_bipartition_masks(int(num_qubits)):
        pt = _partial_transpose(state, int(num_qubits), mask)
        eigvals = np.real(np.linalg.eigvalsh(hermitian_part(pt)))
        trace_norm = float(np.sum(np.abs(eigvals)))
        negativity = 0.5 * max(0.0, trace_norm - 1.0)
        features.extend([
            negativity,
            float(np.min(eigvals)) if eigvals.size else 0.0,
            max(0.0, trace_norm - 1.0),
            float(np.log2(max(trace_norm, _EPS))),
        ])
    return np.asarray(features, dtype=np.float64)


def ppt_negativity_feature_names(num_qubits: int) -> List[str]:
    names: List[str] = []
    for mask in _unique_bipartition_masks(int(num_qubits)):
        label = _mask_label(mask, int(num_qubits))
        names.extend([
            f"ppt_{label}_negativity",
            f"ppt_{label}_min_eigenvalue",
            f"ppt_{label}_trace_norm_excess",
            f"ppt_{label}_log_negativity",
        ])
    return names


def bloch_correlation_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Summarize Pauli/Bloch correlation tensors by qubit subset and order."""

    state = physicalize_density(rho)
    features: List[float] = []
    for order in range(1, int(num_qubits) + 1):
        for subset in combinations(range(int(num_qubits)), order):
            tensor_values = _pauli_subset_expectations(state, int(num_qubits), subset)
            norm = float(np.linalg.norm(tensor_values))
            features.extend([
                norm / np.sqrt(max(tensor_values.size, 1)),
                norm,
                float(np.mean(np.abs(tensor_values))) if tensor_values.size else 0.0,
                float(np.max(np.abs(tensor_values))) if tensor_values.size else 0.0,
            ])
            if order >= 2:
                matrix = tensor_values.reshape(3, -1)
                features.extend(_pad_values(np.linalg.svd(matrix, compute_uv=False), 3))
    return np.asarray(features, dtype=np.float64)


def bloch_correlation_feature_names(num_qubits: int) -> List[str]:
    names: List[str] = []
    for order in range(1, int(num_qubits) + 1):
        for subset in combinations(range(int(num_qubits)), order):
            label = _subset_label(subset)
            names.extend([
                f"bloch_o{order}_{label}_rms",
                f"bloch_o{order}_{label}_l2",
                f"bloch_o{order}_{label}_mean_abs",
                f"bloch_o{order}_{label}_max_abs",
            ])
            if order >= 2:
                names.extend([f"bloch_o{order}_{label}_sv{idx + 1}" for idx in range(3)])
    return names


def correlation_matrix_criterion_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Pairwise correlation-matrix criterion summaries."""

    state = physicalize_density(rho)
    singles, pairs = _pauli_operators(int(num_qubits))
    single_values = {
        key: float(np.real(np.trace(state @ op)))
        for key, op in singles.items()
    }
    features: List[float] = []
    for i, j in combinations(range(int(num_qubits)), 2):
        matrix = np.zeros((3, 3), dtype=np.float64)
        for a_idx, axis_a in enumerate(PAULI_AXES):
            for b_idx, axis_b in enumerate(PAULI_AXES):
                pair_value = float(np.real(np.trace(state @ pairs[(axis_a, axis_b, i, j)])))
                matrix[a_idx, b_idx] = pair_value - single_values[(axis_a, i)] * single_values[(axis_b, j)]
        singular = np.linalg.svd(matrix, compute_uv=False)
        trace_norm = float(np.sum(singular))
        features.extend([
            *list(_pad_values(singular, 3)),
            trace_norm,
            float(np.linalg.norm(matrix, ord="fro")),
            max(0.0, trace_norm - 1.0),
        ])
    return np.asarray(features, dtype=np.float64)


def correlation_matrix_criterion_feature_names(num_qubits: int) -> List[str]:
    names: List[str] = []
    for i, j in combinations(range(int(num_qubits)), 2):
        label = f"q{i + 1}q{j + 1}"
        names.extend([
            f"cm_{label}_sv1",
            f"cm_{label}_sv2",
            f"cm_{label}_sv3",
            f"cm_{label}_trace_norm",
            f"cm_{label}_fro_norm",
            f"cm_{label}_trace_norm_margin",
        ])
    return names


def realignment_moment_features(
    rho: np.ndarray,
    num_qubits: int,
    moments: Sequence[int] = (1, 2, 3, 4),
) -> np.ndarray:
    """Realignment singular moments and CCNR-style violation margins."""

    state = physicalize_density(rho)
    features: List[float] = []
    for mask in _unique_bipartition_masks(int(num_qubits)):
        realigned = _realigned_matrix(state, int(num_qubits), mask)
        singular = np.linalg.svd(realigned, compute_uv=False)
        moment_values = [float(np.sum(singular ** int(moment))) for moment in moments]
        trace_norm = moment_values[0] if moment_values else float(np.sum(singular))
        features.extend([
            *moment_values,
            max(0.0, trace_norm - 1.0),
            float(np.max(singular)) if singular.size else 0.0,
            float(np.linalg.norm(singular)),
        ])
    return np.asarray(features, dtype=np.float64)


def realignment_moment_feature_names(
    num_qubits: int,
    moments: Sequence[int] = (1, 2, 3, 4),
) -> List[str]:
    names: List[str] = []
    for mask in _unique_bipartition_masks(int(num_qubits)):
        label = _mask_label(mask, int(num_qubits))
        names.extend([f"realignment_{label}_moment_{int(moment)}" for moment in moments])
        names.extend([
            f"realignment_{label}_trace_norm_margin",
            f"realignment_{label}_max_singular",
            f"realignment_{label}_singular_l2",
        ])
    return names


def witness_margin_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """W/GHZ witness overlap margins plus single-excitation leakage/coherence."""

    state = physicalize_density(rho)
    w_vec = _w_state_vector(int(num_qubits))
    ghz_plus = _ghz_state_vector(int(num_qubits), phase=1.0)
    ghz_minus = _ghz_state_vector(int(num_qubits), phase=-1.0)
    w_overlap = _state_overlap(state, w_vec)
    ghz_plus_overlap = _state_overlap(state, ghz_plus)
    ghz_minus_overlap = _state_overlap(state, ghz_minus)
    single = single_excitation_indices(int(num_qubits))
    sub = state[np.ix_(single, single)]
    w_population = float(np.clip(np.real(np.trace(sub)), 0.0, 1.0))
    coherence = _single_excitation_coherence_score(sub)
    phase_order = _single_excitation_phase_order(sub)
    return np.asarray([
        w_overlap,
        w_overlap - 0.5,
        ghz_plus_overlap,
        ghz_plus_overlap - 0.5,
        ghz_minus_overlap,
        ghz_minus_overlap - 0.5,
        w_population,
        float(np.clip(1.0 - w_population, 0.0, 1.0)),
        coherence,
        phase_order,
    ], dtype=np.float64)


def witness_margin_feature_names(num_qubits: int) -> List[str]:
    return [
        "witness_w_overlap",
        "witness_w_margin_0p5",
        "witness_ghz_plus_overlap",
        "witness_ghz_plus_margin_0p5",
        "witness_ghz_minus_overlap",
        "witness_ghz_minus_margin_0p5",
        "witness_w_population",
        "witness_w_leakage",
        "witness_w_coherence",
        "witness_w_phase_order",
    ]


def w_witness_tight_margins(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """W-state fidelity witness with tight separability margins (V5 upgrade).

    Unlike :func:`witness_margin_features` (which uses the non-tight 0.5 cut),
    this returns the biseparable ``(N-1)/N`` and fully-separable
    ``((N-1)/N)^(N-1)`` bounds so positive margins certify genuine multipartite
    entanglement and entanglement respectively.  GHZ margins use the biseparable
    bound ``1/2`` (Guhne & Toth, Phys. Rep. 474, 2009).  Additive: the existing
    ``witness_margin_features`` schema is unchanged, so this is opt-in for new
    feature pipelines and audits.
    """

    state = physicalize_density(rho)
    n = int(num_qubits)
    w_overlap = _state_overlap(state, _w_state_vector(n))
    w_genuine_bound = (n - 1.0) / n
    w_separable_bound = w_genuine_bound ** (n - 1.0)
    ghz_plus_overlap = _state_overlap(state, _ghz_state_vector(n, phase=1.0))
    ghz_minus_overlap = _state_overlap(state, _ghz_state_vector(n, phase=-1.0))
    ghz_genuine_bound = 0.5
    return np.asarray([
        w_overlap,
        w_overlap - w_genuine_bound,
        w_overlap - w_separable_bound,
        w_genuine_bound,
        w_separable_bound,
        ghz_plus_overlap,
        ghz_plus_overlap - ghz_genuine_bound,
        ghz_minus_overlap,
        ghz_minus_overlap - ghz_genuine_bound,
    ], dtype=np.float64)


def w_witness_tight_margin_names(num_qubits: int) -> List[str]:
    return [
        "witness_tight_w_overlap",
        "witness_tight_w_genuine_margin",
        "witness_tight_w_separable_margin",
        "witness_tight_w_genuine_bound",
        "witness_tight_w_separable_bound",
        "witness_tight_ghz_plus_overlap",
        "witness_tight_ghz_plus_genuine_margin",
        "witness_tight_ghz_minus_overlap",
        "witness_tight_ghz_minus_genuine_margin",
    ]


def geometric_phase_features(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int,
    history: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Experimental Pancharatnam/Bargmann phase features in the W subspace."""

    validate_density_pair(rho_target, rho_measured)
    target_vec, target_weight, target_coherence = _dominant_w_subspace_vector(rho_target, int(num_qubits))
    measured_vec, measured_weight, measured_coherence = _dominant_w_subspace_vector(rho_measured, int(num_qubits))
    reference = np.ones(int(num_qubits), dtype=complex) / np.sqrt(max(int(num_qubits), 1))
    pancharatnam = np.vdot(target_vec, measured_vec)
    pan_phase = _wrapped_angle(float(np.angle(pancharatnam)))
    visibility = float(np.clip(np.abs(pancharatnam), 0.0, 1.0))
    bargmann = np.vdot(reference, target_vec) * np.vdot(target_vec, measured_vec) * np.vdot(measured_vec, reference)
    bargmann_phase = _wrapped_angle(float(np.angle(bargmann)))
    path_tv, path_max_step, history_count = _w_subspace_phase_path_stats(
        target_vec,
        measured_vec,
        int(num_qubits),
        history,
    )
    coherence_gate = float(np.clip(min(target_coherence, measured_coherence) * min(target_weight, measured_weight), 0.0, 1.0))
    return np.asarray([
        pan_phase / np.pi,
        np.sin(pan_phase),
        1.0 - np.cos(pan_phase),
        visibility,
        bargmann_phase / np.pi,
        np.sin(bargmann_phase),
        1.0 - np.cos(bargmann_phase),
        target_weight,
        measured_weight,
        target_coherence,
        measured_coherence,
        coherence_gate,
        path_tv / np.pi,
        path_max_step / np.pi,
        float(np.clip(history_count / 16.0, 0.0, 1.0)),
    ], dtype=np.float64)


def geometric_phase_feature_names(num_qubits: int) -> List[str]:
    return [
        "geometric_pancharatnam_phase_pi",
        "geometric_pancharatnam_sin",
        "geometric_pancharatnam_cos_loss",
        "geometric_pancharatnam_visibility",
        "geometric_bargmann_phase_pi",
        "geometric_bargmann_sin",
        "geometric_bargmann_cos_loss",
        "geometric_target_w_weight",
        "geometric_measured_w_weight",
        "geometric_target_w_coherence",
        "geometric_measured_w_coherence",
        "geometric_coherence_gate",
        "geometric_path_total_variation_pi",
        "geometric_path_max_step_pi",
        "geometric_history_fraction",
    ]


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    herm = hermitian_part(matrix)
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def _normalized_entropy(rho: np.ndarray) -> float:
    eigvals = np.clip(np.real(np.linalg.eigvalsh(physicalize_density(rho))), 0.0, 1.0)
    eigvals = eigvals[eigvals > _EPS]
    if eigvals.size == 0:
        return 0.0
    entropy = -float(np.sum(eigvals * np.log(eigvals)))
    return entropy / max(float(np.log(rho.shape[0])), _EPS)


def _quantum_jensen_shannon_divergence(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    midpoint = physicalize_density(0.5 * (rho_a + rho_b))
    entropy_mid = _normalized_entropy(midpoint)
    entropy_a = _normalized_entropy(rho_a)
    entropy_b = _normalized_entropy(rho_b)
    return float(np.clip(entropy_mid - 0.5 * (entropy_a + entropy_b), 0.0, 1.0))


def _pauli_operators(num_qubits: int):
    num_qubits = int(num_qubits)
    if num_qubits in _OPERATOR_CACHE:
        return _OPERATOR_CACHE[num_qubits]
    singles = {
        (axis, i): _kron_operator(num_qubits, {i: _PAULI[axis]})
        for axis in PAULI_AXES
        for i in range(num_qubits)
    }
    pairs = {
        (axis_a, axis_b, i, j): _kron_operator(num_qubits, {i: _PAULI[axis_a], j: _PAULI[axis_b]})
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
        for axis_a in PAULI_AXES
        for axis_b in PAULI_AXES
    }
    _OPERATOR_CACHE[num_qubits] = (singles, pairs)
    return singles, pairs


def _kron_operator(num_qubits: int, terms: Dict[int, np.ndarray]) -> np.ndarray:
    matrices: Sequence[np.ndarray] = [terms.get(i, _IDENTITY_2) for i in range(int(num_qubits))]
    out = matrices[0]
    for matrix in matrices[1:]:
        out = np.kron(out, matrix)
    return out


def _offdiag(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=complex)
    return mat - np.diag(np.diag(mat))


def _weighted_phase_rms(target: np.ndarray, measured: np.ndarray) -> float:
    target_abs = np.abs(target)
    measured_abs = np.abs(measured)
    weights = np.sqrt(target_abs * measured_abs)
    phase_delta = np.angle(measured * np.conjugate(target))
    total = float(np.sum(weights))
    if total <= _EPS:
        return 0.0
    return float(np.sqrt(np.sum(weights * phase_delta * phase_delta) / total) / np.pi)


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _physics_v1_feature_names(num_qubits: int) -> List[str]:
    probe = np.outer(_w_state_vector(int(num_qubits)), _w_state_vector(int(num_qubits)).conj())
    feature_count = int(rho_pair_physics_v1_features(probe, probe, int(num_qubits)).size)
    residual_names = [
        "pauli_expectation_residual_rms",
        "pauli_expectation_residual_mean_abs",
        "pauli_expectation_residual_max_abs",
        "pauli_covariance_residual_rms",
        "pauli_covariance_residual_mean_abs",
        "pauli_covariance_residual_max_abs",
    ]
    known = list(RHO_PAIR_DISTANCE_FEATURE_NAMES)
    remaining = feature_count - len(known) - len(residual_names)
    if remaining < 0:
        return [f"physics_feature_{idx:03d}" for idx in range(feature_count)]
    return (
        known
        + [f"single_excitation_pair_feature_{idx:03d}" for idx in range(remaining)]
        + residual_names
    )


def _entanglement_state_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    pieces = [
        ppt_negativity_features(rho, int(num_qubits)),
        bloch_correlation_features(rho, int(num_qubits)),
        correlation_matrix_criterion_features(rho, int(num_qubits)),
        realignment_moment_features(rho, int(num_qubits)),
        witness_margin_features(rho, int(num_qubits)),
    ]
    return np.concatenate([np.asarray(piece, dtype=np.float64).reshape(-1) for piece in pieces])


def _entanglement_state_feature_names(num_qubits: int) -> List[str]:
    return (
        ppt_negativity_feature_names(int(num_qubits))
        + bloch_correlation_feature_names(int(num_qubits))
        + correlation_matrix_criterion_feature_names(int(num_qubits))
        + realignment_moment_feature_names(int(num_qubits))
        + witness_margin_feature_names(int(num_qubits))
    )


def _unique_bipartition_masks(num_qubits: int) -> List[int]:
    full = (1 << int(num_qubits)) - 1
    return [mask for mask in range(1, full) if mask < (full ^ mask)]


def _mask_label(mask: int, num_qubits: int) -> str:
    bits = ["1" if int(mask) & (1 << q) else "0" for q in range(int(num_qubits))]
    return "q" + "".join(bits)


def _subset_label(subset: Sequence[int]) -> str:
    return "q" + "".join(str(int(index) + 1) for index in subset)


def _partial_transpose(rho: np.ndarray, num_qubits: int, mask: int) -> np.ndarray:
    dims = [2] * (2 * int(num_qubits))
    tensor = np.asarray(rho, dtype=complex).reshape(dims)
    axes = list(range(2 * int(num_qubits)))
    for q in range(int(num_qubits)):
        if int(mask) & (1 << q):
            axes[q], axes[int(num_qubits) + q] = axes[int(num_qubits) + q], axes[q]
    return np.transpose(tensor, axes).reshape(rho.shape)


def _realigned_matrix(rho: np.ndarray, num_qubits: int, mask: int) -> np.ndarray:
    a_qubits = [q for q in range(int(num_qubits)) if int(mask) & (1 << q)]
    b_qubits = [q for q in range(int(num_qubits)) if q not in a_qubits]
    tensor = np.asarray(rho, dtype=complex).reshape([2] * (2 * int(num_qubits)))
    axes = (
        a_qubits
        + [int(num_qubits) + q for q in a_qubits]
        + b_qubits
        + [int(num_qubits) + q for q in b_qubits]
    )
    d_a = 2 ** len(a_qubits)
    d_b = 2 ** len(b_qubits)
    return np.transpose(tensor, axes).reshape(d_a * d_a, d_b * d_b)


def _pauli_subset_expectations(rho: np.ndarray, num_qubits: int, subset: Sequence[int]) -> np.ndarray:
    values: List[float] = []
    for axes in product(PAULI_AXES, repeat=len(subset)):
        terms = tuple(sorted((int(qubit), str(axis)) for qubit, axis in zip(subset, axes)))
        operator = _pauli_string_operator(int(num_qubits), terms)
        values.append(float(np.real(np.trace(rho @ operator))))
    return np.asarray(values, dtype=np.float64)


def _pauli_string_operator(num_qubits: int, terms: Tuple[Tuple[int, str], ...]) -> np.ndarray:
    key = (int(num_qubits), tuple(terms))
    if key not in _PAULI_STRING_CACHE:
        _PAULI_STRING_CACHE[key] = _kron_operator(
            int(num_qubits),
            {int(qubit): _PAULI[str(axis)] for qubit, axis in terms},
        )
    return _PAULI_STRING_CACHE[key]


def _pad_values(values: Sequence[float], count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= int(count):
        return arr[:int(count)]
    return np.pad(arr, (0, int(count) - arr.size))


def _w_state_vector(num_qubits: int) -> np.ndarray:
    dim = 2 ** int(num_qubits)
    vec = np.zeros(dim, dtype=complex)
    for i in range(int(num_qubits)):
        vec[2 ** (int(num_qubits) - 1 - i)] = 1.0 / np.sqrt(max(int(num_qubits), 1))
    return vec


def _ghz_state_vector(num_qubits: int, phase: complex = 1.0) -> np.ndarray:
    dim = 2 ** int(num_qubits)
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0 / np.sqrt(2.0)
    vec[-1] = complex(phase) / np.sqrt(2.0)
    return vec


def _state_overlap(rho: np.ndarray, vector: np.ndarray) -> float:
    value = np.vdot(vector, np.asarray(rho, dtype=complex) @ vector)
    return float(np.clip(np.real(value), 0.0, 1.0))


def _single_excitation_coherence_score(subspace: np.ndarray) -> float:
    sub = np.asarray(subspace, dtype=complex)
    if sub.size == 0 or sub.shape[0] <= 1:
        return 0.0
    off_abs = np.abs(_offdiag(sub))
    mean_population = float(np.mean(np.clip(np.real(np.diag(sub)), 0.0, None)))
    denominator = sub.shape[0] * (sub.shape[0] - 1) * max(mean_population, _EPS)
    return float(np.clip(np.sum(off_abs) / denominator, 0.0, 1.0))


def _single_excitation_phase_order(subspace: np.ndarray) -> float:
    sub = np.asarray(subspace, dtype=complex)
    if sub.size == 0 or sub.shape[0] <= 1:
        return 0.0
    off = _offdiag(sub)
    weights = np.abs(off)
    total = float(np.sum(weights))
    if total <= _EPS:
        return 0.0
    return float(np.clip(np.abs(np.sum(weights * np.exp(1j * np.angle(off))) / total), 0.0, 1.0))


def _dominant_w_subspace_vector(rho: np.ndarray, num_qubits: int) -> Tuple[np.ndarray, float, float]:
    single = single_excitation_indices(int(num_qubits))
    arr = np.asarray(rho, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] <= max(single):
        vec = np.ones(int(num_qubits), dtype=complex) / np.sqrt(max(int(num_qubits), 1))
        return vec, 0.0, 0.0
    sub = hermitian_part(arr[np.ix_(single, single)])
    try:
        eigvals, eigvecs = np.linalg.eigh(sub)
        index = int(np.argmax(np.real(eigvals)))
        vec = _gauge_fix_vector(eigvecs[:, index])
        weight = float(np.clip(np.real(eigvals[index]), 0.0, 1.0))
    except np.linalg.LinAlgError:
        vec = np.ones(int(num_qubits), dtype=complex) / np.sqrt(max(int(num_qubits), 1))
        weight = 0.0
    return vec, weight, _single_excitation_coherence_score(sub)


def _gauge_fix_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        return np.ones(max(vec.size, 1), dtype=complex) / np.sqrt(max(vec.size, 1))
    vec = vec / norm
    pivot = int(np.argmax(np.abs(vec)))
    phase = np.angle(vec[pivot])
    return vec * np.exp(-1j * phase)


def _w_subspace_phase_path_stats(
    target_vec: np.ndarray,
    measured_vec: np.ndarray,
    num_qubits: int,
    history: Optional[Sequence[np.ndarray]],
) -> Tuple[float, float, int]:
    vectors = [np.asarray(target_vec, dtype=complex)]
    if history:
        for item in history:
            arr = np.asarray(item, dtype=complex)
            if arr.ndim == 2:
                vec, _, _ = _dominant_w_subspace_vector(arr, int(num_qubits))
                vectors.append(vec)
    vectors.append(np.asarray(measured_vec, dtype=complex))
    steps = [
        abs(_wrapped_angle(float(np.angle(np.vdot(left, right)))))
        for left, right in zip(vectors[:-1], vectors[1:])
    ]
    if not steps:
        return 0.0, 0.0, 0
    return float(np.sum(steps)), float(np.max(steps)), max(len(vectors) - 2, 0)


def _wrapped_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)
