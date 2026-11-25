"""Density-matrix reconstruction with retrieval-efficiency compensation.

The original ``density_reconstruction.py`` is intentionally kept as the raw
reconstruction path.  This module wraps the same least-squares solver but first
applies the atom-array retrieval-rate model used by the sequence generators:

* 3-qubit: 75 operation points = 15 rows x 5 groups, 3 atom cells per group.
* 4-qubit: 64 operation points per cycle = 16 rows x 4 groups, 4 atom cells
  per group.  The 4 basis cycles reuse the same 16x16 retrieval map.

The compensation only affects tomography.  It does not write sequence files or
modify any AOD RF amplitude matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from system_specs import SYSTEM_SPECS as _SPECS

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "matrix"
DEFAULT_IDLER_RATIO_CSV = SCRIPT_DIR.parents[1] / "histogram_2026-06-24_22-01-37.csv"
LEGACY_EFFICIENCY_MAPS = {
    "3qubit": MATRIX_DIR / "retrieval_efficiency_3qubit.txt",
    "4qubit": MATRIX_DIR / "retrieval_efficiency_4qubit.txt",
}
DEFAULT_EFFICIENCY_MAPS = {
    "3qubit": DEFAULT_IDLER_RATIO_CSV,
    "4qubit": DEFAULT_IDLER_RATIO_CSV,
}
EFFICIENCY_SHAPES = {
    "3qubit": _SPECS["3qubit"]["efficiency_shape"],
    "4qubit": _SPECS["4qubit"]["efficiency_shape"],
}
GROUP_SIZES = {
    "3qubit": _SPECS["3qubit"]["group_size"],
    "4qubit": _SPECS["4qubit"]["group_size"],
}
GROUPS_PER_ROW = {
    "3qubit": _SPECS["3qubit"]["groups_per_row"],
    "4qubit": _SPECS["4qubit"]["groups_per_row"],
}
ENTRIES_PER_CYCLE = {
    "3qubit": _SPECS["3qubit"]["entries_per_cycle"],
    "4qubit": _SPECS["4qubit"]["entries_per_cycle"],
}
MIN_RATE = 1e-12

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import density_reconstruction as raw_reconstruction


def default_efficiency_map_path(system_type: str) -> Path:
    """Return the default retrieval-efficiency matrix path for a system."""
    _validate_system(system_type)
    return DEFAULT_EFFICIENCY_MAPS[system_type]


def legacy_efficiency_map_path(system_type: str) -> Path:
    """Return the legacy retrieval-rate text matrix path for a system."""
    _validate_system(system_type)
    return LEGACY_EFFICIENCY_MAPS[system_type]


def load_efficiency_map(system_type: str, efficiency_map_path: Optional[str | Path] = None) -> np.ndarray:
    """Load the retrieval compensation map for tomography.

    ``.csv`` paths are parsed as measured 16x16 idler-count maps, normalized by
    the positive maximum, then trimmed to 15x15 for 3-qubit reconstruction.
    ``.txt`` paths keep the legacy retrieval-efficiency matrix behavior.
    """
    _validate_system(system_type)
    path = _resolve_efficiency_map_path(system_type, efficiency_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval-efficiency matrix not found: {path}")

    if path.suffix.lower() == ".csv":
        return load_idler_ratio_map_from_csv(path, system_type)

    matrix = np.loadtxt(path, dtype=float)
    expected_shape = EFFICIENCY_SHAPES[system_type]
    if matrix.shape != expected_shape:
        raise ValueError(f"{system_type} efficiency matrix must have shape {expected_shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Retrieval-efficiency matrix contains NaN or Inf")
    if np.any(matrix < 0):
        raise ValueError("Retrieval-efficiency matrix must not contain negative values")
    return matrix


def load_idler_ratio_map_from_csv(
    path: str | Path,
    system_type: str,
    row_label: str = "Idler Absolute",
) -> np.ndarray:
    """Load a normalized idler-count ratio map from a measurement CSV."""

    _validate_system(system_type)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Idler calibration CSV not found: {path}")

    target = row_label.strip().casefold()
    values = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().casefold() == target:
                values = [float(item) for item in row[1:] if str(item).strip()]
                break
    if values is None:
        raise ValueError(f"CSV row {row_label!r} not found in {path}")

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 256:
        raise ValueError(f"{row_label!r} row must contain 256 idler counts, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Idler calibration CSV contains NaN or Inf")
    if np.any(arr < 0):
        raise ValueError("Idler calibration CSV must not contain negative counts")

    positive = arr[arr > 0]
    if positive.size == 0:
        raise ValueError("Idler calibration CSV contains no positive counts")
    matrix_16 = (arr / float(np.max(positive))).reshape(16, 16)
    if system_type == "3qubit":
        return matrix_16[:15, :15].copy()
    return matrix_16


def operation_cells_for_entry(system_type: str, entry_index: int) -> tuple[int, list[int]]:
    """Map a basis entry index to the atom cells used by that operation point."""
    _validate_system(system_type)
    group_size = GROUP_SIZES[system_type]
    groups_per_row = GROUPS_PER_ROW[system_type]
    local_index = entry_index % ENTRIES_PER_CYCLE[system_type]
    row = local_index // groups_per_row
    group = local_index % groups_per_row
    col_start = group * group_size
    return row, list(range(col_start, col_start + group_size))


def rate_vectors_for_entries(system_type: str, efficiency_map: np.ndarray, entry_count: int) -> np.ndarray:
    """Return one retrieval-rate vector per basis entry."""
    _validate_system(system_type)
    matrix = np.asarray(efficiency_map, dtype=float)
    if matrix.shape != EFFICIENCY_SHAPES[system_type]:
        raise ValueError(f"efficiency_map must have shape {EFFICIENCY_SHAPES[system_type]}, got {matrix.shape}")

    vectors = []
    for entry_index in range(entry_count):
        row, cols = operation_cells_for_entry(system_type, entry_index)
        vectors.append(matrix[row, cols])
    return np.asarray(vectors, dtype=float)


def reconstruct_density_matrix(
    system_type: str,
    counts: Iterable[float],
    basis_config_path: str | Path,
    efficiency_map_path: Optional[str | Path] = None,
    shots: Optional[Iterable[float]] = None,
    weight_reference: str = "max",
    weight_clip_range: Optional[tuple[float, float]] = (0.1, 10.0),
    regularization: float = 0.0,
    rate_power: float = 1.0,
    phase_sign: float = -1.0,
    method: str = "linear",
    mle_max_iter: int = 500,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict:
    """Return both raw and calibrated density reconstructions.

    The calibrated path applies two corrections:

    * basis-amplitude correction: each entry's amplitude vector is multiplied
      by the retrieval rates of its atom cells before the measurement operator
      is built;
    * count correction: each operation point count is multiplied by
      ``reference_rate / mean(atom_cell_rates)`` before least-squares fitting.

    The ``weight_clip_range`` parameter clips the per-entry count weights to a
    sane range, preventing extreme low-efficiency entries from dominating the
    reconstruction.  Pass ``None`` to disable clipping.  ``rate_power`` keeps
    the historical rate scale at ``1.0``; use ``0.5`` only for the diagnostic
    retrieval-sqrt path where rates are treated as amplitude factors.

    ``method`` selects the calibrated estimator (V5 physics-math upgrade;
    default ``"linear"`` preserves the historical behavior exactly):

    * ``"linear"`` — weighted least squares on probabilities + PSD projection
      (historical path; known to bias fidelity upward in the low-count limit);
    * ``"mle"`` — Poisson maximum-likelihood over raw counts with the
      retrieval-weighted measurement operators (diluted R-rho-R iteration;
      Hradil et al., PRA 55, R1561, 1997).  The count correction is *not*
      applied on this path because the retrieval model already lives inside
      the operators.  The linear calibrated result is still computed and
      exposed under ``result["calibrated_linear"]`` for side-by-side audits.

    ``bootstrap_samples > 0`` (only meaningful with ``method="mle"``) adds a
    parametric-bootstrap posterior summary (purity mean/interval) under
    ``result["mle_posterior"]``.
    """
    _validate_system(system_type)
    method = str(method).lower()
    if method not in {"linear", "mle"}:
        raise ValueError(f"method must be 'linear' or 'mle', got {method!r}")
    rate_power = float(rate_power)
    if rate_power <= 0.0:
        raise ValueError(f"rate_power must be positive, got {rate_power}")
    phase_sign = float(phase_sign)
    basis_config_path = Path(basis_config_path)
    basis_entries = raw_reconstruction.load_basis_entries(system_type, str(basis_config_path))
    basis_count = len(basis_entries)
    counts_array, count_shape_info = _values_for_basis(counts, basis_count, "counts")
    shots_array = _shots_for_basis(shots, basis_count)

    efficiency_path = _resolve_efficiency_map_path(system_type, efficiency_map_path)
    efficiency_map = load_efficiency_map(system_type, efficiency_path)
    if rate_power != 1.0:
        efficiency_map = np.power(efficiency_map, rate_power)
    efficiency_source = _efficiency_source(efficiency_path)
    rate_vectors = rate_vectors_for_entries(system_type, efficiency_map, basis_count)
    operation_rates = _mean_positive(rate_vectors, axis=1)
    reference_rate = _reference_rate(operation_rates, weight_reference)
    count_weights = reference_rate / np.maximum(operation_rates, MIN_RATE)
    if weight_clip_range is not None:
        count_weights = np.clip(count_weights, weight_clip_range[0], weight_clip_range[1])
    corrected_counts = counts_array * count_weights

    raw_result = raw_reconstruction.reconstruct_density_matrix(
        system_type,
        counts_array,
        str(basis_config_path),
        shots=shots_array,
        regularization=regularization,
        phase_sign=phase_sign,
    )
    calibrated_entries = _apply_rate_vectors_to_basis(basis_entries, rate_vectors)
    linear_calibrated_result = _reconstruct_from_entries(
        system_type,
        corrected_counts,
        calibrated_entries,
        str(basis_config_path),
        shots=shots_array,
        regularization=regularization,
        phase_sign=phase_sign,
    )

    mle_result = None
    mle_posterior = None
    if method == "mle":
        mle_result, mle_posterior = _reconstruct_mle_from_entries(
            system_type,
            counts_array,
            calibrated_entries,
            str(basis_config_path),
            phase_sign=phase_sign,
            max_iter=mle_max_iter,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        calibrated_result = mle_result
    else:
        calibrated_result = linear_calibrated_result

    calibration_info = _calibration_diagnostics(
        system_type=system_type,
        efficiency_path=efficiency_path,
        efficiency_source=efficiency_source,
        efficiency_map=efficiency_map,
        basis_count=basis_count,
        count_shape_info=count_shape_info,
        rate_vectors=rate_vectors,
        operation_rates=operation_rates,
        count_weights=count_weights,
        reference_rate=reference_rate,
        weight_reference=weight_reference,
        rate_power=rate_power,
    )

    result = dict(calibrated_result)
    result["raw"] = raw_result
    result["calibrated"] = calibrated_result
    result["reconstruction_method"] = method
    if mle_result is not None:
        result["calibrated_linear"] = linear_calibrated_result
        result["mle_vs_linear_fidelity"] = _state_fidelity(
            mle_result["rho_core"], linear_calibrated_result["rho_core"]
        )
    if mle_posterior is not None:
        result["mle_posterior"] = mle_posterior
    result["raw_rho_full"] = raw_result["rho_full"]
    result["raw_re_rho"] = raw_result["re_rho"]
    result["raw_im_rho"] = raw_result["im_rho"]
    result["calibrated_rho_full"] = calibrated_result["rho_full"]
    result["calibrated_re_rho"] = calibrated_result["re_rho"]
    result["calibrated_im_rho"] = calibrated_result["im_rho"]
    result["raw_counts"] = counts_array
    result["corrected_counts"] = corrected_counts
    result["count_weights"] = count_weights
    result["operation_rates"] = operation_rates
    result["rate_vectors"] = rate_vectors
    result["efficiency_map_path"] = str(efficiency_path)
    result["efficiency_map_source"] = efficiency_source
    result["diagnostics"] = dict(calibrated_result.get("diagnostics", {}))
    result["diagnostics"]["reconstruction_method"] = method
    result["diagnostics"]["calibration"] = calibration_info
    result["diagnostics"]["raw"] = raw_result.get("diagnostics", {})
    return result


def _reconstruct_mle_from_entries(
    system_type: str,
    counts: np.ndarray,
    calibrated_entries: list[dict],
    basis_config_path: str,
    *,
    phase_sign: float,
    max_iter: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, Optional[dict]]:
    """Poisson R-rho-R MLE with retrieval-weighted operators (V5 upgrade).

    The retrieval model lives inside the measurement operators, so the raw
    (uncorrected) counts are the statistically correct input.  Imported lazily
    to keep ``ml_recon.physics_models`` an optional dependency of the
    reconstruction path, mirroring the documented dynamic-import boundaries.
    """

    from ml_recon.physics_models.count_likelihood import (
        bootstrap_density_posterior,
        rrr_density_mle,
    )

    operators = np.stack(
        [raw_reconstruction._measurement_operator(entry, phase_sign=phase_sign) for entry in calibrated_entries]
    )
    rho_mle = rrr_density_mle(counts, operators, max_iter=int(max_iter))

    core_dim = raw_reconstruction.SYSTEM_SPECS[system_type]["core_dim"]
    full_dim = raw_reconstruction.SYSTEM_SPECS[system_type]["full_dim"]
    full_indices = raw_reconstruction.SYSTEM_SPECS[system_type]["full_indices"]
    rho_full = np.zeros((full_dim, full_dim), dtype=complex)
    for i, row_index in enumerate(full_indices):
        for j, col_index in enumerate(full_indices):
            rho_full[row_index, col_index] = rho_mle[i, j]

    probabilities = np.clip(
        np.real(np.einsum("oij,ji->o", operators, rho_mle)), MIN_RATE, None
    )
    scale = float(np.sum(counts)) / float(np.sum(probabilities))
    expected = scale * probabilities
    deviance_terms = np.where(
        counts > 0,
        counts * np.log(np.maximum(counts, MIN_RATE) / expected) - (counts - expected),
        expected,
    )
    diagnostics = {
        "trace": float(np.real(np.trace(rho_mle))),
        "hermitian_residual": float(np.linalg.norm(rho_mle - rho_mle.conj().T)),
        "eigenvalues": np.real(np.linalg.eigvalsh(rho_mle)).tolist(),
        "estimator": "poisson_rrr_mle",
        "mle_max_iter": int(max_iter),
        "poisson_deviance": float(2.0 * np.sum(deviance_terms)),
        "count_scale": scale,
        "phase_sign": phase_sign,
        "basis_phase_convention": raw_reconstruction.BASIS_PHASE_CONVENTION,
    }

    posterior = None
    if int(bootstrap_samples) > 0:
        summary = bootstrap_density_posterior(
            counts,
            operators,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed),
        )
        posterior = {
            "purity_mean": summary.purity_mean,
            "purity_interval": list(summary.purity_interval),
            "bootstrap_samples": summary.bootstrap_samples,
            "diagnostics": summary.diagnostics,
        }

    result = {
        "system_type": system_type,
        "basis_config_path": basis_config_path,
        "rho_linear": rho_mle,
        "rho_core": rho_mle,
        "rho_full": rho_full,
        "re_rho": np.real(rho_full),
        "im_rho": np.imag(rho_full),
        "re_rho_core": np.real(rho_mle),
        "im_rho_core": np.imag(rho_mle),
        "diagnostics": diagnostics,
    }
    return result, posterior


def _state_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    """Squared Uhlmann fidelity between two core density matrices."""

    first = 0.5 * (first + first.conj().T)
    values, vectors = np.linalg.eigh(first)
    root = (vectors * np.sqrt(np.clip(np.real(values), 0.0, None))) @ vectors.conj().T
    product = root @ second @ root
    product = 0.5 * (product + product.conj().T)
    evals = np.clip(np.real(np.linalg.eigvalsh(product)), 0.0, None)
    return float(np.sum(np.sqrt(evals)) ** 2)


def _reconstruct_from_entries(
    system_type: str,
    counts: np.ndarray,
    basis_entries: list[dict],
    basis_config_path: str,
    shots: Optional[Iterable[float]] = None,
    regularization: float = 0.0,
    phase_sign: float = -1.0,
) -> dict:
    if system_type not in raw_reconstruction.SYSTEM_SPECS:
        raise ValueError(f"Unsupported system_type: {system_type}")
    if counts.size != len(basis_entries):
        raise ValueError(f"Expected {len(basis_entries)} counts for {system_type}, got {counts.size}")

    phase_sign = float(phase_sign)
    probabilities, normalization_info = raw_reconstruction._counts_to_probabilities(counts, shots)
    operators = [
        raw_reconstruction._measurement_operator(entry, phase_sign=phase_sign)
        for entry in basis_entries
    ]

    core_dim = raw_reconstruction.SYSTEM_SPECS[system_type]["core_dim"]
    measurement_matrix = np.vstack([operator.reshape(1, core_dim * core_dim) for operator in operators])

    if regularization > 0.0:
        measurement_matrix_h = measurement_matrix.conj().T
        MtM = measurement_matrix_h @ measurement_matrix
        Mtp = measurement_matrix_h @ probabilities
        identity = np.eye(MtM.shape[0])
        solution = np.linalg.solve(MtM + regularization * identity, Mtp)
    else:
        solution, _, _, _ = np.linalg.lstsq(measurement_matrix, probabilities, rcond=None)
    rho_linear = solution.reshape(core_dim, core_dim)
    rho_linear = 0.5 * (rho_linear + rho_linear.conj().T)
    rho_physical = raw_reconstruction._project_physical_density_matrix(rho_linear)

    full_dim = raw_reconstruction.SYSTEM_SPECS[system_type]["full_dim"]
    full_indices = raw_reconstruction.SYSTEM_SPECS[system_type]["full_indices"]
    rho_full = np.zeros((full_dim, full_dim), dtype=complex)
    for i, row_index in enumerate(full_indices):
        for j, col_index in enumerate(full_indices):
            rho_full[row_index, col_index] = rho_physical[i, j]

    diagnostics = {
        "trace": float(np.real(np.trace(rho_physical))),
        "hermitian_residual": float(np.linalg.norm(rho_physical - rho_physical.conj().T)),
        "eigenvalues": np.real(np.linalg.eigvalsh(rho_physical)).tolist(),
        "measurement_rank": int(np.linalg.matrix_rank(measurement_matrix)),
        "normalization": normalization_info,
        "phase_sign": phase_sign,
        "basis_phase_convention": raw_reconstruction.BASIS_PHASE_CONVENTION,
    }

    return {
        "system_type": system_type,
        "basis_config_path": basis_config_path,
        "rho_linear": rho_linear,
        "rho_core": rho_physical,
        "rho_full": rho_full,
        "re_rho": np.real(rho_full),
        "im_rho": np.imag(rho_full),
        "re_rho_core": np.real(rho_physical),
        "im_rho_core": np.imag(rho_physical),
        "diagnostics": diagnostics,
    }


def _apply_rate_vectors_to_basis(basis_entries: list[dict], rate_vectors: np.ndarray) -> list[dict]:
    calibrated_entries = []
    for entry, rates in zip(basis_entries, rate_vectors):
        calibrated = deepcopy(entry)
        amp = np.asarray(entry["amp"], dtype=float)
        if rates.size != amp.size:
            raise ValueError(f"rate vector length {rates.size} does not match basis amp length {amp.size}")
        calibrated["amp"] = (amp * rates).astype(float).tolist()
        calibrated_entries.append(calibrated)
    return calibrated_entries


def _values_for_basis(values: Iterable[float], basis_count: int, name: str) -> tuple[np.ndarray, dict]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    info = {
        f"{name}_input_size": int(arr.size),
        f"{name}_basis_size": int(basis_count),
        f"{name}_trimmed": 0,
        f"{name}_padded": 0,
    }
    if arr.size > basis_count:
        info[f"{name}_trimmed"] = int(arr.size - basis_count)
        return arr[:basis_count].copy(), info
    if arr.size < basis_count:
        out = np.zeros(basis_count, dtype=float)
        out[:arr.size] = arr
        info[f"{name}_padded"] = int(basis_count - arr.size)
        return out, info
    return arr.copy(), info


def _shots_for_basis(shots: Optional[Iterable[float]], basis_count: int):
    if shots is None or np.isscalar(shots):
        return shots
    return _values_for_basis(shots, basis_count, "shots")[0]


def _mean_positive(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    positive = np.where(arr > 0, arr, np.nan)
    means = np.nanmean(positive, axis=axis)
    if np.any(np.isnan(means)):
        means = np.where(np.isnan(means), MIN_RATE, means)
    return means


def _reference_rate(operation_rates: np.ndarray, weight_reference: str) -> float:
    positive = operation_rates[operation_rates > 0]
    if positive.size == 0:
        return 1.0
    if weight_reference == "max":
        return float(np.max(positive))
    if weight_reference == "mean":
        return float(np.mean(positive))
    if weight_reference == "median":
        return float(np.median(positive))
    raise ValueError("weight_reference must be 'max', 'mean', or 'median'")


def _calibration_diagnostics(
    *,
    system_type: str,
    efficiency_path: Path,
    efficiency_source: str,
    efficiency_map: np.ndarray,
    basis_count: int,
    count_shape_info: dict,
    rate_vectors: np.ndarray,
    operation_rates: np.ndarray,
    count_weights: np.ndarray,
    reference_rate: float,
    weight_reference: str,
    rate_power: float,
) -> dict:
    return {
        "enabled": True,
        "efficiency_map_path": str(efficiency_path),
        "efficiency_map_source": efficiency_source,
        "efficiency_map_shape": list(efficiency_map.shape),
        "basis_count": int(basis_count),
        "mapping": _mapping_description(system_type),
        "weight_reference": weight_reference,
        "rate_power": float(rate_power),
        "reference_rate": float(reference_rate),
        "atom_rate_min": float(np.min(rate_vectors)),
        "atom_rate_max": float(np.max(rate_vectors)),
        "operation_rate_min": float(np.min(operation_rates)),
        "operation_rate_max": float(np.max(operation_rates)),
        "operation_rate_mean": float(np.mean(operation_rates)),
        "count_weight_min": float(np.min(count_weights)),
        "count_weight_max": float(np.max(count_weights)),
        "count_weight_mean": float(np.mean(count_weights)),
        **count_shape_info,
    }


def _mapping_description(system_type: str) -> str:
    if system_type == "3qubit":
        return "entry i -> row=i//5, cols=3*(i%5)..3*(i%5)+2"
    return "entry i -> local=i%64, row=local//4, cols=4*(local%4)..4*(local%4)+3"


def _efficiency_source(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "csv_idler_absolute"
    return "retrieval_rate_text_matrix"


def _resolve_efficiency_map_path(system_type: str, requested: Optional[str | Path]) -> Path:
    """Prefer the measured CSV but retain the tracked legacy map as fallback.

    The CSV is a locally generated source-characterization artifact and is
    intentionally not committed.  An explicit missing path still raises; only
    the implicit default obtains this backward-compatible fallback.
    """
    if requested is not None:
        return Path(requested)
    preferred = default_efficiency_map_path(system_type)
    if preferred.exists():
        return preferred
    legacy = legacy_efficiency_map_path(system_type)
    return legacy if legacy.exists() else preferred


def _validate_system(system_type: str) -> None:
    if system_type not in EFFICIENCY_SHAPES:
        raise ValueError(f"Unsupported system_type: {system_type}")


def _json_ready(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _parse_counts(counts_arg: str) -> np.ndarray:
    path = Path(counts_arg)
    if path.exists():
        return np.loadtxt(path, dtype=float).reshape(-1)
    return np.asarray(json.loads(counts_arg), dtype=float).reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct raw and retrieval-calibrated density matrices.")
    parser.add_argument("--system", choices=["3qubit", "4qubit"], required=True)
    parser.add_argument("--counts", required=True, help="JSON list of counts or a text file path.")
    parser.add_argument("--basis-config", required=True)
    parser.add_argument("--efficiency-map", default=None)
    parser.add_argument("--weight-reference", choices=["max", "mean", "median"], default="max")
    parser.add_argument("--phase-sign", type=float, default=-1.0)
    parser.add_argument("--method", choices=["linear", "mle"], default="linear",
                        help="Calibrated estimator; 'mle' enables the V5 Poisson R-rho-R path.")
    parser.add_argument("--mle-max-iter", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help="Parametric bootstrap re-fits for the MLE posterior summary.")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reconstruct_density_matrix(
        args.system,
        _parse_counts(args.counts),
        args.basis_config,
        efficiency_map_path=args.efficiency_map,
        weight_reference=args.weight_reference,
        phase_sign=args.phase_sign,
        method=args.method,
        mle_max_iter=args.mle_max_iter,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = _json_ready(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {output_path}")
    else:
        print(json.dumps(payload["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
