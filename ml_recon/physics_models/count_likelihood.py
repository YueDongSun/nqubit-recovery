"""Count-level quantum likelihoods and positive density reconstruction.

The parameterisation follows the physical acquisition order: Born
probabilities are converted into expected photon counts and only then sampled.
This prevents training code from treating additive Gaussian perturbations of a
reconstructed density matrix as if they were detector observations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from system_control.experiment_control.data_protocol import load_snapshot_manifest


EPS = 1e-12


@dataclass
class CountNoiseProfile:
    dark_counts: float | np.ndarray = 0.0
    accidental_counts: float | np.ndarray = 0.0
    detection_efficiency: float | np.ndarray = 1.0
    overdispersion_alpha: float = 0.0
    multi_click_rates: Dict[str, float] = field(default_factory=dict)
    source_sample_count: int = 0
    quality_flags: list[str] = field(default_factory=list)
    schema_version: int = 1

    def validate(self) -> None:
        for name, value in (
            ("dark_counts", self.dark_counts),
            ("accidental_counts", self.accidental_counts),
            ("detection_efficiency", self.detection_efficiency),
        ):
            arr = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must contain finite non-negative values")
        if self.overdispersion_alpha < 0 or not np.isfinite(self.overdispersion_alpha):
            raise ValueError("overdispersion_alpha must be finite and non-negative")

    @property
    def distribution(self) -> str:
        return "negative_binomial" if self.overdispersion_alpha > 1e-8 else "poisson"

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("dark_counts", "accidental_counts", "detection_efficiency"):
            value = np.asarray(payload[key])
            payload[key] = float(value) if value.ndim == 0 else value.tolist()
        payload["distribution"] = self.distribution
        return payload


@dataclass
class DensityPosteriorSummary:
    mean_density: np.ndarray
    purity_mean: float
    purity_interval: tuple[float, float]
    fidelity_mean: Optional[float] = None
    fidelity_interval: Optional[tuple[float, float]] = None
    bootstrap_samples: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def cholesky_density(lower: np.ndarray) -> np.ndarray:
    matrix = np.asarray(lower, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lower must be a square complex matrix")
    matrix = np.tril(matrix)
    product = matrix @ matrix.conj().T
    trace = float(np.real(np.trace(product)))
    if trace <= EPS:
        raise ValueError("Cholesky factor has zero norm")
    return product / trace


def project_density(matrix: np.ndarray) -> np.ndarray:
    rho = np.asarray(matrix, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    values, vectors = np.linalg.eigh(rho)
    values = np.clip(np.real(values), 0.0, None)
    if float(np.sum(values)) <= EPS:
        values = np.ones_like(values)
    rho = (vectors * values) @ vectors.conj().T
    return rho / np.trace(rho)


def born_probabilities(rho: np.ndarray, measurement_operators: np.ndarray) -> np.ndarray:
    density = project_density(rho)
    operators = np.asarray(measurement_operators, dtype=complex)
    if operators.ndim != 3 or operators.shape[1:] != density.shape:
        raise ValueError("measurement_operators must have shape (operations, dim, dim)")
    probabilities = np.real(np.einsum("oij,ji->o", operators, density))
    return np.clip(probabilities, 0.0, None)


def expected_counts(
    rho: np.ndarray,
    measurement_operators: np.ndarray,
    exposure: float | np.ndarray,
    noise: Optional[CountNoiseProfile] = None,
) -> np.ndarray:
    profile = noise or CountNoiseProfile()
    profile.validate()
    probabilities = born_probabilities(rho, measurement_operators)
    exposure_arr = np.asarray(exposure, dtype=float)
    if np.any(exposure_arr < 0):
        raise ValueError("exposure must be non-negative")
    mean = (
        np.asarray(profile.dark_counts, dtype=float)
        + np.asarray(profile.accidental_counts, dtype=float)
        + np.asarray(profile.detection_efficiency, dtype=float) * exposure_arr * probabilities
    )
    return np.clip(mean, EPS, None)


def poisson_log_likelihood(observed: np.ndarray, mean: np.ndarray) -> float:
    y, mu = np.broadcast_arrays(np.asarray(observed, dtype=float), np.asarray(mean, dtype=float))
    if np.any(y < 0) or np.any(mu <= 0):
        raise ValueError("Poisson observations must be non-negative and means positive")
    lgamma = np.vectorize(math.lgamma)
    return float(np.sum(y * np.log(mu) - mu - lgamma(y + 1.0)))


def negative_binomial_log_likelihood(
    observed: np.ndarray,
    mean: np.ndarray,
    alpha: float,
) -> float:
    """NB2 likelihood with variance ``mu + alpha * mu**2``."""

    if alpha <= 0:
        return poisson_log_likelihood(observed, mean)
    y, mu = np.broadcast_arrays(np.asarray(observed, dtype=float), np.asarray(mean, dtype=float))
    if np.any(y < 0) or np.any(mu <= 0):
        raise ValueError("negative-binomial observations must be non-negative and means positive")
    r = 1.0 / float(alpha)
    p = r / (r + mu)
    lgamma = np.vectorize(math.lgamma)
    values = lgamma(y + r) - lgamma(r) - lgamma(y + 1.0) + r * np.log(p) + y * np.log1p(-p)
    return float(np.sum(values))


def count_log_likelihood(observed: np.ndarray, mean: np.ndarray, noise: CountNoiseProfile) -> float:
    if noise.distribution == "negative_binomial":
        return negative_binomial_log_likelihood(observed, mean, noise.overdispersion_alpha)
    return poisson_log_likelihood(observed, mean)


def rrr_density_mle(
    observed: np.ndarray,
    measurement_operators: np.ndarray,
    *,
    max_iter: int = 500,
    tolerance: float = 1e-9,
    initial_density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Positive maximum-likelihood estimate using the diluted R-rho-R update."""

    counts = np.asarray(observed, dtype=float).reshape(-1)
    operators = np.asarray(measurement_operators, dtype=complex)
    if operators.ndim != 3 or operators.shape[0] != counts.size:
        raise ValueError("one measurement operator is required for every observed count")
    if np.any(counts < 0) or float(np.sum(counts)) <= 0:
        raise ValueError("observed counts must be non-negative with a positive total")
    dim = operators.shape[1]
    if operators.shape[2] != dim:
        raise ValueError("measurement operators must be square")
    rho = project_density(initial_density) if initial_density is not None else np.eye(dim, dtype=complex) / dim
    identity = np.eye(dim, dtype=complex)
    dilution = 0.5

    for _ in range(max(1, int(max_iter))):
        probabilities = np.clip(born_probabilities(rho, operators), EPS, None)
        weights = counts / probabilities
        r_operator = np.einsum("o,oij->ij", weights, operators)
        r_operator /= max(float(np.sum(counts)), EPS)
        update = (1.0 - dilution) * identity + dilution * r_operator
        candidate = project_density(update @ rho @ update.conj().T)
        if np.linalg.norm(candidate - rho, ord="fro") <= tolerance:
            rho = candidate
            break
        rho = candidate
    return rho


def rrr_density_mle_batch(
    observed: np.ndarray,
    measurement_operators: np.ndarray,
    *,
    max_iter: int = 500,
    tolerance: float = 1e-9,
    initial_density: Optional[np.ndarray] = None,
    dilution: float = 0.5,
) -> np.ndarray:
    """Batched diluted R-rho-R MLE (V5 physics-math upgrade).

    Matrix-form twin of :func:`rrr_density_mle`: one snapshot per batch row,
    all einsums vectorized, so a whole campaign (e.g. the 1000-state 4Q
    campaign) iterates simultaneously.  Iteration stops when the largest
    per-snapshot Frobenius step falls below ``tolerance``.
    """

    counts = np.asarray(observed, dtype=float)
    operators = np.asarray(measurement_operators, dtype=complex)
    if counts.ndim == 1:
        counts = counts[None, :]
    if operators.ndim != 3 or operators.shape[0] != counts.shape[1]:
        raise ValueError("one measurement operator is required for every observed count")
    if np.any(counts < 0) or np.any(np.sum(counts, axis=1) <= 0):
        raise ValueError("observed counts must be non-negative with a positive total per row")
    dim = operators.shape[1]
    if operators.shape[2] != dim:
        raise ValueError("measurement operators must be square")
    if initial_density is None:
        rho = np.broadcast_to(np.eye(dim, dtype=complex), (counts.shape[0], dim, dim)).copy()
    else:
        initial = np.asarray(initial_density, dtype=complex)
        if initial.ndim == 2:
            initial = np.broadcast_to(initial, (counts.shape[0], dim, dim)).copy()
        rho = np.stack([project_density(candidate) for candidate in initial])
    identity = np.eye(dim, dtype=complex)
    totals = np.maximum(np.sum(counts, axis=1), EPS)

    for _ in range(max(1, int(max_iter))):
        probabilities = np.clip(np.real(np.einsum("bij,oji->bo", rho, operators)), EPS, None)
        weights = counts / probabilities
        r_operator = np.einsum("bo,oij->bij", weights, operators) / totals[:, None, None]
        update = (1.0 - dilution) * identity + dilution * r_operator
        candidate = update @ rho @ np.swapaxes(update.conj(), -2, -1)
        candidate = _project_density_batch(candidate)
        step = np.max(np.linalg.norm(candidate - rho, ord="fro", axis=(-2, -1)))
        rho = candidate
        if step <= tolerance:
            break
    return rho


def _project_density_batch(matrices: np.ndarray) -> np.ndarray:
    """Batched Hermitize + eigenvalue clip + trace normalization."""

    hermitian = 0.5 * (matrices + np.swapaxes(matrices.conj(), -2, -1))
    values, vectors = np.linalg.eigh(hermitian)
    values = np.clip(np.real(values), 0.0, None)
    degenerate = np.sum(values, axis=1) <= EPS
    if np.any(degenerate):
        values[degenerate] = 1.0
    projected = (vectors * values[:, None, :]) @ np.swapaxes(vectors.conj(), -2, -1)
    traces = np.real(np.trace(projected, axis1=-2, axis2=-1))
    return projected / traces[:, None, None]


def bootstrap_density_posterior(
    observed: np.ndarray,
    measurement_operators: np.ndarray,
    *,
    target_density: Optional[np.ndarray] = None,
    samples: int = 128,
    seed: int = 0,
) -> DensityPosteriorSummary:
    estimate = rrr_density_mle(observed, measurement_operators)
    probabilities = born_probabilities(estimate, measurement_operators)
    total = max(float(np.sum(observed)), 1.0)
    probabilities = probabilities / max(float(np.sum(probabilities)), EPS)
    rng = np.random.default_rng(seed)
    simulated = np.stack([
        rng.multinomial(int(round(total)), probabilities)
        for _ in range(max(1, int(samples)))
    ])
    # V5 upgrade: the bootstrap re-fits are evaluated as one batched matrix
    # iteration instead of a serial per-sample loop (identical update rule).
    densities = list(
        rrr_density_mle_batch(
            simulated,
            measurement_operators,
            max_iter=200,
            initial_density=np.broadcast_to(estimate, (simulated.shape[0], *estimate.shape)),
        )
    )
    purities = [float(np.real(np.trace(candidate @ candidate))) for candidate in densities]
    fidelities = []
    if target_density is not None:
        fidelities = [_density_fidelity(target_density, candidate) for candidate in densities]
    mean_density = project_density(np.mean(densities, axis=0))
    purity_interval = tuple(float(value) for value in np.quantile(purities, [0.025, 0.975]))
    fidelity_interval = None
    fidelity_mean = None
    if fidelities:
        fidelity_mean = float(np.mean(fidelities))
        fidelity_interval = tuple(float(value) for value in np.quantile(fidelities, [0.025, 0.975]))
    return DensityPosteriorSummary(
        mean_density=mean_density,
        purity_mean=float(np.mean(purities)),
        purity_interval=purity_interval,
        fidelity_mean=fidelity_mean,
        fidelity_interval=fidelity_interval,
        bootstrap_samples=len(densities),
        diagnostics={"method": "multinomial_parametric_bootstrap", "count_total": total},
    )


def fit_count_noise_profile(snapshot_dirs: Sequence[str | Path]) -> CountNoiseProfile:
    """Estimate a measured-count prior without inventing unavailable labels."""

    arrays: Dict[str, list[np.ndarray]] = {"signal_bins": [], "idler_bins": [], "coin_bins": []}
    multi_totals = {"multi2": 0.0, "multi3": 0.0, "multi4": 0.0}
    sample_count = 0
    quality_flags: list[str] = []
    for item in snapshot_dirs:
        directory = Path(item)
        manifest = load_snapshot_manifest(directory)
        sample_count += 1
        for field_name in arrays:
            entry = manifest.arrays.get(field_name)
            if entry is not None:
                arrays[field_name].append(np.asarray(np.load(directory / entry["path"]), dtype=float).reshape(-1))
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        for key in multi_totals:
            multi_totals[key] += float(metadata.get(f"{key}_count", 0.0) or 0.0)

    if sample_count == 0:
        raise ValueError("at least one snapshot directory is required")
    usable = [value for streams in arrays.values() for value in streams]
    if not usable:
        raise ValueError("snapshots do not contain signal/idler/coin arrays")
    flattened = np.concatenate(usable)
    mean = float(np.mean(flattened))
    variance = float(np.var(flattened, ddof=1)) if flattened.size > 1 else mean
    alpha = max(0.0, (variance - mean) / max(mean * mean, EPS))

    signal_total = sum(float(np.sum(value)) for value in arrays["signal_bins"])
    idler_total = sum(float(np.sum(value)) for value in arrays["idler_bins"])
    coin_total = sum(float(np.sum(value)) for value in arrays["coin_bins"])
    efficiency = coin_total / max(min(signal_total, idler_total), 1.0)
    # Dark and accidental counts are not identifiable from W-state snapshots
    # alone.  Keep them explicit and zero until dedicated off-window data exist.
    quality_flags.extend(["dark_counts_require_off_window", "accidentals_require_delay_window"])
    total_events = max(signal_total + idler_total + coin_total, 1.0)
    profile = CountNoiseProfile(
        dark_counts=0.0,
        accidental_counts=0.0,
        detection_efficiency=float(np.clip(efficiency, 0.0, 1.0)),
        overdispersion_alpha=alpha,
        multi_click_rates={key: value / total_events for key, value in multi_totals.items()},
        source_sample_count=sample_count,
        quality_flags=quality_flags,
    )
    profile.validate()
    return profile


def _density_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    a = project_density(first)
    b = project_density(second)
    values, vectors = np.linalg.eigh(a)
    sqrt_a = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T
    # Uhlmann fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2: the intermediate
    # product must keep its own trace (Tr = Tr(rho sigma)); renormalizing it here
    # (the old project_density call) collapsed orthogonal pairs to 1.0 and pulled
    # every mixed pair toward 1. Only hermitize against roundoff before eigh.
    middle = sqrt_a @ b @ sqrt_a
    middle = 0.5 * (middle + middle.conj().T)
    # The clip at 0 is an epsilon floor for eigh roundoff only (exact PSD inputs
    # give eigenvalues >= -1e-15); it is not a substitute for a missing rate or
    # weight, and no significant negative mass can be discarded on trace-1 inputs.
    return float(np.clip(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None))) ** 2, 0.0, 1.0))
