"""Formal observation-space covariance and support-restricted whitening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .contracts import sha256_of
from .csm_contract import WhiteningThresholdContract


def _as_square_covariance(covariance: np.ndarray) -> np.ndarray:
    array = np.asarray(covariance, dtype=np.float64)
    if array.ndim == 1:
        if np.any(array < 0.0):
            raise ValueError("diagonal covariance entries must be non-negative")
        array = np.diag(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("covariance must be a square matrix or a diagonal vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("covariance must be finite")
    return 0.5 * (array + array.T)


def multinomial_setting_covariance(probabilities: np.ndarray, shots: float) -> np.ndarray:
    """Return the full categorical covariance for one multinomial setting."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < -1e-12):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(float(p.sum()), 1.0, atol=1e-9):
        raise ValueError("multinomial probabilities must sum to one")
    if not np.isfinite(shots) or float(shots) <= 0.0:
        raise ValueError("multinomial shots must be positive")
    p = np.clip(p, 0.0, 1.0)
    return (np.diag(p) - np.outer(p, p)) / float(shots)


def multinomial_covariance(probabilities: np.ndarray, shots: float) -> np.ndarray:
    """Compatibility alias for the formal per-setting covariance."""
    return multinomial_setting_covariance(probabilities, shots)


def block_diagonal_covariance(blocks: Iterable[np.ndarray]) -> np.ndarray:
    """Build a dense small-N block covariance without adding artificial floors."""
    normalized = tuple(_as_square_covariance(block) for block in blocks)
    if not normalized:
        raise ValueError("at least one covariance block is required")
    size = sum(block.shape[0] for block in normalized)
    result = np.zeros((size, size), dtype=np.float64)
    cursor = 0
    for block in normalized:
        end = cursor + block.shape[0]
        result[cursor:end, cursor:end] = block
        cursor = end
    return result


def sufficient_stat_covariance(
    covariance: np.ndarray,
    statistic_transform: np.ndarray | None = None,
    *,
    blocks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Construct covariance for selected sufficient statistics.

    ``statistic_transform`` maps categorical outcomes to the registered
    sufficient-statistic vector.  ``blocks`` is a convenience for independent
    setting blocks.  Neither path adds an observation-space diagonal floor.
    """
    base = block_diagonal_covariance(blocks) if blocks is not None else _as_square_covariance(covariance)
    if statistic_transform is None:
        return base
    transform = np.asarray(statistic_transform, dtype=np.float64)
    if transform.ndim != 2 or transform.shape[1] != base.shape[0]:
        raise ValueError("statistic_transform must have shape [stats, covariance_dim]")
    projected = transform @ base @ transform.T
    return 0.5 * (projected + projected.T)


@dataclass(frozen=True)
class SupportWhitener:
    """Whitening operator restricted to the observable covariance support."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    absolute_threshold: float
    relative_threshold: float
    covariance_sha: str
    epsilon: float = 1e-12

    @property
    def support_semantics(self) -> str:
        return "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT"

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def matrix(self) -> np.ndarray:
        if self.rank == 0:
            return np.zeros((0, self.dimension), dtype=np.float64)
        return self.eigenvectors.T / np.sqrt(self.eigenvalues)[:, None]

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.basis

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != self.dimension:
            raise ValueError("whitening input has the wrong observation dimension")
        return self.matrix @ array

    def whiten_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        return self.apply(np.asarray(jacobian, dtype=np.float64))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "support_restricted_whitening_v1",
            "rank": self.rank,
            "dimension": self.dimension,
            "eigenvalues": self.eigenvalues.astype(np.float64).tolist(),
            "absolute_threshold": self.absolute_threshold,
            "relative_threshold": self.relative_threshold,
            "covariance_sha": self.covariance_sha,
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class BlockSupportWhitener:
    """Support-restricted whitening for an explicit block-diagonal covariance."""

    block_bases: tuple[np.ndarray, ...]
    block_eigenvalues: tuple[np.ndarray, ...]
    block_sizes: tuple[int, ...]
    absolute_threshold: float
    relative_threshold: float
    covariance_sha: str
    epsilon: float = 1e-12

    @property
    def support_semantics(self) -> str:
        return "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT"

    @property
    def rank(self) -> int:
        return int(sum(values.size for values in self.block_eigenvalues))

    @property
    def dimension(self) -> int:
        return int(sum(self.block_sizes))

    @property
    def eigenvalues(self) -> np.ndarray:
        if not self.block_eigenvalues:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(self.block_eigenvalues)

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != self.dimension:
            raise ValueError("block whitening input has the wrong observation dimension")
        outputs = []
        cursor = 0
        for basis, eigenvalues, size in zip(
            self.block_bases, self.block_eigenvalues, self.block_sizes):
            end = cursor + size
            block = array[cursor:end]
            transform = basis.T / np.sqrt(eigenvalues)[:, None]
            outputs.append(transform @ block)
            cursor = end
        if not outputs:
            return np.zeros((0,) if array.ndim == 1 else (0, array.shape[1]))
        return np.concatenate(outputs, axis=0)

    def whiten_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        return self.apply(np.asarray(jacobian, dtype=np.float64))

    @property
    def matrix(self) -> np.ndarray:
        raise MemoryError(
            "materializing a block whitener matrix defeats the large-N contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "block_support_restricted_whitening_v1",
            "rank": self.rank,
            "dimension": self.dimension,
            "block_sizes": list(self.block_sizes),
            "block_ranks": [int(values.size) for values in self.block_eigenvalues],
            "absolute_threshold": self.absolute_threshold,
            "relative_threshold": self.relative_threshold,
            "covariance_sha": self.covariance_sha,
            "epsilon": self.epsilon,
        }


def block_support_whitening(
    blocks: Iterable[np.ndarray],
    *,
    absolute_threshold: float = 1e-10,
    relative_threshold: float = 1e-8,
    epsilon: float = 1e-12,
    threshold_contract: WhiteningThresholdContract | None = None,
) -> BlockSupportWhitener:
    """Whiten block-diagonal covariance without materializing the global matrix."""
    if threshold_contract is not None:
        threshold_contract.validate()
        absolute_threshold = threshold_contract.absolute
        relative_threshold = threshold_contract.relative
        epsilon = threshold_contract.epsilon
    if absolute_threshold < 0.0 or relative_threshold < 0.0 or epsilon <= 0.0:
        raise ValueError("invalid block support-whitening thresholds")
    normalized = tuple(_as_square_covariance(block) for block in blocks)
    if not normalized:
        raise ValueError("at least one covariance block is required")
    decompositions = tuple(np.linalg.eigh(block) for block in normalized)
    scale = max(
        max(float(values.max(initial=0.0)), 0.0)
        for values, _ in decompositions)
    threshold = max(float(absolute_threshold), float(relative_threshold) * scale)
    bases = []
    eigenvalues = []
    for values, vectors in decompositions:
        if float(values.min(initial=0.0)) < -max(1e-10, 10.0 * threshold):
            raise ValueError("covariance has a materially negative eigenvalue")
        support = values > threshold
        eigenvalues.append(np.asarray(values[support], dtype=np.float64))
        bases.append(np.asarray(vectors[:, support], dtype=np.float64))
    covariance_sha = sha256_of({
        "kind": "block_diagonal_covariance_v1",
        "blocks": [block.tolist() for block in normalized],
    })
    return BlockSupportWhitener(
        block_bases=tuple(bases),
        block_eigenvalues=tuple(eigenvalues),
        block_sizes=tuple(int(block.shape[0]) for block in normalized),
        absolute_threshold=float(absolute_threshold),
        relative_threshold=float(relative_threshold),
        covariance_sha=covariance_sha,
        epsilon=float(epsilon),
    )


@dataclass(frozen=True)
class MultinomialBlockSupportWhitener:
    """Compact exact support operator for multinomial covariance blocks.

    For C=(diag(p)-p p^T)/shots the range is the positive-probability
    simplex with null direction one.  The map
    sqrt(shots) * diag(p)^(-1/2) * P has the same quadratic form as
    C^+ without constructing or diagonalizing the full block.
    """

    block_probabilities: tuple[np.ndarray, ...]
    block_shots: tuple[float, ...]
    block_sizes: tuple[int, ...]
    absolute_threshold: float
    relative_threshold: float
    covariance_sha: str
    epsilon: float = 1e-12
    support_semantics: str = "EXACT_POSITIVE_MULTINOMIAL_SUPPORT"

    @property
    def rank(self) -> int:
        return int(sum(max(int(np.count_nonzero(p > 0.0)) - 1, 0)
                       for p in self.block_probabilities))

    @property
    def dimension(self) -> int:
        return int(sum(self.block_sizes))

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != self.dimension:
            raise ValueError(
                "multinomial block whitening input has the wrong observation dimension")
        result = np.zeros_like(array)
        cursor = 0
        for probabilities, shots, size in zip(
            self.block_probabilities, self.block_shots, self.block_sizes):
            end = cursor + size
            positive = probabilities > 0.0
            block = array[cursor:end]
            if array.ndim == 1:
                centered = block[positive] - block[positive].mean()
                result[cursor:end][positive] = (
                    np.sqrt(shots) * centered / np.sqrt(probabilities[positive]))
            else:
                centered = block[positive] - block[positive].mean(
                    axis=0, keepdims=True)
                result[cursor:end][positive] = (
                    np.sqrt(shots) * centered / np.sqrt(
                        probabilities[positive, None]))
            cursor = end
        return result

    def whiten_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        return self.apply(np.asarray(jacobian, dtype=np.float64))

    @property
    def matrix(self) -> np.ndarray:
        raise MemoryError(
            "materializing a multinomial block whitener defeats the large-N contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "multinomial_block_support_operator_v1",
            "support_semantics": self.support_semantics,
            "rank": self.rank,
            "dimension": self.dimension,
            "block_sizes": list(self.block_sizes),
            "block_shots": list(self.block_shots),
            "absolute_threshold": self.absolute_threshold,
            "relative_threshold": self.relative_threshold,
            "threshold_application": "NOT_APPLIED_EXACT_POSITIVE_SUPPORT",
            "covariance_sha": self.covariance_sha,
            "epsilon": self.epsilon,
        }


def multinomial_block_support_whitening(
    probability_blocks: Iterable[np.ndarray],
    shots: Iterable[float],
    *,
    absolute_threshold: float = 1e-10,
    relative_threshold: float = 1e-8,
    epsilon: float = 1e-12,
    threshold_contract: WhiteningThresholdContract | None = None,
) -> MultinomialBlockSupportWhitener:
    """Build the exact-positive multinomial support operator.

    This path intentionally does not apply eigenvalue thresholds. Its explicit
    support semantics must be carried in the bound observation/CSM contract.
    """
    if threshold_contract is not None:
        threshold_contract.validate()
        absolute_threshold = threshold_contract.absolute
        relative_threshold = threshold_contract.relative
        epsilon = threshold_contract.epsilon
    normalized = tuple(np.asarray(block, dtype=np.float64).reshape(-1)
                      for block in probability_blocks)
    shot_values = tuple(float(value) for value in shots)
    if not normalized or len(normalized) != len(shot_values):
        raise ValueError("probability blocks and shots must be non-empty and aligned")
    for probabilities, shot_value in zip(normalized, shot_values):
        if (probabilities.size == 0 or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < -1e-12)
                or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-9)):
            raise ValueError("probability blocks must be finite and normalized")
        if not np.isfinite(shot_value) or shot_value <= 0.0:
            raise ValueError("multinomial shots must be positive")
    covariance_sha = sha256_of({
        "kind": "multinomial_block_covariance_v1",
        "probability_blocks": [block.tolist() for block in normalized],
        "shots": list(shot_values),
    })
    return MultinomialBlockSupportWhitener(
        block_probabilities=normalized,
        block_shots=shot_values,
        block_sizes=tuple(int(block.size) for block in normalized),
        absolute_threshold=float(absolute_threshold),
        relative_threshold=float(relative_threshold),
        covariance_sha=covariance_sha,
        epsilon=float(epsilon),
        support_semantics="EXACT_POSITIVE_MULTINOMIAL_SUPPORT",
    )

def support_whitening(
    covariance: np.ndarray,
    *,
    absolute_threshold: float = 1e-10,
    relative_threshold: float = 1e-8,
    epsilon: float = 1e-12,
    threshold_contract: WhiteningThresholdContract | None = None,
) -> SupportWhitener:
    """Return ``Lambda_+^{-1/2} U_+^T`` without fabricating simplex support."""
    if threshold_contract is not None:
        threshold_contract.validate()
        absolute_threshold = threshold_contract.absolute
        relative_threshold = threshold_contract.relative
        epsilon = threshold_contract.epsilon
    if absolute_threshold < 0.0 or relative_threshold < 0.0 or epsilon <= 0.0:
        raise ValueError("invalid support-whitening thresholds")
    matrix = _as_square_covariance(covariance)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(values.max(initial=0.0)), 0.0)
    threshold = max(float(absolute_threshold), float(relative_threshold) * scale)
    if float(values.min(initial=0.0)) < -max(1e-10, 10.0 * threshold):
        raise ValueError("covariance has a materially negative eigenvalue")
    support = values > threshold
    positive_values = np.asarray(values[support], dtype=np.float64)
    positive_vectors = np.asarray(vectors[:, support], dtype=np.float64)
    return SupportWhitener(
        basis=positive_vectors,
        eigenvalues=positive_values,
        absolute_threshold=float(absolute_threshold),
        relative_threshold=float(relative_threshold),
        covariance_sha=sha256_of(matrix.tolist()),
        epsilon=float(epsilon),
    )


def covariance_operator_summary(covariance: np.ndarray, *, kind: str) -> dict[str, object]:
    matrix = _as_square_covariance(covariance)
    whitening = support_whitening(matrix)
    return {
        "kind": str(kind),
        "dimension": int(matrix.shape[0]),
        "support_rank": whitening.rank,
        "support_threshold_sha": sha256_of({
            "absolute": whitening.absolute_threshold,
            "relative": whitening.relative_threshold,
            "epsilon": whitening.epsilon,
        }),
        "covariance_sha": whitening.covariance_sha,
    }


__all__ = [
    "BlockSupportWhitener", "MultinomialBlockSupportWhitener",
    "SupportWhitener", "block_diagonal_covariance", "block_support_whitening",
    "covariance_operator_summary", "multinomial_block_support_whitening",
    "multinomial_covariance", "multinomial_setting_covariance",
    "support_whitening", "sufficient_stat_covariance",
]
