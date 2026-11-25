"""Bound, nuisance-profiled Track B CSM support geometry.

This module is deliberately research-only. It consumes a same-forward
Jacobian/covariance draw set, profiles registered nuisance directions after
canonical covariance-support whitening, and records the exact input
identities. It does not create an action or make a causal-gain claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Sequence

import numpy as np

from ..csm_contract import SingularSupportContract, WhiteningThresholdContract
from ..csm_covariance import support_whitening
from .contracts import canonical_sha256, require_sha256
from .forward_draw_set import BoundTrackBForwardDrawSetV1


CSM_GEOMETRY_STATUSES = frozenset({
    "LEGACY_SINGLE_DRAW_UNVERIFIED",
    "POSTERIOR_DRAW_AGGREGATED_NUISANCE_PROFILED",
    "POSTERIOR_DRAW_AGGREGATED_NO_RECOVERABLE_SUPPORT",
})


def array_identity_sha256(values: np.ndarray | Sequence[object]) -> str:
    """Hash finite numeric array bytes with an explicit shape/dtype ABI.

    A bare caller-supplied label cannot stand in for an input identity. The
    representation is independent of NumPy's native endian choice.
    """
    array = np.asarray(values, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise ValueError("array identity requires finite numeric values")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"TrackBArrayIdentityV1\x00")
    digest.update(str(tuple(int(value) for value in contiguous.shape)).encode("ascii"))
    digest.update(b"\x00float64-le\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _finite_matrix(values: np.ndarray | Sequence[object], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite rank-2 matrix")
    return matrix


def _profiled_draw_geometry(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    nuisance_jacobian: np.ndarray | None,
    whitening: WhiteningThresholdContract,
    singular_support: SingularSupportContract,
) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """Return profiled whitened Gram/support for one posterior draw."""
    matrix = _finite_matrix(jacobian, "jacobian")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (matrix.shape[0], matrix.shape[0]) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be finite and align to the Jacobian rows")
    if not np.allclose(cov, cov.T, atol=1.0e-10, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    canonical_whitener = support_whitening(cov, threshold_contract=whitening)
    whitened = canonical_whitener.whiten_jacobian(matrix)
    if nuisance_jacobian is not None:
        nuisance = _finite_matrix(nuisance_jacobian, "nuisance_jacobian")
        if nuisance.shape[0] != matrix.shape[0]:
            raise ValueError("nuisance Jacobian must align to the observation dimension")
        whitened_nuisance = canonical_whitener.whiten_jacobian(nuisance)
        if whitened_nuisance.shape[1]:
            left, singular_nuisance, _right = np.linalg.svd(
                whitened_nuisance, full_matrices=False,
            )
            nuisance_threshold = singular_support.effective_threshold(
                float(singular_nuisance.max(initial=0.0)),
            )
            active = singular_nuisance > nuisance_threshold
            if bool(np.any(active)):
                nuisance_basis = left[:, active]
                # (I - P_nu) Jq after exactly the same covariance whitening.
                whitened = whitened - nuisance_basis @ (nuisance_basis.T @ whitened)
    gram = 0.5 * (whitened.T @ whitened + (whitened.T @ whitened).T)
    if whitened.shape[0] == 0:
        singular = np.zeros((0,), dtype=np.float64)
        right = np.zeros((0, whitened.shape[1]), dtype=np.float64)
    else:
        _left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    threshold = singular_support.effective_threshold(float(singular.max(initial=0.0)))
    rank = int(np.count_nonzero(singular > threshold))
    basis = right[:rank].T if rank else np.zeros((matrix.shape[1], 0), dtype=np.float64)
    condition = float(singular[0] / singular[rank - 1]) if rank else None
    return gram, basis, rank, condition


def _aggregate_support_basis(
    gram: np.ndarray, singular_support: SingularSupportContract,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    singular = np.sqrt(values)
    threshold = singular_support.effective_threshold(float(singular.max(initial=0.0)))
    return vectors[:, singular > threshold]


def _basis_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return 0.0
    projector_left = left @ left.T
    projector_right = right @ right.T
    numerator = float(np.trace(projector_left @ projector_right))
    denominator = math.sqrt(float(left.shape[1] * right.shape[1]))
    return float(np.clip(numerator / max(denominator, 1.0e-15), 0.0, 1.0))


@dataclass(frozen=True)
class RealCSMSupportV1:
    """Target-independent recoverable support with auditable draw aggregation."""

    gram_mean: tuple[tuple[float, ...], ...]
    gram_variance: tuple[tuple[float, ...], ...]
    support_basis: tuple[tuple[float, ...], ...]
    rank_distribution: tuple[int, ...]
    condition_distribution: tuple[float | None, ...]
    recoverable_basis_stability: float
    direction_stability: float
    confidence: float
    forward_sha: str
    jacobian_sha: str
    covariance_sha: str
    operation_descriptor_sha: str
    rf_descriptor_sha: str
    target_independent: bool = True
    nuisance_jacobian_sha: str | None = None
    whitening_contract_sha: str | None = None
    singular_support_contract_sha: str | None = None
    whitening_contract: WhiteningThresholdContract | None = None
    singular_support_contract: SingularSupportContract | None = None
    draw_count: int = 1
    forward_draw_set_sha: str | None = None
    forward_instance_sha: str | None = None
    forward_draw_ids: tuple[str, ...] | None = None
    nuisance_profiled: bool = False
    input_identity_verified: bool = False
    geometry_status: str = "LEGACY_SINGLE_DRAW_UNVERIFIED"
    schema: str = field(default="RealCSMSupportV1", init=False)

    def __post_init__(self) -> None:
        gram = np.asarray(self.gram_mean, dtype=np.float64)
        variance = np.asarray(self.gram_variance, dtype=np.float64)
        basis = np.asarray(self.support_basis, dtype=np.float64)
        if (
            gram.ndim != 2 or gram.shape[0] != gram.shape[1]
            or variance.shape != gram.shape or basis.ndim != 2
            or basis.shape[0] != gram.shape[0]
        ):
            raise ValueError("CSM gram/variance/support basis dimensions disagree")
        if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(variance)) or not np.all(np.isfinite(basis)):
            raise ValueError("CSM values must be finite")
        if not np.allclose(gram, gram.T, atol=1.0e-10) or np.any(variance < 0.0):
            raise ValueError("CSM gram must be symmetric and variance nonnegative")
        if any(type(value) is not int or value < 0 for value in self.rank_distribution):
            raise ValueError("CSM rank distribution must be nonnegative integer values")
        if len(self.rank_distribution) != len(self.condition_distribution) or not self.rank_distribution:
            raise ValueError("CSM rank/condition distributions must be nonempty and aligned")
        for rank, condition in zip(self.rank_distribution, self.condition_distribution):
            if rank == 0:
                if condition is not None:
                    raise ValueError("rank-zero CSM draws require a typed null condition")
            elif condition is None or not math.isfinite(float(condition)) or float(condition) < 1.0:
                raise ValueError("positive-rank CSM draws require a finite condition >= 1")
        if not all(0.0 <= float(value) <= 1.0 for value in (
            self.recoverable_basis_stability, self.direction_stability, self.confidence,
        )):
            raise ValueError("CSM stability/confidence values must lie in [0,1]")
        for name in (
            "forward_sha", "jacobian_sha", "covariance_sha", "operation_descriptor_sha", "rf_descriptor_sha",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "nuisance_jacobian_sha", "whitening_contract_sha", "singular_support_contract_sha",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        for contract_name, hash_name, contract_type in (
            ("whitening_contract", "whitening_contract_sha", WhiteningThresholdContract),
            ("singular_support_contract", "singular_support_contract_sha", SingularSupportContract),
        ):
            contract = getattr(self, contract_name)
            contract_sha = getattr(self, hash_name)
            if (contract is None) != (contract_sha is None):
                raise ValueError("CSM contract object and identity must arrive together")
            if contract is not None:
                if type(contract) is not contract_type:
                    raise TypeError("CSM contract object has an unexpected type")
                contract.validate()
                if contract.sha256() != contract_sha:
                    raise ValueError("CSM contract object differs from its bound identity")
        if type(self.draw_count) is not int or self.draw_count < 1:
            raise ValueError("CSM draw_count must be a positive integer")
        if self.draw_count != len(self.rank_distribution):
            raise ValueError("CSM draw_count must match the rank distribution")
        if (self.forward_draw_set_sha is None) != (self.forward_instance_sha is None):
            raise ValueError("CSM forward draw-set and instance identities must arrive together")
        for name in ("forward_draw_set_sha", "forward_instance_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if self.forward_draw_ids is not None:
            if not self.forward_draw_ids or len(set(self.forward_draw_ids)) != len(self.forward_draw_ids) \
                    or any(type(value) is not str or not value for value in self.forward_draw_ids):
                raise ValueError("CSM forward draw IDs must be unique nonempty strings")
        if self.forward_draw_set_sha is not None and self.forward_draw_ids is None:
            raise ValueError("sealed CSM requires the exact forward draw IDs")
        if self.forward_draw_set_sha is None and self.forward_draw_ids is not None:
            raise ValueError("unsealed CSM cannot claim forward draw IDs")
        if type(self.nuisance_profiled) is not bool or type(self.input_identity_verified) is not bool:
            raise TypeError("CSM profiling/identity flags must be exact booleans")
        if self.input_identity_verified and self.forward_draw_set_sha is None:
            raise ValueError("verified CSM requires a sealed same-forward draw-set identity")
        if self.geometry_status not in CSM_GEOMETRY_STATUSES:
            raise ValueError("CSM geometry status is not registered")
        if self.input_identity_verified:
            if not self.nuisance_profiled:
                raise ValueError("verified CSM requires nuisance-profiled registered draws")
            if self.whitening_contract_sha is None or self.singular_support_contract_sha is None:
                raise ValueError("verified CSM requires bound whitening/singular contracts")
            if self.geometry_status == "LEGACY_SINGLE_DRAW_UNVERIFIED":
                raise ValueError("verified CSM cannot carry a legacy geometry status")
        if self.target_independent is not True:
            raise ValueError("CSM must remain target-independent")

    @classmethod
    def from_jacobian(
        cls, jacobian: np.ndarray, covariance: np.ndarray, *, forward_sha: str, jacobian_sha: str,
        covariance_sha: str, operation_descriptor_sha: str, rf_descriptor_sha: str,
        threshold: float = 1.0e-10,
        nuisance_jacobian: np.ndarray | None = None,
    ) -> "RealCSMSupportV1":
        """Compatibility constructor for one unverified draw.

        New scientific artifacts must use :meth:`from_posterior_draws`; this
        path only parses prior contract fixtures and never treats supplied SHA
        labels as verified array provenance.
        """
        if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
            raise ValueError("singular support threshold must be positive")
        whitening = WhiteningThresholdContract()
        singular = SingularSupportContract(absolute=float(threshold), relative=0.0)
        gram, basis, rank, condition = _profiled_draw_geometry(
            jacobian, covariance, nuisance_jacobian, whitening, singular,
        )
        return cls(
            gram_mean=tuple(tuple(float(value) for value in row) for row in gram),
            gram_variance=tuple(tuple(0.0 for _ in row) for row in gram),
            support_basis=tuple(tuple(float(value) for value in row) for row in basis),
            rank_distribution=(rank,),
            condition_distribution=(condition,),
            recoverable_basis_stability=1.0 if rank else 0.0,
            direction_stability=1.0 if rank else 0.0,
            confidence=1.0 if rank else 0.0,
            forward_sha=forward_sha,
            jacobian_sha=jacobian_sha,
            covariance_sha=covariance_sha,
            operation_descriptor_sha=operation_descriptor_sha,
            rf_descriptor_sha=rf_descriptor_sha,
            nuisance_jacobian_sha=(
                array_identity_sha256(nuisance_jacobian)
                if nuisance_jacobian is not None else None
            ),
            whitening_contract_sha=whitening.sha256(),
            singular_support_contract_sha=singular.sha256(),
            whitening_contract=whitening,
            singular_support_contract=singular,
            nuisance_profiled=nuisance_jacobian is not None,
            forward_draw_set_sha=None,
            forward_instance_sha=None,
            forward_draw_ids=None,
            geometry_status="LEGACY_SINGLE_DRAW_UNVERIFIED",
        )

    @classmethod
    def from_posterior_draws(
        cls,
        jacobian_draws: Sequence[np.ndarray],
        covariance_draws: Sequence[np.ndarray],
        *,
        forward_sha: str,
        operation_descriptor_sha: str,
        rf_descriptor_sha: str,
        nuisance_jacobian_draws: Sequence[np.ndarray] | None,
        whitening: WhiteningThresholdContract | None = None,
        singular_support: SingularSupportContract | None = None,
        expected_jacobian_sha: str | None = None,
        expected_covariance_sha: str | None = None,
        expected_nuisance_jacobian_sha: str | None = None,
        forward_draw_set: BoundTrackBForwardDrawSetV1 | None = None,
    ) -> "RealCSMSupportV1":
        """Build a verified nuisance-profiled CSM from registered posterior draws."""
        draws = tuple(np.asarray(value, dtype=np.float64) for value in jacobian_draws)
        covariances = tuple(np.asarray(value, dtype=np.float64) for value in covariance_draws)
        nuisances = (
            tuple(np.asarray(value, dtype=np.float64) for value in nuisance_jacobian_draws)
            if nuisance_jacobian_draws is not None else None
        )
        if not draws or len(draws) != len(covariances):
            raise ValueError("posterior Jacobian/covariance draws must be nonempty and aligned")
        if nuisances is None or len(nuisances) != len(draws):
            raise ValueError("verified CSM requires one registered nuisance Jacobian per draw")
        sealed_forward = forward_draw_set
        if sealed_forward is not None:
            if type(sealed_forward) is not BoundTrackBForwardDrawSetV1:
                raise TypeError("forward_draw_set must be BoundTrackBForwardDrawSetV1")
            sealed_forward.validate()
            sealed_forward.assert_authoritative()
            if forward_sha != sealed_forward.forward_sha:
                raise ValueError("forward_sha does not match the sealed forward instance")
            if sealed_forward.operation_descriptor_sha != operation_descriptor_sha \
                    or sealed_forward.rf_descriptor_sha != rf_descriptor_sha:
                raise ValueError("CSM descriptor identities do not match the sealed forward draw set")
            if tuple(array_identity_sha256(value) for value in draws) != sealed_forward.jacobian_draw_shas:
                raise ValueError("Jacobian arrays do not match the sealed same-forward draw set")
            if tuple(array_identity_sha256(value) for value in covariances) != sealed_forward.covariance_draw_shas:
                raise ValueError("covariance arrays do not match the sealed same-forward draw set")
            if tuple(array_identity_sha256(value) for value in nuisances) != sealed_forward.nuisance_jacobian_draw_shas:
                raise ValueError("nuisance Jacobian arrays do not match the sealed same-forward draw set")
        else:
            # Retain the old constructor only as an explicitly unverified
            # compatibility path.  It cannot pass the proposal gate.
            if not isinstance(forward_sha, str):
                raise TypeError("legacy forward_sha must be a string")
        if any(draw.shape != draws[0].shape for draw in draws):
            raise ValueError("posterior Jacobian draws must share one shape")
        active_whitening = whitening or WhiteningThresholdContract()
        active_singular = singular_support or SingularSupportContract()
        active_whitening.validate()
        active_singular.validate()
        jacobian_draw_shas = tuple(array_identity_sha256(draw) for draw in draws)
        covariance_draw_shas = tuple(array_identity_sha256(draw) for draw in covariances)
        nuisance_draw_shas = tuple(array_identity_sha256(draw) for draw in nuisances)
        jacobian_sha = canonical_sha256({
            "schema": "TrackBPosteriorJacobianDrawSetV1", "draw_shas": list(jacobian_draw_shas),
        })
        covariance_sha = canonical_sha256({
            "schema": "TrackBPosteriorCovarianceDrawSetV1", "draw_shas": list(covariance_draw_shas),
        })
        nuisance_sha = canonical_sha256({
            "schema": "TrackBPosteriorNuisanceJacobianDrawSetV1", "draw_shas": list(nuisance_draw_shas),
        })
        for supplied, actual, name in (
            (expected_jacobian_sha, jacobian_sha, "expected_jacobian_sha"),
            (expected_covariance_sha, covariance_sha, "expected_covariance_sha"),
            (expected_nuisance_jacobian_sha, nuisance_sha, "expected_nuisance_jacobian_sha"),
        ):
            if supplied is not None:
                require_sha256(supplied, name)
                if supplied != actual:
                    raise ValueError(f"{name} does not match the supplied posterior-draw inputs")
        geometries = tuple(
            _profiled_draw_geometry(draw, cov, nuisance, active_whitening, active_singular)
            for draw, cov, nuisance in zip(draws, covariances, nuisances)
        )
        grams = np.stack([entry[0] for entry in geometries], axis=0)
        gram_mean = np.mean(grams, axis=0)
        gram_variance = np.var(grams, axis=0)
        basis = _aggregate_support_basis(gram_mean, active_singular)
        ranks = tuple(int(entry[2]) for entry in geometries)
        conditions = tuple(entry[3] for entry in geometries)
        overlaps = tuple(_basis_overlap(basis, entry[1]) for entry in geometries)
        support_fraction = float(np.mean([rank > 0 for rank in ranks]))
        basis_stability = float(np.mean(overlaps)) if overlaps else 0.0
        confidence = float(np.clip(support_fraction * basis_stability, 0.0, 1.0))
        status = (
            "POSTERIOR_DRAW_AGGREGATED_NUISANCE_PROFILED"
            if basis.shape[1] else "POSTERIOR_DRAW_AGGREGATED_NO_RECOVERABLE_SUPPORT"
        )
        return cls(
            gram_mean=tuple(tuple(float(value) for value in row) for row in gram_mean),
            gram_variance=tuple(tuple(float(value) for value in row) for row in gram_variance),
            support_basis=tuple(tuple(float(value) for value in row) for row in basis),
            rank_distribution=ranks,
            condition_distribution=conditions,
            recoverable_basis_stability=basis_stability,
            direction_stability=basis_stability,
            confidence=confidence,
            forward_sha=forward_sha,
            jacobian_sha=jacobian_sha,
            covariance_sha=covariance_sha,
            operation_descriptor_sha=operation_descriptor_sha,
            rf_descriptor_sha=rf_descriptor_sha,
            nuisance_jacobian_sha=nuisance_sha,
            whitening_contract_sha=active_whitening.sha256(),
            singular_support_contract_sha=active_singular.sha256(),
            whitening_contract=active_whitening,
            singular_support_contract=active_singular,
            draw_count=len(draws),
            forward_draw_set_sha=(None if sealed_forward is None else sealed_forward.sha256()),
            forward_instance_sha=(None if sealed_forward is None else sealed_forward.forward_instance_sha),
            forward_draw_ids=(None if sealed_forward is None else sealed_forward.draw_ids),
            nuisance_profiled=True,
            input_identity_verified=sealed_forward is not None,
            geometry_status=(status if sealed_forward is not None else "LEGACY_SINGLE_DRAW_UNVERIFIED"),
        )

    def project(self, logical_vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Euclidean compatibility projector; metric projection lives in ``real_rsm``."""
        value = np.asarray(logical_vector, dtype=np.float64)
        basis = np.asarray(self.support_basis, dtype=np.float64)
        if value.ndim != 1 or value.shape[0] != basis.shape[0] or not np.all(np.isfinite(value)):
            raise ValueError("logical vector does not align to CSM support")
        return basis @ (basis.T @ value) if basis.shape[1] else np.zeros_like(value)

    @property
    def forward_draw_ids_sha(self) -> str | None:
        if self.forward_draw_ids is None:
            return None
        return canonical_sha256({
            "schema": "TrackBForwardDrawIDsV1", "draw_ids": list(self.forward_draw_ids),
        })

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete CSM geometry without adding authority."""
        self.__post_init__()
        return {
            "schema": self.schema,
            "gram_mean": [list(row) for row in self.gram_mean],
            "gram_variance": [list(row) for row in self.gram_variance],
            "support_basis": [list(row) for row in self.support_basis],
            "rank_distribution": list(self.rank_distribution),
            "condition_distribution": list(self.condition_distribution),
            "recoverable_basis_stability": self.recoverable_basis_stability,
            "direction_stability": self.direction_stability,
            "confidence": self.confidence,
            "forward_sha": self.forward_sha,
            "jacobian_sha": self.jacobian_sha,
            "covariance_sha": self.covariance_sha,
            "operation_descriptor_sha": self.operation_descriptor_sha,
            "rf_descriptor_sha": self.rf_descriptor_sha,
            "target_independent": True,
            "nuisance_jacobian_sha": self.nuisance_jacobian_sha,
            "whitening_contract_sha": self.whitening_contract_sha,
            "singular_support_contract_sha": self.singular_support_contract_sha,
            "whitening_contract": (
                None if self.whitening_contract is None else self.whitening_contract.to_dict()
            ),
            "singular_support_contract": (
                None if self.singular_support_contract is None
                else self.singular_support_contract.to_dict()
            ),
            "draw_count": self.draw_count,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "forward_instance_sha": self.forward_instance_sha,
            "forward_draw_ids": (
                None if self.forward_draw_ids is None else list(self.forward_draw_ids)
            ),
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
            "nuisance_profiled": self.nuisance_profiled,
            "input_identity_verified": self.input_identity_verified,
            "geometry_status": self.geometry_status,
        }

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "gram_mean": [list(row) for row in self.gram_mean],
            "gram_variance": [list(row) for row in self.gram_variance],
            "support_basis": [list(row) for row in self.support_basis],
            "rank_distribution": list(self.rank_distribution),
            "condition_distribution": list(self.condition_distribution),
            "recoverable_basis_stability": self.recoverable_basis_stability,
            "direction_stability": self.direction_stability,
            "confidence": self.confidence,
            "forward_sha": self.forward_sha,
            "jacobian_sha": self.jacobian_sha,
            "covariance_sha": self.covariance_sha,
            "operation_descriptor_sha": self.operation_descriptor_sha,
            "rf_descriptor_sha": self.rf_descriptor_sha,
            "nuisance_jacobian_sha": self.nuisance_jacobian_sha,
            "whitening_contract_sha": self.whitening_contract_sha,
            "singular_support_contract_sha": self.singular_support_contract_sha,
            "whitening_contract": (
                None if self.whitening_contract is None else self.whitening_contract.to_dict()
            ),
            "singular_support_contract": (
                None if self.singular_support_contract is None
                else self.singular_support_contract.to_dict()
            ),
            "draw_count": self.draw_count,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "forward_instance_sha": self.forward_instance_sha,
            "forward_draw_ids": (None if self.forward_draw_ids is None else list(self.forward_draw_ids)),
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
            "nuisance_profiled": self.nuisance_profiled,
            "input_identity_verified": self.input_identity_verified,
            "geometry_status": self.geometry_status,
            "target_independent": True,
        })


__all__ = [
    "CSM_GEOMETRY_STATUSES", "RealCSMSupportV1", "array_identity_sha256",
]
