"""One target-free nuisance projection on the registered whitened support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import sha256_of
from .csm_contract import SingularSupportContract


NUISANCE_PROJECTION_SCHEMA = "V10CSMNuisanceProjectionV1"


@dataclass(frozen=True)
class CSMNuisanceProjection:
    """Runtime low-rank operator; no residual or target is retained."""

    basis: np.ndarray
    singular_values: np.ndarray
    support_threshold: float
    singular_support_sha: str
    nuisance_jacobian_sha: str
    projection_sha: str

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def project(self, values: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(values):
            raise ValueError("nuisance projection inputs must be real")
        value = np.asarray(values, dtype=np.float64)
        if value.ndim not in (1, 2) or value.shape[0] != self.basis.shape[0]:
            raise ValueError("nuisance projection requires the same whitened row support")
        if not np.all(np.isfinite(value)):
            raise ValueError("nuisance projection inputs must be finite")
        return value - self.basis @ (self.basis.T @ value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": NUISANCE_PROJECTION_SCHEMA,
            "whitened_rows": int(self.basis.shape[0]),
            "nuisance_rank": self.rank,
            "nuisance_singular_values": self.singular_values.tolist(),
            "singular_support_threshold": self.support_threshold,
            "singular_support_sha": self.singular_support_sha,
            "nuisance_jacobian_sha": self.nuisance_jacobian_sha,
            "projection_sha": self.projection_sha,
            "target_independent": True,
        }


def build_nuisance_projection(
    whitened_nuisance_jacobian: np.ndarray,
    *,
    singular_support: SingularSupportContract,
) -> CSMNuisanceProjection:
    """Build ``I-U_nu U_nu.T`` without materializing the dense projector."""
    singular_support.validate()
    if np.iscomplexobj(whitened_nuisance_jacobian):
        raise ValueError("whitened nuisance Jacobian must be real")
    nuisance = np.asarray(whitened_nuisance_jacobian, dtype=np.float64)
    if nuisance.ndim != 2 or not np.all(np.isfinite(nuisance)):
        raise ValueError("whitened nuisance Jacobian must be a finite matrix")
    left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    threshold = singular_support.effective_threshold(
        float(singular.max(initial=0.0)))
    basis = left[:, singular > threshold].copy()
    # Canonical signs avoid arbitrary SVD sign choices in the construction record.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    nuisance_sha = sha256_of({"shape": list(nuisance.shape), "values": nuisance.tolist()})
    support_sha = singular_support.sha256()
    projection_sha = sha256_of({
        "schema": NUISANCE_PROJECTION_SCHEMA,
        "nuisance_jacobian_sha": nuisance_sha,
        "singular_support_sha": support_sha,
        "rank": int(basis.shape[1]),
        "basis": basis.tolist(),
    })
    basis.setflags(write=False)
    singular.setflags(write=False)
    return CSMNuisanceProjection(
        basis=basis, singular_values=singular,
        support_threshold=threshold, singular_support_sha=support_sha,
        nuisance_jacobian_sha=nuisance_sha, projection_sha=projection_sha,
    )
