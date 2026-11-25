"""V9 rho8 contract assembly (charter sections 8, 12.6, 13.4).

Provides the canonical eight-view bundle builder and the hidden-operation masking guard:
when a masked-operation task hides operations, the estimator used for reconstruction must
have the hidden cells zeroed so no hidden count leaks into the view. If a view was
built from hidden operations and then used to predict those same hidden cells, the
leakage test must degrade (fail closed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .contracts import (
    RHO8_CANONICAL_KEYS,
    V9RhoEightViewBundleV1,
    V9RhoViewProvenanceV1,
)
from .rho8_estimators import V9OperationEstimatorResultV1
from .rho8_maps import build_rho8_maps
from .rho8_reconstruction import reconstruct_rho8_view_bundle


def verify_canonical_keys(keys: Sequence[str]) -> None:
    """Assert a key set is exactly the canonical eight rho keys (charter 9.4)."""
    key_tuple = tuple(keys)
    if key_tuple != RHO8_CANONICAL_KEYS:
        raise ValueError(
            f"canonical rho8 keys mismatch:\n  expected={RHO8_CANONICAL_KEYS}\n  got={key_tuple}"
        )


def mask_estimator_hidden_operations(
    estimator: V9OperationEstimatorResultV1,
    operation_mask: np.ndarray,
) -> V9OperationEstimatorResultV1:
    """Return an estimator with hidden operations zeroed (no leak into the view).

    ``operation_mask`` must be boolean with True = visible (kept). Hidden operations
    are zeroed in ``x`` and ``signed_x`` and marked invalid.
    """
    mask = np.asarray(operation_mask, dtype=bool).reshape(-1)
    if mask.size != estimator.x.size:
        raise ValueError(
            f"operation_mask size {mask.size} does not match estimator size {estimator.x.size}"
        )
    x = np.where(mask, estimator.x, 0.0)
    signed = np.where(mask, estimator.signed_x, 0.0)
    zero_signal = estimator.zero_signal_mask | (~mask)
    zero_read = estimator.zero_read_mask | (~mask)
    return V9OperationEstimatorResultV1(
        estimator_key=estimator.estimator_key,
        system_type=estimator.system_type,
        x=x,
        signed_x=signed,
        zero_signal_mask=zero_signal,
        zero_read_mask=zero_read,
        negative_fraction=estimator.negative_fraction,
    )


def build_rho8_view_bundle(
    system_type: str,
    *,
    sample_id: str,
    signal_counts: np.ndarray,
    idler_counts: np.ndarray,
    coin_counts: np.ndarray,
    read_count: np.ndarray | float,
    operation_mask: Optional[np.ndarray] = None,
    retrieval_map_path: Optional[str] = None,
    idler_csv_path: Optional[str] = None,
    basis_config_path: Optional[str] = None,
    count_source_sha256: str = "",
) -> V9RhoEightViewBundleV1:
    """End-to-end builder: maps + estimators + masked reconstruction.

    When ``operation_mask`` is given (True = visible), hidden operations are zeroed in
    the reconstruction estimators so no hidden count leaks into any view. The returned
    bundle records the canonical keys and provenance.
    """
    from .rho8_estimators import build_all_estimators

    estimators = build_all_estimators(
        system_type,
        signal_counts=signal_counts,
        idler_counts=idler_counts,
        coin_counts=coin_counts,
        read_count=read_count,
    )
    if operation_mask is not None:
        estimators = {
            key: mask_estimator_hidden_operations(est, operation_mask)
            for key, est in estimators.items()
        }
    maps = build_rho8_maps(
        system_type,
        retrieval_map_path=retrieval_map_path,
        idler_csv_path=idler_csv_path,
    )
    bundle = reconstruct_rho8_view_bundle(
        system_type,
        sample_id=sample_id,
        estimators=estimators,
        maps=maps,
        basis_config_path=basis_config_path,
        count_source_sha256=count_source_sha256,
    )
    verify_canonical_keys(list(bundle.views.keys()))
    return bundle


def bundle_view_was_built_from_hidden_operations(
    bundle: V9RhoEightViewBundleV1,
    operation_mask: np.ndarray,
) -> bool:
    """Leakage detector: did any view consume hidden-operation counts?

    Returns True when a view's invalid mask leaves ANY hidden operation un-zeroed (i.e.,
    a hidden cell was consumed). Visible operations may LEGITIMATELY be invalid too (a
    visible op with zero detected signal contributes nothing and is zeroed); the
    fail-closed requirement is that every HIDDEN operation is zeroed before
    reconstruction, so a view is never used to predict hidden operations it saw.
    """
    mask = np.asarray(operation_mask, dtype=bool).reshape(-1)
    hidden = ~mask
    leaked = False
    for key, view in bundle.views.items():
        # invalid_operation_mask records which operations were zeroed (True = invalid/zeroed).
        invalid = np.asarray(view.get("invalid_operation_mask", []), dtype=bool)
        if invalid.size != hidden.size:
            return True
        if not np.all(invalid[hidden]):
            leaked = True
    return leaked
