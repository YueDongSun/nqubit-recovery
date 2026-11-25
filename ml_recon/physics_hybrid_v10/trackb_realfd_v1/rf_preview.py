"""Pure RF preview provider for Track B post-damping candidate evaluation.

The provider wraps the immutable numerical mapping core only.  It has no
filesystem, model-loading, sequence, uploader, or system-control dependency;
the model-side selection API can therefore not execute an arbitrary callback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .coordinate_abi import (
    RFCommandDeltaV1,
    RFMaterializationPreviewV1,
    RecoverableLogicalActionV1,
    logical_action_payload_sha,
)
from .rf_mapping_core import RegisteredRFMappingCoreV1, solve_registered_rf_mapping


DEFAULT_SUPPORT_LEAKAGE_TOLERANCE = 1.0e-8
DEFAULT_LOGICAL_ERROR_TOLERANCE = 1.0e-8
DEFAULT_MINIMUM_DIRECTION_COSINE = 0.999999


def preview_registered_rf_action(
    core: RegisteredRFMappingCoreV1,
    action: RecoverableLogicalActionV1,
    *,
    support_leakage_tolerance: float = DEFAULT_SUPPORT_LEAKAGE_TOLERANCE,
    logical_error_tolerance: float = DEFAULT_LOGICAL_ERROR_TOLERANCE,
    minimum_direction_cosine: float = DEFAULT_MINIMUM_DIRECTION_COSINE,
) -> RFMaterializationPreviewV1:
    """Run one pure RF preview against an exact immutable mapping core.

    This is deliberately a module-level function.  The FD path accepts the
    exact core type and never dispatches through a caller-supplied object
    method, so a subclass cannot inject filesystem, network, or uploader work.
    """
    if type(core) is not RegisteredRFMappingCoreV1:
        raise TypeError("RF preview requires the exact registered mapping-core type")
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in (support_leakage_tolerance, logical_error_tolerance)
    ):
        raise ValueError("RF preview tolerances must be finite and nonnegative")
    if not math.isfinite(float(minimum_direction_cosine)) \
            or not -1.0 <= float(minimum_direction_cosine) <= 1.0:
        raise ValueError("RF preview direction threshold must lie in [-1,1]")
    if action.ideal_action.system_type != core.system_type:
        raise ValueError("RF preview action/core system types disagree")
    if action.ideal_action.coordinate_layout_sha != core.logical_coordinate_layout_sha:
        raise ValueError("RF preview action/core coordinate layouts disagree")
    if not action.projection_verified:
        raise ValueError("RF preview requires an actually projected recoverable action")
    result = solve_registered_rf_mapping(core, action.grouped_coordinates)
    command = np.asarray(result.command_delta, dtype=np.float64)
    lower = np.asarray(core.lower_bounds, dtype=np.float64)
    upper = np.asarray(core.upper_bounds, dtype=np.float64)
    bounds_valid = bool(
        np.all(command >= lower - 1.0e-12) and np.all(command <= upper + 1.0e-12)
    )
    rf_delta = RFCommandDeltaV1(
        system_type=core.system_type,
        ordered_tone_ids=tuple(item.coordinate_id for item in core.rf_coordinates),
        amplitude_delta=result.amplitude_delta,
        phase_delta_rad=result.phase_delta_rad,
        other_delta=result.other_delta,
        rf_coordinate_layout_sha=core.rf_coordinate_layout_sha,
        source_logical_action_sha=action.sha256(),
        logical_action_payload_sha=logical_action_payload_sha(
            core.system_type, action.amplitude_coordinates, action.phase_coordinates,
        ),
    )
    accepted = bool(
        result.accepted
        and bounds_valid
        and result.support_leakage_norm <= support_leakage_tolerance
        and result.requested_to_materialized_logical_error <= logical_error_tolerance
        and result.direction_cosine >= minimum_direction_cosine
        and core.phase_wrap_valid(command)
    )
    return RFMaterializationPreviewV1(
        requested_rf_delta=rf_delta,
        support_leakage_norm=float(result.support_leakage_norm),
        requested_to_materialized_logical_error=float(result.requested_to_materialized_logical_error),
        direction_cosine=float(result.direction_cosine),
        bounds_valid=bounds_valid,
        phase_wrap_valid=core.phase_wrap_valid(command),
        actual_loaded_discrepancy=None,
        accepted_for_external_review=accepted,
        mapping_core_sha=core.sha256(),
    )


@dataclass(frozen=True)
class RegisteredRFPreviewProviderV1:
    """Recompute a Track B RF preview through one validated pure core."""

    core: RegisteredRFMappingCoreV1
    support_leakage_tolerance: float = DEFAULT_SUPPORT_LEAKAGE_TOLERANCE
    logical_error_tolerance: float = DEFAULT_LOGICAL_ERROR_TOLERANCE
    minimum_direction_cosine: float = DEFAULT_MINIMUM_DIRECTION_COSINE

    def __post_init__(self) -> None:
        if type(self.core) is not RegisteredRFMappingCoreV1:
            raise TypeError("RF preview provider requires the exact registered mapping-core type")
        self.core.__post_init__()
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.support_leakage_tolerance, self.logical_error_tolerance)
        ):
            raise ValueError("RF preview tolerances must be finite and nonnegative")
        if not math.isfinite(float(self.minimum_direction_cosine)) \
                or not -1.0 <= float(self.minimum_direction_cosine) <= 1.0:
            raise ValueError("RF preview direction threshold must lie in [-1,1]")

    def preview(self, action: RecoverableLogicalActionV1) -> RFMaterializationPreviewV1:
        """Return a deterministic preview for one already-projected logical action."""
        return preview_registered_rf_action(
            self.core, action,
            support_leakage_tolerance=self.support_leakage_tolerance,
            logical_error_tolerance=self.logical_error_tolerance,
            minimum_direction_cosine=self.minimum_direction_cosine,
        )


__all__ = [
    "DEFAULT_LOGICAL_ERROR_TOLERANCE", "DEFAULT_MINIMUM_DIRECTION_COSINE",
    "DEFAULT_SUPPORT_LEAKAGE_TOLERANCE", "RegisteredRFPreviewProviderV1",
    "preview_registered_rf_action",
]
