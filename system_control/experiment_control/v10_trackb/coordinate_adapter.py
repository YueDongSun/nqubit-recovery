"""Versioned conversion between Track B log-amplitudes and materializer coordinates.

Track B logical amplitudes are Helmert coordinates of ``log(a)``.  The
existing feedback materializer uses Aitchison coordinates of ``log(a**2)``.
They have the same shape and gauge but differ by an exact factor of two.  This
adapter makes that physical chart change explicit and hash-bound.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .proposal_reader import ProposalValidationError, canonical_json_bytes, sha256_bytes


COORDINATE_ADAPTER_SCHEMA = "TrackBLogAmplitudeToMaterializerLogProbabilityV1"
LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE = 2.0


def _coordinate_count(system_type: str) -> int:
    values = {"3qubit": 2, "4qubit": 3}
    try:
        return values[system_type]
    except KeyError as error:
        raise ProposalValidationError("coordinate adapter system_type is invalid") from error


def coordinate_adapter_sha(system_type: str) -> str:
    count = _coordinate_count(system_type)
    return sha256_bytes(canonical_json_bytes({
        "schema": COORDINATE_ADAPTER_SCHEMA,
        "system_type": system_type,
        "logical_amplitude_chart": "HELMERT_RELATIVE_LOG_AMPLITUDE",
        "materializer_amplitude_chart": "HELMERT_AITCHISON_LOG_PROBABILITY",
        "log_amplitude_to_log_probability_scale": LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE,
        "phase_chart": "HELMERT_RELATIVE_PHASE_RADIANS",
        "coordinate_count": count,
    }))


def materializer_coordinate_layout_sha(system_type: str) -> str:
    count = _coordinate_count(system_type)
    return sha256_bytes(canonical_json_bytes({
        "schema": "TrackBMaterializerCoordinateLayoutV1",
        "system_type": system_type,
        "coordinate_layout": "GLOBAL_W_PROJECTIVE_AITCHISON_LOG_PROBABILITY_V1",
        "amplitude_chart": "HELMERT_AITCHISON_LOG_PROBABILITY",
        "phase_chart": "HELMERT_RELATIVE_PHASE_RADIANS",
        "coordinate_count": count,
    }))


def trackb_to_materializer_coordinates(
    system_type: str, logical_action: Sequence[float],
) -> Tuple[float, ...]:
    """Map grouped Track B ``[log(a), phase]`` to materializer coordinates."""
    count = _coordinate_count(system_type)
    values = np.asarray(logical_action, dtype=np.float64)
    if values.shape != (2 * count,) or not np.all(np.isfinite(values)):
        raise ProposalValidationError("Track B logical action has wrong finite dimension")
    return tuple(float(value) for value in np.concatenate((
        LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE * values[:count], values[count:],
    )))


def materializer_to_trackb_coordinates(
    system_type: str, materializer_action: Sequence[float],
) -> Tuple[float, ...]:
    """Invert the exact chart conversion for post-materialization auditing."""
    count = _coordinate_count(system_type)
    values = np.asarray(materializer_action, dtype=np.float64)
    if values.shape != (2 * count,) or not np.all(np.isfinite(values)):
        raise ProposalValidationError("materializer action has wrong finite dimension")
    return tuple(float(value) for value in np.concatenate((
        values[:count] / LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE, values[count:],
    )))


def trackb_to_materializer_action_payload(
    system_type: str, logical_action: Sequence[float],
) -> dict[str, object]:
    """Build the legacy action mapping only after the chart conversion."""
    count = _coordinate_count(system_type)
    converted = trackb_to_materializer_coordinates(system_type, logical_action)
    return {
        "schema": "GlobalWProjectiveActionV1",
        "system_type": system_type,
        "amplitude_coordinates": list(converted[:count]),
        "phase_coordinates": list(converted[count:]),
        "brightness_delta": 0.0,
        "read_action": 0.0,
        "local_residual": 0.0,
        "cycle_residual": 0.0,
        "coordinate_layout_sha": materializer_coordinate_layout_sha(system_type),
        "coordinate_adapter_sha": coordinate_adapter_sha(system_type),
    }


__all__ = [
    "COORDINATE_ADAPTER_SCHEMA",
    "LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE",
    "coordinate_adapter_sha",
    "materializer_coordinate_layout_sha",
    "materializer_to_trackb_coordinates",
    "trackb_to_materializer_action_payload",
    "trackb_to_materializer_coordinates",
]
