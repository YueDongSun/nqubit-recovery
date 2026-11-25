"""Operation-point diagnostic maps shared by the 3Q and 4Q data UIs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


POINTWISE_METRICS = (
    "efficiency", "background", "leakage", "variance",
    "phase_residual", "amplitude_residual", "calibration_confidence", "model_uncertainty",
)


def operation_point_maps(system_type: str, values: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Return 15x5 maps for 3Q or four 16x4 maps for 4Q without reindexing."""
    expected = 75 if system_type == "3qubit" else 256
    result: Dict[str, np.ndarray] = {}
    for name in POINTWISE_METRICS:
        raw = values.get(name)
        vector = np.full(expected, np.nan, dtype=float) if raw is None else _fit(raw, expected)
        result[name] = vector.reshape(15, 5) if system_type == "3qubit" else vector.reshape(4, 16, 4)
    return result


def draw_operation_point_diagnostics(figure, system_type: str, values: Mapping[str, Any], *, cycle_index: int = 0) -> None:
    """Draw the requested 3Q map or one of the four 4Q operation-map pages."""
    figure.clear()
    maps = operation_point_maps(system_type, values)
    for index, name in enumerate(POINTWISE_METRICS, start=1):
        axis = figure.add_subplot(2, 4, index)
        matrix = maps[name] if system_type == "3qubit" else maps[name][int(np.clip(cycle_index, 0, 3))]
        image = axis.imshow(matrix, aspect="auto", origin="upper", cmap="viridis")
        axis.set_title(name.replace("_", " "), fontsize=8)
        axis.set_xlabel("group", fontsize=7)
        axis.set_ylabel("row", fontsize=7)
        axis.tick_params(labelsize=6)
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    title = "3Q operation-point map (15×5)" if system_type == "3qubit" else f"4Q operation-point map — cycle {cycle_index + 1} (16×4)"
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))


def pointwise_values_from_shadow(shadow_result: Optional[Mapping[str, Any]], calibration: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Normalize persisted shadow diagnostics into plotting vectors.

    Missing fields stay NaN; no visualizer turns unknown calibration into zero.
    """
    shadow = shadow_result or {}
    count = 75 if str(shadow.get("system_type", "3qubit")) == "3qubit" else 256
    variance = shadow.get("estimator_variance")
    diagnostics = shadow.get("point_diagnostics", {}) or {}
    masks = (shadow.get("diagnostics", {}) or {}).get("point_response_missing_masks", [])
    confidence = np.full(count, np.nan)
    for index, mask in enumerate(masks[:count]):
        cells = (mask or {}).get("cells", [])
        if cells:
            confidence[index] = 1.0 - float(np.mean([bool(item.get("efficiency")) for item in cells]))
    return {
        "variance": _fit(variance, count) if variance is not None else np.full(count, np.nan),
        "calibration_confidence": confidence,
        "efficiency": _calibration_field(calibration, "operation_effective_efficiency", count),
        "background": _calibration_field(calibration, "operation_background_rate", count),
        "leakage": _calibration_field(calibration, "operation_leakage_rate", count),
        "phase_residual": _scalar_vector(diagnostics.get("phase_residual"), count),
        "amplitude_residual": _scalar_vector(diagnostics.get("amplitude_residual"), count),
        "model_uncertainty": _scalar_vector(diagnostics.get("model_uncertainty"), count),
    }


def _fit(values: Any, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    output = np.full(size, np.nan, dtype=float)
    output[:min(size, array.size)] = array[:size]
    return output


def _calibration_field(calibration: Optional[Mapping[str, Any]], name: str, size: int) -> np.ndarray:
    values = (calibration or {}).get(name)
    return np.full(size, np.nan) if values is None else _fit(values, size)


def _scalar_vector(values: Any, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan)
    try:
        return _fit(values, size)
    except ValueError:
        return _fit([np.nan if value is None or len(value) == 0 else float(np.mean(value)) for value in values], size)
