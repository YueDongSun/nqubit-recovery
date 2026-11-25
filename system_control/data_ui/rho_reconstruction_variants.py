"""Extra rho reconstruction variants for slow Data UI diagnostics."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


SYSTEM_CELLS = {
    "3qubit": 75,
    "4qubit": 256,
}

_DATA_UI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DATA_UI_DIR.parents[1]
_QUBIT_RECON_DIR = _REPO_ROOT / "system_control" / "qubit_recon_files"
_MATRIX_DIR = _QUBIT_RECON_DIR / "matrix"


def counts_clip_from_data(data: Optional[Mapping[str, Any]], n_cells: int) -> np.ndarray:
    values = _segment_values(data, n_cells)
    return np.maximum(values, 0.0)


def counts_shift_to_zero_from_data(
    data: Optional[Mapping[str, Any]],
    n_cells: int,
) -> tuple[np.ndarray, float]:
    values = _segment_values(data, n_cells)
    if values.size == 0:
        return values, 0.0
    offset = max(0.0, -float(np.min(values)))
    return np.maximum(values + offset, 0.0), offset


def coin_idler_counts_from_data(
    data: Optional[Mapping[str, Any]],
    n_cells: int,
) -> tuple[np.ndarray, int]:
    """Build the per-cell conditional coincidence estimator ``coin/idler``."""

    data = data or {}
    coin = _sized_values(data.get("coin_bins"), n_cells)
    idler = _sized_values(data.get("idler_bins"), n_cells)
    values = np.zeros(int(n_cells), dtype=float)
    valid = idler > 0.0
    np.divide(coin, idler, out=values, where=valid)
    values[~np.isfinite(values)] = 0.0
    return np.maximum(values, 0.0), int(np.sum(~valid))


def coin_signal_counts_from_data(
    data: Optional[Mapping[str, Any]],
    n_cells: int,
) -> tuple[np.ndarray, int]:
    """Build the per-cell conditional coincidence estimator ``coin/signal``."""

    data = data or {}
    coin = _sized_values(data.get("coin_bins"), n_cells)
    signal = _sized_values(data.get("signal_bins"), n_cells)
    values = np.zeros(int(n_cells), dtype=float)
    valid = signal > 0.0
    np.divide(coin, signal, out=values, where=valid)
    values[~np.isfinite(values)] = 0.0
    return np.maximum(values, 0.0), int(np.sum(~valid))


def build_diagnostic_rho_variants(
    system_type: str,
    data: Optional[Mapping[str, Any]],
    *,
    calibrated_module: Any = None,
) -> dict[str, Any]:
    """Build slow diagnostic-only rho variants.

    The main Data UI update already computes raw/idler/retrieval with clipped
    counts.  This helper is intentionally called only from the 30s diagnosis
    refresh or the explicit ML Correction action.
    """

    system_type = str(system_type)
    if system_type not in SYSTEM_CELLS:
        raise ValueError(f"unsupported system_type: {system_type}")

    diagnostics: dict[str, Any] = {
        "system_type": system_type,
        "available": False,
        "errors": {},
    }
    variants: dict[str, np.ndarray] = {}
    if not data or data.get("segment_values") is None:
        diagnostics["message"] = "segment_values unavailable"
        return {"rho_variants": variants, "diagnostics": diagnostics}

    module = calibrated_module or _load_calibrated_module()
    if module is None:
        diagnostics["message"] = "calibrated reconstruction unavailable"
        return {"rho_variants": variants, "diagnostics": diagnostics}

    n_cells = SYSTEM_CELLS[system_type]
    basis_path = _MATRIX_DIR / f"basis_config_{system_type}.json"
    if not basis_path.exists():
        diagnostics["message"] = f"basis config missing: {basis_path}"
        return {"rho_variants": variants, "diagnostics": diagnostics}

    clipped_counts = counts_clip_from_data(data, n_cells)
    shifted_counts, shift_offset = counts_shift_to_zero_from_data(data, n_cells)
    has_coin_signal_inputs = data.get("coin_bins") is not None and data.get("signal_bins") is not None
    coin_signal_counts, zero_signal_count = coin_signal_counts_from_data(data, n_cells)
    diagnostics["negative_segment_count"] = int(np.sum(_segment_values(data, n_cells) < 0.0))
    diagnostics["shift_offset"] = float(shift_offset)
    diagnostics["coin_signal_zero_denominator_count"] = zero_signal_count

    if has_coin_signal_inputs:
        try:
            result = module.reconstruct_density_matrix(
                system_type,
                coin_signal_counts,
                basis_path,
                efficiency_map_path=module.default_efficiency_map_path(system_type),
            )
            calibrated = result.get("calibrated", result)
            rho = calibrated.get("rho_full")
            if rho is not None:
                variants["coin_signal"] = np.asarray(rho, dtype=complex)
                diagnostics["coin_signal"] = {
                    "estimator": "coin_bins/signal_bins",
                    "zero_denominator_count": zero_signal_count,
                    "amplitude_correction": result.get("diagnostics", {}).get("calibration", {}),
                }
        except Exception as exc:
            diagnostics["errors"]["coin_signal"] = str(exc)
    else:
        diagnostics["errors"]["coin_signal"] = "coin_bins/signal_bins unavailable"

    try:
        retrieval_map = module.legacy_efficiency_map_path(system_type)
        if os.path.exists(retrieval_map):
            result = module.reconstruct_density_matrix(
                system_type,
                clipped_counts,
                basis_path,
                efficiency_map_path=retrieval_map,
                rate_power=0.5,
            )
            calibrated = result.get("calibrated", result)
            rho = calibrated.get("rho_full")
            if rho is not None:
                variants["retrieval_sqrt"] = np.asarray(rho, dtype=complex)
                diagnostics["retrieval_sqrt"] = result.get("diagnostics", {}).get("calibration", {})
        else:
            diagnostics["errors"]["retrieval_sqrt"] = f"retrieval map missing: {retrieval_map}"
    except Exception as exc:
        diagnostics["errors"]["retrieval_sqrt"] = str(exc)

    try:
        result = module.reconstruct_density_matrix(
            system_type,
            shifted_counts,
            basis_path,
            efficiency_map_path=module.default_efficiency_map_path(system_type),
        )
        calibrated = result.get("calibrated", result)
        rho = calibrated.get("rho_full")
        if rho is not None:
            variants["shifted_idler"] = np.asarray(rho, dtype=complex)
            diagnostics["shifted_idler"] = result.get("diagnostics", {}).get("calibration", {})
    except Exception as exc:
        diagnostics["errors"]["shifted_idler"] = str(exc)

    diagnostics["available"] = bool(variants)
    return {"rho_variants": variants, "diagnostics": diagnostics}


def _segment_values(data: Optional[Mapping[str, Any]], n_cells: int) -> np.ndarray:
    values = np.asarray((data or {}).get("segment_values", []), dtype=float).reshape(-1)
    n_cells = int(n_cells)
    if values.size < n_cells:
        values = np.pad(values, (0, n_cells - values.size), constant_values=0.0)
    return values[:n_cells]


def _sized_values(values: Any, n_cells: int) -> np.ndarray:
    arr = np.asarray([] if values is None else values, dtype=float).reshape(-1)
    n_cells = int(n_cells)
    if arr.size < n_cells:
        arr = np.pad(arr, (0, n_cells - arr.size), constant_values=0.0)
    return arr[:n_cells]


def _load_calibrated_module():
    if str(_QUBIT_RECON_DIR) not in sys.path:
        sys.path.insert(0, str(_QUBIT_RECON_DIR))
    module_path = _QUBIT_RECON_DIR / "calibrated_density_reconstruction.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("calibrated_density_reconstruction", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
