"""V9 rho8 reconstruction (charter section 10).

Each estimator-map pair is reconstructed through the same basis registry and phase
convention (``phase_sign=-1``, ``basis_phase_convention=read_aod_negative_order``)
using weighted linear physical reconstruction + PSD projection + unit trace.

Constraints honored here:

- canonical normalized maps only (from :mod:`rho8_maps`);
- ratio estimators (``cs_minus_ir`` / ``cs``) are explicitly distinguished from raw
  integer counts; the ratio reconstruction is never called a Poisson MLE;
- count-weight clipping is recorded;
- invalid-operation masks are recorded;
- the reconstruction output is a weak teacher / fixed-reconstruction evaluation, never
  independent evidence and never ground truth.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .contracts import V9RhoEightViewBundleV1, V9RhoViewProvenanceV1, RHO8_CANONICAL_KEYS
from .rho8_estimators import V9OperationEstimatorResultV1
from .rho8_maps import MAP_KEYS, RETRIEVAL_SHAPES, REPO_ROOT

QUBIT_RECON_DIR = REPO_ROOT / "system_control" / "qubit_recon_files"
MATRIX_DIR = QUBIT_RECON_DIR / "matrix"
PHASE_SIGN = -1
BASIS_PHASE_CONVENTION = "read_aod_negative_order"

ESTIMATOR_KEY_BY_VIEW = {
    "rho_cs_minus_ir_retrieval_raw": "cs_minus_ir",
    "rho_cs_minus_ir_retrieval_sqrt": "cs_minus_ir",
    "rho_cs_minus_ir_idler_raw": "cs_minus_ir",
    "rho_cs_minus_ir_idler_sqrt": "cs_minus_ir",
    "rho_cs_retrieval_raw": "cs",
    "rho_cs_retrieval_sqrt": "cs",
    "rho_cs_idler_raw": "cs",
    "rho_cs_idler_sqrt": "cs",
}
MAP_KEY_BY_VIEW = {
    "rho_cs_minus_ir_retrieval_raw": "retrieval_raw",
    "rho_cs_minus_ir_retrieval_sqrt": "retrieval_sqrt",
    "rho_cs_minus_ir_idler_raw": "idler_raw",
    "rho_cs_minus_ir_idler_sqrt": "idler_sqrt",
    "rho_cs_retrieval_raw": "retrieval_raw",
    "rho_cs_retrieval_sqrt": "retrieval_sqrt",
    "rho_cs_idler_raw": "idler_raw",
    "rho_cs_idler_sqrt": "idler_sqrt",
}


def _load_calibrated_reconstruction_module() -> Any:
    module_path = QUBIT_RECON_DIR / "calibrated_density_reconstruction.py"
    if str(QUBIT_RECON_DIR) not in sys.path:
        sys.path.insert(0, str(QUBIT_RECON_DIR))
    spec = importlib.util.spec_from_file_location(
        "v9_rho8_calibrated_density_reconstruction",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reconstruct_one_view(
    module: Any,
    system_type: str,
    estimator: V9OperationEstimatorResultV1,
    efficiency_map: np.ndarray,
    basis_config_path: str,
) -> Dict[str, Any]:
    """Reconstruct one estimator-map pair with the calibrated linear+PSD path."""
    basis_entries = module.raw_reconstruction.load_basis_entries(
        system_type, basis_config_path
    )
    counts_array, _ = module._values_for_basis(estimator.x, len(basis_entries), "counts")
    rate_vectors = module.rate_vectors_for_entries(
        system_type, efficiency_map, len(basis_entries)
    )
    operation_rates = module._mean_positive(rate_vectors, axis=1)
    reference_rate = module._reference_rate(operation_rates, "max")
    count_weights = np.clip(
        reference_rate / np.maximum(operation_rates, module.MIN_RATE),
        0.1,
        10.0,
    )
    clipping = {
        "lower_bound": float(np.min(count_weights)),
        "upper_bound": float(np.max(count_weights)),
        "clipped_ops": int(np.sum((count_weights == 0.1) | (count_weights == 10.0))),
        "recorded": True,
    }
    calibrated_entries = module._apply_rate_vectors_to_basis(basis_entries, rate_vectors)
    result = module._reconstruct_from_entries(
        system_type,
        counts_array * count_weights,
        calibrated_entries,
        basis_config_path,
        phase_sign=PHASE_SIGN,
    )
    rho_core = np.asarray(result["rho_core"], dtype=np.complex128)
    rho_full = np.asarray(result["rho_full"], dtype=np.complex128)
    purity = float(np.real(np.trace(rho_core @ rho_core)))
    signed = np.asarray(estimator.signed_x, dtype=np.float64)
    denom = float(np.sum(np.abs(signed))) + 1e-30
    w_subspace_mass = float(np.clip(np.sum(estimator.x) / denom, 0.0, 1.0))
    trace_error = float(abs(np.real(np.trace(rho_core)) - 1.0))
    eigenvalues = np.real(np.linalg.eigvalsh(rho_core))
    psd_correction_norm = float(np.linalg.norm(np.asarray(result["rho_linear"]) - rho_core))
    return {
        "rho_full": rho_full,
        "rho_W_core": rho_core,
        "W_subspace_mass": w_subspace_mass,
        "purity": purity,
        "likelihood_diagnostics": {
            "reconstruction_kind": "weighted_linear_physical_psd_trace_one",
            "is_poisson_mle": False,
            "ratio_estimator_not_raw_counts": True,
            "basis_phase_convention": BASIS_PHASE_CONVENTION,
            "phase_sign": PHASE_SIGN,
        },
        "map_diagnostics": {
            "map_sha256": None,  # filled by caller from provenance
            "shape": list(efficiency_map.shape),
            "positive_support": int(np.sum(efficiency_map > 0.0)),
        },
        "estimator_diagnostics": estimator.to_dict(),
        "psd_correction_norm": psd_correction_norm,
        "trace_error": trace_error,
        "eigenvalue_min": float(np.min(eigenvalues)),
        "count_weight_clipping": clipping,
        "invalid_operation_mask": list(np.asarray(estimator.zero_signal_mask, dtype=bool).tolist()),
        "availability_mask": bool(np.all(np.isfinite(rho_core)) and trace_error < 1e-3),
    }


def reconstruct_rho8_view_bundle(
    system_type: str,
    *,
    sample_id: str,
    estimators: Dict[str, V9OperationEstimatorResultV1],
    maps: Dict[str, Tuple[np.ndarray, V9RhoViewProvenanceV1]],
    basis_config_path: str | Path | None = None,
    count_source_sha256: str = "",
) -> V9RhoEightViewBundleV1:
    """Assemble the canonical eight-view bundle from estimators and maps.

    ``maps`` is the output of :func:`rho8_maps.build_rho8_maps` (canonical normalized
    matrices + provenance). Views are keyed by the exact canonical rho8 keys.
    """
    if system_type not in RETRIEVAL_SHAPES:
        raise ValueError(f"system_type must be one of {list(RETRIEVAL_SHAPES)}")
    basis_path = str(
        basis_config_path or MATRIX_DIR / f"basis_config_{system_type}.json"
    )
    if not Path(basis_path).is_file():
        raise FileNotFoundError(basis_path)

    module = _load_calibrated_reconstruction_module()
    views: Dict[str, Any] = {}
    view_provenance: Dict[str, Any] = {}
    availability: Dict[str, bool] = {}
    for key in RHO8_CANONICAL_KEYS:
        est_key = ESTIMATOR_KEY_BY_VIEW[key]
        map_key = MAP_KEY_BY_VIEW[key]
        matrix, prov = maps[map_key]
        view = _reconstruct_one_view(
            module, system_type, estimators[est_key], matrix, basis_path
        )
        view["map_diagnostics"]["map_sha256"] = prov.normalized_matrix_sha256
        views[key] = view
        view_provenance[map_key] = prov.to_dict()
        availability[key] = view["availability_mask"]

    return V9RhoEightViewBundleV1(
        sample_id=sample_id,
        system_type=system_type,
        views=views,
        view_provenance=view_provenance,
        estimator_diagnostics={k: v.to_dict() for k, v in estimators.items()},
        reconstruction_diagnostics={
            "basis_config_path": basis_path,
            "basis_phase_convention": BASIS_PHASE_CONVENTION,
            "phase_sign": PHASE_SIGN,
            "weighted_linear_physical_psd_trace_one": True,
        },
        availability_mask=availability,
        count_source_sha256=count_source_sha256,
    )
