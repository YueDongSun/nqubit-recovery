"""Data UI helpers for saving feedback-training samples."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_recon.feedback_model import train_feedback_model
from system_control.experiment_control.dataset_store import FeedbackDatasetStore, matrix_hash_snapshot
from system_control.experiment_control.model_paths import display_feedback_model_path
from system_control.experiment_control.models import MeasurementSnapshot
from system_control.experiment_control.noise_context import build_noise_context
from system_control.experiment_control.operation_points import build_operation_registry, operation_registry_hash


@dataclass
class FeedbackSampleSaveResult:
    success: bool
    message: str
    path: Optional[Path] = None


def save_feedback_sample_from_window(window, system_type: str) -> FeedbackSampleSaveResult:
    """Save the current data UI frame/rho as a FeedbackSample."""

    context = getattr(window, "feedback_context", None)
    if not context:
        return FeedbackSampleSaveResult(False, "Feedback context is missing. Open this data UI from the experiment UI first.")

    context_system = context.get("system_type")
    if context_system and context_system != system_type:
        return FeedbackSampleSaveResult(False, f"Feedback context system mismatch: {context_system} != {system_type}")

    target_state = context.get("target_state_params")
    if not target_state:
        return FeedbackSampleSaveResult(False, "Feedback context is missing target_state_params.")

    current_data = getattr(window, "current_data", None)
    if current_data is None:
        return FeedbackSampleSaveResult(False, "No current data frame is available.")

    latest_rho = getattr(window, "latest_rho", None)
    if latest_rho is None:
        return FeedbackSampleSaveResult(False, "No reconstructed density matrix is available yet.")
    latest_raw_rho = getattr(window, "latest_raw_rho", None)
    latest_calibrated_rho = getattr(window, "latest_calibrated_rho", None)
    latest_idler_calibrated_rho = getattr(window, "latest_idler_calibrated_rho", latest_calibrated_rho)
    latest_retrieval_calibrated_rho = getattr(window, "latest_retrieval_calibrated_rho", None)
    rho_source = "idler_calibrated_density_reconstruction" if latest_idler_calibrated_rho is not None else "raw_density_reconstruction"

    n_cells = 75 if system_type == "3qubit" else 256
    signal = _array(current_data.get("signal_bins"), n_cells)
    idler = _array(current_data.get("idler_bins"), n_cells)
    coin = _array(current_data.get("coin_bins"), n_cells)
    segment = _array(current_data.get("segment_values"), n_cells)
    coin_idler = _array(
        current_data.get("coin_idler_values", _safe_ratio(coin, idler)),
        n_cells,
    )
    timestamp = float(current_data.get("timestamp", time.time()) or time.time())
    rho_target = target_density_matrix(target_state, system_type)
    rho_measured = np.asarray(latest_rho, dtype=complex)
    retrieval_rate = float(np.sum(coin) / max(float(np.sum(signal)), 1.0))
    purity = float(np.real(np.trace(rho_measured @ rho_measured)))
    fidelity = density_fidelity(rho_target, rho_measured)

    snapshot = MeasurementSnapshot(
        system_type=system_type,
        signal_bins=signal,
        idler_bins=idler,
        coin_bins=coin,
        segment_values=segment,
        read_count=int(current_data.get("read_count", 0) or 0),
        write_count=int(current_data.get("write_count", 0) or 0),
        timestamp=timestamp,
        duration_s=float(current_data.get("duration_s", 0.0) or 0.0),
        density_matrix=rho_measured,
        fidelity=fidelity,
        purity=purity,
        retrieval_rate=retrieval_rate,
        schema_version=2,
        signal_bins_raw=_optional_sized_array(current_data.get("signal_bins_raw"), n_cells),
        idler_bins_raw=_optional_sized_array(current_data.get("idler_bins_raw"), n_cells),
        coin_bins_raw=_optional_sized_array(current_data.get("coin_bins_raw"), n_cells),
        signal_bins_corrected=signal,
        idler_bins_corrected=idler,
        coin_bins_corrected=coin,
        signed_segment_values=_array(current_data.get("signed_segment_values", segment), n_cells),
        legacy_clipped_segment_values=_array(current_data.get("legacy_clipped_segment_values", np.maximum(segment, 0.0)), n_cells),
        coin_idler_values=coin_idler,
        signal_timing_histograms=_optional_array(current_data.get("signal_hist")),
        idler_timing_histograms=_optional_array(current_data.get("idler_hist")),
        multi2_count=_optional_int(current_data.get("multi2_count")),
        multi3_count=_optional_int(current_data.get("multi3_count")),
        multi4_count=_optional_int(current_data.get("multi4_count")),
        multi2_pair_counts={str(k): int(v) for k, v in (current_data.get("multi2_pairs") or {}).items()},
        signal_total=int(current_data.get("signal_total", np.sum(signal))),
        idler_total=int(current_data.get("idler_total", np.sum(idler))),
        coin_total=int(current_data.get("coin_total", np.sum(coin))),
        phase_sign=-1.0,
        basis_phase_convention="read_aod_negative_order",
        fpga_protocol_version=str(current_data.get("fpga_protocol_version", "legacy_counter_frame_v1")),
        feature_schema_version=2,
        external_source_characterization=dict(context.get("external_source_characterization") or {}),
        missing_value_masks={
            "multi2_pairs": not bool(current_data.get("multi2_pairs")),
            "multi3_count": current_data.get("multi3_count") is None,
            "multi4_count": current_data.get("multi4_count") is None,
            "raw_counters": current_data.get("signal_bins_raw") is None,
            "timing_histograms": current_data.get("signal_hist") is None,
        },
        diagnostics={
            "source": "data_ui_manual_feedback_save",
            "rho_source": rho_source,
            "raw_rho_available": latest_raw_rho is not None,
            "idler_calibrated_rho_available": latest_idler_calibrated_rho is not None,
            "retrieval_calibrated_rho_available": latest_retrieval_calibrated_rho is not None,
            "workflow_mode": context.get("workflow_mode"),
            "excitation_mode": context.get("excitation_mode"),
            **{
                key: current_data[key]
                for key in (
                    "read_count_raw",
                    "write_count_raw",
                    "counter_decode_layout",
                    "counter_decode_warning",
                )
                if key in current_data
            },
        },
    )
    registry = build_operation_registry(system_type)
    snapshot.operation_point_registry = [point.to_json_dict() for point in registry]
    snapshot.operation_point_registry_hash = operation_registry_hash(registry)
    snapshot.diagnostics["noise_context"] = build_noise_context(
        snapshot,
        pair_model_mode=str(context.get("pair_model_mode", "disabled")),
        external_dlcz_gsi_reference=context.get("external_dlcz_gsi_reference"),
    ).to_json_dict()

    matrix_dir = REPO_ROOT / "system_control" / "qubit_recon_files" / "matrix"
    basis = _load_basis_config(matrix_dir, system_type)
    store = FeedbackDatasetStore(context.get("samples_dir"))
    sample = store.create_sample(
        system_type=system_type,
        target_state_params=dict(target_state),
        rho_target=rho_target,
        snapshot=snapshot,
        basis_config=basis,
        matrix_snapshot=matrix_hash_snapshot(matrix_dir),
        notes="data_ui_manual_feedback_save",
        data_provenance="real",
    )
    path = store.save_sample(sample)
    return FeedbackSampleSaveResult(True, f"Feedback sample saved: {path}", path)


def train_feedback_model_from_context(window, system_type: str) -> Dict[str, Any]:
    """Train/refresh the feedback model for this data UI context."""

    context = getattr(window, "feedback_context", None) or {}
    effective_system = context.get("system_type") or system_type
    model_path = context.get("model_path") or display_feedback_model_path(effective_system)
    samples_dir = context.get("samples_dir") or "ml_recon/data/feedback_samples"
    return train_feedback_model(
        effective_system,
        samples_dir=samples_dir,
        model_path=model_path,
    )


def target_density_matrix(state_params: Dict[str, Any], system_type: str) -> np.ndarray:
    n = 3 if system_type == "3qubit" else 4
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    amps = np.asarray(state_params.get("amplitudes", [1.0] * n), dtype=float)
    if "phases_rad" in state_params:
        phases = np.asarray(state_params.get("phases_rad", [0.0] * n), dtype=float)
    else:
        phases = np.deg2rad(np.asarray(state_params.get("phases_deg", [0.0] * n), dtype=float))
    amps = np.pad(amps, (0, max(0, n - amps.size)), constant_values=1.0)[:n]
    phases = np.pad(phases, (0, max(0, n - phases.size)), constant_values=0.0)[:n]
    norm = np.linalg.norm(amps)
    if norm <= 0:
        amps = np.ones(n, dtype=float)
        norm = np.linalg.norm(amps)
    for i in range(n):
        idx = 2 ** (n - 1 - i)
        psi[idx] = (amps[i] / norm) * np.exp(1j * phases[i])
    return np.outer(psi, psi.conj())


def density_fidelity(rho_target: np.ndarray, rho_measured: np.ndarray) -> float:
    try:
        eigvals, eigvecs = np.linalg.eigh(rho_target)
        eigvals = np.clip(np.real(eigvals), 0.0, None)
        sqrt_target = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
        middle = sqrt_target @ rho_measured @ sqrt_target
        vals = np.linalg.eigvalsh(0.5 * (middle + middle.conj().T))
        return float(np.real(np.sum(np.sqrt(np.clip(vals, 0.0, None))) ** 2))
    except Exception:
        return float(np.real(np.trace(rho_target @ rho_measured)))


def _load_basis_config(matrix_dir: Path, system_type: str) -> Dict[str, Any]:
    name = "basis_config_3qubit.json" if system_type == "3qubit" else "basis_config_4qubit.json"
    path = matrix_dir / name
    if not path.exists():
        return {"path": str(path), "missing": True}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _array(values, size: int) -> np.ndarray:
    if values is None:
        return np.zeros(size, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size >= size:
        return arr[:size].copy()
    out = np.zeros(size, dtype=np.float64)
    out[:arr.size] = arr
    return out


def _optional_array(values):
    return None if values is None else np.asarray(values).copy()


def _optional_sized_array(values, size: int):
    return None if values is None else _array(values, size)


def _optional_int(value):
    return None if value is None else int(value)


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    result[~np.isfinite(result)] = 0.0
    return result
