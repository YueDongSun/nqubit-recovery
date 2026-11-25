"""Feedback dataset sample persistence."""

from __future__ import annotations

import hashlib
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .models import FeedbackSample, MeasurementSnapshot


class FeedbackDatasetStore:
    """Read and write offline feedback-training samples."""

    def __init__(self, samples_dir: str | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.samples_dir = Path(samples_dir) if samples_dir else repo_root / "ml_recon" / "data" / "feedback_samples"

    def create_sample(
        self,
        system_type: str,
        target_state_params: Dict[str, Any],
        rho_target: np.ndarray,
        snapshot: MeasurementSnapshot,
        basis_config: Dict[str, Any],
        matrix_snapshot: Dict[str, Any],
        notes: str = "",
        data_provenance: str = "real",
    ) -> FeedbackSample:
        if snapshot.density_matrix is None:
            raise ValueError("snapshot.density_matrix is required before saving a feedback sample")
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(snapshot.timestamp))
        sample_id = f"{system_type}_{stamp}_{int(time.time() * 1000) % 100000:05d}"
        counts_snapshot = {
            "signal_bins": snapshot.signal_bins.tolist(),
            "idler_bins": snapshot.idler_bins.tolist(),
            "coin_bins": snapshot.coin_bins.tolist(),
            "segment_values": snapshot.segment_values.tolist(),
            "read_count": int(snapshot.read_count),
            "write_count": int(snapshot.write_count),
            "duration_s": float(snapshot.duration_s),
            "signed_segment_values": _array_payload(snapshot.signed_segment_values, snapshot.segment_values),
            "legacy_clipped_segment_values": _array_payload(snapshot.legacy_clipped_segment_values),
            "signal_bins_raw": _array_payload(snapshot.signal_bins_raw),
            "idler_bins_raw": _array_payload(snapshot.idler_bins_raw),
            "coin_bins_raw": _array_payload(snapshot.coin_bins_raw),
            "signal_bins_corrected": _array_payload(snapshot.signal_bins_corrected),
            "idler_bins_corrected": _array_payload(snapshot.idler_bins_corrected),
            "coin_bins_corrected": _array_payload(snapshot.coin_bins_corrected),
            "signal_timing_histograms": _array_payload(snapshot.signal_timing_histograms),
            "idler_timing_histograms": _array_payload(snapshot.idler_timing_histograms),
            "multi2_count": snapshot.multi2_count,
            "multi3_count": snapshot.multi3_count,
            "multi4_count": snapshot.multi4_count,
            "multi2_pair_counts": snapshot.multi2_pair_counts,
            "signal_total": snapshot.signal_total,
            "idler_total": snapshot.idler_total,
            "coin_total": snapshot.coin_total,
            "phase_sign": float(snapshot.phase_sign),
            "basis_phase_convention": snapshot.basis_phase_convention,
            "fpga_protocol_version": snapshot.fpga_protocol_version,
        }
        for key in (
            "read_count_raw",
            "write_count_raw",
            "counter_decode_layout",
            "counter_decode_warning",
        ):
            if key in snapshot.diagnostics:
                counts_snapshot[key] = snapshot.diagnostics[key]

        return FeedbackSample(
            sample_id=sample_id,
            system_type=system_type,
            target_state_params=target_state_params,
            rho_target=rho_target,
            rho_measured=snapshot.density_matrix,
            counts_snapshot=counts_snapshot,
            retrieval_metrics={
                "retrieval_rate": snapshot.retrieval_rate,
                "fidelity": snapshot.fidelity,
                "purity": snapshot.purity,
                "diagnostics": snapshot.diagnostics,
            },
            basis_config=basis_config,
            matrix_snapshot=matrix_snapshot,
            notes=notes,
            timestamp=snapshot.timestamp,
            feature_schema_version=int(snapshot.feature_schema_version),
            operation_point_registry_snapshot=snapshot.operation_point_registry,
            operation_point_registry_hash=snapshot.operation_point_registry_hash,
            calibration_snapshot=snapshot.calibration_snapshot,
            calibration_snapshot_hash=snapshot.calibration_snapshot_hash,
            noise_context=_json_ready(snapshot.diagnostics.get("noise_context", {})),
            external_source_characterization=_json_ready(snapshot.external_source_characterization),
            missing_value_masks=_json_ready(snapshot.missing_value_masks),
            data_provenance=data_provenance,
        )

    def save_sample(self, sample: FeedbackSample) -> Path:
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        path = self.samples_dir / f"{sample.sample_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sample.to_json_dict(), handle, indent=2)
        return path

    def load_samples(self, system_type: Optional[str] = None) -> List[FeedbackSample]:
        if not self.samples_dir.exists():
            return []
        samples: List[FeedbackSample] = []
        for path in sorted(self.samples_dir.rglob("*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or "sample_id" not in payload:
                continue
            sample = FeedbackSample.from_json_dict(payload)
            if system_type is None or sample.system_type == system_type:
                samples.append(sample)
        for pattern, compressed in (("*.jsonl", False), ("*.jsonl.gz", True)):
            for path in sorted(self.samples_dir.rglob(pattern)):
                opener = gzip.open if compressed else open
                with opener(path, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            sample = FeedbackSample.from_json_dict(json.loads(line))
                        except Exception as exc:
                            raise ValueError(f"invalid feedback JSONL row {path}:{line_number}: {exc}") from exc
                        if system_type is None or sample.system_type == system_type:
                            samples.append(sample)
        return samples


def matrix_hash_snapshot(matrix_dir: str | Path, patterns: Iterable[str] = ("*.txt", "*.json")) -> Dict[str, Any]:
    """Return non-invasive hashes for matrix/config files used during a sample."""

    root = Path(matrix_dir)
    files: Dict[str, Any] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_dir() or path.parent.name == "ml_delta":
                continue
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                files[path.name] = {
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
            except OSError as exc:
                files[path.name] = {"error": str(exc)}
    return {
        "matrix_dir": str(root),
        "files": files,
    }


def _array_payload(values: Optional[np.ndarray], fallback: Optional[np.ndarray] = None):
    if values is None:
        values = fallback
    return None if values is None else np.asarray(values).tolist()


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
