"""Anchor-joint adapter for the authorized 4Q signed-probe campaign.

This module reads completed snapshots only.  It never stages probes, generates
waveforms, uploads AWG state, collects data, or writes feedback deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from system_control.experiment_control.data_protocol import (
    CampaignManifestV1,
    atomic_write_json,
    load_snapshot_manifest,
    stable_json_hash,
)

from ml_recon.physics_hybrid_v4.phase_geometry import project_phase_offsets
from .probe_system_identification import (
    POINT_COUNT_CHANNELS,
    fit_raw_signed_probe_counts,
    pair_safe_anchor_split_identity,
)


def _point_count_matrix(snapshot_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load true point-local channels; never distribute manifest totals."""
    manifest = load_snapshot_manifest(snapshot_path)
    arrays = []
    provenance: Dict[str, Any] = {"point_channels": list(POINT_COUNT_CHANNELS), "global_multi_distributed": False}
    for channel in POINT_COUNT_CHANNELS:
        raw_name = f"{channel}_bins_raw"
        corrected_name = f"{channel}_bins"
        record = manifest.arrays.get(raw_name) or manifest.arrays.get(corrected_name)
        if not record:
            raise ValueError(f"{snapshot_path} is missing {raw_name}/{corrected_name}")
        provenance[f"{channel}_source"] = raw_name if raw_name in manifest.arrays else corrected_name
        arrays.append(np.asarray(np.load(snapshot_path / record["path"], allow_pickle=False), dtype=float).reshape(-1))
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("point-local count arrays disagree in shape")
    operation_count = arrays[0].shape[0]
    acquisition = dict(getattr(manifest, "acquisition", {}) or {})

    def load_exposure_array(name: str) -> Any:
        record = manifest.arrays.get(name)
        if record:
            return np.asarray(np.load(snapshot_path / record["path"], allow_pickle=False), dtype=float)
        return acquisition.get(name)

    explicit = load_exposure_array("gate_exposure")
    if explicit is not None:
        exposure = np.asarray(explicit, dtype=float)
        if exposure.ndim == 0:
            exposure = np.full((operation_count, 3), float(exposure), dtype=float)
        elif exposure.shape == (operation_count,):
            exposure = np.repeat(exposure[:, None], 3, axis=1)
        elif exposure.shape != (operation_count, 3):
            raise ValueError("gate_exposure must be scalar, [O], or [O,3]")
        exposure_provenance = "explicit_gate_exposure"
        high_confidence = True
    else:
        write = load_exposure_array("write_gate_exposure")
        read = load_exposure_array("read_gate_exposure")
        if write is None:
            write = acquisition.get("write_count")
        if read is None:
            read = acquisition.get("read_count")
        if write is not None and read is not None:
            write_array = np.broadcast_to(np.asarray(write, dtype=float), (operation_count,))
            read_array = np.broadcast_to(np.asarray(read, dtype=float), (operation_count,))
            exposure = np.stack((write_array, read_array, np.minimum(write_array, read_array)), axis=-1)
            exposure_provenance = "write_read_gate_counts"
            high_confidence = True
        else:
            duration = float(acquisition.get("exposure_s", manifest.duration_s) or 0.0)
            exposure = np.full((operation_count, 3), duration, dtype=float)
            exposure_provenance = "duration_approximation"
            high_confidence = False
    if np.any(exposure <= 0.0):
        raise ValueError(f"{snapshot_path} has no positive probe exposure")
    provenance["exposure_provenance"] = exposure_provenance
    provenance["high_confidence_probe_label"] = high_confidence
    multi_meta = dict(getattr(manifest, "multi_click_observation", {}) or {})
    provenance["multi_click_field_scopes"] = dict(multi_meta.get("field_scopes") or {})
    return np.stack(arrays, axis=-1), exposure, provenance


def _probe_tensor(snapshot_path: Path) -> np.ndarray:
    manifest = load_snapshot_manifest(snapshot_path)
    record = manifest.arrays.get("phase_probe_tensor")
    if not record:
        raise ValueError(f"{snapshot_path} is missing phase_probe_tensor")
    values = np.asarray(np.load(snapshot_path / record["path"], allow_pickle=False), dtype=float)
    if values.shape != (256, 4):
        raise ValueError("phase_probe_tensor must have shape [256,4]")
    return values


def _zero_anchor_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rates = []
    times = []
    temperatures = []
    for record in records:
        counts, exposure, _ = _point_count_matrix(Path(record["snapshot_path"]))
        rates.append(counts / exposure)
        manifest = load_snapshot_manifest(record["snapshot_path"])
        times.append(float(manifest.ended_at))
        temperatures.append(dict(getattr(manifest, "environment", {}) or {}).get("temperature_c"))
    if not rates:
        return {"available": False}
    matrix = np.stack(rates, axis=0)
    return {
        "available": True,
        "count": len(rates),
        "mean_rate": np.mean(matrix, axis=0),
        "variance_rate": np.var(matrix, axis=0, ddof=1) if len(rates) > 1 else np.zeros_like(matrix[0]),
        "times": np.asarray(times, dtype=np.float64),
        "temperatures": np.asarray([np.nan if value is None else float(value) for value in temperatures], dtype=np.float64),
        "snapshot_hashes": [
            stable_json_hash(load_snapshot_manifest(record["snapshot_path"]).to_json_dict())
            for record in records
        ],
    }


def _broadcast_rate_parameter(value: Any, shape: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(0.0 if value is None else value, dtype=float)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=float)
    return np.broadcast_to(array, shape).astype(float, copy=False)


def _drift_rate_offset(
    snapshot_path: Path,
    shape: Tuple[int, int],
    *,
    drift_report: Mapping[str, Any],
    reference_time: float,
    reference_temperature_c: float,
) -> np.ndarray:
    """Evaluate the preregistered time/temperature/block baseline offset."""

    manifest = load_snapshot_manifest(snapshot_path)
    elapsed_s = float(manifest.ended_at) - float(reference_time)
    environment = dict(getattr(manifest, "environment", {}) or {})
    temperature = environment.get("temperature_c")
    temperature_delta = 0.0 if temperature is None else float(temperature) - float(reference_temperature_c)
    time_slope = _broadcast_rate_parameter(
        drift_report.get("rate_slope_per_s", drift_report.get("time_rate_slope_per_s")), shape,
    )
    temperature_slope = _broadcast_rate_parameter(
        drift_report.get("temperature_rate_slope_per_c"), shape,
    )
    schedule = dict(getattr(manifest, "schedule", {}) or {})
    block_id = str(schedule.get("time_block") or schedule.get("block_id") or "")
    block_offsets = dict(drift_report.get("time_block_rate_offsets") or {})
    block_offset = _broadcast_rate_parameter(block_offsets.get(block_id), shape)
    return elapsed_s * time_slope + temperature_delta * temperature_slope + block_offset


def analyze_joint_probe_records(
    campaign: CampaignManifestV1,
    output_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    drift_report: Mapping[str, Any],
    ridge: float = 1e-4,
    rank_threshold: int = 3,
    condition_threshold: float = 1e3,
    nb2_dispersion: float = 0.0,
) -> Dict[str, Any]:
    """Fit all six signed pairs jointly within each anchor identity."""
    output = Path(output_dir)
    label_root = output / "probe_labels"
    label_root.mkdir(parents=True, exist_ok=True)
    by_anchor: Dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_anchor.setdefault(str(record.get("anchor_id") or "missing_anchor"), []).append(record)

    snapshots: Dict[str, Any] = {}
    anchor_reports = []
    pair_reports = []
    total_complete_pairs = 0
    for anchor_id, anchor_records in sorted(by_anchor.items()):
        zero_records = [record for record in anchor_records if int(record["phase_probe"].get("pair_sign", 0)) == 0]
        zero_summary = _zero_anchor_summary(zero_records)
        if zero_summary.get("available"):
            reference_time = float(np.mean(zero_summary["times"]))
            finite_temperatures = zero_summary["temperatures"][np.isfinite(zero_summary["temperatures"])]
            reference_temperature = float(np.mean(finite_temperatures)) if finite_temperatures.size else 25.0
        else:
            reference_time = float(np.mean([
                load_snapshot_manifest(record["snapshot_path"]).ended_at for record in anchor_records
            ]))
            reference_temperature = 25.0
        signed_pairs: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        for record in anchor_records:
            probe = dict(record.get("phase_probe") or {})
            sign = int(probe.get("pair_sign", 0))
            if sign:
                signed_pairs.setdefault(str(probe.get("code_pair_id")), {})[sign] = record
        complete_pairs = [(pair_id, signed) for pair_id, signed in sorted(signed_pairs.items()) if set(signed) == {-1, 1}]
        total_complete_pairs += len(complete_pairs)
        if len(complete_pairs) < 3:
            anchor_reports.append({
                "anchor_id": anchor_id, "quality_pass": False, "reason": "fewer_than_three_complete_directions",
                "complete_pair_count": len(complete_pairs),
            })
            continue

        directions = []
        amplitudes = []
        plus_counts = []
        minus_counts = []
        plus_exposure = []
        minus_exposure = []
        pair_provenance = []
        signed_records = []
        for pair_id, signed in complete_pairs:
            plus, minus = signed[1], signed[-1]
            plus_probe, minus_probe = dict(plus["phase_probe"]), dict(minus["phase_probe"])
            amplitude = float(plus_probe.get("amplitude_rad", 0.0))
            if amplitude <= 0.0 or not np.isclose(amplitude, float(minus_probe.get("amplitude_rad", 0.0))):
                raise ValueError(f"invalid paired amplitude for {pair_id}")
            plus_tensor = _probe_tensor(Path(plus["snapshot_path"]))
            minus_tensor = _probe_tensor(Path(minus["snapshot_path"]))
            if not np.allclose(plus_tensor, -minus_tensor, atol=1e-9):
                raise ValueError(f"signed tensors are not antipodal for {pair_id}")
            direction = project_phase_offsets(plus_tensor / amplitude)
            plus_matrix, plus_duration, plus_source = _point_count_matrix(Path(plus["snapshot_path"]))
            minus_matrix, minus_duration, minus_source = _point_count_matrix(Path(minus["snapshot_path"]))
            plus_offset = _drift_rate_offset(
                Path(plus["snapshot_path"]), plus_matrix.shape,
                drift_report=drift_report,
                reference_time=reference_time,
                reference_temperature_c=reference_temperature,
            )
            minus_offset = _drift_rate_offset(
                Path(minus["snapshot_path"]), minus_matrix.shape,
                drift_report=drift_report,
                reference_time=reference_time,
                reference_temperature_c=reference_temperature,
            )
            plus_matrix = np.maximum(plus_matrix - plus_offset * plus_duration, 0.0)
            minus_matrix = np.maximum(minus_matrix - minus_offset * minus_duration, 0.0)
            directions.append(direction)
            amplitudes.append(amplitude)
            plus_counts.append(plus_matrix)
            minus_counts.append(minus_matrix)
            plus_exposure.append(plus_duration)
            minus_exposure.append(minus_duration)
            signed_records.extend((plus, minus))
            pair_provenance.append({
                "pair_id": pair_id,
                "plus_snapshot": str(plus["snapshot_path"]),
                "minus_snapshot": str(minus["snapshot_path"]),
                "plus_count_provenance": plus_source,
                "minus_count_provenance": minus_source,
            })

        fit = fit_raw_signed_probe_counts(
            np.stack(directions), np.asarray(amplitudes), np.stack(plus_counts), np.stack(minus_counts),
            np.asarray(plus_exposure), np.asarray(minus_exposure), nb2_dispersion=nb2_dispersion,
            ridge=ridge, rank_threshold=rank_threshold, condition_threshold=condition_threshold,
        )
        mean_counts = 0.5 * (np.mean(np.stack(plus_counts), axis=0) + np.mean(np.stack(minus_counts), axis=0))
        mean_exposure = 0.5 * (
            np.mean(np.stack(plus_exposure), axis=0)
            + np.mean(np.stack(minus_exposure), axis=0)
        )
        baseline_rate = np.maximum(mean_counts / np.maximum(mean_exposure, 1e-12), 1e-12)
        log_rate_jacobian = fit.joint_jacobian / baseline_rate[..., None]
        coefficient = mean_counts / (1.0 + max(float(nb2_dispersion), 0.0) * mean_counts)
        fisher = np.einsum("oc,ocm,ocn->omn", coefficient, log_rate_jacobian, log_rate_jacobian, optimize=True)
        posterior_covariance = np.linalg.pinv(fisher + ridge * np.eye(3)[None, :, :], hermitian=True)
        # V4 compatibility retains six slots, but the three nonexistent global
        # slots are explicitly masked and remain zero.
        compatibility_jacobian = np.zeros((fit.joint_jacobian.shape[0], 6, 3), dtype=np.float32)
        compatibility_mask = np.zeros((fit.joint_jacobian.shape[0], 6), dtype=np.float32)
        compatibility_jacobian[:, :3] = log_rate_jacobian.astype(np.float32)
        compatibility_mask[:, :3] = fit.jacobian_supervision_mask.astype(np.float32)
        split_identity = pair_safe_anchor_split_identity(anchor_id, fit.direction_hashes)
        exposure_provenance = sorted({
            source["exposure_provenance"]
            for pair in pair_provenance
            for source in (pair["plus_count_provenance"], pair["minus_count_provenance"])
        })
        high_confidence_exposure = all(
            bool(source["high_confidence_probe_label"])
            for pair in pair_provenance
            for source in (pair["plus_count_provenance"], pair["minus_count_provenance"])
        )
        label_confidence = fit.jacobian_supervision_mask & bool(high_confidence_exposure)
        drift_model_hash = stable_json_hash({
            "drift_report": dict(drift_report),
            "reference_time": reference_time,
            "reference_temperature_c": reference_temperature,
        })
        zero_anchor_hashes = list(zero_summary.get("snapshot_hashes") or [])
        label_path = label_root / f"{stable_json_hash({'anchor_id': anchor_id, 'directions': fit.direction_hashes})[:16]}.npz"
        save_payload = {
            "joint_jacobian": fit.joint_jacobian.astype(np.float64),
            "phase_rate_jacobian": fit.joint_jacobian.astype(np.float64),
            "phase_log_rate_jacobian": log_rate_jacobian.astype(np.float64),
            "drift_corrected_phase_rate_jacobian": fit.joint_jacobian.astype(np.float64),
            "drift_corrected_phase_log_rate_jacobian": log_rate_jacobian.astype(np.float64),
            "phase_jacobian_covariance": fit.jacobian_covariance.astype(np.float64),
            "phase_jacobian_mask": fit.jacobian_supervision_mask.astype(np.bool_),
            "amplitude_empirical_evidence_mask": np.zeros_like(
                fit.jacobian_supervision_mask, dtype=np.bool_,
            ),
            "count_jacobian": compatibility_jacobian,
            "jacobian_covariance": fit.jacobian_covariance.astype(np.float64),
            "jacobian_inverse_variance": fit.jacobian_inverse_variance.astype(np.float64),
            "jacobian_supervision_mask": fit.jacobian_supervision_mask.astype(np.bool_),
            "count_jacobian_mask": compatibility_mask,
            "fisher_matrix": fisher.astype(np.float64),
            "posterior_covariance": posterior_covariance.astype(np.float64),
            "rank": fit.rank.astype(np.int64),
            "singular_values": fit.singular_values.astype(np.float64),
            "condition_number": fit.condition_number.astype(np.float64),
            "complete_pair_count": fit.complete_pair_count.astype(np.int64),
            "directional_prediction": fit.directional_prediction.astype(np.float64),
            "directional_residuals": fit.directional_residuals.astype(np.float64),
            "held_out_direction_error": fit.held_out_direction_error.astype(np.float64),
            "direction_hashes": np.asarray(fit.direction_hashes, dtype="U64"),
            "baseline_rate": baseline_rate.astype(np.float64),
            "exposure": mean_exposure.astype(np.float64),
            "exposure_provenance": np.asarray(exposure_provenance, dtype="U64"),
            "drift_model_hash": np.asarray(drift_model_hash),
            "zero_anchor_hashes": np.asarray(zero_anchor_hashes, dtype="U64"),
            "label_confidence": label_confidence.astype(np.bool_),
            "global_multi_distributed": np.asarray(False),
        }
        if zero_summary.get("available"):
            save_payload["zero_anchor_mean_rate"] = zero_summary["mean_rate"]
            save_payload["zero_anchor_variance_rate"] = zero_summary["variance_rate"]
        np.savez_compressed(label_path, **save_payload)
        operation_quality = np.any(fit.jacobian_supervision_mask, axis=-1)
        quality = bool(np.mean(operation_quality) >= 0.5 and len(complete_pairs) >= 6)
        report = {
            "anchor_id": anchor_id,
            "pair_safe_split_identity": split_identity,
            "label_path": str(label_path.resolve()),
            "complete_pair_count": len(complete_pairs),
            "direction_hashes": list(fit.direction_hashes),
            "design_audit": fit.design_audit(),
            "quality_pass": quality,
            "zero_anchor_count": len(zero_records),
            "global_multi_distributed": False,
            "point_channels": list(POINT_COUNT_CHANNELS),
            "hessian_retained": False,
            "pair_provenance": pair_provenance,
            "exposure_provenance": exposure_provenance,
            "high_confidence_probe_label": bool(high_confidence_exposure),
            "drift_model_hash": drift_model_hash,
            "zero_anchor_hashes": zero_anchor_hashes,
        }
        anchor_reports.append(report)
        for pair_id, signed in complete_pairs:
            pair_reports.append({
                "pair_id": pair_id, "anchor_id": anchor_id,
                "pair_safe_split_identity": split_identity,
                "label_path": str(label_path.resolve()), "quality_pass": quality,
            })
        for record in signed_records:
            snapshots[str(Path(record["snapshot_path"]).resolve())] = {
                "pair_split_key": split_identity,
                "anchor_split_identity": split_identity,
                "count_jacobian_label_path": str(label_path.resolve()),
                "quality_pass": quality,
                "joint_directional_supervision": True,
            }

    index_payload = {
        "schema": "PhysicsHybridV5JointProbeLabelIndex",
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "point_channels": list(POINT_COUNT_CHANNELS),
        "global_multi_distributed": False,
        "anchors": anchor_reports,
        "pairs": pair_reports,
        "snapshots": snapshots,
        "ridge": float(ridge),
        "rank_threshold": int(rank_threshold),
        "condition_threshold": float(condition_threshold),
        "nb2_dispersion": float(nb2_dispersion),
    }
    index_path = output / "real_probe_label_index.json"
    atomic_write_json(index_path, index_payload)
    return {
        "campaign_id": campaign.campaign_id,
        "completed_anchor_count": len(records),
        "complete_pair_count": total_complete_pairs,
        "quality_pair_count": sum(bool(row.get("quality_pass")) for row in pair_reports),
        "quality_anchor_count": sum(bool(row.get("quality_pass")) for row in anchor_reports),
        "drift_selection": str(output / "drift_selection.json"),
        "probe_label_index": str(index_path),
        "global_multi_distributed": False,
    }


__all__ = ["analyze_joint_probe_records"]
