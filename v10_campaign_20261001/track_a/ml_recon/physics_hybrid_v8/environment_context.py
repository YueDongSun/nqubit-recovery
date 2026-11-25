"""Environment-context hooks for measured-data contract V2.

Temperature and humidity are weak, observational data-quality context.  They
never create a hard label, remove a row, change a rho/action mask, or reweight
the raw-count likelihood.  Causal features use only sensor samples available
at or before a measurement endpoint.  Whole-window summaries are explicitly
retrospective and are never valid predictive inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.measured_data.contracts import MeasuredSampleRecordV1


ENVIRONMENT_CONTRACT_SCHEMA = "physics_hybrid_v8_environment_contract_v2"
ENVIRONMENT_FEATURE_SCHEMA = "physics_hybrid_v8_environment_features_v2"
DEFAULT_SENSOR_SERIAL = "HXA1379GS1A"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_HIGH_TEMPERATURE_C = 21.5
DEFAULT_SENSOR_DISTANCE_M = 2.0
DEFAULT_MAX_CAUSAL_AGE_S = 600.0
DEFAULT_STABILITY_WINDOW_S = 1800.0
DEFAULT_STABILITY_SCALE_C = 0.20
TEMPERATURE_COLUMN = "field1(温度 ºC )"
HUMIDITY_COLUMN = "field2(湿度)"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EnvironmentSensorContractV2:
    source_sha256: str
    sensor_serial: str = DEFAULT_SENSOR_SERIAL
    timezone: str = DEFAULT_TIMEZONE
    temperature_unit: str = "degree_celsius"
    humidity_unit: str = "percent_rh"
    sensor_distance_from_platform_m_approx: float = DEFAULT_SENSOR_DISTANCE_M
    high_temperature_threshold_c: float = DEFAULT_HIGH_TEMPERATURE_C
    stability_window_s: float = DEFAULT_STABILITY_WINDOW_S
    stability_scale_c: float = DEFAULT_STABILITY_SCALE_C
    maximum_causal_sensor_age_s: float = DEFAULT_MAX_CAUSAL_AGE_S
    role: str = "weak_data_quality_analysis_only"
    hard_label_or_exclusion_authority: bool = False
    raw_count_likelihood_weight_change_allowed: bool = False
    rho_or_action_mask_change_allowed: bool = False
    schema: str = ENVIRONMENT_CONTRACT_SCHEMA

    def validate(self) -> None:
        if self.schema != ENVIRONMENT_CONTRACT_SCHEMA:
            raise ValueError("unsupported environment contract schema")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("environment source_sha256 must be lowercase SHA-256")
        if self.sensor_serial != DEFAULT_SENSOR_SERIAL:
            raise ValueError("environment sensor serial changed")
        if self.timezone != DEFAULT_TIMEZONE:
            raise ValueError("environment timezone changed")
        if self.humidity_unit != "percent_rh":
            raise ValueError("humidity must retain %RH semantics")
        if not math.isclose(
            float(self.sensor_distance_from_platform_m_approx),
            DEFAULT_SENSOR_DISTANCE_M,
        ):
            raise ValueError("environment sensor distance changed")
        if not math.isclose(
            float(self.high_temperature_threshold_c),
            DEFAULT_HIGH_TEMPERATURE_C,
        ):
            raise ValueError("temperature threshold changed")
        if min(
            float(self.stability_window_s),
            float(self.stability_scale_c),
            float(self.maximum_causal_sensor_age_s),
        ) <= 0.0:
            raise ValueError("environment window/scale values must be positive")
        if (
            self.role != "weak_data_quality_analysis_only"
            or self.hard_label_or_exclusion_authority
            or self.raw_count_likelihood_weight_change_allowed
            or self.rho_or_action_mask_change_allowed
        ):
            raise ValueError("environment context exceeded its weak-analysis authority")

    def to_json_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentLogV2:
    timestamps: np.ndarray
    temperature_c: np.ndarray
    humidity_percent_rh: np.ndarray
    contract: EnvironmentSensorContractV2
    row_count: int
    stable_hash: str

    def __post_init__(self) -> None:
        self.contract.validate()
        timestamp = np.asarray(self.timestamps, dtype=np.float64)
        temperature = np.asarray(self.temperature_c, dtype=np.float64)
        humidity = np.asarray(self.humidity_percent_rh, dtype=np.float64)
        if timestamp.shape != temperature.shape or timestamp.shape != humidity.shape:
            raise ValueError("environment arrays must have identical shapes")
        if timestamp.ndim != 1 or timestamp.size != int(self.row_count):
            raise ValueError("environment arrays must be one-dimensional and non-empty")
        if timestamp.size == 0 or not np.all(np.isfinite(timestamp)):
            raise ValueError("environment timestamps must be finite and non-empty")
        if np.any(np.diff(timestamp) <= 0.0):
            raise ValueError("environment timestamps must be strictly increasing")
        if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(humidity)):
            raise ValueError("environment values must be finite")
        if np.any((humidity < 0.0) | (humidity > 100.0)):
            raise ValueError("humidity values must be in percent RH")
        if len(self.stable_hash) != 64:
            raise ValueError("environment stable_hash must be SHA-256")


@dataclass(frozen=True)
class EnvironmentFeaturesV2:
    sample_ids: Tuple[str, ...]
    causal_temperature_c: np.ndarray
    causal_humidity_percent_rh: np.ndarray
    causal_sensor_age_s: np.ndarray
    causal_context_mask: np.ndarray
    trailing_temperature_std_c: np.ndarray
    trailing_temperature_range_c: np.ndarray
    trailing_temperature_slope_c_per_hour: np.ndarray
    trailing_stability_mask: np.ndarray
    high_temperature_mask: np.ndarray
    weak_quality_prior: np.ndarray
    weak_quality_prior_mask: np.ndarray
    retrospective_temperature_mean_c: np.ndarray
    retrospective_temperature_std_c: np.ndarray
    retrospective_humidity_mean_percent_rh: np.ndarray
    retrospective_window_mask: np.ndarray
    environment_log_hash: str
    stable_hash: str
    schema: str = ENVIRONMENT_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_FEATURE_SCHEMA:
            raise ValueError("unsupported environment feature schema")
        row_count = len(self.sample_ids)
        if row_count == 0 or len(set(self.sample_ids)) != row_count:
            raise ValueError("environment features require unique sample IDs")
        for name, value in asdict(self).items():
            if isinstance(value, np.ndarray) and value.shape != (row_count,):
                raise ValueError(f"environment feature {name} has wrong shape")
        for name in (
            "causal_context_mask",
            "trailing_stability_mask",
            "high_temperature_mask",
            "weak_quality_prior_mask",
            "retrospective_window_mask",
        ):
            values = np.asarray(getattr(self, name), dtype=np.int64)
            if not set(np.unique(values)).issubset({0, 1}):
                raise ValueError(f"{name} must be binary")
        if np.any(
            (self.weak_quality_prior[self.weak_quality_prior_mask.astype(bool)] < 0.0)
            | (self.weak_quality_prior[self.weak_quality_prior_mask.astype(bool)] > 1.0)
        ):
            raise ValueError("weak quality prior must lie in [0, 1]")


def load_environment_log(
    csv_path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_sensor_serial: str = DEFAULT_SENSOR_SERIAL,
) -> EnvironmentLogV2:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"environment CSV missing: {path}")
    source_sha256 = _sha256_file(path)
    if expected_sha256 is not None and source_sha256 != str(expected_sha256):
        raise ValueError("environment CSV SHA-256 mismatch")
    timestamps = []
    temperatures = []
    humidities = []
    serials = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"created_at", TEMPERATURE_COLUMN, HUMIDITY_COLUMN, "serial"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("environment CSV lacks required timestamp/T/RH/serial columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(str(row["created_at"])).timestamp()
                temperature = float(row[TEMPERATURE_COLUMN])
                humidity = float(row[HUMIDITY_COLUMN])
            except Exception as exc:
                raise ValueError(f"invalid environment row {row_number}") from exc
            serial = str(row.get("serial") or "").strip()
            if not serial:
                raise ValueError(f"missing sensor serial at row {row_number}")
            serials.add(serial)
            timestamps.append(timestamp)
            temperatures.append(temperature)
            humidities.append(humidity)
    if serials != {expected_sensor_serial}:
        raise ValueError(f"unexpected environment sensor serials: {sorted(serials)}")
    contract = EnvironmentSensorContractV2(source_sha256=source_sha256)
    contract.validate()
    payload = {
        "contract": contract.to_json_dict(),
        "timestamps": timestamps,
        "temperature_c": temperatures,
        "humidity_percent_rh": humidities,
    }
    return EnvironmentLogV2(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        temperature_c=np.asarray(temperatures, dtype=np.float64),
        humidity_percent_rh=np.asarray(humidities, dtype=np.float64),
        contract=contract,
        row_count=len(timestamps),
        stable_hash=_stable_hash(payload),
    )


def _linear_slope_per_hour(timestamps: np.ndarray, values: np.ndarray) -> float:
    if timestamps.size < 2 or float(np.ptp(timestamps)) <= 0.0:
        return math.nan
    centered = timestamps - float(np.mean(timestamps))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return math.nan
    slope_per_second = float(np.dot(centered, values - float(np.mean(values))) / denominator)
    return slope_per_second * 3600.0


def align_environment_context(
    records: Sequence[MeasuredSampleRecordV1],
    environment_log: EnvironmentLogV2,
) -> EnvironmentFeaturesV2:
    """Align causal and retrospective environment features to manifest rows."""

    if not records:
        raise ValueError("environment alignment requires measured records")
    contract = environment_log.contract
    contract.validate()
    sensor_t = np.asarray(environment_log.timestamps, dtype=np.float64)
    sensor_temperature = np.asarray(environment_log.temperature_c, dtype=np.float64)
    sensor_humidity = np.asarray(environment_log.humidity_percent_rh, dtype=np.float64)
    row_count = len(records)

    causal_temperature = np.full(row_count, np.nan, dtype=np.float64)
    causal_humidity = np.full(row_count, np.nan, dtype=np.float64)
    causal_age = np.full(row_count, np.nan, dtype=np.float64)
    causal_mask = np.zeros(row_count, dtype=np.int64)
    trailing_std = np.full(row_count, np.nan, dtype=np.float64)
    trailing_range = np.full(row_count, np.nan, dtype=np.float64)
    trailing_slope = np.full(row_count, np.nan, dtype=np.float64)
    stability_mask = np.zeros(row_count, dtype=np.int64)
    high_temperature_mask = np.zeros(row_count, dtype=np.int64)
    weak_quality_prior = np.full(row_count, np.nan, dtype=np.float64)
    weak_quality_prior_mask = np.zeros(row_count, dtype=np.int64)
    retrospective_temperature_mean = np.full(row_count, np.nan, dtype=np.float64)
    retrospective_temperature_std = np.full(row_count, np.nan, dtype=np.float64)
    retrospective_humidity_mean = np.full(row_count, np.nan, dtype=np.float64)
    retrospective_window_mask = np.zeros(row_count, dtype=np.int64)

    for row_index, record in enumerate(records):
        endpoint = float(record.timestamp)
        latest_index = int(np.searchsorted(sensor_t, endpoint, side="right") - 1)
        if latest_index >= 0:
            age_s = endpoint - float(sensor_t[latest_index])
            if 0.0 <= age_s <= contract.maximum_causal_sensor_age_s:
                causal_temperature[row_index] = sensor_temperature[latest_index]
                causal_humidity[row_index] = sensor_humidity[latest_index]
                causal_age[row_index] = age_s
                causal_mask[row_index] = 1
                high_temperature_mask[row_index] = int(
                    sensor_temperature[latest_index]
                    > contract.high_temperature_threshold_c
                )

        trailing_start = endpoint - contract.stability_window_s
        trailing_indices = np.flatnonzero(
            (sensor_t >= trailing_start) & (sensor_t <= endpoint)
        )
        if causal_mask[row_index] and trailing_indices.size >= 3:
            trailing_values = sensor_temperature[trailing_indices]
            trailing_std[row_index] = float(np.std(trailing_values, ddof=0))
            trailing_range[row_index] = float(np.ptp(trailing_values))
            trailing_slope[row_index] = _linear_slope_per_hour(
                sensor_t[trailing_indices],
                trailing_values,
            )
            stability_mask[row_index] = 1
            temperature_score = 1.0 / (
                1.0
                + math.exp(
                    (
                        causal_temperature[row_index]
                        - contract.high_temperature_threshold_c
                    )
                    / 0.5
                )
            )
            stability_score = math.exp(
                -0.5
                * (
                    trailing_std[row_index] / contract.stability_scale_c
                ) ** 2
            )
            weak_quality_prior[row_index] = temperature_score * stability_score
            weak_quality_prior_mask[row_index] = 1

        duration_s = float(record.duration_s)
        if duration_s > 0.0:
            window_start = endpoint - duration_s
            window_indices = np.flatnonzero(
                (sensor_t >= window_start) & (sensor_t <= endpoint)
            )
            if window_indices.size >= 2:
                retrospective_temperature_mean[row_index] = float(
                    np.mean(sensor_temperature[window_indices])
                )
                retrospective_temperature_std[row_index] = float(
                    np.std(sensor_temperature[window_indices], ddof=0)
                )
                retrospective_humidity_mean[row_index] = float(
                    np.mean(sensor_humidity[window_indices])
                )
                retrospective_window_mask[row_index] = 1

    feature_payload = {
        "sample_ids": [record.sample_id for record in records],
        "environment_log_hash": environment_log.stable_hash,
        "contract": contract.to_json_dict(),
        "causal_temperature_c": causal_temperature.tolist(),
        "causal_humidity_percent_rh": causal_humidity.tolist(),
        "causal_sensor_age_s": causal_age.tolist(),
        "causal_context_mask": causal_mask.tolist(),
        "trailing_temperature_std_c": trailing_std.tolist(),
        "trailing_temperature_range_c": trailing_range.tolist(),
        "trailing_temperature_slope_c_per_hour": trailing_slope.tolist(),
        "trailing_stability_mask": stability_mask.tolist(),
        "high_temperature_mask": high_temperature_mask.tolist(),
        "weak_quality_prior": weak_quality_prior.tolist(),
        "weak_quality_prior_mask": weak_quality_prior_mask.tolist(),
        "retrospective_temperature_mean_c": retrospective_temperature_mean.tolist(),
        "retrospective_temperature_std_c": retrospective_temperature_std.tolist(),
        "retrospective_humidity_mean_percent_rh": retrospective_humidity_mean.tolist(),
        "retrospective_window_mask": retrospective_window_mask.tolist(),
    }
    return EnvironmentFeaturesV2(
        sample_ids=tuple(record.sample_id for record in records),
        causal_temperature_c=causal_temperature,
        causal_humidity_percent_rh=causal_humidity,
        causal_sensor_age_s=causal_age,
        causal_context_mask=causal_mask,
        trailing_temperature_std_c=trailing_std,
        trailing_temperature_range_c=trailing_range,
        trailing_temperature_slope_c_per_hour=trailing_slope,
        trailing_stability_mask=stability_mask,
        high_temperature_mask=high_temperature_mask,
        weak_quality_prior=weak_quality_prior,
        weak_quality_prior_mask=weak_quality_prior_mask,
        retrospective_temperature_mean_c=retrospective_temperature_mean,
        retrospective_temperature_std_c=retrospective_temperature_std,
        retrospective_humidity_mean_percent_rh=retrospective_humidity_mean,
        retrospective_window_mask=retrospective_window_mask,
        environment_log_hash=environment_log.stable_hash,
        stable_hash=_stable_hash(feature_payload),
    )


def environment_count_quality_analysis(
    records: Sequence[MeasuredSampleRecordV1],
    features: EnvironmentFeaturesV2,
) -> Dict[str, Any]:
    """Report observational count associations without causal promotion."""

    if tuple(record.sample_id for record in records) != features.sample_ids:
        raise ValueError("environment features and measured records are misaligned")
    result: Dict[str, Any] = {
        "schema": "physics_hybrid_v8_environment_count_analysis_v2",
        "causal_claim_allowed": False,
        "hard_label_or_exclusion_authority": False,
        "raw_count_likelihood_weight_change_allowed": False,
        "gsi_metric_available": False,
        "gsi_direction_is_operator_observation_not_measured_label": True,
        "interpretation": (
            "temperature_above_21_5c_and_temperature_instability_are_weak_priors_for_"
            "lower_global_optical_throughput_counts_and_experiment_gsi"
        ),
        "systems": {},
    }
    causal_mask = features.causal_context_mask.astype(bool)
    for system_type in ("3qubit", "4qubit"):
        system_mask = np.asarray(
            [record.system_type == system_type for record in records],
            dtype=bool,
        )
        valid = system_mask & causal_mask
        temperature = features.causal_temperature_c[valid]
        system_report: Dict[str, Any] = {
            "row_count": int(np.sum(system_mask)),
            "environment_matched_row_count": int(np.sum(valid)),
            "temperature_count_correlations_are_confounded": True,
            "correlation": {},
        }
        for count_name in ("signal_total", "idler_total", "coin_total", "read_count"):
            counts = np.asarray(
                [float(getattr(record, count_name)) for record in records],
                dtype=np.float64,
            )[valid]
            finite = np.isfinite(temperature) & np.isfinite(counts)
            correlation = None
            if np.sum(finite) >= 3 and np.std(temperature[finite]) > 0.0:
                correlation = float(
                    np.corrcoef(
                        temperature[finite],
                        np.log1p(np.maximum(counts[finite], 0.0)),
                    )[0, 1]
                )
            system_report["correlation"][f"temperature_vs_log1p_{count_name}"] = (
                correlation
            )
        high = valid & (features.high_temperature_mask.astype(bool))
        low = valid & ~features.high_temperature_mask.astype(bool)
        system_report["temperature_strata"] = {
            "above_21_5c_row_count": int(np.sum(high)),
            "at_or_below_21_5c_row_count": int(np.sum(low)),
            "median_coin_total_above_21_5c": (
                float(np.median([records[i].coin_total for i in np.flatnonzero(high)]))
                if np.any(high) else None
            ),
            "median_coin_total_at_or_below_21_5c": (
                float(np.median([records[i].coin_total for i in np.flatnonzero(low)]))
                if np.any(low) else None
            ),
        }
        result["systems"][system_type] = system_report
    result["analysis_sha256"] = _stable_hash(result)
    return result


def build_environment_analysis_artifact(
    records: Sequence[MeasuredSampleRecordV1],
    environment_log: EnvironmentLogV2,
) -> Dict[str, Any]:
    """Build a hash-bound, aggregate-only V2 weak-quality report."""

    features = align_environment_context(records, environment_log)
    analysis = environment_count_quality_analysis(records, features)
    payload: Dict[str, Any] = {
        "schema": "physics_hybrid_v8_environment_analysis_artifact_v2",
        "contract": environment_log.contract.to_json_dict(),
        "environment_log_stable_hash": environment_log.stable_hash,
        "environment_feature_stable_hash": features.stable_hash,
        "measured_sample_count": len(records),
        "measured_sample_ids_sha256": _stable_hash(
            [record.sample_id for record in records]
        ),
        "causal_context_available_count": int(
            np.sum(features.causal_context_mask)
        ),
        "trailing_stability_available_count": int(
            np.sum(features.trailing_stability_mask)
        ),
        "retrospective_window_available_count": int(
            np.sum(features.retrospective_window_mask)
        ),
        "temperature_above_21_5c_count": int(
            np.sum(
                features.high_temperature_mask.astype(bool)
                & features.causal_context_mask.astype(bool)
            )
        ),
        "temperature_or_humidity_changes_hard_label": False,
        "temperature_or_humidity_changes_likelihood_weight": False,
        "temperature_or_humidity_changes_rho_or_action_mask": False,
        "selection_authority": False,
        "analysis": analysis,
    }
    payload["artifact_sha256"] = _stable_hash(payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-csv", required=True)
    parser.add_argument(
        "--manifest-dir",
        default="ml_recon/reports/measured_data_v2_3",
    )
    parser.add_argument(
        "--v8-config",
        default="ml_recon/configs/physics_hybrid_v8_feedback.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from .feedback_data import load_v8_measured_data_bundle

    config = json.loads(Path(args.v8_config).read_text(encoding="utf-8"))
    bundle = load_v8_measured_data_bundle(
        args.manifest_dir,
        audit_waiver=config.get("measured_audit_waiver"),
        curated_corpus_contract=config.get("operator_curated_corpus"),
    )
    environment_log = load_environment_log(args.environment_csv)
    artifact = build_environment_analysis_artifact(bundle.records, environment_log)
    artifact.update({
        "measured_manifest_hash": bundle.manifest_hash,
        "measured_split_hash": bundle.split_hash,
        "measured_summary_sha256": bundle.summary_sha256,
        "curated_corpus_contract_sha256": bundle.curated_corpus_audit.get(
            "contract_sha256"
        ),
    })
    artifact_without_hash = dict(artifact)
    artifact_without_hash.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _stable_hash(artifact_without_hash)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_HIGH_TEMPERATURE_C",
    "DEFAULT_SENSOR_DISTANCE_M",
    "DEFAULT_SENSOR_SERIAL",
    "ENVIRONMENT_CONTRACT_SCHEMA",
    "ENVIRONMENT_FEATURE_SCHEMA",
    "EnvironmentFeaturesV2",
    "EnvironmentLogV2",
    "EnvironmentSensorContractV2",
    "align_environment_context",
    "build_environment_analysis_artifact",
    "environment_count_quality_analysis",
    "load_environment_log",
]


if __name__ == "__main__":
    main()
