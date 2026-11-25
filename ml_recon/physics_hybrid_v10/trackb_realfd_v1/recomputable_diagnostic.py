"""Source-bound orchestration for independent Track B AD1 metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import math

import torch

from ml_recon.data_v2_4.loaders import V24TrainingDataset

from .artifact_io import PUBLICATION_COMMITTED, publish_new_bytes_with_state
from .contracts import canonical_json_bytes, canonical_sha256
from .diagnostic_runtime import preserve_python_numpy_torch_rng
from .inverse_transformer_pilot import (
    GLOBAL_ACTION_LOSS_NORMALIZATION,
    PER_ROUND_ACTION_LOSS_NORMALIZATION,
    RANDOM_TRAINING_SAMPLING,
    ROUND_COVERAGE_TRAINING_SAMPLING,
    STANDARDIZED_HUBER_DELTA,
    TEN_ROUND_COUNT,
    TrackBTargetNormalizationV1,
    _round_coverage_batch_schedule,
    _training_target_normalization,
)
from .policy_ablation import (
    _bound_v24_source,
    _dataset_source_identity,
    _load_bound_model,
    _matching_trackb_cell,
    _require_matching_source_identity,
    _require_oracle_history_reader,
)
from .recomputable_metrics import (
    MEDIAN_CONVENTION,
    RECOMPUTABLE_METRICS_SCHEMA,
    ROW_EQUAL_METRIC_AGGREGATION,
    ROW_LEDGER_SCHEMA,
    _strict_real,
    evaluate_recomputable_teacher_forced,
    legacy_arrays_from_recomputable_metrics,
    reduce_recomputable_row_records,
    require_legacy_metric_reconstruction,
)
from .train_fit_diagnostic import (
    EQUAL_BATCH_METRIC_AGGREGATION,
    _bound_batch_plan,
    _checkpoint_path_from_receipt,
    _diagnostic_repository_identity,
    _prepare_new_output_path,
    _read_verified_pilot_receipt,
    _require_isolated_report_output_path,
)


RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA = "TrackBRecomputableTrainDiagnosticV1"
SCHEDULE_RECONSTRUCTION_KIND = "RECONSTRUCTED_FROM_BOUND_SCHEDULE_NOT_RUNTIME_RECORDED"
VISIT_NOT_RECORDED = "NOT_RECORDED"

_legacy_arrays_from_recomputable_metrics = legacy_arrays_from_recomputable_metrics
_require_legacy_metric_reconstruction = require_legacy_metric_reconstruction


def _publish_ad1_report(output_path: Path, output: Mapping[str, object]) -> str:
    """Publish an AD1 report under the admitted parent lease, retaining its state."""
    payload = canonical_json_bytes(output)
    publication = publish_new_bytes_with_state(
        output_path, payload, expected_sha=canonical_sha256(output),
    )
    if publication.status != PUBLICATION_COMMITTED:
        raise RuntimeError(
            "Track B AD1 report committed with incomplete staging cleanup: "
            f"target={publication.target_path}, artifact_sha={publication.artifact_sha256}, "
            f"staging={publication.staging_path}, error={publication.cleanup_error}"
        )
    return publication.artifact_sha256


@dataclass(frozen=True)
class TrackBRecomputableTrainDiagnosticConfigV1:
    """Immutable inputs for one independent no-gradient TRAIN diagnostic."""

    pilot_receipt_path: str
    expected_pilot_receipt_sha: str
    output_path: str
    pilot_batch_size: int = 8
    inference_batch_size: int = 64
    device: str = "cpu"
    schema: str = RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "output_path"):
            if not Path(str(getattr(self, name))).is_absolute():
                raise ValueError(f"{name} must be absolute")
        if type(self.expected_pilot_receipt_sha) is not str or len(self.expected_pilot_receipt_sha) != 64:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value")
        try:
            int(self.expected_pilot_receipt_sha, 16)
        except ValueError as error:
            raise ValueError("expected_pilot_receipt_sha must be a SHA-256 hex value") from error
        for name in ("pilot_batch_size", "inference_batch_size"):
            if type(getattr(self, name)) is not int or int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} is invalid")
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("recomputable diagnostic device is invalid") from error
        if device.type != "cpu":
            raise ValueError("recomputable diagnostic supports CPU only")
        if self.schema != RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA:
            raise ValueError("unsupported recomputable Track B diagnostic schema")


def _strict_scale_sequence(value: object, *, name: str) -> tuple[float, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an ordered sequence")
    return tuple(
        _strict_real(item, name=f"{name}[{index}]", positive=True)
        for index, item in enumerate(value)
    )


def _strict_normalization(payload: Mapping[str, Any]) -> TrackBTargetNormalizationV1:
    required = {
        "amplitude_scale", "phase_scale", "gain_center", "gain_scale",
        "action_loss_normalization_kind", "amplitude_scale_by_round",
        "phase_scale_by_round", "schema",
    }
    if set(payload) != required:
        raise ValueError("Track B target normalization fields are invalid")
    kind = payload["action_loss_normalization_kind"]
    if type(kind) is not str or kind not in (
        GLOBAL_ACTION_LOSS_NORMALIZATION,
        PER_ROUND_ACTION_LOSS_NORMALIZATION,
    ):
        raise ValueError("Track B action-loss normalization kind is invalid")
    amplitude_by_round = _strict_scale_sequence(
        payload["amplitude_scale_by_round"],
        name="amplitude_scale_by_round",
    )
    phase_by_round = _strict_scale_sequence(
        payload["phase_scale_by_round"],
        name="phase_scale_by_round",
    )
    if kind == PER_ROUND_ACTION_LOSS_NORMALIZATION:
        if len(amplitude_by_round) != TEN_ROUND_COUNT or len(phase_by_round) != TEN_ROUND_COUNT:
            raise ValueError("per-round Track B normalization requires all ten scales")
    elif amplitude_by_round or phase_by_round:
        raise ValueError("global Track B normalization cannot carry per-round scales")
    if payload["schema"] != "TrackBTargetNormalizationV1":
        raise ValueError("Track B target normalization schema is invalid")
    result = TrackBTargetNormalizationV1(
        amplitude_scale=_strict_real(
            payload["amplitude_scale"], name="amplitude_scale", positive=True,
        ),
        phase_scale=_strict_real(
            payload["phase_scale"], name="phase_scale", positive=True,
        ),
        gain_center=_strict_real(payload["gain_center"], name="gain_center"),
        gain_scale=_strict_real(payload["gain_scale"], name="gain_scale", positive=True),
        action_loss_normalization_kind=kind,
        amplitude_scale_by_round=amplitude_by_round,
        phase_scale_by_round=phase_by_round,
    )
    result.validate()
    return result


def _normalization_from_bound_records(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> TrackBTargetNormalizationV1:
    model_record = receipt.get("model")
    if not isinstance(model_record, Mapping):
        raise ValueError("Track B pilot receipt lacks a model record")
    receipt_payload = model_record.get("target_normalization")
    checkpoint_payload = checkpoint.get("target_normalization")
    if not isinstance(receipt_payload, Mapping) or not isinstance(checkpoint_payload, Mapping):
        raise ValueError("Track B pilot artifacts lack target normalization")
    if canonical_sha256(receipt_payload) != canonical_sha256(checkpoint_payload):
        raise ValueError("Track B checkpoint and receipt target normalization differ")
    normalization = _strict_normalization(checkpoint_payload)
    receipt_config = receipt.get("config")
    if not isinstance(receipt_config, Mapping):
        raise ValueError("Track B pilot receipt lacks configuration")
    if receipt_config.get("action_loss_normalization_kind") != normalization.action_loss_normalization_kind:
        raise ValueError("Track B action-loss normalization declaration differs from its target normalization")
    gain_loss_weight = _strict_real(
        receipt_config.get("gain_loss_weight"),
        name="gain_loss_weight",
    )
    if gain_loss_weight < 0.0:
        raise ValueError("gain_loss_weight must be nonnegative")
    return normalization


def _require_recomputed_train_normalization(
    recorded: TrackBTargetNormalizationV1,
    dataset: V24TrainingDataset,
    *,
    pilot_batch_size: int,
) -> dict[str, object]:
    recomputed = _training_target_normalization(
        dataset,
        batch_size=pilot_batch_size,
        action_loss_normalization_kind=recorded.action_loss_normalization_kind,
    )
    recorded_payload = recorded.to_dict()
    recomputed_payload = recomputed.to_dict()
    if canonical_sha256(recorded_payload) != canonical_sha256(recomputed_payload):
        raise ValueError("Track B recorded target normalization differs from TRAIN recomputation")
    return {
        "recorded_sha": canonical_sha256(recorded_payload),
        "recomputed_sha": canonical_sha256(recomputed_payload),
        "recomputed_from_train": True,
        "scale_floor": 1e-6,
        "gain_variance_kind": "POPULATION_VARIANCE",
    }


def reconstruct_bound_coverage_visits(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    ordered_sample_ids: Sequence[str],
) -> dict[str, object]:
    """Rebuild the coverage plan while denying runtime-observation status."""
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Track B pilot receipt lacks configuration")
    sampling_kind = config.get("training_sampling_kind", RANDOM_TRAINING_SAMPLING)
    if sampling_kind not in (RANDOM_TRAINING_SAMPLING, ROUND_COVERAGE_TRAINING_SAMPLING):
        raise ValueError("Track B pilot receipt has an unknown training sampling kind")
    if sampling_kind != ROUND_COVERAGE_TRAINING_SAMPLING:
        if "batch_schedule_sha" in receipt or "batch_schedule_sha" in checkpoint:
            raise ValueError("legacy random Track B artifacts cannot carry coverage batch metadata")
        return {
            "status": VISIT_NOT_RECORDED,
            "reason": "legacy random visits were not bound to a reconstructable schedule",
            "runtime_visits_recorded": False,
        }
    sample_ids = tuple(str(value) for value in ordered_sample_ids)
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Track B coverage reconstruction requires unique ordered sample identities")
    required = ("batch_size", "optimizer_steps", "seed", "train_samples")
    if any(type(config.get(name)) is not int for name in required):
        raise ValueError("Track B coverage configuration is incomplete")
    if int(config["train_samples"]) != len(sample_ids):
        raise ValueError("Track B coverage source length differs from the receipt")
    batches, rounds = _round_coverage_batch_schedule(
        dataset_size=len(sample_ids),
        batch_size=int(config["batch_size"]),
        optimizer_steps=int(config["optimizer_steps"]),
        seed=int(config["seed"]),
    )
    batch_schedule_sha = canonical_sha256(batches)
    round_schedule_sha = canonical_sha256(rounds)
    for label, source in (("receipt", receipt), ("checkpoint", checkpoint)):
        if source.get("batch_schedule_sha") != batch_schedule_sha:
            raise ValueError(f"reconstructed Track B batch schedule differs from the {label}")
        if source.get("round_schedule_sha") != round_schedule_sha:
            raise ValueError(f"reconstructed Track B round schedule differs from the {label}")
    visit_count_matrix = [[0] * TEN_ROUND_COUNT for _ in sample_ids]
    for batch, round_index in zip(batches, rounds, strict=True):
        for row_index in batch:
            if not 0 <= row_index < len(sample_ids):
                raise ValueError("Track B coverage schedule contains an invalid row index")
            visit_count_matrix[row_index][round_index] += 1
    flattened = [count for row in visit_count_matrix for count in row]
    histogram = Counter(flattened)
    per_round = []
    for round_index in range(TEN_ROUND_COUNT):
        counts = [row[round_index] for row in visit_count_matrix]
        per_round.append({
            "round_index": round_index + 1,
            "minimum_visits": min(counts),
            "maximum_visits": max(counts),
            "mean_visits": math.fsum(counts) / len(counts),
        })
    return {
        "status": SCHEDULE_RECONSTRUCTION_KIND,
        "runtime_visits_recorded": False,
        "ordered_sample_ids_sha": canonical_sha256(list(sample_ids)),
        "batch_schedule_sha": batch_schedule_sha,
        "round_schedule_sha": round_schedule_sha,
        "unique_row_round_pairs": sum(count > 0 for count in flattened),
        "total_presentations": sum(flattened),
        "visit_count_histogram": {str(count): histogram[count] for count in sorted(histogram)},
        "per_round": per_round,
        "visit_count_matrix": visit_count_matrix,
        "visit_count_matrix_sha": canonical_sha256(visit_count_matrix),
    }


def describe_bound_training_progress(receipt: Mapping[str, Any]) -> dict[str, object]:
    progress = receipt.get("training_progress")
    if not isinstance(progress, list):
        return {"status": VISIT_NOT_RECORDED, "reason": "training_progress is absent"}
    round_counts: Counter[int] = Counter()
    optimizer_steps = []
    for record in progress:
        if not isinstance(record, Mapping):
            raise ValueError("Track B training_progress contains an invalid record")
        round_index = record.get("round_index")
        optimizer_step = record.get("optimizer_step")
        if type(round_index) is not int or not 1 <= round_index <= TEN_ROUND_COUNT:
            raise ValueError("Track B training_progress round index is invalid")
        if type(optimizer_step) is not int or optimizer_step <= 0:
            raise ValueError("Track B training_progress optimizer step is invalid")
        round_counts[round_index] += 1
        optimizer_steps.append(optimizer_step)
    represented = sorted(round_counts)
    return {
        "status": "RECORDED_SPARSE_PROGRESS_ONLY",
        "record_count": len(progress),
        "optimizer_steps": optimizer_steps,
        "round_record_counts": {str(index): round_counts[index] for index in represented},
        "all_rounds_represented": represented == list(range(1, TEN_ROUND_COUNT + 1)),
        "historical_optimizer_or_clip_state": VISIT_NOT_RECORDED,
    }


def _run_recomputable_diagnostic(
    config: TrackBRecomputableTrainDiagnosticConfigV1,
) -> tuple[Path, dict[str, object]]:
    output_path = _prepare_new_output_path(Path(config.output_path))
    diagnostic_source_commit, diagnostic_source_tree = _diagnostic_repository_identity()
    receipt_path = Path(config.pilot_receipt_path).resolve()
    receipt = _read_verified_pilot_receipt(receipt_path, config.expected_pilot_receipt_sha)
    _require_oracle_history_reader(receipt, label="Track B recomputable diagnostic")
    batch_plan = _bound_batch_plan(receipt, requested_batch_size=config.pilot_batch_size)
    root, build_id, _identity, manifest = _bound_v24_source(receipt)
    checkpoint_path = _checkpoint_path_from_receipt(receipt)
    _require_isolated_report_output_path(
        output_path,
        dataset_root=root,
        receipt_path=receipt_path,
        checkpoint_path=checkpoint_path,
    )
    model, model_identity, checkpoint = _load_bound_model(receipt)
    normalization = _normalization_from_bound_records(receipt, checkpoint)
    cell = _matching_trackb_cell(manifest, system_type=model.config.system_type)
    train_dataset = V24TrainingDataset(
        root, build_id=build_id, cell_id=str(cell["cell_id"]),
        split="train", task="feedback", arm="A0_RAW",
    )
    validation_dataset = V24TrainingDataset(
        root, build_id=build_id, cell_id=str(cell["cell_id"]),
        split="validation", task="feedback", arm="A0_RAW",
    )
    try:
        receipt_dataset = receipt.get("dataset")
        if not isinstance(receipt_dataset, Mapping):
            raise ValueError("Track B pilot receipt lacks dataset identities")
        source_records = (
            ("train", checkpoint.get("train_source"), receipt_dataset.get("train"), train_dataset),
            (
                "validation", checkpoint.get("validation_source"),
                receipt_dataset.get("validation"), validation_dataset,
            ),
        )
        source_identities: dict[str, dict[str, Any]] = {}
        for label, checkpoint_source, receipt_source, dataset in source_records:
            if not isinstance(checkpoint_source, Mapping) or not isinstance(receipt_source, Mapping):
                raise ValueError(f"Track B pilot receipt lacks bound {label} identity")
            opened_source = _dataset_source_identity(dataset)
            _require_matching_source_identity(
                checkpoint_source, receipt_source, label=f"checkpoint/receipt {label}",
            )
            _require_matching_source_identity(
                checkpoint_source, opened_source, label=f"checkpoint/opened {label}",
            )
            source_identities[label] = opened_source
        normalization_recomputation = _require_recomputed_train_normalization(
            normalization,
            train_dataset,
            pilot_batch_size=config.pilot_batch_size,
        )
        recomputable = evaluate_recomputable_teacher_forced(
            model,
            train_dataset,
            pilot_batch_size=config.pilot_batch_size,
            inference_batch_size=config.inference_batch_size,
            device=torch.device(config.device),
            normalization=normalization,
        )
        coverage = reconstruct_bound_coverage_visits(
            receipt,
            checkpoint,
            ordered_sample_ids=tuple(str(value) for value in train_dataset.sample_ids),
        )
        output = {
            "schema": RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA,
            "config": asdict(config),
            "diagnostic_source_commit": diagnostic_source_commit,
            "diagnostic_source_tree": diagnostic_source_tree,
            "pilot": {
                "receipt_path": str(receipt_path),
                "receipt_file_sha": config.expected_pilot_receipt_sha,
                "pilot_source_commit": receipt.get("source_commit"),
                "pilot_source_tree": receipt.get("source_tree"),
                "checkpoint_sha": model_identity["checkpoint_sha"],
                "checkpoint_model_state_sha": model_identity["model_state_sha"],
            },
            "model": model_identity,
            "batch_plan": {
                **batch_plan,
                "pilot_batch_size": config.pilot_batch_size,
                "inference_batch_size": config.inference_batch_size,
                "shadow_metric_aggregation": ROW_EQUAL_METRIC_AGGREGATION,
            },
            "train_source": source_identities["train"],
            "validation_source": source_identities["validation"],
            "normalization_recomputation": normalization_recomputation,
            "equal_batch_reconstruction": legacy_arrays_from_recomputable_metrics(recomputable),
            "recomputable_evaluation": recomputable,
            "coverage_visit_reconstruction": coverage,
            "training_progress_observation": describe_bound_training_progress(receipt),
            "loss_definition": {
                "action_normalization_kind": normalization.action_loss_normalization_kind,
                "gain_loss_weight": _strict_real(
                    receipt["config"].get("gain_loss_weight"), name="gain_loss_weight",
                ),
                "gain_standardized_huber_beta": STANDARDIZED_HUBER_DELTA,
            },
            "gradient_history": VISIT_NOT_RECORDED,
            "optimizer_history": VISIT_NOT_RECORDED,
            "diagnostic_only": True,
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
            "external_waveform_write": "NOT_IMPLEMENTED",
        }
        return output_path, output
    finally:
        train_dataset.close()
        validation_dataset.close()


def run_trackb_recomputable_train_diagnostic(
    config: TrackBRecomputableTrainDiagnosticConfigV1,
) -> dict[str, object]:
    """Publish one source-bound AD1 diagnostic without model or data mutation."""
    config.validate()
    with preserve_python_numpy_torch_rng():
        output_path, output = _run_recomputable_diagnostic(config)
        output_sha = _publish_ad1_report(output_path, output)
    output["output_path"] = str(output_path)
    output["output_sha"] = output_sha
    output["caller_rng_restored"] = True
    return output


__all__ = [
    "EQUAL_BATCH_METRIC_AGGREGATION",
    "MEDIAN_CONVENTION",
    "RECOMPUTABLE_METRICS_SCHEMA",
    "RECOMPUTABLE_TRAIN_DIAGNOSTIC_SCHEMA",
    "ROW_EQUAL_METRIC_AGGREGATION",
    "ROW_LEDGER_SCHEMA",
    "SCHEDULE_RECONSTRUCTION_KIND",
    "TrackBRecomputableTrainDiagnosticConfigV1",
    "describe_bound_training_progress",
    "evaluate_recomputable_teacher_forced",
    "reconstruct_bound_coverage_visits",
    "reduce_recomputable_row_records",
    "run_trackb_recomputable_train_diagnostic",
]
