"""Bounded, observation-only training artifacts; never a scientific gate.

The trainer owns optimizer steps. This module receives detached scalar records,
does not inspect checkpoints with torch.load, and does not touch model RNGs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict

from .checkpoint_storage import MAX_METRICS_JOURNAL_BYTES, _safe_read_bytes


TELEMETRY_SCHEMA = "V10TrainingTelemetryV1"
TELEMETRY_INTERVAL_STEPS = 20
TELEMETRY_INTERVAL_SECONDS = 60.0
MAX_TELEMETRY_JOURNAL_BYTES = MAX_METRICS_JOURNAL_BYTES
MAX_TELEMETRY_CELLS = 64
AUTHORITY = MappingProxyType({"research_only": True, "selection_evidence": False,
             "promotion_ready": False, "ten_k_training_authorized": False,
             "hardware_action_authorized": False})


def atomic_training_json(path: Path, value: object) -> None:
    """Replace a producer-owned JSON file using an exclusive same-volume temp."""
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=True,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=path.name + "_", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_training_record(path: Path, value: object) -> None:
    """Append complete JSON lines; never repair or overwrite a damaged history."""
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=True,
                          allow_nan=False) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                or info.st_nlink != 1 or info.st_size + len(encoded) > MAX_TELEMETRY_JOURNAL_BYTES):
            raise ValueError("unsafe or oversized training journal")
        if info.st_size:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError("training journal has an incomplete final record")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def append_committed_metric(path: Path, value: Dict[str, Any], *,
                            execution_id: str, generation_id: str,
                            checkpoint_sha256: str, global_step: int) -> None:
    """Append one metric only after its checkpoint generation is committed."""
    required = {"execution_id": execution_id, "generation_id": generation_id,
                "checkpoint_sha256": checkpoint_sha256, "global_step": global_step}
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("metric record is not bound to the committed checkpoint")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("metrics journal global step is invalid")
    if path.exists():
        prior_rows = []
        for raw in path.read_bytes().splitlines():
            try:
                prior = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("metrics journal is malformed") from exc
            if not isinstance(prior, dict):
                raise ValueError("metrics journal contains a non-object record")
            for key in ("execution_id", "generation_id", "checkpoint_sha256", "global_step"):
                if key not in prior:
                    raise ValueError("metrics journal record is not checkpoint-bound")
            if type(prior["global_step"]) is not int or prior["global_step"] < 0:
                raise ValueError("metrics journal global step is invalid")
            prior_rows.append(prior)
            if (prior.get("execution_id"), prior.get("generation_id"),
                    prior.get("global_step")) == (execution_id, generation_id, global_step):
                raise ValueError("duplicate committed metric record")
        if prior_rows:
            previous_step = prior_rows[-1]["global_step"]
            if global_step < previous_step:
                raise ValueError("metrics journal global step regressed")
            if global_step == previous_step and prior_rows[-1].get("training_complete") is True:
                raise ValueError("metrics journal appended after completion")
    append_training_record(path, value)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def validate_committed_metrics(output_dir: Path, *, execution_id: str,
                               final_commit: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Validate the complete journal against every retained generation manifest."""
    journal_binding = final_commit.get("metrics_journal")
    if not isinstance(journal_binding, dict) or not isinstance(journal_binding.get("path"), str):
        raise ValueError("descriptor lacks an immutable metrics journal")
    relative = journal_binding["path"]
    if relative.startswith("/") or "\\" in relative or ".." in relative.split("/"):
        raise ValueError("metrics journal path is unsafe")
    path = (output_dir / relative).resolve()
    generation_root = (output_dir / ".checkpoint_generations" / final_commit["generation_id"]).resolve()
    if path.parent != generation_root or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError("metrics journal is unavailable")
    payload = _safe_read_bytes(path, maximum_bytes=MAX_METRICS_JOURNAL_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    if {"sha256": digest, "bytes": len(payload)} != {
            "sha256": journal_binding.get("sha256"), "bytes": journal_binding.get("bytes")}:
        raise ValueError("metrics journal identity mismatch")
    manifest_path = output_dir / final_commit["manifest"]["path"]
    manifest = json.loads(_safe_read_bytes(manifest_path),
                          object_pairs_hook=_reject_duplicate_json_keys,
                          parse_constant=_reject_nonfinite_json)
    if manifest.get("metrics_journal") != {"sha256": journal_binding.get("sha256"),
                                            "bytes": journal_binding.get("bytes")}:
        raise ValueError("metrics journal manifest binding mismatch")
    rows = []
    for raw in payload.splitlines():
        try:
            row = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys,
                             parse_constant=_reject_nonfinite_json)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("metrics journal is malformed") from exc
        if not isinstance(row, dict):
            raise ValueError("metrics journal record is not an object")
        required = {"execution_id", "generation_id", "checkpoint_sha256", "global_step", "epoch",
                    "training_complete"}
        if not required.issubset(row):
            raise ValueError("metrics journal record identity is invalid")
        if (type(row["global_step"]) is not int or row["global_step"] < 0
                or type(row["epoch"]) is not int or row["epoch"] < 0
                or type(row["training_complete"]) is not bool):
            raise ValueError("metrics journal record types are invalid")
        rows.append(row)
    if not rows or rows[-1]["generation_id"] != final_commit["generation_id"]:
        raise ValueError("metrics journal does not terminate at final generation")
    previous_step = -1
    previous_epoch = -1
    seen_generations = set()
    current_generations = set()
    last_by_execution = {execution: max(index for index, item in enumerate(rows)
                                        if item["execution_id"] == execution)
                         for execution in {item["execution_id"] for item in rows}}
    for index, row in enumerate(rows):
        if row["training_complete"] is True and index != last_by_execution[row["execution_id"]]:
            raise ValueError("metrics journal contains a non-final completed record")
        if row["global_step"] < previous_step or row["epoch"] < previous_epoch:
            raise ValueError("metrics journal is not monotonic")
        if row["generation_id"] in seen_generations:
            raise ValueError("metrics journal generation is duplicated")
        seen_generations.add(row["generation_id"])
        if row["execution_id"] == execution_id:
            current_generations.add(row["generation_id"])
        manifest = row.get("generation_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("metrics row lacks immutable generation binding")
        if (manifest.get("execution_id") != row["execution_id"]
                or manifest.get("checkpoint_sha256") != row["checkpoint_sha256"]
                or manifest.get("global_step") != row["global_step"]
                or manifest.get("epoch") != row["epoch"]
                or manifest.get("training_complete") != row["training_complete"]):
            raise ValueError("metrics journal generation binding mismatch")
        previous_step = row["global_step"]
        previous_epoch = row["epoch"]
    if rows[-1].get("training_complete") is not True:
        raise ValueError("metrics journal final record is incomplete")
    return rows


def archive_training_summary(path: Path) -> Path | None:
    """Preserve old completion bytes before an explicit continuation starts."""
    if not path.exists():
        return None
    info = path.lstat()
    if (path.name not in {"training_summary.json", "gradient_audit_summary.json", "training_completion.json", "checkpoint_validation.json"}
            or not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink != 1
            or info.st_size > 2 * 1024 * 1024):
        raise ValueError("unsafe previous training summary")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    destination = path.with_name(path.stem + "_previous_" + digest + ".json")
    if destination.parent.resolve() != path.parent.resolve():
        raise ValueError("summary archive escapes its output directory")
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != content:
            raise ValueError("previous summary archive identity mismatch")
        path.unlink()
    else:
        os.replace(path, destination)
    return destination


class Training_telemetry:
    """Record scalar facts at first step, step/time boundaries and lifecycle edges."""

    def __init__(self, output_dir: Path, *, identity: Dict[str, Any], model_id: str,
                 track_id: str, run_id: str, enabled: bool = True,
                 interval_steps: int = TELEMETRY_INTERVAL_STEPS,
                 interval_seconds: float = TELEMETRY_INTERVAL_SECONDS,
                 monotonic_clock: Callable[[], float] = time.monotonic):
        self.output_dir = output_dir
        self.identity = dict(identity)
        self.enabled = enabled
        self.interval_steps = interval_steps
        self.interval_seconds = interval_seconds
        self.clock = monotonic_clock
        self.started_at = self.clock()
        self.last_record_at = self.started_at
        self.last_record_step = 0
        self.first_step_recorded = False
        self.error_kind = None
        self.execution_id = uuid.uuid4().hex
        self.current = {
            "schema": TELEMETRY_SCHEMA, "track_id": track_id,
            "run_id": run_id or None, "model_id": model_id,
            "source_identity": dict(identity), "producer_pid": os.getpid(),
            "execution_id": self.execution_id, "initial_global_step": 0,
            "optimizer_steps_this_execution": 0,
            "producer_started_at": datetime.now(timezone.utc).isoformat(),
            "phase": "training_preflight", "optimizer_started": False,
            "global_step": 0, "epoch": None, "cell_id": None,
            "latest_step": None, "cell_metrics": {}, "checkpoint": None,
            "field_status": {"gpu_utilization": "NOT_COLLECTED",
                             "science_audit": "NOT_RUN",
                             "run_id": "RECORDED" if run_id else "NOT_REGISTERED"},
            "authority": dict(AUTHORITY), "scientific_gate_evaluated": False,
            "cadence": {"steps": interval_steps, "seconds": interval_seconds},
        }

    def _publish(self, event: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = self.clock()
        step = self.current["global_step"]
        first_step = event == "step" and not self.first_step_recorded
        if (not force and not first_step
                and step // self.interval_steps <= self.last_record_step // self.interval_steps
                and now - self.last_record_at < self.interval_seconds):
            return
        self.current.update(observed_at=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=max(0.0, now - self.started_at), event=event)
        try:
            append_training_record(self.output_dir / "training_steps.jsonl", self.current)
            atomic_training_json(self.output_dir / "training_progress.json", self.current)
        except (OSError, ValueError, TypeError) as exc:
            # Optional observability cannot interrupt an optimizer. Consumers see
            # stale/missing artifacts and the final summary records this failure.
            self.error_kind = type(exc).__name__
            return
        self.last_record_at = now
        self.last_record_step = step
        self.first_step_recorded |= event == "step"

    def begin(self, global_step: int, epoch: int) -> None:
        self.current.update(global_step=global_step, epoch=epoch,
                            initial_global_step=global_step, optimizer_steps_this_execution=0,
                            optimizer_started=False, phase="training_preflight")
        self._publish("preflight", force=True)

    def record_step(self, record: Dict[str, Any]) -> None:
        current_steps = record["global_step"] - self.current["initial_global_step"]
        self.current.update(global_step=record["global_step"], epoch=record["epoch"],
                            cell_id=record.get("cell_id"), latest_step=record,
                            optimizer_steps_this_execution=current_steps,
                            optimizer_started=current_steps > 0,
                            phase="training" if current_steps > 0 else "training_preflight")
        cell_id = record.get("cell_id")
        if cell_id is not None:
            cells = self.current["cell_metrics"]
            if cell_id not in cells and len(cells) >= MAX_TELEMETRY_CELLS:
                self.error_kind = "CellLimitExceeded"
            else:
                cells[cell_id] = record
        skipped = record.get("optimizer_step_executed") is False
        self._publish("optimizer_step_skipped" if skipped else "step", force=skipped)

    def record_epoch(self, epoch: int, train: Dict[str, float],
                     validation: Dict[str, float]) -> None:
        self.current.update(epoch=epoch, epoch_metrics={"train": dict(train),
                                                       "validation": dict(validation)})
        self._publish("epoch_end", force=True)

    def record_phase(self, phase: str, *, error_kind: str | None = None) -> None:
        if phase not in {"training_preflight", "training", "evaluating", "completed", "failed"}:
            raise ValueError("unknown training telemetry phase")
        self.current.update(phase=phase, failure_kind=error_kind)
        self._publish(phase, force=True)

    def record_checkpoint(self, path: Path, *, epoch: int, global_step: int,
                          training_complete: bool, model_state_sha: str,
                          commit: Dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if commit is None:
            raise ValueError("checkpoint observation requires a committed descriptor")
        if commit.get("execution_id") != self.execution_id:
            raise ValueError("checkpoint observation execution identity mismatch")
        if commit.get("global_step") != global_step or commit.get("model_state_sha") != model_state_sha:
            raise ValueError("checkpoint observation identity mismatch")
        self.current["checkpoint"] = dict(commit)

    def summary(self) -> Dict[str, Any]:
        return {"schema": TELEMETRY_SCHEMA, "enabled": self.enabled,
                "execution_id": self.execution_id,
                "status": "DISABLED" if not self.enabled else (
                    "OBSERVATION_FAILED" if self.error_kind else "RECORDED"),
                "error_kind": self.error_kind,
                "progress_path": str(self.output_dir / "training_progress.json") if self.enabled else None,
                "scientific_gate_evaluated": False}


def training_step_record(*, batch: Dict[str, Any], scalar_losses: Dict[str, float],
                         epoch: int, global_step: int, optimizer_step_executed: bool,
                         gradient_norm: float, learning_rate: float, amp_active: bool,
                         amp_scale: float, overflow_count: int, step_seconds: float,
                         attempt_index: int, execution_id: str, amp_scale_before: float,
                         overflow_counter_scope: str,
                         generator_wait_seconds: float, device: str) -> Dict[str, Any]:
    """Extract declared sample/mask counts without persisting observations or truth."""
    import torch

    cell = batch.get("cell_id")
    if isinstance(cell, (list, tuple)) and cell and len(set(cell)) == 1:
        cell = cell[0]
    if not isinstance(cell, str) or not cell or len(cell) > 128:
        cell = None
    dimensions = {int(value.shape[0]) for value in batch.values()
                  if isinstance(value, torch.Tensor) and value.ndim > 0}
    batch_samples = next(iter(dimensions)) if len(dimensions) == 1 else None
    row_mask = batch.get("row_valid_mask")
    valid_samples = None
    if (isinstance(row_mask, torch.Tensor) and row_mask.dtype == torch.bool
            and row_mask.ndim == 1 and row_mask.shape[0] == batch_samples):
        valid_samples = int(row_mask.sum().item())
    finite_norm = math.isfinite(gradient_norm)
    return {
        "epoch": epoch, "global_step": global_step, "cell_id": cell,
        "attempt_index": attempt_index, "execution_id": execution_id,
        "optimizer_step_executed": bool(optimizer_step_executed),
        "losses": dict(scalar_losses), "loss_reduction": "trainer_step_scalar",
        "batch_samples": batch_samples, "valid_samples": valid_samples,
        "valid_samples_status": "RECORDED" if valid_samples is not None else "NOT_COLLECTED",
        "gradient_norm_before_clip": gradient_norm if finite_norm else None,
        "gradient_norm_status": "RECORDED" if finite_norm else "NONFINITE",
        "learning_rate": learning_rate, "device": device,
        "amp": {"active": bool(amp_active), "scale": amp_scale,
                "scale_used_for_backward": amp_scale_before, "scale_after_update": amp_scale,
                "overflow_count": overflow_count, "overflow_counter_scope": overflow_counter_scope},
        "throughput": {"step_seconds": step_seconds,
                       "generator_wait_seconds": generator_wait_seconds,
                       "samples_per_second": (batch_samples / step_seconds
                                              if batch_samples is not None and step_seconds > 0 else None)},
    }
