"""Fail-closed V2.4-bound ten-round logical guidance protocol for Track B.

The protocol intentionally owns only model-side logical amplitude/phase
guidance.  It never creates RF commands, waveforms, uploads, or hardware
authority.  Every next guidance record requires an externally supplied
post-action observation so an autoregressive model cannot silently treat its
own prediction as a real measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import math
import os
from numbers import Real

from .contracts import canonical_json_bytes, canonical_sha256, require_sha256


TEN_ROUND_COUNT = 10
GUIDANCE_SOURCE_KINDS = frozenset({"v24_synthetic_transition", "external_observation"})
SESSION_STATUSES = frozenset({
    "READY_FOR_FIRST_GUIDANCE",
    "READY_FOR_NEXT_GUIDANCE",
    "WAITING_FOR_EXTERNAL_OBSERVATION",
    "COMPLETE",
})
GUIDANCE_SCHEMA = "TrackBV24TenRoundGuidanceSessionV1"


def _require_git_identity(value: str, name: str) -> None:
    if type(value) is not str or len(value) not in (40, 64):
        raise ValueError(f"{name} must be a Git SHA-1 or SHA-256 identity")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must use lowercase hexadecimal")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_vector(values: Sequence[object], *, count: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__") \
            or not hasattr(values, "__getitem__") or len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} finite values")
    return tuple(_finite_float(value, name) for value in values)


def _measurement_tokens(values: Sequence[Sequence[object]], *, count: int,
                        name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__") \
            or not hasattr(values, "__getitem__") or len(values) != count:
        raise ValueError(f"{name} must contain one two-value token per mode")
    tokens = []
    for index, value in enumerate(values):
        if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") \
                or not hasattr(value, "__getitem__") or len(value) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two values")
        tokens.append((_finite_float(value[0], name), _finite_float(value[1], name)))
    return tuple(tokens)


@dataclass(frozen=True)
class TrackBV24GuidanceSourceBindingV1:
    """Exact V2.4/model identity required before ten-round guidance begins."""

    system_type: str
    v24_contract_sha: str
    v24_schema_sha: str
    v24_generator_sha: str
    train_range_sha: str
    validation_range_sha: str
    model_checkpoint_sha: str
    model_state_sha: str
    model_source_commit: str
    model_source_tree: str
    training_cells: tuple[str, ...]
    validation_cells: tuple[str, ...]
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBV24GuidanceSourceBindingV1"

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("ten-round guidance supports only 3qubit or 4qubit")
        for name in (
            "v24_contract_sha",
            "v24_schema_sha",
            "v24_generator_sha",
            "train_range_sha",
            "validation_range_sha",
            "model_checkpoint_sha",
            "model_state_sha",
        ):
            require_sha256(getattr(self, name), name)
        _require_git_identity(self.model_source_commit, "model_source_commit")
        _require_git_identity(self.model_source_tree, "model_source_tree")
        if not self.training_cells or not self.validation_cells:
            raise ValueError("ten-round guidance requires nonempty V2.4 train and validation cells")
        expected_history_cell = f"feedback_history_n{self.system_type[0]}"
        if expected_history_cell not in self.training_cells or expected_history_cell not in self.validation_cells:
            raise ValueError("ten-round guidance requires the matching V2.4 feedback_history cell")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("ten-round guidance cannot assert selection, promotion, or hardware authority")
        if self.schema != "TrackBV24GuidanceSourceBindingV1":
            raise ValueError("unsupported ten-round source binding schema")

    @property
    def mode_count(self) -> int:
        return int(self.system_type[0])

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "system_type": self.system_type,
            "v24_contract_sha": self.v24_contract_sha,
            "v24_schema_sha": self.v24_schema_sha,
            "v24_generator_sha": self.v24_generator_sha,
            "train_range_sha": self.train_range_sha,
            "validation_range_sha": self.validation_range_sha,
            "model_checkpoint_sha": self.model_checkpoint_sha,
            "model_state_sha": self.model_state_sha,
            "model_source_commit": self.model_source_commit,
            "model_source_tree": self.model_source_tree,
            "training_cells": list(self.training_cells),
            "validation_cells": list(self.validation_cells),
            "research_only": True,
            "selection_evidence": False,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        })


@dataclass(frozen=True)
class TrackBGuidanceObservationV1:
    """One V2.4-shaped observation supplied before or after a logical action."""

    sample_id: str
    round_index: int
    observation_sha: str
    model_input_sha: str
    measurement_tokens: tuple[tuple[float, float], ...]
    source_kind: str
    observed_gain: float | None = None
    preceding_guidance_sha: str | None = None
    hardware_action_authorized: bool = False
    schema: str = "TrackBGuidanceObservationV1"

    def validate(self, *, mode_count: int) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("guidance observation sample_id is required")
        if type(self.round_index) is not int or not 0 <= self.round_index <= TEN_ROUND_COUNT:
            raise ValueError("guidance observation round index is invalid")
        require_sha256(self.observation_sha, "observation_sha")
        require_sha256(self.model_input_sha, "model_input_sha")
        _measurement_tokens(self.measurement_tokens, count=mode_count, name="measurement_tokens")
        if self.source_kind not in GUIDANCE_SOURCE_KINDS:
            raise ValueError("guidance observation source kind is unsupported")
        if self.observed_gain is not None:
            _finite_float(self.observed_gain, "observed_gain")
        if self.preceding_guidance_sha is not None:
            require_sha256(self.preceding_guidance_sha, "preceding_guidance_sha")
        if self.hardware_action_authorized:
            raise ValueError("guidance observations cannot self-authorize hardware")
        if self.schema != "TrackBGuidanceObservationV1":
            raise ValueError("unsupported guidance observation schema")


@dataclass(frozen=True)
class TrackBLogicalGuidanceV1:
    """One model-only logical amplitude/phase recommendation."""

    amplitude_delta: tuple[float, ...]
    phase_delta: tuple[float, ...]
    model_output_sha: str
    predicted_gain: float
    predicted_harm_probability: float
    predicted_stop_probability: float
    hardware_action_authorized: bool = False
    schema: str = "TrackBLogicalGuidanceV1"

    def validate(self, *, mode_count: int) -> None:
        _finite_vector(self.amplitude_delta, count=mode_count, name="amplitude_delta")
        _finite_vector(self.phase_delta, count=mode_count, name="phase_delta")
        require_sha256(self.model_output_sha, "model_output_sha")
        _finite_float(self.predicted_gain, "predicted_gain")
        for name in ("predicted_harm_probability", "predicted_stop_probability"):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.hardware_action_authorized:
            raise ValueError("logical guidance cannot self-authorize hardware")
        if self.schema != "TrackBLogicalGuidanceV1":
            raise ValueError("unsupported logical guidance schema")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "amplitude_delta": list(self.amplitude_delta),
            "phase_delta": list(self.phase_delta),
            "model_output_sha": self.model_output_sha,
            "predicted_gain": self.predicted_gain,
            "predicted_harm_probability": self.predicted_harm_probability,
            "predicted_stop_probability": self.predicted_stop_probability,
            "hardware_action_authorized": False,
        })


@dataclass(frozen=True)
class TrackBTenRoundGuidanceRecordV1:
    """One requested action and its required externally supplied outcome."""

    round_index: int
    before_observation: TrackBGuidanceObservationV1
    guidance: TrackBLogicalGuidanceV1
    after_observation: TrackBGuidanceObservationV1 | None = None
    schema: str = "TrackBTenRoundGuidanceRecordV1"

    def validate(self, *, mode_count: int) -> None:
        if type(self.round_index) is not int or not 1 <= self.round_index <= TEN_ROUND_COUNT:
            raise ValueError("guidance record round index is invalid")
        self.before_observation.validate(mode_count=mode_count)
        if self.before_observation.round_index != self.round_index - 1:
            raise ValueError("guidance record before observation does not bind the preceding round")
        self.guidance.validate(mode_count=mode_count)
        if self.after_observation is not None:
            self.after_observation.validate(mode_count=mode_count)
            if self.after_observation.round_index != self.round_index:
                raise ValueError("guidance record after observation has the wrong round index")
            if self.after_observation.preceding_guidance_sha != self.guidance.sha256():
                raise ValueError("guidance record after observation does not bind the issued guidance")
            if self.after_observation.observed_gain is None:
                raise ValueError("post-action observation must carry an observed gain")
        if self.schema != "TrackBTenRoundGuidanceRecordV1":
            raise ValueError("unsupported ten-round guidance record schema")

    def history_token(self, *, mode_count: int) -> tuple[tuple[float, ...], ...]:
        """Return the V2.4 `(y_t, u_t, y_t_plus_1, G_t)` token for this round."""
        self.validate(mode_count=mode_count)
        if self.after_observation is None:
            raise ValueError("a history token requires an externally observed post-action result")
        before = _measurement_tokens(
            self.before_observation.measurement_tokens, count=mode_count, name="before_measurement_tokens")
        after = _measurement_tokens(
            self.after_observation.measurement_tokens, count=mode_count, name="after_measurement_tokens")
        gain = _finite_float(self.after_observation.observed_gain, "observed_gain")
        return tuple(
            (before[index][0], before[index][1], self.guidance.amplitude_delta[index],
             self.guidance.phase_delta[index], after[index][0], after[index][1], gain)
            for index in range(mode_count)
        )


@dataclass(frozen=True)
class TrackBTenRoundGuidanceSessionV1:
    """Immutable, resumable ten-round session with no waveform capability."""

    source_binding: TrackBV24GuidanceSourceBindingV1
    records: tuple[TrackBTenRoundGuidanceRecordV1, ...] = ()
    status: str = "READY_FOR_FIRST_GUIDANCE"
    schema: str = GUIDANCE_SCHEMA

    def __post_init__(self) -> None:
        self.source_binding.__post_init__()
        if self.status not in SESSION_STATUSES:
            raise ValueError("ten-round guidance session status is invalid")
        if len(self.records) > TEN_ROUND_COUNT:
            raise ValueError("ten-round guidance session exceeds its fixed round budget")
        for index, record in enumerate(self.records, start=1):
            record.validate(mode_count=self.source_binding.mode_count)
            if record.round_index != index:
                raise ValueError("ten-round guidance records must be contiguous")
            if index > 1:
                previous = self.records[index - 2]
                if previous.after_observation is None:
                    raise ValueError("a later guidance record requires the prior external observation")
                if record.before_observation.observation_sha != previous.after_observation.observation_sha:
                    raise ValueError("guidance record lost the preceding externally observed state")
        expected_status = self._expected_status()
        if self.status != expected_status:
            raise ValueError("ten-round guidance session status does not match its records")
        if self.schema != GUIDANCE_SCHEMA:
            raise ValueError("unsupported ten-round guidance session schema")

    def _expected_status(self) -> str:
        if not self.records:
            return "READY_FOR_FIRST_GUIDANCE"
        latest = self.records[-1]
        if latest.after_observation is None:
            return "WAITING_FOR_EXTERNAL_OBSERVATION"
        if len(self.records) == TEN_ROUND_COUNT:
            return "COMPLETE"
        return "READY_FOR_NEXT_GUIDANCE"

    @property
    def mode_count(self) -> int:
        return self.source_binding.mode_count

    @property
    def completed_round_count(self) -> int:
        return sum(record.after_observation is not None for record in self.records)

    def history_tokens(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        return tuple(
            record.history_token(mode_count=self.mode_count)
            for record in self.records if record.after_observation is not None
        )

    def issue_guidance(
        self,
        *,
        before_observation: TrackBGuidanceObservationV1,
        guidance: TrackBLogicalGuidanceV1,
    ) -> "TrackBTenRoundGuidanceSessionV1":
        if self.status not in ("READY_FOR_FIRST_GUIDANCE", "READY_FOR_NEXT_GUIDANCE"):
            raise ValueError("guidance can be issued only after the prior external observation")
        next_round = len(self.records) + 1
        before_observation.validate(mode_count=self.mode_count)
        guidance.validate(mode_count=self.mode_count)
        if before_observation.round_index != next_round - 1:
            raise ValueError("guidance must use the current round's preceding observation")
        if self.records:
            previous = self.records[-1]
            if previous.after_observation is None:
                raise ValueError("guidance cannot skip the prior external observation")
            if before_observation.observation_sha != previous.after_observation.observation_sha:
                raise ValueError("guidance must consume the exact prior external observation")
        record = TrackBTenRoundGuidanceRecordV1(next_round, before_observation, guidance)
        return replace(self, records=(*self.records, record),
                       status="WAITING_FOR_EXTERNAL_OBSERVATION")

    def record_external_observation(
        self,
        after_observation: TrackBGuidanceObservationV1,
    ) -> "TrackBTenRoundGuidanceSessionV1":
        if self.status != "WAITING_FOR_EXTERNAL_OBSERVATION" or not self.records:
            raise ValueError("an external observation is not currently expected")
        latest = self.records[-1]
        after_observation.validate(mode_count=self.mode_count)
        if after_observation.round_index != latest.round_index:
            raise ValueError("external observation does not belong to the pending round")
        if after_observation.preceding_guidance_sha != latest.guidance.sha256():
            raise ValueError("external observation does not bind the pending logical guidance")
        if after_observation.observed_gain is None:
            raise ValueError("external observation must report its observed gain")
        completed = replace(latest, after_observation=after_observation)
        new_records = (*self.records[:-1], completed)
        new_status = "COMPLETE" if len(new_records) == TEN_ROUND_COUNT else "READY_FOR_NEXT_GUIDANCE"
        return replace(self, records=new_records, status=new_status)

    def selected_final_record(self) -> TrackBTenRoundGuidanceRecordV1:
        """The user-defined final candidate is round ten, never a post-hoc best pick."""
        if self.status != "COMPLETE" or len(self.records) != TEN_ROUND_COUNT:
            raise ValueError("the final logical candidate exists only after all ten observed rounds")
        return self.records[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_binding": {
                "schema": self.source_binding.schema,
                "sha256": self.source_binding.sha256(),
                "system_type": self.source_binding.system_type,
                "v24_contract_sha": self.source_binding.v24_contract_sha,
                "v24_schema_sha": self.source_binding.v24_schema_sha,
                "v24_generator_sha": self.source_binding.v24_generator_sha,
                "train_range_sha": self.source_binding.train_range_sha,
                "validation_range_sha": self.source_binding.validation_range_sha,
                "model_checkpoint_sha": self.source_binding.model_checkpoint_sha,
                "model_state_sha": self.source_binding.model_state_sha,
                "model_source_commit": self.source_binding.model_source_commit,
                "model_source_tree": self.source_binding.model_source_tree,
                "training_cells": list(self.source_binding.training_cells),
                "validation_cells": list(self.source_binding.validation_cells),
                "research_only": True,
                "selection_evidence": False,
                "promotion_ready": False,
                "hardware_action_authorized": False,
            },
            "round_budget": TEN_ROUND_COUNT,
            "completed_round_count": self.completed_round_count,
            "status": self.status,
            "records": [_record_payload(record, mode_count=self.mode_count) for record in self.records],
            "history_token_count": len(self.history_tokens()),
            "final_round_is_selected": self.status == "COMPLETE",
            "selected_round_index": TEN_ROUND_COUNT if self.status == "COMPLETE" else None,
            "waveform_generation": "NOT_IMPLEMENTED",
            "hardware_action_authorized": False,
        }


def _observation_payload(value: TrackBGuidanceObservationV1) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "sample_id": value.sample_id,
        "round_index": value.round_index,
        "observation_sha": value.observation_sha,
        "model_input_sha": value.model_input_sha,
        "measurement_tokens": [list(token) for token in value.measurement_tokens],
        "source_kind": value.source_kind,
        "observed_gain": value.observed_gain,
        "preceding_guidance_sha": value.preceding_guidance_sha,
        "hardware_action_authorized": False,
    }


def _guidance_payload(value: TrackBLogicalGuidanceV1) -> dict[str, Any]:
    return {
        "schema": value.schema,
        "sha256": value.sha256(),
        "amplitude_delta": list(value.amplitude_delta),
        "phase_delta": list(value.phase_delta),
        "model_output_sha": value.model_output_sha,
        "predicted_gain": value.predicted_gain,
        "predicted_harm_probability": value.predicted_harm_probability,
        "predicted_stop_probability": value.predicted_stop_probability,
        "hardware_action_authorized": False,
    }


def _record_payload(value: TrackBTenRoundGuidanceRecordV1, *, mode_count: int) -> dict[str, Any]:
    value.validate(mode_count=mode_count)
    return {
        "schema": value.schema,
        "round_index": value.round_index,
        "before_observation": _observation_payload(value.before_observation),
        "guidance": _guidance_payload(value.guidance),
        "after_observation": (
            None if value.after_observation is None else _observation_payload(value.after_observation)),
        "history_token": (
            None if value.after_observation is None
            else [list(token) for token in value.history_token(mode_count=mode_count)]),
    }


def write_ten_round_guidance_session(
    path: str | Path,
    session: TrackBTenRoundGuidanceSessionV1,
) -> str:
    """Atomically persist a model-only session record and return its SHA-256."""
    destination = Path(path)
    payload = canonical_json_bytes(session.to_dict())
    if destination.exists():
        existing = destination.read_bytes()
        if existing != payload:
            raise ValueError("ten-round guidance record already exists with different bytes")
        return canonical_sha256(session.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return canonical_sha256(session.to_dict())


__all__ = [
    "GUIDANCE_SCHEMA",
    "GUIDANCE_SOURCE_KINDS",
    "SESSION_STATUSES",
    "TEN_ROUND_COUNT",
    "TrackBGuidanceObservationV1",
    "TrackBLogicalGuidanceV1",
    "TrackBTenRoundGuidanceRecordV1",
    "TrackBTenRoundGuidanceSessionV1",
    "TrackBV24GuidanceSourceBindingV1",
    "write_ten_round_guidance_session",
]
