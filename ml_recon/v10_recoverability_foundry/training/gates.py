"""Immutable, dependency-bound Track A gate and authorization evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..config import GateConfig
from ..contracts import GateRecord, GateStatus, TrackAAuthority
from .gate_specs import (
    RESEARCH_AUTHORITY,
    gate_spec,
    validate_gate_evidence,
)
from ml_recon.physics_hybrid_v10.evidence_snapshot import (
    load_json_snapshot,
    reject_json_constant,
    strict_object_pairs,
)


TRACK_A_GATE_ORDER = (
    "P0",
    "E0",
    "E1",
    "E2",
    "E3",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
)

TRACK_A_GATE_DEPENDENCIES = {
    gate_id: TRACK_A_GATE_ORDER[:index]
    for index, gate_id in enumerate(TRACK_A_GATE_ORDER)
}

TRACK_A_GATE_LEDGER_SCHEMA = "V10TrackAGateLedgerV4"
TRACK_A_EXTERNAL_GATE_SCHEMA = "V10TrackAExternalGateArtifactV3"
HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string SHA-256 identity")
    raw = value
    normalized = raw.strip().lower()
    if (
        raw != normalized
        or len(normalized) != 64
        or any(character not in HEX_DIGITS for character in normalized)
    ):
        raise ValueError(f"{field_name} must be a canonical hexadecimal SHA-256")
    return normalized


def _require_git_identity(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string Git identity")
    raw = value
    normalized = raw.strip().lower()
    if (
        raw != normalized
        or len(normalized) not in (40, 64)
        or any(character not in HEX_DIGITS for character in normalized)
    ):
        raise ValueError(f"{field_name} must be a complete hexadecimal Git identity")
    return normalized


@dataclass(frozen=True)
class GateLedgerEvent:
    event_index: int
    action: str
    gate_id: str
    status: GateStatus
    evidence_sha: str
    notes: str
    parent_evidence_shas: Tuple[Tuple[str, str], ...]
    parent_content_shas: Tuple[Tuple[str, str], ...]
    previous_event_sha: str
    previous_gate_content_sha: str
    evidence_origin: str
    artifact_file_sha: str
    authorization_scope: str
    gate_spec_sha: str
    content_sha: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_index": self.event_index,
            "action": self.action,
            "gate_id": self.gate_id,
            "status": self.status.value,
            "evidence_sha": self.evidence_sha,
            "notes": self.notes,
            "parent_evidence_shas": dict(self.parent_evidence_shas),
            "parent_content_shas": dict(self.parent_content_shas),
            "previous_event_sha": self.previous_event_sha,
            "previous_gate_content_sha": self.previous_gate_content_sha,
            "evidence_origin": self.evidence_origin,
            "artifact_file_sha": self.artifact_file_sha,
            "authorization_scope": self.authorization_scope,
            "gate_spec_sha": self.gate_spec_sha,
            "content_sha": self.content_sha,
        }


def validate_external_gate_artifact_snapshot(
    *,
    raw_bytes: bytes,
    parsed_value: Mapping[str, object],
    expected_gate_id: str,
    expected_config_sha: str,
    expected_source_tree_sha: str,
    expected_parent_evidence_shas: Mapping[str, str],
    expected_parent_content_shas: Mapping[str, str],
    requested_scope: str,
    gate_config: GateConfig,
) -> Dict[str, Any]:
    """Validate a Gate artifact already decoded from one bounded snapshot."""

    if not isinstance(raw_bytes, bytes):
        raise TypeError("gate artifact snapshot bytes are required")
    if len(raw_bytes) > 32 * 1024 * 1024:
        raise ValueError("gate artifact exceeds the admission size limit")
    try:
        value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=strict_object_pairs,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("gate artifact snapshot is not canonical JSON") from exc
    if value != parsed_value:
        raise ValueError("gate artifact parsed value differs from snapshot bytes")
    if not isinstance(value, dict):
        raise ValueError("gate artifact root must be a JSON object")

    required_fields = {
        "schema",
        "gate_id",
        "status",
        "evidence_sha",
        "evidence",
        "gate_spec_sha",
        "config_sha",
        "source_tree_sha",
        "parent_evidence_shas",
        "parent_content_shas",
        "authorization_scope",
        "issuer",
        "issued_at",
        "authority",
        "artifact_content_sha",
    }
    if set(value) != required_fields:
        missing = sorted(required_fields - set(value))
        extra = sorted(set(value) - required_fields)
        raise ValueError(
            f"gate artifact fields mismatch: missing={missing}, extra={extra}"
        )
    if value["schema"] != TRACK_A_EXTERNAL_GATE_SCHEMA:
        raise ValueError("unknown Track A external gate artifact schema")
    if (
        value["gate_id"] != expected_gate_id
        or expected_gate_id not in TRACK_A_GATE_ORDER
    ):
        raise ValueError("external gate artifact gate identity mismatch")
    if value["status"] != GateStatus.PASS.value:
        raise ValueError("only an externally reviewed PASS artifact can close a gate")

    resolved_spec = gate_spec(expected_gate_id, gate_config)
    gate_spec_sha = _require_sha256(value["gate_spec_sha"], "gate_spec_sha")
    if gate_spec_sha != resolved_spec["gate_spec_sha"]:
        raise ValueError("external gate artifact specification is stale")
    evidence_sha = _require_sha256(value["evidence_sha"], "evidence_sha")
    if evidence_sha != sha256_of(value["evidence"]):
        raise ValueError(
            "external gate artifact evidence SHA does not match evidence content"
        )
    config_sha = _require_sha256(value["config_sha"], "config_sha")
    source_tree_sha = _require_git_identity(value["source_tree_sha"], "source_tree_sha")
    if config_sha != _require_sha256(expected_config_sha, "expected_config_sha"):
        raise ValueError("gate artifact config identity mismatch")
    if source_tree_sha != _require_git_identity(
        expected_source_tree_sha, "expected_source_tree_sha"
    ):
        raise ValueError("gate artifact source tree identity mismatch")

    parent_evidence = value["parent_evidence_shas"]
    parent_content = value["parent_content_shas"]
    if not isinstance(parent_evidence, dict) or not isinstance(parent_content, dict):
        raise ValueError("gate artifact parent identities must be JSON objects")
    normalized_parent_evidence = {
        str(gate_id): _require_sha256(identity, f"parent evidence {gate_id}")
        for gate_id, identity in parent_evidence.items()
    }
    normalized_parent_content = {
        str(gate_id): _require_sha256(identity, f"parent content {gate_id}")
        for gate_id, identity in parent_content.items()
    }
    expected_evidence = {
        str(gate_id): _require_sha256(identity, f"expected parent evidence {gate_id}")
        for gate_id, identity in expected_parent_evidence_shas.items()
    }
    expected_content = {
        str(gate_id): _require_sha256(identity, f"expected parent content {gate_id}")
        for gate_id, identity in expected_parent_content_shas.items()
    }
    if normalized_parent_evidence != expected_evidence:
        raise ValueError(
            "gate artifact parent evidence identities are stale or incomplete"
        )
    if normalized_parent_content != expected_content:
        raise ValueError(
            "gate artifact parent content identities are stale or incomplete"
        )

    canonical_scope = resolved_spec["authorization_scope"]
    if requested_scope != canonical_scope:
        raise ValueError("requested authorization scope is not canonical for the Gate")
    scopes = value["authorization_scope"]
    if scopes != [canonical_scope]:
        raise ValueError(
            "gate artifact must contain exactly the Gate-specific authorization scope"
        )
    issuer = value["issuer"]
    if (
        not isinstance(issuer, dict)
        or set(issuer) != {"identity", "role"}
        or any(not isinstance(issuer[key], str) or not issuer[key] for key in issuer)
    ):
        raise ValueError("gate artifact requires an identified independent issuer")
    try:
        issued_at = datetime.fromisoformat(
            str(value["issued_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("gate artifact issued_at must be ISO-8601") from exc
    if issued_at.tzinfo is None:
        raise ValueError("gate artifact issued_at must include a timezone")
    if value["authority"] != RESEARCH_AUTHORITY:
        raise ValueError(
            "external gate artifacts cannot grant selection, promotion, or hardware authority"
        )

    validate_gate_evidence(
        value["evidence"],
        expected_gate_id=expected_gate_id,
        expected_config_sha=expected_config_sha,
        expected_source_tree_sha=expected_source_tree_sha,
        gate_config=gate_config,
    )

    declared_content_sha = _require_sha256(
        value["artifact_content_sha"], "artifact_content_sha"
    )
    content = dict(value)
    content.pop("artifact_content_sha")
    calculated_content_sha = sha256_of(content)
    if declared_content_sha != calculated_content_sha:
        raise ValueError("gate artifact content SHA does not match canonical content")

    verified = dict(value)
    verified["artifact_file_sha"] = hashlib.sha256(raw_bytes).hexdigest()
    verified["verified"] = True
    return verified


def verify_external_gate_artifact(
    artifact_path: str | Path,
    *,
    expected_gate_id: str,
    expected_config_sha: str,
    expected_source_tree_sha: str,
    expected_parent_evidence_shas: Mapping[str, str],
    expected_parent_content_shas: Mapping[str, str],
    requested_scope: str,
    gate_config: GateConfig,
) -> Dict[str, Any]:
    """Backward-compatible file wrapper over the pure snapshot validator."""

    snapshot, value = load_json_snapshot(
        artifact_path, "external gate artifact", maximum_bytes=32 * 1024 * 1024
    )
    return validate_external_gate_artifact_snapshot(
        raw_bytes=snapshot.raw_bytes,
        parsed_value=value,
        expected_gate_id=expected_gate_id,
        expected_config_sha=expected_config_sha,
        expected_source_tree_sha=expected_source_tree_sha,
        expected_parent_evidence_shas=expected_parent_evidence_shas,
        expected_parent_content_shas=expected_parent_content_shas,
        requested_scope=requested_scope,
        gate_config=gate_config,
    )


class TrackAGateLedger:
    def __init__(self, authority: TrackAAuthority | None = None):
        self.authority = authority or TrackAAuthority()
        self.authority.validate()
        self._records: Dict[str, GateRecord] = {
            gate_id: GateRecord(gate_id, GateStatus.NOT_RUN)
            for gate_id in TRACK_A_GATE_ORDER
        }
        self._current_events: Dict[str, GateLedgerEvent] = {}
        self._events: list[GateLedgerEvent] = []

    def _parent_identities(
        self,
        gate_id: str,
    ) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]:
        parent_evidence = []
        parent_content = []
        for dependency in TRACK_A_GATE_DEPENDENCIES[gate_id]:
            event = self._current_events.get(dependency)
            if event is None:
                continue
            parent_evidence.append((dependency, self._records[dependency].evidence_sha))
            parent_content.append((dependency, event.content_sha))
        return tuple(parent_evidence), tuple(parent_content)

    def _append_event(
        self,
        record: GateRecord,
        *,
        action: str,
        previous_gate_content_sha: str = "",
        evidence_origin: str = "LOCAL_EVIDENCE",
        artifact_file_sha: str = "",
        authorization_scope: str = "",
        gate_spec_sha: str = "",
    ) -> GateLedgerEvent:
        allowed_origins = {"LOCAL_EVIDENCE", "EXTERNAL_ARTIFACT", "INVALIDATION"}
        if evidence_origin not in allowed_origins:
            raise ValueError("unknown gate evidence origin")
        if evidence_origin == "EXTERNAL_ARTIFACT":
            artifact_file_sha = _require_sha256(artifact_file_sha, "artifact_file_sha")
            gate_spec_sha = _require_sha256(gate_spec_sha, "gate_spec_sha")
            if not authorization_scope:
                raise ValueError(
                    "external gate evidence requires a Gate-specific scope"
                )
        elif artifact_file_sha or authorization_scope or gate_spec_sha:
            raise ValueError(
                "only external gate evidence may bind artifact, scope, or Gate spec"
            )
        parent_evidence, parent_content = self._parent_identities(record.gate_id)
        previous_event_sha = self._events[-1].content_sha if self._events else ""
        content = {
            "schema": "V10TrackAGateLedgerEventV4",
            "event_index": len(self._events),
            "action": action,
            "gate_id": record.gate_id,
            "status": record.status.value,
            "evidence_sha": record.evidence_sha,
            "notes": record.notes,
            "parent_evidence_shas": dict(parent_evidence),
            "parent_content_shas": dict(parent_content),
            "previous_event_sha": previous_event_sha,
            "previous_gate_content_sha": previous_gate_content_sha,
            "evidence_origin": evidence_origin,
            "artifact_file_sha": artifact_file_sha,
            "authorization_scope": authorization_scope,
            "gate_spec_sha": gate_spec_sha,
        }
        event = GateLedgerEvent(
            event_index=len(self._events),
            action=action,
            gate_id=record.gate_id,
            status=record.status,
            evidence_sha=record.evidence_sha,
            notes=record.notes,
            parent_evidence_shas=parent_evidence,
            parent_content_shas=parent_content,
            previous_event_sha=previous_event_sha,
            previous_gate_content_sha=previous_gate_content_sha,
            evidence_origin=evidence_origin,
            artifact_file_sha=artifact_file_sha,
            authorization_scope=authorization_scope,
            gate_spec_sha=gate_spec_sha,
            content_sha=sha256_of(content),
        )
        self._events.append(event)
        self._current_events[record.gate_id] = event
        self._records[record.gate_id] = record
        return event

    def _lineage_is_current(
        self,
        gate_id: str,
        *,
        require_external_artifact: bool = False,
    ) -> bool:
        record = self._records[gate_id]
        event = self._current_events.get(gate_id)
        if record.status is not GateStatus.PASS or event is None:
            return False
        expected_evidence, expected_content = self._parent_identities(gate_id)
        return (
            event.parent_evidence_shas == expected_evidence
            and event.parent_content_shas == expected_content
            and (
                not require_external_artifact
                or (
                    event.evidence_origin == "EXTERNAL_ARTIFACT"
                    and bool(event.artifact_file_sha)
                    and bool(event.authorization_scope)
                    and bool(event.gate_spec_sha)
                )
            )
            and all(
                self._records[dependency].status is GateStatus.PASS
                for dependency in TRACK_A_GATE_DEPENDENCIES[gate_id]
            )
        )

    def _record_decision(
        self,
        gate_id: str,
        status: GateStatus | str,
        *,
        evidence_sha: str = "",
        notes: str = "",
        evidence_origin: str,
        artifact_file_sha: str = "",
        authorization_scope: str = "",
        gate_spec_sha: str = "",
    ) -> GateRecord:
        if gate_id not in self._records:
            raise ValueError(f"unknown Track A gate {gate_id}")
        if self._records[gate_id].status is not GateStatus.NOT_RUN:
            raise ValueError(
                f"gate {gate_id} already has an immutable decision; create a new ledger "
                "or append an explicit invalidation"
            )
        selected = GateStatus(status)
        normalized_evidence = evidence_sha
        if selected in {GateStatus.PASS, GateStatus.FAIL}:
            normalized_evidence = _require_sha256(normalized_evidence, "evidence_sha")
        elif normalized_evidence:
            normalized_evidence = _require_sha256(normalized_evidence, "evidence_sha")
        if selected is GateStatus.PASS:
            blocked = [
                dependency
                for dependency in TRACK_A_GATE_DEPENDENCIES[gate_id]
                if not self._lineage_is_current(dependency)
            ]
            if blocked:
                raise ValueError(
                    f"gate {gate_id} cannot pass before current dependencies {blocked}"
                )
        record = GateRecord(gate_id, selected, normalized_evidence, notes)
        record.validate()
        self._append_event(
            record,
            action="RECORD",
            evidence_origin=evidence_origin,
            artifact_file_sha=artifact_file_sha,
            authorization_scope=authorization_scope,
            gate_spec_sha=gate_spec_sha,
        )
        return record

    def record(
        self,
        gate_id: str,
        status: GateStatus | str,
        *,
        evidence_sha: str = "",
        notes: str = "",
    ) -> GateRecord:
        """Record local evaluation evidence without granting server readiness."""

        return self._record_decision(
            gate_id,
            status,
            evidence_sha=evidence_sha,
            notes=notes,
            evidence_origin="LOCAL_EVIDENCE",
        )

    def record_external_artifact(
        self,
        artifact_path: str | Path,
        *,
        gate_id: str,
        expected_config_sha: str,
        expected_source_tree_sha: str,
        requested_scope: str,
        gate_config: GateConfig,
        notes: str = "",
    ) -> GateRecord:
        if gate_id not in self._records:
            raise ValueError(f"unknown Track A gate {gate_id}")
        parent_evidence, parent_content = self._parent_identities(gate_id)
        artifact = verify_external_gate_artifact(
            artifact_path,
            expected_gate_id=gate_id,
            expected_config_sha=expected_config_sha,
            expected_source_tree_sha=expected_source_tree_sha,
            expected_parent_evidence_shas=dict(parent_evidence),
            expected_parent_content_shas=dict(parent_content),
            requested_scope=requested_scope,
            gate_config=gate_config,
        )
        artifact_note = notes or (
            f"External gate artifact {artifact['artifact_file_sha']} verified for {requested_scope}"
        )
        return self._record_decision(
            gate_id,
            GateStatus.PASS,
            evidence_sha=str(artifact["artifact_content_sha"]),
            notes=artifact_note,
            evidence_origin="EXTERNAL_ARTIFACT",
            artifact_file_sha=str(artifact["artifact_file_sha"]),
            authorization_scope=requested_scope,
            gate_spec_sha=str(artifact["gate_spec_sha"]),
        )

    def invalidate(
        self,
        gate_id: str,
        *,
        evidence_sha: str,
        notes: str,
    ) -> Tuple[GateRecord, ...]:
        """Append an explicit revocation and block every decided dependent gate."""

        if gate_id not in self._records:
            raise ValueError(f"unknown Track A gate {gate_id}")
        if self._records[gate_id].status is GateStatus.NOT_RUN:
            raise ValueError(f"gate {gate_id} has no decision to invalidate")
        if not notes.strip():
            raise ValueError("gate invalidation requires a non-empty reason")
        normalized_evidence = _require_sha256(evidence_sha, "invalidation evidence_sha")
        invalidated = []
        start = TRACK_A_GATE_ORDER.index(gate_id)
        for index, affected_gate in enumerate(TRACK_A_GATE_ORDER[start:]):
            prior = self._records[affected_gate]
            if prior.status is GateStatus.NOT_RUN:
                continue
            prior_event = self._current_events[affected_gate]
            selected = GateStatus.FAIL if index == 0 else GateStatus.BLOCKED
            reason = (
                notes
                if index == 0
                else f"Invalidated by upstream gate {gate_id}: {notes}"
            )
            record = GateRecord(affected_gate, selected, normalized_evidence, reason)
            record.validate()
            self._append_event(
                record,
                action="INVALIDATE",
                previous_gate_content_sha=prior_event.content_sha,
                evidence_origin="INVALIDATION",
            )
            invalidated.append(record)
        return tuple(invalidated)

    def status(self, gate_id: str) -> GateStatus:
        return self._records[gate_id].status

    def engineering_server_ready(self) -> bool:
        return all(
            self._lineage_is_current(gate, require_external_artifact=True)
            for gate in TRACK_A_GATE_ORDER[:5]
        )

    def scientific_server_ready(self) -> bool:
        return all(
            self._lineage_is_current(gate, require_external_artifact=True)
            for gate in TRACK_A_GATE_ORDER
        )

    def report(self) -> Dict[str, object]:
        self.authority.validate()
        records: Dict[str, object] = {}
        for gate_id, record in self._records.items():
            value: Dict[str, object] = asdict(record)
            value["status"] = record.status.value
            event = self._current_events.get(gate_id)
            if event is not None:
                value.update(
                    {
                        "parent_evidence_shas": dict(event.parent_evidence_shas),
                        "parent_content_shas": dict(event.parent_content_shas),
                        "evidence_origin": event.evidence_origin,
                        "artifact_file_sha": event.artifact_file_sha,
                        "authorization_scope": event.authorization_scope,
                        "gate_spec_sha": event.gate_spec_sha,
                        "content_sha": event.content_sha,
                    }
                )
            else:
                value.update(
                    {
                        "parent_evidence_shas": {},
                        "parent_content_shas": {},
                        "evidence_origin": "",
                        "artifact_file_sha": "",
                        "authorization_scope": "",
                        "gate_spec_sha": "",
                        "content_sha": "",
                    }
                )
            records[gate_id] = value
        payload: Dict[str, object] = {
            "schema": TRACK_A_GATE_LEDGER_SCHEMA,
            "gate_order": TRACK_A_GATE_ORDER,
            "records": records,
            "events": tuple(event.to_dict() for event in self._events),
            "engineering_server_ready": self.engineering_server_ready(),
            "scientific_server_ready": self.scientific_server_ready(),
            "authority": asdict(self.authority),
        }
        payload["ledger_sha"] = sha256_of(payload)
        return payload


def evidence_sha(*values: object) -> str:
    return sha256_of(values)
