"""Fail-closed binding of a Track A smoke report to trusted source evidence."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from .gates import TRACK_A_GATE_LEDGER_SCHEMA, TRACK_A_GATE_ORDER


ENGINEERING_SMOKE_SCHEMA = "V10TrackAEngineeringSmokeV1"
SOURCE_ACCEPTANCE_SCHEMA = "V10TrackASourceAcceptanceV2"
TRUSTED_SCHEDULER_ENVELOPE_SCHEMA = "TailnetMeshTrustedJobEnvelopeV1"
GIT_TIMEOUT_SECONDS = 15
SOURCE_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
HEX_PATTERN = re.compile(r"^[0-9a-f]+$")
RESEARCH_AUTHORITY = {
    "research_only": True,
    "selection_evidence": False,
    "promotion_ready": False,
    "hardware_action_authorized": False,
    "ten_k_training_authorized": False,
}


class SourceAcceptanceError(RuntimeError):
    """A structured fail-closed source acceptance error."""


def _require_hex_identity(
    value: object,
    field_name: str,
    *,
    allowed_lengths: Sequence[int],
) -> str:
    if not isinstance(value, str):
        raise SourceAcceptanceError(f"{field_name} must be a string identity")
    raw = value
    normalized = raw.strip().lower()
    if raw != normalized or len(normalized) not in allowed_lengths or not HEX_PATTERN.fullmatch(
            normalized):
        lengths = "/".join(str(length) for length in allowed_lengths)
        raise SourceAcceptanceError(
            f"{field_name} must be canonical lowercase hexadecimal with length {lengths}"
        )
    return normalized


def _git_output(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SourceAcceptanceError(
            f"Git provenance check timed out after {GIT_TIMEOUT_SECONDS} seconds"
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        raise SourceAcceptanceError(f"Git provenance check could not start: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise SourceAcceptanceError(f"Git provenance check failed: {detail[:500]}")
    return result.stdout.strip()


def _validate_authority(value: object, field_name: str) -> None:
    if not isinstance(value, dict) or value != RESEARCH_AUTHORITY:
        raise SourceAcceptanceError(
            f"{field_name} must preserve the exact research-only, all-false authority contract"
        )


def _current_platform_identity() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
    }


def validate_empty_smoke_ledger(value: object) -> None:
    """Require the exact no-decision ledger emitted by engineering smoke."""

    required_fields = {
        "schema",
        "gate_order",
        "records",
        "events",
        "engineering_server_ready",
        "scientific_server_ready",
        "authority",
        "ledger_sha",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise SourceAcceptanceError(
            "engineering smoke requires a complete current Track A gate ledger"
        )
    if (value["schema"] != TRACK_A_GATE_LEDGER_SCHEMA
            or tuple(value["gate_order"]) != TRACK_A_GATE_ORDER
            or value["events"] not in ([], ())
            or value["engineering_server_ready"] is not False
            or value["scientific_server_ready"] is not False):
        raise SourceAcceptanceError(
            "engineering smoke ledger must contain no decisions or readiness"
        )
    _validate_authority(value["authority"], "gate ledger authority")
    records = value["records"]
    if not isinstance(records, dict) or set(records) != set(TRACK_A_GATE_ORDER):
        raise SourceAcceptanceError("engineering smoke ledger Gate records are incomplete")
    empty_record_fields = {
        "evidence_sha": "",
        "notes": "",
        "parent_evidence_shas": {},
        "parent_content_shas": {},
        "evidence_origin": "",
        "artifact_file_sha": "",
        "authorization_scope": "",
        "gate_spec_sha": "",
        "content_sha": "",
    }
    for gate_id in TRACK_A_GATE_ORDER:
        record = records[gate_id]
        if (not isinstance(record, dict)
                or record.get("gate_id") != gate_id
                or record.get("status") != "NOT_RUN"
                or any(record.get(name) != expected
                       for name, expected in empty_record_fields.items())):
            raise SourceAcceptanceError(
                f"engineering smoke ledger Gate {gate_id} is not an empty NOT_RUN record"
            )
    declared_ledger_sha = _require_hex_identity(
        value["ledger_sha"], "gate ledger_sha", allowed_lengths=(64,))
    unhashed = dict(value)
    unhashed.pop("ledger_sha")
    if declared_ledger_sha != sha256_of(unhashed):
        raise SourceAcceptanceError("engineering smoke gate ledger SHA is stale")


def _validate_unbound_report(
    report: Mapping[str, Any],
    *,
    expected_config_sha: str,
) -> Dict[str, Any]:
    if not isinstance(report, Mapping):
        raise SourceAcceptanceError("source acceptance report must be a mapping")
    value = dict(report)
    if value.get("schema") != ENGINEERING_SMOKE_SCHEMA:
        raise SourceAcceptanceError("source acceptance requires a V10 Track A engineering smoke report")
    if value.get("status") != "PASS_ENGINEERING_SMOKE":
        raise SourceAcceptanceError("only a passing engineering smoke can be source-bound")
    if value.get("source_bound") is not False:
        raise SourceAcceptanceError("input report must be an unbound engineering smoke")

    report_config_sha = _require_hex_identity(
        value.get("config_sha", ""), "report config_sha", allowed_lengths=(64,))
    expected_sha = _require_hex_identity(
        expected_config_sha, "expected_config_sha", allowed_lengths=(64,))
    if report_config_sha != expected_sha:
        raise SourceAcceptanceError("engineering smoke config identity mismatch")

    checks = value.get("checks")
    if (not isinstance(checks, dict) or not checks
            or any(type(result) is not bool for result in checks.values())
            or not all(checks.values())):
        raise SourceAcceptanceError("engineering smoke checks must be explicit booleans and all pass")
    _validate_authority(value.get("authority"), "report authority")
    validate_empty_smoke_ledger(value.get("gate_ledger"))
    gate_assessment = value.get("gate_assessment")
    if (not isinstance(gate_assessment, dict)
            or gate_assessment.get("status") != "PARTIAL_LOCAL_EVIDENCE"
            or gate_assessment.get("gate_decisions_written") is not False
            or gate_assessment.get("external_gate_artifact_required") is not True):
        raise SourceAcceptanceError(
            "engineering smoke gate assessment must remain local and write no Gate decisions"
        )
    _require_hex_identity(
        gate_assessment.get("engineering_evidence_sha", ""),
        "engineering evidence SHA",
        allowed_lengths=(64,),
    )

    runtime = value.get("runtime")
    required_runtime = {
        "device": str,
        "torch_version": str,
        "cuda_available": bool,
        "torch_cuda_version": str,
        "cuda_device_name": str,
    }
    if not isinstance(runtime, dict):
        raise SourceAcceptanceError("engineering smoke runtime block is missing")
    for field_name, expected_type in required_runtime.items():
        if field_name not in runtime or type(runtime[field_name]) is not expected_type:
            raise SourceAcceptanceError(f"runtime field {field_name} is missing or has the wrong type")
    if not runtime["device"] or not runtime["torch_version"]:
        raise SourceAcceptanceError("runtime device and torch_version must be non-empty")
    if runtime["device"].startswith("cuda") and (
            not runtime["cuda_available"]
            or not runtime["torch_cuda_version"]
            or not runtime["cuda_device_name"]):
        raise SourceAcceptanceError("CUDA smoke runtime identity is incomplete")

    declared_report_sha = _require_hex_identity(
        value.get("report_sha", ""), "report_sha", allowed_lengths=(64,))
    unhashed_report = dict(value)
    unhashed_report.pop("report_sha")
    if declared_report_sha != sha256_of(unhashed_report):
        raise SourceAcceptanceError("engineering smoke report_sha does not match canonical content")

    current_platform = _current_platform_identity()
    declared_platform = value.get("platform")
    if declared_platform is not None and declared_platform != current_platform:
        raise SourceAcceptanceError("declared platform identity does not match the accepting runtime")
    bound_runtime = dict(runtime)
    declared_python = bound_runtime.get("python_version")
    if declared_python is not None and declared_python != current_platform["python_version"]:
        raise SourceAcceptanceError("runtime python_version does not match the accepting runtime")
    bound_runtime["python_version"] = current_platform["python_version"]
    value["runtime"] = bound_runtime
    value["platform"] = current_platform
    return value


def _load_trusted_scheduler_envelope(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise SourceAcceptanceError("trusted scheduler envelope must be a readable JSON file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAcceptanceError(f"trusted scheduler envelope is invalid JSON: {exc}") from exc
    required_fields = {
        "schema",
        "source_repository",
        "source_revision",
        "job_id",
        "scheduler_identity",
        "node",
        "rank",
        "world_size",
        "origin_urls_sha256",
        "envelope_content_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise SourceAcceptanceError("trusted scheduler envelope fields are incomplete or unexpected")
    if value["schema"] != TRUSTED_SCHEDULER_ENVELOPE_SCHEMA:
        raise SourceAcceptanceError("unknown trusted scheduler envelope schema")
    if (not isinstance(value["source_repository"], str)
            or not SOURCE_REPOSITORY_PATTERN.fullmatch(value["source_repository"])):
        raise SourceAcceptanceError("trusted scheduler repository identity is invalid")
    _require_hex_identity(
        value["source_revision"], "scheduler source_revision", allowed_lengths=(40, 64))
    if (not isinstance(value["job_id"], str) or not value["job_id"]
            or not isinstance(value["scheduler_identity"], str)
            or not value["scheduler_identity"]
            or not isinstance(value["node"], str)
            or not value["node"]):
        raise SourceAcceptanceError("trusted scheduler identity fields must be non-empty strings")
    if (type(value["rank"]) is not int or type(value["world_size"]) is not int
            or value["world_size"] < 1 or not 0 <= value["rank"] < value["world_size"]):
        raise SourceAcceptanceError("trusted scheduler rank/world_size are invalid")
    if value["origin_urls_sha256"]:
        _require_hex_identity(
            value["origin_urls_sha256"], "origin_urls_sha256", allowed_lengths=(64,))
    declared_sha = _require_hex_identity(
        value["envelope_content_sha256"],
        "envelope_content_sha256",
        allowed_lengths=(64,),
    )
    content = dict(value)
    content.pop("envelope_content_sha256")
    if declared_sha != sha256_of(content):
        raise SourceAcceptanceError("trusted scheduler envelope content SHA mismatch")
    return value


def bind_source_provenance(
    report: Mapping[str, Any],
    *,
    expected_config_sha: str,
    repository_root: str | Path | None = None,
    trusted_repository: str | None = None,
    trusted_scheduler_envelope: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate and bind a smoke report to a clean source and optional scheduler trust."""

    validated_report = _validate_unbound_report(
        report, expected_config_sha=expected_config_sha)
    source_repository = os.environ.get("TAILNET_MESH_SOURCE_REPOSITORY", "").strip()
    source_revision = os.environ.get("TAILNET_MESH_SOURCE_REVISION", "").strip()
    job_id = os.environ.get("TAILNET_MESH_JOB_ID", "").strip()
    if not SOURCE_REPOSITORY_PATTERN.fullmatch(source_repository):
        raise SourceAcceptanceError("TAILNET_MESH_SOURCE_REPOSITORY is missing or invalid")
    source_revision = _require_hex_identity(
        source_revision, "TAILNET_MESH_SOURCE_REVISION", allowed_lengths=(40, 64))
    if not job_id:
        raise SourceAcceptanceError("TAILNET_MESH_JOB_ID is required for source-bound acceptance")
    if trusted_repository is not None:
        if not SOURCE_REPOSITORY_PATTERN.fullmatch(trusted_repository):
            raise SourceAcceptanceError("trusted_repository is invalid")
        if source_repository != trusted_repository:
            raise SourceAcceptanceError("scheduler repository does not match the trusted repository")

    root = Path(repository_root or Path.cwd()).resolve()
    git_root = Path(_git_output(root, ("rev-parse", "--show-toplevel"))).resolve()
    if git_root != root:
        raise SourceAcceptanceError("repository_root does not match Git's top-level checkout")
    actual_revision = _require_hex_identity(
        _git_output(root, ("rev-parse", "HEAD")),
        "checked-out revision",
        allowed_lengths=(40, 64),
    )
    source_tree_sha = _require_hex_identity(
        _git_output(root, ("rev-parse", "HEAD^{tree}")),
        "checked-out tree",
        allowed_lengths=(40, 64),
    )
    worktree_status = _git_output(
        root, ("status", "--porcelain=v1", "--untracked-files=normal"))
    if actual_revision != source_revision:
        raise SourceAcceptanceError("Tailnet source revision does not match the checked-out Git HEAD")
    if worktree_status:
        raise SourceAcceptanceError("source-bound acceptance requires a clean Git worktree")

    remote_names = tuple(
        line.strip() for line in _git_output(root, ("remote",)).splitlines() if line.strip()
    )
    origin_urls = ()
    if "origin" in remote_names:
        origin_urls = tuple(
            sorted(line.strip() for line in _git_output(
                root, ("remote", "get-url", "--all", "origin")).splitlines() if line.strip())
        )
    origin_urls_sha256 = sha256_of(origin_urls)

    envelope_path = trusted_scheduler_envelope or os.environ.get(
        "TAILNET_MESH_TRUSTED_ENVELOPE", "").strip()
    envelope = None
    if envelope_path:
        envelope = _load_trusted_scheduler_envelope(envelope_path)
        if envelope["source_repository"] != source_repository:
            raise SourceAcceptanceError("trusted scheduler envelope repository mismatch")
        if envelope["source_revision"] != source_revision:
            raise SourceAcceptanceError("trusted scheduler envelope revision mismatch")
        if envelope["job_id"] != job_id:
            raise SourceAcceptanceError("trusted scheduler envelope job identity mismatch")
        if envelope["origin_urls_sha256"] and envelope["origin_urls_sha256"] != origin_urls_sha256:
            raise SourceAcceptanceError("trusted scheduler envelope remote identity mismatch")
        environment_identity = {
            "node": os.environ.get("TAILNET_MESH_NODE", "").strip(),
            "rank": os.environ.get("TAILNET_MESH_RANK", "").strip(),
            "world_size": os.environ.get("TAILNET_MESH_WORLD_SIZE", "").strip(),
        }
        if environment_identity["node"] and environment_identity["node"] != envelope["node"]:
            raise SourceAcceptanceError("trusted scheduler envelope node mismatch")
        try:
            environment_rank = (
                int(environment_identity["rank"])
                if environment_identity["rank"] else None
            )
            environment_world_size = (
                int(environment_identity["world_size"])
                if environment_identity["world_size"] else None
            )
        except ValueError as exc:
            raise SourceAcceptanceError(
                "trusted scheduler rank/world_size environment values must be integers"
            ) from exc
        if environment_rank is not None and environment_rank != envelope["rank"]:
            raise SourceAcceptanceError("trusted scheduler envelope rank mismatch")
        if (environment_world_size is not None
                and environment_world_size != envelope["world_size"]):
            raise SourceAcceptanceError("trusted scheduler envelope world_size mismatch")

    unbound_report_sha = validated_report.pop("report_sha")
    bound_report = dict(validated_report)
    bound_report["source_bound"] = True
    bound_report["source"] = {
        "repository": source_repository,
        "revision": actual_revision,
        "tree_sha": source_tree_sha,
        "job_id": job_id,
        "worktree_clean": True,
        "origin_remote_present": bool(origin_urls),
        "origin_urls_sha256": origin_urls_sha256,
    }
    bound_report["source_acceptance"] = {
        "schema": SOURCE_ACCEPTANCE_SCHEMA,
        "unbound_report_sha": unbound_report_sha,
        "config_identity_verified": True,
        "authority_verified": True,
        "runtime_platform_verified": True,
        "trusted_repository_verified": trusted_repository is not None or envelope is not None,
        "scheduler_envelope_verified": envelope is not None,
        "scheduler_envelope_content_sha256": (
            envelope["envelope_content_sha256"] if envelope is not None else ""
        ),
        "git_timeout_seconds": GIT_TIMEOUT_SECONDS,
    }
    bound_report["report_sha"] = sha256_of(bound_report)
    return bound_report
