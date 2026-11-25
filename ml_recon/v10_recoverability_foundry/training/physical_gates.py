"""R16 physical evidence validation on the existing append-only Gate engine.

Hashes validate content and lineage, not an issuer's real-world identity. A
separate, source-bound Chat/Pro review is required by the execution admission.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.control_coordinate_layout import (
    INTERLEAVED_CONTROL_LAYOUT_SHA,
)
from ml_recon.physics_hybrid_v10.evidence_file_io import (
    Evidence_file_identity,
    sha256_evidence_file,
    validate_evidence_directory,
)
from ml_recon.physics_hybrid_v10.evidence_snapshot import (
    load_json_snapshot,
    reject_json_constant,
    strict_object_pairs,
)
from ..config import GateConfig
from ..contracts import GateRecord, GateStatus
from .gates import GateLedgerEvent, TrackAGateLedger, TRACK_A_GATE_DEPENDENCIES
from .physical_gate_templates import (
    PHYSICAL_EVIDENCE_SCHEMA,
    PHYSICAL_IDENTITY_FIELDS,
    P0_SEMANTIC_BINDING_SCHEMA,
    P0_SEMANTIC_SOURCE_PATHS,
    P0_CLASSIFIER_OBJECTIVE_BINDING,
    physical_gate_spec,
)
from .physical_plan import AUTHORITY, PHYSICAL_PROGRAM_NAMESPACE
from ..runtime.memory_event_reducer import (
    LEGACY_MEMORY_EVENT_SCHEMA_V3,
    LEGACY_MEMORY_EVENT_SCHEMA_V4,
    MEMORY_EVENT_SCHEMA,
)


PHYSICAL_LEDGER_SCHEMA = "V10TrackAGateLedgerV5"
PHYSICAL_EVENT_SCHEMA = "V10TrackAGateLedgerEventV5"
PHYSICAL_ARTIFACT_SCHEMA = "V10PhysicalExternalGateArtifactV4"
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024


def require_digest(value: object, name: str, *, git: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in ((40, 64) if git else (64,))
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} requires a canonical lowercase digest")
    return value


def validate_physical_identity(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(PHYSICAL_IDENTITY_FIELDS):
        raise ValueError("physical identity fields mismatch")
    return {
        name: require_digest(value[name], name) for name in PHYSICAL_IDENTITY_FIELDS
    }


def _digest_map(
    value: object, name: str, *, allow_empty: bool = False
) -> dict[str, str]:
    if not isinstance(value, dict) or (not allow_empty and not value):
        raise ValueError(f"{name} requires an identified digest mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be nonempty strings")
    return {
        key: require_digest(digest, f"{name}.{key}") for key, digest in value.items()
    }


def validate_physical_gate_evidence(
    evidence: object,
    *,
    expected_gate_id: str,
    expected_source_commit: str,
    expected_source_tree_sha: str,
    expected_config_sha: str,
    expected_physical_identity: Mapping[str, str],
    gate_config: GateConfig | None = None,
    expected_threshold_registration_sha: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "namespace",
        "gate_id",
        "gate_spec_sha",
        "evaluator_schema",
        "evaluator_sha",
        "source_commit",
        "source_tree_sha",
        "config_sha",
        "physical_identity",
        "checkpoint_shas",
        "dataset_range_shas",
        "generator_sha",
        "contract_sha",
        "dataset_schema_sha",
        "seed_panel",
        "sample_rung",
        "checks",
        "metrics",
        "evidence_file_roles",
        "authority",
        "threshold_registration_sha",
        "minimum_meaningful_effect",
        "registered_at",
        "confirmation_started_at",
        "development_seed_panel",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError("physical gate evidence has an exact field contract")
    spec = physical_gate_spec(expected_gate_id, gate_config)
    if (
        evidence["schema"] != PHYSICAL_EVIDENCE_SCHEMA
        or evidence["namespace"] != PHYSICAL_PROGRAM_NAMESPACE
        or evidence["gate_id"] != expected_gate_id
        or evidence["gate_spec_sha"] != spec["gate_spec_sha"]
        or evidence["evaluator_schema"] != spec["evaluator_schema"]
    ):
        raise ValueError("physical namespace, Gate or evaluator specification is stale")
    for name, expected, is_git in (
        ("source_commit", expected_source_commit, True),
        ("source_tree_sha", expected_source_tree_sha, True),
        ("config_sha", expected_config_sha, False),
    ):
        if require_digest(evidence[name], name, git=is_git) != require_digest(
            expected, f"expected_{name}", git=is_git
        ):
            raise ValueError(f"physical evidence {name} mismatch")
    if validate_physical_identity(
        evidence["physical_identity"]
    ) != validate_physical_identity(expected_physical_identity):
        raise ValueError("physical evidence ABI/budget/operation/protocol mismatch")
    require_digest(evidence["evaluator_sha"], "evaluator_sha")
    _digest_map(
        evidence["checkpoint_shas"],
        "checkpoint_shas",
        allow_empty=not spec["require_checkpoints"],
    )
    _digest_map(
        evidence["dataset_range_shas"],
        "dataset_range_shas",
        allow_empty=not spec["require_dataset_ranges"],
    )
    if (
        not isinstance(evidence["evidence_file_roles"], dict)
        or not evidence["evidence_file_roles"]
    ):
        raise ValueError("physical evidence requires role-bound evidence files")
    dataset_fields = ("generator_sha", "contract_sha", "dataset_schema_sha")
    if spec["require_dataset_identity"] or any(
        evidence[name] != "" for name in dataset_fields
    ):
        for name in dataset_fields:
            require_digest(evidence[name], name)
    seeds = evidence["seed_panel"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < spec["minimum_seed_count"]
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or seeds != sorted(set(seeds))
    ):
        raise ValueError("physical evidence seed panel is incomplete or noncanonical")
    rung = evidence["sample_rung"]
    if (
        type(rung) is not int
        or not spec["minimum_sample_rung"] <= rung <= spec["maximum_sample_rung"]
    ):
        raise ValueError("physical evidence sample rung is outside its registered Gate")
    checks = evidence["checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != set(spec["required_checks"])
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("physical evidence required checks are missing or not true")
    metrics = evidence["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {
        rule["metric"] for rule in spec["metric_rules"]
    }:
        raise ValueError("physical Gate metrics do not match the registered estimands")
    for rule in spec["metric_rules"]:
        value = metrics[rule["metric"]]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError("physical Gate metrics must be finite real values")
        threshold = rule["threshold"]
        passed = {
            "minimum": value >= threshold,
            "strict_minimum": value > threshold,
            "maximum": value <= threshold,
        }[rule["comparator"]]
        if not passed:
            raise ValueError(f"physical Gate metric failed: {rule['metric']}")
    if evidence["authority"] != AUTHORITY or any(
        type(value) is not bool for value in evidence["authority"].values()
    ):
        raise ValueError("physical Gate evidence cannot grant authority")
    if expected_gate_id == "S5":
        if expected_threshold_registration_sha is None:
            raise ValueError("S5 requires an expected threshold registration SHA")
        if require_digest(
            evidence["threshold_registration_sha"], "threshold_registration_sha"
        ) != require_digest(
            expected_threshold_registration_sha, "expected_threshold_registration_sha"
        ):
            raise ValueError("S5 threshold registration SHA mismatch")
        effect = evidence["minimum_meaningful_effect"]
        if (
            isinstance(effect, bool)
            or not isinstance(effect, Real)
            or not math.isfinite(effect)
            or effect <= 0.0
        ):
            raise ValueError("S5 minimum meaningful effect must be positive")
        try:
            registered_at = datetime.fromisoformat(
                evidence["registered_at"].replace("Z", "+00:00")
            )
            confirmation_at = datetime.fromisoformat(
                evidence["confirmation_started_at"].replace("Z", "+00:00")
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("S5 timestamps must be timezone-aware ISO-8601") from exc
        if (
            registered_at.tzinfo is None
            or confirmation_at.tzinfo is None
            or confirmation_at < registered_at
        ):
            raise ValueError("S5 confirmation cannot precede threshold registration")
        development_seeds = evidence["development_seed_panel"]
        if (
            not isinstance(development_seeds, list)
            or not development_seeds
            or any(type(seed) is not int or seed < 0 for seed in development_seeds)
            or development_seeds != sorted(set(development_seeds))
            or set(development_seeds).intersection(seeds)
        ):
            raise ValueError("S5 development and confirmation seeds must be disjoint")
        if any(
            metrics[name] < effect
            for name in ("model_value_over_baseline_lcb95", "confirmation_gain_lcb95")
        ):
            raise ValueError("S5 LCB is below the preregistered meaningful effect")
    elif (
        evidence["threshold_registration_sha"] != ""
        or evidence["minimum_meaningful_effect"] is not None
        or evidence["registered_at"] != ""
        or evidence["confirmation_started_at"] != ""
        or evidence["development_seed_panel"] != []
    ):
        raise ValueError("only S5 may carry threshold registration fields")
    return dict(evidence)


def verify_evidence_files(
    evidence_root: str | Path, evidence_file_roles: Mapping[str, object]
) -> dict[str, str]:
    """Verify immutable role-to-file bindings, without traversal or symlink escapes."""
    verified, _ = _verify_evidence_file_records(evidence_root, evidence_file_roles)
    return verified


def _verify_evidence_file_records(
    evidence_root: str | Path, evidence_file_roles: Mapping[str, object]
) -> tuple[dict[str, str], dict[str, Evidence_file_identity]]:
    """Return content and descriptor identities from the same bounded file pass."""
    root = validate_evidence_directory(evidence_root, "evidence_root")
    verified: dict[str, str] = {}
    identities: dict[str, Evidence_file_identity] = {}
    if not isinstance(evidence_file_roles, Mapping) or not evidence_file_roles:
        raise ValueError("evidence_file_roles requires nonempty role bindings")
    for role, binding in evidence_file_roles.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
        ):
            raise ValueError("evidence roles require a path and SHA-256")
        relative = binding["path"]
        digest = require_digest(binding["sha256"], f"evidence_file_roles.{role}.sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError("evidence role paths must be nonempty strings")
        suffix = Path(relative).suffix.lower()
        required_suffix = {
            "evaluator": ".py",
            "report": ".json",
            "generator": ".json",
            "contract": ".json",
            "dataset_schema": ".json",
            "threshold_registration": ".json",
        }.get(role)
        if required_suffix is not None and suffix != required_suffix:
            raise ValueError(f"evidence role {role} has an invalid file type")
        if role.startswith("checkpoint:") and suffix not in {".ckpt", ".pt", ".bin"}:
            raise ValueError("checkpoint roles require a checkpoint file")
        if role.startswith("dataset_range:") and suffix != ".json":
            raise ValueError("dataset-range roles require a JSON range file")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or ":" in relative
        ):
            raise ValueError("evidence paths must be contained relative paths")
        candidate = root / relative_path
        maximum_size = (
            MAX_CHECKPOINT_BYTES
            if role.startswith("checkpoint")
            else MAX_EVIDENCE_BYTES
        )
        _, observed, file_identity = sha256_evidence_file(
            candidate, f"evidence role {role}", maximum_bytes=maximum_size
        )
        if observed != digest:
            raise ValueError(f"evidence file SHA mismatch: {relative}")
        verified[role] = observed
        identities[role] = file_identity
    return verified, identities


def _require_exact_evidence_roles(
    evidence: Mapping[str, Any],
    verified: Mapping[str, str],
    identities: Mapping[str, Evidence_file_identity],
    evidence_root: Path,
) -> None:
    """Bind declared identities to their actual evaluator, report, and input bytes."""
    required_roles = {"evaluator", "report"}
    required_roles.update(f"checkpoint:{name}" for name in evidence["checkpoint_shas"])
    required_roles.update(
        f"dataset_range:{name}" for name in evidence["dataset_range_shas"]
    )
    for field, role in (
        ("generator_sha", "generator"),
        ("contract_sha", "contract"),
        ("dataset_schema_sha", "dataset_schema"),
    ):
        if evidence[field]:
            required_roles.add(role)
    if evidence["gate_id"] == "S5":
        required_roles.add("threshold_registration")
    provided_roles = set(evidence["evidence_file_roles"])
    unexpected = {
        role for role in provided_roles if not role.startswith("external_receipt:")
    }
    if required_roles != unexpected:
        raise ValueError(
            "evidence roles do not exactly cover declared evaluator and inputs"
        )
    if verified["evaluator"] != evidence["evaluator_sha"]:
        raise ValueError("evaluator SHA is not bound to the evaluator file")
    for name, digest in evidence["checkpoint_shas"].items():
        if verified[f"checkpoint:{name}"] != digest:
            raise ValueError("checkpoint SHA is not bound to its checkpoint file")
    for name, digest in evidence["dataset_range_shas"].items():
        if verified[f"dataset_range:{name}"] != digest:
            raise ValueError("dataset-range SHA is not bound to its range file")
    for field, role in (
        ("generator_sha", "generator"),
        ("contract_sha", "contract"),
        ("dataset_schema_sha", "dataset_schema"),
    ):
        if evidence[field] and verified[role] != evidence[field]:
            raise ValueError(f"{field} is not bound to its role file")
    try:
        _, report = load_json_snapshot(
            evidence_root / evidence["evidence_file_roles"]["report"]["path"],
            "evaluation report",
            expected_sha=verified["report"],
            expected_identity=identities["report"],
            maximum_bytes=MAX_EVIDENCE_BYTES,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"evaluation report must be readable JSON: {exc}") from exc
    report_fields = (
        "namespace",
        "gate_id",
        "source_commit",
        "source_tree_sha",
        "config_sha",
        "physical_identity",
        "checks",
        "metrics",
        "seed_panel",
        "sample_rung",
    )
    if not isinstance(report, dict) or any(
        report.get(name) != evidence[name] for name in report_fields
    ):
        raise ValueError(
            "evaluation report does not bind the declared evidence content"
        )
    if evidence["gate_id"] == "P0":
        required_flags = (
            "source_clean",
            "source_stable_before_after",
            "fixed_nodes_collected",
            "optimizer_steps_forbidden_by_harness",
        )
        if (
            report.get("schema") != "V10PhysicalP0ContractEvaluatorV3"
            or report.get("status") != "READY_FOR_EXTERNAL_REVIEW"
            or report.get("pytest_returncode") != 0
            or any(report.get(name) is not True for name in required_flags)
            or report.get("evaluator_sha256") != verified["evaluator"]
        ):
            raise ValueError(
                "P0 report is not a source-stable no-optimizer contract evaluation"
            )
        test_nodes = report.get("test_nodes")
        test_hashes = report.get("test_file_sha256")
        if (
            not isinstance(test_nodes, list)
            or not test_nodes
            or len(test_nodes) != len(set(test_nodes))
            or any(
                not isinstance(node, str)
                or node.count("::") != 2
                or not node.startswith("tests/")
                for node in test_nodes
            )
            or not isinstance(test_hashes, dict)
            or set(test_hashes) != {node.split("::", 1)[0] for node in test_nodes}
        ):
            raise ValueError("P0 report does not bind its exact collected test nodes")
        for path, digest in test_hashes.items():
            if require_digest(digest, f"P0 test file {path} SHA") != digest:
                raise ValueError("P0 report test file digest is invalid")
        for name in ("stdout_sha256", "junit_sha256", "collect_stdout_sha256"):
            require_digest(report.get(name), f"P0 report {name}")
        bindings = report.get("semantic_bindings")
        if not isinstance(bindings, dict) or set(bindings) != {
            "schema",
            "active_classifier_nll",
            "csm_estimands",
            "memory_reducer",
            "e0_route",
            "source_file_sha256",
        }:
            raise ValueError("P0 report lacks exact active semantic bindings")
        if bindings["schema"] != P0_SEMANTIC_BINDING_SCHEMA:
            raise ValueError("P0 active semantic binding schema is stale")
        classifier = bindings["active_classifier_nll"]
        if any(
            classifier.get(name) != value
            or type(classifier.get(name)) is not type(value)
            for name, value in P0_CLASSIFIER_OBJECTIVE_BINDING.items()
        ):
            raise ValueError(
                "P0 physical objective and effective-coverage binding is stale"
            )
        if (
            classifier.get("program_entrypoint") != "physical_program.main"
            or classifier.get("model_identity") != "C_N_FULL_V10"
            or classifier.get("physical_step") != "physical_classifier_step"
            or classifier.get("registered_step") != "classifier_registered_step"
            or classifier.get("physical_residual_builder") != "belief_control_residual"
            or classifier.get("registered_residual_builder")
            != "belief_residual_details"
            or classifier.get("coordinate_interleaver")
            != "interleave_control_coordinates"
            or classifier.get("covariance_builder")
            != (
                "ml_recon.physics_hybrid_v10.control_coordinate_layout."
                "interleaved_joint_covariance"
            )
            or classifier.get("coordinate_layout_sha") != INTERLEAVED_CONTROL_LAYOUT_SHA
        ):
            raise ValueError("P0 classifier/NLL binding differs from the active route")
        csm = bindings["csm_estimands"]
        if (
            csm.get("raw_schema") != "V10AodRawCSMGeometryV1"
            or csm.get("raw_identity") != "RAW_CSM"
            or csm.get("profiled_schema") != "V10AodProfiledCSMGeometryV1"
            or csm.get("profiled_identity") != "PROFILED_CSM"
            or csm.get("profile_contract_schema") != "V10ProfiledCSMContractV2"
            or csm.get("profile_provenance_fields")
            != [
                "profile_contract_sha",
                "profile_calibration_source_sha",
                "profile_contract_path",
                "profile_contract_file_sha",
            ]
        ):
            raise ValueError("P0 CSM estimand/profile binding is incomplete")
        memory = bindings["memory_reducer"]
        if (
            memory.get("event_schema") != MEMORY_EVENT_SCHEMA
            or memory.get("rejected_legacy_event_schemas")
            != [LEGACY_MEMORY_EVENT_SCHEMA_V3, LEGACY_MEMORY_EVENT_SCHEMA_V4]
            or memory.get("online_reducer") != "apply_memory_event"
            or memory.get("offline_reducer") != "replay_memory_events"
            or memory.get("legacy_v1_migration") != "REJECT"
            or memory.get("legacy_v3_migration") != "REJECT"
            or memory.get("legacy_v4_migration") != "REJECT"
            or memory.get("protocol_contract") != "Memory_horizon_protocol"
            or memory.get("outcome_builder") != "evaluate_memory_attempt"
        ):
            raise ValueError("P0 online/offline reducer binding is incomplete")
        e0_route = bindings["e0_route"]
        if e0_route != {
            "stage": "E0",
            "component": "classifier",
            "model_profile": "C-TEST",
            "training_plan_schema": "V10PhysicalTrainingPlanV4",
            "n_values": [3, 4],
            "requested_mode_shots": 1024,
            "samples_per_n": 256,
            "split_sizes": [8, 4, 4],
            "train_n_mode_counts": {"3": 4, "4": 4},
            "validation_n_mode_counts": {"3": 2, "4": 2},
            "data_seed": 20261001,
            "memory_targets": "NOT_AVAILABLE",
            "memory_losses_enabled": False,
        }:
            raise ValueError("P0 E0 route admits an undeclared fallback")
        source_hashes = bindings["source_file_sha256"]
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(
            P0_SEMANTIC_SOURCE_PATHS
        ):
            raise ValueError("P0 semantic implementation files are incomplete")
        for path, digest in source_hashes.items():
            require_digest(digest, f"P0 semantic source {path} SHA")
    if evidence["gate_id"] == "S5":
        if verified["threshold_registration"] != evidence["threshold_registration_sha"]:
            raise ValueError("S5 threshold registration SHA is not bound to its file")
        _, registration = load_json_snapshot(
            evidence_root
            / evidence["evidence_file_roles"]["threshold_registration"]["path"],
            "S5 threshold registration",
            expected_sha=verified["threshold_registration"],
            expected_identity=identities["threshold_registration"],
            maximum_bytes=MAX_EVIDENCE_BYTES,
        )
        fields = (
            "source_commit",
            "source_tree_sha",
            "config_sha",
            "physical_identity",
            "minimum_meaningful_effect",
            "registered_at",
        )
        if not isinstance(registration, dict) or any(
            registration.get(name) != evidence[name] for name in fields
        ):
            raise ValueError(
                "S5 threshold registration is not bound to source and protocol identity"
            )


def validate_physical_gate_artifact_snapshot(
    *,
    raw_bytes: bytes,
    parsed_value: Mapping[str, object],
    expected_gate_id: str,
    expected_source_commit: str,
    expected_source_tree_sha: str,
    expected_config_sha: str,
    expected_physical_identity: Mapping[str, str],
    expected_parent_evidence_shas: Mapping[str, str],
    expected_parent_content_shas: Mapping[str, str],
    evidence_root: str | Path,
    gate_config: GateConfig | None = None,
    expected_threshold_registration_sha: str | None = None,
    expected_artifact_file_sha: str | None = None,
) -> dict[str, Any]:
    """Validate one already-read physical artifact snapshot without reopening it.

    The supplied parsed mapping is treated as an untrusted convenience value. The
    semantic object is always re-parsed from ``raw_bytes`` so descriptor-bound
    callers cannot substitute a separately read artifact between hashing and
    validation. Referenced evidence files remain independently file-backed.
    """
    if not isinstance(raw_bytes, bytes):
        raise ValueError("physical gate artifact snapshot requires bytes")
    if len(raw_bytes) > MAX_EVIDENCE_BYTES:
        raise ValueError("physical gate artifact exceeds its size limit")
    try:
        snapshot_value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=strict_object_pairs,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("physical gate artifact must be UTF-8 JSON") from exc
    if not isinstance(snapshot_value, dict):
        raise ValueError("physical gate artifact must be a JSON object")
    if not isinstance(parsed_value, Mapping) or dict(parsed_value) != snapshot_value:
        raise ValueError(
            "physical gate artifact parsed value differs from snapshot bytes"
        )
    raw = raw_bytes
    artifact_file_sha = hashlib.sha256(raw).hexdigest()
    if expected_artifact_file_sha is not None and artifact_file_sha != require_digest(
        expected_artifact_file_sha, "expected_artifact_file_sha"
    ):
        raise ValueError("physical gate artifact bytes changed; snapshot SHA mismatch")
    value = snapshot_value
    required = {
        "schema",
        "namespace",
        "gate_id",
        "status",
        "evidence_sha",
        "evidence",
        "gate_spec_sha",
        "source_commit",
        "source_tree_sha",
        "config_sha",
        "physical_identity",
        "parent_evidence_shas",
        "parent_content_shas",
        "authorization_scope",
        "issuer",
        "issued_at",
        "authority",
        "artifact_content_sha",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("physical artifact field contract mismatch")
    spec = physical_gate_spec(expected_gate_id, gate_config)
    if (
        value["schema"] != PHYSICAL_ARTIFACT_SCHEMA
        or value["namespace"] != PHYSICAL_PROGRAM_NAMESPACE
        or value["gate_id"] != expected_gate_id
        or value["status"] != "PASS"
        or value["gate_spec_sha"] != spec["gate_spec_sha"]
        or value["authorization_scope"] != [spec["authorization_scope"]]
    ):
        raise ValueError(
            "physical artifact has stale schema, namespace, scope, Gate or status"
        )
    evidence = validate_physical_gate_evidence(
        value["evidence"],
        expected_gate_id=expected_gate_id,
        expected_source_commit=expected_source_commit,
        expected_source_tree_sha=expected_source_tree_sha,
        expected_config_sha=expected_config_sha,
        expected_physical_identity=expected_physical_identity,
        gate_config=gate_config,
        expected_threshold_registration_sha=expected_threshold_registration_sha,
    )
    for name in ("source_commit", "source_tree_sha", "config_sha", "physical_identity"):
        if value[name] != evidence[name]:
            raise ValueError(f"physical artifact and evidence disagree on {name}")
    if value["evidence_sha"] != sha256_of(evidence):
        raise ValueError("physical artifact evidence SHA mismatch")
    for name, expected in (
        ("parent_evidence_shas", expected_parent_evidence_shas),
        ("parent_content_shas", expected_parent_content_shas),
    ):
        if _digest_map(value[name], name, allow_empty=True) != _digest_map(
            dict(expected), name, allow_empty=True
        ):
            raise ValueError("physical artifact parent lineage mismatch")
    issuer = value["issuer"]
    if (
        not isinstance(issuer, dict)
        or set(issuer) != {"identity", "role"}
        or any(not isinstance(item, str) or not item for item in issuer.values())
    ):
        raise ValueError("physical artifact must identify its independent reviewer")
    if (
        not isinstance(value["issued_at"], str)
        or datetime.fromisoformat(value["issued_at"].replace("Z", "+00:00")).tzinfo
        is None
    ):
        raise ValueError("physical artifact needs a timezone-aware issuance time")
    if value["authority"] != AUTHORITY or any(
        type(item) is not bool for item in value["authority"].values()
    ):
        raise ValueError("physical artifact authority is not research-only")
    content = {
        key: item for key, item in value.items() if key != "artifact_content_sha"
    }
    if value["artifact_content_sha"] != sha256_of(content):
        raise ValueError("physical artifact content SHA mismatch")
    resolved_evidence_root = validate_evidence_directory(evidence_root, "evidence_root")
    verified_files, verified_identities = _verify_evidence_file_records(
        resolved_evidence_root, evidence["evidence_file_roles"]
    )
    _require_exact_evidence_roles(
        evidence,
        verified_files,
        verified_identities,
        resolved_evidence_root,
    )
    identity_payload = {
        role: {
            "path": evidence["evidence_file_roles"][role]["path"],
            "sha256": verified_files[role],
            "file_identity": asdict(verified_identities[role]),
        }
        for role in sorted(verified_files)
    }
    return {
        **value,
        "artifact_file_sha": artifact_file_sha,
        "verified_evidence_files": verified_files,
        "verified_evidence_identity_sha": sha256_of(identity_payload),
        "verified": True,
    }


def verify_physical_gate_artifact(
    artifact_path: str | Path,
    *,
    expected_gate_id: str,
    expected_source_commit: str,
    expected_source_tree_sha: str,
    expected_config_sha: str,
    expected_physical_identity: Mapping[str, str],
    expected_parent_evidence_shas: Mapping[str, str],
    expected_parent_content_shas: Mapping[str, str],
    evidence_root: str | Path,
    gate_config: GateConfig | None = None,
    expected_threshold_registration_sha: str | None = None,
    expected_artifact_file_sha: str | None = None,
) -> dict[str, Any]:
    snapshot, value = load_json_snapshot(
        artifact_path,
        "physical gate artifact",
        expected_sha=expected_artifact_file_sha,
        maximum_bytes=MAX_EVIDENCE_BYTES,
    )
    return validate_physical_gate_artifact_snapshot(
        raw_bytes=snapshot.raw_bytes,
        parsed_value=value,
        expected_gate_id=expected_gate_id,
        expected_source_commit=expected_source_commit,
        expected_source_tree_sha=expected_source_tree_sha,
        expected_config_sha=expected_config_sha,
        expected_physical_identity=expected_physical_identity,
        expected_parent_evidence_shas=expected_parent_evidence_shas,
        expected_parent_content_shas=expected_parent_content_shas,
        evidence_root=evidence_root,
        gate_config=gate_config,
        expected_threshold_registration_sha=expected_threshold_registration_sha,
        expected_artifact_file_sha=expected_artifact_file_sha,
    )


class Physical_gate_ledger(TrackAGateLedger):
    """Reuse event ordering/invalidation while binding every event to R16."""

    def _append_event(self, record: GateRecord, **options: Any) -> GateLedgerEvent:
        event = super()._append_event(record, **options)
        content = event.to_dict()
        content.pop("content_sha")
        content.update(
            schema=PHYSICAL_EVENT_SCHEMA, namespace=PHYSICAL_PROGRAM_NAMESPACE
        )
        bound = replace(event, content_sha=sha256_of(content))
        self._events[-1] = bound
        self._current_events[record.gate_id] = bound
        return bound

    def record_external_artifact(
        self,
        artifact_path: str | Path,
        *,
        gate_id: str,
        expected_source_commit: str,
        expected_source_tree_sha: str,
        expected_config_sha: str,
        expected_physical_identity: Mapping[str, str],
        evidence_root: str | Path,
        gate_config: GateConfig | None = None,
        notes: str = "",
        expected_threshold_registration_sha: str | None = None,
        expected_artifact_file_sha: str | None = None,
    ) -> GateRecord:
        if gate_id not in TRACK_A_GATE_DEPENDENCIES:
            raise ValueError("unknown physical Gate")
        self._validate_identity_pin(
            source_commit=expected_source_commit,
            source_tree_sha=expected_source_tree_sha,
            config_sha=expected_config_sha,
            physical_identity=expected_physical_identity,
            gate_id=gate_id,
        )
        if any(
            not self._lineage_is_current(parent, require_external_artifact=True)
            for parent in TRACK_A_GATE_DEPENDENCIES[gate_id]
        ):
            raise ValueError(
                "physical external artifact requires current external parent decisions"
            )
        parent_evidence, parent_content = self._parent_identities(gate_id)
        artifact = verify_physical_gate_artifact(
            artifact_path,
            expected_gate_id=gate_id,
            expected_source_commit=expected_source_commit,
            expected_source_tree_sha=expected_source_tree_sha,
            expected_config_sha=expected_config_sha,
            expected_physical_identity=expected_physical_identity,
            expected_parent_evidence_shas=dict(parent_evidence),
            expected_parent_content_shas=dict(parent_content),
            evidence_root=evidence_root,
            gate_config=gate_config,
            expected_threshold_registration_sha=expected_threshold_registration_sha,
            expected_artifact_file_sha=expected_artifact_file_sha,
        )
        result = self._record_decision(
            gate_id,
            GateStatus.PASS,
            evidence_sha=artifact["artifact_content_sha"],
            notes=notes
            or "Verified physical external artifact; execution authority remains separate",
            evidence_origin="EXTERNAL_ARTIFACT",
            artifact_file_sha=artifact["artifact_file_sha"],
            authorization_scope=physical_gate_spec(gate_id, gate_config)[
                "authorization_scope"
            ],
            gate_spec_sha=artifact["gate_spec_sha"],
        )
        if gate_id == "P0" and self._physical_identity_pin is None:
            self._physical_identity_pin = {
                "source_commit": artifact["source_commit"],
                "source_tree_sha": artifact["source_tree_sha"],
                "config_sha": artifact["config_sha"],
                "physical_identity": artifact["physical_identity"],
            }
        return result

    def report(self) -> dict[str, Any]:
        report = super().report()
        report.pop("ledger_sha")
        report.update(
            schema=PHYSICAL_LEDGER_SCHEMA,
            namespace=PHYSICAL_PROGRAM_NAMESPACE,
            authority=dict(AUTHORITY),
            physical_identity_pin=self._physical_identity_pin,
        )
        report["events"] = tuple(
            {
                **event,
                "schema": PHYSICAL_EVENT_SCHEMA,
                "namespace": PHYSICAL_PROGRAM_NAMESPACE,
            }
            for event in report["events"]
        )
        return {**report, "ledger_sha": sha256_of(report)}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._physical_identity_pin: dict[str, Any] | None = None

    def _validate_identity_pin(
        self,
        *,
        source_commit: str,
        source_tree_sha: str,
        config_sha: str,
        physical_identity: Mapping[str, str],
        gate_id: str,
    ) -> None:
        identity = {
            "source_commit": require_digest(source_commit, "source_commit", git=True),
            "source_tree_sha": require_digest(
                source_tree_sha, "source_tree_sha", git=True
            ),
            "config_sha": require_digest(config_sha, "config_sha"),
            "physical_identity": validate_physical_identity(physical_identity),
        }
        if (
            self._physical_identity_pin is not None
            and identity != self._physical_identity_pin
        ):
            raise ValueError(
                "physical ledger identity differs from the immutable P0 pin"
            )
        if gate_id != "P0" and self._physical_identity_pin is None:
            raise ValueError(
                "physical descendants require a pinned external P0 identity"
            )
