"""Read-only source and physical-contract snapshots for a future teacher runner.

The binding is a consistency record, not an unforgeable authorization token.
The caller independently pins the source, sample key and recipe, and revalidates
against a fresh capture before future effects. Digests identify bytes; they do
not authenticate a caller or reviewer. These bounded reads do not continuously
lock Git or the filesystem and cannot exclude changes after revalidation.

This module does not execute the teacher kernel, publish an artifact, or grant
execution admission. Loading the V8 experimental contract does not qualify the
TrackB global_pauli_v24 forward as a complete DLCZ physical implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot, read_stable_bytes
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_bytes
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.physics_hybrid_v8.experimental_physics_contract import (
    V8ExperimentalPhysicsContractV1,
    experimental_physics_contract_audit,
    validate_experimental_physics_contract,
)

from . import pilot_runtime_checks
from .contracts import require_git_oid
from .teacher_diagnostic_v2 import Teacher_diagnostic_config_v2


# Keep the lexical anchor so a reparse component is rejected before resolution.
repository_root = Path(__file__).absolute().parents[3]
physical_config_path = "ml_recon/configs/physics_hybrid_v8_experimental_physics.json"
config_max_bytes = 256 * 1024
document_max_bytes = 2 * 1024 * 1024
reference_fields = (
    ("operator_ask_v2_amendment_path", "operator_ask_v2_amendment_sha256"),
    ("dlcz_double_excitation_contract_path", "dlcz_double_excitation_contract_sha256"),
    ("aod_optical_detuning_contract_path", "aod_optical_detuning_contract_sha256"),
)
reparse_attribute = 0x400


@dataclass(frozen=True)
class Physical_file_snapshot:
    """Exact owned immutable bytes, identified by a portable repository path."""

    relative_path: str
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if type(self.payload) is not bytes or type(self.sha256) is not str:
            raise TypeError("physical snapshot requires immutable bytes and a text digest")
        if sha256_bytes(self.payload) != self.sha256:
            raise ValueError("physical snapshot bytes differ from their digest")


@dataclass(frozen=True)
class Teacher_diagnostic_preflight_binding:
    """Portable evidence with raw bytes; use identity_summary for JSON reports.

    Neither the binding nor its JSON summary conveys execution authority. The
    summary omits raw documents/configuration and cannot replace revalidation.
    """

    source_commit: str
    source_tree: str
    sample_key: Canonical_json_snapshot
    recipe: Canonical_json_snapshot
    policy_id: str
    policy_sha256: str
    physical_config: Physical_file_snapshot
    physical_semantics: Canonical_json_snapshot
    physical_audit: Canonical_json_snapshot
    reference_documents: tuple[Physical_file_snapshot, ...]
    schema: str = "TrackBTeacherDiagnosticPreflightBindingV1"
    forward_domain: str = "global_pauli_v24"
    physical_check_scope: str = "REGISTERED_V8_CONTRACT_AND_REFERENCE_DOCUMENT_BYTES"
    full_dlcz_forward_qualified: bool = False
    kernel_executed: bool = False
    execution_admission: str = "NOT_GRANTED_PREFLIGHT_CONSISTENCY_ONLY"
    dedicated_runner_wiring: str = "NOT_ATTESTED_BY_PREFLIGHT"
    model_gain_measured: bool = False
    training_authorized: bool = False
    research_only: bool = True
    gate_decision: str = "NOT_PERFORMED"
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def identity_summary(self) -> dict[str, Any]:
        """Return fresh JSON-compatible identity metadata, without raw payloads."""
        return {
            "schema": "TrackBTeacherDiagnosticPreflightIdentityV1",
            "binding_schema": self.schema,
            "source_commit": self.source_commit, "source_tree": self.source_tree,
            "sample_key_sha256": self.sample_key.sha256, "recipe_sha256": self.recipe.sha256,
            "policy_id": self.policy_id, "policy_sha256": self.policy_sha256,
            "physical_config": {"relative_path": self.physical_config.relative_path,
                                "sha256": self.physical_config.sha256},
            "physical_semantic_sha256": self.physical_semantics.sha256,
            "physical_audit_sha256": self.physical_audit.sha256,
            "reference_documents": [{"relative_path": item.relative_path, "sha256": item.sha256}
                                    for item in self.reference_documents],
            "forward_domain": self.forward_domain, "physical_check_scope": self.physical_check_scope,
            "full_dlcz_forward_qualified": self.full_dlcz_forward_qualified,
            "kernel_executed": self.kernel_executed, "execution_admission": self.execution_admission,
            "dedicated_runner_wiring": self.dedicated_runner_wiring,
            "model_gain_measured": self.model_gain_measured, "training_authorized": self.training_authorized,
            "research_only": self.research_only, "gate_decision": self.gate_decision,
            "selection_evidence": self.selection_evidence, "promotion_ready": self.promotion_ready,
            "hardware_action_authorized": self.hardware_action_authorized,
        }


def _validate_relative_path(value: str) -> None:
    if (type(value) is not str or not value or "\\" in value or ":" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix() != value
            or not PurePosixPath(value).parts
            or any(part in (".", "..") for part in PurePosixPath(value).parts)):
        raise ValueError("physical-contract path must be a canonical relative repository path")


def _path_identities(relative_path: str) -> tuple[Path, tuple[tuple[int, int], ...]]:
    """Reject reparse points in every ancestor and check repository confinement."""
    _validate_relative_path(relative_path)
    root = repository_root
    if not root.is_absolute() or root != pilot_runtime_checks.PILOT_REPOSITORY_ROOT:
        raise ValueError("physical-contract repository anchor differs from pilot source anchor")
    source = root.joinpath(*PurePosixPath(relative_path).parts)
    paths = tuple(reversed(source.parents)) + (source,)
    identities = []
    try:
        for path in paths:
            observed = path.lstat()
            if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & reparse_attribute:
                raise ValueError("physical-contract path must not contain reparse points")
            if path != source and not stat.S_ISDIR(observed.st_mode):
                raise ValueError("physical-contract ancestor is not a directory")
            if path == source and not stat.S_ISREG(observed.st_mode):
                raise ValueError("physical-contract source is not a regular file")
            identities.append((observed.st_dev, observed.st_ino))
        if not source.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise ValueError("physical-contract source escapes its repository")
    except OSError as error:
        raise ValueError(f"physical-contract path is unavailable: {relative_path}") from error
    return source, tuple(identities)


def _read_physics_file(
    relative_path: str, *, max_bytes: int, expected_sha: str | None = None,
) -> Physical_file_snapshot:
    source, before = _path_identities(relative_path)
    payload = read_stable_bytes(
        source, label=f"physical contract {relative_path}", expected_sha=expected_sha,
        max_bytes=max_bytes, reject_reparse=True)
    _, after = _path_identities(relative_path)
    if before != after:
        raise ValueError("physical-contract path identity changed during stable read")
    return Physical_file_snapshot(relative_path, payload, sha256_bytes(payload))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"physical contract contains duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"physical contract contains nonfinite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("physical contract contains an overflowing JSON number")
    return result


def _parse_physics_config(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("physical contract is not a bounded valid UTF-8 JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("physical contract must contain a JSON object")
    return value


def capture_teacher_diagnostic_preflight(
    sample_key: V24SampleKey, *, config: Teacher_diagnostic_config_v2,
    expected_source: tuple[str, str],
) -> Teacher_diagnostic_preflight_binding:
    """Capture one read-only snapshot against independently pinned caller inputs.

    Source must be a clean exact commit/tree before and after reads. The fixed
    configuration is parsed and hashed from the same stable bytes. Reference
    hashes are those registered by the validated physical configuration.
    """
    if not isinstance(config, Teacher_diagnostic_config_v2):
        raise ValueError("a teacher diagnostic V2 configuration is required")
    config.validate(sample_key)
    if type(expected_source) is not tuple or len(expected_source) != 2:
        raise ValueError("expected_source must independently pin (commit, tree)")
    require_git_oid(expected_source[0], "expected_source_commit")
    require_git_oid(expected_source[1], "expected_source_tree")
    observed_source = pilot_runtime_checks.pilot_source_identity()
    if observed_source != expected_source:
        raise RuntimeError("teacher diagnostic source differs from independently pinned commit/tree")

    config_file = _read_physics_file(physical_config_path, max_bytes=config_max_bytes)
    payload = _parse_physics_config(config_file.payload)
    validate_experimental_physics_contract(payload)
    # Preserve V8's semantic encoding (ensure_ascii=False), with finite numbers.
    semantic_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    semantics = Canonical_json_snapshot(semantic_bytes, sha256_bytes(semantic_bytes))
    references = tuple(_read_physics_file(
        payload["decision_source"][path_key], max_bytes=document_max_bytes,
        expected_sha=payload["decision_source"][hash_key]) for path_key, hash_key in reference_fields)
    # Only this fresh, internally owned mapping enters the legacy audit type.
    contract = V8ExperimentalPhysicsContractV1(
        payload=payload, file_sha256=config_file.sha256,
        semantic_sha256=semantics.sha256, path=physical_config_path)
    audit_bytes = canonical_json_bytes(experimental_physics_contract_audit(contract))
    key_bytes, recipe_bytes = canonical_json_bytes(asdict(sample_key)), canonical_json_bytes(asdict(config))
    pilot_runtime_checks.require_unchanged_pilot_source(expected_source)
    return Teacher_diagnostic_preflight_binding(
        source_commit=observed_source[0], source_tree=observed_source[1],
        sample_key=Canonical_json_snapshot(key_bytes, sha256_bytes(key_bytes)),
        recipe=Canonical_json_snapshot(recipe_bytes, sha256_bytes(recipe_bytes)),
        policy_id=config.history_policy, policy_sha256=config.history_policy_sha256,
        physical_config=config_file, physical_semantics=semantics,
        physical_audit=Canonical_json_snapshot(audit_bytes, sha256_bytes(audit_bytes)),
        reference_documents=references)


def require_unchanged_teacher_diagnostic_preflight(
    expected_binding: Teacher_diagnostic_preflight_binding, *, sample_key: V24SampleKey,
    config: Teacher_diagnostic_config_v2, expected_source: tuple[str, str],
) -> Teacher_diagnostic_preflight_binding:
    """Re-read and compare all evidence using caller-pinned inputs, not claims.

    Do not derive expected_source/config/key from an untrusted candidate binding.
    Equality establishes a fresh consistency observation only. A future runner
    remains responsible for wiring this check before its own side effects.
    """
    if type(expected_binding) is not Teacher_diagnostic_preflight_binding:
        raise ValueError("expected binding must be a teacher diagnostic preflight snapshot")
    fresh = capture_teacher_diagnostic_preflight(sample_key, config=config, expected_source=expected_source)
    if fresh != expected_binding:
        raise RuntimeError("teacher diagnostic preflight binding changed or does not match pinned inputs")
    return fresh


__all__ = ["Physical_file_snapshot", "Teacher_diagnostic_preflight_binding",
           "capture_teacher_diagnostic_preflight", "require_unchanged_teacher_diagnostic_preflight"]
