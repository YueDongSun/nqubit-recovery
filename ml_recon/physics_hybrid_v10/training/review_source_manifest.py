"""Strict legacy/V2 review-source-manifest recognition for physical admission."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable, Mapping

from ml_recon.v10_recoverability_foundry.training.physical_gates import require_digest
from ml_recon.physics_hybrid_v10.evidence_file_io import (
    sha256_evidence_file,
    validate_evidence_path,
)


LEGACY_MANIFEST_FIELDS = {
    "repo_root", "branch", "source_commit", "source_tree_sha", "files"
}
V2_MANIFEST_FIELDS = LEGACY_MANIFEST_FIELDS | {
    "manifest_schema",
    "runtime_dependencies",
    "runtime_dependency_audit",
    "excluded_data_assets",
}
V2_MANIFEST_SCHEMA = "V10ReviewSourceManifestV2"
EXCLUDED_DATA_REASON = "NO_RAW_EXPERIMENTAL_DATA_OR_LEGACY_NUMERIC_CALIBRATION_UPLOAD"
UNBOUND_REAL_DATA_SCOPE = "NOT_AVAILABLE_WITHOUT_SEPARATE_DATA_ADMISSION"


def _canonical_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or ":" in value or "\\" in value:
        raise ValueError(f"{label} path is invalid")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
        or not candidate.parts
        or any(part.rstrip(". ") != part for part in candidate.parts)
    ):
        raise ValueError(f"{label} path escapes the imported repository")
    return value


def _resolved_identity(
    root: Path, relative: str, label: str, *, require_file: bool
) -> tuple[Path, tuple[int, int] | None]:
    candidate = validate_evidence_path(
        root / relative, label, allow_missing=not require_file
    )
    if require_file and not candidate.exists():
        raise ValueError(f"{label} file escapes the imported repository")
    if not candidate.exists():
        return candidate, None
    stat_result = candidate.lstat()
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError(f"{label} file escapes the imported repository")
    return candidate, (stat_result.st_dev, stat_result.st_ino)


def _validate_file_hashes(
    entries: object,
    *,
    root: Path,
    sha256_file: Callable[[Path], str],
    label: str,
) -> dict[str, str]:
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{label} requires nonempty file hashes")
    validated: dict[str, str] = {}
    resolved_identities: set[tuple[int, int]] = set()
    for relative, digest in entries.items():
        relative = _canonical_relative_path(relative, label)
        candidate, identity = _resolved_identity(
            root, relative, label, require_file=True
        )
        if identity is None or identity in resolved_identities:
            raise ValueError(f"{label} contains duplicate filesystem path aliases")
        resolved_identities.add(identity)
        expected = require_digest(digest, f"{label} SHA")
        _, observed, descriptor_identity = sha256_evidence_file(
            candidate, f"{label} file"
        )
        if (descriptor_identity.device, descriptor_identity.inode) != identity:
            raise ValueError(f"{label} file identity changed")
        if observed != expected:
            raise ValueError(f"{label} file bytes changed")
        validated[relative] = expected
    return validated


def _validate_v2_metadata(
    manifest: Mapping[str, Any], files: Mapping[str, str], root: Path, sha256_file: Callable[[Path], str]
) -> None:
    if manifest["manifest_schema"] != V2_MANIFEST_SCHEMA:
        raise ValueError("source manifest V2 schema marker is invalid")
    runtime_dependencies = _validate_file_hashes(
        manifest["runtime_dependencies"],
        root=root,
        sha256_file=sha256_file,
        label="runtime dependency",
    )
    if any(files.get(path) != digest for path, digest in runtime_dependencies.items()):
        raise ValueError("runtime dependency hashes must equal reviewed file hashes")
    audit = manifest["runtime_dependency_audit"]
    audit_fields = {
        "method",
        "local_import_prefixes",
        "closure_rounds",
        "local_import_closure_complete",
        "files",
    }
    if (
        not isinstance(audit, dict)
        or set(audit) != audit_fields
        or audit["method"] != "AST_LITERAL_AND_LOCAL_IMPORT_FIXED_POINT_V2"
        or audit["local_import_prefixes"] != ["ml_recon", "system_control"]
        or type(audit["closure_rounds"]) is not int
        or not 1 <= audit["closure_rounds"] <= 32
        or audit["local_import_closure_complete"] is not True
    ):
        raise ValueError("runtime dependency audit has an exact invalid structure")
    audit_files = _validate_file_hashes(
        audit["files"], root=root, sha256_file=sha256_file, label="runtime dependency audit"
    )
    if audit_files != runtime_dependencies:
        raise ValueError("runtime dependency audit hashes must equal declared dependencies")
    excluded = manifest["excluded_data_assets"]
    excluded_fields = {"reason", "paths", "dependent_real_paths"}
    if (
        not isinstance(excluded, dict)
        or set(excluded) != excluded_fields
        or excluded["reason"] != EXCLUDED_DATA_REASON
        or excluded["dependent_real_paths"] != UNBOUND_REAL_DATA_SCOPE
        or not isinstance(excluded["paths"], list)
        or not excluded["paths"]
    ):
        raise ValueError("excluded data assets have an exact invalid structure")
    excluded_paths = [_canonical_relative_path(path, "excluded data asset") for path in excluded["paths"]]
    excluded_identities: set[tuple[int, int]] = set()
    excluded_resolved_paths: set[Path] = set()
    for relative in excluded_paths:
        candidate, identity = _resolved_identity(
            root, relative, "excluded data asset", require_file=False
        )
        if candidate in excluded_resolved_paths or (
            identity is not None and identity in excluded_identities
        ):
            raise ValueError("excluded data asset paths must be unique")
        if identity is not None:
            excluded_identities.add(identity)
        excluded_resolved_paths.add(candidate)
    if len(set(excluded_paths)) != len(excluded_paths):
        raise ValueError("excluded data asset paths must be unique")
    included_resolved = [
        _resolved_identity(root, relative, "included source", require_file=True)
        for relative in set(files) | set(runtime_dependencies)
    ]
    included_identities = {identity for _, identity in included_resolved if identity is not None}
    included_resolved_paths = {candidate for candidate, _ in included_resolved}
    if (
        excluded_identities & included_identities
        or excluded_resolved_paths & included_resolved_paths
        or any(
            included_path.is_relative_to(excluded_path)
            for excluded_path in excluded_resolved_paths
            for included_path in included_resolved_paths
        )
    ):
        raise ValueError("excluded data assets cannot appear in included source dependencies")


def validate_review_source_manifest(
    manifest: object,
    admission: Mapping[str, Any],
    *,
    root: Path,
    sha256_file: Callable[[Path], str],
    required_core: set[str],
) -> dict[str, str]:
    """Validate only legacy five-field or producer V2 nine-field manifests."""
    if not isinstance(manifest, dict):
        raise ValueError("source manifest has an exact repository schema")
    manifest_fields = set(manifest)
    if manifest_fields == LEGACY_MANIFEST_FIELDS:
        is_v2 = False
    elif manifest_fields == V2_MANIFEST_FIELDS:
        is_v2 = True
    else:
        raise ValueError("source manifest has an exact repository schema")
    if (
        manifest["branch"] != admission["branch"]
        or any(manifest[name] != admission[name] for name in ("source_commit", "source_tree_sha"))
        or manifest["repo_root"] not in ("", ".")
    ):
        raise ValueError("reviewed source manifest and admission identities disagree")
    files = _validate_file_hashes(
        manifest["files"], root=root, sha256_file=sha256_file, label="source manifest"
    )
    if not required_core.issubset(files):
        raise ValueError("source manifest does not cover required admitted core files")
    if is_v2:
        _validate_v2_metadata(manifest, files, root, sha256_file)
    return {str((root / relative).resolve()): digest for relative, digest in files.items()}
