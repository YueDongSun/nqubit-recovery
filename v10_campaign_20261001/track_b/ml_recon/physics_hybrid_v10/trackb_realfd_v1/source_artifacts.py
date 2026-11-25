"""File-backed provenance bindings for actionful Track B research proposals.

The FD bridge remains research-only, but an actionful proposal must not be
constructed from caller-supplied digest strings.  This module owns the narrow
boundary that rehashes the producer's actual local artifacts every time they
are admitted or replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .posterior_producer import REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH


FORWARD_SOURCE_ARTIFACT_NAMES = (
    "operation_descriptor",
    "rf_descriptor",
    "session_context",
    "exposure_context",
    "draw_parameters",
    "base_point",
    "nuisance_point",
    "forward_callable_source",
    "covariance_callable_source",
)
POSTERIOR_SOURCE_ARTIFACT_NAMES = (
    "posterior_payload",
    "model_checkpoint",
    "model_output_bundle",
    "data_manifest",
    "row_identity",
    "batch_plan",
    "producer_source",
    "producer_config",
    "calibration_coverage",
    "producer_manifest",
)
ACTIONFUL_SOURCE_ARTIFACT_NAMES = tuple(
    sorted(set(FORWARD_SOURCE_ARTIFACT_NAMES) | set(POSTERIOR_SOURCE_ARTIFACT_NAMES))
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


SOURCE_CODE_ARTIFACT_NAMES = frozenset({
    "producer_source",
    "forward_callable_source",
    "covariance_callable_source",
})


def _registered_posterior_producer_path(checkout_root: Path, source_path: Path) -> Path:
    try:
        relative_path = source_path.resolve().relative_to(checkout_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Track B posterior producer must live inside its verified checkout") from error
    if relative_path != REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH:
        raise ValueError("Track B actionful posterior requires the registered producer source path")
    return source_path.resolve()


def _run_registered_posterior_producer(
    *, checkout_root: Path, source_commit: str, source_tree: str,
    producer_source_path: Path, producer_source_git_blob_oid: str,
    artifact_paths: Mapping[str, Path],
) -> bytes:
    """Execute a source-byte snapshot of the only registered posterior producer."""
    registered_source = _registered_posterior_producer_path(checkout_root, producer_source_path)
    source_bytes = registered_source.read_bytes()
    with tempfile.TemporaryDirectory(prefix="trackb_posterior_producer_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        snapshot_path = temporary_root / "audited_posterior_producer.py"
        request_path = temporary_root / "producer_request.json"
        snapshot_path.write_bytes(source_bytes)
        request_path.write_bytes(canonical_json_bytes({
            "schema": "TrackBPosteriorProducerRequestV1",
            "source_commit": source_commit,
            "source_tree": source_tree,
            "producer_source_git_blob_oid": producer_source_git_blob_oid,
            "artifact_paths": {
                name: str(path.resolve())
                for name, path in sorted(artifact_paths.items())
            },
        }))
        completed = subprocess.run(
            [sys.executable, "-I", str(snapshot_path), str(request_path)],
            cwd=checkout_root,
            check=False,
            capture_output=True,
            text=False,
            timeout=60,
        )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or b"posterior producer failed").decode(
            "utf-8", errors="replace",
        ).strip()
        raise ValueError(f"Track B registered posterior producer failed: {diagnostic[:2000]}")
    return completed.stdout


def emit_registered_posterior_producer_receipt(
    *, checkout_root: str | Path, artifact_paths: Mapping[str, str | Path],
) -> bytes:
    """Emit a receipt by executing the registered source-byte producer once."""
    resolved_root = Path(checkout_root).resolve()
    source_commit, source_tree = _clean_checkout_identity(resolved_root)
    paths = {name: Path(path).resolve() for name, path in artifact_paths.items()}
    if "producer_manifest" in paths:
        raise ValueError("posterior producer receipt input must exclude its output manifest")
    producer_source_path = paths.get("producer_source")
    if producer_source_path is None:
        raise ValueError("posterior producer receipt input requires producer_source")
    producer_source = _registered_posterior_producer_path(resolved_root, producer_source_path)
    producer_blob = _tracked_source_blob_oid(resolved_root, producer_source)
    return _run_registered_posterior_producer(
        checkout_root=resolved_root,
        source_commit=source_commit,
        source_tree=source_tree,
        producer_source_path=producer_source,
        producer_source_git_blob_oid=producer_blob,
        artifact_paths=paths,
    )


def _git_output(checkout_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Track B source bundle requires a readable Git checkout") from error
    return completed.stdout.strip()


def _clean_checkout_identity(checkout_root: Path) -> tuple[str, str]:
    if not checkout_root.is_absolute() or not checkout_root.is_dir():
        raise ValueError("Track B source bundle checkout root must be an existing absolute directory")
    commit = _git_output(checkout_root, "rev-parse", "HEAD")
    tree = _git_output(checkout_root, "rev-parse", "HEAD^{tree}")
    status = _git_output(checkout_root, "status", "--porcelain")
    require_git_oid(commit, "checkout_source_commit")
    require_git_oid(tree, "checkout_source_tree")
    if status:
        raise ValueError("Track B source bundle requires a clean Git checkout")
    return commit, tree


def _tracked_source_blob_oid(checkout_root: Path, source_path: Path) -> str:
    try:
        relative_path = source_path.resolve().relative_to(checkout_root.resolve())
    except ValueError as error:
        raise ValueError("Track B producer source must live inside its verified checkout") from error
    relative_text = relative_path.as_posix()
    try:
        subprocess.run(
            ["git", "-C", str(checkout_root), "ls-files", "--error-unmatch", "--", relative_text],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout_root), "diff", "--quiet", "HEAD", "--", relative_text],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Track B producer source is not a clean tracked checkout file") from error
    blob_oid = _git_output(checkout_root, "rev-parse", f"HEAD:{relative_text}")
    require_git_oid(blob_oid, "producer_source_git_blob_oid")
    return blob_oid


@dataclass(frozen=True)
class TrackBSourceArtifactFileV1:
    """One local file whose present bytes must equal the bound digest."""

    name: str
    path: str
    sha256: str
    git_blob_oid: str | None = None

    def __post_init__(self) -> None:
        if self.name not in ACTIONFUL_SOURCE_ARTIFACT_NAMES:
            raise ValueError("Track B source artifact name is not registered")
        if type(self.path) is not str:
            raise TypeError("Track B source artifact path must be a string")
        artifact_path = Path(self.path)
        if not artifact_path.is_absolute() or not artifact_path.is_file():
            raise ValueError("Track B source artifact path must be an existing absolute file")
        require_sha256(self.sha256, "source_artifact_sha256")
        if self.git_blob_oid is not None:
            require_git_oid(self.git_blob_oid, "source_artifact_git_blob_oid")
        if _file_sha256(artifact_path) != self.sha256:
            raise ValueError("Track B source artifact bytes do not match their declared SHA")

    def assert_current(self) -> None:
        if _file_sha256(Path(self.path)) != self.sha256:
            raise ValueError("Track B source artifact changed after admission")

    def to_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "sha256": self.sha256, "git_blob_oid": self.git_blob_oid}


@dataclass(frozen=True)
class TrackBActionfulSourceArtifactBundleV1:
    """Rehashable source bundle shared by posterior and forward producers."""

    source_commit: str
    source_tree: str
    checkout_root: str
    artifacts: tuple[TrackBSourceArtifactFileV1, ...]

    def __post_init__(self) -> None:
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        if type(self.checkout_root) is not str:
            raise TypeError("Track B source bundle checkout root must be a string")
        checkout_root = Path(self.checkout_root).resolve()
        checkout_commit, checkout_tree = _clean_checkout_identity(checkout_root)
        if (checkout_commit, checkout_tree) != (self.source_commit, self.source_tree):
            raise ValueError("Track B source bundle commit/tree do not match its clean checkout")
        if not self.artifacts or any(type(item) is not TrackBSourceArtifactFileV1 for item in self.artifacts):
            raise TypeError("Track B source bundle requires exact file bindings")
        names = tuple(item.name for item in self.artifacts)
        if len(set(names)) != len(names) or tuple(sorted(names)) != names:
            raise ValueError("Track B source bundle artifact names must be unique and sorted")
        for artifact in self.artifacts:
            artifact.assert_current()
            if artifact.name in SOURCE_CODE_ARTIFACT_NAMES:
                if artifact.git_blob_oid != _tracked_source_blob_oid(checkout_root, Path(artifact.path)):
                    raise ValueError("Track B source artifact does not match its tracked Git blob")
        if set(POSTERIOR_SOURCE_ARTIFACT_NAMES).issubset(names):
            self._assert_registered_posterior_producer_receipt(checkout_root)

    @classmethod
    def from_paths(
        cls,
        *,
        checkout_root: str | Path,
        artifact_paths: Mapping[str, str | Path],
    ) -> "TrackBActionfulSourceArtifactBundleV1":
        unknown = set(artifact_paths) - set(ACTIONFUL_SOURCE_ARTIFACT_NAMES)
        if unknown:
            raise ValueError("Track B source bundle has an unregistered artifact name")
        resolved_root = Path(checkout_root).resolve()
        source_commit, source_tree = _clean_checkout_identity(resolved_root)
        bindings = []
        for name in sorted(artifact_paths):
            path = Path(artifact_paths[name]).resolve()
            bindings.append(TrackBSourceArtifactFileV1(
                name=name,
                path=str(path),
                sha256=_file_sha256(path),
                git_blob_oid=(
                    _tracked_source_blob_oid(resolved_root, path)
                    if name in SOURCE_CODE_ARTIFACT_NAMES else None
                ),
            ))
        return cls(
            source_commit=source_commit,
            source_tree=source_tree,
            checkout_root=str(resolved_root),
            artifacts=tuple(bindings),
        )

    def artifact(self, name: str) -> TrackBSourceArtifactFileV1:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise ValueError(f"Track B source bundle is missing {name}")

    def artifact_sha256(self, name: str) -> str:
        return self.artifact(name).sha256

    def read_bytes(self, name: str) -> bytes:
        artifact = self.artifact(name)
        payload = Path(artifact.path).read_bytes()
        if _bytes_sha256(payload) != artifact.sha256:
            raise ValueError("Track B source artifact changed before its bytes were consumed")
        return payload

    def read_canonical_json_object(self, name: str) -> dict[str, object]:
        payload = self.read_bytes(name)

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Track B source artifact JSON has duplicate keys")
                result[key] = value
            return result

        try:
            decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Track B source artifact is not UTF-8 canonical JSON") from error
        if type(decoded) is not dict or canonical_json_bytes(decoded) != payload:
            raise ValueError("Track B source artifact JSON is not canonical")
        return decoded

    def _assert_registered_posterior_producer_receipt(self, checkout_root: Path) -> None:
        producer_source = self.artifact("producer_source")
        producer_source_path = _registered_posterior_producer_path(
            checkout_root, Path(producer_source.path),
        )
        if producer_source.git_blob_oid is None:
            raise ValueError("Track B registered posterior producer has no Git blob identity")
        artifact_paths = {
            name: Path(self.artifact(name).path)
            for name in ACTIONFUL_SOURCE_ARTIFACT_NAMES
            if name != "producer_manifest"
        }
        expected_receipt = _run_registered_posterior_producer(
            checkout_root=checkout_root,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            producer_source_path=producer_source_path,
            producer_source_git_blob_oid=producer_source.git_blob_oid,
            artifact_paths=artifact_paths,
        )
        if self.read_bytes("producer_manifest") != expected_receipt:
            raise ValueError("Track B producer manifest was not emitted by the registered source snapshot")

    def assert_has(self, names: Sequence[str]) -> None:
        expected = tuple(sorted(names))
        available = tuple(item.name for item in self.artifacts)
        if any(name not in available for name in expected):
            raise ValueError("Track B source bundle is missing a required artifact")

    def assert_current(self, names: Iterable[str] | None = None) -> None:
        selected = tuple(names) if names is not None else tuple(item.name for item in self.artifacts)
        available = tuple(item.name for item in self.artifacts)
        self.assert_has(selected)
        checkout_root = Path(self.checkout_root).resolve()
        if _clean_checkout_identity(checkout_root) != (self.source_commit, self.source_tree):
            raise ValueError("Track B source bundle checkout identity changed after admission")
        for artifact in self.artifacts:
            if artifact.name in SOURCE_CODE_ARTIFACT_NAMES and artifact.git_blob_oid != _tracked_source_blob_oid(
                checkout_root,
                Path(artifact.path),
            ):
                raise ValueError("Track B source artifact Git blob changed after admission")
        for name in selected:
            self.artifact(name).assert_current()
        if set(POSTERIOR_SOURCE_ARTIFACT_NAMES).issubset(available):
            self._assert_registered_posterior_producer_receipt(checkout_root)

    def binding_sha256(self, names: Sequence[str]) -> str:
        selected = tuple(sorted(names))
        self.assert_current(selected)
        return canonical_sha256({
            "schema": "TrackBSourceArtifactBindingV1",
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "artifacts": [self.artifact(name).to_dict() for name in selected],
        })

    @property
    def forward_binding_sha256(self) -> str:
        return self.binding_sha256(FORWARD_SOURCE_ARTIFACT_NAMES)

    @property
    def actionful_binding_sha256(self) -> str:
        self.assert_has(ACTIONFUL_SOURCE_ARTIFACT_NAMES)
        return self.binding_sha256(ACTIONFUL_SOURCE_ARTIFACT_NAMES)

    def to_dict(self) -> dict[str, object]:
        self.assert_current()
        return {
            "schema": "TrackBActionfulSourceArtifactBundleV1",
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = [
    "ACTIONFUL_SOURCE_ARTIFACT_NAMES",
    "FORWARD_SOURCE_ARTIFACT_NAMES",
    "POSTERIOR_SOURCE_ARTIFACT_NAMES",
    "SOURCE_CODE_ARTIFACT_NAMES",
    "emit_registered_posterior_producer_receipt",
    "TrackBActionfulSourceArtifactBundleV1",
    "TrackBSourceArtifactFileV1",
]
