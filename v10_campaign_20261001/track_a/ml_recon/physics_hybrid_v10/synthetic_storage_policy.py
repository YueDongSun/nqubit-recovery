"""Confine historical materialized tests to an owned, disposable directory."""

from __future__ import annotations

import hashlib
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, Sequence, TypeVar


DEFAULT_MAXIMUM_ROWS = 128
MAXIMUM_FIXTURE_ROWS = 256
FIXTURE_DIRECTORY_PREFIX = "v10_legacy_fixture_"


class Closeable_dataset(Protocol):
    def close(self) -> None:
        ...


Dataset_type = TypeVar("Dataset_type", bound=Closeable_dataset)


@dataclass
class _Fixture_scope_state:
    directory: tempfile.TemporaryDirectory
    root_identity: tuple[int, int]
    active: bool = True
    admission: Legacy_fixture_admission | None = None
    resources: ExitStack = field(default_factory=ExitStack)


@dataclass(frozen=True)
class Legacy_fixture_admission:
    """In-process capability issued only for a live disposable test scope.

    This is resource ownership, not a scientific or training admission. A
    copied dataclass or serialized declaration cannot recreate a live scope.
    """

    disposable_root: Path
    disposable_root_sha: str
    test_id: str
    maximum_rows: int
    _scope: _Fixture_scope_state = field(repr=False, compare=False)

    def own_dataset(self, dataset: Dataset_type) -> Dataset_type:
        """Close a fixture reader even when construction or training later fails."""
        self.validate_paths(())
        close = getattr(dataset, "close", None)
        if not callable(close):
            raise TypeError("a fixture dataset must expose deterministic close")
        self._scope.resources.callback(close)
        return dataset

    def validate_paths(self, paths: Sequence[str | Path]) -> None:
        scope = self._scope
        if not scope.active or scope.admission is not self:
            raise ValueError("legacy fixture admission is not an active owned scope")
        root = self.disposable_root
        if root != Path(scope.directory.name).resolve() or not root.is_dir():
            raise ValueError("legacy fixture root ownership changed")
        root_stat = root.lstat()
        if (root_stat.st_dev, root_stat.st_ino) != scope.root_identity:
            raise ValueError("legacy fixture root identity changed")
        expected_sha = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        if self.disposable_root_sha != expected_sha:
            raise ValueError("legacy fixture disposable-root SHA mismatch")
        for value in paths:
            if not value:
                continue
            path = Path(value).absolute()
            resolved = path.resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError("legacy fixture paths must stay inside the disposable root")
            for component in (path, *path.parents):
                if component.exists() or component.is_symlink():
                    attributes = getattr(component.lstat(), "st_file_attributes", 0)
                    if component.is_symlink() or attributes & getattr(
                            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                        raise ValueError("legacy fixture paths cannot contain links or junctions")
                if component == root:
                    break


@contextmanager
def legacy_fixture_scope(
    *, test_id: str, maximum_rows: int = DEFAULT_MAXIMUM_ROWS,
) -> Iterator[Legacy_fixture_admission]:
    """Own and remove every materialized sample/output of a bounded test."""
    if not isinstance(test_id, str) or not test_id.strip():
        raise ValueError("legacy fixture requires an explicit test identity")
    if type(maximum_rows) is not int or not 1 <= maximum_rows <= MAXIMUM_FIXTURE_ROWS:
        raise ValueError("legacy fixture row budget must be a bounded positive integer")
    directory = tempfile.TemporaryDirectory(prefix=FIXTURE_DIRECTORY_PREFIX)
    root = Path(directory.name).resolve()
    root_stat = root.lstat()
    scope = _Fixture_scope_state(directory, (root_stat.st_dev, root_stat.st_ino))
    admission = Legacy_fixture_admission(
        disposable_root=root,
        disposable_root_sha=hashlib.sha256(str(root).encode("utf-8")).hexdigest(),
        test_id=test_id,
        maximum_rows=maximum_rows,
        _scope=scope,
    )
    scope.admission = admission
    try:
        yield admission
    finally:
        # Validate the exact owned absolute target before recursive cleanup.
        # Never adopt or remove an arbitrary caller-supplied directory.
        try:
            scope.resources.close()
            if root.exists():
                admission.validate_paths((root,))
        finally:
            scope.active = False
        directory.cleanup()


def require_legacy_fixture_mode(
    run_mode: str, *, fixture_admission: Legacy_fixture_admission | None = None,
    output_dir: str = "", dataset_root: str = "",
    artifact_paths: Sequence[str | Path] = (),
) -> Legacy_fixture_admission:
    """Reject public persistence before rendering, model creation, or updates."""
    if run_mode != "smoke":
        raise ValueError(
            "materialized synthetic production training is retired; use the reviewed stream-only entrypoint")
    if not isinstance(fixture_admission, Legacy_fixture_admission):
        raise ValueError("materialized smoke requires an active disposable fixture admission")
    if not output_dir:
        raise ValueError("legacy fixture requires an explicit disposable output directory")
    fixture_admission.validate_paths((output_dir, dataset_root, *artifact_paths))
    return fixture_admission


def require_legacy_fixture_rows(
    fixture_admission: Legacy_fixture_admission, row_count: int,
) -> None:
    """Check both planned generation and selected rows before optimization."""
    fixture_admission.validate_paths(())
    if type(row_count) is not int or not 1 <= row_count <= fixture_admission.maximum_rows:
        raise ValueError("materialized smoke exceeds its disposable fixture row budget")


def reject_materialized_cli(parser) -> None:
    """No public legacy CLI can silently create a new synthetic corpus."""
    parser.error(
        "materialized synthesis CLI is retired; classifier uses "
        "ml_recon.physics_hybrid_v10.train_streaming_classifier; "
        "online feedback remains blocked pending reviewed frozen-output admission")
