"""Store experiment settings beside derived results and bind reusable model exports."""

from contextlib import contextmanager
import hashlib
import json
from ml_recon.physics_hybrid_v10.matrix_campaign import register_once
from pathlib import Path


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def register(path, settings):
    # Normalize tuples and integer dictionary keys before checking an existing JSON record.
    register_once(path, json.loads(json.dumps(settings, allow_nan=False)))


@contextmanager
def pending_panel(root, tag):
    """Coordinate POSIX workers sharing a result directory; the finished row table commits the panel."""
    import fcntl

    locks = root / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / (tag + ".lock")).open("a+") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield not (root / "rows" / (tag + ".csv.gz")).exists()
