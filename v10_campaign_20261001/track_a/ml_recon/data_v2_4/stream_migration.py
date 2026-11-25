"""Run and consume isolated, digest-only V2.4 stream migration verification."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from .stream_migration_contract import (
    GLOBAL_SEED, REFERENCE_COMMIT, REFERENCE_TREE, WORKER_PATHS,
    TRAIN_ROWS, VALIDATION_ROWS, THREAD_COUNT,
    build_migration_evidence, canonical_bytes, load_projection,
    validate_migration_evidence, digest, require_digest,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECTION_RELATIVE_PATH = "tests/fixtures/v24_historical_ready_projection.json"
GIT_TIMEOUT_SECONDS = 15
WORKER_TIMEOUT_SECONDS = 3600
RUNTIME_FIELDS = ("python_version", "numpy_version", "scipy_version", "torch_version")
_LIVE_PROOF_SEAL = object()


@dataclass(frozen=True, init=False, slots=True)
class Verified_stream_migration:
    """Process-local result of actual workers, never a deserialized authority.

    This API capability rejects caller-authored JSON and accidental fabrication;
    it is not a sandbox against arbitrary Python code modifying this module.
    """

    evidence_sha256: str
    source: tuple
    runtime: tuple
    issuer_pid: int
    rows_checked: int
    seal: object

    def __init__(self, report: Mapping[str, Any], *, seal: object = None):
        if seal is not _LIVE_PROOF_SEAL:
            raise ValueError("live migration proof can only be issued by actual verification")
        object.__setattr__(self, "evidence_sha256", digest(report))
        object.__setattr__(self, "source", tuple(sorted(report["candidate"]["source"].items())))
        object.__setattr__(self, "runtime", tuple(sorted(report["candidate"]["runtime"].items())))
        object.__setattr__(self, "issuer_pid", os.getpid())
        object.__setattr__(self, "rows_checked", len(report["candidate"]["rows"]))
        object.__setattr__(self, "seal", seal)

    def __reduce_ex__(self, protocol):
        raise TypeError("live migration proof cannot be serialized or transferred")


def require_live_migration(
    proof: object, expected_sha256: str, *, source_commit: str, source_tree: str,
    runtime_identity: Mapping[str, Any],
) -> None:
    if (type(proof) is not Verified_stream_migration
            or getattr(proof, "seal", None) is not _LIVE_PROOF_SEAL
            or proof.issuer_pid != os.getpid()):
        raise ValueError("V2 pilot requires live migration verification, not a supplied digest list")
    expected_source = tuple(sorted({"commit": source_commit, "tree": source_tree}.items()))
    expected_runtime = tuple(sorted((key, runtime_identity.get(key)) for key in RUNTIME_FIELDS))
    if (proof.evidence_sha256 != expected_sha256 or proof.source != expected_source
            or proof.runtime != expected_runtime):
        raise ValueError("live migration proof does not match this source/runtime/evidence")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "--no-optional-locks", *args), cwd=root,
                                   text=True, timeout=GIT_TIMEOUT_SECONDS).strip()


def source_binding(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    source = {"commit": _git(root, "rev-parse", "HEAD"),
              "tree": _git(root, "show", "-s", "--format=%T", "HEAD")}
    blobs = {}
    for relative in WORKER_PATHS:
        blob = _git(root, "rev-parse", f"HEAD:{relative}")
        if blob != _git(root, "hash-object", "--", relative):
            raise ValueError("migration verifier differs from its committed source")
        blobs[relative] = blob
    return source, blobs


def runtime_fingerprint() -> dict[str, str]:
    import numpy
    import scipy
    import torch

    return {"python_version": platform.python_version(), "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__, "torch_version": str(torch.__version__)}


def fixed_migration_recipe_binding() -> dict[str, str]:
    """Derive the admitted recipe from fixed generation scope, not receipt claims."""
    from .rung_plan import plan_registered_rung
    from .streaming import classifier_stream_datasets

    plan = plan_registered_rung("small_n_classifier", TRAIN_ROWS, VALIDATION_ROWS)
    _, _, recipe = classifier_stream_datasets(plan, global_seed=GLOBAL_SEED)
    return {"rung_plan_sha256": plan.sha256(),
            "recipe_payload_sha256": recipe["recipe_sha256"],
            "recipe_file_sha256": digest(recipe)}


def validate_current_migration(
    path: str | Path, expected_sha256: str, *, source_commit: str, source_tree: str,
    runtime_identity: Mapping[str, Any], rung_plan_sha256: str,
    recipe_payload_sha256: str, recipe_file_sha256: str,
) -> dict[str, Any]:
    """Check complete migration evidence against the imported consumer source."""
    source, blobs = source_binding(SOURCE_ROOT)
    if source != {"commit": source_commit, "tree": source_tree}:
        raise ValueError("migration consumer source differs from pilot identity")
    expected_recipe = fixed_migration_recipe_binding()
    supplied_recipe = dict(rung_plan_sha256=rung_plan_sha256,
                           recipe_payload_sha256=recipe_payload_sha256,
                           recipe_file_sha256=recipe_file_sha256)
    if supplied_recipe != expected_recipe:
        raise ValueError("migration requires the fixed migration recipe (plan and seed)")
    return validate_migration_evidence(
        path, expected_sha256, candidate_source=source, worker_blobs=blobs,
        runtime={key: runtime_identity.get(key) for key in RUNTIME_FIELDS},
        projection_path=SOURCE_ROOT / PROJECTION_RELATIVE_PATH,
        **expected_recipe)


def _execute_source_migration(reference_root: str | Path) -> dict[str, Any]:
    """Verify both processes before issuing any receipt or live capability."""
    reference_root = Path(reference_root).resolve()
    if _git(SOURCE_ROOT, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("migration verification requires a clean isolated candidate checkout")
    if (_git(reference_root, "rev-parse", "HEAD") != REFERENCE_COMMIT
            or _git(reference_root, "show", "-s", "--format=%T", "HEAD") != REFERENCE_TREE):
        raise ValueError("migration reference must be the frozen historical source")
    source, blobs = source_binding(SOURCE_ROOT)
    projection_path = SOURCE_ROOT / PROJECTION_RELATIVE_PATH
    projection = load_projection(projection_path)
    recipe_binding = fixed_migration_recipe_binding()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(THREAD_COUNT),
                       MKL_NUM_THREADS=str(THREAD_COUNT), OPENBLAS_NUM_THREADS=str(THREAD_COUNT))
    worker = SOURCE_ROOT / WORKER_PATHS[0]
    reports = {}
    for mode, root in (("reference", reference_root), ("stream", SOURCE_ROOT)):
        result = subprocess.run(
            (sys.executable, "-B", "-u", str(worker), "--source-root", str(root),
             "--projection", str(projection_path), "--mode", mode),
            cwd=root, env=environment, stdout=subprocess.PIPE, text=True,
            check=True, timeout=WORKER_TIMEOUT_SECONDS)
        reports[mode] = json.loads(result.stdout)
    report = build_migration_evidence(
        reports["reference"], reports["stream"], candidate_source=source, worker_blobs=blobs,
        runtime=runtime_fingerprint(), projection=projection, **recipe_binding)
    if source_binding(SOURCE_ROOT) != (source, blobs):
        raise ValueError("candidate source changed before migration publication")
    return report


def qualify_source_migration(reference_root: str | Path, expected_sha256: str) -> Verified_stream_migration:
    """Replay now and bind the live result to the separately supplied receipt."""
    require_digest(expected_sha256, "expected migration evidence")
    report = _execute_source_migration(reference_root)
    if digest(report) != expected_sha256:
        raise ValueError("live replay differs from the supplied migration receipt")
    return Verified_stream_migration(report, seal=_LIVE_PROOF_SEAL)


def qualify_and_publish_source_migration(
    reference_root: str | Path, output_path: str | Path,
) -> Verified_stream_migration:
    """Publish verified metadata and retain its live proof for this caller.

    The caller still needs independently supplied pilot admission. Neither
    this function nor its receipt creates an optimizer or grants execution.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("migration evidence output already exists")
    report = _execute_source_migration(reference_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(report)
    with output_path.open("xb") as stream:
        stream.write(raw)
    return Verified_stream_migration(report, seal=_LIVE_PROOF_SEAL)


def run_source_migration(reference_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Preserve the metadata-only CLI/API without exposing a serialized proof."""
    proof = qualify_and_publish_source_migration(reference_root, output_path)
    return {"evidence_path": str(Path(output_path).resolve()), "evidence_sha256": proof.evidence_sha256,
            "source_equivalence_verified": True, "rows_checked": proof.rows_checked,
            "historical_corpus_byte_parity_verified": False, "optimizer_steps": 0}
