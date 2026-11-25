"""External, source-bound admission for the bounded Track A stream pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ml_recon.data_v2_4.hashing import (
    canonical_json_bytes, read_canonical_json, sha256_file, sha256_json,
)
from ml_recon.physics_hybrid_v10.loss_profiles import resolve_loss_weights


PILOT_ADMISSION_SCHEMA = "V10StreamPilotAdmissionV1"
PILOT_MIGRATION_ADMISSION_SCHEMA = "V10StreamPilotAdmissionV2"
FORMAL_BRANCHES = frozenset({"v10", "codex/v10_lowdisk_training_prep"})
REQUIRED_FALSE_FLAGS = frozenset({
    "scientific_gate_evaluated", "selection_evidence", "promotion_ready",
    "hardware_action_authorized", "ten_k_training_authorized",
})
SOURCE_TIMEOUT_SECONDS = 15
STREAM_WORKER_COUNTS = (0, 3)
STREAM_WORKER_TIMEOUT_SECONDS = 300
TRACK_A_PILOT_SEED = 20261001


def assert_stream_source_identity(expected_source: Mapping[str, str]) -> None:
    """Fail closed when a stream worker or publisher sees source drift."""
    expected = {
        key: expected_source.get(key)
        for key in ("branch", "source_commit", "source_tree_sha")
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise ValueError("stream source identity guard requires a complete source identity")
    actual = _current_stream_source()
    if actual != expected:
        raise ValueError("stream source identity changed after admission")


@dataclass(frozen=True)
class StreamSourceIdentityGuard:
    """Picklable DataLoader worker hook that rechecks the admitted checkout."""

    expected_source: dict[str, str]

    def __call__(self, worker_id: int) -> None:
        del worker_id
        assert_stream_source_identity(self.expected_source)


def validate_stream_worker_count(num_workers: object) -> int:
    """Admit only the serial or measured three-process stream profile."""
    if type(num_workers) is not int or num_workers not in STREAM_WORKER_COUNTS:
        raise ValueError("stream num_workers must be an integer in the registered profiles (0, 3)")
    return num_workers


def stream_runtime_identity(device: str) -> dict[str, object]:
    """Fingerprint observed runtime facts, without claiming qualification tests."""
    import numpy
    import scipy

    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("stream training supports cpu or a specific cuda ordinal")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA stream runtime is unavailable")
    ordinal = (torch.cuda.current_device() if resolved.type == "cuda" and resolved.index is None
               else resolved.index)
    identity = {
        "requested_device": device,
        "resolved_device": f"cuda:{ordinal}" if resolved.type == "cuda" else "cpu",
        "device_ordinal": ordinal, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
        "machine": platform.node(), "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "cuda_device_name": torch.cuda.get_device_name(ordinal) if resolved.type == "cuda" else None,
    }
    identity["runtime_identity_sha256"] = sha256_json(identity)
    return identity


def _current_stream_source() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=root, text=True,
            timeout=SOURCE_TIMEOUT_SECONDS).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("stream pilot core requires a clean reviewed source checkout")
    return {"branch": git("branch", "--show-current"),
            "source_commit": git("rev-parse", "HEAD"),
            "source_tree_sha": git("show", "-s", "--format=%T", "HEAD")}


def validate_core_stream_pilot(
    *, admission_path: str | Path | None, config: Mapping[str, Any],
    identity: Mapping[str, Any], device: str, model: Any,
    model_profile: str | None = None,
    migration_proof: object = None,
) -> dict[str, Any]:
    """Guard direct trainer callers before output or optimizer construction."""
    from .training_identity import config_identity_sha
    if identity.get("config_sha") != config_identity_sha(config):
        raise ValueError("stream pilot core config_sha does not match the actual V10 training config")
    if not admission_path or not identity.get("stream_pilot_admission_sha"):
        raise ValueError("stream pilot core requires external admission path and SHA")
    try:
        num_workers = validate_stream_worker_count(config.get("num_workers"))
    except ValueError as exc:
        raise ValueError("stream pilot core configuration differs from the bounded C-S contract (worker profile)") from exc
    fixed_config = {
        "run_mode": "pilot", "model_id": "C_N_FULL_V10", "track_id": "track_a",
        "model_profile": "C-S",
        "epochs": 12, "batch_size": 16, "amp_initial_scale": 256.0,
        "sample_count": 1000, "global_sample_count": 1000,
        "max_train_batches": None, "max_validation_batches": None,
        "learning_rate": 3e-4, "weight_decay": 1e-5, "gradient_clip_norm": 1.0,
        "amp_enabled": True, "scheduler_kind": "cosine",
        "early_stopping_patience": 10, "metric_name": "validation_loss",
        "metric_direction": "min", "training_authorized": False,
        "global_seed": TRACK_A_PILOT_SEED,
        "seed_panel": (TRACK_A_PILOT_SEED,),
    }
    if any(config.get(key) != value for key, value in fixed_config.items()):
        raise ValueError("stream pilot core configuration differs from the bounded C-S contract")
    from .classifier_n_full import V10ClassifierNFull
    expected_model_spec = {
        "architecture_class": "V10ClassifierNFull",
        "profile": "C-S",
        "descriptor_dim": 15,
        "mode_feature_dim": 12,
        "certificate_dim": 10,
    }
    if type(model) is not V10ClassifierNFull:
        raise ValueError("stream pilot core requires the canonical V10ClassifierNFull model")
    actual_spec_fn = getattr(model, "checkpoint_build_spec", None)
    actual_spec = actual_spec_fn() if callable(actual_spec_fn) else None
    if actual_spec != expected_model_spec:
        raise ValueError("stream pilot core model build spec differs from canonical C-S")
    if model_profile is not None and model_profile != actual_spec["profile"]:
        raise ValueError("stream pilot core model profile does not match its build spec")
    if config.get("model_profile") != expected_model_spec["profile"]:
        raise ValueError("stream pilot core configuration differs from the bounded C-S contract")
    if not config.get("run_id") or tuple(config.get("seed_panel", ())) != (config.get("global_seed"),):
        raise ValueError("stream pilot core requires an explicit run and one bound seed")
    source = _current_stream_source()
    if any(identity.get(key) != value for key, value in source.items()):
        raise ValueError("stream pilot core source identity does not match the loaded checkout")
    from ml_recon.data_v2_4.rung_plan import plan_registered_rung
    from ml_recon.data_v2_4.streaming import classifier_stream_datasets
    plan = plan_registered_rung("small_n_classifier", 1000, 200)
    _, _, recipe = classifier_stream_datasets(plan, global_seed=config["global_seed"])
    expected_recipe = {
        "stream_recipe_sha": recipe["recipe_sha256"],
        "stream_recipe_file_sha": hashlib.sha256(canonical_json_bytes(recipe)).hexdigest(),
        "dataset_index_sha": recipe["recipe_sha256"],
        "split_hash": sha256_json(recipe["splits"]),
        "dataset_contract_sha": plan.contract_sha,
        "dataset_cell_id": ",".join(cell.cell_id for cell in plan.cell_allocations),
    }
    if config["rung_plan_sha"] != plan.sha256() or any(
            identity.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("stream pilot core recipe differs from its registered source/seed/plan")
    validate_stream_pilot_admission(
        admission_path, identity["stream_pilot_admission_sha"], **source,
        canonical_recipe_sha256=identity["stream_recipe_sha"],
        recipe_file_sha256=identity["stream_recipe_file_sha"],
        rung_plan_sha256=config["rung_plan_sha"], run_id=config["run_id"],
        runtime_identity=stream_runtime_identity(device), migration_proof=migration_proof,
        loss_weight_profile=config.get("loss_weight_profile", "default"),
        num_workers=num_workers)
    return recipe


def validate_stream_pilot_datasets(train_dataset, validation_dataset, recipe: Mapping[str, Any]) -> None:
    """Match the actual virtual ranges to the admitted recipe before fitting."""
    from ml_recon.data_v2_4.multi_cell import V24MultiCellRangeDataset
    from ml_recon.data_v2_4.streaming import V24_stream_dataset

    for split, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        specs = recipe["splits"][split]
        if not isinstance(dataset, V24MultiCellRangeDataset) or tuple(dataset.cell_ids) != tuple(
                spec["cell"]["cell_id"] for spec in specs) or len(dataset.datasets) != len(specs):
            raise ValueError("stream pilot dataset cell membership differs from its admission")
        for member, expected in zip(dataset.datasets, specs, strict=True):
            if (not isinstance(member, V24_stream_dataset) or not member.include_truth
                    or member.spec.recipe() != expected):
                raise ValueError("stream pilot dataset range or renderer differs from its admission")


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_external_admission(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    admission_path = Path(path)
    _require_digest(expected_sha256, "expected_admission_sha256")
    if not admission_path.is_file():
        raise ValueError("stream pilot requires an external admission file")
    content = admission_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("stream pilot admission byte SHA mismatch")
    try:
        payload = json.loads(content)
        if canonical_json_bytes(payload) != content:
            raise ValueError("admission JSON is not canonical")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("stream pilot admission is malformed or noncanonical") from exc
    if not isinstance(payload, dict):
        raise ValueError("stream pilot admission must be a JSON object")
    return payload


def preserve_stream_pilot_evidence(
    admission_path: str | Path, expected_sha256: str, output_dir: Path,
) -> dict[str, dict[str, str]]:
    """Retain immutable, content-addressed metadata copies; never raw samples."""
    admission = _read_external_admission(admission_path, expected_sha256)
    evidence_kind, path_field, sha_field = _data_evidence_fields(admission)
    parity_path = Path(admission[path_field])
    if not parity_path.is_absolute():
        parity_path = Path(admission_path).resolve().parent / parity_path
    sources = (("admission", Path(admission_path), expected_sha256),
               (evidence_kind, parity_path, admission[sha_field]))
    records = {}
    for kind, source, digest in sources:
        _require_digest(digest, kind + "_sha256")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("stream pilot evidence changed before preservation")
        destination = output_dir / "admission_evidence" / (kind + "_" + digest + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError("immutable stream pilot evidence was modified")
        else:
            with destination.open("xb") as stream:
                stream.write(content)
        records[kind] = {"path": destination.relative_to(output_dir).as_posix(), "sha256": digest}
    return records


def _data_evidence_fields(admission: Mapping[str, Any]) -> tuple[str, str, str]:
    if admission.get("schema") == PILOT_ADMISSION_SCHEMA:
        return "historical_parity", "historical_parity_manifest_path", "historical_parity_manifest_sha256"
    if (admission.get("schema") == PILOT_MIGRATION_ADMISSION_SCHEMA
            and admission.get("data_evidence_kind") == "frozen_source_stream_equivalence"):
        return "source_migration", "migration_evidence_path", "migration_evidence_sha256"
    raise ValueError("stream pilot admission schema or explicit data evidence kind mismatch")


def read_bound_stream_pilot_metadata(
    path: str | Path,
    expected_sha256: str,
    *,
    branch: str,
    source_commit: str,
    source_tree_sha: str,
    canonical_recipe_sha256: str,
    recipe_file_sha256: str,
    rung_plan_sha256: str,
    run_id: str,
    runtime_identity: Mapping[str, object],
    loss_weight_profile: str = "default",
    num_workers: int = 0,
) -> dict[str, Any]:
    """Reject invalid metadata before costly qualification; grant no admission."""
    payload = _read_external_admission(path, expected_sha256)
    resolve_loss_weights(loss_weight_profile)
    payload_profile = (payload["loss_weight_profile"]
                       if "loss_weight_profile" in payload else "default")
    resolve_loss_weights(payload_profile)
    if payload_profile != loss_weight_profile:
        raise ValueError("stream pilot admission loss-weight profile mismatch")
    evidence_kind, path_field, sha_field = _data_evidence_fields(payload)
    workers = validate_stream_worker_count(num_workers)
    declared_workers = validate_stream_worker_count(payload.get("num_workers", 0))
    if declared_workers != workers or (evidence_kind != "source_migration" and workers != 0):
        raise ValueError("stream worker profile differs from its versioned external admission")
    expected = {
        "allowed_branch": branch,
        "source_commit": source_commit,
        "source_tree_sha": source_tree_sha,
        "canonical_recipe_sha256": canonical_recipe_sha256,
        "stream_recipe_file_sha256": recipe_file_sha256,
        "rung_plan_sha256": rung_plan_sha256,
        "run_id": run_id,
        "train_rows": 1_000,
        "validation_rows": 200,
        "epochs": 12,
        "batch_size": 16,
        "amp_initial_scale": 256.0,
        "global_seed": TRACK_A_PILOT_SEED,
        "runtime_identity": dict(runtime_identity),
    }
    if branch not in FORMAL_BRANCHES:
        raise ValueError("stream pilot branch is not formally allowed")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"stream pilot admission mismatch: {key}")
    flags = payload.get("authority")
    if not isinstance(flags, dict) or any(flags.get(key) is not False for key in REQUIRED_FALSE_FLAGS):
        raise ValueError("stream pilot admission must retain all authority flags as false")
    parity_path = payload.get(path_field)
    parity_sha = payload.get(sha_field)
    if not isinstance(parity_path, str) or not parity_path or not isinstance(parity_sha, str):
        raise ValueError("stream pilot admission requires data evidence path and SHA binding")
    _require_digest(parity_sha, "data evidence SHA")
    return payload


def validate_stream_pilot_admission(
    path: str | Path, expected_sha256: str, *, branch: str, source_commit: str,
    source_tree_sha: str, canonical_recipe_sha256: str, recipe_file_sha256: str,
    rung_plan_sha256: str, run_id: str, runtime_identity: Mapping[str, object],
    migration_proof: object = None,
    loss_weight_profile: str = "default",
    num_workers: int = 0,
) -> dict[str, Any]:
    """Require evidence verification in addition to bound metadata."""
    payload = read_bound_stream_pilot_metadata(
        path, expected_sha256, branch=branch, source_commit=source_commit,
        source_tree_sha=source_tree_sha, canonical_recipe_sha256=canonical_recipe_sha256,
        recipe_file_sha256=recipe_file_sha256, rung_plan_sha256=rung_plan_sha256,
        run_id=run_id, runtime_identity=runtime_identity,
        loss_weight_profile=loss_weight_profile, num_workers=num_workers)
    evidence_kind, path_field, sha_field = _data_evidence_fields(payload)
    parity_path, parity_sha = payload[path_field], payload[sha_field]
    evidence_path = ((Path(path).resolve().parent / parity_path)
                     if not Path(parity_path).is_absolute() else Path(parity_path))
    if evidence_kind == "source_migration":
        from ml_recon.data_v2_4.stream_migration import validate_current_migration, require_live_migration
        require_live_migration(
            migration_proof, parity_sha, source_commit=source_commit, source_tree=source_tree_sha,
            runtime_identity=runtime_identity)
        validate_current_migration(
            evidence_path, parity_sha, source_commit=source_commit, source_tree=source_tree_sha,
            runtime_identity=runtime_identity, recipe_payload_sha256=canonical_recipe_sha256,
            recipe_file_sha256=recipe_file_sha256, rung_plan_sha256=rung_plan_sha256)
        return payload
    from ml_recon.data_v2_4.stream_historical_parity import validate_stream_historical_parity

    parity = validate_stream_historical_parity(
        evidence_path,
        parity_sha, recipe_payload_sha256=canonical_recipe_sha256,
        recipe_file_sha256=recipe_file_sha256, rung_plan_sha256=rung_plan_sha256)
    if parity.get("admission_ready") is not True:
        raise ValueError("stream pilot historical parity admission is not ready")
    return payload
