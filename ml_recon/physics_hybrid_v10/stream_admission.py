"""External, source-bound admission for the bounded Track A stream pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import torch

from ml_recon.data_v2_4.hashing import (
    canonical_json_bytes, read_canonical_json, sha256_file, sha256_json,
)


PILOT_ADMISSION_SCHEMA = "V10StreamPilotAdmissionV1"
FORMAL_BRANCHES = frozenset({"v10", "codex/v10_lowdisk_training_prep"})
REQUIRED_FALSE_FLAGS = frozenset({
    "scientific_gate_evaluated", "selection_evidence", "promotion_ready",
    "hardware_action_authorized", "ten_k_training_authorized",
})
SOURCE_TIMEOUT_SECONDS = 15


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
    identity: Mapping[str, Any], device: str, model_profile: str | None,
) -> dict[str, Any]:
    """Guard direct trainer callers before output or optimizer construction."""
    if not admission_path or not identity.get("stream_pilot_admission_sha"):
        raise ValueError("stream pilot core requires external admission path and SHA")
    fixed_config = {
        "run_mode": "pilot", "model_id": "C_N_FULL_V10", "track_id": "track_a",
        "epochs": 12, "batch_size": 16, "amp_initial_scale": 256.0,
        "sample_count": 1000, "global_sample_count": 1000,
        "max_train_batches": None, "max_validation_batches": None,
        "learning_rate": 3e-4, "weight_decay": 1e-5, "gradient_clip_norm": 1.0,
        "amp_enabled": True, "scheduler_kind": "cosine", "num_workers": 0,
        "early_stopping_patience": 10, "metric_name": "validation_loss",
        "metric_direction": "min", "training_authorized": False,
    }
    if any(config.get(key) != value for key, value in fixed_config.items()) or model_profile != "C-S":
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
        runtime_identity=stream_runtime_identity(device))
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
    parity_path = Path(admission["historical_parity_manifest_path"])
    if not parity_path.is_absolute():
        parity_path = Path(admission_path).resolve().parent / parity_path
    sources = (("admission", Path(admission_path), expected_sha256),
               ("historical_parity", parity_path, admission["historical_parity_manifest_sha256"]))
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


def validate_stream_pilot_admission(
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
) -> dict[str, Any]:
    """Validate an independent admission before model or optimizer creation."""
    payload = _read_external_admission(path, expected_sha256)
    if payload.get("schema") != PILOT_ADMISSION_SCHEMA:
        raise ValueError("stream pilot admission schema mismatch")
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
    parity_path = payload.get("historical_parity_manifest_path")
    parity_sha = payload.get("historical_parity_manifest_sha256")
    if not isinstance(parity_path, str) or not parity_path or not isinstance(parity_sha, str):
        raise ValueError("stream pilot admission requires historical parity binding")
    from ml_recon.data_v2_4.stream_historical_parity import validate_stream_historical_parity

    parity = validate_stream_historical_parity(
        (Path(path).resolve().parent / parity_path) if not Path(parity_path).is_absolute() else parity_path,
        parity_sha, recipe_payload_sha256=canonical_recipe_sha256,
        recipe_file_sha256=recipe_file_sha256, rung_plan_sha256=rung_plan_sha256)
    if parity.get("admission_ready") is not True:
        raise ValueError("stream pilot historical parity admission is not ready")
    return payload
