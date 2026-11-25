"""Fail-closed admission records for the R16 physical classifier program."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.control_coordinate_layout import (
    INTERLEAVED_CONTROL_LAYOUT_SHA,
)
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_file
from ml_recon.physics_hybrid_v10.evidence_snapshot import (
    DEFAULT_JSON_BYTES,
    load_json_snapshot,
    parse_json_object,
    read_snapshot,
    reject_json_constant as _reject_json_constant,
    strict_object_pairs as _strict_object_pairs,
)
from ml_recon.physics_hybrid_v10.evidence_file_io import validate_evidence_directory
from ml_recon.v10_recoverability_foundry.training.physical_gates import (
    require_digest,
    validate_physical_identity,
)
from .review_source_manifest import validate_review_source_manifest

PHYSICAL_NAMESPACE = "TRACK_A_PHYSICAL_V2"
PHYSICAL_BRANCH = "codex/v10_tracka_physical_budget_v2"
ADMISSION_SCHEMA = "V10PhysicalProgramAdmissionV1"
PHYSICAL_MODE_BINDING_FIELDS = {
    "operation_input_recipe",
    "excitation_mode",
    "output_training_manifest_sha",
}
REVIEW_SCHEMA = "V10ChatProCodeReviewReceiptV2"
REVIEW_RESPONSE_FORMAT = "plain_rendered_text_with_final_json_marker"
CANONICAL_MARKER_EXTRACTOR_ID = "physical_admission.final_top_level_marker.v1"
REGISTERED_RENDER_TRAILER = (
    "\n来源\nConvert to PDF\nChatGPT 也可能会犯错。请核查重要信息。\n\nPro"
)
MAX_ADMISSION_OBJECT_BYTES = DEFAULT_JSON_BYTES


def _validate_e0_recipe(payload: Mapping[str, object], plan) -> None:
    if (
        payload.get("n_values") != plan.qualification_n_values
        or payload.get("requested_mode_shots")
        != plan.qualification_requested_mode_shots
        or payload.get("samples_per_n") != plan.qualification_samples_per_n
        or payload.get("noise_per_cell_success_probability")
        != plan.qualification_noise_per_cell_success_probability
        or payload.get("split_sizes") != plan.qualification_split_sizes
    ):
        raise ValueError(
            "E0 recipe must bind the exact N=3/4 C-TEST budget, pool and split contract"
        )


def physical_runtime_identity(device: str) -> dict[str, str]:
    """Read the actual numerical runtime without installing or changing it."""
    import numpy as np
    import scipy
    import torch

    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("physical qualification supports only explicit CPU or CUDA")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("the admitted CUDA device is not available")
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "device": str(selected),
        "device_name": torch.cuda.get_device_name(selected)
        if selected.type == "cuda"
        else "CPU",
    }


def _load_object(
    path: str | Path,
    name: str,
    *,
    expected_sha: str | None = None,
    maximum_bytes: int = MAX_ADMISSION_OBJECT_BYTES,
) -> tuple[Path, dict[str, Any], str]:
    """Preserve the existing tuple API over a single bound evidence read."""
    snapshot, value = load_json_snapshot(
        path, name, expected_sha=expected_sha, maximum_bytes=maximum_bytes
    )
    return snapshot.path, value, snapshot.sha256


def _extract_last_review_marker(response_bytes: bytes) -> dict[str, Any]:
    """Extract one terminal top-level marker, never nested JSON examples."""
    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("full review response must be UTF-8 text") from exc
    response_text = response_text.rstrip()
    if response_text.endswith(REGISTERED_RENDER_TRAILER):
        response_text = response_text[: -len(REGISTERED_RENDER_TRAILER)].rstrip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_object_pairs, parse_constant=_reject_json_constant
    )
    candidates: list[dict[str, Any]] = []
    marker_keys = {"decision", "reviewed_source_manifest_sha", "blockers"}
    line_start = True
    for index, character in enumerate(response_text):
        if character != "{" or not line_start:
            line_start = character == "\n"
            continue
        try:
            candidate, end = decoder.raw_decode(response_text[index:])
        except json.JSONDecodeError:
            line_start = False
            continue
        remainder = response_text[index + end :].strip()
        if remainder in ("", "```"):
            if not isinstance(candidate, dict) or set(candidate) != marker_keys:
                raise ValueError(
                    "final review marker must be an exact top-level object"
                )
            candidates.append(candidate)
        line_start = False
    if len(candidates) != 1:
        raise ValueError(
            "full review response requires exactly one terminal top-level JSON marker"
        )
    return candidates[0]


def validate_review_receipt(
    path: str | Path,
    *,
    expected_source_manifest_sha: str,
    expected_receipt_sha: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Validate a separately captured ChatGPT web Pro review; never create one."""
    receipt_path, receipt, receipt_sha = _load_object(
        path, "review receipt", expected_sha=expected_receipt_sha
    )
    required = {
        "schema",
        "source",
        "implementation_complete",
        "review_url",
        "mode",
        "prompt_file_path",
        "full_response_path",
        "extracted_marker_path",
        "extraction_tool_path",
        "extraction_tool_identity",
        "prompt_file_sha",
        "full_response_sha",
        "extracted_marker_sha",
        "extraction_tool_sha",
        "response_format",
        "reviewed_source_manifest_sha",
        "decision",
        "blockers",
    }
    if set(receipt) != required or receipt["schema"] != REVIEW_SCHEMA:
        raise ValueError("review receipt has an exact schema")
    if (
        receipt["source"] != "chatgpt_web_chat"
        or receipt["implementation_complete"] is not True
        or not isinstance(receipt["review_url"], str)
        or not receipt["review_url"]
        or receipt["mode"] != "pro"
        or receipt["response_format"] != REVIEW_RESPONSE_FORMAT
        or receipt["decision"] != "PASS"
        or receipt["blockers"] != []
    ):
        raise ValueError(
            "a complete ChatGPT web Pro PASS review with no blockers is required"
        )
    from urllib.parse import urlparse

    parsed = urlparse(receipt["review_url"])
    if (
        parsed.scheme != "https"
        or parsed.netloc not in ("chatgpt.com", "www.chatgpt.com")
        or not (
            parsed.path.startswith("/c/")
            or "/c/" in parsed.path
            and parsed.path.startswith("/g/")
        )
    ):
        raise ValueError("review URL must be a ChatGPT conversation URL")
    for name in (
        "prompt_file_sha",
        "full_response_sha",
        "extracted_marker_sha",
        "extraction_tool_sha",
        "reviewed_source_manifest_sha",
    ):
        require_digest(receipt[name], name)
    if receipt["reviewed_source_manifest_sha"] != expected_source_manifest_sha:
        raise ValueError("review receipt was captured for a different source manifest")
    parent = receipt_path.parent
    review_snapshots = {}
    for path_key, sha_key, label in (
        ("prompt_file_path", "prompt_file_sha", "review prompt"),
        ("full_response_path", "full_response_sha", "full response"),
        ("extracted_marker_path", "extracted_marker_sha", "extracted marker"),
    ):
        candidate = _contained_file(parent, receipt[path_key], path_key)
        if candidate.parent != parent:
            raise ValueError(
                "review prompt, full response, marker, or extraction tool bytes are not bound to the receipt"
            )
        review_snapshots[path_key] = read_snapshot(
            candidate, label, expected_sha=receipt[sha_key]
        )
    extractor_path = Path(receipt["extraction_tool_path"]).resolve(strict=True)
    if (
        receipt["extraction_tool_identity"] != CANONICAL_MARKER_EXTRACTOR_ID
        or extractor_path != Path(__file__).resolve()
        or sha256_file(extractor_path) != receipt["extraction_tool_sha"]
    ):
        raise ValueError(
            "review receipt must bind the registered canonical marker extractor"
        )
    full_response = review_snapshots["full_response_path"]
    extracted_marker = review_snapshots["extracted_marker_path"]
    response = _extract_last_review_marker(full_response.raw_bytes)
    marker_file = parse_json_object(extracted_marker, "extracted review marker")
    if (
        not isinstance(response, dict)
        or not isinstance(marker_file, dict)
        or set(response) != {"decision", "reviewed_source_manifest_sha", "blockers"}
        or response != marker_file
        or response["decision"] != "PASS"
        or response["reviewed_source_manifest_sha"] != expected_source_manifest_sha
        or response["blockers"] != []
    ):
        raise ValueError(
            "review response does not contain an unambiguous final PASS marker"
        )
    return receipt_sha, {
        **{str(snapshot.path): snapshot.sha256 for snapshot in review_snapshots.values()},
        str(extractor_path): receipt["extraction_tool_sha"],
    }


def _validate_source_manifest(
    path: Path, admission: Mapping[str, Any]
) -> dict[str, str]:
    _, manifest, _ = _load_object(
        path,
        "source manifest",
        expected_sha=require_digest(admission.get("source_manifest_sha"), "source manifest SHA"),
    )
    root = Path(__file__).resolve().parents[3]
    if not root.is_dir():
        raise ValueError("source manifest repo root is invalid")

    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"source Git verification failed: {exc}") from exc

    if (
        Path(git("rev-parse", "--show-toplevel")).resolve() != root
        or git("branch", "--show-current") != admission["branch"]
        or git("rev-parse", "HEAD") != admission["source_commit"]
        or git("rev-parse", "HEAD^{tree}") != admission["source_tree_sha"]
        or git("status", "--porcelain")
    ):
        raise ValueError(
            "reviewed source checkout is not the current clean admitted checkout"
        )
    required_core = {
        "ml_recon/physics_hybrid_v10/evidence_snapshot.py",
        "ml_recon/physics_hybrid_v10/training/physical_admission.py",
        "ml_recon/physics_hybrid_v10/training/review_source_manifest.py",
        "ml_recon/physics_hybrid_v10/training/physical_program.py",
        "ml_recon/physics_hybrid_v10/training_runtime.py",
        "ml_recon/physics_hybrid_v10/training/physical_dataset.py",
        "ml_recon/v10_recoverability_foundry/evaluation/physical_contracts.py",
        "ml_recon/physics_hybrid_v10/classifier_n_full.py",
        "ml_recon/physics_hybrid_v10/train_classifier.py",
        "ml_recon/physics_hybrid_v10/control_coordinate_layout.py",
        "ml_recon/physics_hybrid_v10/belief_geometry.py",
        "ml_recon/physics_hybrid_v10/w_structure_head.py",
        "ml_recon/physics_hybrid_v10/physical_classifier.py",
        "ml_recon/v10_recoverability_foundry/training/physical_supervision.py",
        "ml_recon/physics_hybrid_v10/calibration_admission.py",
        "ml_recon/physics_hybrid_v10/csm_profile_contract.py",
        "ml_recon/physics_hybrid_v10/training/physical_feedback.py",
        "ml_recon/v10_recoverability_foundry/physics/aod_feedback_forward.py",
        "ml_recon/v10_recoverability_foundry/contracts/memory_event_reducer.py",
        "ml_recon/v10_recoverability_foundry/contracts/memory_queue.py",
        "ml_recon/v10_recoverability_foundry/physics/memory_queue.py",
        "ml_recon/v10_recoverability_foundry/physics/memory_transition.py",
        "ml_recon/v10_recoverability_foundry/evaluation/memory_fusion.py",
        "tests/test_v10_physical_supervision.py",
        "tests/test_v10_aod_feedback_forward.py",
        "tests/test_v10_physical_feedback_pipeline.py",
        "tests/test_v10_memory_queue.py",
        "tests/test_v10_memory_fusion_evaluation.py",
    }
    return validate_review_source_manifest(
        manifest,
        admission,
        root=root,
        sha256_file=sha256_file,
        required_core=required_core,
    )


def _contained_file(parent: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} requires a relative path")
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or ":" in relative:
        raise ValueError(f"{label} path escapes its root")
    candidate = parent / item
    if not candidate.exists():
        raise ValueError(f"{label} is not a contained file")
    return candidate


def _contained_directory(parent: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} requires a relative path")
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or ":" in relative:
        raise ValueError(f"{label} path escapes its root")
    return validate_evidence_directory(parent / item, label)


def _validate_p0_report_source_bindings(
    report: Mapping[str, Any], source_files: Mapping[str, str]
) -> None:
    """Cross-check P0 semantic and test bytes against the reviewed source manifest."""
    bindings = report.get("semantic_bindings", {})
    bound_files = dict(bindings.get("source_file_sha256", {}))
    bound_files.update(report.get("test_file_sha256", {}))
    if not bound_files:
        raise ValueError("P0 report lacks source-bound semantic and test files")
    repo_root = Path(__file__).resolve().parents[3]
    for relative, digest in bound_files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("P0 report source path escapes the admitted checkout")
        source_path = (repo_root / relative).resolve(strict=True)
        if source_files.get(str(source_path)) != require_digest(
            digest, f"P0 report source {relative} SHA"
        ):
            raise ValueError(
                "P0 semantic/test source binding differs from the reviewed manifest"
            )


def _validate_file_backed_gates(
    admission: Mapping[str, Any], parent: Path, source_files: Mapping[str, str]
) -> dict[str, str]:
    from ml_recon.v10_recoverability_foundry.training.physical_gates import (
        MAX_EVIDENCE_BYTES,
        Physical_gate_ledger,
    )

    required_gates = ("P0",) if admission["stage"] == "E0" else ("P0", "E0", "E1", "E2")
    bindings = admission["gate_artifact_files"]
    if not isinstance(bindings, dict) or set(bindings) != set(required_gates):
        raise ValueError(
            "admission requires exact file-backed artifacts for every required Gate"
        )
    evidence_root = _contained_directory(
        parent, admission["evidence_root"], "evidence root"
    )
    ledger = Physical_gate_ledger()
    nested_files: dict[str, str] = {}
    for gate_id in required_gates:
        binding = bindings[gate_id]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("gate artifact bindings require path and SHA")
        artifact = _contained_file(parent, binding["path"], f"{gate_id} artifact")
        artifact_sha = require_digest(binding["sha256"], f"{gate_id} artifact SHA")
        _, artifact_payload, _ = _load_object(
            artifact, f"{gate_id} artifact", expected_sha=artifact_sha,
            maximum_bytes=MAX_EVIDENCE_BYTES,
        )
        ledger.record_external_artifact(
            artifact,
            gate_id=gate_id,
            expected_source_commit=admission["source_commit"],
            expected_source_tree_sha=admission["source_tree_sha"],
            expected_config_sha=admission["config_sha"],
            expected_physical_identity=admission["physical_identity"],
            evidence_root=evidence_root,
            expected_artifact_file_sha=artifact_sha,
        )
        roles = artifact_payload["evidence"]["evidence_file_roles"]
        for role, role_binding in roles.items():
            role_path = _contained_file(
                evidence_root, role_binding["path"], f"{gate_id} evidence role {role}"
            )
            nested_files[str(role_path)] = require_digest(
                role_binding["sha256"], f"{gate_id} evidence role {role} SHA"
            )
        if gate_id == "P0":
            report_path = _contained_file(
                evidence_root, roles["report"]["path"], "P0 evidence report"
            )
            _, report, _ = _load_object(
                report_path, "P0 evidence report", expected_sha=roles["report"]["sha256"],
                maximum_bytes=MAX_EVIDENCE_BYTES,
            )
            _validate_p0_report_source_bindings(report, source_files)
    reconstructed = ledger.report()
    _, supplied, _ = _load_object(
        _contained_file(parent, admission["gate_ledger_path"], "gate ledger"),
        "gate ledger", expected_sha=admission["gate_ledger_file_sha"],
    )
    if (
        not isinstance(supplied, dict)
        or supplied.get("ledger_sha") != admission["gate_ledger_sha"]
        or supplied.get("ledger_sha") != reconstructed["ledger_sha"]
    ):
        raise ValueError(
            "file-backed reconstructed physical ledger differs from the supplied ledger"
        )
    return nested_files


def validate_physical_admission(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Validate all external bindings before a trainer can allocate output or optimizer state."""
    admission_path, admission, admission_sha = _load_object(path, "physical admission")
    required = {
        "schema",
        "namespace",
        "branch",
        "source_commit",
        "source_tree_sha",
        "config_sha",
        "run_id",
        "device",
        "coordinate_layout_sha",
        "physical_identity",
        "source_manifest_path",
        "source_manifest_sha",
        "review_receipt_path",
        "review_receipt_sha",
        "gate_ledger_path",
        "gate_ledger_sha",
        "gate_ledger_file_sha",
        "plan_path",
        "plan_sha",
        "recipe_path",
        "recipe_sha",
        "recipe_file_sha",
        "stage",
    }
    required.update({"evidence_root", "gate_artifact_files", "runtime_identity"})
    supplied_mode_fields = set(admission) & PHYSICAL_MODE_BINDING_FIELDS
    if (
        set(admission) not in (required, required | PHYSICAL_MODE_BINDING_FIELDS)
        or admission["schema"] != ADMISSION_SCHEMA
    ):
        raise ValueError("physical admission has an exact schema")
    if (
        admission["namespace"] != PHYSICAL_NAMESPACE
        or admission["branch"] != PHYSICAL_BRANCH
    ):
        raise ValueError("physical admission namespace or branch is invalid")
    if admission["runtime_identity"] != physical_runtime_identity(device):
        raise ValueError(
            "actual numerical runtime differs from the independently admitted runtime"
        )
    for name in (
        "source_manifest_sha",
        "review_receipt_sha",
        "gate_ledger_sha",
        "gate_ledger_file_sha",
        "plan_sha",
        "recipe_sha",
        "recipe_file_sha",
        "config_sha",
    ):
        require_digest(admission[name], name)
    for name in ("source_commit", "source_tree_sha"):
        require_digest(admission[name], name, git=True)
    expected = {
        name: identity.get(name)
        for name in ("branch", "source_commit", "source_tree_sha", "config_sha")
    }
    if any(admission[name] != expected[name] for name in expected):
        raise ValueError("physical admission differs from the runtime identity")
    if sha256_of(dict(config)) != admission["config_sha"]:
        raise ValueError("admission config SHA is not the actual runtime config")
    if admission["run_id"] != config.get("run_id") or admission["device"] != str(
        device
    ):
        raise ValueError("physical admission differs from the requested run or device")
    if (
        admission["coordinate_layout_sha"] != INTERLEAVED_CONTROL_LAYOUT_SHA
        or identity.get("coordinate_layout_sha") != INTERLEAVED_CONTROL_LAYOUT_SHA
        or admission["coordinate_layout_sha"] != identity.get("coordinate_layout_sha")
    ):
        raise ValueError(
            "physical admission coordinate layout SHA differs from runtime identity"
        )
    if validate_physical_identity(
        admission["physical_identity"]
    ) != validate_physical_identity(identity.get("physical_identity")):
        raise ValueError(
            "physical admission identity differs from the runtime identity"
        )
    if supplied_mode_fields:
        if (
            admission["operation_input_recipe"]
            != identity.get("operation_input_recipe")
            or admission["excitation_mode"] != identity.get("excitation_mode")
            or admission["output_training_manifest_sha"]
            != identity.get("output_training_manifest_sha")
        ):
            raise ValueError("physical admission mode binding differs from runtime identity")
    elif (
        identity.get("operation_input_recipe", "legacy_v1") != "legacy_v1"
        or identity.get("excitation_mode", "legacy") != "legacy"
        or identity.get("output_training_manifest_sha")
    ):
        raise ValueError("nonlegacy physical admission requires an explicit mode binding")
    source_path = _contained_file(
        admission_path.parent, admission["source_manifest_path"], "source manifest"
    )
    source_files = _validate_source_manifest(source_path, admission)
    receipt_path = _contained_file(
        admission_path.parent, admission["review_receipt_path"], "review receipt"
    )
    review_receipt_sha, review_files = validate_review_receipt(
        receipt_path, expected_source_manifest_sha=admission["source_manifest_sha"],
        expected_receipt_sha=admission["review_receipt_sha"],
    )
    if review_receipt_sha != admission["review_receipt_sha"]:
        raise ValueError("review receipt checksum mismatch")
    for field, label in (
        ("gate_ledger_path", "gate ledger"),
        ("plan_path", "plan"),
        ("recipe_path", "recipe"),
    ):
        _contained_file(admission_path.parent, admission[field], label)
    gate_evidence_files = _validate_file_backed_gates(
        admission, admission_path.parent, source_files
    )
    from ml_recon.v10_recoverability_foundry.training.physical_plan import (
        Physical_training_plan,
    )

    _, plan_payload, _ = _load_object(
        _contained_file(admission_path.parent, admission["plan_path"], "plan"),
        "plan", expected_sha=admission["plan_sha"],
    )
    plan = Physical_training_plan.from_mapping(plan_payload)
    for field_name in (
        "learning_rate",
        "weight_decay",
        "amp_enabled",
        "amp_initial_scale",
        "telemetry_interval_seconds",
    ):
        if config.get(field_name) != getattr(plan, field_name):
            raise ValueError(
                f"runtime {field_name} differs from the reviewed training plan"
            )
    if config.get("training_authorized") is not False or config.get("num_workers") != 0:
        raise ValueError(
            "physical admission cannot self-authorize scale or unreviewed worker processes"
        )
    if admission["stage"] not in ("E0", "E3"):
        raise ValueError("only registered E0 or E3 physical stages may be admitted")
    if admission["stage"] == "E0" and (
        config.get("sample_count") != plan.qualification_rows
        or config.get("max_train_batches") != plan.qualification_optimizer_steps
        or config.get("epochs") != 1
        or config.get("batch_size") != 4
        or config.get("global_seed") not in plan.development_seeds
        or tuple(config.get("seed_panel", ())) != (config.get("global_seed"),)
        or config.get("run_mode") != "pilot"
    ):
        raise ValueError(
            "E0 admission requires the bounded eight-row two-step qualification config"
        )
    if admission["stage"] == "E3" and (
        config.get("sample_count") != plan.anti_collapse_rows
        or config.get("epochs") != plan.epochs
        or tuple(config.get("seed_panel", ())) != plan.development_seeds
        or config.get("global_seed") not in plan.development_seeds
    ):
        raise ValueError("E3 admission requires the registered 2k three-seed panel")
    recipe_path = _contained_file(
        admission_path.parent, admission["recipe_path"], "recipe"
    )
    recipe_snapshot, recipe = load_json_snapshot(
        recipe_path, "recipe", expected_sha=admission["recipe_file_sha"]
    )
    if recipe_snapshot.raw_bytes != canonical_json_bytes(recipe):
        raise ValueError(f"JSON is not canonical: {recipe_path}")
    from .physical_dataset import Physical_dataset_recipe

    operation_input_recipe = (
        recipe.get("operation_input_recipe", "legacy_v1")
        if isinstance(recipe, dict) else ""
    )
    recipe_template = Physical_dataset_recipe(
        operation_input_recipe=operation_input_recipe
    )
    if not isinstance(recipe, dict) or set(recipe) != {
        *recipe_template.to_dict(), "recipe_sha256",
    }:
        raise ValueError("recipe payload SHA mismatch")

    payload = {
        name: value for name, value in recipe.items()
        if name not in ("recipe_sha256", "schema")
    }
    payload["n_values"] = tuple(payload["n_values"])
    if payload["split_sizes"] is not None:
        payload["split_sizes"] = tuple(payload["split_sizes"])
    canonical_recipe_sha = Physical_dataset_recipe(**payload).sha256()
    if payload["global_seed"] != plan.data_seed:
        raise ValueError("dataset seed must remain fixed across the model seed panel")
    if (
        recipe["recipe_sha256"] != canonical_recipe_sha
        or admission["recipe_sha"] != canonical_recipe_sha
    ):
        raise ValueError("recipe SHA is not its canonical recipe identity")
    if (
        identity.get("stream_recipe_sha") != admission["recipe_sha"]
        or identity.get("stream_recipe_file_sha") != admission["recipe_file_sha"]
    ):
        raise ValueError("runtime stream recipe identity differs from admission")
    operation_recipe = payload.get("operation_input_recipe", "legacy_v1")
    expected_contract = {
        "legacy_v1": ("legacy", "V10PhysicalAodDatasetRecipeV3"),
        "operation_poisson_v1": (
            identity.get("excitation_mode"),
            "V10PhysicalOperationPoissonDatasetRecipeV4",
        ),
    }.get(operation_recipe)
    if (
        expected_contract is None
        or identity.get("operation_input_recipe", "legacy_v1") != operation_recipe
        or identity.get("excitation_mode", "legacy") not in (
            ("legacy",) if operation_recipe == "legacy_v1"
            else ("diagnostic", "coupled")
        )
        or (
            identity.get("dataset_contract_version") is not None
            and identity.get("dataset_contract_version") != expected_contract[1]
        )
    ):
        raise ValueError("physical admission recipe/mode/data-contract binding is invalid")
    from ml_recon.physics_hybrid_v10.published_output_registry import (
        physical_output_training_manifest_sha,
    )

    expected_manifest_sha = physical_output_training_manifest_sha(
        identity.get("excitation_mode", "legacy")
    )
    identity_manifest_sha = identity.get("output_training_manifest_sha") or ""
    if identity_manifest_sha not in ("", expected_manifest_sha):
        raise ValueError("runtime output manifest SHA differs from excitation mode")
    if operation_recipe != "legacy_v1" and (
        not supplied_mode_fields
        or identity_manifest_sha != expected_manifest_sha
    ):
        raise ValueError("operation-Poisson admission requires an explicit output manifest SHA")
    if admission["stage"] == "E0":
        _validate_e0_recipe(payload, plan)
    immutable_files = {
        str(admission_path): admission_sha,
        str(source_path): admission["source_manifest_sha"],
        str(receipt_path): admission["review_receipt_sha"],
        str(
            _contained_file(
                admission_path.parent, admission["gate_ledger_path"], "gate ledger"
            )
        ): admission["gate_ledger_file_sha"],
        str(
            _contained_file(admission_path.parent, admission["plan_path"], "plan")
        ): admission["plan_sha"],
        str(recipe_path): admission["recipe_file_sha"],
        **source_files,
        **review_files,
        **gate_evidence_files,
    }
    for binding in admission["gate_artifact_files"].values():
        immutable_files[
            str(
                _contained_file(admission_path.parent, binding["path"], "gate artifact")
            )
        ] = binding["sha256"]
    return {
        "admission_path": str(admission_path),
        "admission_sha": admission_sha,
        "review_receipt_sha": admission["review_receipt_sha"],
        "stage": admission["stage"],
        "recipe_sha": admission["recipe_sha"],
        "recipe_file_sha": admission["recipe_file_sha"],
        "immutable_files": immutable_files,
    }
