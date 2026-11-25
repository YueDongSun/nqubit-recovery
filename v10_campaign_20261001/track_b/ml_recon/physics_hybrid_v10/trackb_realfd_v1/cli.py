"""Source-bound Track B bounded pilots and read-only admission preflights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from itertools import islice
from typing import Any, Mapping

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .datasets import open_trackb_v24_real_static_fold, trackb_real_static_collate
from .observation_plate import (
    MaskedSetQueryObservationV1,
    TrainableMaskedObservationPlateConfigV1,
    TrainableMaskedObservationPlateV1,
    deterministic_query_mask,
)
from .losses import trackb_masked_real_static_objective
from .training import (
    MAX_SMOKE_OPTIMIZER_STEPS,
    TrackBRealRuntimeConfigV1,
    TrackBRealRuntimeIdentityV1,
    TrackBRealTrainingRuntimeV1,
    _move_to_device,
    run_short_smoke_training,
    seed_trackb_runtime,
    trackb_model_state_sha,
)


def _smoke_observation() -> MaskedSetQueryObservationV1:
    operation_ids = tuple(f"op_{index:02d}" for index in range(8))
    families = tuple("family_a" if index < 4 else "family_b" for index in range(8))
    cycles = tuple(0 if index % 2 == 0 else 1 for index in range(8))
    mask = deterministic_query_mask(
        sample_id="trackb_cli_smoke", epoch=0, view_seed=0, operation_ids=operation_ids,
        operation_families=families, cycle_indices=cycles,
    )
    return MaskedSetQueryObservationV1(
        sample_id="trackb_cli_smoke",
        four_cell_counts=tuple((5, 2, 1, 1) for _ in operation_ids),
        query_mask=tuple(bool(value) for value in mask),
        operation_ids=operation_ids,
        operation_families=families,
        cycle_indices=cycles,
        prepared_command_coefficients=(0.0, 0.0),
        global_multi=(0, 0, 0, 0),
        global_multi_scope="global",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-steps", type=int, default=0, help=f"bounded local smoke steps (0..{MAX_SMOKE_OPTIMIZER_STEPS})")
    parser.add_argument("--long-train", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-contract-smoke", action="store_true", help="run one no-training CUDA availability/contract probe")
    parser.add_argument("--real-pilot", action="store_true", help="run a source-bound V2.4 real-static pilot (at most 2,000 steps)")
    parser.add_argument("--real-preflight", action="store_true", help="validate the source-bound V2.4 real-static fold without optimization")
    parser.add_argument("--observation-model", choices=["legacy_v1", "effective_v2", "effective_calibration_v2", "effective_fixed_pair_v2", "effective_family_context_v3"], default="legacy_v1",
                        help="explicit model identity; V2 uses a separate checkpoint/receipt namespace")
    parser.add_argument("--dataset-root")
    parser.add_argument("--build-id")
    parser.add_argument("--system-type", choices=["3qubit", "4qubit"])
    parser.add_argument("--basis-config-root")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    parser.add_argument("--expected-operation-registry-sha")
    parser.add_argument("--expected-command-table-sha")
    parser.add_argument("--operation-metadata-sidecar")
    parser.add_argument("--expected-operation-metadata-sidecar-sha")
    parser.add_argument("--global-initialization")
    parser.add_argument("--expected-global-initialization-sha")
    parser.add_argument("--output-root")
    parser.add_argument("--pilot-steps", type=int, default=2)
    parser.add_argument("--pilot-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-sha")
    return parser


def _current_source_acceptance() -> tuple[str, str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("real Track B pilot requires a readable clean Git checkout") from error
    require_git_oid(commit, "checkout_commit")
    require_git_oid(tree, "checkout_tree")
    if status:
        raise SystemExit("real Track B pilot requires a clean checkout with no uncommitted files")
    acceptance_sha = canonical_sha256({
        "schema": "TrackBSourceAcceptanceV1", "commit": commit, "tree": tree, "clean": True,
    })
    return commit, tree, acceptance_sha


def _require_real_pilot_arguments(args: argparse.Namespace) -> tuple[str, str, str]:
    initialization_path = getattr(args, "global_initialization", None)
    initialization_sha = getattr(args, "expected_global_initialization_sha", None)
    if bool(initialization_path) != bool(initialization_sha):
        raise SystemExit("global initialization requires both file path and expected SHA")
    if initialization_path:
        if _pilot_variant(args) == "legacy_v1":
            raise SystemExit("global initialization is supported only by explicit effective V2 models")
        require_sha256(initialization_sha, "global_initialization_file_sha")
    required = (
        "dataset_root", "build_id", "system_type", "source_commit", "source_tree",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"real Track B pilot/preflight requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    require_git_oid(args.source_commit, "source_commit")
    require_git_oid(args.source_tree, "source_tree")
    if bool(args.basis_config_root) == bool(
        args.expected_operation_registry_sha or args.expected_command_table_sha
    ):
        raise SystemExit(
            "real Track B pilot requires authoritative basis_config_root or both expected registry/table SHAs"
        )
    if (args.expected_operation_registry_sha is None) != (args.expected_command_table_sha is None):
        raise SystemExit("expected registry/table SHA arguments must be supplied together")
    for name in ("expected_operation_registry_sha", "expected_command_table_sha"):
        value = getattr(args, name)
        if value is not None:
            require_sha256(value, name)
    if bool(args.operation_metadata_sidecar) != bool(args.expected_operation_metadata_sidecar_sha):
        raise SystemExit(
            "real Track B pilot requires both --operation-metadata-sidecar and "
            "--expected-operation-metadata-sidecar-sha when either is supplied"
        )
    if args.expected_operation_metadata_sidecar_sha is not None:
        require_sha256(
            args.expected_operation_metadata_sidecar_sha,
            "expected_operation_metadata_sidecar_sha",
        )
    checkout_commit, checkout_tree, acceptance_sha = _current_source_acceptance()
    if args.source_commit != checkout_commit or args.source_tree != checkout_tree:
        raise SystemExit("declared source commit/tree do not match the clean pilot checkout")
    return checkout_commit, checkout_tree, acceptance_sha


def _open_real_pilot_dataset(args: argparse.Namespace):
    _require_real_pilot_arguments(args)
    return open_trackb_v24_real_static_fold(
        args.dataset_root,
        build_id=args.build_id,
        system_type=args.system_type,
        basis_config_root=args.basis_config_root,
        expected_operation_registry_sha=args.expected_operation_registry_sha or "",
        expected_command_table_sha=args.expected_command_table_sha or "",
        operation_metadata_sidecar_path=args.operation_metadata_sidecar,
        expected_operation_metadata_sidecar_sha=(
            args.expected_operation_metadata_sidecar_sha or ""
        ),
        outer_role="train",
    )


def resolve_trackb_device(requested: str, *, cuda_available: bool) -> str:
    """Resolve ``auto`` explicitly; never pass an unresolved device name to Torch."""
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested not in ("cpu", "cuda"):
        raise ValueError("Track B device must be cpu, cuda, or auto")
    return requested


def _all_tensors_finite(value: Any, torch_module: Any) -> bool:
    if isinstance(value, torch_module.Tensor):
        return bool(torch_module.all(torch_module.isfinite(value)).item())
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item, torch_module) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensors_finite(item, torch_module) for item in value)
    return True


def _pilot_variant(args: argparse.Namespace) -> str:
    variant = getattr(args, "observation_model", "legacy_v1")
    if variant not in {"legacy_v1", "effective_v2", "effective_calibration_v2", "effective_fixed_pair_v2", "effective_family_context_v3"}:
        raise ValueError("unregistered Track B observation model")
    return variant


def _build_real_pilot_model(args: argparse.Namespace, first: Mapping[str, Any], session_count: int, *, global_initialization=None, operation_metadata_sha=None):
    """Construct only a registered model; keep optional Torch imports lazy."""
    common = {
        "system_type": args.system_type, "mode_count": 3 if args.system_type == "3qubit" else 4,
        "projector_dim": int(first["projector_descriptor"].shape[-1]),
        "prepared_command_dim": int(first["prepared_command"].shape[-1]),
    }
    variant = _pilot_variant(args)
    if bool(getattr(args, "global_initialization", None)) != (global_initialization is not None):
        raise ValueError("model construction requires the verified global initialization")
    if variant == "legacy_v1":
        return TrainableMaskedObservationPlateV1(TrainableMaskedObservationPlateConfigV1(
            **common, raw_rf_dim=int(first["raw_rf_descriptor"].shape[-1]), session_count=session_count,
        ))
    from .effective_observation_v2 import Effective_observation_config, Effective_singleton_observation
    from .measurement_context import Effective_measurement_context
    from .effective_family_context_v3 import MODEL_CONFIG_SCHEMA as FAMILY_CONFIG_SCHEMA

    context = None
    if variant == "effective_family_context_v3":
        context = Effective_measurement_context.from_operation_metadata(
            admission_manifest_sha=first["admission_manifest_sha"],
            operation_metadata_sha=operation_metadata_sha,
            operation_ids=first["operation_ids"], operation_families=first["operation_families"],
        )

    return Effective_singleton_observation(Effective_observation_config(
        **common, structure_enabled=variant != "effective_calibration_v2", read_trials_source="native_scalar",
        pair_residual_enabled=variant not in {"effective_fixed_pair_v2", "effective_family_context_v3"},
        global_initialization=global_initialization,
        measurement_context=context,
        schema=FAMILY_CONFIG_SCHEMA if context is not None else "TrackBEffectiveSingletonConfigV2",
    ))


def _verified_pilot_global_initialization(args, dataset, source_commit, source_tree):
    path = getattr(args, "global_initialization", None)
    if not path:
        return None
    from .global_count_initialization import fit_global_count_initialization, load_global_initialization_bytes

    initialization = load_global_initialization_bytes(
        Path(path).read_bytes(), expected_sha=args.expected_global_initialization_sha,
    )
    if (initialization.source_commit, initialization.source_tree) != (source_commit, source_tree):
        raise ValueError("initializer source must match the exact current pilot source")
    derived = fit_global_count_initialization(dataset, source_commit=source_commit, source_tree=source_tree)
    if initialization.canonical_bytes() != derived.canonical_bytes():
        raise ValueError("global initialization does not reproduce from the exact admitted train fold")
    return initialization


def _pilot_objective(args: argparse.Namespace, prediction: Mapping[str, Any], batch: Mapping[str, Any]):
    if _pilot_variant(args) == "legacy_v1":
        return trackb_masked_real_static_objective(prediction, batch)
    from .effective_observation_loss import effective_observation_objective

    return effective_observation_objective(prediction, batch)


def real_pilot_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Read and validate the admitted train fold without optimizer or output writes."""
    try:
        import torch
    except ImportError as error:
        raise SystemExit("real Track B preflight requires PyTorch") from error
    checkout_commit, checkout_tree, acceptance_sha = _require_real_pilot_arguments(args)
    dataset = _open_real_pilot_dataset(args)
    try:
        if len(dataset) == 0:
            raise ValueError("Track B real pilot source contains no admitted train rows")
        first = dataset[0]
        sample_ids = tuple(str(dataset[index]["sample_id"]) for index in range(len(dataset)))
        group_ids = tuple(sorted({str(dataset[index]["group_id"]) for index in range(len(dataset))}))
        initialization = _verified_pilot_global_initialization(args, dataset, checkout_commit, checkout_tree)
        seed_trackb_runtime(args.seed)
        model = _build_real_pilot_model(args, first, len(dataset.session_index), global_initialization=initialization,
                                       operation_metadata_sha=getattr(dataset, "operation_metadata_sha", None))
        resolved_device = resolve_trackb_device(
            args.device, cuda_available=torch.cuda.is_available(),
        )
        if _pilot_variant(args) == "effective_family_context_v3" and resolved_device != "cpu":
            raise ValueError("family-context V3 currently requires CPU execution for independent replay")
        device = torch.device(resolved_device)
        model.to(device)
        batch_plan_sha = canonical_sha256({
            "schema": "TrackBBatchPlanV1", "sample_ids": list(sample_ids),
            "batch_size": args.pilot_batch_size, "view_seed": 20261001,
            "query_fraction": 0.25, "shuffle": False, "drop_last": False,
        })
        from torch.utils.data import DataLoader

        all_row_forward_loss_finite = True
        all_row_b0_probability_parity = True
        all_row_global_initialization_parity = True
        batch_count = 0
        next_batch_index = 0

        def preflight_collate(items):
            nonlocal next_batch_index
            batch_index = next_batch_index
            next_batch_index += 1
            return trackb_real_static_collate(
                items, epoch=0, view_seed=20261001, query_fraction=0.25,
                batch_plan_sha=batch_plan_sha, batch_index=batch_index,
                cursor_start=batch_index * args.pilot_batch_size,
            )

        for batch_items in DataLoader(
            dataset, batch_size=args.pilot_batch_size, shuffle=False, drop_last=False,
            collate_fn=preflight_collate,
        ):
            batch_count += 1
            device_batch = _move_to_device(batch_items, device)
            with torch.no_grad():
                prediction = model(device_batch)
                losses = _pilot_objective(args, prediction, device_batch)
                if _pilot_variant(args) != "legacy_v1":
                    from .evaluation import build_trackb_fixed_visible_marginal_prediction

                    baseline = build_trackb_fixed_visible_marginal_prediction(device_batch)
                    all_row_b0_probability_parity = all_row_b0_probability_parity and all(
                        torch.equal(prediction[name], baseline[name].double()) for name in (
                            "signal_probability", "coin_given_signal_probability",
                            "idler_given_no_signal_probability", "four_cell_concentration",
                        )
                    )
                if initialization is not None:
                    expected_globals = {
                        "multi_means": torch.tensor(initialization.mean_counts, dtype=torch.float64, device="cpu"),
                        "multi_dispersion": torch.tensor(initialization.dispersion, dtype=torch.float64, device="cpu"),
                        "pair_logits": torch.tensor(initialization.pair_probabilities, dtype=torch.float64, device="cpu").log(),
                    }
                    all_row_global_initialization_parity = all_row_global_initialization_parity and all(
                        torch.equal(prediction[name], value.to(device).expand_as(prediction[name]))
                        for name, value in expected_globals.items()
                    )
            if not _all_tensors_finite(prediction, torch) \
                    or not _all_tensors_finite(losses, torch):
                all_row_forward_loss_finite = False
                break
        return {
            "schema": ("TrackBRealPilotPreflightReceiptV1" if _pilot_variant(args) == "legacy_v1" else
                       "TrackBEffectiveFamilyContextPreflightReceiptV3" if _pilot_variant(args) == "effective_family_context_v3" else
                       "TrackBEffectivePilotPreflightReceiptV2"),
            **({} if _pilot_variant(args) == "legacy_v1" else {
                "observation_model": _pilot_variant(args),
                "model_config": model.config.to_dict(),
                "model_config_sha": model.config.sha256(),
                "all_row_b0_probability_parity": all_row_b0_probability_parity,
            }),
            "status": (
                "PARTIAL_DATA_FOLD_PREFLIGHT"
                if all_row_forward_loss_finite and all_row_b0_probability_parity and all_row_global_initialization_parity
                else "FAILED_GLOBAL_INITIALIZATION_PARITY" if all_row_forward_loss_finite and not all_row_global_initialization_parity
                else "FAILED_B0_PROBABILITY_PARITY" if all_row_forward_loss_finite
                else "FAILED_NONFINITE_FORWARD_OR_LOSS"
            ),
            "system_type": args.system_type,
            **({} if initialization is None else {"all_row_global_initialization_parity": all_row_global_initialization_parity}),
            "source_commit": checkout_commit,
            "source_tree": checkout_tree,
            "source_acceptance_sha": acceptance_sha,
            "admission_manifest_sha": dataset.admission_manifest.sha256(),
            "operation_metadata_sidecar_file_sha": (
                dataset.admission_manifest.operation_metadata_sidecar_file_sha
            ),
            "operation_metadata_sidecar_semantic_sha": (
                dataset.admission_manifest.operation_metadata_sidecar_semantic_sha
            ),
            "operation_metadata_origin": dataset.admission_manifest.operation_metadata_origin,
            "outer_fold_sha": dataset.fold_plan.sha256(),
            "row_identity_sha": dataset.row_identity_sha,
            "outer_role": dataset.outer_role,
            "sample_count": len(sample_ids),
            "group_count": len(group_ids),
            "projector_dim": int(first["projector_descriptor"].shape[-1]),
            "raw_rf_dim": int(first["raw_rf_descriptor"].shape[-1]),
            "prepared_command_dim": int(first["prepared_command"].shape[-1]),
            "session_count": len(dataset.session_index),
            "optimizer_started": False,
            "hardware_action_authorized": False,
            "authoritative_registry_binding": bool(
                args.basis_config_root
                or (args.expected_operation_registry_sha and args.expected_command_table_sha)
            ),
            "all_row_forward_loss_finite": all_row_forward_loss_finite,
            "forward_loss_batch_count": batch_count,
            "initial_model_state_sha": trackb_model_state_sha(model),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "resolved_device": resolved_device,
            "output_root_checked": False,
            "exact_resume_checked": False,
        }
    finally:
        dataset.close()


def _require_successful_real_pilot_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, object], str]:
    """Replay and bind the no-optimizer admission before a pilot can write output."""
    preflight = real_pilot_preflight(args)
    if type(preflight) is not dict \
            or preflight.get("status") != "PARTIAL_DATA_FOLD_PREFLIGHT" \
            or preflight.get("all_row_forward_loss_finite") is not True:
        raise SystemExit("--real-pilot requires a successful in-process Track B real preflight")
    if preflight.get("optimizer_started") is not False \
            or preflight.get("hardware_action_authorized") is not False \
            or preflight.get("outer_role") != "train" \
            or preflight.get("authoritative_registry_binding") is not True \
            or preflight.get("output_root_checked") is not False \
            or preflight.get("exact_resume_checked") is not False:
        raise SystemExit("Track B real preflight does not satisfy the pilot safety contract")
    if preflight.get("source_commit") != args.source_commit \
            or preflight.get("source_tree") != args.source_tree:
        raise SystemExit("Track B real preflight source identity does not match the requested pilot")
    if preflight.get("resolved_device") not in {"cpu", "cuda"} \
            or type(preflight.get("cuda_available")) is not bool \
            or type(preflight.get("sample_count")) is not int \
            or type(preflight.get("group_count")) is not int \
            or int(preflight["sample_count"]) <= 0 \
            or int(preflight["group_count"]) <= 0:
        raise SystemExit("Track B real preflight payload is incomplete")
    require_git_oid(str(preflight["source_commit"]), "preflight_source_commit")
    require_git_oid(str(preflight["source_tree"]), "preflight_source_tree")
    for name in (
        "source_acceptance_sha", "admission_manifest_sha", "outer_fold_sha",
        "row_identity_sha", "initial_model_state_sha",
    ):
        require_sha256(str(preflight.get(name, "")), f"preflight_{name}")
    if _pilot_variant(args) != "legacy_v1" and (
        preflight.get("schema") != ("TrackBEffectiveFamilyContextPreflightReceiptV3"
                                    if _pilot_variant(args) == "effective_family_context_v3" else
                                    "TrackBEffectivePilotPreflightReceiptV2")
        or preflight.get("observation_model") != _pilot_variant(args)
        or preflight.get("all_row_b0_probability_parity") is not True
    ):
        raise SystemExit("effective pilot requires its own model-bound B0-parity preflight")
    if getattr(args, "global_initialization", None) and preflight.get("all_row_global_initialization_parity") is not True:
        raise SystemExit("initialized global pilot requires all-row global-baseline parity")
    return preflight, canonical_sha256(preflight)


def run_real_pilot(args: argparse.Namespace) -> dict[str, object]:
    """Run only a source-bound, train-role, at-most-2,000-step real-static pilot."""
    try:
        import torch
    except ImportError as error:
        raise SystemExit("--real-pilot requires PyTorch") from error
    if args.output_root is None:
        raise SystemExit("--real-pilot requires --output-root")
    if type(args.pilot_steps) is not int or not 1 <= args.pilot_steps <= MAX_SMOKE_OPTIMIZER_STEPS:
        raise SystemExit(f"--pilot-steps must be in [1,{MAX_SMOKE_OPTIMIZER_STEPS}]")
    preflight, preflight_sha = _require_successful_real_pilot_preflight(args)
    checkout_commit = str(preflight["source_commit"])
    checkout_tree = str(preflight["source_tree"])
    acceptance_sha = str(preflight["source_acceptance_sha"])
    output_root = Path(args.output_root)
    if args.resume_checkpoint is not None:
        checkpoint = Path(args.resume_checkpoint).resolve()
        if checkpoint.parent != output_root.resolve():
            raise SystemExit("--resume-checkpoint must be directly inside --output-root")
        if args.resume_sha is None:
            raise SystemExit("--resume-checkpoint requires --resume-sha")
        require_sha256(args.resume_sha, "resume_sha")
    elif output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("--output-root must be empty for a new pilot; use --resume-checkpoint to continue")
    dataset = _open_real_pilot_dataset(args)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        initialization = _verified_pilot_global_initialization(args, dataset, checkout_commit, checkout_tree)
        seed_trackb_runtime(args.seed)
        first = dataset[0]
        model = _build_real_pilot_model(args, first, len(dataset.session_index), global_initialization=initialization,
                                       operation_metadata_sha=getattr(dataset, "operation_metadata_sha", None))
        variant = _pilot_variant(args)
        artifact_prefix = "trackb_real_pilot" if variant == "legacy_v1" else f"trackb_{variant}_pilot"
        if variant != "legacy_v1" and (
            model.config.to_dict() != preflight.get("model_config")
            or model.config.sha256() != preflight.get("model_config_sha")
        ):
            raise RuntimeError("effective model configuration changed after preflight")
        config = TrackBRealRuntimeConfigV1(
            optimizer_steps=args.pilot_steps,
            batch_size=args.pilot_batch_size,
            execution_mode="pilot",
        )
        initial_model_state_sha = trackb_model_state_sha(model)
        if initial_model_state_sha != preflight["initial_model_state_sha"]:
            raise RuntimeError("Track B pilot initial model state changed after its in-process preflight")
        sample_ids = tuple(str(dataset[index]["sample_id"]) for index in range(len(dataset)))
        group_ids = tuple(sorted({str(dataset[index]["group_id"]) for index in range(len(dataset))}))
        row_identities = tuple(
            (sample_id, group_id, content_sha, role)
            for sample_id in sorted(sample_ids)
            for group_id, content_sha, role in (dataset.row_identity_records[sample_id],)
        )
        session_index_bindings = tuple(
            (sample_id, int(dataset[index]["session_index"].item()))
            for index, sample_id in enumerate(sample_ids)
        )
        batch_plan_sha = canonical_sha256({
            "schema": "TrackBBatchPlanV1",
            "sample_ids": list(sample_ids),
            "batch_size": args.pilot_batch_size,
            "view_seed": 20261001,
            "query_fraction": 0.25,
            "shuffle": False,
            "drop_last": False,
        })
        identity = TrackBRealRuntimeIdentityV1(
            run_id=f"{artifact_prefix}_{args.build_id}",
            system_type=args.system_type,
            admission_manifest_sha=dataset.admission_manifest.sha256(),
            outer_fold_sha=dataset.fold_plan.sha256(),
            model_config_sha=model.config.sha256(),
            source_commit=checkout_commit,
            source_tree=checkout_tree,
            initial_model_state_sha=initial_model_state_sha,
            seed=args.seed,
            allowed_sample_ids=sample_ids,
            allowed_group_ids=group_ids,
            allowed_roles=("train",),
            allowed_row_identities=row_identities,
            row_identity_sha=dataset.row_identity_sha,
            batch_plan_sha=batch_plan_sha,
            batch_plan_sample_ids=sample_ids,
            batch_plan_batch_size=args.pilot_batch_size,
            batch_plan_view_seed=20261001,
            batch_plan_query_fraction=0.25,
            session_index_bindings=session_index_bindings,
            operation_metadata_sidecar_file_sha=(
                dataset.admission_manifest.operation_metadata_sidecar_file_sha
            ),
            operation_metadata_sidecar_semantic_sha=(
                dataset.admission_manifest.operation_metadata_sidecar_semantic_sha
            ),
        )
        resolved_device = resolve_trackb_device(
            args.device, cuda_available=torch.cuda.is_available(),
        )
        if resolved_device != preflight["resolved_device"]:
            raise RuntimeError("Track B pilot device changed after its in-process preflight")
        actual_preflight_bindings = {
            "admission_manifest_sha": dataset.admission_manifest.sha256(),
            "outer_fold_sha": dataset.fold_plan.sha256(),
            "row_identity_sha": dataset.row_identity_sha,
            "projector_dim": int(first["projector_descriptor"].shape[-1]),
            "raw_rf_dim": int(first["raw_rf_descriptor"].shape[-1]),
            "prepared_command_dim": int(first["prepared_command"].shape[-1]),
            "session_count": len(dataset.session_index),
            "operation_metadata_sidecar_file_sha": (
                dataset.admission_manifest.operation_metadata_sidecar_file_sha
            ),
            "operation_metadata_sidecar_semantic_sha": (
                dataset.admission_manifest.operation_metadata_sidecar_semantic_sha
            ),
        }
        if any(preflight.get(name) != value for name, value in actual_preflight_bindings.items()):
            raise RuntimeError("Track B pilot source changed after its in-process preflight")
        runtime_type = TrackBRealTrainingRuntimeV1
        runtime_options = {}
        if variant != "legacy_v1":
            from .effective_observation_runtime import Effective_observation_runtime, Family_context_observation_runtime
            from .pilot_diagnostics import Pilot_diagnostic_config

            runtime_type = (Family_context_observation_runtime if variant == "effective_family_context_v3"
                            else Effective_observation_runtime)
            runtime_options["diagnostic_config"] = Pilot_diagnostic_config(enabled=True)
        runtime = runtime_type(
            model=model, identity=identity, config=config, device=resolved_device, **runtime_options,
        )
        checkpoint_path = output_root / f"{artifact_prefix}_checkpoint.pt"
        monitor_path = output_root / f"{artifact_prefix}_monitor.json"
        if args.resume_checkpoint is not None:
            runtime.resume_checkpoint(args.resume_checkpoint, expected_sha256=args.resume_sha)

        batches_per_pass = max(
            1, (len(dataset) + args.pilot_batch_size - 1) // args.pilot_batch_size,
        )

        def batch_iterator_factory(global_step: int):
            from torch.utils.data import DataLoader

            pass_index = global_step // batches_per_pass
            first_batch_index = global_step % batches_per_pass
            next_batch_index = 0

            def pilot_collate(items):
                nonlocal next_batch_index
                batch_index = next_batch_index
                next_batch_index += 1
                return trackb_real_static_collate(
                    items, epoch=pass_index, view_seed=20261001, query_fraction=0.25,
                    batch_plan_sha=batch_plan_sha, batch_index=batch_index,
                    cursor_start=batch_index * args.pilot_batch_size,
                )

            loader = DataLoader(
                dataset,
                batch_size=args.pilot_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=pilot_collate,
            )
            return islice(loader, first_batch_index, None)

        receipt = runtime.run(
            batch_iterator_factory,
            checkpoint_path=checkpoint_path,
            monitor_path=monitor_path,
            **({} if variant == "legacy_v1" else {"diagnostics_dir": output_root / f"{artifact_prefix}_diagnostics"}),
        )
        result: dict[str, object] = {
            **receipt,
            **({} if variant == "legacy_v1" else {
                "observation_model": variant,
                "model_config": model.config.to_dict(),
                # These are parsed options captured during this invocation,
                # not reconstructed historical argv or an inferred receipt.
                "parsed_pilot_request": dict(vars(args)),
                "admission_manifest": dataset.admission_manifest.to_dict(),
            }),
            "source_commit": checkout_commit,
            "source_tree": checkout_tree,
            "source_acceptance_sha": acceptance_sha,
            "admission_manifest_sha": identity.admission_manifest_sha,
            "outer_fold_sha": identity.outer_fold_sha,
            "pilot_budget_max": MAX_SMOKE_OPTIMIZER_STEPS,
            "hardware_action_authorized": False,
            "preflight_replayed_in_process": True,
            "preflight_receipt_sha": preflight_sha,
            "preflight_receipt": preflight,
        }
        (output_root / f"{artifact_prefix}_receipt.json").write_bytes(canonical_json_bytes(result))
        return result
    finally:
        dataset.close()


def cuda_contract_smoke() -> dict[str, object]:
    """Probe CUDA capability without claiming it as a training or hardware pass."""
    try:
        import torch
    except ImportError:
        return {
            "schema": "TrackBCudaContractSmokeV1",
            "status": "CUDA_CONTRACT_SMOKE_SKIPPED_TORCH_UNAVAILABLE",
            "cuda_available": False,
            "long_training_started": False,
        }
    if not torch.cuda.is_available():
        return {
            "schema": "TrackBCudaContractSmokeV1",
            "status": "CUDA_CONTRACT_SMOKE_SKIPPED_CUDA_UNAVAILABLE",
            "cuda_available": False,
            "torch_version": str(torch.__version__),
            "long_training_started": False,
        }
    # One allocation/finite check only.  There is no optimizer, checkpoint,
    # dataset materialization, or escalation to a long training arm.
    vector = torch.tensor((0.0, 1.0), device="cuda", dtype=torch.float32)
    finite = bool(torch.isfinite(vector).all().item())
    return {
        "schema": "TrackBCudaContractSmokeV1",
        "status": "CUDA_CONTRACT_SMOKE_LOCAL_FINITE_CHECK_ONLY" if finite else "CUDA_CONTRACT_SMOKE_FAILED_NONFINITE",
        "cuda_available": True,
        "torch_version": str(torch.__version__),
        "long_training_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.real_preflight:
        result = real_pilot_preflight(args)
        print(canonical_json_bytes(result).decode("utf-8"))
        success = result["all_row_forward_loss_finite"] and (
            _pilot_variant(args) == "legacy_v1" or result.get("all_row_b0_probability_parity") is True
        ) and (not args.global_initialization or result.get("all_row_global_initialization_parity") is True)
        return 0 if success else 2
    if args.real_pilot:
        print(canonical_json_bytes(run_real_pilot(args)).decode("utf-8"))
        return 0
    if args.long_train or args.execute:
        raise SystemExit("Track B CLI has no long-training or execution authority; wait for explicit follow-up")
    if args.cuda_contract_smoke:
        print(canonical_json_bytes(cuda_contract_smoke()).decode("utf-8"))
        return 0
    receipt = run_short_smoke_training(_smoke_observation(), optimizer_steps=args.smoke_steps)
    print(canonical_json_bytes({
        "schema": receipt.schema,
        "requested_steps": receipt.requested_steps,
        "committed_optimizer_steps": receipt.committed_optimizer_steps,
        "final_query_ratio_of_sums_error": receipt.final_query_ratio_of_sums_error,
        "checkpoint_roundtrip_verified": receipt.checkpoint_roundtrip_verified,
        "long_training_started": False,
        "status": receipt.status,
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
