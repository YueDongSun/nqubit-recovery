"""Evaluate frozen V8 feedback checkpoints without contaminating selection.

Development reevaluation is repeatable and sees validation plus the declared
development simulator family only.  Locked final evaluation is a different
entry point: it consumes a hash-bound, one-time access declaration, writes an
exclusive ledger before reading final rows, never changes weights, and never
returns a model-selection score.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ml_recon.physics_hybrid_v7 import train_feedback_models as v7_training

from .contracts import (
    LockedEnsembleReleaseAccessDeclarationV1,
    V8_LOCKED_ENSEMBLE_ACCESS_LEDGER_SCHEMA,
    V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA,
)
from .finalize_feedback_candidate import FINALIZATION_SCHEMA
from .train_feedback_models import (
    CHECKPOINT_SCHEMA,
    TRAINING_REPORT_SCHEMA,
    _file_sha256,
    _stable_json_hash,
    build_causal_context_gate,
    collect_source_tree_audit,
    load_v8_config,
    load_v8_contexts,
    resolve_v8_profiles,
    v8_v7_reuse_seam,
)


DEVELOPMENT_EVALUATION_SCHEMA = "v8_feedback_development_reevaluation_v1"
LOCKED_FINAL_EVALUATION_SCHEMA = "v8_feedback_locked_final_ensemble_evaluation_v1"
LOCKED_RELEASE_DECLARATION_SCHEMA = V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA
LOCKED_ACCESS_LEDGER_SCHEMA = V8_LOCKED_ENSEMBLE_ACCESS_LEDGER_SCHEMA


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("V8 feedback evaluation requires torch")


def locked_release_declaration_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("declaration_sha256", None)
    payload.setdefault("schema", LOCKED_RELEASE_DECLARATION_SCHEMA)
    payload.setdefault("schema_version", 1)
    return _stable_json_hash(payload)


def validate_locked_release_declaration(
    declaration: Mapping[str, Any],
    *,
    parent_manifest_hash: str,
    ensemble_sha256: str,
    config_sha256: str,
    locked_test_declaration_sha256: str,
) -> Dict[str, Any]:
    value = copy.deepcopy(dict(declaration))
    if value.get("schema") != LOCKED_RELEASE_DECLARATION_SCHEMA:
        raise ValueError("locked release-access declaration schema changed")
    contract = LockedEnsembleReleaseAccessDeclarationV1.from_json_dict(value)
    contract.validate()
    expected = {
        "parent_manifest_hash": str(parent_manifest_hash),
        "ensemble_sha256": str(ensemble_sha256),
        "config_sha256": str(config_sha256),
        "locked_test_declaration_sha256": str(
            locked_test_declaration_sha256
        ),
    }
    for field_name, expected_value in expected.items():
        if str(value.get(field_name, "")) != expected_value:
            raise ValueError(f"locked release declaration {field_name} mismatch")
    return value


def claim_locked_final_access(
    declaration: Mapping[str, Any],
    *,
    parent_manifest_hash: str,
    ensemble_sha256: str,
    config_sha256: str,
    locked_test_declaration_sha256: str,
) -> Dict[str, Any]:
    """Atomically consume the only allowed final-test evaluation budget."""

    validated = validate_locked_release_declaration(
        declaration,
        parent_manifest_hash=parent_manifest_hash,
        ensemble_sha256=ensemble_sha256,
        config_sha256=config_sha256,
        locked_test_declaration_sha256=locked_test_declaration_sha256,
    )
    path = Path(validated["canonical_ledger_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": LOCKED_ACCESS_LEDGER_SCHEMA,
        "status": "claimed_before_final_rows_loaded",
        "access_count": 1,
        "maximum_access_count": 1,
        "declaration_sha256": validated["declaration_sha256"],
        "parent_manifest_hash": parent_manifest_hash,
        "ensemble_sha256": ensemble_sha256,
        "config_sha256": config_sha256,
        "locked_test_declaration_sha256": locked_test_declaration_sha256,
        "canonical_ledger_path_sha256": validated[
            "canonical_ledger_path_sha256"
        ],
        "allowed_for_model_selection": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    ledger["ledger_sha256"] = _stable_json_hash(ledger)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(ledger, handle, indent=2, sort_keys=True)
    except FileExistsError as exc:
        raise RuntimeError(
            f"locked final-test access was already consumed: {path}"
        ) from exc
    return {**ledger, "path": str(path)}


@contextlib.contextmanager
def _v8_checkpoint_schema_for_v7_reevaluation() -> Iterator[None]:
    previous = v7_training.CHECKPOINT_SCHEMA
    v7_training.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    try:
        yield
    finally:
        v7_training.CHECKPOINT_SCHEMA = previous


def _load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: "torch.device",
) -> tuple[Dict[str, Any], Any]:
    from .feedback_models import V8FeedbackModelBundle, V8FeedbackModelConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported V8 feedback checkpoint")
    if (
        checkpoint.get("frozen_inference") is not True
        or checkpoint.get("online_training_allowed") is not False
        or checkpoint.get("promotion_ready") is not False
        or checkpoint.get("hardware_action_authorized") is not False
    ):
        raise ValueError("V8 evaluation requires a frozen no-authority checkpoint")
    model = V8FeedbackModelBundle(
        V8FeedbackModelConfig.from_mapping(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def _validate_evaluation_provenance(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if str(checkpoint.get("config_sha256", "")) != str(config["_config_sha256"]):
        raise ValueError("checkpoint and evaluation config hashes differ")
    current_source = collect_source_tree_audit(config["_config_path"])
    trained_source = dict(checkpoint.get("source_tree_audit") or {})
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": str(config["_config_sha256"]),
        "training_source_tree_sha256": str(
            trained_source.get("source_tree_sha256", "")
        ),
        "evaluation_source_tree_sha256": str(
            current_source.get("source_tree_sha256", "")
        ),
        "source_tree_matches_training": bool(
            trained_source.get("source_tree_sha256")
            == current_source.get("source_tree_sha256")
        ),
        "training_dirty_patch_sha256": str(
            trained_source.get("dirty_patch_sha256", "")
        ),
        "evaluation_dirty_patch_sha256": str(
            current_source.get("dirty_patch_sha256", "")
        ),
    }


def _resolve_manifest_bound_path(path_value: str | Path, *, manifest_path: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_frozen_parent_manifest(
    parent_manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
) -> tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    """Validate the complete frozen ensemble identity before model loading."""

    path = Path(parent_manifest_path).resolve()
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent.get("schema") != FINALIZATION_SCHEMA:
        raise ValueError("locked final evaluation requires a frozen V8 ensemble manifest")
    expected_manifest_hash = _stable_json_hash({
        key: item for key, item in parent.items() if key != "manifest_hash"
    })
    if str(parent.get("manifest_hash", "")) != expected_manifest_hash:
        raise ValueError("frozen parent ensemble manifest hash mismatch")
    if (
        parent.get("selection_scope") != "development_only"
        or parent.get("frozen_inference") is not True
        or parent.get("online_training_allowed") is not False
        or parent.get("research_only") is not True
        or parent.get("promotion_ready") is not False
        or parent.get("hardware_action_authorized") is not False
    ):
        raise ValueError("locked final evaluation requires a frozen no-authority ensemble")
    locked_state = dict(parent.get("locked_final_test") or {})
    if locked_state.get("status") != "sealed_not_accessed":
        raise ValueError("frozen parent ensemble does not preserve a sealed final test")
    if str(parent.get("config_sha256", "")) != str(config["_config_sha256"]):
        raise ValueError("frozen parent ensemble and evaluation config hashes differ")

    member_rows = [copy.deepcopy(dict(value)) for value in parent.get(
        "ensemble_members", []
    )]
    if (
        not member_rows
        or int(parent.get("ensemble_member_count", 0)) != len(member_rows)
        or int(parent.get("seed_count", 0)) != len(member_rows)
    ):
        raise ValueError("frozen parent ensemble member declaration is incomplete")
    seeds = [int(value.get("seed")) for value in member_rows]
    checkpoint_hashes = [str(value.get("checkpoint_sha256", "")) for value in member_rows]
    if len(set(seeds)) != len(seeds) or len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("frozen parent ensemble repeats a seed or checkpoint")
    weights = [float(value.get("weight", 0.0)) for value in member_rows]
    if any(weight <= 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("frozen parent ensemble weights must be positive and sum to one")
    ensemble_identity = {
        "profile": str(parent.get("profile", "")),
        "ablation_id": str(parent.get("ablation_id", "")),
        "members": [
            {
                "seed": int(value["seed"]),
                "checkpoint_sha256": str(value["checkpoint_sha256"]),
                "weight": float(value["weight"]),
            }
            for value in sorted(member_rows, key=lambda item: int(item["seed"]))
        ],
    }
    if str(parent.get("ensemble_sha256", "")) != _stable_json_hash(ensemble_identity):
        raise ValueError("frozen parent ensemble identity hash mismatch")
    for value in member_rows:
        checkpoint_path = _resolve_manifest_bound_path(
            value["checkpoint_path"], manifest_path=path
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if _file_sha256(checkpoint_path) != str(value["checkpoint_sha256"]):
            raise ValueError("frozen parent ensemble member checkpoint hash mismatch")
        value["resolved_checkpoint_path"] = str(checkpoint_path)
    return path, parent, sorted(member_rows, key=lambda item: int(item["seed"]))


def _weighted_ensemble_evaluation_output(
    member_outputs: Sequence[Mapping[str, "torch.Tensor"]],
    weights: Sequence[float],
) -> Dict[str, "torch.Tensor"]:
    """Average every compatible floating output and restore count invariants."""

    if not member_outputs or len(member_outputs) != len(weights):
        raise ValueError("ensemble evaluation requires one weight per member output")
    common_keys = set(member_outputs[0])
    for output in member_outputs[1:]:
        common_keys.intersection_update(output)
    result: Dict[str, "torch.Tensor"] = dict(member_outputs[0])
    for key in common_keys:
        values = [output[key] for output in member_outputs]
        first = values[0]
        if (
            torch.is_tensor(first)
            and all(torch.is_tensor(value) and value.shape == first.shape for value in values)
            and (first.is_floating_point() or torch.is_complex(first))
        ):
            result[key] = sum(
                value * float(weight) for value, weight in zip(values, weights)
            )
    expected_counts = result["expected_counts"]
    result["expected_total"] = expected_counts.sum(dim=-1)
    result["count_probability"] = expected_counts / expected_counts.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return result


def _build_frozen_ensemble_model(
    member_models: Sequence[Any],
    weights: Sequence[float],
) -> Any:
    """Expose one read-only model_a surface backed by every frozen member."""

    class _FrozenEnsembleModelA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.members = torch.nn.ModuleList([model.model_a for model in member_models])
            self.weights = tuple(float(value) for value in weights)

        def forward(self, *args: Any, **kwargs: Any) -> Dict[str, "torch.Tensor"]:
            outputs = [member(*args, **kwargs) for member in self.members]
            return _weighted_ensemble_evaluation_output(outputs, self.weights)

    class _FrozenEnsembleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_a = _FrozenEnsembleModelA()

    model = _FrozenEnsembleModel()
    model.eval()
    return model


def _load_frozen_parent_ensemble(
    parent_manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
    device: "torch.device",
) -> Dict[str, Any]:
    manifest_path, parent, member_rows = _validate_frozen_parent_manifest(
        parent_manifest_path, config=config
    )
    loaded_members: List[Dict[str, Any]] = []
    invariant_hashes: Dict[str, set[str]] = {
        "model_config": set(),
        "training_profile": set(),
        "classifier_prior": set(),
        "locked_final_test": set(),
        "state_pair_transport_weak": set(),
    }
    for row in member_rows:
        checkpoint_path = Path(row["resolved_checkpoint_path"])
        checkpoint, model = _load_checkpoint_model(checkpoint_path, device=device)
        if int(checkpoint.get("seed", -1)) != int(row["seed"]):
            raise ValueError("frozen ensemble member seed differs from checkpoint")
        if str(checkpoint.get("config_sha256", "")) != str(parent["config_sha256"]):
            raise ValueError("frozen ensemble member config hash differs from parent")
        if str(dict(checkpoint.get("training_profile") or {}).get("name", "")) != str(
            parent["profile"]
        ):
            raise ValueError("frozen ensemble member profile differs from parent")
        for field_name in invariant_hashes:
            invariant_hashes[field_name].add(_stable_json_hash(checkpoint.get(field_name)))
        loaded_members.append({
            "seed": int(row["seed"]),
            "weight": float(row["weight"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "checkpoint": checkpoint,
            "model": model,
            "provenance": _validate_evaluation_provenance(
                checkpoint,
                checkpoint_path=checkpoint_path,
                config=config,
            ),
        })
    if any(len(values) != 1 for values in invariant_hashes.values()):
        raise ValueError("frozen ensemble members do not share one inference contract")
    if _stable_json_hash(loaded_members[0]["checkpoint"]["locked_final_test"]) != _stable_json_hash(
        parent["locked_final_test"]
    ):
        raise ValueError("frozen ensemble locked-final declaration differs from parent")
    ensemble_model = _build_frozen_ensemble_model(
        [value["model"] for value in loaded_members],
        [value["weight"] for value in loaded_members],
    ).to(device)
    return {
        "manifest_path": manifest_path,
        "parent": parent,
        "members": loaded_members,
        "model": ensemble_model,
    }


def _classifier_prior_for_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    contexts: Mapping[str, Any],
    device: "torch.device",
) -> Any:
    contract = dict(
        checkpoint.get("classifier_prior") or {"source": "none", "frozen": True}
    )
    if contract.get("frozen") is not True:
        raise ValueError("evaluation refuses a mutable classifier prior")
    source = str(contract.get("source", "none"))
    if source == "none":
        return None
    prior = v7_training.FrozenFeedbackClassifierPrior(
        source=source,
        model_dir="ml_recon/models",
        expected_latent_dim=int(checkpoint["model_config"]["classifier_latent_dim"]),
        device=device,
    )
    actual = prior.audit_dict()["checkpoints"]
    expected = dict(contract.get("checkpoints") or {})
    for system_type in contexts:
        if expected.get(system_type, {}).get("sha256") != actual[system_type]["sha256"]:
            raise ValueError("frozen classifier checkpoint hash changed")
    return prior


def _locked_measured_evaluation(
    model: Any,
    reference_checkpoint: Mapping[str, Any],
    contexts: Mapping[str, Any],
    *,
    evaluation_seed: int,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    profile_payload = dict(reference_checkpoint["training_profile"])
    profile = v7_training.FeedbackTrainingProfile.from_mapping(
        str(profile_payload["name"]), profile_payload
    )
    declaration = reference_checkpoint["locked_final_test"]["declaration"]
    split_names = set(str(item) for item in declaration["final_test_split_names"])
    hidden_fraction = float(
        checkpoint["v8_training_contract"]["real_loss_contract"].get(
            "hidden_operation_fraction",
            0.08,
        )
    )
    result: Dict[str, Any] = {}
    for offset, (system_type, context) in enumerate(contexts.items()):
        selected = np.flatnonzero(np.asarray([
            str(name) in split_names for name in context.arrays.split_names
        ], dtype=bool))
        if selected.size == 0:
            raise ValueError(f"{system_type} locked final split is empty")
        prediction = v7_training._collect_measured_predictions(
            model,
            context,
            selected,
            batch_size=profile.real_batch_size,
            hidden_fraction=hidden_fraction,
            seed=int(evaluation_seed + 9000011 + offset),
            device=device,
            classifier_prior=classifier_prior,
        )
        # The deployable frozen ensemble averages raw member count predictions;
        # it does not select one member's development-fit temperature.
        temperature = 1.0
        result[system_type] = {
            "splits": {
                "locked_final": v7_training._measured_prediction_metrics(
                    prediction,
                    temperature=temperature,
                    seed=int(evaluation_seed + offset),
                )
            },
            "temperature_source": "frozen_ensemble_raw_inference_semantics",
            "row_count": int(selected.size),
            "allowed_for_model_selection": False,
        }
    return result


def _synthetic_log_count_error(
    output: Mapping[str, "torch.Tensor"],
    expected: "torch.Tensor",
) -> float:
    return float(torch.mean(torch.abs(
        torch.log1p(output["expected_counts"])
        - torch.log1p(expected)
    )).detach().cpu())


def _synthetic_log_count_error_per_row(
    output: Mapping[str, "torch.Tensor"],
    expected: "torch.Tensor",
) -> "torch.Tensor":
    return torch.mean(torch.abs(
        torch.log1p(output["expected_counts"])
        - torch.log1p(expected)
    ), dim=-1)


def _causal_negative_control_contract(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve the hash-bound V8 contract across standalone and V7 reuse paths."""

    value = config.get("causal_negative_control_contract")
    if value is None:
        value = config.get("_v8_causal_negative_control_contract")
    if not isinstance(value, Mapping):
        raise ValueError("V8 causal negative-control contract is missing")
    required = {
        "bootstrap_replicates",
        "confidence_level",
        "minimum_groups",
        "minimum_relative_degradation",
        "seed",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(
            f"V8 causal negative-control contract is incomplete: {missing}"
        )
    return copy.deepcopy(dict(value))


def _synthetic_derangement_identity(
    batch: Mapping[str, Any],
    *,
    system_type: str,
    family_id: str,
    generation_seed: int,
    sample_count: int,
) -> tuple[List[str], List[str], List[str], str]:
    """Bind strict synthetic groups to independent generator row draws."""

    causal_mask = np.asarray(batch["causal_history_bound"], dtype=np.float32)
    if causal_mask.shape != (sample_count,) or not np.all(causal_mask == 1.0):
        raise ValueError("synthetic causal controls require bound same-plant histories")
    family_hash = str(np.asarray(batch["synthetic_family_hash"]).item())
    family_role = str(np.asarray(batch["synthetic_family_role"]).item())
    stratum_id = f"{system_type}|{family_role}|{family_id}"
    item_ids = [
        f"{family_hash}:{int(generation_seed)}:row_{index}"
        for index in range(sample_count)
    ]
    # generate_v8_transition_batch independently samples one latent plant and
    # causal rollout per row.  The seed plus row index is therefore the strict
    # synthetic episode/provenance identity; it is never reused as real proof.
    episode_group_ids = [
        f"synthetic_episode:{family_hash}:{int(generation_seed)}:{index}"
        for index in range(sample_count)
    ]
    return item_ids, [stratum_id] * sample_count, episode_group_ids, family_hash


def evaluate_synthetic_causal_negative_controls(
    model: Any,
    contexts: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: "torch.device",
    classifier_prior: Any,
    family_ids: Sequence[str],
    sample_count: int,
    base_seed: int,
    allow_locked_evaluation: bool,
    channel: str,
    allowed_for_model_selection: bool,
    state_endpoint_contract_reference: Optional[str],
) -> Dict[str, Any]:
    from .causal_negative_control import (
        CausalNegativeControlConfig,
        build_target_action_derangement,
        build_target_context_derangement,
        evaluate_paired_vs_shuffled_loss,
    )
    from .feedback_synthetic import generate_v8_transition_batch
    from .train_feedback_models import _condition_v8_model_a

    if int(sample_count) < 2:
        raise ValueError("synthetic negative controls require at least two rows")
    sample_count = int(sample_count)
    family_ids = tuple(str(value) for value in family_ids)
    result: Dict[str, Any] = {}
    for system_offset, (system_type, context) in enumerate(contexts.items()):
        system_result: Dict[str, Any] = {}
        for family_offset, family_id in enumerate(family_ids):
            generation_seed = int(
                base_seed
                + system_offset * 100003
                + family_offset * 1009
            )
            batch = generate_v8_transition_batch(
                context.arrays,
                context.profile,
                batch_size=sample_count,
                seed=generation_seed,
                family_id=str(family_id),
                allow_locked_evaluation=bool(allow_locked_evaluation),
            )
            tensor = v7_training._synthetic_tensors(batch, device=device)
            tensor["context_target_features"] = tensor["target_features"][:, None, :].expand(
                -1, tensor["history"].shape[1], -1
            )
            tensor["context_mask"] = tensor["history_mask"]
            tensor.update(v7_training._classifier_prior_features(
                classifier_prior,
                system_type=system_type,
                target_features=tensor["target_features"],
                rho_available_mask=tensor["rho_available_mask"],
                core_rho=tensor["current_rho"],
            ))
            graph = torch.from_numpy(context.arrays.graph_kernel).to(device)
            item_ids, stratum_ids, episode_group_ids, family_hash = (
                _synthetic_derangement_identity(
                    batch,
                    system_type=system_type,
                    family_id=str(family_id),
                    generation_seed=generation_seed,
                    sample_count=sample_count,
                )
            )
            action_plan = build_target_action_derangement(
                item_ids,
                stratum_ids,
                episode_group_ids,
                seed=int(generation_seed + 4001),
            )
            context_plan = build_target_context_derangement(
                item_ids,
                stratum_ids,
                episode_group_ids,
                seed=int(generation_seed + 8009),
            )
            action_donors = torch.as_tensor(
                action_plan.donor_indices, dtype=torch.long, device=device
            )
            context_donors = torch.as_tensor(
                context_plan.donor_indices, dtype=torch.long, device=device
            )
            context_shuffled_tensor = dict(tensor)
            for name in (
                "history",
                "history_mask",
                "context_target_features",
                "context_mask",
            ):
                context_shuffled_tensor[name] = tensor[name][context_donors]

            def model_a_forward(
                condition_tensor: Mapping[str, "torch.Tensor"],
                *,
                action: "torch.Tensor",
                action_mask: "torch.Tensor",
            ) -> Mapping[str, "torch.Tensor"]:
                with _condition_v8_model_a(model.model_a, condition_tensor):
                    return model.model_a(
                        tensor["operation_features"],
                        tensor["target_features"],
                        graph,
                        action,
                        action_mask,
                        system_type=system_type,
                        classifier_latent=tensor.get("classifier_latent"),
                        classifier_uncertainty=tensor.get("classifier_uncertainty"),
                    )

            with torch.no_grad():
                correct = model_a_forward(
                    tensor,
                    action=tensor["action"],
                    action_mask=tensor["action_mask"],
                )
                shuffled_action = model_a_forward(
                    tensor,
                    action=tensor["action"][action_donors],
                    action_mask=tensor["action_mask"][action_donors],
                )
                shuffled_context = model_a_forward(
                    context_shuffled_tensor,
                    action=tensor["action"],
                    action_mask=tensor["action_mask"],
                )
            correct_per_row = _synthetic_log_count_error_per_row(
                correct, tensor["next_expected_counts"]
            ).detach().float().cpu().numpy()
            action_per_row = _synthetic_log_count_error_per_row(
                shuffled_action, tensor["next_expected_counts"]
            ).detach().float().cpu().numpy()
            context_per_row = _synthetic_log_count_error_per_row(
                shuffled_context, tensor["next_expected_counts"]
            ).detach().float().cpu().numpy()
            negative_contract = _causal_negative_control_contract(config)
            negative_config = CausalNegativeControlConfig(
                bootstrap_replicates=int(negative_contract["bootstrap_replicates"]),
                confidence_level=float(negative_contract["confidence_level"]),
                minimum_groups=int(negative_contract["minimum_groups"]),
                minimum_relative_degradation=float(
                    negative_contract["minimum_relative_degradation"]
                ),
                seed=int(negative_contract["seed"]),
            )
            action_result = evaluate_paired_vs_shuffled_loss(
                correct_per_row.tolist(),
                action_per_row.tolist(),
                episode_group_ids,
                evidence_source="synthetic_transition",
                shuffle_kind="target_action",
                config=negative_config,
            )
            context_result = evaluate_paired_vs_shuffled_loss(
                correct_per_row.tolist(),
                context_per_row.tolist(),
                episode_group_ids,
                evidence_source="synthetic_transition",
                shuffle_kind="target_context",
                config=negative_config,
            )
            correct_error = float(np.mean(correct_per_row))
            action_error = float(np.mean(action_per_row))
            context_error = float(np.mean(context_per_row))
            action_payload = asdict(action_result)
            action_payload["result_hash"] = action_result.result_hash
            action_payload["derangement_plan_hash"] = action_plan.plan_hash
            context_payload = asdict(context_result)
            context_payload["result_hash"] = context_result.result_hash
            context_payload["derangement_plan_hash"] = context_plan.plan_hash
            system_result[str(family_id)] = {
                "family_hash": family_hash,
                "sample_count": sample_count,
                "correct_pair_log_count_mae": correct_error,
                "target_context_block_derangement_log_count_mae": context_error,
                "target_action_block_derangement_log_count_mae": action_error,
                "target_context_block_derangement_degradation": (
                    context_error - correct_error
                ),
                "target_action_block_derangement_degradation": (
                    action_error - correct_error
                ),
                "negative_controls": {
                    "target_action": action_payload,
                    "target_context": context_payload,
                    "state_endpoint": {
                        "status": "reported_in_state_pair_transport_weak_channel",
                        "evidence_scope": "weak",
                        "counts_as_real_causality_evidence": False,
                        "contract_reference": state_endpoint_contract_reference,
                    },
                },
                "derangement_stratum": "system_type|synthetic_family_role|family_id",
                "derangement_group_source": (
                    "independent_generator_seed_family_row_episode"
                ),
                "whole_context_block_fields": [
                    "history",
                    "history_mask",
                    "context_target_features",
                    "context_mask",
                ],
                "channel": channel,
                "counts_as_real_causality_evidence": False,
                "allowed_for_model_selection": bool(allowed_for_model_selection),
            }
        result[system_type] = system_result
    return result


def _locked_synthetic_ood_evaluation(
    model: Any,
    checkpoint: Mapping[str, Any],
    contexts: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: "torch.device",
    classifier_prior: Any,
) -> Dict[str, Any]:
    return evaluate_synthetic_causal_negative_controls(
        model,
        contexts,
        config,
        device=device,
        classifier_prior=classifier_prior,
        family_ids=config["synthetic_plant_families"]["locked_ood"],
        sample_count=min(64, int(config["selection"]["synthetic_evaluation_samples"])),
        base_seed=int(checkpoint["seed"] + 12000017),
        allow_locked_evaluation=True,
        channel="synthetic_locked_ood_only",
        allowed_for_model_selection=False,
        state_endpoint_contract_reference=checkpoint[
            "state_pair_transport_weak"
        ]["contract_sha256"],
    )


def evaluate_v8_checkpoint(
    checkpoint_path: Optional[str | Path] = None,
    *,
    parent_manifest_path: Optional[str | Path] = None,
    config: Mapping[str, Any],
    output_dir: str | Path,
    device_name: str = "auto",
    scope: str = "development",
    locked_declaration_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    _require_torch()
    if scope not in {"development", "locked_final"}:
        raise ValueError("scope must be development or locked_final")
    device = v7_training._resolve_device(device_name)

    if scope == "development":
        if checkpoint_path is None or parent_manifest_path is not None:
            raise ValueError("development evaluation requires exactly one checkpoint")
        source_path = Path(checkpoint_path).resolve()
        checkpoint, _ = _load_checkpoint_model(source_path, device=device)
        provenance = _validate_evaluation_provenance(
            checkpoint,
            checkpoint_path=source_path,
            config=config,
        )
        runtime_config, _ = resolve_v8_profiles(
            config,
            [str(checkpoint["training_profile"]["name"])],
        )
        contexts = load_v8_contexts(config, runtime_config)
        with v8_v7_reuse_seam(
            development_families=config["synthetic_plant_families"]["development"]
        ), _v8_checkpoint_schema_for_v7_reevaluation():
            report = v7_training.reevaluate_feedback_checkpoint(
                source_path,
                config=runtime_config,
                contexts=contexts,
                output_dir=output_dir,
                device=device,
            )
        report.update({
            "schema": DEVELOPMENT_EVALUATION_SCHEMA,
            "evaluation_scope": "development_only",
            "allowed_for_model_selection": True,
            "locked_final_rows_accessed": 0,
            "locked_final_metrics_present": False,
            "provenance": provenance,
            "causal_context_gate": copy.deepcopy(
                checkpoint["causal_context_gate"]
            ),
            "weights_changed": False,
            "online_training_used": False,
            "research_only": True,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        })
        report.pop("report_hash", None)
        report["report_hash"] = _stable_json_hash(report)
        report_path = (
            Path(output_dir)
            / str(checkpoint["training_profile"]["name"])
            / f"seed_{int(checkpoint['seed'])}"
            / "checkpoint_reevaluation.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    if checkpoint_path is not None:
        raise ValueError(
            "member-only checkpoint locked-final evaluation is prohibited; "
            "provide the frozen parent ensemble manifest"
        )
    if parent_manifest_path is None:
        raise ValueError("locked final evaluation requires --parent-manifest")
    if locked_declaration_path is None:
        raise ValueError("locked final evaluation requires --locked-declaration")
    frozen = _load_frozen_parent_ensemble(
        parent_manifest_path,
        config=config,
        device=device,
    )
    parent = frozen["parent"]
    members = frozen["members"]
    ensemble_model = frozen["model"]
    reference_checkpoint = members[0]["checkpoint"]
    runtime_config, _ = resolve_v8_profiles(
        config,
        [str(parent["profile"])],
    )
    release_declaration = json.loads(
        Path(locked_declaration_path).read_text(encoding="utf-8")
    )
    locked_state = dict(parent["locked_final_test"])
    ledger = claim_locked_final_access(
        release_declaration,
        parent_manifest_hash=str(parent["manifest_hash"]),
        ensemble_sha256=str(parent["ensemble_sha256"]),
        config_sha256=str(config["_config_sha256"]),
        locked_test_declaration_sha256=str(locked_state["declaration_sha256"]),
    )
    # The single-use claim is durable before any context loader can expose a
    # locked row.  Changing output_dir can only move the report, never ledger.
    contexts = load_v8_contexts(config, runtime_config)
    classifier_prior = _classifier_prior_for_checkpoint(
        reference_checkpoint,
        contexts=contexts,
        device=device,
    )
    evaluation_seed = int(str(parent["ensemble_sha256"])[:8], 16) & 0x7FFFFFFF
    evaluation_checkpoint = copy.deepcopy(reference_checkpoint)
    evaluation_checkpoint["seed"] = evaluation_seed
    with v8_v7_reuse_seam(
        development_families=config["synthetic_plant_families"]["development"]
    ):
        measured = _locked_measured_evaluation(
            ensemble_model,
            evaluation_checkpoint,
            contexts,
            evaluation_seed=evaluation_seed,
            device=device,
            classifier_prior=classifier_prior,
        )
        synthetic = _locked_synthetic_ood_evaluation(
            ensemble_model,
            evaluation_checkpoint,
            contexts,
            config,
            device=device,
            classifier_prior=classifier_prior,
        )
    report = {
        "schema": LOCKED_FINAL_EVALUATION_SCHEMA,
        "evaluation_scope": "locked_final_single_use",
        "allowed_for_model_selection": False,
        "parent_ensemble": {
            "manifest_path": str(frozen["manifest_path"]),
            "manifest_file_sha256": _file_sha256(frozen["manifest_path"]),
            "manifest_hash": str(parent["manifest_hash"]),
            "manifest_schema": FINALIZATION_SCHEMA,
            "ensemble_sha256": str(parent["ensemble_sha256"]),
            "ensemble_member_count": len(members),
            "members": [
                {
                    "seed": int(value["seed"]),
                    "weight": float(value["weight"]),
                    "checkpoint_path": str(value["checkpoint_path"]),
                    "checkpoint_sha256": str(value["checkpoint_sha256"]),
                    "checkpoint_schema": CHECKPOINT_SCHEMA,
                }
                for value in members
            ],
        },
        "parent_manifest_hash": str(parent["manifest_hash"]),
        "ensemble_sha256": str(parent["ensemble_sha256"]),
        "evaluation_seed": evaluation_seed,
        "profile": copy.deepcopy(reference_checkpoint["training_profile"]),
        "provenance": {
            "all_frozen_members_loaded": True,
            "ensemble_member_count": len(members),
            "member_provenance": [
                {
                    "seed": int(value["seed"]),
                    "weight": float(value["weight"]),
                    **copy.deepcopy(value["provenance"]),
                }
                for value in members
            ],
        },
        "ensemble_inference_contract": {
            "all_members_evaluated": True,
            "member_outputs_aggregated_before_metrics": True,
            "member_weights_sha256": _stable_json_hash([
                {
                    "seed": int(value["seed"]),
                    "weight": float(value["weight"]),
                    "checkpoint_sha256": str(value["checkpoint_sha256"]),
                }
                for value in members
            ]),
            "single_member_selection_used": False,
        },
        "locked_release_declaration": release_declaration,
        "locked_access_ledger": ledger,
        "measured_locked_final": measured,
        "synthetic_locked_ood": synthetic,
        "causal_context_gate": build_causal_context_gate(
            config,
            real_paired_or_signed_probe_count=0,
        ),
        "weights_changed": False,
        "online_training_used": False,
        "selection_score_emitted": False,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    report["report_hash"] = _stable_json_hash(report)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "locked_final_evaluation.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--parent-manifest")
    parser.add_argument("--config", default="ml_recon/configs/physics_hybrid_v8_feedback.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scope", choices=("development", "locked_final"), default="development"
    )
    parser.add_argument("--locked-declaration")
    args = parser.parse_args(argv)
    evaluate_v8_checkpoint(
        args.checkpoint,
        parent_manifest_path=args.parent_manifest,
        config=load_v8_config(args.config),
        output_dir=args.output_dir,
        device_name=args.device,
        scope=args.scope,
        locked_declaration_path=args.locked_declaration,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_EVALUATION_SCHEMA",
    "LOCKED_ACCESS_LEDGER_SCHEMA",
    "LOCKED_FINAL_EVALUATION_SCHEMA",
    "LOCKED_RELEASE_DECLARATION_SCHEMA",
    "claim_locked_final_access",
    "evaluate_v8_checkpoint",
    "locked_release_declaration_hash",
    "main",
    "validate_locked_release_declaration",
]
