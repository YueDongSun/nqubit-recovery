"""V10 checkpoint envelope; V9 weights cannot be relabelled as V10 final."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .contracts import TRACKB_TRAINING_BRANCH, V10_AUTHORIZED_BRANCHES, V10_MODEL_IDS
from .control_coordinate_layout import INTERLEAVED_CONTROL_LAYOUT_SHA


PHYSICAL_MODE_BINDING_SCHEMA = "V10PhysicalModeBindingV1"
TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH = "track_a_operation_poisson_research"


def _validate_physical_mode_binding(envelope: Mapping[str, Any]) -> None:
    """Keep V3 replay and V4 operation-Poisson checkpoints non-interchangeable."""
    recipe = envelope.get("operation_input_recipe")
    mode = envelope.get("excitation_mode")
    manifest_sha = envelope.get("output_training_manifest_sha")
    binding = envelope.get("physical_mode_binding")
    version = envelope.get("dataset_contract_version")
    historical_v3 = all(value is None for value in (recipe, mode, manifest_sha, binding))
    if historical_v3:
        if version != "V10PhysicalAodDatasetRecipeV3":
            raise ValueError("physical checkpoint has an invalid admission identity")
        return
    if not (
        isinstance(recipe, str)
        and isinstance(mode, str)
        and isinstance(manifest_sha, str)
        and len(manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha)
        and isinstance(binding, Mapping)
        and dict(binding)
        == {
            "schema": PHYSICAL_MODE_BINDING_SCHEMA,
            "operation_input_recipe": recipe,
            "excitation_mode": mode,
            "output_training_manifest_sha": manifest_sha,
        }
    ):
        raise ValueError("physical checkpoint mode binding is malformed")
    expected_version = {
        ("legacy_v1", "legacy"): "V10PhysicalAodDatasetRecipeV3",
        ("operation_poisson_v1", "diagnostic"): "V10PhysicalOperationPoissonDatasetRecipeV4",
        ("operation_poisson_v1", "coupled"): "V10PhysicalOperationPoissonDatasetRecipeV4",
    }.get((recipe, mode))
    if version != expected_version:
        raise ValueError("physical checkpoint recipe and excitation mode are not an admitted pair")
    from .published_output_registry import (
        physical_output_training_manifest,
        physical_output_training_manifest_sha,
        validate_physical_output_training_manifest,
    )

    manifest = envelope.get("published_output_training")
    validate_physical_output_training_manifest(manifest)
    if (
        manifest != physical_output_training_manifest(mode)
        or manifest_sha != physical_output_training_manifest_sha(mode)
    ):
        raise ValueError("physical checkpoint output-training manifest mismatch")


def validate_input_delivery(envelope: Dict[str, Any]) -> None:
    """Legacy defaults never turn a materialized checkpoint into stream evidence."""
    delivery = envelope.get("input_delivery", "materialized")
    index_kind = envelope.get("dataset_index_kind", "materialized")
    recipe_sha = envelope.get("stream_recipe_sha", "")
    recipe_file_sha = envelope.get("stream_recipe_file_sha", "")
    if delivery == "stream_only":
        if (
            index_kind != "virtual_stream"
            or not isinstance(recipe_sha, str)
            or len(recipe_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in recipe_sha)
            or not isinstance(recipe_file_sha, str)
            or len(recipe_file_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in recipe_file_sha)
            or envelope.get("dataset_index_sha") != recipe_sha
        ):
            raise ValueError(
                "stream identity requires canonical and file recipe SHA bindings"
            )
    elif (
        delivery != "materialized"
        or index_kind != "materialized"
        or recipe_sha
        or recipe_file_sha
    ):
        raise ValueError("invalid or mixed input delivery identity")


def validate_checkpoint_envelope(envelope: Dict[str, Any]) -> None:
    validate_input_delivery(envelope)
    required = {
        "schema",
        "model_id",
        "branch",
        "source_commit",
        "source_tree_sha",
        "v23_manifest_sha",
        "synthetic_manifest_sha",
        "measurement_contract_sha",
        "shot_budget_contract_sha",
        "rho8_sha",
        "config_sha",
        "seed_panel",
        "exact_parameter_count",
        "calibration",
        "negative_controls",
        "selection_evidence",
        "research_only",
    }
    runtime_required = {
        "run_mode",
        "dataset_hashes",
        "split_hash",
        "model_state_sha",
        "epoch",
        "global_step",
        "best_metric",
        "metric_name",
        "metric_direction",
        "training_complete",
    }
    required |= runtime_required
    missing = required - set(envelope)
    if missing:
        raise ValueError(f"V10 checkpoint envelope missing: {sorted(missing)}")
    if (
        envelope["schema"] != "V10CheckpointV1"
        or envelope["model_id"] not in V10_MODEL_IDS
    ):
        raise ValueError("checkpoint does not identify a formal V10 model")
    physical_branch = "codex/v10_tracka_physical_budget_v2"
    operation_poisson_branch = TRACK_A_OPERATION_POISSON_RESEARCH_BRANCH
    if envelope["branch"] not in V10_AUTHORIZED_BRANCHES and envelope["branch"] not in (
        physical_branch,
        operation_poisson_branch,
    ):
        raise ValueError(
            "V10 checkpoint must bind v10 or the authorized V10 low-disk preparation branch"
        )
    if envelope["branch"] == TRACKB_TRAINING_BRANCH and envelope["run_mode"] != "smoke":
        raise ValueError("Track B checkpoints require the authorized smoke run mode")
    if envelope["branch"] == physical_branch:
        required_physical = {
            "physical_namespace",
            "physical_admission_sha",
            "strict_stream_only_identity",
            "coordinate_layout_sha",
            "physical_identity",
            "published_output_training",
        }
        if not required_physical.issubset(envelope):
            raise ValueError("physical checkpoint requires admission identity fields")
        if (
            envelope["physical_namespace"] != "TRACK_A_PHYSICAL_V2"
            or envelope["strict_stream_only_identity"] is not True
            or envelope["coordinate_layout_sha"] != INTERLEAVED_CONTROL_LAYOUT_SHA
            or envelope.get("input_delivery") != "stream_only"
            or envelope.get("dataset_index_kind") != "virtual_stream"
            or not isinstance(envelope["physical_admission_sha"], str)
            or len(envelope["physical_admission_sha"]) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in envelope["physical_admission_sha"]
            )
        ):
            raise ValueError("physical checkpoint has an invalid admission identity")
        from ml_recon.v10_recoverability_foundry.training.physical_gates import (
            validate_physical_identity,
        )

        validate_physical_identity(envelope["physical_identity"])
        from .published_output_registry import (
            validate_physical_output_training_manifest,
        )

        validate_physical_output_training_manifest(
            envelope["published_output_training"]
        )
        _validate_physical_mode_binding(envelope)
    elif envelope["branch"] == operation_poisson_branch:
        required_research = {
            "operation_input_recipe",
            "excitation_mode",
            "output_training_manifest_sha",
            "physical_mode_binding",
            "published_output_training",
        }
        if not required_research.issubset(envelope):
            raise ValueError("operation-Poisson checkpoint requires a mode-bound V4 identity")
        if (
            envelope.get("input_delivery") != "materialized"
            or envelope.get("dataset_index_kind") != "materialized"
            or envelope.get("stream_pilot_admission_sha")
            or any(
                envelope.get(key)
                for key in (
                    "physical_namespace",
                    "physical_admission_sha",
                    "strict_stream_only_identity",
                    "coordinate_layout_sha",
                    "physical_identity",
                )
            )
        ):
            raise ValueError(
                "operation-Poisson research checkpoint cannot carry stream or physical admission identity"
            )
        _validate_physical_mode_binding(envelope)
    elif any(
        envelope.get(key)
        for key in (
            "physical_namespace",
            "physical_admission_sha",
            "strict_stream_only_identity",
            "coordinate_layout_sha",
            "physical_identity",
            "operation_input_recipe",
            "excitation_mode",
            "output_training_manifest_sha",
            "physical_mode_binding",
        )
    ):
        raise ValueError(
            "nonphysical checkpoint cannot carry physical admission fields"
        )
    if envelope.get("source_generation") in ("V8", "V9", "V9-R2"):
        raise ValueError("V8/V9 checkpoint cannot be relabelled as V10")
    if (
        envelope["selection_evidence"] is not False
        or envelope["research_only"] is not True
    ):
        raise ValueError(
            "offline gates are not complete; authority flags must remain false/true"
        )
    if envelope["run_mode"] not in ("smoke", "pilot", "full"):
        raise ValueError("checkpoint run_mode is invalid")
    admission_sha = envelope.get("stream_pilot_admission_sha", "")
    if envelope["branch"] == physical_branch:
        if admission_sha or envelope["run_mode"] != "pilot":
            raise ValueError(
                "physical checkpoints use their own pilot admission, not a legacy stream receipt"
            )
    elif (
        envelope["run_mode"] == "pilot"
        and envelope.get("input_delivery") == "stream_only"
    ):
        if (
            not isinstance(admission_sha, str)
            or len(admission_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in admission_sha)
        ):
            raise ValueError("stream pilot checkpoint requires an admission file SHA")
    elif admission_sha:
        raise ValueError("only stream pilot checkpoints may bind an admission file SHA")
    if envelope["metric_direction"] not in ("min", "max"):
        raise ValueError("checkpoint metric direction is invalid")
    if (
        not isinstance(envelope["dataset_hashes"], dict)
        or not envelope["dataset_hashes"]
    ):
        raise ValueError("checkpoint must bind non-empty dataset hashes")
    if (
        len(str(envelope["model_state_sha"])) != 64
        or len(str(envelope["split_hash"])) != 64
    ):
        raise ValueError("checkpoint state and split hashes must be SHA-256")
