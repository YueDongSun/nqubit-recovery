"""V10 checkpoint envelope; V9 weights cannot be relabelled as V10 final."""

from __future__ import annotations

from typing import Any, Dict

from .contracts import V10_MODEL_IDS


def validate_input_delivery(envelope: Dict[str, Any]) -> None:
    """Legacy defaults never turn a materialized checkpoint into stream evidence."""
    delivery = envelope.get("input_delivery", "materialized")
    index_kind = envelope.get("dataset_index_kind", "materialized")
    recipe_sha = envelope.get("stream_recipe_sha", "")
    recipe_file_sha = envelope.get("stream_recipe_file_sha", "")
    if delivery == "stream_only":
        if (index_kind != "virtual_stream" or not isinstance(recipe_sha, str)
                or len(recipe_sha) != 64 or any(ch not in "0123456789abcdef" for ch in recipe_sha)
                or not isinstance(recipe_file_sha, str) or len(recipe_file_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in recipe_file_sha)
                or envelope.get("dataset_index_sha") != recipe_sha):
            raise ValueError("stream identity requires canonical and file recipe SHA bindings")
    elif delivery != "materialized" or index_kind != "materialized" or recipe_sha or recipe_file_sha:
        raise ValueError("invalid or mixed input delivery identity")


def validate_checkpoint_envelope(envelope: Dict[str, Any]) -> None:
    validate_input_delivery(envelope)
    required = {"schema", "model_id", "branch", "source_commit", "source_tree_sha",
                "v23_manifest_sha", "synthetic_manifest_sha", "measurement_contract_sha",
                "shot_budget_contract_sha", "rho8_sha", "config_sha", "seed_panel",
                "exact_parameter_count", "calibration", "negative_controls",
                "selection_evidence", "research_only"}
    runtime_required = {"run_mode", "dataset_hashes", "split_hash", "model_state_sha",
                        "epoch", "global_step", "best_metric", "metric_name",
                        "metric_direction", "training_complete", "execution_id"}
    required |= runtime_required
    missing = required - set(envelope)
    if missing:
        raise ValueError(f"V10 checkpoint envelope missing: {sorted(missing)}")
    if envelope["schema"] != "V10CheckpointV1" or envelope["model_id"] not in V10_MODEL_IDS:
        raise ValueError("checkpoint does not identify a formal V10 model")
    if envelope["branch"] != "v10" and envelope["branch"] != "codex/v10_lowdisk_training_prep":
        raise ValueError("V10 checkpoint must bind v10 or the authorized V10 low-disk preparation branch")
    if envelope.get("source_generation") in ("V8", "V9", "V9-R2"):
        raise ValueError("V8/V9 checkpoint cannot be relabelled as V10")
    if envelope["selection_evidence"] is not False or envelope["research_only"] is not True:
        raise ValueError("offline gates are not complete; authority flags must remain false/true")
    if envelope["run_mode"] not in ("smoke", "pilot", "full"):
        raise ValueError("checkpoint run_mode is invalid")
    admission_sha = envelope.get("stream_pilot_admission_sha", "")
    if envelope["run_mode"] == "pilot" and envelope.get("input_delivery") == "stream_only":
        if (not isinstance(admission_sha, str) or len(admission_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in admission_sha)):
            raise ValueError("stream pilot checkpoint requires an admission file SHA")
    elif admission_sha:
        raise ValueError("only stream pilot checkpoints may bind an admission file SHA")
    if envelope["metric_direction"] not in ("min", "max"):
        raise ValueError("checkpoint metric direction is invalid")
    if not isinstance(envelope["dataset_hashes"], dict) or not envelope["dataset_hashes"]:
        raise ValueError("checkpoint must bind non-empty dataset hashes")
    if len(str(envelope["model_state_sha"])) != 64 or len(str(envelope["split_hash"])) != 64:
        raise ValueError("checkpoint state and split hashes must be SHA-256")
