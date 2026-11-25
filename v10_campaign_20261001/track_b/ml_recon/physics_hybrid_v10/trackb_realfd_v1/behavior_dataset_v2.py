"""Dedicated source/evidence-bound V2 behavior dataset publication.

The generic V2.4 smoke/range writers remain closed to V2. This producer renders
the fixed explicit sample keys, retains the unchanged persisted field schema,
and publishes each validated byte payload through the caller's native owner.
It does not start training, admit learner views, or grant Gate/hardware authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot
from ml_recon.data_v2_4.contracts import NQubitDatasetContractV240
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_json
from ml_recon.data_v2_4.manifests import V24BuildManifest, V24ReadyManifest
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.schema_registry import schema_sha256
from ml_recon.data_v2_4.synthetic.generator import _render_dataset_family, _row_sha, _stack
from ml_recon.data_v2_4.validators import validate_global_multi, validate_split_groups, validate_view_keys

from .behavior_artifact_owner import Behavior_artifact_owner
from .behavior_dataset_bytes import encode_behavior_hdf5, encode_behavior_index
from .behavior_dataset_validation import behavior_validation_scope, validate_behavior_split
from .behavior_v2_contract import Behavior_stage_plan_v2
from .contracts import require_sha256
from .pilot_dataset_admission import Pilot_dataset_admission, admit_pilot_dataset
from .pilot_runtime_checks import require_unchanged_pilot_source
from .teacher_diagnostic_admission import Teacher_evidence_admission, require_unchanged_teacher_evidence
from .teacher_diagnostic_preflight import require_unchanged_teacher_diagnostic_preflight


dataset_prefix = "dataset"
progress_interval = 100


@dataclass(frozen=True)
class Behavior_dataset_result:
    admission: Pilot_dataset_admission
    receipt: Canonical_json_snapshot


def _render_split(
    keys: Sequence[V24SampleKey], cell: dict[str, Any], *, check_inputs: Callable[[], None],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Render one bounded split; no file writes or truth-to-input substitution."""
    if not keys:
        raise ValueError("behavior split cannot be empty")
    samples = []
    for index, key in enumerate(keys):
        if index % progress_interval == 0:
            check_inputs()
        samples.append(_render_dataset_family(key, cell=cell))
        if on_progress is not None and ((index + 1) % progress_interval == 0 or index + 1 == len(keys)):
            on_progress({"phase": "DATASET_RENDER", "split": key.split, "completed": index + 1,
                         "total": len(keys)})
    check_inputs()
    sample_ids = [key.sample_id for key in keys]
    observations = _stack([sample["observation"] for sample in samples], sample_ids)
    truths = _stack([sample["truth"] for sample in samples], sample_ids)
    validate_global_multi(observations["global_multi"], observations["global_multi_mask"], ["state"] * len(keys))
    validate_view_keys("feedback_student_base", observations)
    return observations, truths, samples


def publish_behavior_dataset(
    plan: Behavior_stage_plan_v2, owner: Behavior_artifact_owner,
    teacher_admission: Teacher_evidence_admission, *, expected_source: tuple[str, str],
    expected_plan_sha256: str, on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> Behavior_dataset_result:
    """Publish a new V2 dataset only inside the independently pinned live stage."""
    if type(plan) is not Behavior_stage_plan_v2 or type(owner) is not Behavior_artifact_owner:
        raise ValueError("behavior production requires the explicit stage plan and live owner")
    plan.validate_identity()
    require_sha256(expected_plan_sha256, "behavior_stage_plan_sha")
    if (plan.expected_source != expected_source or plan.sha256() != expected_plan_sha256
            or owner.root.path != plan.output_root):
        raise ValueError("behavior dataset differs from independently pinned source, plan or owner")
    require_unchanged_pilot_source(expected_source)
    owner.verify()
    borrowed_teacher_parent = (
        owner.parent_lease if owner.parent_lease.path == plan.teacher_evidence_pin.plan.trusted_output_parent else None)
    require_unchanged_teacher_evidence(
        teacher_admission, plan.teacher_evidence_pin, expected_source=expected_source,
        expected_plan_sha256=plan.teacher_evidence_pin.expected_plan_sha256,
        borrowed_parent=borrowed_teacher_parent)
    owner.verify()
    if dataset_prefix in owner.directories or plan.dataset_root.exists():
        raise ValueError("behavior dataset must be a fresh phase; no reuse")
    recipe = plan.recipe
    pin = plan.teacher_evidence_pin

    def check_source() -> None:
        require_unchanged_teacher_diagnostic_preflight(
            pin.plan.preflight_bindings[0], sample_key=pin.plan.sample_keys[0],
            config=pin.plan.config, expected_source=expected_source)

    def publish(relative: str, payload: bytes):
        return owner.publish(relative, payload, before_publish=check_source)

    cell = recipe.dataset_cell()
    build_id = "trackb_v2_" + sha256_json({"stage_plan_sha256": expected_plan_sha256,
                                          "dataset_recipe": recipe.dataset_recipe()})[:24]
    contract = NQubitDatasetContractV240.default()
    contract_sha = contract.sha256()
    schema_sha = schema_sha256()
    for relative in (dataset_prefix, "dataset/observations/synthetic", "dataset/truth/synthetic",
                     "dataset/index", "dataset/manifests", "dataset/reports"):
        owner.create_directory(relative, before_create=check_source)
    publish("dataset/CONTRACT.json", canonical_json_bytes(contract.to_dict()))
    shard_hashes: dict[str, str] = {}
    index_rows = []
    for split, keys in (("train", plan.train_keys), ("validation", plan.validation_keys)):
        observations, truths, samples = _render_split(
            keys, cell, check_inputs=check_source, on_progress=on_progress)
        validate_behavior_split(observations, truths, sample_keys=keys, recipe=recipe)
        check_source()
        metadata = {"provenance": "synthetic", "dataset_family": recipe.dataset_family,
                    "store": "observation", "cell_id": recipe.cell_id, "split": split,
                    "system_type": recipe.system_type, "N": recipe.mode_count,
                    "measurement_domain": recipe.measurement_domain, "schedule_id": "full_pauli",
                    "shot_mode": "fixed_per_setting", "storage_tier": "CORE",
                    "contract_sha": contract_sha, "build_id": build_id}
        observation_relative = f"observations/synthetic/{recipe.cell_id}_{split}.h5"
        truth_relative = f"truth/synthetic/{recipe.cell_id}_{split}.h5"
        for relative, arrays, shard_metadata in (
                (observation_relative, observations, metadata),
                (truth_relative, truths, {**metadata, "store": "truth", "storage_tier": "EXTENDED"})):
            payload = encode_behavior_hdf5(arrays, shard_metadata)
            binding = publish(f"dataset/{relative}", payload)
            shard_hashes[relative] = binding.sha256
        for row, (key, sample) in enumerate(zip(keys, samples)):
            identities = sample["identities"]
            index_rows.append({
                "sample_id": key.sample_id, "latent_state_id": identities["latent_state_id"],
                "transition_id": identities["transition_id"], "crn_group_id": identities["crn_group_id"],
                "session_id": "", "acquisition_run_id": "", "repeat_group_id": "", "trajectory_id": "",
                "provenance": "synthetic", "dataset_family": recipe.dataset_family, "cell_id": recipe.cell_id,
                "split": split, "system_type": recipe.system_type, "N": recipe.mode_count,
                "measurement_domain": recipe.measurement_domain, "observation_shard": observation_relative,
                "observation_row": row, "truth_shard": truth_relative, "truth_row": row,
                "sample_key_sha": key.sha256(), "row_content_sha": _row_sha(observations, truths, row),
                "contract_sha": contract_sha, "build_id": build_id,
                "derived_rho8": 0, "derived_count_mle": 0, "derived_classifier": 0, "derived_control": 0})
        del observations, truths, samples
    index_binding = publish("dataset/index/index.sqlite", encode_behavior_index(index_rows))
    manifest = V24BuildManifest(
        build_id=build_id, contract_sha=contract_sha, schema_sha=schema_sha, generator_sha=recipe.generator_sha,
        source_commit=expected_source[0], source_tree_sha=expected_source[1], global_seed=recipe.dataset_seed,
        cells=(recipe.dataset_recipe()["cell"],), generator_profile_id=recipe.generator_profile_id)
    manifest.validate()
    manifest_binding = publish(f"dataset/manifests/{build_id}.json", canonical_json_bytes(manifest.to_dict()))
    split_report = validate_split_groups(index_rows)
    if split_report["unique_samples"] != recipe.train_samples + recipe.validation_samples:
        raise ValueError("behavior dataset does not cover its full disjoint sample set")
    validation = {"build_id": build_id, "rows": len(index_rows), "split": split_report,
                  "truth_visibility": True, "physics_validators": True,
                  "behavior_validation_scope": behavior_validation_scope,
                  "classifier_checkpoint_sha": None, "control_forward_sha": None,
                  "research_only": True, "selection_evidence": False,
                  "promotion_ready": False, "hardware_action_authorized": False}
    validation_binding = publish(f"dataset/reports/{build_id}_validation.json", canonical_json_bytes(validation))
    ready = V24ReadyManifest(
        build_id=build_id, build_manifest_sha=manifest_binding.sha256, contract_sha=contract_sha,
        schema_sha=schema_sha, index_sha=index_binding.sha256,
        split_sha=sha256_json({row["sample_id"]: row["split"] for row in index_rows}),
        shard_hashes=shard_hashes, validation_report_sha=validation_binding.sha256,
        readiness_checks=("shard_hashes", "sqlite_integrity", "split_disjoint", "group_isolation",
                          "truth_visibility", "physics_validators"))
    ready.validate()
    ready_binding = publish("dataset/READY.json", canonical_json_bytes(ready.to_dict()))
    admission = admit_pilot_dataset(
        plan.dataset_root, build_id=build_id, cell_id=recipe.cell_id, history_policy=recipe.history_policy,
        history_policy_sha256=recipe.history_policy_sha256, strict_shard_audit=True)
    identity = admission.identity_dict()
    if (identity["ready_sha"] != ready_binding.sha256 or identity["source_commit"] != expected_source[0]
            or identity["source_tree"] != expected_source[1] or identity["rows"] != len(index_rows)):
        raise ValueError("strict admitted dataset differs from the newly published artifacts")
    for split, keys in (("train", plan.train_keys), ("validation", plan.validation_keys)):
        admitted = {row["sample_id"]: row["sample_key_sha"] for row in admission.rows_for_split(split)}
        if admitted != {key.sample_id: key.sha256() for key in keys}:
            raise ValueError("admitted dataset does not bind every independently planned sample key")
    require_unchanged_teacher_evidence(
        teacher_admission, plan.teacher_evidence_pin, expected_source=expected_source,
        expected_plan_sha256=plan.teacher_evidence_pin.expected_plan_sha256,
        borrowed_parent=borrowed_teacher_parent)
    receipt_payload = canonical_json_bytes({
        "schema": "TrackBBehaviorDatasetReceiptV2", "stage_plan_sha256": expected_plan_sha256,
        "source_commit": expected_source[0], "source_tree": expected_source[1],
        "recipe": recipe.dataset_recipe(), "dataset_recipe_sha256": sha256_json(recipe.dataset_recipe()),
        "teacher_evidence": teacher_admission.identity_summary(), "dataset_identity": identity,
        "ready_file_identity": ready_binding.to_dict(),
        "inventory": [binding.to_dict() for binding in owner.inventory()
                      if binding.relative_path.startswith("dataset/")],
        "strict_shard_audit": True, "derived_classifier_or_control_cache": "NOT_CREATED_OR_USED",
        "research_only": True, "training_authorized": False, "model_gain_measured": False,
        "scientific_gate": "NOT_PERFORMED", "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False})
    receipt_binding = publish("dataset/behavior_dataset_receipt.json", receipt_payload)
    return Behavior_dataset_result(admission, Canonical_json_snapshot(receipt_payload, receipt_binding.sha256))


__all__ = ["Behavior_dataset_result", "publish_behavior_dataset"]
