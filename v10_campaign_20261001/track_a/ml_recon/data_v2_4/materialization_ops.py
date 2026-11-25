"""High-level planning operations for persistent eval and ephemeral train shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .hashing import read_canonical_json, sha256_json, write_canonical_json
from .materialization import V24MaterializationProfileV1
from .rung_plan import V24CellAllocationV2, V24RungPlanV2
from .shard_producer import (
    V24ShardProducer,
    make_shard_plan,
    rows_per_shard_from_profile,
)
from .storage_profile import EMPIRICAL_STORAGE_PROFILE_REPORT_KIND
from .synthetic.negative_control_renderers import (
    PERSISTENT_NEGATIVE_CONTROL_KINDS,
    persistent_negative_controls_for_family,
)


LEGACY_EMPIRICAL_STORAGE_PROFILE_REPORT_KIND = "v24_empirical_storage_profile_v1"


def load_empirical_storage_profile(
    path: str | Path, *, expected_global_seed: int | None = None
) -> Tuple[Dict[str, Any], V24MaterializationProfileV1]:
    """Load V2 profiles or V1 profiles without fabricating seed provenance."""
    if (expected_global_seed is not None
            and type(expected_global_seed) is not int):
        raise ValueError("expected global_seed must be an integer")
    report = dict(read_canonical_json(path))
    profile_sha = str(report.pop("profile_sha", ""))
    if len(profile_sha) != 64 or sha256_json(report) != profile_sha:
        raise ValueError("empirical storage profile SHA mismatch")
    report_kind = report.get("report_kind")
    if report_kind == EMPIRICAL_STORAGE_PROFILE_REPORT_KIND:
        if "global_seed" not in report:
            raise ValueError("V2 empirical storage profile is missing global_seed")
        if type(report["global_seed"]) is not int:
            raise ValueError("V2 empirical storage profile global_seed must be an integer")
        if (expected_global_seed is not None
                and report["global_seed"] != expected_global_seed):
            raise ValueError("empirical storage profile global_seed mismatch")
    elif report_kind == LEGACY_EMPIRICAL_STORAGE_PROFILE_REPORT_KIND:
        if "global_seed" in report:
            raise ValueError("legacy V1 empirical storage profile cannot add a seed retroactively")
        if expected_global_seed is not None:
            raise ValueError("legacy V1 empirical storage profile has no global_seed")
    else:
        raise ValueError("unsupported empirical storage profile report kind")
    report["profile_sha"] = profile_sha
    profile = V24MaterializationProfileV1.from_dict(
        dict(report["materialization_profile"]))
    if profile.sha256() != report["materialization_profile_sha"]:
        raise ValueError("materialization profile identity mismatch")
    return report, profile


def _selected_cell(plan: V24RungPlanV2, cell_id: str) -> V24CellAllocationV2:
    matches = [cell for cell in plan.cell_allocations if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"rung plan does not contain exactly one cell {cell_id}")
    return matches[0]


def validate_range_training_plan_binding(
    *,
    identity: Mapping[str, Any],
    plan: V24RungPlanV2,
    storage_profile_path: str | Path,
    global_seed: int,
) -> Dict[str, str]:
    """Bind range-indexed train/eval shards to one rung and storage profile."""

    report, profile = load_empirical_storage_profile(storage_profile_path)
    if report["rung_plan_sha"] != plan.sha256():
        raise ValueError("storage profile belongs to another rung plan")
    train_build_id = "v24_" + sha256_json({
        "rung_plan_sha": plan.sha256(),
        "materialization_profile_sha": profile.sha256(),
        "global_seed": int(global_seed),
    })[:20]
    evaluation_build_id = "v24_eval_" + sha256_json({
        "rung_plan_sha": plan.sha256(),
        "storage_profile_sha": report["profile_sha"],
        "global_seed": int(global_seed),
    })[:20]
    if identity.get("split_build_ids") != {
        "train": [train_build_id],
        "validation": [evaluation_build_id],
    }:
        raise ValueError("range build IDs do not match rung/storage-profile binding")
    if identity.get("split_global_seeds") != {
        "train": [int(global_seed)],
        "validation": [int(global_seed)],
    }:
        raise ValueError("range materialization global seed mismatch")
    return {
        "storage_profile_sha": str(report["profile_sha"]),
        "materialization_profile_sha": profile.sha256(),
        "train_build_id": train_build_id,
        "evaluation_build_id": evaluation_build_id,
    }


def producer_from_plan(*, plan: V24RungPlanV2,
                       storage_profile_report: Mapping[str, Any],
                       materialization_profile: V24MaterializationProfileV1,
                       cell_id: str, split: str, local_root: str | Path,
                       global_seed: int, max_buffer_bytes: int,
                       free_bytes_provider=None) -> V24ShardProducer:
    plan.validate()
    cell = _selected_cell(plan, cell_id)
    if storage_profile_report["rung_plan_sha"] != plan.sha256():
        raise ValueError("storage profile belongs to another rung plan")
    profile_cells = {
        item["cell_id"]: item for item in storage_profile_report["cells"]}
    if cell_id not in profile_cells:
        raise ValueError("storage profile is missing the requested cell")
    empirical_bytes = int(round(
        profile_cells[cell_id]["compressed_observation_truth_bytes_per_sample"]))
    rows_per_shard = rows_per_shard_from_profile(
        max(empirical_bytes, 1), materialization_profile)
    count = cell.train_count if split == "train" else cell.validation_count
    if count <= 0:
        raise ValueError("selected split has zero allocated samples")
    build_id = "v24_" + sha256_json({
        "rung_plan_sha": plan.sha256(),
        "materialization_profile_sha": materialization_profile.sha256(),
        "global_seed": int(global_seed),
    })[:20]
    shard_plan = make_shard_plan(
        rung_plan_sha=plan.sha256(),
        materialization_profile_sha=materialization_profile.sha256(),
        build_id=build_id, global_seed=global_seed,
        cell=cell.to_generator_cell(), split=split,
        start_index=0, stop_index=count, rows_per_shard=rows_per_shard)
    return V24ShardProducer(
        local_root, shard_plan, materialization_profile,
        max_buffer_bytes=max_buffer_bytes,
        free_bytes_provider=free_bytes_provider)


def generate_persistent_eval(*, plan: V24RungPlanV2,
                             storage_profile_path: str | Path,
                             dataset_root: str | Path,
                             global_seed: int = 20261001,
                             free_bytes_provider=None) -> Dict[str, Any]:
    report, profile = load_empirical_storage_profile(storage_profile_path)
    if "validation" not in profile.persistent_splits:
        raise ValueError("materialization profile does not persist validation")
    root = Path(dataset_root)
    profile_cells = {item["cell_id"]: item for item in report["cells"]}
    build_id = "v24_eval_" + sha256_json({
        "rung_plan_sha": plan.sha256(), "storage_profile_sha": report["profile_sha"],
        "global_seed": global_seed,
    })[:20]
    manifest_path = root / "manifests" / f"{build_id}_persistent_eval.json"
    results = []
    split_identities: Dict[str, str] = {}
    range_index_paths: Dict[str, str] = {}
    for cell in plan.cell_allocations:
        evaluation_specs = [
            ("validation", cell.dataset_family, cell.cell_id, cell.validation_count),
            ("test", cell.dataset_family, cell.cell_id, cell.validation_count),
            ("audit", cell.dataset_family,
             f"{cell.cell_id}_audit", min(128, max(1, cell.validation_count))),
        ]
        if cell.dataset_family == "small_n_family":
            evaluation_specs.append((
                "locked_ood", f"{cell.dataset_family}_locked_ood",
                f"{cell.cell_id}_locked_ood",
                max(1, cell.validation_count // 5)))
        evaluation_specs.extend(
            ("negative_control",
             f"{cell.dataset_family}_negative_control_{kind}",
             f"{cell.cell_id}_negative_control_{kind}",
             max(1, cell.validation_count // 5))
            for kind in persistent_negative_controls_for_family(cell.dataset_family)
        )
        empirical_bytes = int(round(
            profile_cells[cell.cell_id]["compressed_observation_truth_bytes_per_sample"]))
        rows_per_shard = rows_per_shard_from_profile(max(empirical_bytes, 1), profile)
        for split, family, eval_cell_id, count in evaluation_specs:
            generator_cell = cell.to_generator_cell()
            generator_cell.update({"cell_id": eval_cell_id, "dataset_family": family})
            shard_plan = make_shard_plan(
                rung_plan_sha=plan.sha256(),
                materialization_profile_sha=profile.sha256(),
                build_id=build_id, global_seed=global_seed,
                cell=generator_cell, split=split,
                start_index=0, stop_index=count, rows_per_shard=rows_per_shard)
            producer = V24ShardProducer(
                root, shard_plan, profile, max_buffer_bytes=2**63 - 1,
                free_bytes_provider=free_bytes_provider)
            for shard_id, _, _ in producer.plan.shard_ranges():
                results.append(producer.produce_shard(shard_id))
            range_index_paths[f"{eval_cell_id}:{split}"] = (
                producer.range_index_path.relative_to(root).as_posix())
            split_identities[f"{eval_cell_id}:{split}"] = sha256_json([
                f"v2_4:{family}:{eval_cell_id}:{split}:N{cell.N}:{index:012d}"
                for index in range(count)
            ])
    from .synthetic.ood_audit import audit_persistent_locked_ood
    locked_ood_audits = []
    for cell in plan.cell_allocations:
        if cell.dataset_family != "small_n_family":
            continue
        locked_ood_audits.append(audit_persistent_locked_ood(
            root,
            validation_range_index_path=root / range_index_paths[
                f"{cell.cell_id}:validation"],
            locked_ood_range_index_path=root / range_index_paths[
                f"{cell.cell_id}_locked_ood:locked_ood"],
            validation_cell_id=cell.cell_id,
            locked_ood_cell_id=f"{cell.cell_id}_locked_ood"))
    manifest = {
        "manifest_kind": "v24_frozen_persistent_eval_v2",
        "build_id": build_id,
        "rung_plan_sha": plan.sha256(),
        "storage_profile_sha": report["profile_sha"],
        "materialization_profile_sha": profile.sha256(),
        "global_seed": global_seed,
        "split_identities": split_identities,
        "range_index_paths": range_index_paths,
        "locked_ood_audits": locked_ood_audits,
        "locked_ood_metrics_authoritative": bool(
            locked_ood_audits
            and all(item["metrics_authoritative"]
                    for item in locked_ood_audits)),
        "negative_control_kinds": list(PERSISTENT_NEGATIVE_CONTROL_KINDS),
        "negative_control_cells_by_kind": {
            kind: sorted(
                cell.cell_id for cell in plan.cell_allocations
                if kind in persistent_negative_controls_for_family(cell.dataset_family))
            for kind in PERSISTENT_NEGATIVE_CONTROL_KINDS
        },
        "negative_control_plan_sha": sha256_json(
            list(PERSISTENT_NEGATIVE_CONTROL_KINDS)),
        "persistent_eval_shards": len(results),
        "persistent_eval_rows": sum(int(item["row_count"]) for item in results),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if manifest_path.exists():
        existing = read_canonical_json(manifest_path)
        if existing != manifest:
            raise ValueError("frozen persistent eval manifest changed")
    manifest_sha = write_canonical_json(manifest_path, manifest)
    return {**manifest, "persistent_eval_manifest_sha": manifest_sha,
            "manifest_path": str(manifest_path)}
