"""Executable C_N_FULL_V10 training and registered loss assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ml_recon.data_v2_4.multi_cell import (
    parse_cell_keyed_paths as _parse_cell_path_map,
)

from .classifier_n_full import V10ClassifierNFull
from .contracts import V10MeasurementSchedule, sha256_of
from .loss_profiles import LOSS_WEIGHT_PROFILES, resolve_loss_weights
from .measurement_scheduler import V10MeasurementScheduler
from .operation_adapters import v10_rho8_contract
from .shot_budget import RUNG_SHOT_MODE_TO_V10, allocate_measurement_shots
from .sample_scaling import SAMPLE_RUNGS, validate_rung_authorization
from .structure_objective import masked_structure_logit_loss
from .synthetic_storage_policy import (
    Legacy_fixture_admission, require_legacy_fixture_mode, require_legacy_fixture_rows,
)
from .training_runtime import (
    V10Trainer, V10TrainingConfig, V10TrainingIdentity, resolve_range_mode,
)


CLASSIFIER_LOSS_WEIGHTS = LOSS_WEIGHT_PROFILES["default"]
CLASSIFIER_AMP_INITIAL_SCALE = 256.0
CLASSIFIER_ALIGNMENT_MARGIN = 0.02
# Materialized-path provenance must be hashed over the schedule the dataset was
# actually rendered with.  The V2.4 structure renderer hardcodes
# ``V10MeasurementScheduler(N).schedule(max_hyperedges=16)`` (see
# ``ml_recon/data_v2_4/synthetic/generator.py``, ``_render_structure``), so this
# constant mirrors that cap.  The scheduler default (128) coincides at N <= 5
# but registers 35 hyperedges instead of 16 at N >= 6, which would bind the
# provenance hash to a schedule the dataset was NOT rendered with.  Keep in
# sync with generator.py until the cap is centralized in measurement_scheduler.
RENDERER_MAX_HYPEREDGES = 16


def build_seeded_classifier(profile: str, seed: int) -> V10ClassifierNFull:
    """Bind initialization, not only subsequent minibatches, to the run seed."""
    if type(seed) is not int or seed < 0:
        raise ValueError("classifier initialization seed must be a nonnegative integer")
    torch.manual_seed(seed)
    return V10ClassifierNFull(profile=profile)


def mean_per_valid_setting(loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weight = valid_mask.to(loss.dtype)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.expand_as(loss).sum().clamp_min(1)

def safe_binary_cross_entropy(probability: torch.Tensor,
                              target: torch.Tensor,
                              *, reduction: str = "mean") -> torch.Tensor:
    with torch.autocast(device_type=probability.device.type, enabled=False):
        return F.binary_cross_entropy(probability.float(), target.float(), reduction=reduction)

def cap_rho8_weak_loss(rho8_loss: torch.Tensor, real_primary_count_objective: torch.Tensor,
                       maximum_fraction: float = 0.05) -> torch.Tensor:
    return torch.minimum(rho8_loss, maximum_fraction * real_primary_count_objective.detach())


def classifier_total_loss(
    losses: Dict[str, torch.Tensor], loss_weight_profile: str = "default",
) -> torch.Tensor:
    weights = resolve_loss_weights(loss_weight_profile)
    required = {"measurement", "denoise", "small_n_family", "coarse", "pair_map",
                "hyperedge", "partition", "depth", "intactness", "stretchability",
                "w_r1", "w_coherence", "w_sector",
                "operational_noise", "psd", "rho8_weak", "ood", "calibration",
                "setting_alignment"}
    missing = required - set(losses)
    if missing:
        raise ValueError(f"missing C_N losses: {sorted(missing)}")
    return sum(weights[name] * losses[name] for name in sorted(required))


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


def normalized_w_coherence_loss(
    prediction_re: torch.Tensor,
    prediction_im: torch.Tensor,
    target_re: torch.Tensor,
    target_im: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Match normalized off-diagonal W coherence with the registered mask.

    The existing ``w_r1`` objective fits raw density-matrix entries. The Gate
    metric evaluates their diagonal-normalized coherence, so this term aligns
    training with that metric without changing the physical target or gauge.
    """
    tensors = (prediction_re, prediction_im, target_re, target_im)
    if any(tensor.ndim != 3 or tensor.shape[-1] != tensor.shape[-2] for tensor in tensors):
        raise ValueError("W coherence tensors must have shape [batch, N, N]")
    if not (prediction_re.shape == prediction_im.shape == target_re.shape == target_im.shape):
        raise ValueError("W coherence tensors must have identical shapes")
    if valid_mask.dtype is not torch.bool or valid_mask.shape != prediction_re.shape[:1]:
        raise ValueError("W coherence mask must be Boolean with one value per sample")
    if prediction_re.shape[-1] < 2:
        raise ValueError("W coherence loss requires at least one off-diagonal pair")
    active = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
    if active.numel() == 0:
        safe_real = torch.nan_to_num(
            prediction_re.float(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_imag = torch.nan_to_num(
            prediction_im.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return (safe_real.sum() + safe_imag.sum()) * 0.0
    prediction_re = prediction_re.index_select(0, active)
    prediction_im = prediction_im.index_select(0, active)
    target_re = target_re.index_select(0, active)
    target_im = target_im.index_select(0, active)
    if not all(torch.isfinite(tensor).all() for tensor in (
            prediction_re, prediction_im, target_re, target_im)):
        raise ValueError("active W coherence tensors must be finite")
    with torch.autocast(device_type=prediction_re.device.type, enabled=False):
        predicted = torch.complex(prediction_re.float(), prediction_im.float())
        target = torch.complex(target_re.float(), target_im.float())
        predicted_diagonal = torch.diagonal(predicted.real, dim1=-2, dim2=-1)
        target_diagonal = torch.diagonal(target.real, dim1=-2, dim2=-1).clamp_min(1e-12)
        predicted_product = predicted_diagonal.unsqueeze(-1) * predicted_diagonal.unsqueeze(-2)
        if (predicted_diagonal < 0).any():
            raise ValueError("active predicted W coherence diagonal must be nonnegative")
        # Match V10WStructureHead exactly: sqrt first, then clamp the scale.
        # sqrt(max(x, eps**2)) is value-equivalent to max(sqrt(x), eps)
        # while removing the sqrt(0) backward singularity.
        coherence_epsilon = 1e-12
        predicted_scale = torch.sqrt(
            predicted_product.clamp_min(coherence_epsilon ** 2))
        target_scale = torch.sqrt(
            target_diagonal.unsqueeze(-1) * target_diagonal.unsqueeze(-2))
        off_diagonal = ~torch.eye(
            prediction_re.shape[-1], dtype=torch.bool, device=prediction_re.device)
        residual = (predicted / predicted_scale - target / target_scale).abs()
        per_sample = residual[:, off_diagonal].mean(-1)
        return per_sample.mean()


def _derange_alignment_descriptors(
    batch: Dict[str, object],
) -> tuple[Dict[str, object], bool]:
    """Break descriptor/value pairing while preserving every measured value."""

    modified = dict(batch)
    changed = False
    sample_ids = batch.get("sample_id")

    def _shift(row: int, length: int, field_name: str) -> int:
        if length < 2:
            return 0
        sample_id = (
            str(sample_ids[row])
            if isinstance(sample_ids, (list, tuple)) and row < len(sample_ids)
            else f"row_{row}"
        )
        digest = hashlib.sha256(
            f"v10_train_alignment_v1|{field_name}|{sample_id}".encode("utf-8")
        ).digest()
        return 1 + int.from_bytes(digest[:8], "big") % (length - 1)

    descriptors = batch.get("basis_descriptors")
    setting_mask = batch.get("setting_mask")
    if isinstance(descriptors, torch.Tensor) and isinstance(setting_mask, torch.Tensor):
        deranged = descriptors.clone()
        for row in range(descriptors.shape[0]):
            valid = torch.nonzero(setting_mask[row].bool(), as_tuple=False).reshape(-1)
            if valid.numel() >= 2:
                deranged[row, valid] = descriptors[
                    row, torch.roll(
                        valid,
                        shifts=_shift(row, valid.numel(), "basis_descriptors"),
                    )]
                changed = True
        modified["basis_descriptors"] = deranged

    pair_axes = batch.get("pair_stat_axes")
    pair_qubits = batch.get("pair_stat_qubits")
    pair_coverage = batch.get("pair_stat_coverage_mask")
    if (isinstance(pair_axes, torch.Tensor)
            and isinstance(pair_qubits, torch.Tensor)
            and isinstance(pair_coverage, torch.Tensor)):
        deranged_axes = pair_axes.clone()
        deranged_qubits = pair_qubits.clone()
        for row in range(pair_axes.shape[0]):
            valid = torch.nonzero(pair_coverage[row].bool(), as_tuple=False).reshape(-1)
            if valid.numel() >= 2:
                permutation = torch.roll(
                    valid,
                    shifts=_shift(row, valid.numel(), "pair_stat_descriptors"),
                )
                deranged_axes[row, valid] = pair_axes[row, permutation]
                deranged_qubits[row, valid] = pair_qubits[row, permutation]
                changed = True
        modified["pair_stat_axes"] = deranged_axes
        modified["pair_stat_qubits"] = deranged_qubits
    return modified, changed


def _alignment_objective(
    output: Dict[str, torch.Tensor | bool],
    batch: Dict[str, object],
    loss_weight_profile: str = "default",
) -> torch.Tensor:
    """Primary differentiable objective used by the train-only alignment margin."""

    required_targets = {
        "target_p_clean",
        "target_fine",
        "target_fine_mask",
        "target_fine_identifiable_mask",
        "target_pair",
        "target_pair_mask",
        "target_R1_re",
        "target_R1_im",
        "target_w_mask",
        "target_sector",
    }
    if not required_targets <= set(batch):
        return output["semantic_latent"].sum() * 0.0
    measurement = -(batch["target_p_clean"] * torch.log(
        output["p_clean"].clamp_min(1e-9))).sum(-1).mean()
    fine_raw = F.cross_entropy(
        output["fine_family_logits"], batch["target_fine"].long(),
        reduction="none")
    fine_mask = batch["target_fine_mask"].bool() \
        & batch["target_fine_identifiable_mask"].bool()
    family = _masked_mean(fine_raw, fine_mask)
    pair = masked_structure_logit_loss(
        output["pair_entanglement_logits"], batch["target_pair"], batch["target_pair_mask"])
    w_r1_raw = ((output["R1_re"] - batch["target_R1_re"]) ** 2
                + (output["R1_im"] - batch["target_R1_im"]) ** 2).mean((-1, -2))
    w_r1 = _masked_mean(w_r1_raw, batch["target_w_mask"])
    w_coherence = normalized_w_coherence_loss(
        output["R1_re"], output["R1_im"], batch["target_R1_re"], batch["target_R1_im"],
        batch["target_w_mask"])
    sector = torch.stack((output["p0"], output["p1"], output["p_ge2"]), -1)
    w_sector = _masked_mean(
        ((sector - batch["target_sector"]) ** 2).mean(-1),
        batch["target_w_mask"])
    weights = resolve_loss_weights(loss_weight_profile)
    return (
        weights["measurement"] * measurement
        + weights["small_n_family"] * family
        + weights["pair_map"] * pair
        + weights["w_r1"] * w_r1
        + weights["w_coherence"] * w_coherence
        + weights["w_sector"] * w_sector
    )


def classifier_registered_step(N: int, loss_weight_profile: str = "default"):
    """Complete mask-aware V2.4 classifier objective used by every run mode."""
    def step(model: torch.nn.Module, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        small_n = N in (3, 4)
        output = model(batch, N=N, fine_identifiable=small_n)
        zero = output["semantic_latent"].sum() * 0.0
        if small_n:
            measurement = -(batch["target_p_clean"] * torch.log(
                output["p_clean"].clamp_min(1e-9))).sum(-1).mean()
            denoise = F.mse_loss(output["p_clean"], batch["target_p_clean"])
            fine_raw = F.cross_entropy(
                output["fine_family_logits"], batch["target_fine"].long(),
                reduction="none")
            fine_mask = batch["target_fine_mask"].bool() \
                & batch["target_fine_identifiable_mask"].bool()
            small_n_family = _masked_mean(fine_raw, fine_mask)
        else:
            # No dense clean-law or fine-family truth exists for compact
            # general-N samples. Do not optimize their shape placeholders.
            measurement = zero
            denoise = zero
            small_n_family = zero
        pair_map = masked_structure_logit_loss(
            output["pair_entanglement_logits"], batch["target_pair"], batch["target_pair_mask"])
        hyperedge_raw = safe_binary_cross_entropy(
            output["hyperedge_entanglement_probs"], batch["target_hyperedge"],
            reduction="none")
        hyperedge = _masked_mean(
            hyperedge_raw, batch["target_hyperedge_mask"])
        partition = masked_structure_logit_loss(
            output["partition_link_logits"], batch["target_partition"], batch["target_partition_mask"])
        structure_mask = batch["target_structure_mask"].bool()
        depth = _masked_mean(F.nll_loss(
            torch.log(output["depth_posterior"].clamp_min(1e-9)),
            batch["target_depth"].long(), reduction="none"), structure_mask)
        intactness = _masked_mean(F.nll_loss(
            torch.log(output["intactness_posterior"].clamp_min(1e-9)),
            batch["target_intactness"].long(), reduction="none"), structure_mask)
        stretchability = _masked_mean(F.nll_loss(
            torch.log(output["stretchability_posterior"].clamp_min(1e-9)),
            batch["target_stretchability"].long(), reduction="none"), structure_mask)
        coarse_raw = F.nll_loss(
            torch.log(output["coarse_template_posterior"].clamp_min(1e-9)),
            batch["target_coarse"].long(), reduction="none")
        coarse = _masked_mean(coarse_raw, batch["target_coarse_mask"])
        w_raw = ((output["R1_re"] - batch["target_R1_re"]) ** 2
                 + (output["R1_im"] - batch["target_R1_im"]) ** 2).mean((-1, -2))
        w_r1 = _masked_mean(w_raw, batch["target_w_mask"])
        w_coherence = normalized_w_coherence_loss(
            output["R1_re"], output["R1_im"], batch["target_R1_re"], batch["target_R1_im"],
            batch["target_w_mask"])
        sector_prediction = torch.stack((output["p0"], output["p1"], output["p_ge2"]), -1)
        w_sector_raw = ((sector_prediction - batch["target_sector"]) ** 2).mean(-1)
        w_sector = _masked_mean(w_sector_raw, batch["target_w_mask"])
        operational_noise_raw = F.cross_entropy(
            output["operational_noise_logits"],
            batch["target_operational_noise"].long(), reduction="none")
        operational_noise = _masked_mean(
            operational_noise_raw, batch["target_operational_noise_mask"])
        complex_r1 = torch.complex(output["R1_re"], output["R1_im"])
        psd = torch.relu(-torch.linalg.eigvalsh(complex_r1).real).mean()
        rho8_weak = zero
        ood = safe_binary_cross_entropy(output["OOD_probability"],
                                     batch["target_ood"])
        if small_n:
            one_hot = F.one_hot(
                batch["target_fine"].long(),
                output["fine_family_probs"].shape[-1]).to(
                    output["fine_family_probs"].dtype)
            calibration = _masked_mean(
                ((output["fine_family_probs"] - one_hot) ** 2).mean(-1),
                fine_mask)
            deranged_batch, alignment_applicable = _derange_alignment_descriptors(
                batch)
            if alignment_applicable:
                deranged_output = model(
                    deranged_batch, N=N, fine_identifiable=True)
                setting_alignment = F.relu(
                    CLASSIFIER_ALIGNMENT_MARGIN
                    + _alignment_objective(output, batch, loss_weight_profile)
                    - _alignment_objective(deranged_output, batch, loss_weight_profile)
                )
            else:
                setting_alignment = zero
        else:
            calibration = zero
            setting_alignment = zero
        components = {
            "measurement": measurement,
            "denoise": denoise,
            "small_n_family": small_n_family,
            "pair_map": pair_map,
            "hyperedge": hyperedge,
            "partition": partition,
            "depth": depth,
            "intactness": intactness,
            "stretchability": stretchability,
            "coarse": coarse,
            "w_r1": w_r1,
            "w_coherence": w_coherence,
            "w_sector": w_sector,
            "operational_noise": operational_noise,
            "psd": psd,
            "rho8_weak": rho8_weak,
            "ood": ood,
            "calibration": calibration,
            "setting_alignment": setting_alignment,
        }
        total = classifier_total_loss(components, loss_weight_profile)
        return {"loss": total, **components}
    return step


def classifier_registered_multi_cell_step(
    cell_N: Dict[str, int],
    loss_weight_profile: str = "default",
):
    """Mask-aware V2.4 classifier objective for homogeneous multi-cell jobs."""

    def step(model: torch.nn.Module, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        cell_id = str(batch["cell_id"][0])
        if cell_id not in cell_N:
            raise ValueError(f"training batch references unknown cell: {cell_id}")
        return classifier_registered_step(
            cell_N[cell_id], loss_weight_profile)(model, batch)
    return step


def _git_identity() -> tuple[str, str, str]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"), text=True).strip()
    patch = subprocess.check_output(("git", "diff", "--binary", "HEAD"))
    return commit, tree, sha256_of(patch)


def _git_branch() -> str:
    branch = subprocess.check_output(
        ("git", "branch", "--show-current"), text=True).strip()
    return branch or "detached:" + subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True).strip()


def _build_observers(model, output_dir: str, run_mode: str):
    """Build training observers.  Gradient audit runs on all non-smoke modes."""
    observers = []
    try:
        from .classifier_output_audit import Classifier_output_audit
        from .gradient_audit import V10GradientAuditConfig, V10GradientAuditObserver
        config = V10GradientAuditConfig(
            enabled=True,
            interval_steps=20,
            first_step=True,
            final_step=True,
        )
        shared_params = list(model.shared_parameters_for_audit())
        observers.append(V10GradientAuditObserver(
            config=config, shared_parameters=shared_params,
            output_dir=Path(output_dir)))
        observers.append(Classifier_output_audit(model, output_dir))
    except Exception as exc:
        observers = []
        if run_mode != "smoke":
            raise RuntimeError("required gradient audit observer initialization failed") from exc
    return observers


def renderer_bound_schedule(N: int) -> V10MeasurementSchedule:
    """Typed schedule bound to the data renderer's hyperedge cap.

    Materialized datasets are rendered with
    ``schedule(max_hyperedges=RENDERER_MAX_HYPEREDGES)``; provenance hashing on
    this path must use exactly that schedule, never the scheduler default.
    """
    return V10MeasurementScheduler(N).typed_schedule(
        max_hyperedges=RENDERER_MAX_HYPEREDGES)


def run_classifier_training(
    *,
    N: Optional[int] = None,
    output_dir: str,
    run_mode: str = "smoke",
    seed: int = 20261001,
    device: str = "cpu",
    dataset_root: str = "",
    dataset_build_id: str = "",
    dataset_cell: str = "",
    task_pool: str = "",
    cells: tuple[str, ...] = (),
    derived_checkpoint_sha: str = "",
    training_authorized: bool = False,
    phase_1_to_7_gate_sha: str = "",
    phase_8_gate_sha: str = "",
    sample_rung: Optional[int] = None,
    rung_plan_path: str = "",
    storage_profile_path: str = "",
    train_range_index_path: str = "",
    validation_range_index_path: str = "",
    train_range_index_paths: Optional[Dict[str, str]] = None,
    validation_range_index_paths: Optional[Dict[str, str]] = None,
    completed_sample_rungs: tuple[int, ...] = (),
    track_id: str = "unspecified",
    run_id: str = "",
    fixture_admission: Legacy_fixture_admission | None = None,
) -> Dict[str, object]:
    fixture_admission = require_legacy_fixture_mode(
        run_mode, fixture_admission=fixture_admission,
        output_dir=output_dir, dataset_root=dataset_root,
        artifact_paths=(rung_plan_path, storage_profile_path,
                        train_range_index_path, validation_range_index_path,
                        *(train_range_index_paths or {}).values(),
                        *(validation_range_index_paths or {}).values()))
    from ml_recon.data_v2_4.loaders import V24TrainingDataset, dataset_training_identity
    from ml_recon.data_v2_4.synthetic.generator import DEFAULT_SMOKE_PLAN, generate_smoke_build

    if not dataset_root:
        planned_rows = 6 * len(cells) if cells else sum(
            int(cell["train"]) + int(cell["validation"]) for cell in DEFAULT_SMOKE_PLAN)
        require_legacy_fixture_rows(fixture_admission, planned_rows)

    if cells and task_pool not in ("small_n_classifier", "general_n_structure"):
        raise ValueError(
            "classifier multi-cell training requires task_pool "
            "small_n_classifier or general_n_structure")
    if cells and len(set(cells)) != len(cells):
        raise ValueError("classifier multi-cell cell ids must be unique")
    if cells and dataset_cell:
        raise ValueError("--cells and --dataset-cell are mutually exclusive")
    if sample_rung is None:
        if run_mode == "full":
            raise ValueError("full training requires --sample-rung 2000000")
        if run_mode == "pilot" and not dataset_root:
            raise ValueError("pilot training requires a READY V2.4 --dataset-root")
    else:
        if sample_rung not in SAMPLE_RUNGS:
            raise ValueError(f"sample_rung must be one of {SAMPLE_RUNGS}")
        expected_mode = "full" if sample_rung == SAMPLE_RUNGS[-1] else "pilot"
        if run_mode != expected_mode:
            raise ValueError(
                f"sample rung {sample_rung} requires run_mode={expected_mode}")
        validate_rung_authorization(
            completed_sample_rungs, sample_rung,
            training_authorized=training_authorized,
            phase_1_to_7_gate_sha=phase_1_to_7_gate_sha,
            phase_8_gate_sha=phase_8_gate_sha)
    smoke_cell_plan: tuple[Dict[str, object], ...] = ()
    range_mode = resolve_range_mode(
        train_range_index_path, validation_range_index_path,
        train_range_index_paths, validation_range_index_paths)
    if not dataset_root:
        if range_mode:
            raise ValueError("range training requires an explicit --dataset-root")
        elif cells:
            dataset_root = str(Path(output_dir) / "v2_4_fixture")
            from ml_recon.data_v2_4.rung_plan import registered_task_cells
            pool = {
                str(cell["cell_id"]): dict(cell)
                for cell in registered_task_cells(task_pool)
            }
            missing = [cell_id for cell_id in cells if cell_id not in pool]
            if missing:
                raise ValueError(
                    f"cells are not registered in task pool {task_pool}: {missing}")
            plan = []
            for cell_id in cells:
                cell = dict(pool[cell_id])
                cell["train"] = 4
                cell["validation"] = 2
                plan.append(cell)
            smoke_cell_plan = tuple(plan)
            generated = generate_smoke_build(
                dataset_root, global_seed=seed, plan=smoke_cell_plan)
        else:
            dataset_root = str(Path(output_dir) / "v2_4_fixture")
            generated = generate_smoke_build(dataset_root, global_seed=seed)
        if not range_mode:
            dataset_build_id = str(generated["build_id"])
    if range_mode and not storage_profile_path:
        raise ValueError("range training requires --storage-profile")
    if not dataset_build_id:
        if not (cells and range_mode):
            raise ValueError("training requires --dataset-build-id")
    cell_N: Dict[str, int] = {}
    if cells:
        from ml_recon.data_v2_4.multi_cell import (
            V24CellHomogeneousBatchSampler,
            V24MultiCellRangeDataset,
        )
        cell_bundles = []
        common_identity: Dict[str, str] = {}
        dataset_hashes: Dict[str, str] = {}
        if range_mode:
            if not rung_plan_path:
                raise ValueError("range training requires --rung-plan")
            from ml_recon.data_v2_4.ephemeral_loader import (
                V24EphemeralRangeDataset,
                range_training_identity,
            )
            from ml_recon.data_v2_4.materialization_ops import (
                validate_range_training_plan_binding,
            )
            from ml_recon.data_v2_4.rung_plan import read_rung_plan
            rung_plan = read_rung_plan(rung_plan_path)
            from ml_recon.data_v2_4.multi_cell import resolve_cell_keyed_paths
            train_index_paths = resolve_cell_keyed_paths(
                cells, legacy_path=train_range_index_path,
                cell_paths=train_range_index_paths)
            validation_index_paths = resolve_cell_keyed_paths(
                cells, legacy_path=validation_range_index_path,
                cell_paths=validation_range_index_paths)
            missing_indexes = [
                cell_id for cell_id in cells
                if not train_index_paths.get(cell_id)
                or not validation_index_paths.get(cell_id)]
            if missing_indexes:
                raise ValueError(
                    "multi-cell classifier range training requires per-cell "
                    f"range indexes; missing {missing_indexes}")
            combined_build_id = "v24_multi_cell_classifier_" + "0" * 16
            build_parts = {}
            index_parts = {}
            manifest_parts = {}
            source_commit, source_tree, _ = _git_identity()
            for cell_id in cells:
                range_identity = range_training_identity(
                    dataset_root,
                    train_range_index_path=train_index_paths[cell_id],
                    validation_range_index_path=validation_index_paths[cell_id],
                    cell_id=cell_id)
                validate_range_training_plan_binding(
                    identity=range_identity,
                    plan=rung_plan,
                    storage_profile_path=storage_profile_path,
                    global_seed=seed,
                )
                if not common_identity:
                    common_identity = {
                        "contract_sha": range_identity["contract_sha"],
                        "schema_sha": range_identity["schema_sha"],
                        "generator_sha": range_identity["generator_sha"],
                        "build_manifest_sha": range_identity["build_manifest_sha"],
                        "split_sha": range_identity["split_sha"],
                        "source_commit": source_commit,
                        "source_tree": source_tree,
                    }
                build_parts[cell_id] = range_identity["build_id"]
                index_parts[cell_id] = range_identity["index_sha"]
                manifest_parts[cell_id] = range_identity["build_manifest_sha"]
                from ml_recon.data_v2_4.range_index import read_range_index
                cell_N[cell_id] = int(next(
                    entry.N for entry in read_range_index(train_index_paths[cell_id])
                    if entry.cell_id == cell_id and entry.split == "train"))
                train_dataset = fixture_admission.own_dataset(V24EphemeralRangeDataset(
                    dataset_root, train_index_paths[cell_id], cell_id=cell_id,
                    split="train"))
                validation_dataset = fixture_admission.own_dataset(V24EphemeralRangeDataset(
                    dataset_root, validation_index_paths[cell_id], cell_id=cell_id,
                    split="validation"))
                cell_bundles.append({
                    "cell_id": cell_id,
                    "train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "train_identity": {
                        **common_identity,
                        "cell_split_sha": range_identity["cell_split_hashes"]["train"],
                    },
                    "validation_identity": {
                        **common_identity,
                        "cell_split_sha": range_identity["cell_split_hashes"]["validation"],
                    },
                })
                dataset_hashes[f"train:{cell_id}"] = (
                    range_identity["cell_split_hashes"]["train"])
                dataset_hashes[f"validation:{cell_id}"] = (
                    range_identity["cell_split_hashes"]["validation"])
            combined_build_id = "v24_multi_cell_classifier_" + sha256_of(build_parts)[:16]
            if dataset_build_id and dataset_build_id != combined_build_id:
                raise ValueError("multi-cell range build ID mismatch")
            dataset_build_id = combined_build_id
            combined_index_sha = sha256_of(index_parts)
            combined_manifest_sha = sha256_of(manifest_parts)
        else:
            if not smoke_cell_plan:
                from ml_recon.data_v2_4.rung_plan import registered_task_cells
                pool = {
                    str(cell["cell_id"]): dict(cell)
                    for cell in registered_task_cells(task_pool)
                }
                for cell_id in cells:
                    cell_N[cell_id] = int(pool[cell_id]["N"])
            else:
                for cell in smoke_cell_plan:
                    cell_N[str(cell["cell_id"])] = int(cell["N"])
            source_commit, source_tree, _ = _git_identity()
            for cell_id in cells:
                train_dataset = fixture_admission.own_dataset(V24TrainingDataset(
                    dataset_root, build_id=dataset_build_id, cell_id=cell_id,
                    split="train", task="classifier",
                    derived_checkpoint_sha=derived_checkpoint_sha))
                validation_dataset = fixture_admission.own_dataset(V24TrainingDataset(
                    dataset_root, build_id=dataset_build_id, cell_id=cell_id,
                    split="validation", task="classifier",
                    derived_checkpoint_sha=derived_checkpoint_sha))
                train_identity = dataset_training_identity(
                    dataset_root, build_id=dataset_build_id,
                    cell_id=cell_id, split="train")
                validation_identity = dataset_training_identity(
                    dataset_root, build_id=dataset_build_id,
                    cell_id=cell_id, split="validation")
                if not common_identity:
                    common_identity = {
                        "contract_sha": train_identity["contract_sha"],
                        "schema_sha": train_identity["schema_sha"],
                        "generator_sha": train_identity["generator_sha"],
                        "build_manifest_sha": train_identity["build_manifest_sha"],
                        "split_sha": train_identity["split_sha"],
                        "index_sha": train_identity["index_sha"],
                        "source_commit": source_commit,
                        "source_tree": source_tree,
                    }
                cell_bundles.append({
                    "cell_id": cell_id,
                    "train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "train_identity": {
                        **common_identity,
                        "cell_split_sha": train_identity["cell_split_sha"],
                    },
                    "validation_identity": {
                        **common_identity,
                        "cell_split_sha": validation_identity["cell_split_sha"],
                    },
                })
                dataset_hashes[f"train:{cell_id}"] = train_identity["cell_split_sha"]
                dataset_hashes[f"validation:{cell_id}"] = (
                    validation_identity["cell_split_sha"])
            combined_index_sha = common_identity["index_sha"]
            combined_manifest_sha = ""
        if N is not None and N not in set(cell_N.values()):
            raise ValueError(
                f"--N {N} does not match any requested cell N values "
                f"{sorted(set(cell_N.values()))}")
        train_dataset = fixture_admission.own_dataset(V24MultiCellRangeDataset(
            [bundle["train_dataset"] for bundle in cell_bundles], cells))
        validation_dataset = fixture_admission.own_dataset(V24MultiCellRangeDataset(
            [bundle["validation_dataset"] for bundle in cell_bundles], cells))
        train_identity = {
            **common_identity,
            "index_sha": combined_index_sha,
            "build_manifest_sha": (
                combined_manifest_sha or common_identity["build_manifest_sha"]),
            "split_sha": sha256_of({
                cell_id: {
                    "train": dataset_hashes[f"train:{cell_id}"],
                    "validation": dataset_hashes[f"validation:{cell_id}"],
                }
                for cell_id in cells}),
            "cell_split_hashes": dataset_hashes,
        }
        validation_identity = dict(train_identity)
    elif range_mode:
        from ml_recon.data_v2_4.ephemeral_loader import (
            V24EphemeralRangeDataset,
            range_training_identity,
        )
        if N is None:
            N = 3
        dataset_cell = dataset_cell or f"classifier_n{N}"
        range_identity = range_training_identity(
            dataset_root,
            train_range_index_path=train_range_index_path,
            validation_range_index_path=validation_range_index_path,
            cell_id=dataset_cell,
        )
        if not rung_plan_path:
            raise ValueError("range training requires --rung-plan")
        from ml_recon.data_v2_4.materialization_ops import (
            validate_range_training_plan_binding,
        )
        from ml_recon.data_v2_4.rung_plan import read_rung_plan
        validate_range_training_plan_binding(
            identity=range_identity,
            plan=read_rung_plan(rung_plan_path),
            storage_profile_path=storage_profile_path,
            global_seed=seed,
        )
        if dataset_build_id != range_identity["build_id"]:
            raise ValueError("range materialization build ID mismatch")
        train_dataset = fixture_admission.own_dataset(V24EphemeralRangeDataset(
            dataset_root, train_range_index_path,
            cell_id=dataset_cell, split="train"))
        validation_dataset = fixture_admission.own_dataset(V24EphemeralRangeDataset(
            dataset_root, validation_range_index_path,
            cell_id=dataset_cell, split="validation"))
        source_commit, source_tree, _ = _git_identity()
        common_identity = {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "contract_sha": range_identity["contract_sha"],
            "schema_sha": range_identity["schema_sha"],
            "generator_sha": range_identity["generator_sha"],
            "build_manifest_sha": range_identity["build_manifest_sha"],
            "index_sha": range_identity["index_sha"],
            "split_sha": range_identity["split_sha"],
        }
        train_identity = {
            **common_identity,
            "cell_split_sha": range_identity["cell_split_hashes"]["train"],
        }
        validation_identity = {
            **common_identity,
            "cell_split_sha": range_identity["cell_split_hashes"]["validation"],
        }
    else:
        if N is None:
            N = 3
        dataset_cell = dataset_cell or f"classifier_n{N}"
        train_dataset = fixture_admission.own_dataset(V24TrainingDataset(
            dataset_root, build_id=dataset_build_id, cell_id=dataset_cell,
            split="train", task="classifier",
            derived_checkpoint_sha=derived_checkpoint_sha))
        validation_dataset = fixture_admission.own_dataset(V24TrainingDataset(
            dataset_root, build_id=dataset_build_id, cell_id=dataset_cell,
            split="validation", task="classifier",
            derived_checkpoint_sha=derived_checkpoint_sha))
        train_identity = dataset_training_identity(
            dataset_root, build_id=dataset_build_id,
            cell_id=dataset_cell, split="train")
        validation_identity = dataset_training_identity(
            dataset_root, build_id=dataset_build_id,
            cell_id=dataset_cell, split="validation")
    train_count = len(train_dataset)
    require_legacy_fixture_rows(fixture_admission, train_count + len(validation_dataset))
    registered_rung_plan_sha = ""
    global_sample_count = 0
    if rung_plan_path:
        from ml_recon.data_v2_4.rung_plan import (
            read_rung_plan,
            validate_registered_multi_cell_materialization,
            validate_registered_cell_materialization,
        )
        rung_plan = read_rung_plan(rung_plan_path)
        if sample_rung is None and rung_plan.total_train_samples != 1_000:
            raise ValueError("only the 1k engineering pilot may omit --sample-rung")
        requested_global_count = sample_rung or 1_000
        if cells:
            validate_registered_multi_cell_materialization(
                rung_plan,
                expected_task_pool=task_pool,
                requested_sample_rung=requested_global_count,
                cell_ids=cells,
                train_counts={
                    bundle["cell_id"]: len(bundle["train_dataset"])
                    for bundle in cell_bundles},
                validation_counts={
                    bundle["cell_id"]: len(bundle["validation_dataset"])
                    for bundle in cell_bundles},
            )
        else:
            validate_registered_cell_materialization(
                rung_plan,
                expected_task_pool=task_pool or "small_n_classifier",
                requested_sample_rung=requested_global_count,
                cell_id=dataset_cell,
                train_count=train_count,
                validation_count=len(validation_dataset),
            )
        registered_rung_plan_sha = rung_plan.sha256()
        global_sample_count = rung_plan.total_train_samples
    elif sample_rung is not None:
        raise ValueError("registered sample rung requires --rung-plan")
    if sample_rung is not None:
        from ml_recon.data_v2_4.gate_artifact import validate_gate_artifact
        validate_gate_artifact(
            phase_1_to_7_gate_sha, requested_rung=sample_rung,
            expected_kind="phase_1_to_7", expected_dependencies={
                "source_commit": train_identity["source_commit"],
                "source_tree": train_identity["source_tree"],
                "contract_sha": train_identity["contract_sha"],
                "schema_sha": train_identity["schema_sha"],
                "build_program_sha": train_identity["generator_sha"],
            })
    if cells:
        # When range mode, derive shot_mode/budget from each cell's rung
        # allocation instead of hard-coding classifier defaults.
        if range_mode:
            _allocs = {cell.cell_id: cell for cell in rung_plan.cell_allocations}
        schedule_hashes = {
            cell_id: renderer_bound_schedule(cell_N[cell_id]).sha256()
            for cell_id in cells}
        shot_budget_hashes = {}
        for cell_id in cells:
            if range_mode:
                alloc = _allocs[cell_id]
                shot_mode = RUNG_SHOT_MODE_TO_V10[alloc.shot_mode]
                budget = alloc.shot_budget
            else:
                shot_mode, budget = "fixed_per_setting", 64
            shot_budget_hashes[cell_id] = allocate_measurement_shots(
                renderer_bound_schedule(cell_N[cell_id]).setting_ids,
                shot_mode, budget).sha256()
        measurement_contract_sha = sha256_of({"cells": schedule_hashes})
        shot_budget_contract_sha = sha256_of({"cells": shot_budget_hashes})
        schedule = None
        shot_budget = None
    else:
        schedule = renderer_bound_schedule(N)
        shot_budget = allocate_measurement_shots(
            schedule.setting_ids, "fixed_per_setting", 64)
        measurement_contract_sha = schedule.sha256()
        shot_budget_contract_sha = shot_budget.sha256()
    config = V10TrainingConfig(
        model_id="C_N_FULL_V10", output_dir=output_dir, run_mode=run_mode,
        track_id=track_id, run_id=run_id,
        global_seed=seed, seed_panel=(seed,), batch_size=2 if run_mode == "smoke" else 16,
        epochs=1 if run_mode == "smoke" else 12,
        amp_initial_scale=CLASSIFIER_AMP_INITIAL_SCALE,
        max_train_batches=2 if run_mode == "smoke" else None,
        max_validation_batches=1 if run_mode == "smoke" else None,
        training_authorized=training_authorized,
        phase_1_to_7_gate_sha=phase_1_to_7_gate_sha,
        phase_8_gate_sha=phase_8_gate_sha,
        sample_count=train_count,
        global_sample_count=global_sample_count,
        rung_plan_sha=registered_rung_plan_sha,
        completed_sample_rungs=completed_sample_rungs)
    commit, tree, dirty_patch_sha = _git_identity()
    identity = V10TrainingIdentity(
        branch=_git_branch(), source_commit=commit, source_tree_sha=tree,
        dirty_patch_sha=dirty_patch_sha,
        v23_manifest_sha=sha256_of({"source_contract_version": "2.3"}),
        synthetic_manifest_sha=train_identity["build_manifest_sha"],
        measurement_contract_sha=measurement_contract_sha,
        shot_budget_contract_sha=shot_budget_contract_sha,
        rho8_sha=sha256_of(v10_rho8_contract()),
        config_sha=sha256_of(asdict(config)),
        dataset_hashes=(train_identity["cell_split_hashes"]
                        if cells else {"train": train_identity["cell_split_sha"],
                                       "validation": validation_identity["cell_split_sha"]}),
        split_hash=train_identity["split_sha"],
        dataset_contract_sha=train_identity["contract_sha"],
        dataset_build_id=dataset_build_id,
        dataset_index_sha=train_identity["index_sha"],
        dataset_cell_id=",".join(cells) if cells else dataset_cell)
    model = build_seeded_classifier("C-TEST" if run_mode == "smoke" else "C-S", seed)
    if cells:
        step = classifier_registered_multi_cell_step(cell_N)
        def prediction_function(network, batch):
            return network(
                batch,
                N=cell_N[str(batch["cell_id"][0])],
                fine_identifiable=True,
            )

        def batch_sampler_factory(dataset, epoch, shuffle):
            return V24CellHomogeneousBatchSampler(
                dataset, batch_size=config.batch_size, shuffle=shuffle,
                seed=seed + 1_000_003 * epoch)
    else:
        step = classifier_registered_step(N)
        def prediction_function(network, batch):
            return network(batch, N=N, fine_identifiable=True)

        batch_sampler_factory = None
    trainer = V10Trainer(
        model=model, config=config, identity=identity, step_function=step,
        prediction_function=prediction_function, device=device,
        batch_sampler_factory=batch_sampler_factory,
        observers=_build_observers(model, output_dir, run_mode))
    summary = trainer.fit(train_dataset, validation_dataset)
    summary["task_pool"] = task_pool or "small_n_classifier"
    summary["cells"] = cells
    summary["cell_N"] = cell_N
    if cells:
        train_dataset.close()
        validation_dataset.close()
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Executable V10 classifier training")
    parser.add_argument("--N", type=int, default=None,
                        choices=(3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256))
    parser.add_argument("--run-mode", default="smoke", choices=("smoke", "pilot", "full"))
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="ml_recon/reports/physics_hybrid_v10/classifier_smoke")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset-build-id", default="")
    parser.add_argument("--dataset-cell", default="")
    parser.add_argument("--task-pool", default="",
                        choices=("small_n_classifier", "general_n_structure"))
    parser.add_argument("--cells", default="")
    parser.add_argument("--derived-checkpoint-sha", default="")
    parser.add_argument("--training-authorized", action="store_true")
    parser.add_argument("--phase-1-to-7-gate", "--phase-1-to-7-gate-sha",
                        dest="phase_1_to_7_gate_sha", default="")
    parser.add_argument("--phase-8-gate", "--phase-8-gate-sha",
                        dest="phase_8_gate_sha", default="")
    parser.add_argument("--sample-rung", type=int, choices=SAMPLE_RUNGS)
    parser.add_argument("--rung-plan", default="")
    parser.add_argument("--storage-profile", default="")
    parser.add_argument("--train-range-index", action="append", default=[])
    parser.add_argument("--validation-range-index", action="append", default=[])
    parser.add_argument("--completed-sample-rung", action="append", type=int,
                        choices=SAMPLE_RUNGS, default=[])
    args = parser.parse_args(argv)
    from .synthetic_storage_policy import reject_materialized_cli
    reject_materialized_cli(parser)
    cells = tuple(cell for cell in args.cells.split(",") if cell)
    train_index_map = _parse_cell_path_map(args.train_range_index)
    validation_index_map = _parse_cell_path_map(args.validation_range_index)
    train_index_paths = {
        cell_id: path for cell_id, path in train_index_map.items() if cell_id}
    validation_index_paths = {
        cell_id: path for cell_id, path in validation_index_map.items() if cell_id}
    summary = run_classifier_training(
        N=args.N, output_dir=args.output_dir, run_mode=args.run_mode, seed=args.seed,
        device=args.device, dataset_root=args.dataset_root,
        dataset_build_id=args.dataset_build_id, dataset_cell=args.dataset_cell,
        task_pool=args.task_pool, cells=cells,
        derived_checkpoint_sha=args.derived_checkpoint_sha,
        training_authorized=args.training_authorized,
        phase_1_to_7_gate_sha=args.phase_1_to_7_gate_sha,
        phase_8_gate_sha=args.phase_8_gate_sha,
        sample_rung=args.sample_rung,
        rung_plan_path=args.rung_plan,
        storage_profile_path=args.storage_profile,
        train_range_index_path=train_index_map.get("", ""),
        validation_range_index_path=validation_index_map.get("", ""),
        train_range_index_paths=train_index_paths,
        validation_range_index_paths=validation_index_paths,
        completed_sample_rungs=tuple(args.completed_sample_rung))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
