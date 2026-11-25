"""Executable eight-stage V6 joint feedback research trainer.

Run with ``python -m ml_recon.physics_hybrid_v5.train_joint_feedback``.  The
entry point consumes compact immutable command tables only; it cannot upload,
collect, stage a hardware bundle, or promote a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .action_conditioned_model import command_tensor_batch
from .action_optimizer import ConstrainedInverseTeacher, InverseTeacherConfig
from .command_features import command_tone_features
from .joint_dataset import JointFeedbackDataset
from .joint_training import RESEARCH_CURRICULUM_STAGES, build_research_feedback_model
from .projective_action import ACTION_GEOMETRY_SCHEMA, ACTION_GEOMETRY_VERSION
from .runtime_checkpoint import (
    build_joint_checkpoint_metadata,
    canonical_sha256,
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from .source_multiplicity import SOURCE_MULTIPLICITY_SCHEMA

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass(frozen=True)
class CurriculumStageSpec:
    name: str
    trainable_prefixes: Tuple[str, ...]
    objective: str


CURRICULUM_STAGE_SPECS: Tuple[CurriculumStageSpec, ...] = (
    CurriculumStageSpec("command_aod_forward_pretrain", ("count_forward_model.command_encoder", "count_forward_model.write_forward", "count_forward_model.read_forward"), "state"),
    CurriculumStageSpec("write_read_born_count_pretrain", ("count_forward_model",), "count"),
    CurriculumStageSpec("source_detector_nuisance_calibration", ("count_forward_model.source_model", "count_forward_model.detector_model"), "nuisance"),
    CurriculumStageSpec("joint_signed_probe_phase_adaptation", ("count_forward_model",), "phase_probe"),
    CurriculumStageSpec("constrained_inverse_teacher_generation", (), "teacher_generation"),
    CurriculumStageSpec("controller_distillation", ("trunk", "policy_", "amplitude_coordinate_decoder", "phase_coordinate_decoder"), "distillation"),
    CurriculumStageSpec("real_count_adapter", ("count_forward_model",), "count_adapter"),
    CurriculumStageSpec("uq_conformal_calibration", ("uncertainty_covariance_head",), "uncertainty"),
)


def _enable_joint_research_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(config))
    training = value.setdefault("training", {})
    for name in ("command_conditioning", "joint_amp_phase_action", "autograd_count_jacobian"):
        training.setdefault(name, {})["enabled"] = True
    policy = training["command_conditioning"].setdefault("policy_conditioning", {})
    policy["enabled"] = True
    policy["required_for_joint_action"] = True
    policy["share_forward_command_encoder"] = True
    training.setdefault("research_curriculum", {})["enabled"] = True
    return value


def configure_stage_trainability(model: Any, stage: CurriculumStageSpec) -> Tuple[str, ...]:
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = any(name.startswith(prefix) for prefix in stage.trainable_prefixes)
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(name)
    return tuple(trainable)


def command_counterfactual_policy_loss(
    predicted_action: Any,
    teacher_action: Any,
    counterfactual_pair_id: Any,
) -> Any:
    """Match table-specific actions and their within-pair command response."""
    predicted = torch.as_tensor(predicted_action)
    teacher = torch.as_tensor(teacher_action, dtype=predicted.dtype, device=predicted.device)
    pair_ids = torch.as_tensor(counterfactual_pair_id, device=predicted.device).reshape(-1)
    loss = F.mse_loss(predicted, teacher)
    pair_loss = predicted.sum() * 0.0
    pair_count = 0
    for pair_id in torch.unique(pair_ids):
        indices = torch.nonzero(pair_ids == pair_id, as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            continue
        first, second = indices[0], indices[1]
        pair_loss = pair_loss + F.mse_loss(
            predicted[second] - predicted[first],
            teacher[second] - teacher[first],
        )
        pair_count += 1
    return loss + pair_loss / max(1, pair_count)


def command_shuffle_audit(
    model: Any,
    dataset: JointFeedbackDataset,
    *,
    device: str | Any = "cpu",
    maximum_pairs: int = 4,
) -> Dict[str, Any]:
    """Verify same-observation predictions react to physical command tensors."""
    model.eval()
    action_deltas = []
    covariance_deltas = []
    count_deltas = []
    with torch.no_grad():
        for pair_number, (left, right) in enumerate(dataset.counterfactual_pairs()):
            if pair_number >= int(maximum_pairs):
                break
            observation = {
                "observation_features": torch.as_tensor(left["observation_features"], dtype=torch.float32, device=device).unsqueeze(0),
                "exposure_s": torch.ones(1, dtype=torch.float32, device=device),
            }
            outputs = []
            for sample in (left, right):
                table = dataset.command_table(int(sample["command_table_index"]))
                command = command_tensor_batch(
                    table, device=device,
                    normalization_contract=dataset.normalization,
                    expected_normalization_sha256=dataset.normalization["sha256"],
                )
                outputs.append(model(observation, command))
            left_action = torch.cat((outputs[0]["amplitude_action_coordinates"], outputs[0]["phase_action_coordinates"]), dim=-1)
            right_action = torch.cat((outputs[1]["amplitude_action_coordinates"], outputs[1]["phase_action_coordinates"]), dim=-1)
            action_deltas.append(float(torch.linalg.vector_norm(left_action - right_action).cpu()))
            covariance_deltas.append(float(torch.linalg.vector_norm(outputs[0]["action_covariance"] - outputs[1]["action_covariance"]).cpu()))
            count_deltas.append(float(torch.linalg.vector_norm(outputs[0]["expected_counts"] - outputs[1]["expected_counts"]).cpu()))
    if not action_deltas:
        raise ValueError("command shuffle audit requires counterfactual pairs")
    result = {
        "pair_count": len(action_deltas),
        "mean_action_delta": float(np.mean(action_deltas)),
        "mean_covariance_delta": float(np.mean(covariance_deltas)),
        "mean_count_delta": float(np.mean(count_deltas)),
        "table_identity_used_as_numeric_input": False,
    }
    result["passed"] = bool(
        result["mean_action_delta"] > 1e-9
        and result["mean_covariance_delta"] > 1e-9
        and result["mean_count_delta"] > 1e-9
    )
    return result


def generate_inverse_teacher_labels(
    model: Any,
    dataset: JointFeedbackDataset,
    output_path: str | Path,
    *,
    device: str | Any = "cpu",
    maximum_samples: int | None = None,
    optimizer_config: InverseTeacherConfig | None = None,
) -> Path:
    teacher = ConstrainedInverseTeacher(model, optimizer_config)
    actions = []
    covariances = []
    masks = []
    converged = []
    objectives = []
    limit = len(dataset) if maximum_samples is None else min(len(dataset), int(maximum_samples))
    for index in range(limit):
        sample = dataset[index]
        table = dataset.command_table(int(sample["command_table_index"]))
        command = command_tensor_batch(
            table, device=device,
            normalization_contract=dataset.normalization,
            expected_normalization_sha256=dataset.normalization["sha256"],
        )
        observation = {
            "observation_features": torch.as_tensor(sample["observation_features"], dtype=torch.float32, device=device).unsqueeze(0),
            "exposure_s": torch.as_tensor([float(sample["exposure_s"])], dtype=torch.float32, device=device),
        }
        result = teacher.solve(
            observation, command,
            target_counts=sample["point_counts"],
            target_amplitude=sample["target_amplitude"],
            target_phase_rad=sample["target_phase_rad"],
            warm_start=sample["projected_teacher_action"],
            actionable_mask=sample["teacher_actionable_mask"],
        )
        actions.append(result.action_coordinates.detach().cpu().numpy()[0])
        covariances.append(result.covariance.detach().cpu().numpy()[0])
        masks.append(result.per_coordinate_actionable_mask.detach().cpu().numpy()[0])
        converged.append(result.converged)
        objectives.append(result.objective)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        action_coordinates=np.asarray(actions),
        covariance=np.asarray(covariances),
        actionable_mask=np.asarray(masks),
        converged=np.asarray(converged, dtype=np.bool_),
        objective=np.asarray(objectives, dtype=np.float64),
        amplitude_empirical_evidence_mask=np.zeros(
            (limit, action_dimensions_for_dataset(dataset)["amplitude"]), dtype=np.bool_,
        ),
    )
    return output


def action_dimensions_for_dataset(dataset: JointFeedbackDataset) -> Dict[str, int]:
    from .projective_action import action_dimensions

    return action_dimensions(str(dataset.manifest["system_type"]))


def _stage_loss(stage: CurriculumStageSpec, output: Mapping[str, Any], sample: Mapping[str, Any]) -> Any:
    dtype = output["expected_counts"].dtype
    device = output["expected_counts"].device
    target_counts = torch.as_tensor(sample["point_counts"], dtype=dtype, device=device).reshape_as(output["expected_counts"])
    target_action = torch.as_tensor(sample["projected_teacher_action"], dtype=dtype, device=device).reshape(1, -1)
    predicted_action = torch.cat((output["amplitude_action_coordinates"], output["phase_action_coordinates"]), dim=-1)
    if stage.objective in {"count", "count_adapter"}:
        mean = output["expected_counts"].clamp_min(1e-8)
        log_dispersion = output.get("count_log_dispersion")
        if log_dispersion is None:
            return torch.mean(mean - target_counts * torch.log(mean) + torch.lgamma(target_counts + 1.0))
        dispersion = torch.exp(log_dispersion).clamp_min(1e-10)
        size = 1.0 / dispersion
        return torch.mean(-(
            torch.lgamma(target_counts + size)
            - torch.lgamma(size)
            - torch.lgamma(target_counts + 1.0)
            + size * (torch.log(size) - torch.log(size + mean))
            + target_counts * (torch.log(mean) - torch.log(size + mean))
        ))
    if stage.objective == "state":
        target_amp = torch.as_tensor(sample["target_amplitude"], dtype=dtype, device=device).reshape_as(output["shared_state_amplitude"])
        target_phase = torch.as_tensor(sample["target_phase_rad"], dtype=dtype, device=device).reshape_as(output["shared_state_phase_rad"])
        weighted_amplitude = output["shared_state_amplitude"] * target_amp
        phase_delta = output["shared_state_phase_rad"] - target_phase
        overlap_real = torch.sum(weighted_amplitude * torch.cos(phase_delta), dim=-1)
        overlap_imag = torch.sum(weighted_amplitude * torch.sin(phase_delta), dim=-1)
        return torch.mean(1.0 - torch.square(overlap_real) - torch.square(overlap_imag))
    if stage.objective == "nuisance":
        return output["detector_regularization"] + output["source_regularization"]
    if stage.objective == "phase_probe":
        amp_dim = output["amplitude_action_coordinates"].shape[-1]
        return F.mse_loss(predicted_action[..., amp_dim:], target_action[..., amp_dim:])
    if stage.objective == "distillation":
        return F.mse_loss(predicted_action, target_action)
    if stage.objective == "uncertainty":
        residual = (target_action - predicted_action.detach()).unsqueeze(-1)
        covariance = output["action_covariance"]
        precision = torch.linalg.pinv(covariance)
        return torch.mean((residual.transpose(-1, -2) @ precision @ residual).reshape(-1)) + torch.logdet(covariance).mean()
    raise ValueError(f"unsupported trainable curriculum objective: {stage.objective}")


def run_curriculum(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    stage_indices: Iterable[int] = range(8),
    epochs_per_stage: int = 1,
    learning_rate: float = 1e-3,
    device: str | Any = "cpu",
    resume: bool = False,
    maximum_samples: int | None = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("V6 joint training requires torch")
    dataset = JointFeedbackDataset(dataset_dir)
    first_table = dataset.command_table(0)
    enabled_config = _enable_joint_research_config(config)
    model = build_research_feedback_model(enabled_config, first_table).to(device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_receipts = []
    for stage_index in stage_indices:
        stage = CURRICULUM_STAGE_SPECS[int(stage_index)]
        if stage.name != RESEARCH_CURRICULUM_STAGES[int(stage_index)]:
            raise RuntimeError("curriculum stage declaration drift")
        trainable_names = configure_stage_trainability(model, stage)
        checkpoint_path = output_root / f"stage_{int(stage_index) + 1:02d}_{stage.name}.pt"
        if stage.objective == "teacher_generation":
            teacher_path = generate_inverse_teacher_labels(
                model, dataset, output_root / "inverse_teacher_labels.npz",
                device=device,
                maximum_samples=maximum_samples,
                optimizer_config=InverseTeacherConfig(maximum_steps=12 if maximum_samples else 120),
            )
            metrics = {"teacher_path": teacher_path.name, "trainable_parameter_count": 0}
            optimizer = None
            scheduler = None
        else:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError(f"stage {stage.name} has no trainable parameters")
            optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(epochs_per_stage)),
            )
            start_epoch = 0
            if resume and checkpoint_path.exists():
                try:
                    loaded = load_joint_checkpoint(
                        checkpoint_path, model, optimizer=optimizer,
                        scheduler=scheduler, map_location=device,
                    )
                except Exception:
                    dataset.close()
                    raise
                start_epoch = int(loaded["training_state"].get("completed_epochs", 0))
            losses = []
            limit = len(dataset) if maximum_samples is None else min(len(dataset), int(maximum_samples))
            generated_teacher = None
            teacher_path = output_root / "inverse_teacher_labels.npz"
            if stage.objective == "distillation":
                if not teacher_path.is_file():
                    generate_inverse_teacher_labels(
                        model,
                        dataset,
                        teacher_path,
                        device=device,
                        maximum_samples=limit,
                        optimizer_config=InverseTeacherConfig(
                            maximum_steps=12 if maximum_samples else 120
                        ),
                    )
                    configure_stage_trainability(model, stage)
                with np.load(teacher_path, allow_pickle=False) as teacher_payload:
                    generated_teacher = np.asarray(teacher_payload["action_coordinates"])
                if generated_teacher.shape[0] < limit:
                    raise ValueError("generated inverse teacher has fewer rows than the distillation stage")
            model.train()
            for _epoch in range(start_epoch, int(epochs_per_stage)):
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                for sample_index in range(limit):
                    sample = dict(dataset[sample_index])
                    if generated_teacher is not None:
                        sample["projected_teacher_action"] = generated_teacher[sample_index]
                    table = dataset.command_table(int(sample["command_table_index"]))
                    command = command_tensor_batch(
                        table, device=device,
                        normalization_contract=dataset.normalization,
                        expected_normalization_sha256=dataset.normalization["sha256"],
                    )
                    observation = {
                        "observation_features": torch.as_tensor(sample["observation_features"], dtype=torch.float32, device=device).unsqueeze(0),
                        "exposure_s": torch.as_tensor([float(sample["exposure_s"])], dtype=torch.float32, device=device),
                    }
                    output = model(observation, command)
                    loss = _stage_loss(stage, output, sample)
                    (loss / max(int(batch_size), 1)).backward()
                    pending += 1
                    if pending >= max(int(batch_size), 1) or sample_index + 1 == limit:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        pending = 0
                    losses.append(float(loss.detach().cpu()))
                scheduler.step()
            metrics = {
                "mean_loss": float(np.mean(losses)) if losses else None,
                "last_loss": float(losses[-1]) if losses else None,
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
                "resume_noop": bool(resume and not losses),
                "optimizer_step_granularity": "batch",
                "batch_size": int(batch_size),
                "generated_teacher_consumed": bool(generated_teacher is not None),
            }
        shuffle = command_shuffle_audit(model, dataset, device=device)
        metrics["command_shuffle_audit"] = shuffle
        feature_order = command_tone_features(first_table)[1].ordered_names
        manifest = dataset.manifest
        metadata = build_joint_checkpoint_metadata(
            architecture_id="OperationComplexFeedbackModel",
            architecture_config={"hidden_dim": 64, "system_type": first_table.system_type},
            system_type=first_table.system_type,
            command_schema=first_table.schema,
            command_schema_hash=first_table.feature_schema_hash,
            command_table_hashes=manifest["command_table_hashes"],
            command_normalization_sha256=dataset.normalization["sha256"],
            command_feature_order=feature_order,
            policy_config=enabled_config["training"]["command_conditioning"]["policy_conditioning"],
            action_geometry_schema=ACTION_GEOMETRY_SCHEMA,
            action_geometry_version=ACTION_GEOMETRY_VERSION,
            detector_schema="FourDetectorClickModelV1",
            source_multiplicity_schema=SOURCE_MULTIPLICITY_SCHEMA,
            probe_schema="phase_signed_probe_3d_v1",
            registry_graph_hash=_as_sha256(first_table.operation_registry_hash),
            curriculum_stage_index=int(stage_index),
            curriculum_stage_name=stage.name,
            dataset_sha256=manifest["dataset_sha256"],
            calibration_sha256=_as_sha256(first_table.calibration_snapshot_hash),
            backend=str(device),
            config_sha256=canonical_sha256(enabled_config),
            metrics=metrics,
        )
        checkpoint_sha256 = save_joint_checkpoint(
            checkpoint_path, model, metadata,
            optimizer=optimizer, scheduler=scheduler,
            training_state={
                "completed_epochs": int(epochs_per_stage),
                "stage_index": int(stage_index),
                "stage_name": stage.name,
                "frozen_parameter_names": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
                "trainable_parameter_names": list(trainable_names),
            },
        )
        stage_receipts.append({
            "stage_index": int(stage_index), "stage_name": stage.name,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256, "metrics": metrics,
        })
    receipt = {
        "schema": "PhysicsHybridV6JointTrainingReceipt",
        "schema_version": 1,
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "stages": stage_receipts,
        "held_out_axes": dataset.manifest["held_out_axes"],
        "research_only": True,
        "promotion_ready": False,
        "hardware_actions_executed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (output_root / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
    )
    dataset.close()
    return receipt


def _as_sha256(value: Any) -> str:
    text = str(value or "")
    if len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text):
        return text.lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_stage_indices(value: str) -> Tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(len(CURRICULUM_STAGE_SPECS)))
    indices = tuple(int(item.strip()) - 1 for item in value.split(",") if item.strip())
    if not indices or any(index < 0 or index >= len(CURRICULUM_STAGE_SPECS) for index in indices):
        raise argparse.ArgumentTypeError("stages must be all or comma-separated values in 1..8")
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the research-only V6 joint feedback model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="ml_recon/configs/physics_hybrid_v4.json")
    parser.add_argument("--stages", type=_parse_stage_indices, default=tuple(range(8)))
    parser.add_argument("--epochs-per-stage", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "directml", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from .run_small_model_screen import resolve_device

    resolved_device = resolve_device(args.device)
    receipt = run_curriculum(
        args.dataset_dir, args.output_dir, config,
        stage_indices=args.stages,
        epochs_per_stage=args.epochs_per_stage,
        learning_rate=args.learning_rate,
        device=resolved_device,
        resume=args.resume,
        maximum_samples=2 if args.smoke else None,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "training_receipt": str(Path(args.output_dir) / "training_receipt.json"),
        "receipt_sha256": receipt["receipt_sha256"],
        "research_only": True,
        "promotion_ready": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CURRICULUM_STAGE_SPECS",
    "CurriculumStageSpec",
    "command_counterfactual_policy_loss",
    "command_shuffle_audit",
    "configure_stage_trainability",
    "generate_inverse_teacher_labels",
    "run_curriculum",
]
