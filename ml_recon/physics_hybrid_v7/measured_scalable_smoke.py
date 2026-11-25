"""Measured masked-count post-training smoke for the scalable V7 model.

The executable comparison keeps the registered measured splits fixed and
compares synthetic pretraining against an otherwise identical zero-pretraining
control.  Full coin arrays are labels only on masked operation points; the
model input receives the complementary observed subset plus full signal/idler
covariates and snapshot-bound program-state commands.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover
    torch = None
    functional = None

from ml_recon.physics_identification.operation_point_calibration import (
    _split_masks,
    load_operation_point_dataset,
)

from .geometry import OperationFrequencyGeometryV1, build_operation_frequency_geometry
from .noise_taxonomy import (
    build_operation_input_features,
    build_synthetic_physics_support,
    encode_target_program_context,
    generate_noise_taxonomy_batch,
    numpy_batch_to_torch,
)
from .physical_ceiling import load_retrieval_rate_vectors, retrieval_ceiling_features
from .program_commands import resolve_program_commands
from .scalable_model import (
    V7ScalableModelConfig,
    V7ScalablePhysicsModel,
    parameter_count,
    v7_scalable_multitask_loss,
)


DEFAULT_CONFIG_PATH = Path("ml_recon/configs/physics_hybrid_v7_scalable.json")


@dataclass(frozen=True)
class MeasuredScalableArraysV1:
    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    target_context: np.ndarray
    target_overlap: np.ndarray
    program_overlap: np.ndarray
    retrieval_ceiling: np.ndarray
    split_names: np.ndarray


def _target_arrays(records: Sequence[Any], qubit_count: int) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = []
    phases = []
    for record in records:
        state = dict(record.exact_target_state_params)
        amplitude = np.asarray(state.get("amplitudes", []), dtype=float).reshape(-1)
        phase = np.asarray(state.get("phases_rad", []), dtype=float).reshape(-1)
        if amplitude.shape != (qubit_count,) or phase.shape != amplitude.shape:
            raise ValueError("measured target state is incomplete")
        amplitudes.append(amplitude)
        phases.append(phase)
    return np.stack(amplitudes), np.stack(phases)


def build_measured_scalable_arrays(
    dataset: Any,
    data_root: str | Path,
    geometry: OperationFrequencyGeometryV1,
    retrieval_rates: np.ndarray,
) -> Tuple[MeasuredScalableArraysV1, Dict[str, Any]]:
    commands, command_audit = resolve_program_commands(
        dataset.records, data_root=data_root,
    )
    target_amplitude, target_phase = _target_arrays(
        dataset.records, geometry.qubit_count,
    )
    program_amplitude = np.stack([command.amplitudes for command in commands])
    program_phase = np.stack([command.phases_rad for command in commands])
    command_available = np.asarray([command.available for command in commands], dtype=float)
    target_context = encode_target_program_context(
        target_amplitude,
        target_phase,
        program_amplitude,
        program_phase,
        command_available,
        qubit_count=geometry.qubit_count,
    )
    target_overlap = np.stack([
        geometry.target_measurement_overlap(record.exact_target_state_params)
        for record in dataset.records
    ])
    basis = geometry.basis_amplitudes * np.exp(1j * geometry.basis_phases_rad)
    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    program_state = program_amplitude * np.exp(1j * program_phase)
    program_norm = np.linalg.norm(program_state, axis=1, keepdims=True)
    normalized_program = np.divide(
        program_state,
        np.maximum(program_norm, 1e-12),
        out=np.zeros_like(program_state),
        where=program_norm > 1e-12,
    )
    program_overlap = np.square(np.abs(normalized_program @ basis.conj().T))
    program_overlap *= command_available[:, None]
    retrieval_ceiling = np.asarray([
        retrieval_ceiling_features(
            record.exact_target_state_params, retrieval_rates, geometry,
        )[0][0]
        for record in dataset.records
    ])
    arrays = MeasuredScalableArraysV1(
        coin_counts=np.asarray(dataset.coin_counts, dtype=np.float32),
        signal_counts=np.asarray(dataset.signal_counts, dtype=np.float32),
        idler_counts=np.asarray(dataset.idler_counts, dtype=np.float32),
        target_context=target_context.astype(np.float32),
        target_overlap=target_overlap.astype(np.float32),
        program_overlap=program_overlap.astype(np.float32),
        retrieval_ceiling=retrieval_ceiling.astype(np.float32),
        split_names=dataset.split_names.astype(str),
    )
    return arrays, {
        "schema": "v7_measured_scalable_input_audit_v1",
        "command_audit": command_audit,
        "snapshot_program_amplitude_used": True,
        "snapshot_program_phase_used": True,
        "target_and_program_are_separate_inputs": True,
        "hidden_coin_counts_used_for_input": False,
        "signal_idler_full_covariates_used": True,
        "current_matrix_directory_read": False,
        "historical_command_invented": False,
    }


def _operation_masks(row_indices: np.ndarray, operation_count: int, seed: int) -> np.ndarray:
    operation_index = np.arange(operation_count)
    masks = np.zeros((row_indices.size, operation_count), dtype=bool)
    for local_index, row_index in enumerate(row_indices):
        kind = (int(row_index) + int(seed)) % 4
        if kind == 0:
            masks[local_index] = operation_index % 2 == 0
        elif kind == 1:
            masks[local_index] = operation_index % 2 == 1
        elif kind == 2:
            masks[local_index] = np.isin(operation_index % 4, (0, 3))
        else:
            rng = np.random.default_rng(int(seed) * 1000003 + int(row_index))
            masks[local_index] = rng.random(operation_count) < 0.5
    return masks


def _measured_batch(
    arrays: MeasuredScalableArraysV1,
    support: Any,
    row_indices: np.ndarray,
    *,
    seed: int,
    device: Any,
) -> Dict[str, Any]:
    rows = np.asarray(row_indices, dtype=int).reshape(-1)
    mask = _operation_masks(rows, arrays.coin_counts.shape[1], seed)
    observed_coin = arrays.coin_counts[rows] * mask
    operation_inputs = build_operation_input_features(
        support,
        observed_coin_counts=observed_coin,
        signal_counts=arrays.signal_counts[rows],
        idler_counts=arrays.idler_counts[rows],
        observed_mask=mask,
        target_overlap=arrays.target_overlap[rows],
        program_overlap=arrays.program_overlap[rows],
    )
    payload = {
        "operation_inputs": operation_inputs,
        "target_context": arrays.target_context[rows],
        "signal_counts": arrays.signal_counts[rows],
        "idler_counts": arrays.idler_counts[rows],
        "coin_counts": arrays.coin_counts[rows],
        "observed_mask": mask.astype(np.float32),
        "retrieval_ceiling": arrays.retrieval_ceiling[rows],
    }
    return numpy_batch_to_torch(payload, device)


def _real_masked_loss(output: Mapping[str, Any], batch: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    observed = batch["coin_counts"].to(dtype=output["expected_counts"].dtype)
    hidden_mask = 1.0 - batch["observed_mask"].to(dtype=observed.dtype)
    mean = output["expected_counts"].clamp_min(1e-8)
    alpha = output["nb2_dispersion"].clamp_min(1e-8)[:, None]
    shape = 1.0 / alpha
    log_probability = (
        torch.lgamma(observed + shape) - torch.lgamma(shape)
        - torch.lgamma(observed + 1.0)
        + shape * (torch.log(shape) - torch.log(shape + mean))
        + observed * (torch.log(mean) - torch.log(shape + mean))
    )
    nb2_nll = -(log_probability * hidden_mask).sum() / hidden_mask.sum().clamp_min(1.0)
    hidden_probability = (output["probability"] * hidden_mask).clamp_min(1e-12)
    hidden_probability /= hidden_probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
    hidden_total = (observed * hidden_mask).sum(dim=1).clamp_min(1.0)
    per_row_conditional = -(
        observed * hidden_mask * torch.log(hidden_probability)
    ).sum(dim=1) / hidden_total
    retrieval_loss = functional.smooth_l1_loss(
        output["retrieval_ceiling"],
        batch["retrieval_ceiling"].to(dtype=mean.dtype),
    )
    total = nb2_nll + 0.35 * per_row_conditional.mean() + 0.1 * retrieval_loss
    return total, {
        "nb2_nll_per_hidden_operation": nb2_nll,
        "conditional_nll_per_click": per_row_conditional,
        "retrieval_loss": retrieval_loss,
    }


def _evaluate_model(
    model: Any,
    arrays: MeasuredScalableArraysV1,
    support: Any,
    row_mask: np.ndarray,
    *,
    seed: int,
    device: Any,
    batch_size: int,
) -> Dict[str, Any]:
    indices = np.flatnonzero(row_mask)
    conditional_rows = []
    nb2_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            batch = _measured_batch(
                arrays, support, indices[start:start + batch_size], seed=seed, device=device,
            )
            output = model(
                batch["operation_inputs"], batch["target_context"],
                batch["signal_counts"], batch["idler_counts"],
            )
            _, metrics = _real_masked_loss(output, batch)
            conditional_rows.extend(
                metrics["conditional_nll_per_click"].detach().cpu().numpy().tolist()
            )
            nb2_values.append(float(metrics["nb2_nll_per_hidden_operation"].cpu()))
    values = np.asarray(conditional_rows, dtype=float)
    return {
        "sample_count": int(indices.size),
        "conditional_nll_per_click_mean": float(np.mean(values)),
        "conditional_nll_per_click_median": float(np.median(values)),
        "conditional_nll_per_click_rows": values.tolist(),
        "nb2_nll_per_hidden_operation_mean": float(np.mean(nb2_values)),
    }


def _train_synthetic(
    model: Any,
    optimizer: Any,
    geometry: Any,
    support: Any,
    config: Mapping[str, Any],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: Any,
) -> None:
    model.train()
    for step in range(int(steps)):
        batch = numpy_batch_to_torch(generate_noise_taxonomy_batch(
            geometry, support, batch_size=batch_size, seed=seed + step,
            low_count_fraction=0.6,
        ), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["operation_inputs"], batch["target_context"],
            batch["signal_counts"], batch["idler_counts"],
        )
        loss, _ = v7_scalable_multitask_loss(output, batch, config["loss_weights"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _posttrain_real(
    model: Any,
    optimizer: Any,
    arrays: MeasuredScalableArraysV1,
    support: Any,
    train_mask: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: Any,
) -> None:
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(int(steps)):
        replace = train_indices.size < batch_size
        rows = rng.choice(train_indices, size=batch_size, replace=replace)
        batch = _measured_batch(arrays, support, rows, seed=seed + step, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["operation_inputs"], batch["target_context"],
            batch["signal_counts"], batch["idler_counts"],
        )
        loss, _ = _real_masked_loss(output, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _gain(pretrained: Mapping[str, Any], control: Mapping[str, Any]) -> float:
    return float(
        control["conditional_nll_per_click_mean"]
        - pretrained["conditional_nll_per_click_mean"]
    )


def _paired_bootstrap_gain(
    pretrained: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    pretrained_rows = np.asarray(
        pretrained["conditional_nll_per_click_rows"], dtype=float,
    )
    control_rows = np.asarray(
        control["conditional_nll_per_click_rows"], dtype=float,
    )
    if pretrained_rows.shape != control_rows.shape or pretrained_rows.size == 0:
        raise ValueError("paired measured metrics have incompatible rows")
    difference = control_rows - pretrained_rows
    rng = np.random.default_rng(int(seed))
    bootstrap = np.asarray([
        np.mean(difference[rng.integers(0, difference.size, difference.size)])
        for _ in range(int(repetitions))
    ])
    return {
        "mean_nll_gain": float(np.mean(difference)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_positive_fraction": float(np.mean(bootstrap > 0.0)),
        "paired_sample_count": int(difference.size),
    }


def _blend_model_state(
    initial_state: Mapping[str, Any],
    pretrained_state: Mapping[str, Any],
    strength: float,
) -> Dict[str, Any]:
    blend = float(strength)
    if not 0.0 <= blend <= 1.0 or initial_state.keys() != pretrained_state.keys():
        raise ValueError("transfer strength/state schema is invalid")
    result: Dict[str, Any] = {}
    for name, initial_value in initial_state.items():
        pretrained_value = pretrained_state[name]
        if torch.is_floating_point(initial_value):
            result[name] = (1.0 - blend) * initial_value + blend * pretrained_value
        else:
            result[name] = initial_value.clone()
    return result


def _transfer_key(strength: float) -> str:
    return f"transfer_{float(strength):g}".replace(".", "p")


def run_measured_scalable_smoke(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    manifest_dir: str | Path = "ml_recon/reports/measured_data",
    data_root: str | Path = "data",
    output_dir: str | Path = "ml_recon/reports/physics_hybrid_v7/measured_scalable_smoke",
    device: Optional[str] = None,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("measured scalable smoke requires torch")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    profile = dict(config["profiles"].get("measured_smoke") or {
        "width": 128, "latent_tokens": 16, "layers": 4, "attention_heads": 8,
        "feedforward_multiplier": 3.0, "dropout": 0.0, "count_smoothing": 0.5,
        "use_activation_checkpointing": False, "batch_size": 8,
        "synthetic_steps": 160, "real_steps": 80, "learning_rate": 0.0003,
    })
    system_type = str(config["system_type"])
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, retrieval_audit = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates, graph_mode_count=int(config["graph_mode_count"]),
        graph_length_mhz=float(config["graph_length_mhz"]),
    )
    dataset = load_operation_point_dataset(
        manifest_dir, data_root=data_root, system_type=system_type,
    )
    arrays, input_audit = build_measured_scalable_arrays(
        dataset, data_root, geometry, retrieval_rates,
    )
    train_mask, validation_mask, holdouts = _split_masks(dataset)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = V7ScalableModelConfig.from_mapping(
        {name: profile[name] for name in (
            "width", "latent_tokens", "layers", "attention_heads",
            "feedforward_multiplier", "dropout", "count_smoothing",
            "use_activation_checkpointing",
        )},
        system_type=system_type,
        operation_count=geometry.operation_count,
        qubit_count=geometry.qubit_count,
    )
    seed = int(config["seed"]) + 40000 + int(seed_offset)
    torch.manual_seed(seed)
    initial_model = V7ScalablePhysicsModel(model_config, support.graph_kernel)
    initial_state = copy.deepcopy(initial_model.state_dict())
    learning_rate = float(profile["learning_rate"])
    started = time.perf_counter()
    pretraining_model = V7ScalablePhysicsModel(
        model_config, support.graph_kernel,
    ).to(selected_device)
    pretraining_model.load_state_dict(initial_state)
    pretraining_optimizer = torch.optim.AdamW(
        pretraining_model.parameters(), lr=learning_rate, weight_decay=0.01,
    )
    _train_synthetic(
        pretraining_model, pretraining_optimizer,
        geometry, support, config, steps=int(profile["synthetic_steps"]),
        batch_size=int(profile["batch_size"]), seed=seed, device=selected_device,
    )
    pretrained_state = {
        name: value.detach().cpu().clone()
        for name, value in pretraining_model.state_dict().items()
    }
    transfer_strengths = tuple(float(value) for value in profile["transfer_strengths"])
    if 0.0 not in transfer_strengths or 1.0 not in transfer_strengths:
        raise ValueError("measured transfer strengths must contain 0 and 1")
    models = {
        _transfer_key(strength): V7ScalablePhysicsModel(
            model_config, support.graph_kernel,
        ).to(selected_device)
        for strength in transfer_strengths
    }
    optimizers = {}
    for strength in transfer_strengths:
        name = _transfer_key(strength)
        models[name].load_state_dict(_blend_model_state(
            initial_state, pretrained_state, strength,
        ))
        optimizers[name] = torch.optim.AdamW(
            models[name].parameters(), lr=learning_rate, weight_decay=0.01,
        )
        _posttrain_real(
            models[name], optimizers[name], arrays, support, train_mask,
            steps=int(profile["real_steps"]), batch_size=int(profile["batch_size"]),
            seed=seed + 10000, device=selected_device,
        )
    split_masks = {"validation": validation_mask, **holdouts}
    metrics: Dict[str, Dict[str, Any]] = {}
    for split_index, (split_name, split_mask) in enumerate(split_masks.items()):
        metrics[split_name] = {
            name: _evaluate_model(
                model, arrays, support, split_mask, seed=seed + 20000 + split_index,
                device=selected_device, batch_size=max(8, int(profile["batch_size"])),
            )
            for name, model in models.items()
        }
    minimum_gain = float(config["selection_gates"]["minimum_validation_nll_gain"])
    control_name = _transfer_key(0.0)
    selected_name = min(
        models,
        key=lambda name: (
            metrics["validation"][name]["conditional_nll_per_click_mean"], name,
        ),
    )
    validation_gain = _gain(
        metrics["validation"][selected_name], metrics["validation"][control_name],
    )
    if validation_gain < minimum_gain:
        selected_name = control_name
        validation_gain = 0.0
    for split_index, split_name in enumerate(metrics):
        metrics[split_name]["selected_transfer"] = metrics[split_name][selected_name]
        metrics[split_name]["zero_pretraining_control"] = metrics[split_name][control_name]
        metrics[split_name]["selected_transfer_nll_gain"] = _gain(
            metrics[split_name][selected_name], metrics[split_name][control_name],
        )
        metrics[split_name]["full_pretraining_nll_gain"] = _gain(
            metrics[split_name][_transfer_key(1.0)], metrics[split_name][control_name],
        )
        metrics[split_name]["paired_bootstrap_gain"] = _paired_bootstrap_gain(
            metrics[split_name][selected_name], metrics[split_name][control_name],
            repetitions=1000, seed=seed + 30000 + split_index,
        )
    validation_ci_low = float(metrics["validation"]["paired_bootstrap_gain"]["ci95_low"])
    required_ood = tuple(
        name for name in ("target_ood", "time_run_ood", "q4_plus90_intervention_test")
        if name in metrics and metrics[name]["selected_transfer"]["sample_count"] > 0
    )
    maximum_ood_regression = float(
        config["selection_gates"]["maximum_ood_nll_regression"]
    )
    ood_noninferior = all(
        float(metrics[name]["paired_bootstrap_gain"]["ci95_low"])
        >= -maximum_ood_regression
        for name in required_ood
    )
    selected = bool(
        selected_name != control_name
        and validation_gain >= minimum_gain
        and validation_ci_low > 0.0
        and ood_noninferior
    )
    summary = {
        "schema": "physics_hybrid_v7_measured_scalable_smoke_v1",
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "production_subtraction_allowed": False,
        "system_type": system_type,
        "device": str(selected_device),
        "parameter_count": parameter_count(models[control_name]),
        "model_config": model_config.to_json_dict(),
        "synthetic_steps": int(profile["synthetic_steps"]),
        "real_posttraining_steps_per_model": int(profile["real_steps"]),
        "transfer_strengths": list(transfer_strengths),
        "seed": seed,
        "seed_offset": int(seed_offset),
        "elapsed_s": float(time.perf_counter() - started),
        "input_audit": input_audit,
        "retrieval_audit": retrieval_audit,
        "metrics": metrics,
        "selection": {
            "minimum_validation_nll_gain": minimum_gain,
            "validation_gain": validation_gain,
            "validation_gain_ci95_low": validation_ci_low,
            "validation_gain_ci95_high": float(
                metrics["validation"]["paired_bootstrap_gain"]["ci95_high"]
            ),
            "validation_gain_ci_excludes_zero": bool(validation_ci_low > 0.0),
            "selected_transfer_name": selected_name,
            "selected_transfer_strength": float(
                next(
                    strength for strength in transfer_strengths
                    if _transfer_key(strength) == selected_name
                )
            ),
            "required_ood_splits": list(required_ood),
            "maximum_ood_nll_regression": maximum_ood_regression,
            "ood_noninferior": ood_noninferior,
            "synthetic_pretraining_selected": selected,
            "selection_interpretation": (
                "transfer strength is selected on measured validation only; it is retained "
                "only if paired validation confidence and every registered OOD audit pass"
            ),
        },
        "unevaluated": {
            "posterior_predictive_checks": True,
            "simulation_based_calibration": True,
            "four_gpu_execution": True,
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "v7_measured_scalable_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--measured-manifest-dir", default="ml_recon/reports/measured_data")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-dir", default="ml_recon/reports/physics_hybrid_v7/measured_scalable_smoke",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_measured_scalable_smoke(
        config_path=args.config,
        manifest_dir=args.measured_manifest_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        seed_offset=args.seed_offset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MeasuredScalableArraysV1",
    "build_measured_scalable_arrays",
    "run_measured_scalable_smoke",
]
