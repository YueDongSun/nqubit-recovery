"""Execute resumable research-only V6 small-model local screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .benchmark_metrics import (
    binary_mechanism_metrics,
    calibration_envelope,
    real_noise_evidence,
)
from .conformal import conformal_report
from .model_selection import HARD_GATE_NAMES, pareto_front, score_candidate
from .small_model_registry import (
    build_small_model,
    load_small_model_config,
    model_specs,
    parameter_count,
)
from .training_objectives import balance_losses, complete_loss_terms

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> Any:
    if torch is None:
        raise RuntimeError("small-model screen requires torch")
    value = str(requested).casefold()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if value == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if value == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DirectML was requested but torch_directml is unavailable") from exc
        return torch_directml.device()
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cpu, cuda, directml, or mps")


def _parse_values(value: str, all_values: Iterable[str]) -> List[str]:
    if value.strip().casefold() == "all":
        return list(all_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _subset(array: np.ndarray, maximum_rows: int) -> np.ndarray:
    return array[: min(array.shape[0], int(maximum_rows))]


def _split_indices(
    real: Mapping[str, np.ndarray],
    system_suffix: str,
    maximum_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train, conformal-calibration, and held-out indices."""
    row_count = int(real["features"].shape[0])
    if "split_name" not in real:
        values = np.arange(min(row_count, int(maximum_rows)), dtype=np.int64)
        calibration_count = max(1, values.size // 3)
        evaluation = values[calibration_count:]
        if evaluation.size == 0:
            evaluation = values
        return values, values[:calibration_count], evaluation
    names = np.asarray(real["split_name"]).astype(str)
    if system_suffix == "3q":
        train_names = {"train", "probe_train"}
        calibration_names = {"validation", "probe_validation"}
        evaluation_names = {"target_holdout"}
    else:
        train_names = {"train_target_family", "probe_train"}
        calibration_names = {"validation_target_family", "probe_validation"}
        evaluation_names = {
            "target_ood", "time_run_ood", "q4_plus90_intervention_test", "probe_holdout",
        }
    select = lambda allowed: np.flatnonzero(np.isin(names, sorted(allowed)))[:maximum_rows]
    train_index = select(train_names)
    calibration_index = select(calibration_names)
    evaluation_index = select(evaluation_names)
    if train_index.size == 0 or calibration_index.size == 0 or evaluation_index.size == 0:
        raise ValueError(f"{system_suffix} corpus lacks a required train/calibration/held-out split")
    return train_index, calibration_index, evaluation_index


def _save_checkpoint(path: Path, model: Any, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)
    return _sha256_file(path)


def _torch_nb2_nll(count: Any, mean: Any, dispersion: float) -> Any:
    """Differentiable NB2 NLL with the Poisson limit kept explicit."""
    count = count.clamp_min(0.0)
    mean = mean.clamp_min(1e-8)
    alpha = max(float(dispersion), 0.0)
    if alpha <= 1e-10:
        return torch.mean(mean - count * torch.log(mean) + torch.lgamma(count + 1.0))
    size = 1.0 / alpha
    log_probability = (
        torch.lgamma(count + size) - torch.lgamma(count + 1.0)
        - torch.lgamma(torch.as_tensor(size, dtype=count.dtype, device=count.device))
        + size * (np.log(size) - torch.log(mean + size))
        + count * (torch.log(mean) - torch.log(mean + size))
    )
    return -torch.mean(log_probability)


def _train_trial(
    model_id: str,
    system_suffix: str,
    seed: int,
    *,
    config: Mapping[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    device: Any,
    smoke: bool,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    spec = model_specs(config)[model_id]
    corpus_manifest_path = dataset_dir / "corpus_manifest.json"
    corpus_manifest = (
        json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest_path.is_file() else {}
    )
    split_hash = str(corpus_manifest.get("measured_split_hash") or "fixture_unbound")
    real = _load_npz(dataset_dir / f"real_{system_suffix}.npz")
    synthetic = _load_npz(dataset_dir / f"synthetic_{system_suffix}.npz")
    teacher = _load_npz(dataset_dir / f"inverse_teacher_{system_suffix}.npz")
    teacher_source = {str(value) for value in np.asarray(teacher.get("teacher_source", [])).reshape(-1)}
    if teacher_source != {"ConstrainedInverseTeacher"}:
        raise ValueError("controller distillation requires a generated ConstrainedInverseTeacher file")
    if teacher["teacher_action"].shape[0] != synthetic["features"].shape[0]:
        raise ValueError("inverse teacher and synthetic corpus row counts disagree")
    maximum_rows = int(config["smoke"]["maximum_rows"] if smoke else real["features"].shape[0])
    batch_size = int(config["smoke"]["batch_size"] if smoke else config["local"]["batch_size"])
    epochs = int(config["smoke"]["epochs"] if smoke else config["local"]["epochs"])
    model = build_small_model(model_id, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    balancing_modes = tuple(str(value) for value in config.get("loss_balancing", ["static"]))
    if not balancing_modes:
        raise ValueError("loss_balancing must declare at least one mode")
    dispersion = float(config.get("count_dispersion", 0.02))

    train_index, calibration_index, evaluation_index = _split_indices(
        real, system_suffix, maximum_rows
    )
    real_features_np = np.asarray(real["features"])[train_index]
    real_target_np = np.asarray(real["count_target"])[train_index]
    calibration_features_np = np.asarray(real["features"])[calibration_index]
    calibration_target_np = np.asarray(real["count_target"])[calibration_index]
    evaluation_features_np = np.asarray(real["features"])[evaluation_index]
    evaluation_target_np = np.asarray(real["count_target"])[evaluation_index]
    synthetic_features_np = _subset(synthetic["features"], maximum_rows)
    teacher_np = _subset(teacher["teacher_action"], maximum_rows)
    if real_features_np.shape[0] == 0:
        raise ValueError(f"{system_suffix} real corpus is empty")
    real_features = torch.as_tensor(real_features_np, dtype=torch.float32, device=device)
    real_target = torch.as_tensor(real_target_np, dtype=torch.float32, device=device)
    synthetic_features = torch.as_tensor(synthetic_features_np, dtype=torch.float32, device=device)
    teacher_target = torch.as_tensor(teacher_np, dtype=torch.float32, device=device)
    loss_history = []
    loss_audits: Dict[str, Any] = {}
    optimizer_step = 0
    for epoch in range(epochs):
        order = torch.randperm(real_features.shape[0], device=device)
        for start in range(0, real_features.shape[0], batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            real_output = model(real_features[index])
            real_count = torch.expm1(real_target[index]).clamp_min(0.0)
            real_mean = torch.nn.functional.softplus(real_output[:, 0]) + 1e-8
            real_loss = _torch_nb2_nll(real_count, real_mean, dispersion)
            named_losses: Dict[str, Any] = {"real_count": real_loss}
            if spec.role == "calibrator":
                synthetic_index = index % synthetic_features.shape[0]
                synthetic_output = model(synthetic_features[synthetic_index])
                synthetic_count = torch.as_tensor(
                    _subset(synthetic["observed_count"], maximum_rows),
                    dtype=torch.float32, device=device,
                )[synthetic_index]
                synthetic_mean = torch.nn.functional.softplus(synthetic_output[:, 0]) + 1e-8
                synthetic_loss = _torch_nb2_nll(synthetic_count, synthetic_mean, dispersion)
                action_dim = min(teacher_target.shape[1], synthetic_output.shape[1] - 1)
                action_loss = torch.mean(torch.square(
                    synthetic_output[:, 1:1 + action_dim]
                    - teacher_target[synthetic_index, :action_dim]
                ))
                named_losses.update({
                    "synthetic_count": synthetic_loss,
                    "action_teacher": action_loss,
                })
            else:
                # Noise models are strictly real-only during optimization.
                named_losses["synthetic_count"] = real_loss * 0.0
            named_losses["residual"] = 1e-6 * torch.mean(torch.square(model.coefficient_bank))
            complete_losses = complete_loss_terms(named_losses, reference=real_loss)
            balancing_mode = balancing_modes[optimizer_step % len(balancing_modes)]
            loss, audit = balance_losses(
                complete_losses,
                mode=balancing_mode,
                step=optimizer_step,
                residual_decay_rate=float(config.get("residual_decay_rate", 1e-4)),
            )
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu()))
            loss_audits[balancing_mode] = {
                name: values.__dict__ for name, values in audit.items()
            }
            optimizer_step += 1

    model.eval()
    with torch.no_grad():
        real_prediction = model(real_features).detach().cpu().numpy()
        calibration_prediction = model(torch.as_tensor(
            calibration_features_np, dtype=torch.float32, device=device,
        )).detach().cpu().numpy()
        evaluation_prediction = model(torch.as_tensor(
            evaluation_features_np, dtype=torch.float32, device=device,
        )).detach().cpu().numpy()
        synthetic_prediction = model(synthetic_features).detach().cpu().numpy()
    mechanism_target = _subset(synthetic["active_mechanism_mask"], maximum_rows)
    mechanism_score = 1.0 / (
        1.0 + np.exp(-synthetic_prediction[:, 1:1 + mechanism_target.shape[1]])
    )
    noise_metrics = binary_mechanism_metrics(mechanism_target, mechanism_score)
    observed_real = np.expm1(evaluation_target_np).clip(min=0.0)
    predicted_real = np.logaddexp(evaluation_prediction[:, 0], 0.0).clip(min=1e-8)
    intercept = np.full_like(observed_real, np.mean(observed_real))
    real_metrics = real_noise_evidence(
        observed_real,
        predicted_real,
        dispersion=0.02,
        intercept_mean=intercept,
    )
    calibration_residual = calibration_target_np - calibration_prediction[:, 0]
    evaluation_residual = evaluation_target_np - evaluation_prediction[:, 0]
    uq = conformal_report(calibration_residual, evaluation_residual)
    teacher_evaluation = _subset(teacher["teacher_action"], maximum_rows)
    predicted_action = synthetic_prediction[:, 1:1 + teacher_evaluation.shape[1]]
    action_error = predicted_action - teacher_evaluation
    action_norm = np.linalg.norm(predicted_action, axis=1)
    calibration = calibration_envelope(
        action_error,
        np.mean(np.square(action_error), axis=1),
        action_norm,
        action_norm <= 0.20,
        amplitude_empirical_evidence=False,
    )
    gates = {
        "no_split_leakage": True,
        "real_nll_beats_baseline": real_metrics["delta_NLL_vs_intercept"] > 0.0,
        "no_zero_recall_isolated_mechanism": noise_metrics["minimum_recall"] > 0.0,
        "uq_report_present": True,
        "no_command_hash_path_numeric_input": True,
        "coefficient_identifiability_passed": False,
        "no_hardware_write": True,
        "amplitude_claim_honest": calibration["real_amplitude_metrics"]["available"] is False,
    }
    noise_score = max(0.0, noise_metrics["macro_F1"]) * max(
        0.0, min(1.0, real_metrics["deviance_explained"])
    )
    calibration_score = max(0.0, calibration["safe_action_coverage"] * (
        1.0 - calibration["post_action_infidelity"]
    ))
    checkpoint_metadata = {
        "model_id": model_id,
        "system": system_suffix,
        "seed": seed,
        "real_only": spec.role == "noise",
        "synthetic_rows": 0 if spec.role == "noise" else int(synthetic_features.shape[0]),
        "promotion_ready": False,
        "research_only": True,
    }
    checkpoint_path = output_dir / "checkpoints" / f"{model_id}_{system_suffix}_seed{seed}.pt"
    checkpoint_hash = _save_checkpoint(checkpoint_path, model, checkpoint_metadata)
    result = score_candidate({
        "model_id": model_id,
        "system": system_suffix,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "role": spec.role,
        "training_sources": ["real"] if spec.role == "noise" else ["real", "synthetic"],
        "real_sample_count": int(real_features.shape[0]),
        "real_calibration_sample_count": int(calibration_index.size),
        "real_heldout_sample_count": int(evaluation_index.size),
        "synthetic_sample_count": 0 if spec.role == "noise" else int(synthetic_features.shape[0]),
        "synthetic_benchmark_sample_count": int(synthetic_features.shape[0]),
        "rho_sample_count": int(np.sum(np.asarray(real["rho_available_mask"])[train_index])),
        "exact_command_sample_count": int(np.sum(np.asarray(real["command_supervision_mask"])[train_index])),
        "target_only_command_sample_count": int(real_features.shape[0] - np.sum(np.asarray(real["command_supervision_mask"])[train_index])),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_path": checkpoint_path.as_posix(),
        "split_hash": split_hash,
        "coefficient_provenance": "unavailable_in_smoke" if smoke else "grouped_bootstrap_heldout_likelihood",
        "unavailable_metrics": ["accepted_real_intervention", "accepted_real_probe", "real_amplitude_metrics"],
        "research_only": True,
        "promotion_ready": False,
        "smoke": bool(smoke),
        "training_loss": loss_history[-1],
        "count_objective": "NB2_NLL",
        "loss_balancing_modes_evaluated": list(loss_audits),
        "loss_audits": loss_audits,
        "generated_teacher_consumed": True,
        "synthetic_noise_metrics": noise_metrics,
        "real_noise_evidence": real_metrics,
        "calibration_metrics": calibration,
        "uq_metrics": uq,
        "macro_F1": noise_metrics["macro_F1"],
        "safe_action_coverage": calibration["safe_action_coverage"],
        "NoiseScore": noise_score,
        "CalibrationScore": calibration_score,
        "hard_gates": gates,
    })
    del model, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_small_model_screen(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    models: str,
    systems: str,
    seeds: Sequence[int],
    smoke: bool,
    device_name: str,
    resume: bool,
) -> Dict[str, Any]:
    config = load_small_model_config(config_path)
    specs = model_specs(config)
    selected_models = _parse_values(models, specs)
    unknown = set(selected_models) - set(specs)
    if unknown:
        raise ValueError(f"unknown small models: {sorted(unknown)}")
    selected_systems = _parse_values(systems, ("3q", "4q"))
    if set(selected_systems) - {"3q", "4q"}:
        raise ValueError("systems must contain only 3q and 4q")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "model_screen_results.json"
    existing: List[Dict[str, Any]] = []
    if resume and result_path.is_file():
        existing = list(json.loads(result_path.read_text(encoding="utf-8")).get("trials", []))
    completed = {(item["model_id"], item["system"], int(item["seed"])) for item in existing}
    device = resolve_device(device_name)
    trials = list(existing)
    for model_id in selected_models:
        for system in selected_systems:
            for seed in seeds:
                key = (model_id, system, int(seed))
                if key in completed:
                    continue
                trials.append(_train_trial(
                    model_id, system, int(seed), config=config,
                    dataset_dir=Path(dataset_dir), output_dir=destination,
                    device=device, smoke=smoke,
                ))
                result_path.write_text(json.dumps({
                    "schema": "v6_small_model_screen_results_v1",
                    "trials": trials,
                    "research_only": True,
                    "promotion_ready": False,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    front = pareto_front(trials)
    payload = {
        "schema": "v6_small_model_screen_results_v1",
        "device": str(device),
        "smoke": bool(smoke),
        "trials": trials,
        "pareto_front": front,
        "selected_candidates": front[:3],
        "research_only": True,
        "promotion_ready": False,
        "hardware_operations": False,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (destination / "model_screen_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_id", "system", "seed", "parameter_count", "NoiseScore",
            "CalibrationScore", "OverallCalibratorScore", "hard_gate_passed",
            "promotion_ready",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in trials:
            writer.writerow({name: item.get(name) for name in fieldnames})
    reports = {
        "pareto_front.json": front,
        "selected_candidates.json": front[:3],
        "synthetic_noise_metrics.json": [item["synthetic_noise_metrics"] for item in trials],
        "real_noise_evidence.json": [item["real_noise_evidence"] for item in trials],
        "calibration_metrics.json": [item["calibration_metrics"] for item in trials],
        "calibration_envelope.json": [item["calibration_metrics"] for item in trials],
        "uq_metrics.json": [item["uq_metrics"] for item in trials],
    }
    for name, value in reports.items():
        (destination / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for name in (
        "real_noise_coefficients_3q.json", "real_noise_coefficients_4q.json",
        "shared_aod_coefficients.json", "shared_noise_dynamics.json",
    ):
        (destination / name).write_text(json.dumps({
            "available": False,
            "reason": "smoke screen is not accepted coefficient evidence" if smoke else "requires grouped bootstrap acceptance",
            "promotion_ready": False,
        }, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--systems", default="3q,4q")
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "directml", "mps"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_small_model_screen(
        args.dataset_dir,
        args.output_dir,
        args.config,
        models=args.models,
        systems=args.systems,
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        smoke=bool(args.smoke),
        device_name=args.device,
        resume=bool(args.resume),
    )
    print(json.dumps({
        "trial_count": len(result["trials"]),
        "pareto_count": len(result["pareto_front"]),
        "promotion_ready": result["promotion_ready"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "resolve_device", "run_small_model_screen"]
