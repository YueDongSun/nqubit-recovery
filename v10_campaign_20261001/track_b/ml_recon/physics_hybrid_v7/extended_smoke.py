"""Multi-profile (local / mid / server) V7 smoke with 5-seed PPC + SBC.

The runner trains the v2 scalable model on three profiles:

* ``local_smoke`` — existing local single-GPU config (width 384, L=8).
* ``mid_smoke`` — a new mid-tier profile sized for the local 4070 Ti with
  activation checkpointing; target ~3-5 GiB peak reserved.
* ``server_4x4090_plan`` — a planning-only FSDP estimate, no real 4-GPU run.

For each profile the runner runs five seeds, fits the closed-form hierarchical
posterior on a held-out subset, then evaluates posterior predictive checks and
simulation-based calibration.  Only the local and mid profiles execute; the
server profile emits a planning estimate that is not execution evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .geometry import build_operation_frequency_geometry
from .hierarchical_posterior import (
    HierarchicalPosteriorArrays,
    HierarchicalPosteriorConfig,
    fit_hierarchical_posterior,
    predict_hierarchical_posterior,
    mechanism_attribution,
)
from .noise_taxonomy import (
    build_synthetic_physics_support,
    generate_noise_taxonomy_batch,
    numpy_batch_to_torch,
)
from .physical_ceiling import load_retrieval_rate_vectors
from .posterior_calibration import (
    five_seed_aggregator,
    posterior_predictive_check,
    simulation_based_calibration,
)
from .scalable_model import (
    V7ScalableModelConfig,
    V7ScalablePhysicsModel,
    estimate_fsdp_memory_bytes,
    parameter_count,
    v7_scalable_multitask_loss,
)
from .scalable_model_v2 import (
    V7ScalableModelV2Config,
    V7ScalablePhysicsModelV2,
    build_cycle_index,
)


DEFAULT_CONFIG_PATH = Path("ml_recon/configs/physics_hybrid_v7_scalable.json")


def _resolve_aod_features(geometry: Any) -> Tuple[np.ndarray, np.ndarray]:
    static, _ = geometry.static_features()
    # AOD frequency features: 2 = (x, y) per operation
    aod_freq = static[:, :2]
    aod_resp = static[:, 2:6]
    return aod_freq.astype(np.float64), aod_resp.astype(np.float64)


def _fit_closed_form(
    batch_numpy: Mapping[str, np.ndarray],
    geometry: Any,
    retrieval_rates: np.ndarray,
) -> "fit_hierarchical_posterior":
    sample_count, operation_count = batch_numpy["coin_counts"].shape
    qubit_count = geometry.qubit_count
    cycle_count = max(int(operation_count // 64), 1) if operation_count >= 64 else max(int(operation_count // 15), 1)
    aod_freq, aod_resp = _resolve_aod_features(geometry)
    target_context = np.asarray(batch_numpy["target_context"], dtype=np.float64)
    program_amp = target_context[:, :qubit_count]
    program_phase = np.arctan2(target_context[:, qubit_count:2*qubit_count], target_context[:, 2*qubit_count:3*qubit_count])
    config = HierarchicalPosteriorConfig(
        operation_count=operation_count,
        qubit_count=qubit_count,
        cycle_count=cycle_count,
    )
    multi_total = (
        np.sum(np.asarray(batch_numpy["coin_counts"], dtype=np.float64), axis=1)
        * 0.02  # heuristic: 2% of coin count is multi-excitation
    )
    cycle_index = build_cycle_index(operation_count, qubit_count, sample_count, seed=20260807)
    arrays = HierarchicalPosteriorArrays(
        coin_counts=np.asarray(batch_numpy["coin_counts"], dtype=np.float64),
        signal_counts=np.asarray(batch_numpy["signal_counts"], dtype=np.float64),
        idler_counts=np.asarray(batch_numpy["idler_counts"], dtype=np.float64),
        program_amplitudes=program_amp,
        program_phases_rad=program_phase,
        multi_click_total=multi_total,
        exposure_proxy=np.ones(sample_count, dtype=np.float64),
        cycle_index=cycle_index,
        retrieval_field_vectors=retrieval_rates,
        aod_frequency_features=aod_freq,
        aod_response_features=aod_resp,
        sample_weights=np.ones(sample_count, dtype=np.float64),
    )
    return fit_hierarchical_posterior(arrays, config), arrays


def _train_v2_profile(
    profile: Mapping[str, Any],
    geometry: Any,
    support: Any,
    *,
    seed: int,
    device: Any,
) -> Tuple[V7ScalablePhysicsModelV2, float, float, float, float]:
    if torch is None:
        raise RuntimeError("v2 training requires torch")
    config_payload = {name: profile[name] for name in (
        "width", "latent_tokens", "layers", "attention_heads",
        "feedforward_multiplier", "dropout", "count_smoothing",
        "use_activation_checkpointing",
    )}
    # Physics-math v3/v4/v5 add-ons are opt-in via profile keys; when a
    # profile omits them the defaults (off) apply and the forward / loss
    # path is byte-identical to v2.
    for optional_name in (
        "use_phase_residual", "use_qfi_fisher_weight",
        "use_nb2_nll_observed", "use_count_calibration",
        "use_physical_fisher_weight", "use_adaptive_count_calibration",
        "use_phase_residual_consistency",
        "use_log_rate_self_consistency", "use_frequency_feedback_base",
        "use_poisson_observed",
        "use_spectral_smoothness", "use_operation_equivariance",
        "nb2_observed_weight", "count_calibration_weight",
        "adaptive_count_calibration_weight", "phase_residual_consistency_weight",
        "log_rate_residual_weight", "frequency_feedback_base_weight",
        "poisson_observed_weight",
        "spectral_smoothness_weight", "operation_equivariance_weight",
        "qfi_fisher_weight_floor", "physical_fisher_weight_floor",
    ):
        if optional_name in profile:
            config_payload[optional_name] = profile[optional_name]
    config = V7ScalableModelV2Config.from_mapping(
        config_payload,
        system_type=geometry.system_type,
        operation_count=geometry.operation_count,
        qubit_count=geometry.qubit_count,
    )
    model = V7ScalablePhysicsModelV2(config, support.graph_kernel).to(device)
    torch.manual_seed(int(seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    use_amp = bool(profile.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    training_losses = []
    for step in range(int(profile["optimizer_steps"])):
        batch_numpy = generate_noise_taxonomy_batch(
            geometry, support,
            batch_size=int(profile["batch_size"]),
            seed=int(seed) + step,
            low_count_fraction=float(profile.get("low_count_fraction", 0.6)),
        )
        batch = numpy_batch_to_torch(batch_numpy, device)
        cycle_index = torch.as_tensor(
            build_cycle_index(
                geometry.operation_count, geometry.qubit_count,
                int(profile["batch_size"]), int(seed) + step,
            ), device=device, dtype=torch.long,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(
                batch["operation_inputs"], batch["target_context"],
                batch["signal_counts"], batch["idler_counts"],
                cycle_index=cycle_index,
            )
            loss_weights = {
                "nb2_nll": 1.0, "conditional_nll": 0.35, "teacher_cross_entropy": 0.15,
                "noise_type": 0.25, "noise_coarse_type": 2.0, "noise_strength": 0.5,
                "mixture": 0.2, "dispersion": 0.03, "retrieval": 0.1, "best_fidelity": 0.2,
            }
            # Wire the v3/v4/v5 loss components through only when the
            # profile opted in; otherwise they contribute zero and the
            # trainer is byte-identical to v2.
            if config.use_nb2_nll_observed:
                loss_weights["nb2_nll_observed"] = float(
                    profile.get("nb2_observed_weight", 0.15),
                )
            if config.use_count_calibration:
                loss_weights["count_calibration"] = float(
                    profile.get("count_calibration_weight", 0.02),
                )
            if config.use_adaptive_count_calibration:
                loss_weights["adaptive_count_calibration"] = float(
                    profile.get("adaptive_count_calibration_weight", 0.02),
                )
            if config.use_phase_residual_consistency:
                loss_weights["phase_residual_consistency"] = float(
                    profile.get("phase_residual_consistency_weight", 0.02),
                )
            if config.use_log_rate_self_consistency:
                loss_weights["log_rate_consistency"] = float(
                    profile.get("log_rate_residual_weight", 0.02),
                )
            if config.use_frequency_feedback_base:
                loss_weights["frequency_feedback_base"] = float(
                    profile.get("frequency_feedback_base_weight", 0.02),
                )
            if config.use_poisson_observed:
                loss_weights["poisson_nll_observed"] = float(
                    profile.get("poisson_observed_weight", 0.15),
                )
            if config.use_spectral_smoothness:
                loss_weights["spectral_smoothness"] = float(
                    profile.get("spectral_smoothness_weight", 0.05),
                )
            if config.use_operation_equivariance:
                loss_weights["operation_equivariance"] = float(
                    profile.get("operation_equivariance_weight", 0.05),
                )
            loss, _ = v7_scalable_multitask_loss(output, batch, loss_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(profile.get("gradient_clip_norm", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()
        training_losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    final_loss = float(training_losses[-1]) if training_losses else 0.0
    mean_loss = float(sum(training_losses) / max(len(training_losses), 1))
    return model, elapsed, float(peak), final_loss, mean_loss


def _evaluate_posterior(
    geometry: Any,
    support: Any,
    retrieval_rates: np.ndarray,
    *,
    seed: int,
) -> Dict[str, Any]:
    batch_numpy = generate_noise_taxonomy_batch(
        geometry, support, batch_size=320, seed=int(seed) + 50000,
        low_count_fraction=0.5,
    )
    posterior, arrays = _fit_closed_form(batch_numpy, geometry, retrieval_rates)
    ppc = posterior_predictive_check(posterior, arrays, draw_count=80, seed=int(seed) + 1)
    sbc = simulation_based_calibration(
        lambda arr: _fit_closed_form(batch_numpy, geometry, retrieval_rates)[0],
        arrays, draw_count=20, seed=int(seed) + 2,
    )
    attribution = mechanism_attribution(posterior)
    return {
        "ppc": ppc.to_json_dict(),
        "sbc": sbc,
        "attribution": attribution,
        "posterior_loss": float(posterior.training_loss),
    }


def run_extended_smoke(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = "ml_recon/reports/physics_hybrid_v7/extended_smoke",
    device: Optional[str] = None,
    seed_offsets: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("extended smoke requires torch")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    system_type = str(config["system_type"])
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, retrieval_audit = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates,
        graph_mode_count=int(config["graph_mode_count"]),
        graph_length_mhz=float(config["graph_length_mhz"]),
    )
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    profiles = {
        "local_smoke": config["profiles"]["local_smoke"],
        "mid_smoke": config["profiles"].get("mid_smoke") or {
            "width": 640, "latent_tokens": 64, "layers": 12, "attention_heads": 8,
            "feedforward_multiplier": 3.5, "dropout": 0.0, "count_smoothing": 0.5,
            "use_activation_checkpointing": True, "batch_size": 4, "optimizer_steps": 200,
            "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "low_count_fraction": 0.6, "mixed_precision": True,
        },
        "server_4x4090_plan": config["profiles"].get("server_4x4090") or {
            "width": 1024, "latent_tokens": 128, "layers": 24, "attention_heads": 16,
            "feedforward_multiplier": 4.0, "dropout": 0.05, "count_smoothing": 0.5,
            "use_activation_checkpointing": True,
        },
    }
    # The RTX 4070 Ti ceiling profile (width=1024, layers=20,
    # latent_tokens=128, batch=4, 6.65 GiB peak) is executed like
    # local/mid when present in the config; otherwise it is skipped.
    if config["profiles"].get("ceiling_1024"):
        profiles["ceiling_1024"] = config["profiles"]["ceiling_1024"]
    results: Dict[str, Any] = {
        "schema": "physics_hybrid_v7_extended_smoke_v1",
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "production_subtraction_allowed": False,
        "system_type": system_type,
        "device": str(selected_device),
        "device_name": (
            torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "cpu"
        ),
        "retrieval_audit": retrieval_audit,
        "profiles": {},
    }
    # Executable profiles run 5 seeds + posterior PPC/SBC; the server
    # plan emits a planning-only estimate.
    executable_names = ["local_smoke", "mid_smoke"]
    if "ceiling_1024" in profiles:
        executable_names.append("ceiling_1024")
    ppc_per_seed: Dict[str, list] = {name: [] for name in executable_names}
    sbc_per_seed: Dict[str, list] = {name: [] for name in executable_names}
    five_seed_ppc: Dict[str, Any] = {}
    five_seed_sbc: Dict[str, Any] = {}
    for profile_name, profile in profiles.items():
        profile_summary: Dict[str, Any] = {
            "config_keys": sorted(name for name in profile if not name.startswith("_")),
        }
        if profile_name in executable_names:
            seeds = []
            for seed_offset in seed_offsets:
                seed = int(config["seed"]) + 10000 * (seed_offset + 1)
                model, elapsed, peak, final_loss, mean_loss = _train_v2_profile(
                    profile, geometry, support, seed=seed, device=selected_device,
                )
                seed_summary = {
                    "seed": seed,
                    "seed_offset": int(seed_offset),
                    "parameter_count": int(parameter_count(model)),
                    "elapsed_s": float(elapsed),
                    "peak_reserved_gib": float(peak / 1024 ** 3),
                    "final_training_loss": float(final_loss),
                    "mean_training_loss": float(mean_loss),
                }
                if profile_name in ppc_per_seed:
                    evaluation = _evaluate_posterior(
                        geometry, support, retrieval_rates, seed=seed,
                    )
                    seed_summary.update({
                        "posterior_loss": evaluation["posterior_loss"],
                        "attribution": evaluation["attribution"],
                    })
                    ppc_per_seed[profile_name].append(evaluation["ppc"])
                    sbc_per_seed[profile_name].append(evaluation["sbc"])
                seeds.append(seed_summary)
            profile_summary["seeds"] = seeds
            if profile_name in ppc_per_seed and len(ppc_per_seed[profile_name]) >= 5:
                aggregated = five_seed_aggregator(
                    ppc_per_seed[profile_name],
                    ppc=ppc_per_seed[profile_name][-1],
                    sbc=sbc_per_seed[profile_name][-1],
                )
                five_seed_ppc[profile_name] = aggregated
                five_seed_sbc[profile_name] = sbc_per_seed[profile_name][-1]
        else:
            v1_config = V7ScalableModelConfig.from_mapping(
                {name: profile[name] for name in (
                    "width", "latent_tokens", "layers", "attention_heads",
                    "feedforward_multiplier", "dropout", "count_smoothing",
                    "use_activation_checkpointing",
                )},
                system_type=geometry.system_type,
                operation_count=geometry.operation_count,
                qubit_count=geometry.qubit_count,
            )
            with torch.device("meta"):
                meta_kernel = torch.empty(
                    (geometry.operation_count, geometry.operation_count), device="meta",
                )
                server_model = V7ScalablePhysicsModel(v1_config, meta_kernel)
            params = int(parameter_count(server_model))
            memory = estimate_fsdp_memory_bytes(
                params, v1_config,
                world_size=int(profile.get("world_size", 4)),
                batch_size_per_gpu=int(profile.get("batch_size_per_gpu", 4)),
            )
            profile_summary.update({
                "parameter_count": params,
                "memory_estimate": memory,
                "execution_evidence": False,
            })
        results["profiles"][profile_name] = profile_summary
    results["five_seed_ppc"] = five_seed_ppc
    results["five_seed_sbc"] = five_seed_sbc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    (destination / "v7_extended_smoke_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output-dir",
        default="ml_recon/reports/physics_hybrid_v7/extended_smoke",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    summary = run_extended_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )

    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_extended_smoke"]
