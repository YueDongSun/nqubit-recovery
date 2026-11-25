"""GPU memory pressure test for the V7 v2 scalable model.

The test sweeps width / layers / batch_size combinations on the local
4070 Ti (11.99 GiB) and reports the peak reserved memory for each
profile.  Profiles that exceed the device budget are tagged as
``out_of_memory`` and skipped; the largest profile that runs is the
recommended long-training configuration.

Usage::

    python -m ml_recon.physics_hybrid_v7.vram_sweep \
        --output ml_recon/reports/physics_hybrid_v7/vram_sweep.json

The script is research-only: it never imports ExperimentRunner, never
writes matrix/ml_delta, never generates AWG sequences.  The output is a
JSON summary with one record per profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .geometry import build_operation_frequency_geometry
from .noise_taxonomy import (
    build_synthetic_physics_support,
    generate_noise_taxonomy_batch,
    numpy_batch_to_torch,
)
from .physical_ceiling import load_retrieval_rate_vectors
from .scalable_model_v2 import (
    V7ScalableModelV2Config,
    V7ScalablePhysicsModelV2,
)


PROFILES: List[Dict[str, Any]] = [
    # Smoke, very small.
    {"name": "smoke_64_2", "width": 64, "latent_tokens": 4, "layers": 2, "batch_size": 4},
    {"name": "smoke_128_4", "width": 128, "latent_tokens": 8, "layers": 4, "batch_size": 4},
    # Local 4070 Ti baseline (matches local_smoke config).
    {"name": "local_smoke", "width": 384, "latent_tokens": 48, "layers": 8, "batch_size": 8},
    {"name": "local_smoke_x2", "width": 384, "latent_tokens": 48, "layers": 8, "batch_size": 16},
    # Mid profile from the configurable smoke.
    {"name": "mid_smoke", "width": 512, "latent_tokens": 64, "layers": 12, "batch_size": 4},
    {"name": "mid_smoke_x2", "width": 512, "latent_tokens": 64, "layers": 12, "batch_size": 8},
    # Large profile sized for the 11.99 GiB ceiling.
    {"name": "large_640", "width": 640, "latent_tokens": 96, "layers": 16, "batch_size": 4},
    {"name": "large_768", "width": 768, "latent_tokens": 96, "layers": 16, "batch_size": 4},
    {"name": "large_896", "width": 896, "latent_tokens": 128, "layers": 16, "batch_size": 2},
    {"name": "ceiling_1024", "width": 1024, "latent_tokens": 128, "layers": 20, "batch_size": 2},
    {"name": "ceiling_1024_x2", "width": 1024, "latent_tokens": 128, "layers": 20, "batch_size": 4},
    # Pushing past the existing ceiling toward the 11.99 GiB wall.
    {"name": "ceiling_1280", "width": 1280, "latent_tokens": 160, "layers": 24, "batch_size": 4},
    {"name": "ceiling_1280_deep", "width": 1280, "latent_tokens": 160, "layers": 28, "batch_size": 4},
    {"name": "ceiling_1536", "width": 1536, "latent_tokens": 192, "layers": 24, "batch_size": 2},
    {"name": "ceiling_2048", "width": 2048, "latent_tokens": 256, "layers": 24, "batch_size": 2},
]


def _build_model(width: int, layers: int, latent_tokens: int, geometry, graph_kernel):
    config = V7ScalableModelV2Config.from_mapping(
        {
            "width": int(width),
            "latent_tokens": int(latent_tokens),
            "layers": int(layers),
            "attention_heads": 8,
            "feedforward_multiplier": 3.5,
            "dropout": 0.0,
            "count_smoothing": 0.5,
            "use_activation_checkpointing": True,
            "use_phase_residual": False,
            "use_qfi_fisher_weight": False,
            "use_aod_positive_response": True,
            "use_retrieval_field_amplitude": True,
            "use_cycle_random_effect": True,
            "use_multi_excitation_mixture": True,
        },
        system_type=geometry.system_type,
        operation_count=geometry.operation_count,
        qubit_count=geometry.qubit_count,
    )
    return V7ScalablePhysicsModelV2(config, graph_kernel), config


def _summarise_profile(
    name: str,
    width: int,
    layers: int,
    latent_tokens: int,
    batch_size: int,
    geometry,
    support,
    device: "torch.device",
    step_count: int,
) -> Dict[str, Any]:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, config = _build_model(width, layers, latent_tokens, geometry, support.graph_kernel)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        model.train()
        peak_reserved = 0
        last_loss = 0.0
        for step in range(int(step_count)):
            batch_numpy = generate_noise_taxonomy_batch(
                geometry, support,
                batch_size=int(batch_size), seed=20260807 + step,
                low_count_fraction=0.6,
            )
            batch = numpy_batch_to_torch(batch_numpy, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch["operation_inputs"], batch["target_context"],
                    batch["signal_counts"], batch["idler_counts"],
                )
                # Use the same loss call shape as a real trainer; here we
                # only care about peak memory, so NB2 NLL alone is enough.
                from .scalable_model import v7_scalable_multitask_loss
                loss, _ = v7_scalable_multitask_loss(
                    output, batch,
                    {
                        "nb2_nll": 1.0, "conditional_nll": 0.35,
                        "teacher_cross_entropy": 0.15, "noise_type": 0.25,
                        "noise_coarse_type": 2.0, "noise_strength": 0.5,
                        "mixture": 0.2, "dispersion": 0.03,
                        "retrieval": 0.1, "best_fidelity": 0.2,
                        "nb2_nll_observed": 0.0, "count_calibration": 0.0,
                    },
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            last_loss = float(loss.detach().cpu())
            if device.type == "cuda":
                peak_reserved = max(
                    peak_reserved,
                    int(torch.cuda.max_memory_reserved(device)),
                )
        param_count = int(sum(p.numel() for p in model.parameters()))
        return {
            "profile": name,
            "width": width,
            "layers": layers,
            "latent_tokens": latent_tokens,
            "batch_size": batch_size,
            "parameter_count": param_count,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_gib": peak_reserved / 1024 ** 3,
            "step_count": int(step_count),
            "last_loss": last_loss,
            "out_of_memory": False,
            "execution_evidence": True,
            "research_only": True,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "OOM" in str(exc).upper():
            return {
                "profile": name,
                "width": width,
                "layers": layers,
                "latent_tokens": latent_tokens,
                "batch_size": batch_size,
                "peak_reserved_bytes": 0,
                "peak_reserved_gib": 0.0,
                "step_count": 0,
                "last_loss": 0.0,
                "out_of_memory": True,
                "execution_evidence": False,
                "research_only": True,
                "error": str(exc).splitlines()[0],
            }
        raise


def run_vram_sweep(
    *,
    output_path: str | Path,
    system_type: str = "4qubit",
    step_count: int = 20,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("vram_sweep requires torch")
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, _ = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates, graph_mode_count=12, graph_length_mhz=8.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    device_total_gib = (
        torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        if device.type == "cuda" else 0.0
    )
    profiles_summary = []
    for entry in PROFILES:
        record = _summarise_profile(
            entry["name"], entry["width"], entry["layers"],
            entry["latent_tokens"], entry["batch_size"],
            geometry, support, device, step_count,
        )
        profiles_summary.append(record)
    summary = {
        "schema": "v7_vram_sweep_v1",
        "system_type": system_type,
        "device": str(device),
        "device_name": device_name,
        "device_total_gib": device_total_gib,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "profiles": profiles_summary,
    }
    Path(output_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="ml_recon/reports/physics_hybrid_v7/vram_sweep.json",
    )
    parser.add_argument("--system", default="4qubit", choices=["3qubit", "4qubit"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    summary = run_vram_sweep(
        output_path=args.output, system_type=args.system, step_count=args.steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = ["run_vram_sweep", "PROFILES"]


if __name__ == "__main__":
    raise SystemExit(main())