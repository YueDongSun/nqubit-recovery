"""Memory-bounded V8 stop-threshold calibration.

The calibration population and generator seed remain identical to the legacy
V7/V8 path.  One complete synthetic policy batch is generated per system,
then only model/classifier inference is split into configurable microbatches.
Probabilities are concatenated in original row order before any threshold is
fit, so microbatching changes peak forward size rather than calibration data.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ml_recon.physics_hybrid_v7 import train_feedback_models as v7_training


DEFAULT_STOP_CALIBRATION_MICROBATCH_SIZE = 32
STOP_CALIBRATION_MICROBATCH_CONFIG_KEY = "stop_calibration_microbatch_size"
STOP_CALIBRATION_AUDIT_SCHEMA = "v8_batched_stop_calibration_audit_v1"


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("batched V8 stop calibration requires torch")


def _resolved_microbatch_size(
    policy_config: Mapping[str, Any],
    microbatch_size: Optional[int],
) -> int:
    selected = (
        policy_config.get(
            STOP_CALIBRATION_MICROBATCH_CONFIG_KEY,
            DEFAULT_STOP_CALIBRATION_MICROBATCH_SIZE,
        )
        if microbatch_size is None
        else microbatch_size
    )
    if isinstance(selected, bool) or int(selected) <= 0:
        raise ValueError("stop calibration microbatch size must be a positive integer")
    if float(selected) != float(int(selected)):
        raise ValueError("stop calibration microbatch size must be an integer")
    return int(selected)


def _slice_generated_batch(
    batch: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    sample_count: int,
) -> Dict[str, Any]:
    """Slice row-aligned batch fields while preserving static metadata."""

    result: Dict[str, Any] = {}
    for name, value in batch.items():
        array = np.asarray(value)
        result[name] = (
            array[start:stop]
            if array.ndim > 0 and int(array.shape[0]) == int(sample_count)
            else value
        )
    return result


def calibrate_policy_stop_thresholds(
    model: Any,
    contexts: Mapping[str, Any],
    profile: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: "torch.device",
    classifier_prior: Any = None,
    classifier_rho_available: Optional[bool] = None,
    microbatch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Calibrate 3Q/4Q stop heads with one full batch and bounded forwards."""

    _require_torch()
    del profile  # Kept for drop-in compatibility with the legacy seam.
    policy_config = dict(config.get("policy") or {})
    selected_microbatch_size = _resolved_microbatch_size(
        policy_config, microbatch_size
    )
    minimum_specificity = float(
        policy_config.get("stop_calibration_minimum_specificity", 0.80)
    )
    minimum_specificity_by_history = {
        str(key): float(value)
        for key, value in dict(
            policy_config.get("stop_calibration_minimum_specificity_by_history")
            or {"0": 0.99, "1": 0.98, "2_plus": minimum_specificity}
        ).items()
    }
    configured_samples = int(
        policy_config.get(
            "stop_calibration_samples",
            config["selection"]["synthetic_evaluation_samples"],
        )
    )
    sample_count = max(64, configured_samples)
    result: Dict[str, Any] = {}
    model.eval()

    for offset, (system_type, context) in enumerate(contexts.items()):
        generator_seed = int(seed + 7919 * (offset + 1))
        batch = v7_training.generate_feedback_policy_batch(
            context.arrays,
            context.profile,
            batch_size=sample_count,
            seed=generator_seed,
        )
        graph = torch.from_numpy(context.arrays.graph_kernel).to(device)
        probability_parts = []
        forward_batch_sizes = []
        classifier_available_row_count = 0

        for start in range(0, sample_count, selected_microbatch_size):
            stop = min(start + selected_microbatch_size, sample_count)
            microbatch = _slice_generated_batch(
                batch,
                start=start,
                stop=stop,
                sample_count=sample_count,
            )
            tensor = v7_training._synthetic_tensors(microbatch, device=device)
            classifier_mask = v7_training._classifier_rho_availability_mask(
                tensor["rho_available_mask"], classifier_rho_available
            )
            classifier_available_row_count += int(
                torch.sum(classifier_mask > 0.5).detach().cpu()
            )
            with torch.no_grad():
                tensor.update(v7_training._classifier_prior_features(
                    classifier_prior,
                    system_type=system_type,
                    target_features=tensor["target_features"],
                    rho_available_mask=classifier_mask,
                    core_rho=tensor["current_rho"],
                ))
                encoded = model.encoder(
                    tensor["operation_features"],
                    tensor["target_features"],
                    graph,
                    system_type=system_type,
                    classifier_latent=tensor.get("classifier_latent"),
                    classifier_uncertainty=tensor.get(
                        "classifier_uncertainty"
                    ),
                )
                policy = model.model_b(
                    encoded["state_embedding"],
                    tensor["history"],
                    tensor["history_mask"],
                    tensor["action_mask"],
                    system_type=system_type,
                )
                probability_parts.append(
                    torch.sigmoid(policy["stop_logit"])
                    .float()
                    .cpu()
                    .numpy()
                )
            forward_batch_sizes.append(int(stop - start))

        probability = np.concatenate(probability_parts, axis=0)
        if int(probability.shape[0]) != sample_count:
            raise RuntimeError("microbatch stop probabilities lost calibration rows")
        history_length = np.sum(batch["history_mask"], axis=1).astype(np.int64)
        teacher_mode = np.asarray(batch["teacher_mode"], dtype=np.int64)
        target = teacher_mode == v7_training.ACTION_MODE_NAMES.index("stop")
        global_metrics = v7_training._fit_stop_probability_threshold(
            probability,
            target,
            minimum_specificity=minimum_specificity,
        )
        bucket_masks = {
            "0": history_length == 0,
            "1": history_length == 1,
            "2_plus": history_length >= 2,
        }
        bucket_metrics: Dict[str, Dict[str, float]] = {}
        for bucket_name, bucket_mask in bucket_masks.items():
            if int(np.sum(bucket_mask)) < 16:
                fallback = dict(global_metrics)
                fallback["fallback_to_global"] = 1.0
                bucket_metrics[bucket_name] = fallback
            else:
                bucket_metrics[bucket_name] = (
                    v7_training._fit_stop_probability_threshold(
                        probability[bucket_mask],
                        target[bucket_mask],
                        minimum_specificity=minimum_specificity_by_history[
                            bucket_name
                        ],
                    )
                )
                bucket_metrics[bucket_name]["fallback_to_global"] = 0.0
        result[system_type] = {
            **global_metrics,
            "by_history_bucket": bucket_metrics,
            "microbatch_audit": {
                "schema": STOP_CALIBRATION_AUDIT_SCHEMA,
                "system_type": str(system_type),
                "generator_seed": generator_seed,
                "full_batch_generated_once": True,
                "full_calibration_sample_count": sample_count,
                "configured_microbatch_size": selected_microbatch_size,
                "microbatch_count": len(forward_batch_sizes),
                "forward_batch_sizes": forward_batch_sizes,
                "peak_forward_batch_size": max(forward_batch_sizes),
                "probability_row_count": int(probability.shape[0]),
                "probabilities_concatenated_before_threshold_fit": True,
                "threshold_fit_population_unchanged": True,
                "no_grad_forward": True,
                "classifier_prior_enabled": classifier_prior is not None,
                "classifier_rho_available_override": classifier_rho_available,
                "classifier_available_row_count": classifier_available_row_count,
                "classifier_missing_row_count": (
                    sample_count - classifier_available_row_count
                ),
            },
        }
    return result


__all__ = [
    "DEFAULT_STOP_CALIBRATION_MICROBATCH_SIZE",
    "STOP_CALIBRATION_AUDIT_SCHEMA",
    "STOP_CALIBRATION_MICROBATCH_CONFIG_KEY",
    "calibrate_policy_stop_thresholds",
]
