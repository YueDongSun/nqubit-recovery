"""Deployable variable-N feedback student with no privileged input path."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .feedback_moe import V10FeedbackMoE
from .gain_quantiles import signed_monotonic_gain_quantiles

STUDENT_ALLOWED_INPUTS = {
    "noisy_measurement_tokens", "denoised_measurement_tokens", "covariance",
    "target_mode_tokens", "esm_mode_tokens", "physical_mode_tokens",
    "history_tokens", "history_mask", "gate_features", "analytic_action", "mode_mask",
}
PRIVILEGED_NAMES = {
    "clean_measurements", "clean_measurement_tokens", "true_noise_algorithm",
    "true_mechanism", "true_severity", "true_esm_mode_tokens", "true_partition",
    "oracle_action", "state_truth", "partition_truth",
}


class V10FeedbackNStudent(nn.Module):
    model_id = "F_N_RESEARCH_V10"

    def __init__(self, input_dim: int, gate_input_dim: int, width: int = 192,
                 heads: int = 8, blocks: int = 6):
        super().__init__()
        self.mode_input = nn.Linear(input_dim, width)
        self.measurement_input = nn.Linear(input_dim, width)
        self.history_input = nn.Linear(7, width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.mode_encoder = nn.TransformerEncoder(layer, blocks)
        self.moe = V10FeedbackMoE(width, gate_input_dim)
        self.global_head = nn.Linear(width, 12)
        self.recoverability_class_head = nn.Linear(width, 2)
        self.next_measurement_head = nn.Linear(width, 2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        leaked = PRIVILEGED_NAMES & set(inputs)
        if leaked:
            raise ValueError(f"deployable student received privileged inputs: {sorted(leaked)}")
        unknown = set(inputs) - STUDENT_ALLOWED_INPUTS
        if unknown:
            raise ValueError(f"unknown student inputs: {sorted(unknown)}")
        required = STUDENT_ALLOWED_INPUTS - {"history_tokens", "history_mask"}
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"student missing inputs: {sorted(missing)}")
        mode_input = torch.cat((inputs["target_mode_tokens"], inputs["esm_mode_tokens"],
                                inputs["physical_mode_tokens"]), dim=-1)
        modes = self.mode_input(mode_input)
        if ("history_tokens" in inputs) != ("history_mask" in inputs):
            raise ValueError("feedback history tokens and mask must be provided together")
        if "history_tokens" in inputs:
            history = inputs["history_tokens"]
            history_mask = inputs["history_mask"].bool()
            if history.ndim != 4 or history.shape[2] != modes.shape[1] \
                    or history.shape[-1] != 7 \
                    or history_mask.shape != history.shape[:2]:
                raise ValueError("feedback history must have shape [batch,rounds,N,7]")
            encoded_history = self.history_input(history)
            history_weight = history_mask.to(encoded_history.dtype)[:, :, None, None]
            history_context = (encoded_history * history_weight).sum(1) \
                / history_weight.sum(1).clamp_min(1.0)
            modes = modes + history_context
        measurements = self.measurement_input(torch.cat((inputs["noisy_measurement_tokens"],
                                                          inputs["denoised_measurement_tokens"],
                                                          inputs["covariance"]), dim=-1))
        fused, _ = self.cross(modes, measurements, measurements)
        modes = self.mode_encoder(modes + fused,
                                  src_key_padding_mask=~inputs["mode_mask"].bool())
        result = self.moe(modes, inputs["gate_features"], inputs["analytic_action"],
                          inputs["mode_mask"].float())
        pooled = ((modes * inputs["mode_mask"].unsqueeze(-1)).sum(1) /
                  inputs["mode_mask"].sum(1, keepdim=True).clamp_min(1))
        global_out = self.global_head(pooled)
        recoverability_class_logits = self.recoverability_class_head(pooled)
        with torch.autocast(device_type=modes.device.type, enabled=False):
            next_measurement = self.next_measurement_head(modes.float())
        result.update({"recoverability": torch.sigmoid(global_out[:, 0]),
                       "recoverability_class_logits": recoverability_class_logits,
                       "recoverability_class_probs": torch.softmax(
                           recoverability_class_logits, dim=-1),
                       "correctable_fraction": torch.sigmoid(global_out[:, 1]),
                       "irrecoverable_floor": torch.sigmoid(global_out[:, 2]),
                       "gain_quantiles": signed_monotonic_gain_quantiles(
                           global_out[:, 3:8]),
                       "harmful_probability": torch.sigmoid(global_out[:, 8]),
                       "stop_probability": torch.sigmoid(global_out[:, 9]),
                       "noop_gain": global_out[:, 10],
                       "history_gain": global_out[:, 11],
                       "next_measurement": next_measurement,
                       "abstain_probability": torch.maximum(
                           torch.sigmoid(global_out[:, 8]), torch.sigmoid(global_out[:, 9]))})
        return result
