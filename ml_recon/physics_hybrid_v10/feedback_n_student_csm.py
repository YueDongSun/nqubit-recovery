"""Optional CSM side-input student adapter with legacy-checkpoint parity.

The historical ``feedback_n_student.py`` remains untouched as the A0-A4
baseline.  This adapter mirrors its architecture, adds only the zero-init
relational sidecar, and provides an explicit checkpoint conversion seam.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn

from .feedback_moe import V10FeedbackMoE
from .feedback_n_student import PRIVILEGED_NAMES, STUDENT_ALLOWED_INPUTS
from .gain_quantiles import signed_monotonic_gain_quantiles
from .partition_router import V10ZeroInitRelationalRouter


CSM_SIDE_INPUTS = frozenset({
    "partition_context", "partition_current_context", "partition_target_context",
    "partition_current_defined", "partition_target_defined", "partition_mask",
    "csm_node_context", "csm_edge_context", "csm_global_context",
    "structure_confidence", "partition_current_confidence",
    "partition_target_confidence",
    "real_sector_confidence", "w_coherence_confidence",
})
CSM_STUDENT_ALLOWED_INPUTS = frozenset(STUDENT_ALLOWED_INPUTS) | CSM_SIDE_INPUTS
DEFAULT_CSM_NODE_CONTEXT_DIM = 9
DEFAULT_CSM_GLOBAL_CONTEXT_DIM = 4


class V10FeedbackNStudentCSM(nn.Module):
    """V10 student with optional, zero-init CSM/partition side inputs."""

    model_id = "F_N_RESEARCH_V10_CSM_SHADOW"

    def __init__(self, input_dim: int, gate_input_dim: int, width: int = 192,
                 heads: int = 8, blocks: int = 6,
                 csm_node_context_dim: int = DEFAULT_CSM_NODE_CONTEXT_DIM,
                 csm_global_context_dim: int = DEFAULT_CSM_GLOBAL_CONTEXT_DIM):
        super().__init__()
        self.mode_input = nn.Linear(input_dim, width)
        self.measurement_input = nn.Linear(input_dim, width)
        self.history_input = nn.Linear(7, width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.mode_encoder = nn.TransformerEncoder(layer, blocks)
        self.relational_router = V10ZeroInitRelationalRouter(
            width, node_context_dim=csm_node_context_dim,
            global_context_dim=csm_global_context_dim)
        self.moe = V10FeedbackMoE(width, gate_input_dim)
        self.global_head = nn.Linear(width, 12)
        self.recoverability_class_head = nn.Linear(width, 2)
        self.next_measurement_head = nn.Linear(width, 2)

    def load_legacy_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True):
        """Load a historical student state dict and keep new sidecar gains at zero."""
        current = self.state_dict()
        converted = {}
        unexpected = []
        for key, value in state_dict.items():
            if key in current:
                converted[key] = value
            else:
                unexpected.append(key)
        missing = sorted(key for key in set(current) - set(converted)
                        if not key.startswith("relational_router."))
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"legacy student conversion mismatch; missing={missing}, unexpected={unexpected}")
        result = self.load_state_dict(converted, strict=False)
        if float(self.relational_router.alpha_partition.detach()) != 0.0 \
                or float(self.relational_router.alpha_csm.detach()) != 0.0:
            raise RuntimeError("legacy conversion must retain zero-init relational gains")
        return result

    def forward(
        self, inputs: Dict[str, torch.Tensor], *,
        hierarchy_residual: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        leaked = PRIVILEGED_NAMES & set(inputs)
        if leaked:
            raise ValueError(f"deployable CSM student received privileged inputs: {sorted(leaked)}")
        unknown = set(inputs) - CSM_STUDENT_ALLOWED_INPUTS
        if unknown:
            raise ValueError(f"unknown CSM student inputs: {sorted(unknown)}")
        required = set(STUDENT_ALLOWED_INPUTS) - {"history_tokens", "history_mask"}
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"CSM student missing inputs: {sorted(missing)}")
        mode_input = torch.cat((inputs["target_mode_tokens"], inputs["esm_mode_tokens"],
                                inputs["physical_mode_tokens"]), dim=-1)
        modes = self.mode_input(mode_input)
        if ("history_tokens" in inputs) != ("history_mask" in inputs):
            raise ValueError("feedback history tokens and mask must be provided together")
        if "history_tokens" in inputs:
            history = inputs["history_tokens"]
            history_mask = inputs["history_mask"].bool()
            if history.ndim != 4 or history.shape[2] != modes.shape[1] \
                    or history.shape[-1] != 7 or history_mask.shape != history.shape[:2]:
                raise ValueError("feedback history must have shape [batch,rounds,N,7]")
            encoded_history = self.history_input(history)
            history_weight = history_mask.to(encoded_history.dtype)[:, :, None, None]
            history_context = (encoded_history * history_weight).sum(1) \
                / history_weight.sum(1).clamp_min(1.0)
            modes = modes + history_context
        measurements = self.measurement_input(torch.cat((
            inputs["noisy_measurement_tokens"],
            inputs["denoised_measurement_tokens"], inputs["covariance"]), dim=-1))
        fused, _ = self.cross(modes, measurements, measurements)
        modes = self.mode_encoder(modes + fused,
                                  src_key_padding_mask=~inputs["mode_mask"].bool())
        if hierarchy_residual is not None:
            if not isinstance(hierarchy_residual, torch.Tensor) \
                    or hierarchy_residual.shape != modes.shape \
                    or hierarchy_residual.dtype != modes.dtype \
                    or hierarchy_residual.device != modes.device \
                    or not bool(torch.isfinite(hierarchy_residual).all()):
                raise ValueError("hierarchy residual must match the finite inherited mode representation")
            modes = modes + hierarchy_residual
        structure_confidence_value = inputs.get("structure_confidence")
        if structure_confidence_value is None \
                and "real_sector_confidence" in inputs \
                and "w_coherence_confidence" in inputs:
            structure_confidence_value = torch.minimum(
                inputs["real_sector_confidence"],
                inputs["w_coherence_confidence"])
        routed = self.relational_router(
            modes,
            partition_context=inputs.get("partition_context"),
            partition_current_context=inputs.get("partition_current_context"),
            partition_target_context=inputs.get("partition_target_context"),
            partition_current_defined=inputs.get("partition_current_defined"),
            partition_target_defined=inputs.get("partition_target_defined"),
            partition_mask=inputs.get("partition_mask"),
            csm_node_context=inputs.get("csm_node_context"),
            csm_edge_context=inputs.get("csm_edge_context"),
            csm_global_context=inputs.get("csm_global_context"),
            structure_confidence_value=structure_confidence_value,
            partition_current_confidence=inputs.get("partition_current_confidence"),
            partition_target_confidence=inputs.get("partition_target_confidence"),
            mode_mask=inputs["mode_mask"],
        )
        modes = routed["mode_tokens"]
        result = self.moe(modes, inputs["gate_features"], inputs["analytic_action"],
                          inputs["mode_mask"].float())
        pooled = ((modes * inputs["mode_mask"].unsqueeze(-1)).sum(1) /
                  inputs["mode_mask"].sum(1, keepdim=True).clamp_min(1))
        global_out = self.global_head(pooled)
        recoverability_class_logits = self.recoverability_class_head(pooled)
        with torch.autocast(device_type=modes.device.type, enabled=False):
            next_measurement = self.next_measurement_head(modes.float())
        result.update({
            "recoverability": torch.sigmoid(global_out[:, 0]),
            "recoverability_class_logits": recoverability_class_logits,
            "recoverability_class_probs": torch.softmax(recoverability_class_logits, dim=-1),
            "correctable_fraction": torch.sigmoid(global_out[:, 1]),
            "irrecoverable_floor": torch.sigmoid(global_out[:, 2]),
            "gain_quantiles": signed_monotonic_gain_quantiles(global_out[:, 3:8]),
            "harmful_probability": torch.sigmoid(global_out[:, 8]),
            "stop_probability": torch.sigmoid(global_out[:, 9]),
            "noop_gain": global_out[:, 10],
            "history_gain": global_out[:, 11],
            "next_measurement": next_measurement,
            "relational_bias": routed["relational_bias"],
            "partition_channels": routed.get("partition_channels"),
            "partition_defined": routed.get("partition_defined"),
            "structure_confidence": routed["structure_confidence"],
            "alpha_partition": routed["alpha_partition"],
            "alpha_csm": routed["alpha_csm"],
            "abstain_probability": torch.maximum(
                torch.sigmoid(global_out[:, 8]), torch.sigmoid(global_out[:, 9])),
        })
        return result


__all__ = [
    "CSM_SIDE_INPUTS", "CSM_STUDENT_ALLOWED_INPUTS",
    "DEFAULT_CSM_GLOBAL_CONTEXT_DIM", "DEFAULT_CSM_NODE_CONTEXT_DIM",
    "V10FeedbackNStudentCSM",
]
