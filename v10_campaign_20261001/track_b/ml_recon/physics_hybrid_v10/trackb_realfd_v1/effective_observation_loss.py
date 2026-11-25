"""New observation-only loss; no raw-count anchor or free Gaussian variance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from .effective_observation_v2 import MODEL_SCHEMA
from .effective_family_context_v3 import MODEL_SCHEMA as FAMILY_MODEL_SCHEMA
from .losses import native_global_multi_nll, native_masked_four_cell_nll


@dataclass(frozen=True)
class Effective_observation_loss_config:
    global_multi_weight: float = 1.0
    effective_response_prior_weight: float = 0.001
    prepared_state_prior_weight: float = 0.001
    schema: str = "TrackBEffectiveSingletonLossV2"

    def validate(self) -> None:
        if self.schema != "TrackBEffectiveSingletonLossV2":
            raise ValueError("effective observation loss schema is invalid")
        for value in (self.global_multi_weight, self.effective_response_prior_weight, self.prepared_state_prior_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("effective observation loss weights must be finite nonnegative")
        if self.global_multi_weight <= 0:
            raise ValueError("observed native global multi likelihood remains active")


def effective_observation_objective(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], *,
    config: Effective_observation_loss_config = Effective_observation_loss_config(),
) -> dict[str, torch.Tensor]:
    """Native primary score plus weak dimensionless hypothesis priors.

Prepared state is not a target. Its soft prior acts only on available command
rows and can be overridden by the native observation likelihood. Priors do
not scale with R, exposure, or raw cell magnitudes. There is no concentration
reward, reliability floor, causal label or operation-broadcast multi input.
"""
    config.validate()
    if prediction.get("model_schema") not in {MODEL_SCHEMA, FAMILY_MODEL_SCHEMA}:
        raise ValueError("V2 effective objective cannot reinterpret another model")
    four_cell = native_masked_four_cell_nll(prediction, batch)
    global_multi, global_rows = native_global_multi_nll(prediction, batch)
    response_prior = prediction["effective_offsets"].square().mean() + prediction["effective_contrasts"].square().mean()
    state_distance = (prediction["latent_state"] - prediction["prepared_prior"]).abs().square().sum((-1, -2))
    prior_mask = prediction["prepared_prior_available"]
    state_prior = (state_distance * prior_mask).sum() / prior_mask.sum().clamp_min(1)
    total = four_cell + config.global_multi_weight * global_multi \
        + config.effective_response_prior_weight * response_prior \
        + config.prepared_state_prior_weight * state_prior
    if not bool(torch.isfinite(total)):
        raise RuntimeError("effective observation objective became nonfinite")
    return {
        "loss": total, "four_cell_nll": four_cell, "global_multi_nll": global_multi,
        "global_multi_available_rows": total.new_tensor(global_rows),
        "effective_response_prior": response_prior, "prepared_state_prior": state_prior,
    }
