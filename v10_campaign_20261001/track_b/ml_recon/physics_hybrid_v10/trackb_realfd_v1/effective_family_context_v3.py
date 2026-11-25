"""Low-capacity public measurement-family contrasts for Track B P1."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from .measurement_context import Effective_measurement_context


VARIANT_NAME = "effective_family_context_v3"
MODEL_SCHEMA = "TrackBEffectiveFamilyContextObservationV3"
MODEL_CONFIG_SCHEMA = "TrackBEffectiveFamilyContextConfigV3"
CHECKPOINT_SCHEMA = "TrackBEffectiveFamilyContextCheckpointV3"
RECEIPT_SCHEMA = "TrackBEffectiveFamilyContextTrainingReceiptV3"
PREFLIGHT_SCHEMA = "TrackBEffectiveFamilyContextPreflightReceiptV3"
FAMILY_BRANCHES = (
    "signal_logit", "coin_given_signal_integrated_h", "idler_given_no_signal_integrated_h",
)


def _helmert_contrast_matrix(family_count: int) -> tuple[tuple[float, ...], ...]:
    """Return the deterministic orthonormal K-by-(K-1) Helmert basis."""
    if type(family_count) is not int or family_count < 2:
        raise ValueError("Helmert family contrast requires at least two families")
    rows = [[0.0] * (family_count - 1) for _ in range(family_count)]
    for column in range(family_count - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        for row in range(column + 1):
            rows[row][column] = 1.0 / scale
        rows[column + 1][column] = -(column + 1) / scale
    return tuple(tuple(value for value in row) for row in rows)


def _bounded_radial_scale(vectors: torch.Tensor, bound: float) -> torch.Tensor:
    """Smoothly bound a vector norm while preserving its centered direction."""
    norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    small = norm / bound < 1.0e-4
    # Do not differentiate a zero norm in either branch. Merely masking the
    # result after sqrt/norm leaves undefined second derivatives in autograd.
    regular_vectors = torch.where(small, torch.ones_like(vectors), vectors)
    regular_ratio = torch.linalg.vector_norm(regular_vectors, dim=1, keepdim=True) / bound
    regular = torch.tanh(regular_ratio) / regular_ratio
    ratio_squared = vectors.square().sum(dim=1, keepdim=True) / (bound * bound)
    series = 1 - ratio_squared / 3 + 2 * ratio_squared.square() / 15
    return torch.where(small, series, regular)


class Effective_family_context(nn.Module):
    """Three bounded Helmert coefficient vectors with no ID embeddings.

    Coefficients are bounded in the registered orthonormal family basis.  The
    resulting operation field is centered over visible operations separately
    for every row and response branch, then radially scaled so its available-
    operation L2 norm is below the registered total-correction bound.
    """

    def __init__(self, context: Effective_measurement_context, *, prototype: torch.Tensor) -> None:
        super().__init__()
        context.validate()
        self.context = context
        # new_zeros is deterministic and consumes no RNG after the V2 backbone.
        self.raw_coefficients = nn.Parameter(prototype.new_zeros(
            (len(FAMILY_BRANCHES), len(context.family_vocabulary) - 1),
        ))

    def forward(
        self, batch: Mapping[str, Any], *, operation_mask: torch.Tensor,
        visible_operation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if operation_mask.dtype != torch.bool or visible_operation_mask.dtype != torch.bool \
                or operation_mask.ndim != 2 or visible_operation_mask.shape != operation_mask.shape \
                or bool((visible_operation_mask & ~operation_mask).any()) \
                or not bool(visible_operation_mask.any(-1).all()):
            raise ValueError("family context requires aligned bool available/visible masks")
        if "visible_operation_mask" in batch:
            supplied = batch["visible_operation_mask"]
            if not isinstance(supplied, torch.Tensor) or supplied.dtype != torch.bool \
                    or supplied.shape != visible_operation_mask.shape \
                    or not torch.equal(supplied, visible_operation_mask):
                raise ValueError("batch visible-operation mask differs from the family context mask")
        family_index_values = self.context.batch_family_indices(
            batch, operation_count=operation_mask.shape[1],
        )
        basis = self.raw_coefficients.new_tensor(
            _helmert_contrast_matrix(len(self.context.family_vocabulary)),
        )
        bounded_coefficients = self.context.maximum_family_coefficient * torch.tanh(
            self.raw_coefficients,
        )
        family_values = basis @ bounded_coefficients.transpose(0, 1)
        family_indices = torch.tensor(
            family_index_values, dtype=torch.long, device=self.raw_coefficients.device,
        )
        operation_values = family_values.index_select(0, family_indices)
        visible = visible_operation_mask[..., None]
        visible_mean = (
            operation_values[None] * visible
        ).sum(1, keepdim=True) / visible.sum(1, keepdim=True)
        centered = torch.where(
            operation_mask[..., None], operation_values[None] - visible_mean,
            torch.zeros_like(operation_values)[None],
        )
        scale = _bounded_radial_scale(
            centered, self.context.maximum_family_correction_l2,
        )
        correction = centered * scale
        return {
            "effective_family_coefficients": bounded_coefficients,
            "effective_family_correction": correction,
            "effective_family_correction_l2": torch.linalg.vector_norm(correction, dim=1),
            "effective_family_branches": FAMILY_BRANCHES,
            "measurement_context_sha": self.context.sha256(),
            "family_coefficient_interpretation": self.context.coefficient_interpretation,
            "family_correction_interpretation": self.context.correction_interpretation,
        }
