"""Hard physical contracts used by the V8 feedback model.

The helpers in this module deliberately separate identities that are easy to
conflate in a learned world model:

* an authoritative exposure from a numerical reference exposure;
* a conditional density operator inside the single-excitation W subspace from
  the probability mass that the experiment actually occupies that subspace;
* an action-induced residual from the uncontrolled/history-conditioned drift.

The module is research/shadow only and has no experiment-control imports.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # pragma: no cover - repository supports NumPy-only installs
    torch = None
    functional = None
    nn = None

from ml_recon.physics_hybrid_v7.feedback_synthetic import HISTORY_FEATURE_NAMES


HISTORY_IDENTIFICATION_RESPONSE_NAMES = (
    "utility_delta",
    "retrieval_delta",
    "multi_risk_delta",
)


def _batch_vector(
    value: "torch.Tensor | float",
    *,
    batch_size: int,
    device: "torch.device",
    dtype: "torch.dtype",
    name: str,
) -> "torch.Tensor":
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(batch_size)
    if tensor.shape == (batch_size, 1):
        return tensor[:, 0]
    if tensor.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar, [batch], or [batch, 1]")
    return tensor


def resolve_exposure(
    *,
    batch_size: int,
    reference_exposure_s: float,
    device: "torch.device",
    dtype: "torch.dtype",
    exposure_s: Optional["torch.Tensor | float"] = None,
    exposure_known_mask: Optional["torch.Tensor | float"] = None,
) -> Dict[str, "torch.Tensor"]:
    """Resolve rate exposure without turning a legacy zero into a fact.

    ``effective_exposure_s`` is always positive, but for an unknown row it is
    only a normalization reference and must not supervise a per-second rate.
    It equals the requested exposure only when the provenance mask is true.
    Otherwise the world model uses its independent uncalibrated-total head and
    ``rate_equivariance_verifiable`` is false.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("V8 exposure resolution requires torch")
    if batch_size <= 0 or float(reference_exposure_s) <= 0.0:
        raise ValueError("batch_size and reference_exposure_s must be positive")

    exposure_supplied = exposure_s is not None
    requested = _batch_vector(
        reference_exposure_s if exposure_s is None else exposure_s,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        name="exposure_s",
    )
    if bool(torch.any(requested < 0.0)):
        raise ValueError("exposure_s cannot be negative")

    if exposure_known_mask is None:
        # A positive number is not provenance.  Synthetic generators and the
        # prepared-command bridge must assert the mask explicitly; legacy
        # measured rows therefore remain unknown even when they contain a
        # nominal duration.
        known = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        supplied_mask = _batch_vector(
            exposure_known_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name="exposure_known_mask",
        ) > 0.5
        if bool(torch.any(supplied_mask & (requested <= 0.0))):
            raise ValueError("a known exposure must be strictly positive")
        known = supplied_mask & (requested > 0.0)

    reference = torch.full(
        (batch_size,),
        float(reference_exposure_s),
        dtype=dtype,
        device=device,
    )
    effective = torch.where(known, requested, reference)
    known_float = known.to(dtype=dtype)
    return {
        "input_exposure_s": requested,
        "effective_exposure_s": effective,
        "exposure_supplied_mask": torch.full(
            (batch_size,),
            float(exposure_supplied),
            dtype=dtype,
            device=device,
        ),
        "exposure_known_mask": known_float,
        "reference_exposure_used_mask": 1.0 - known_float,
        "rate_equivariance_verifiable": known_float,
    }


def bounded_control_residual(
    raw_residual: "torch.Tensor",
    *,
    bound: float,
) -> "torch.Tensor":
    """Map a zero-preserving raw control response into a hard open bound."""

    if float(bound) <= 0.0:
        raise ValueError("residual bound must be positive")
    return float(bound) * torch.tanh(raw_residual)


def analytic_history_identification(
    history: "torch.Tensor",
    history_mask: "torch.Tensor",
    *,
    action_dim: int,
    action_step_limits: Sequence[float],
    response_scales: Sequence[float],
) -> "torch.Tensor":
    """Compute the frozen 26-value secant/system-identification summary.

    The mask is applied before every statistic.  Consequently arbitrary bytes
    in padded history positions cannot change the result.
    """

    if history.ndim != 3 or history_mask.shape != history.shape[:2]:
        raise ValueError("history/history_mask shape changed")
    if history.shape[-1] != len(HISTORY_FEATURE_NAMES):
        raise ValueError("history feature contract changed")
    if len(action_step_limits) != action_dim:
        raise ValueError("action_step_limits and action_dim disagree")
    if len(response_scales) != len(HISTORY_IDENTIFICATION_RESPONSE_NAMES):
        raise ValueError("history response scale contract changed")

    dtype = history.dtype
    device = history.device
    valid_boolean = history_mask > 0.5
    valid = valid_boolean.to(dtype=dtype)
    masked_history = torch.where(
        valid_boolean[:, :, None],
        history,
        torch.zeros_like(history),
    )
    limits = torch.as_tensor(action_step_limits, dtype=dtype, device=device)
    normalized_action = (
        masked_history[:, :, :action_dim] / limits[None, None, :]
    ) * valid[:, :, None]
    information = torch.square(normalized_action).sum(dim=1)

    response_gradients = []
    for response_name, response_scale in zip(
        HISTORY_IDENTIFICATION_RESPONSE_NAMES,
        response_scales,
    ):
        response = torch.tanh(
            masked_history[:, :, HISTORY_FEATURE_NAMES.index(response_name)]
            / float(response_scale)
        ) * valid
        numerator = (normalized_action * response[:, :, None]).sum(dim=1)
        response_gradients.append(numerator / (information + 0.25))

    valid_count = valid.sum(dim=1).clamp_min(1.0)
    accepted_rate = (
        masked_history[:, :, HISTORY_FEATURE_NAMES.index("accepted")] * valid
    ).sum(dim=1) / valid_count
    rollback_rate = (
        masked_history[:, :, HISTORY_FEATURE_NAMES.index("rollback")] * valid
    ).sum(dim=1) / valid_count
    return torch.cat(
        (
            *response_gradients,
            torch.log1p(information),
            accepted_rate[:, None],
            rollback_rate[:, None],
        ),
        dim=1,
    )


if nn is not None:

    class ConditionalWCoreDensityHead(nn.Module):
        """PSD, trace-one density conditioned on the W subspace being occupied."""

        def __init__(
            self,
            width: int,
            dimension: int,
            *,
            mass_enabled: bool,
        ) -> None:
            super().__init__()
            self.dimension = int(dimension)
            self.mass_enabled = bool(mass_enabled)
            lower_count = self.dimension * (self.dimension + 1) // 2
            self.cholesky_projection = nn.Linear(width, 2 * lower_count)
            self.mass_projection = nn.Linear(width, 1)
            indices = torch.tril_indices(self.dimension, self.dimension)
            self.register_buffer("lower_rows", indices[0], persistent=False)
            self.register_buffer("lower_columns", indices[1], persistent=False)

        def forward(self, latent: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            with torch.autocast(device_type=latent.device.type, enabled=False):
                real_value, imag_value = self.cholesky_projection(
                    latent.float()
                ).chunk(2, dim=-1)
                batch_size = latent.shape[0]
                real = torch.zeros(
                    batch_size,
                    self.dimension,
                    self.dimension,
                    dtype=torch.float32,
                    device=latent.device,
                )
                imag = torch.zeros_like(real)
                real[:, self.lower_rows, self.lower_columns] = real_value
                imag[:, self.lower_rows, self.lower_columns] = imag_value
                diagonal = torch.arange(self.dimension, device=latent.device)
                real[:, diagonal, diagonal] = (
                    functional.softplus(real[:, diagonal, diagonal]) + 1e-5
                )
                imag[:, diagonal, diagonal] = 0.0
                lower = torch.complex(real, imag)
                density = lower @ lower.mH
                trace = torch.diagonal(density, dim1=-2, dim2=-1).real.sum(dim=-1)
                density = density / trace.clamp_min(1e-8)[:, None, None]
                if self.mass_enabled:
                    mass = torch.sigmoid(
                        self.mass_projection(latent.float()).squeeze(-1)
                    ).clamp(1e-6, 1.0 - 1e-6)
                else:
                    mass = torch.ones(
                        batch_size,
                        dtype=torch.float32,
                        device=latent.device,
                    )
                return {
                    "conditional_density": density,
                    "w_subspace_mass": mass,
                    "leakage": 1.0 - mass,
                }


else:  # pragma: no cover

    class ConditionalWCoreDensityHead:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("V8 conditional density head requires torch")


__all__ = [
    "ConditionalWCoreDensityHead",
    "HISTORY_IDENTIFICATION_RESPONSE_NAMES",
    "analytic_history_identification",
    "bounded_control_residual",
    "resolve_exposure",
]
