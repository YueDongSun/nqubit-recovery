"""One registered R1 belief chart for training, calibration, and feedback.

Covariance residuals use detached point beliefs. R1 is trained by its state
objective; covariance fitting must not backpropagate through a degenerate
eigenspace. A principal phase difference is formed in mode space before Helmert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch

from .contracts import sha256_of
from .control_coordinate_layout import interleave_control_coordinates

BELIEF_VALIDITY_CONTRACT = "V10R1ValidatedTopEigenvectorPointApproximationV4"
BELIEF_MEAN_SOURCE = "validated_r1_top_eigenvector_point_belief_v3"
RESIDUAL_CONTRACT = "prediction_reference_principal_mode_difference_then_helmert_v1"
NUMERICAL_TOLERANCE_MULTIPLIER = 8.0


@dataclass(frozen=True)
class Belief_validity_config:
    minimum_trace: float = 0.0
    minimum_top_eigengap: float = 0.0
    minimum_phase_support: float = 0.0
    psd_tolerance: float = 0.0
    minimum_sector_probability: float | None = None
    minimum_confidence: float | None = None

    def validate(self) -> None:
        for name in (
            "minimum_trace",
            "minimum_top_eigengap",
            "minimum_phase_support",
            "psd_tolerance",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    "belief validity thresholds must be finite nonnegative numbers"
                )
        for value in (self.minimum_sector_probability, self.minimum_confidence):
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "optional belief probabilities must be finite values in [0,1]"
                )

    def sha256(self) -> str:
        self.validate()
        return sha256_of(
            {
                "schema": BELIEF_VALIDITY_CONTRACT,
                "config": asdict(self),
                "residual_contract": RESIDUAL_CONTRACT,
                "numerical_tolerance_multiplier": NUMERICAL_TOLERANCE_MULTIPLIER,
                "reference": "largest_predicted_amplitude_first_mode_on_tie",
                "mode_support": "every_active_mode_full_dimension_only",
                "sector_probability_index": 1,
                "aggregate_sector_evidence": "explicit_k1_probability_without_tail_split_v1",
            }
        )


def _validate_r1_shape(real: torch.Tensor, imag: torch.Tensor) -> None:
    if (
        not isinstance(real, torch.Tensor)
        or not isinstance(imag, torch.Tensor)
        or real.ndim != 3
        or real.shape != imag.shape
        or real.shape[-1] < 2
        or real.shape[-2] != real.shape[-1]
        or real.dtype not in (torch.float32, torch.float64)
        or imag.dtype != real.dtype
        or imag.device != real.device
    ):
        raise ValueError(
            "R1 requires aligned float32/float64 [batch,N,N] real and imaginary matrices"
        )


def _probability_rows(
    value: torch.Tensor | None, rows: int, device, *, sector: bool
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.device != device:
        raise ValueError(
            "confidence and sector evidence must be tensors on the R1 device"
        )
    if sector:
        if tuple(value.shape) != (rows, 4):
            raise ValueError("sector evidence must use [batch,p0/p1/p2/pge3]")
        return value.detach()
    if tuple(value.shape) not in {(rows,), (rows, 1)}:
        raise ValueError("confidence must have one value per R1 row")
    return value.detach().reshape(rows)


def validated_belief_from_r1(
    r1_re: torch.Tensor,
    r1_im: torch.Tensor,
    *,
    config: Belief_validity_config,
    sector_posterior: torch.Tensor | None = None,
    sector_k1_probability: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str, ...], ...]]:
    """Return a detached valid direction or an exactly zero abstention direction."""
    if not isinstance(config, Belief_validity_config):
        raise ValueError("a registered belief validity configuration is required")
    config.validate()
    _validate_r1_shape(r1_re, r1_im)
    rows, modes = r1_re.shape[:2]
    if sector_posterior is not None and sector_k1_probability is not None:
        raise ValueError("provide full-sector or aggregate K1 evidence, never both")
    sector = _probability_rows(sector_posterior, rows, r1_re.device, sector=True)
    k1_probability = _probability_rows(sector_k1_probability, rows, r1_re.device, sector=False)
    if k1_probability is not None and k1_probability.dtype not in (torch.float32, torch.float64):
        raise ValueError("aggregate K1 probability requires floating-point evidence")
    confidence_values = _probability_rows(confidence, rows, r1_re.device, sector=False)
    with torch.no_grad():
        real, imag = r1_re.detach().double(), r1_im.detach().double()
        raw = torch.complex(real, imag)
        finite = torch.isfinite(raw).all((-1, -2))
        safe = torch.where(finite[:, None, None], raw, torch.zeros_like(raw))
        scale = safe.abs().amax((-1, -2)).clamp_min(1)
        tolerance = (
            torch.finfo(r1_re.dtype).eps
            * NUMERICAL_TOLERANCE_MULTIPLIER
            * modes
            * scale
        )
        hermitian = (safe - safe.mH).abs().amax((-1, -2)) <= tolerance
        eligible = finite & hermitian & torch.isfinite(scale)
        eigenvalues = real.new_zeros((rows, modes))
        vectors = raw.new_zeros((rows, modes))
        if bool(eligible.any()):
            values, basis = torch.linalg.eigh(
                (safe[eligible] + safe[eligible].mH) * 0.5
            )
            eigenvalues[eligible] = values
            vectors[eligible] = basis[..., -1]
        trace = eigenvalues.sum(-1)
        reference = vectors.abs().argmax(-1, keepdim=True)
        vectors = vectors * torch.exp(-1j * vectors.gather(1, reference).angle())
        support_floor = max(
            config.minimum_phase_support,
            torch.finfo(r1_re.dtype).eps * NUMERICAL_TOLERANCE_MULTIPLIER,
        )
        checks = {
            "NONFINITE_R1": finite,
            "NONHERMITIAN_R1": hermitian,
            "NOT_PSD": eigenvalues[:, 0]
            >= -torch.maximum(
                tolerance, torch.full_like(tolerance, config.psd_tolerance)
            ),
            "TRACE_UNSUPPORTED": (
                trace
                > torch.maximum(
                    tolerance, torch.full_like(tolerance, config.minimum_trace)
                )
            )
            & (trace <= 1 + tolerance),
            "TOP_EIGENSPACE_DEGENERATE": eigenvalues[:, -1] - eigenvalues[:, -2]
            > torch.maximum(
                tolerance, torch.full_like(tolerance, config.minimum_top_eigengap)
            ),
            "PHASE_UNSUPPORTED": (vectors.abs() > support_floor).all(-1),
        }
        if config.minimum_sector_probability is not None:
            if k1_probability is not None:
                accepted_sector = (
                    torch.isfinite(k1_probability)
                    & (k1_probability >= config.minimum_sector_probability)
                    & (k1_probability <= 1)
                    & ((k1_probability.double() - trace).abs() <= tolerance)
                )
            elif sector is None:
                accepted_sector = torch.zeros_like(finite)
            else:
                accepted_sector = (
                    torch.isfinite(sector).all(-1)
                    & (sector >= 0).all(-1)
                    & (sector <= 1).all(-1)
                    & ((sector.sum(-1) - 1).abs() <= tolerance)
                    & (sector[:, 1] >= config.minimum_sector_probability)
                    & ((sector[:, 1].double() - trace).abs() <= tolerance)
                )
            checks["SECTOR_UNSUPPORTED"] = accepted_sector
        if config.minimum_confidence is not None:
            checks["CONFIDENCE_UNSUPPORTED"] = (
                torch.zeros_like(finite)
                if confidence_values is None
                else torch.isfinite(confidence_values)
                & (confidence_values >= config.minimum_confidence)
                & (confidence_values <= 1)
            )
        valid = torch.stack(tuple(checks.values())).all(0) & eligible
        reasons = tuple(
            tuple(name for name, values in checks.items() if not bool(values[row]))
            for row in range(rows)
        )
        belief = torch.where(valid[:, None], vectors, torch.zeros_like(vectors))
        return (
            belief.to(
                torch.complex64 if r1_re.dtype == torch.float32 else torch.complex128
            ),
            valid,
            reasons,
        )


def helmert_tensor(n_modes: int, reference: torch.Tensor) -> torch.Tensor:
    matrix = reference.new_zeros((n_modes - 1, n_modes))
    for index in range(1, n_modes):
        matrix[index - 1, :index] = 1 / math.sqrt(index * (index + 1))
        matrix[index - 1, index] = -index / math.sqrt(index * (index + 1))
    return matrix


@dataclass(frozen=True)
class Belief_residual_details:
    predicted_belief: torch.Tensor
    target_belief: torch.Tensor
    reference_indices: torch.Tensor
    mode_residual: torch.Tensor
    control_residual: torch.Tensor
    valid_mask: torch.Tensor


def belief_residual_details(
    predicted_re: torch.Tensor,
    predicted_im: torch.Tensor,
    target_re: torch.Tensor,
    target_im: torch.Tensor,
    mode_mask: torch.Tensor,
    *,
    config: Belief_validity_config | None = None,
    sector_posterior: torch.Tensor | None = None,
    sector_k1_probability: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> Belief_residual_details:
    """Predict-minus-target errors in one gauge, with finite graph-connected zeros."""
    _validate_r1_shape(predicted_re, predicted_im)
    _validate_r1_shape(target_re, target_im)
    if (
        target_re.shape != predicted_re.shape
        or target_re.device != predicted_re.device
        or target_re.dtype != predicted_re.dtype
        or not isinstance(mode_mask, torch.Tensor)
        or mode_mask.dtype is not torch.bool
        or tuple(mode_mask.shape) != tuple(predicted_re.shape[:2])
        or mode_mask.device != predicted_re.device
    ):
        raise ValueError("prediction, target, and strict mode masks must align")
    config = config or Belief_validity_config()
    predicted, valid, _ = validated_belief_from_r1(
        predicted_re,
        predicted_im,
        config=config,
        sector_posterior=sector_posterior,
        sector_k1_probability=sector_k1_probability,
        confidence=confidence,
    )
    # Prediction confidence is not a required observable of known synthetic truth.
    truth_config = replace(
        config, minimum_confidence=None, minimum_sector_probability=None
    )
    target, target_valid, _ = validated_belief_from_r1(
        target_re, target_im, config=truth_config
    )
    valid = valid & target_valid & mode_mask.all(-1)
    reference = predicted.abs().argmax(-1)
    target = target * torch.exp(-1j * target.gather(1, reference[:, None]).angle())
    floor = torch.finfo(predicted_re.dtype).tiny
    amplitude = (
        predicted.abs().clamp_min(floor).log() - target.abs().clamp_min(floor).log()
    )
    reference_product = (predicted * target.conj()).gather(1, reference[:, None])
    phase = torch.angle(predicted * target.conj() * reference_product.conj())
    mode_residual = torch.stack(
        (
            amplitude - amplitude.mean(-1, keepdim=True),
            phase - phase.mean(-1, keepdim=True),
        ),
        -1,
    )
    mode_residual = torch.where(
        valid[:, None, None], mode_residual, torch.zeros_like(mode_residual)
    ).detach()
    # The zero anchor has finite zero derivative even for nonfinite/degenerate
    # source rows; no eigensolver graph reaches the covariance loss.
    anchor = sum(
        (torch.where(torch.isfinite(value), value, torch.zeros_like(value)) * 0).sum(
            (-1, -2)
        )
        for value in (predicted_re, predicted_im)
    )
    mode_residual = mode_residual + anchor[:, None, None]
    helmert = helmert_tensor(predicted.shape[-1], predicted.real)
    control = interleave_control_coordinates(
        mode_residual[..., 0] @ helmert.T, mode_residual[..., 1] @ helmert.T
    )
    return Belief_residual_details(
        predicted, target, reference, mode_residual, control, valid
    )


def belief_control_residual(
    predicted_re: torch.Tensor,
    predicted_im: torch.Tensor,
    target_re: torch.Tensor,
    target_im: torch.Tensor,
    mode_mask: torch.Tensor,
    *,
    config: Belief_validity_config | None = None,
    sector_posterior: torch.Tensor | None = None,
    sector_k1_probability: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    details = belief_residual_details(
        predicted_re,
        predicted_im,
        target_re,
        target_im,
        mode_mask,
        config=config,
        sector_posterior=sector_posterior,
        sector_k1_probability=sector_k1_probability,
        confidence=confidence,
    )
    return details.control_residual, details.valid_mask


def mode_marginal_variances(
    control_covariance: torch.Tensor, n_modes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift the logical covariance to the same zero-sum mode chart used above."""
    if (
        control_covariance.ndim != 3
        or control_covariance.shape[-2:] != (2 * (n_modes - 1),) * 2
    ):
        raise ValueError("control covariance must have the full interleaved dimension")
    helmert = helmert_tensor(n_modes, control_covariance)
    amplitude = torch.einsum(
        "in,bij,jn->bn", helmert, control_covariance[:, 0::2, 0::2], helmert
    )
    phase = torch.einsum(
        "in,bij,jn->bn", helmert, control_covariance[:, 1::2, 1::2], helmert
    )
    return amplitude, phase
