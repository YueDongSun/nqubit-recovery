"""Differentiable DLCZ single-clock positive control, not a real-window decoder.

One unconditioned source shot has P(K=k)=(1-chi)*chi**k, matching the
TrackA source kernel. Given K, photons in one selected collective optical
mode undergo independent losses into four downstream SPADs. The idler event
is exactly one SPAD present at this clock, NOT any-click and NOT the sticky
OR over the real 19-clock window. No historical model or data contract uses
this module implicitly. Parameters are hypotheses, not real calibrations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROBABILITY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class Detected_count_control_config:
    """Explicit scope prevents a simplified control being sold as real parity."""

    source_law: str = "tracka_geometric_chi"
    opportunity: str = "unconditioned_source_shot"
    idler_event: str = "single_clock_exact_singleton"
    accepted_clock_count: int = 1
    parameter_status: str = "HYPOTHESIS_UNBOUND"
    schema: str = "TrackBDetectedCountControlV1"

    def validate(self) -> None:
        if self != Detected_count_control_config() or type(self.accepted_clock_count) is not int:
            raise ValueError("only the explicit single-clock source-shot positive control is implemented")


def require_probability_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    """Validate a physical probability without narrowing exact 0/1 limits."""
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating tensor")
    result = value.to(torch.float64)
    if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
        raise ValueError(f"{name} must be finite in [0,1]")
    return result


def selected_mode_detection_probability(
    state: torch.Tensor, projector: torch.Tensor, throughput: torch.Tensor,
) -> torch.Tensor:
    """Contract an actual subnormalized PSD state, without double survival.

``state`` is not implicitly constructed from a prepared-command setpoint.
Its trace carries population survival; coherence loss may change off-diagonal
entries without changing that trace. Throughput covers subsequent conversion,
path and detection only. No factor is claimed individually identifiable.
"""
    if state.ndim < 2 or state.shape[-1] != state.shape[-2] or projector.shape[-2:] != state.shape[-2:]:
        raise ValueError("state/projector require matching square matrices")
    state = state.to(torch.complex128)
    projector = projector.to(torch.complex128)
    for name, matrix in (("state", state), ("projector", projector)):
        if not bool(torch.isfinite(matrix).all()) or not torch.allclose(
            matrix, matrix.mH, atol=PROBABILITY_TOLERANCE, rtol=0,
        ):
            raise ValueError(f"{name} must be finite Hermitian")
        if bool((torch.linalg.eigvalsh(matrix) < -PROBABILITY_TOLERANCE).any()):
            raise ValueError(f"{name} must be PSD")
    if bool((state.diagonal(dim1=-2, dim2=-1).sum(-1).real > 1 + PROBABILITY_TOLERANCE).any()):
        raise ValueError("state trace must not exceed one")
    if bool((torch.linalg.eigvalsh(projector) > 1 + PROBABILITY_TOLERANCE).any()):
        raise ValueError("projector must be bounded by identity")
    result = torch.einsum("...ij,...ji->...", projector, state).real
    return result.clamp(0, 1) * require_probability_tensor("throughput", throughput)


def single_clock_detected_probability(
    *, chi: torch.Tensor, signal_efficiency: torch.Tensor,
    idler_path_probability: torch.Tensor, signal_background: torch.Tensor,
    idler_background: torch.Tensor,
    config: Detected_count_control_config = Detected_count_control_config(),
) -> torch.Tensor:
    """Return [00,10,01,11] using exact geometric PGFs and SPAD presence.

``idler_path_probability[...,j]`` is the probability for ONE photon to
register in downstream SPAD j. Their sum cannot exceed one. Backgrounds
are independent per-clock presence probabilities after gating, not rates.
There is no tail cutoff. Arbitrary multi-particle states, inter-clock timing,
dead-time and detector memory are outside this positive control.
"""
    config.validate()
    source = require_probability_tensor("chi", chi)
    if bool((source >= 1).any()):
        raise ValueError("TrackA geometric chi must be less than one")
    signal = require_probability_tensor("signal_efficiency", signal_efficiency)
    paths = require_probability_tensor("idler_path_probability", idler_path_probability)
    dark_signal = require_probability_tensor("signal_background", signal_background)
    dark_idler = require_probability_tensor("idler_background", idler_background)
    if paths.ndim < 1 or paths.shape[-1] != 4 or dark_idler.shape != paths.shape:
        raise ValueError("four downstream SPAD paths and aligned backgrounds are required")
    path_sum = paths.sum(-1)
    if bool((path_sum > 1).any()):
        raise ValueError("idler paths are exclusive single-photon outcomes")
    source, signal, dark_signal, path_sum = torch.broadcast_tensors(source, signal, dark_signal, path_sum)
    paths = torch.broadcast_to(paths, (*source.shape, 4))
    dark_idler = torch.broadcast_to(dark_idler, paths.shape)
    vacuum = 1 - source
    missed = 1 - path_sum

    def singleton_with_signal_absence(signal_survival: torch.Tensor) -> torch.Tensor:
        # Sum disjoint events: detector j present and all other detectors absent.
        # G(z*(missed+q_j))-G(z*missed) is evaluated as a rational difference
        # to retain rare singletons without subtracting nearly equal PGFs.
        denominator = 1 - source * signal_survival * missed
        terms = []
        for index in range(4):
            other = [item for item in range(4) if item != index]
            no_other_dark = (1 - dark_idler[..., other]).prod(-1)
            added = source * signal_survival * paths[..., index]
            increment = vacuum * added / (denominator * (denominator - added))
            terms.append(no_other_dark * (increment + dark_idler[..., index] * vacuum / denominator))
        return torch.stack(terms, -1).sum(-1)

    singleton = singleton_with_signal_absence(torch.ones_like(signal))
    no_signal_singleton = (1 - dark_signal) * singleton_with_signal_absence(1 - signal)
    signal_probability = dark_signal + (1 - dark_signal) * source * signal / (vacuum + source * signal)
    coin = singleton - no_signal_singleton
    probability = torch.stack((
        (1 - signal_probability) - no_signal_singleton,
        signal_probability - coin, no_signal_singleton, coin,
    ), -1)
    if not bool(torch.isfinite(probability).all()) or bool((probability < -PROBABILITY_TOLERANCE).any()):
        raise ArithmeticError("single-clock PGF lost four-cell probability support")
    # Only floating-point roundoff at deterministic boundaries, never target clipping.
    return probability.clamp_min(0)
