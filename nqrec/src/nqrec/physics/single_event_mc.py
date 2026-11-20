"""Differentiable effective single-event MC, aligned on 20 November 2026.

One WRITE is followed by one READ. Signal rates include background. The public
observation has native order 00,10,01,11; it never exposes photon multiplicity.
K>=2 responses are explicit ESM hypotheses, independent of the K1 state.
An effective event kernel marginalizes detector/time details; it is not a new
implementation of the FPGA sticky-window instrument.
"""

from __future__ import annotations

import math

import numpy as np
import torch

EVENT_LAW = "effective_single_event_v1"
CELL_ORDER = ("00", "10", "01", "11")


def _probability(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must be finite and in [0,1]")


def _normalized(value: torch.Tensor, dim: int | tuple[int, ...], name: str) -> None:
    _probability(value, name)
    total = value.sum(dim=dim)
    tolerance = 32 * torch.finfo(value.dtype).eps
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0):
        raise ValueError(f"{name} must be normalized; no mass is clipped or renormalized")


def calibrate_source_weights(
    total_signal_rate: torch.Tensor,
    signal_efficiency: torch.Tensor,
    signal_background: torch.Tensor,
    *,
    double_probability: torch.Tensor,
    double_signal_event_probability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Solve p0,p1,p2 from a total signal rate and an explicit source P(K=2).

    This is a declared K<=2 sensitivity family, not a thermal-source claim. The
    supplied K2 signal probability already includes its background/acceptance law.
    K0/K1 use an independent effective-background OR model. The source p2 is NOT
    the herald-conditioned double fraction. Invalid operating points are rejected.
    """
    values = {
        "total_signal_rate": total_signal_rate,
        "signal_efficiency": signal_efficiency,
        "signal_background": signal_background,
        "double_probability": double_probability,
    }
    for name, value in values.items():
        _probability(value, name)
    s, eta, background, p2 = torch.broadcast_tensors(*values.values())
    if torch.any(eta <= 0) or torch.any(background >= 1):
        raise ValueError("signal efficiency must be positive and background below one")
    if double_signal_event_probability is None:
        if torch.any(p2 > 0):
            raise ValueError("K2 requires an explicit signal-event probability")
        h2 = torch.zeros_like(s)  # Zero-weight sector; no K2 physics is inferred.
    else:
        _probability(double_signal_event_probability, "K2 signal-event probability")
        s, eta, background, p2, h2 = torch.broadcast_tensors(
            s, eta, background, p2, double_signal_event_probability
        )
    h0 = background
    h1 = background + (1 - background) * eta
    p1 = (s - h0 - p2 * (h2 - h0)) / (h1 - h0)
    source = torch.stack((1 - p1 - p2, p1, p2), dim=-1)
    _normalized(source, -1, "source probabilities")
    signal = torch.stack((h0, h1, h2), dim=-1)
    joint = source.unsqueeze(-1) * torch.stack((1 - signal, signal), dim=-1)
    return {"source_probability": source, "signal_probability": signal, "joint": joint}


def four_cell_probabilities(
    joint_source_signal: torch.Tensor,
    idler_event_probability: torch.Tensor,
) -> torch.Tensor:
    """Marginalize K from w[...,K,2] and r[...,K,2], returning [...,4].

    The final axis of w/r is S=0,1. The idler responses may depend on S because a
    coherent herald can change the conditional spatial state. K1 and K2 must not
    be forced to share a collective mode. Leading batch/operation axes broadcast.
    """
    w, r = joint_source_signal, idler_event_probability
    if w.ndim < 2 or r.ndim < 2 or w.shape[-1] != 2 or r.shape[-1] != 2:
        raise ValueError("w and r require trailing [sector, signal=0/1] axes")
    if w.shape[-2] < 1 or w.shape[-2] != r.shape[-2]:
        raise ValueError("w and r must describe the same nonempty excitation sectors")
    _normalized(w, (-2, -1), "joint source/signal probabilities")
    _probability(r, "idler effective-event probabilities")
    w, r = torch.broadcast_tensors(w, r)
    n00 = (w[..., 0] * (1 - r[..., 0])).sum(-1)
    n10 = (w[..., 1] * (1 - r[..., 1])).sum(-1)
    n01 = (w[..., 0] * r[..., 0]).sum(-1)
    n11 = (w[..., 1] * r[..., 1]).sum(-1)
    cells = torch.stack((n00, n10, n01, n11), dim=-1)
    _normalized(cells, -1, "native event probabilities")
    return cells


def k1_event_probability(
    density: torch.Tensor,
    read_transfer: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Effective-background OR with t^dagger rho t for a normalized K1 state.

    The analyzer ket t includes amplitude, the authoritative read phase convention,
    and loss once. Its squared norm is at most one. Leading dimensions broadcast;
    for B states and O operations use rho[B,1,N,N] and t[B,O,N].
    """
    _validate_transfer(read_transfer)
    _probability(background, "idler background")
    if not (density.is_floating_point() or density.is_complex()):
        raise TypeError("K1 density must be floating point or complex")
    common_dtype = torch.promote_types(density.dtype, read_transfer.dtype)
    density = density.to(dtype=common_dtype)
    read_transfer = read_transfer.to(dtype=common_dtype)
    n = read_transfer.shape[-1]
    if density.ndim < 2 or density.shape[-2:] != (n, n):
        raise ValueError("K1 density and transfer dimensions must agree")
    if not torch.isfinite(density).all():
        raise ValueError("K1 density must be finite")
    tolerance = 64 * torch.finfo(density.real.dtype).eps
    if not torch.allclose(density, density.mH, atol=tolerance, rtol=0):
        raise ValueError("K1 density must be Hermitian")
    trace = torch.diagonal(density, dim1=-2, dim2=-1).sum(-1)
    if not torch.allclose(trace, torch.ones_like(trace), atol=tolerance, rtol=0):
        raise ValueError("K1 density must have unit trace")
    if torch.any(torch.linalg.eigvalsh(density.detach()) < -tolerance):
        raise ValueError("K1 density must be positive semidefinite")
    q = torch.einsum("...i,...ij,...j->...", read_transfer.conj(), density, read_transfer).real
    response = background + (1 - background) * q
    _probability(response, "K1 effective-event response")
    return response


def _validate_transfer(read_transfer: torch.Tensor) -> None:
    if not isinstance(read_transfer, torch.Tensor) or read_transfer.ndim < 1:
        raise ValueError("read transfer must have a mode axis")
    if not (read_transfer.is_floating_point() or read_transfer.is_complex()):
        raise TypeError("read transfer must be floating point or complex")
    if read_transfer.shape[-1] < 2 or not torch.isfinite(read_transfer).all():
        raise ValueError("read transfer requires at least two finite modes")
    if torch.any(read_transfer.abs().square().sum(-1) > 1):
        raise ValueError(
            "read transfer squared norm must not exceed one; normalize the analyzer "
            "before applying its calibrated loss (this API does not clip efficiencies)"
        )


def diagonal_fock2_event_probability(
    read_transfer: torch.Tensor,
    local_weights: torch.Tensor,
    cross_weights: torch.Tensor,
    *,
    two_photon_single_event_acceptance: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Explicit ESM mixture of |2_i> and |1_i,1_j>, independent of K1.

    local[...,N] and strictly upper-triangular cross[...,N,N] jointly sum to one.
    For a cross pair, P(two photons in selected output)=2|t_i t_j|^2; the bosonic
    factor two is retained. kappa=P(accepted single event | two photons in output)
    is a supplied effective sensitivity parameter, not an inferred TTL histogram.
    No two-event output or automatic coherent |2_c> lifting is generated.
    """
    _validate_transfer(read_transfer)
    n = read_transfer.shape[-1]
    if local_weights.shape[-1:] != (n,) or cross_weights.shape[-2:] != (n, n):
        raise ValueError("local/cross weights must match the transfer mode count")
    _probability(local_weights, "local weights")
    _probability(cross_weights, "cross weights")
    _probability(two_photon_single_event_acceptance, "two-photon single-event acceptance")
    _probability(background, "idler background")
    if torch.any(torch.tril(cross_weights) != 0):
        raise ValueError("cross weights must use only the strict upper triangle")
    total = local_weights.sum(-1) + cross_weights.sum((-2, -1))
    tolerance = 32 * torch.finfo(total.dtype).eps
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0):
        raise ValueError("local and cross weights must jointly sum to one")
    a = read_transfer.abs().square()
    a_i, a_j = a.unsqueeze(-1), a.unsqueeze(-2)
    mean = 2 * (local_weights * a).sum(-1)
    mean = mean + (cross_weights * (a_i + a_j)).sum((-2, -1))
    two = (local_weights * a.square()).sum(-1)
    two = two + 2 * (cross_weights * a_i * a_j).sum((-2, -1))
    one = mean - 2 * two
    accepted = one + two_photon_single_event_acceptance * two
    response = background + (1 - background) * accepted
    _probability(response, "K2 effective-event response")
    return response


def sample_counts(
    probabilities: torch.Tensor,
    read_opportunities: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one fixed-total multinomial per operation; all READs stay in R."""
    if type(read_opportunities) is not int or read_opportunities < 0:
        raise ValueError("READ opportunities must be a nonnegative integer")
    if probabilities.ndim < 1 or probabilities.shape[-1] != 4:
        raise ValueError("probabilities must end in native four-cell order")
    _normalized(probabilities, -1, "event probabilities")
    values = probabilities.detach().cpu().numpy()
    draws = [rng.multinomial(read_opportunities, row) for row in values.reshape(-1, 4)]
    return np.asarray(draws, dtype=np.int64).reshape(values.shape)


def thermal_mean_from_total_signal(
    total_signal_rate: float, signal_efficiency: float, signal_background: float
) -> float:
    """Exact OLD single-thermal threshold reference, not a universal DLCZ prior."""
    if not all(math.isfinite(x) for x in (total_signal_rate, signal_efficiency, signal_background)):
        raise ValueError("reference parameters must be finite")
    if not 0 <= signal_background <= total_signal_rate < 1 or not 0 < signal_efficiency <= 1:
        raise ValueError("require 0 <= background <= signal < 1 and 0 < efficiency <= 1")
    return (total_signal_rate - signal_background) / (signal_efficiency * (1 - total_signal_rate))
