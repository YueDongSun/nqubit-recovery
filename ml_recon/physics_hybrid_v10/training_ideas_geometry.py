"""Synthetic geometry and counterfactual labels for the declared projector law.

This teacher operates only on a known, generated conditional single-excitation
density matrix. It does not derive causal labels from real-static counts.
The probability owner is training_ideas_data.readout_probability_from_rho.
The normalized diagonal action is a declared preparation/filter model; it is
not a claim that an arbitrary deterministic quantum channel implements it.
"""

from __future__ import annotations

from typing import Callable

import torch

from .training_geometry import joint_profiled_count_geometry, profile_observation_residual


def apply_diagonal_control(
    density: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Apply relative log-amplitude/phase actions and normalize once.

    density[B,N,N], actions[B,K,N,2] -> conditional density[B,K,N,N].
    """
    if density.ndim != 3 or actions.ndim != 4 or actions.shape[-1] != 2:
        raise ValueError("Expected density[B,N,N] and actions[B,K,N,2]")
    if actions.shape[0] != density.shape[0] or actions.shape[-2] != density.shape[-1]:
        raise ValueError("Control actions and density modes do not align")
    diagonal = torch.exp(torch.complex(actions[..., 0], actions[..., 1]))
    changed = diagonal.unsqueeze(-1) * density.unsqueeze(1) * diagonal.conj().unsqueeze(-2)
    mass = changed.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    if not bool(torch.isfinite(changed).all()) or bool((mass <= 0).any()):
        raise ValueError("Finite diagonal control must retain positive conditional mass")
    return changed / mass[..., None, None]


def public_control_candidates(
    batch_size: int,
    n_mode: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Seven fixed, truth-independent candidates including exact NO_OP."""
    if n_mode < 3:
        raise ValueError("The registered candidate schedule requires at least three modes")
    first = torch.linspace(-1.0, 1.0, n_mode, device=device, dtype=dtype)
    second = first.square()
    second = second - second.mean()
    second = second / second.abs().amax()
    actions = torch.zeros((7, n_mode, 2), device=device, dtype=dtype)
    actions[1, :, 0], actions[2, :, 0] = 0.15 * first, -0.15 * first
    actions[3, :, 1], actions[4, :, 1] = 0.35 * first, -0.35 * first
    actions[5, :, 1], actions[6, :, 1] = 0.35 * second, -0.35 * second
    return actions.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()


def _apparatus(
    value: float | torch.Tensor,
    shape: tuple[int, int],
    reference: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 1 and tensor.shape[0] == shape[0]:
        tensor = tensor.unsqueeze(-1)
    try:
        result = torch.broadcast_to(tensor, shape)
    except RuntimeError as exc:
        raise ValueError("Apparatus parameters must broadcast to [B,S]") from exc
    if not bool(torch.isfinite(result).all()) or bool(((result <= 0) | (result >= 1)).any()):
        raise ValueError("This regular teacher requires apparatus probabilities in (0,1)")
    return result


def projector_mode_jacobian(
    density: torch.Tensor,
    analyzers: torch.Tensor,
    *,
    herald_probability: float | torch.Tensor = 0.35,
    read_efficiency: float | torch.Tensor = 0.7,
    background_probability: float | torch.Tensor = 0.015,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic same-law derivatives in interleaved physical mode coordinates.

    Returns control J[B,S,4,2N] and shared-nuisance J_nu[B,S,4,3].
    Nuisance coordinates are common logit-background, logit-efficiency and
    logit-herald shifts, in that order. Native category order is 00,10,01,11.
    """
    if density.ndim != 3 or analyzers.ndim != 3:
        raise ValueError("Expected conditional density[B,N,N] and analyzers[B,S,N]")
    if density.shape[0] != analyzers.shape[0] or density.shape[-1] != analyzers.shape[-1]:
        raise ValueError("Density and analyzer dimensions do not align")
    shape = tuple(analyzers.shape[:2])
    reference = density.real
    h = _apparatus(herald_probability, shape, reference)
    eta = _apparatus(read_efficiency, shape, reference)
    bg = _apparatus(background_probability, shape, reference)
    rho_v = torch.einsum("bij,bsj->bsi", density, analyzers)
    term = analyzers.conj() * rho_v
    overlap = term.sum(-1).real
    population = density.diagonal(dim1=-2, dim2=-1).real
    d_amp = 2.0 * term.real - 2.0 * population[:, None, :] * overlap[..., None]
    d_phase = -2.0 * term.imag
    d_overlap = torch.stack((d_amp, d_phase), dim=-1).flatten(-2)
    d_ret = ((1.0 - bg) * eta).unsqueeze(-1) * d_overlap
    jacobian = d_ret.new_zeros((*shape, 4, 2 * density.shape[-1]))
    jacobian[..., 1, :] = -h.unsqueeze(-1) * d_ret
    jacobian[..., 3, :] = h.unsqueeze(-1) * d_ret
    q = bg + (1.0 - bg) * eta * overlap
    zeros = torch.zeros_like(q)
    dq_bg, dq_eta = 1.0 - eta * overlap, (1.0 - bg) * overlap
    d_bg = torch.stack((-(1.0 - h), -h * dq_bg, 1.0 - h, h * dq_bg), -1)
    d_eta = torch.stack((zeros, -h * dq_eta, zeros, h * dq_eta), -1)
    d_h = torch.stack((-(1.0 - bg), 1.0 - q, -bg, q), -1)
    nuisance = torch.stack(
        (
            d_bg * (bg * (1.0 - bg)).unsqueeze(-1),
            d_eta * (eta * (1.0 - eta)).unsqueeze(-1),
            d_h * (h * (1.0 - h)).unsqueeze(-1),
        ),
        -1,
    )
    return jacobian, nuisance


@torch.no_grad()
def build_projector_geometry_targets(
    rho_true: torch.Tensor,
    analyzers: torch.Tensor,
    shots: torch.Tensor,
    prepared: torch.Tensor,
    *,
    candidate_actions: torch.Tensor | None = None,
    herald_probability: float | torch.Tensor = 0.35,
    read_efficiency: float | torch.Tensor = 0.7,
    background_probability: float | torch.Tensor = 0.015,
    profile_nuisance: bool = False,
    visible_mask: torch.Tensor | None = None,
    probability_forward: Callable[..., torch.Tensor] | None = None,
    derivative_forward: Callable | None = None,
    effective_calibration: torch.Tensor | None = None,
    report_fixed_geometry: bool = False,
) -> dict[str, torch.Tensor]:
    """Build detached synthetic labels plus a public, truth-independent action set.

    Utility is conditional target-overlap gain. Observation-Hellinger gain is
    returned separately and never relabelled as fidelity. Phase-edge weights
    are the fixed-nuisance isolated-pair Fisher even when CSM is profiled.
    """
    if rho_true.ndim != 3 or rho_true.shape[-1] != rho_true.shape[-2]:
        raise ValueError("Known synthetic rho_true must have shape [B,N,N]")
    batch_size, n_mode = rho_true.shape[:2]
    real_dtype = rho_true.real.dtype
    density = rho_true.detach().to(torch.complex128)
    if analyzers.ndim == 2:
        analyzers = analyzers.unsqueeze(0).expand(batch_size, -1, -1)
    analyzer = analyzers.detach().to(device=density.device, dtype=density.dtype)
    if analyzer.shape[0] != batch_size or analyzer.shape[-1] != n_mode:
        raise ValueError("Analyzers must align with each generated state")
    if not torch.allclose(
        analyzer.abs().square().sum(-1),
        torch.ones_like(analyzer[..., 0].real),
        atol=1e-7,
        rtol=1e-6,
    ):
        raise ValueError("Analyzer vectors must be normalized")
    # Remove only accepted float32-to-float64 normalization roundoff before
    # calling the strict double-precision owner of the same physical law.
    analyzer = analyzer / torch.linalg.vector_norm(analyzer, dim=-1, keepdim=True)
    if not torch.allclose(density, density.mH, atol=1e-7, rtol=1e-6):
        raise ValueError("Synthetic conditional states must be Hermitian")
    if not torch.allclose(
        density.diagonal(dim1=-2, dim2=-1).real.sum(-1),
        torch.ones(batch_size, device=density.device, dtype=torch.float64),
        atol=1e-7,
        rtol=1e-6,
    ) or bool((torch.linalg.eigvalsh(density).amin(-1) < -1e-7).any()):
        raise ValueError("Synthetic conditional states must be PSD and trace one")
    density = (density + density.mH) * 0.5
    density = density / density.diagonal(dim1=-2, dim2=-1).real.sum(-1)[:, None, None]
    target = prepared.detach().to(device=density.device, dtype=density.dtype)
    if target.shape != (batch_size, n_mode):
        raise ValueError("Prepared public target must be complex [B,N]")
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("Prepared targets must be finite and nonzero")
    if shots.ndim == 1:
        shots = shots.unsqueeze(0).expand(batch_size, -1)
    exposure = shots.detach().to(device=density.device, dtype=torch.float64)
    if exposure.shape != analyzer.shape[:2] or bool((exposure < 0).any()):
        raise ValueError("Nonnegative exposure must align with each operation")
    if visible_mask is not None:
        if visible_mask.dtype != torch.bool or visible_mask.shape != exposure.shape:
            raise ValueError("Visible observation mask must be Boolean [B,S]")
        exposure = exposure * visible_mask.to(device=density.device)
    if not bool((exposure.sum(-1) > 0).all()):
        raise ValueError("Each teacher row requires a positive total measurement budget")
    from .training_ideas_data import readout_probability_from_rho

    if effective_calibration is not None:
        if probability_forward is not None or derivative_forward is not None:
            raise ValueError("effective calibration selects its own probability and derivative")
        from .effective_projector_law import effective_probabilities, effective_jacobians
        probability_forward, derivative_forward = effective_probabilities, effective_jacobians
        apparatus = {"calibration": effective_calibration}
    else:
        if probability_forward is not None and probability_forward is not readout_probability_from_rho and derivative_forward is None:
            raise ValueError("an alternate probability requires its paired same-law derivative")
        shape = tuple(analyzer.shape[:2])
        apparatus = dict(herald_probability=_apparatus(herald_probability, shape, density.real),
                         read_efficiency=_apparatus(read_efficiency, shape, density.real),
                         background_probability=_apparatus(background_probability, shape, density.real))
        probability_forward = probability_forward or readout_probability_from_rho
        derivative_forward = derivative_forward or projector_mode_jacobian

    def probabilities(state: torch.Tensor) -> torch.Tensor:
        return probability_forward(state, analyzer, **apparatus)

    probability = probabilities(density)
    jacobian, nuisance = derivative_forward(density, analyzer, **apparatus)
    geometry = joint_profiled_count_geometry(
        probability,
        jacobian,
        exposure,
        nuisance if profile_nuisance else None,
    )
    target_density = target.unsqueeze(-1) * target.conj().unsqueeze(-2)
    desired_probability = probabilities(target_density)
    residual = profile_observation_residual(desired_probability - probability, geometry)
    left, singular, _ = torch.linalg.svd(geometry["profiled_jacobian"], full_matrices=False)
    support = singular > torch.maximum(
        singular.amax(-1, keepdim=True) * 1e-6,
        singular.new_full((batch_size, 1), 1e-10),
    )
    basis = left * support.unsqueeze(-2)
    parallel = (basis @ (basis.transpose(-1, -2) @ residual.unsqueeze(-1))).squeeze(-1)
    energy = residual.square().sum(-1)
    rsm = parallel.square().sum(-1) / energy.clamp_min(1e-20)

    edges = torch.stack(
        (
            torch.zeros(n_mode - 1, device=density.device, dtype=torch.long),
            torch.arange(1, n_mode, device=density.device),
        ),
        -1,
    )
    edge_coherence = density[:, edges[:, 0], edges[:, 1]]
    edge_phase = torch.stack((edge_coherence.real, edge_coherence.imag), -1)
    edge_phase = edge_phase / edge_coherence.abs().clamp_min(1e-12).unsqueeze(-1)
    occupied = analyzer.abs() > 1e-10
    phase_weights = []
    for left_mode, right_mode in edges.tolist():
        isolated = occupied.sum(-1).eq(2) & occupied[..., left_mode] & occupied[..., right_mode]
        phase_derivative = jacobian[..., 2 * left_mode + 1]
        edge_information = (phase_derivative.square() * exposure.unsqueeze(-1) / probability).sum(
            -1
        )
        phase_weights.append((edge_information * isolated).sum(-1))
    edge_weight = torch.stack(phase_weights, -1)
    edge_mask = (edge_coherence.abs() > 1e-10) & (edge_weight > 0)

    if candidate_actions is None:
        actions = public_control_candidates(
            batch_size,
            n_mode,
            device=density.device,
            dtype=torch.float64,
        )
    else:
        actions = candidate_actions.detach().to(device=density.device, dtype=torch.float64)
        if actions.ndim == 3:
            actions = actions.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if actions.shape[0] != batch_size or actions.shape[-2:] != (n_mode, 2):
        raise ValueError("Public candidate actions must have shape [B,K,N,2]")
    no_op = actions.abs().sum((-1, -2)) == 0
    if not bool(no_op.any(-1).all()):
        raise ValueError("Each registered candidate library must include an exact NO_OP")
    after = apply_diagonal_control(density, actions)
    before_fidelity = torch.einsum("bi,bij,bj->b", target.conj(), density, target).real
    after_fidelity = torch.einsum("bi,bkij,bj->bk", target.conj(), after, target).real
    gain = after_fidelity - before_fidelity.unsqueeze(-1)
    gain = torch.where(no_op, torch.zeros_like(gain), gain)
    expected = torch.stack([probabilities(after[:, index]) for index in range(after.shape[1])], 1)
    baseline_h2 = (
        (torch.sqrt(probability) - torch.sqrt(desired_probability)).square().sum(-1) * exposure
    ).sum(-1) / (2.0 * exposure.sum(-1))
    after_h2 = (
        (torch.sqrt(expected) - torch.sqrt(desired_probability[:, None])).square().sum(-1)
        * exposure[:, None]
    ).sum(-1) / (2.0 * exposure.sum(-1, keepdim=True))
    observation_gain = baseline_h2[:, None] - after_h2
    observation_gain = torch.where(no_op, torch.zeros_like(observation_gain), observation_gain)

    def value(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(real_dtype) if tensor.is_floating_point() else tensor.detach()

    labels = {
        "target_csm_gram": geometry["gram"],
        "target_csm_mask": torch.ones(batch_size, dtype=torch.bool, device=density.device),
        "target_csm_profiled": torch.full(
            (batch_size,),
            profile_nuisance,
            dtype=torch.bool,
            device=density.device,
        ),
        "phase_edges": edges.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        "target_phase_edge_cos_sin": edge_phase,
        "target_phase_edge_weight": edge_weight,
        "target_phase_edge_mask": edge_mask,
        "target_rsm_fraction": rsm.clamp(0, 1),
        "target_rsm_mask": energy > 1e-16,
        "target_fd_utility": gain,
        "target_fd_utility_mask": torch.ones_like(gain, dtype=torch.bool),
        "target_fd_harm": (gain < -1e-12).to(gain.dtype),
        "target_fd_harm_mask": torch.ones_like(gain, dtype=torch.bool),
        "target_fd_observation_gain": observation_gain,
        "target_fd_next_probabilities": expected,
        "public_candidate_actions": actions,
        "public_candidate_valid_mask": torch.ones_like(gain, dtype=torch.bool),
        "public_noop_candidate_index": no_op.to(torch.long).argmax(-1),
        "target_geometry_is_synthetic": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=density.device,
        ),
        "target_geometry_observed_mask": exposure > 0,
    }
    if report_fixed_geometry:
        labels["target_csm_fixed_trace"] = geometry["raw_gram"].diagonal(dim1=-2, dim2=-1).sum(-1)
    return {key: value(tensor) for key, tensor in labels.items()}


__all__ = [
    "apply_diagonal_control",
    "public_control_candidates",
    "projector_mode_jacobian",
    "build_projector_geometry_targets",
]
