"""Differentiable, mask-aware geometry and feedback-decision supervision.

CSM targets describe a declared observation law, independently of the desired
target. Phase-edge supervision acts on the predicted density matrix itself.
FD means feedback decision here; finite differences are a separate derivative
audit, never action/gain labels. Nothing in this module executes an action.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

GEOMETRY_LOSS_NAMES = (
    "csm_gram",
    "phase_laplacian",
    "rsm",
    "fd_utility",
    "fd_harm",
)


def _anchor(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for name in ("R1_re", "idea_csm_gram", "idea_fd_logits", "semantic_latent"):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value.sum() * 0.0
    raise ValueError("A differentiable model output is required")


def _teacher(
    batch: Mapping[str, object],
    target_key: str,
    mask_key: str,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    target, mask = batch.get(target_key), batch.get(mask_key)
    if target is None and mask is None:
        return None
    if mask is None:
        raise ValueError(f"{target_key} requires an explicit {mask_key}")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError(f"{mask_key} must be a Boolean tensor")
    mask = mask.to(device=reference.device)
    if target is None:
        if bool(mask.any()):
            raise ValueError(f"Valid {mask_key} entries have no {target_key}")
        return None
    if not isinstance(target, torch.Tensor):
        raise ValueError(f"{target_key} must be a tensor")
    return target.detach().to(device=reference.device, dtype=reference.dtype), mask


def _finite_selected(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Valid {name} entries must be finite")


def incidence_matrix(
    edges: torch.Tensor,
    n_mode: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Incidence rows follow the ordered relative phase phi_left - phi_right."""
    if edges.dtype not in (torch.int32, torch.int64) or edges.ndim not in (2, 3):
        raise ValueError("Edges must be integer [edge,2] or [batch,edge,2]")
    if edges.shape[-1] != 2 or n_mode < 2:
        raise ValueError("An edge has two endpoints and n_mode >= 2")
    if bool(((edges < 0) | (edges >= n_mode)).any()):
        raise ValueError("Edge endpoint is outside the mode range")
    if bool((edges[..., 0] == edges[..., 1]).any()):
        raise ValueError("Self edges carry no relative-phase information")
    return (F.one_hot(edges[..., 0].long(), n_mode) - F.one_hot(edges[..., 1].long(), n_mode)).to(
        dtype
    )


def weighted_phase_laplacian(
    edges: torch.Tensor,
    edge_weights: torch.Tensor,
    n_mode: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build B.T diag(w) B without inventing weights for unmeasured edges."""
    if edge_weights.ndim != 2 or not edge_weights.is_floating_point():
        raise ValueError("Edge weights must be floating [batch,edge]")
    mask = (
        torch.ones_like(edge_weights, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=edge_weights.device)
    )
    if mask.dtype != torch.bool or mask.shape != edge_weights.shape:
        raise ValueError("Edge validity must be Boolean and align with weights")
    selected = edge_weights[mask]
    _finite_selected(selected, "edge weights")
    if bool((selected < 0).any()):
        raise ValueError("Fisher edge weights cannot be negative")
    weight = torch.where(mask, edge_weights, torch.zeros_like(edge_weights))
    incidence = incidence_matrix(
        edges.to(edge_weights.device),
        n_mode,
        dtype=edge_weights.dtype,
    )
    if incidence.ndim == 2:
        incidence = incidence.unsqueeze(0).expand(edge_weights.shape[0], -1, -1)
    if incidence.shape[:2] != edge_weights.shape:
        raise ValueError("Incidence and edge weights do not align")
    return incidence.transpose(-1, -2) @ (weight.unsqueeze(-1) * incidence)


def joint_profiled_count_geometry(
    probabilities: torch.Tensor,
    jacobian: torch.Tensor,
    shots: torch.Tensor,
    nuisance_jacobian: torch.Tensor | None = None,
    *,
    relative_threshold: float = 1e-6,
    absolute_threshold: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Exact conditional-multinomial score geometry, joint before profiling.

    Inputs are p[B,S,C], J[B,S,C,D], and exposures[B,S]. On the simplex
    tangent, sqrt(shots/p)*J is an exact score factor and retains the
    covariance nullspace without jitter. Shared nuisance columns are stacked
    across settings before the low-rank projection. No dense projector is made.

    This function requires a regular, positive-probability conditional-iid law.
    The nuisance support is frozen for gradient propagation; gradients still
    flow through the control Jacobian and its score weighting. A caller using
    a shared-latent marginal law must supply its joint covariance elsewhere.
    """
    if probabilities.ndim != 3 or jacobian.ndim != 4:
        raise ValueError("Expected p[B,S,C] and J[B,S,C,D]")
    if jacobian.shape[:3] != probabilities.shape or shots.shape != probabilities.shape[:2]:
        raise ValueError("Probability, Jacobian and exposure shapes do not align")
    if probabilities.device != jacobian.device or shots.device != probabilities.device:
        raise ValueError("Geometry inputs must share a device")
    if relative_threshold < 0 or absolute_threshold < 0:
        raise ValueError("Support thresholds must be nonnegative")
    _finite_selected(probabilities, "probabilities")
    _finite_selected(jacobian, "Jacobian")
    _finite_selected(shots, "shots")
    if bool((probabilities <= 0).any()) or bool((shots < 0).any()):
        raise ValueError("Positive probability support and nonnegative exposures required")
    if not torch.allclose(
        probabilities.sum(-1),
        torch.ones_like(probabilities.sum(-1)),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Categorical probabilities must sum to one per setting")
    if not torch.allclose(
        jacobian.sum(-2),
        torch.zeros_like(jacobian.sum(-2)),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Probability derivatives must preserve the simplex")
    # Keep zero-exposure rows differentiable with respect to probabilities:
    # sqrt(shots / p) has a singular sqrt derivative at shots=0 even though
    # the physical derivative is zero. This algebraically identical factor
    # has the correct finite zero-row gradient.
    scale = torch.sqrt(shots.to(probabilities.dtype)).unsqueeze(-1) * probabilities.rsqrt()
    whitened = (jacobian * scale.unsqueeze(-1)).flatten(1, 2)
    profiled = whitened
    basis = whitened.new_zeros((whitened.shape[0], whitened.shape[1], 0))
    nuisance_rank = torch.zeros(whitened.shape[0], device=whitened.device, dtype=torch.long)
    if nuisance_jacobian is not None:
        if nuisance_jacobian.shape[:3] != probabilities.shape or nuisance_jacobian.ndim != 4:
            raise ValueError("Nuisance derivatives must be [B,S,C,nuisance]")
        _finite_selected(nuisance_jacobian, "nuisance Jacobian")
        if not torch.allclose(
            nuisance_jacobian.sum(-2),
            torch.zeros_like(nuisance_jacobian.sum(-2)),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Nuisance derivatives must preserve the simplex")
        if nuisance_jacobian.shape[-1]:
            nuisance = (nuisance_jacobian * scale.unsqueeze(-1)).flatten(1, 2)
            # Rank selection is a local numerical-support decision, not a
            # differentiable objective or a parameter the model may inflate.
            with torch.no_grad():
                left, singular, _ = torch.linalg.svd(nuisance.detach(), full_matrices=False)
                threshold = torch.maximum(
                    singular.amax(-1, keepdim=True) * relative_threshold,
                    singular.new_full((singular.shape[0], 1), absolute_threshold),
                )
                support = singular > threshold
                basis = left * support.unsqueeze(-2)
                nuisance_rank = support.sum(-1)
            profiled = whitened - basis @ (basis.transpose(-1, -2) @ whitened)
    gram = profiled.transpose(-1, -2) @ profiled
    return {
        "gram": (gram + gram.transpose(-1, -2)) * 0.5,
        "raw_gram": whitened.transpose(-1, -2) @ whitened,
        "whitened_jacobian": whitened,
        "profiled_jacobian": profiled,
        "score_scale": scale,
        "nuisance_basis": basis,
        "nuisance_rank": nuisance_rank,
    }


def profile_observation_residual(
    residual: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Use exactly the same score metric and shared-nuisance support for RSM."""
    if residual.shape != geometry["score_scale"].shape:
        raise ValueError("Residual must align with the categorical observation law")
    value = (residual * geometry["score_scale"]).flatten(1)
    basis = geometry["nuisance_basis"]
    return value - (basis @ (basis.transpose(-1, -2) @ value.unsqueeze(-1))).squeeze(-1)


def _csm_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != teacher.shape or prediction.ndim != 3:
        raise ValueError("CSM prediction and teacher must be aligned [B,D,D]")
    if prediction.shape[-1] != prediction.shape[-2] or mask.shape != prediction.shape[:1]:
        raise ValueError("CSM matrices must be square with a per-row validity mask")
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    predicted, target = prediction[mask], teacher[mask]
    _finite_selected(predicted, "CSM prediction")
    _finite_selected(target, "CSM teacher")
    target = (target + target.transpose(-1, -2)) * 0.5
    predicted = (predicted + predicted.transpose(-1, -2)) * 0.5
    p_trace = predicted.diagonal(dim1=-2, dim2=-1).sum(-1)
    t_trace = target.diagonal(dim1=-2, dim2=-1).sum(-1)
    if bool((p_trace < -1e-6).any()) or bool((t_trace < -1e-6).any()):
        raise ValueError("CSM Gram traces cannot be negative")
    p_trace, t_trace = p_trace.clamp_min(0), t_trace.clamp_min(0)
    # Shape and raw information strength are distinct. log1p avoids allowing
    # high-exposure rows to dominate merely through their physical units.
    p_shape = predicted / p_trace.clamp_min(1e-8)[:, None, None]
    t_shape = target / t_trace.clamp_min(1e-8)[:, None, None]
    shape_loss = (p_shape - t_shape).square().sum((-1, -2))
    strength_loss = (torch.log1p(p_trace) - torch.log1p(t_trace)).square()
    return (shape_loss + strength_loss).mean()


def _phase_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    zero: torch.Tensor,
) -> torch.Tensor:
    conditional = "conditional_R1_re" in output
    prefix = "conditional_R1" if conditional else "R1"
    real, imaginary = output.get(prefix + "_re"), output.get(prefix + "_im")
    if real is None or imaginary is None:
        return zero
    labelled = _teacher(
        batch,
        "target_phase_edge_cos_sin",
        "target_phase_edge_mask",
        real,
    )
    if labelled is None:
        return zero
    target, mask = labelled
    if not bool(mask.any()):
        return (real.sum() + imaginary.sum()) * 0.0
    edges, weights = batch.get("phase_edges"), batch.get("target_phase_edge_weight")
    if not isinstance(edges, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise ValueError("Valid phase labels require ordered edges and physical edge weights")
    if real.shape != imaginary.shape or real.ndim != 3 or real.shape[-1] != real.shape[-2]:
        raise ValueError("R1 real/imaginary parts must be [B,N,N]")
    edges = edges.to(real.device)
    incidence_matrix(edges, real.shape[-1], dtype=real.dtype)  # Validates endpoint semantics.
    if edges.ndim == 2:
        edges = edges.unsqueeze(0).expand(real.shape[0], -1, -1)
    if mask.shape != edges.shape[:2] or target.shape != (*mask.shape, 2):
        raise ValueError("Phase target, edges and masks must align")
    weights = weights.detach().to(real)
    if weights.shape != mask.shape:
        raise ValueError("Phase Fisher weights must align with edge labels")
    _finite_selected(target[mask], "phase teacher")
    _finite_selected(weights[mask], "phase weights")
    if bool((weights[mask] < 0).any()):
        raise ValueError("Phase information weights must be nonnegative")
    if not torch.allclose(
        target[mask].square().sum(-1),
        torch.ones_like(weights[mask]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("Valid phase teachers must be unit cosine/sine vectors")
    row = torch.arange(real.shape[0], device=real.device)[:, None]
    re = real[row, edges[..., 0], edges[..., 1]]
    im = imaginary[row, edges[..., 0], edges[..., 1]]
    # Normalize R1 to the conditional block before applying the numerical
    # near-zero scale. Phase itself is unchanged by positive sector mass.
    sector = output.get("p1")
    if sector is not None and not conditional:
        sector = sector.reshape(real.shape[0], 1).clamp_min(1e-8)
        re, im = re / sector, im / sector
    safe_target = torch.where(mask.unsqueeze(-1), target, torch.zeros_like(target))
    safe_weight = torch.where(mask, weights, torch.zeros_like(weights))
    magnitude = torch.sqrt(re.square() + im.square() + 1e-12)
    cosine = (re * safe_target[..., 0] + im * safe_target[..., 1]) / magnitude
    error = 1.0 - cosine
    row_weight = safe_weight.sum(-1)
    usable = row_weight > 0
    if not bool(usable.any()):
        return (re.sum() + im.sum()) * 0.0
    per_row = (error * safe_weight).sum(-1) / row_weight.clamp_min(1e-12)
    return per_row[usable].mean()


def _candidate_mask(
    batch: Mapping[str, object],
    logits: torch.Tensor,
) -> torch.Tensor:
    value = batch.get("public_candidate_valid_mask")
    if value is None:
        return torch.ones_like(logits, dtype=torch.bool)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.shape != logits.shape
    ):
        raise ValueError("Public candidate validity must be Boolean [B,K]")
    mask = value.to(logits.device)
    if not bool(mask.any(-1).all()):
        raise ValueError("Each row must retain at least one public candidate")
    return mask


def compute_training_geometry_losses(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Return five real autograd objectives; absent supervision stays masked.

    Required teacher keys are paired with explicit masks. NaNs in unavailable
    label slots are ignored by selecting valid entries before arithmetic.
    A partially labelled candidate library does not become a different policy:
    expected regret is trained only when every public candidate has a utility.
    """
    zero = _anchor(output)
    result = {name: zero for name in GEOMETRY_LOSS_NAMES}
    prediction = output.get("idea_csm_gram")
    if prediction is not None:
        result["csm_gram"] = prediction.sum() * 0.0
        labelled = _teacher(batch, "target_csm_gram", "target_csm_mask", prediction)
        if labelled is not None:
            result["csm_gram"] = _csm_loss(prediction, *labelled)
    result["phase_laplacian"] = _phase_loss(output, batch, zero)
    prediction = output.get("idea_rsm_fraction")
    if prediction is not None:
        result["rsm"] = prediction.sum() * 0.0
        labelled = _teacher(batch, "target_rsm_fraction", "target_rsm_mask", prediction)
        if labelled is not None:
            target, mask = labelled
            if prediction.shape != target.shape or mask.shape != prediction.shape:
                raise ValueError("RSM fraction and validity must be aligned [B]")
            if bool(mask.any()):
                _finite_selected(target[mask], "RSM teacher")
                if bool(((target[mask] < 0) | (target[mask] > 1)).any()):
                    raise ValueError("RSM teacher fractions must be in [0,1]")
                result["rsm"] = (prediction[mask] - target[mask]).square().mean()
            else:
                result["rsm"] = prediction.sum() * 0.0
    logits = output.get("idea_fd_logits")
    if logits is not None:
        result["fd_utility"] = logits.sum() * 0.0
        labelled = _teacher(batch, "target_fd_utility", "target_fd_utility_mask", logits)
        if labelled is not None:
            utility, label_mask = labelled
            public_mask = _candidate_mask(batch, logits)
            if utility.shape != logits.shape or label_mask.shape != logits.shape:
                raise ValueError("FD utilities and masks must align with [B,K] logits")
            rows = (label_mask | ~public_mask).all(-1)
            if bool(rows.any()):
                candidate_mask = public_mask[rows]
                selected = utility[rows]
                _finite_selected(selected[candidate_mask], "FD utility")
                scores = logits[rows].masked_fill(~candidate_mask, -torch.inf)
                probability = torch.softmax(scores, dim=-1)
                usable_utility = torch.where(candidate_mask, selected, torch.zeros_like(selected))
                best = selected.masked_fill(~candidate_mask, -torch.inf).amax(-1)
                result["fd_utility"] = (best - (probability * usable_utility).sum(-1)).mean()
            else:
                result["fd_utility"] = logits.sum() * 0.0
    logits = output.get("idea_fd_harm_logits")
    if logits is not None:
        result["fd_harm"] = logits.sum() * 0.0
        labelled = _teacher(batch, "target_fd_harm", "target_fd_harm_mask", logits)
        if labelled is not None:
            target, mask = labelled
            if target.shape != logits.shape or mask.shape != logits.shape:
                raise ValueError("FD harm labels and masks must align with [B,K] logits")
            mask = mask & _candidate_mask(batch, logits)
            if bool(mask.any()):
                _finite_selected(target[mask], "FD harm")
                if bool(((target[mask] != 0) & (target[mask] != 1)).any()):
                    raise ValueError("Counterfactual harm labels must be binary")
                result["fd_harm"] = F.binary_cross_entropy_with_logits(
                    logits[mask],
                    target[mask],
                )
            else:
                result["fd_harm"] = logits.sum() * 0.0
    return result


training_geometry_losses = compute_training_geometry_losses


__all__ = [
    "GEOMETRY_LOSS_NAMES",
    "compute_training_geometry_losses",
    "training_geometry_losses",
    "incidence_matrix",
    "weighted_phase_laplacian",
    "joint_profiled_count_geometry",
    "profile_observation_residual",
]
