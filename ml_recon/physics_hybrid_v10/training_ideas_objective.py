"""Registered losses for the main V10 model's projector-readout experiment.

The new experiment has its own declared single-excitation count law. It does
not reuse the legacy Pauli-family labels or claim native FPGA calibration.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F


def _masked_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("loss values and availability mask must have identical shapes")
    if not torch.isfinite(values).all():
        raise ValueError("non-finite per-row training loss")
    return (values * mask).sum() / mask.sum().clamp_min(1)


def normalized_state(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "conditional_R1_re" in output:
        return torch.complex(output["conditional_R1_re"], output["conditional_R1_im"])
    rho = torch.complex(output["R1_re"].float(), output["R1_im"].float())
    trace = rho.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    if torch.any(trace <= 0) or not torch.isfinite(trace).all():
        raise ValueError("conditional W prediction requires a positive finite sector trace")
    return rho / trace[:, None, None]


def projector_predictions(
    output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    from .training_ideas_data import readout_probability_from_rho

    if "effective_calibration" in batch:
        from .effective_projector_law import effective_probabilities
        return effective_probabilities(normalized_state(output),
            torch.complex(batch["analyzer_re"], batch["analyzer_im"]), batch["effective_calibration"])
    params = batch["readout_parameters"]
    if params.ndim != 2 or params.shape[-1] != 3:
        raise ValueError("readout_parameters must be public [batch,3] calibration")
    analyzers = torch.complex(batch["analyzer_re"], batch["analyzer_im"])
    return readout_probability_from_rho(
        normalized_state(output),
        analyzers,
        herald_probability=params[:, 0],
        read_efficiency=params[:, 1],
        background_probability=params[:, 2],
    )


def uncertainty_scores(output, target):
    """Proper second-moment score for centered log amplitude and wrapped phase error."""
    rho = normalized_state(output).detach()
    diagonal = target.diagonal(dim1=-2, dim2=-1).real
    predicted_diagonal = rho.diagonal(dim1=-2, dim2=-1).real
    if (diagonal <= 0).any() or (predicted_diagonal <= 0).any():
        raise ValueError("centered log amplitude requires occupied modes")
    amplitude = .5 * (predicted_diagonal.log() - diagonal.log())
    amplitude = amplitude - amplitude.mean(-1, keepdim=True)
    reference = target[:, :, 0]
    # A missing target coherence has no reference phase. Exclude the fixed phase
    # reference itself. Circular differences avoid a discontinuity across +/-pi.
    phase_valid = reference.abs() > 1e-7
    phase_valid[:, 0] = False
    phase = (rho[:, :, 0] * reference.conj()).angle()
    va, vp = output["W_amplitude_variance"], output["W_phase_variance"]
    a_score = .5 * (va.log() + amplitude.square() / va)
    p_score = .5 * (vp.log() + phase.square() / vp)
    return dict(nll=a_score.mean() + (p_score * phase_valid).sum() / phase_valid.sum().clamp_min(1),
                amplitude_residual=amplitude, phase_residual=phase, phase_valid=phase_valid)


def projector_primary_losses(
    output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    rho = normalized_state(output)
    target = torch.complex(batch["target_R1_re"], batch["target_R1_im"]).detach()
    state_per_row = (rho - target).abs().square().sum((-1, -2))
    state = _masked_rows(state_per_row, batch["target_w_mask"].bool())
    probability = projector_predictions(output, batch)
    if torch.any(probability <= 0) or not torch.isfinite(probability).all():
        raise ValueError("registered interior count law must give positive probabilities")
    query = batch["measurement_query_mask"].bool()
    counts = batch["query_counts"].to(probability.dtype)
    trials = batch["shots"].to(probability.dtype)
    if query.shape != trials.shape or counts.shape != probability.shape:
        raise ValueError("query counts, opportunity and probability shapes disagree")
    if torch.any(query & (trials <= 0)) or not torch.allclose(counts.sum(-1)[query], trials[query]):
        raise ValueError("query cells must conserve independently known opportunities")
    if torch.any(query.sum(-1) == 0):
        raise ValueError("each row must contain a held-out measurement")
    # Factorial constants are retained. Division by known R declares a
    # per-opportunity training scale, followed by operation-then-row averaging.
    log_coefficient = torch.lgamma(trials + 1) - torch.lgamma(counts + 1).sum(-1)
    log_probability = (counts * probability.log()).sum(-1)
    per_operation = -(log_coefficient + log_probability) / trials.clamp_min(1)
    count_per_row = (per_operation * query).sum(-1) / query.sum(-1)
    count_loss = count_per_row.mean()
    noise_raw = F.cross_entropy(
        output["operational_noise_logits"],
        batch["target_operational_noise"].long(),
        reduction="none",
    )
    noise = _masked_rows(noise_raw, batch["target_operational_noise_mask"].bool())
    primary = state + count_loss + 0.1 * noise
    extra = {}
    if "effective_calibration" in batch:
        uq = uncertainty_scores(output, target)["nll"]
        primary = primary + .02 * uq
        extra["uncertainty_nll"] = uq
    return {
        **extra,
        "primary": primary,
        "state_risk": state,
        "query_count_nll": count_loss,
        "noise_classification": noise,
    }


def projector_classifier_step(model, batch: Dict[str, torch.Tensor], *, N: int):
    """Use the main classifier, physical decoder and every enabled auxiliary loss."""
    from .training_geometry import compute_training_geometry_losses
    from .training_ideas import compute_training_ideas_losses

    output = model(batch, N=N, fine_identifiable=False)
    config = model.training_ideas_config
    primary = projector_primary_losses(output, batch)
    zero = primary["primary"] * 0.0
    esm = zero
    if config.esm:
        pairs = F.binary_cross_entropy(
            output["pair_entanglement_probs"].float(),
            batch["target_pair"].float(),
            reduction="none",
        )
        mask = batch["target_pair_mask"].bool()
        if mask.shape != pairs.shape or not torch.any(mask):
            raise ValueError("ESM experiment requires available pair targets")
        esm = (pairs * mask).sum() / mask.sum()
        edges = F.binary_cross_entropy(
            output["hyperedge_entanglement_probs"].float(),
            batch["target_hyperedge"].float(),
            reduction="none",
        )
        edge_mask = batch["target_hyperedge_mask"].bool()
        if edge_mask.shape != edges.shape or not torch.any(edge_mask):
            raise ValueError("ESM experiment requires available hyperedge targets")
        esm = esm + (edges * edge_mask).sum() / edge_mask.sum()
    auxiliary = compute_training_ideas_losses(
        output, batch, config, primary_count_loss=primary["query_count_nll"]
    )
    geometry = compute_training_geometry_losses(output, batch)
    geometry_total = zero
    if config.csm:
        geometry_total = geometry_total + config.csm_weight * geometry["csm_gram"]
    if config.laplacian:
        geometry_total = geometry_total + config.laplacian_weight * geometry["phase_laplacian"]
    if config.fd:
        geometry_total = geometry_total + config.rsm_weight * geometry["rsm"]
        geometry_total = geometry_total + config.fd_weight * geometry["fd_utility"]
        geometry_total = geometry_total + config.fd_harm_weight * geometry["fd_harm"]
    total = primary["primary"] + config.esm_weight * esm + auxiliary["ideas_total"] + geometry_total
    return {
        "loss": total,
        **primary,
        "ideas_esm": esm,
        **auxiliary,
        **{f"ideas_{name}": value for name, value in geometry.items()},
    }
