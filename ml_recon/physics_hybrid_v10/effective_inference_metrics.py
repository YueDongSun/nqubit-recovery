"""Paired per-state estimation, geometry and uncertainty diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from .effective_projector_law import effective_probabilities
from .training_ideas_objective import normalized_state, uncertainty_scores


def command_and_truth(batch):
    command = torch.complex(batch["prepared_re"], batch["prepared_im"])
    truth = torch.complex(batch["target_R1_re"], batch["target_R1_im"])
    return command.unsqueeze(-1) * command.conj().unsqueeze(-2), truth


def fit_prior(train):
    command, truth = command_and_truth(train)
    off = command - torch.diag_embed(command.diagonal(dim1=-2, dim2=-1))
    coefficient = float(((off.conj() * truth).real.sum() / off.abs().square().sum()).clamp(0, 1))
    # The restricted [0,1] argmin yields a convex dephasing channel.
    pairs = train["target_pair"][train["target_pair_mask"]]
    return coefficient, float(pairs.mean())


def prior_density(command, retention):
    diag = torch.diag_embed(command.diagonal(dim1=-2, dim2=-1))
    return retention * command + (1 - retention) * diag


def classical_estimate(batch, prior, steps=30):
    """Visible multinomial MAP with a public-command TRAIN prior; batched independent fits."""
    analyzer = torch.complex(batch["analyzer_re"], batch["analyzer_im"])
    calibration = batch["effective_calibration"]
    eig, vec = torch.linalg.eigh(prior)
    if torch.any(eig < -1e-6):
        raise ValueError("TRAIN command prior is not PSD")
    factor = (vec * eig.clamp_min(0).sqrt().unsqueeze(-2)).detach().requires_grad_()
    optimizer = torch.optim.Adam([factor], lr=0.015)
    counts = batch["counts"]
    trials = counts.sum((-2, -1))

    def estimate():
        raw = factor @ factor.mH
        return raw / raw.diagonal(dim1=-2, dim2=-1).real.sum(-1)[:, None, None]

    for _ in range(steps):
        optimizer.zero_grad()
        rho = estimate()
        pi = effective_probabilities(rho, analyzer, calibration)
        nll = -(counts * pi.log()).sum((-2, -1)) / trials
        score = nll + 0.0005 * (rho - prior).abs().square().sum((-2, -1))
        score.sum().backward()
        optimizer.step()
    return estimate().detach()


def state_risk(rho, truth):
    return (rho - truth).abs().square().sum((-2, -1))


def rank_floor(truth, rank):
    values = torch.linalg.eigvalsh(
        truth.double() if not truth.is_complex() else truth.to(torch.complex128)
    )
    if rank >= values.shape[-1]:
        return torch.zeros_like(values[:, 0])
    discarded = values[:, :-rank]
    return discarded.square().sum(-1) + discarded.sum(-1).square() / rank


def paired_estimate(value, family, draws=1000):
    value, family = np.asarray(value, dtype=np.float64), np.asarray(family)
    if not np.isfinite(value).all():
        raise ValueError("non-finite paired estimate")
    rng = np.random.default_rng(729171)
    boot = np.zeros(draws)
    for kind in np.unique(family):
        sample = value[family == kind]
        boot += rng.choice(sample, size=(draws, len(sample)), replace=True).sum(1)
    return {
        "mean": float(value.mean()),
        "ci95": np.quantile(boot / len(value), [0.025, 0.975]).tolist(),
    }


def public_uncertainty_score(output, mode_mask):
    """Freeze selection using active public modes and the declared reference."""
    phase_mask = mode_mask.bool().clone()
    phase_mask[:, 0] = False
    av, pv = output["W_amplitude_variance"], output["W_phase_variance"]
    mean_amp = (av * mode_mask).sum(-1) / mode_mask.sum(-1)
    return mean_amp + (pv * phase_mask).sum(-1) / phase_mask.sum(-1).clamp_min(1)


@torch.no_grad()
def score_model(model, batch, *, n, retention, pair_prior, classical):
    from .training_ideas_experiment import evaluate_training_ideas

    summary, samples = evaluate_training_ideas(model, batch, N=n)
    output = model(batch, N=n)
    rho = normalized_state(output)
    command, truth = command_and_truth(batch)
    prior = prior_density(command, retention)
    family = (
        torch.where(batch["target_is_nominal"], -1, batch["target_operational_noise"]).cpu().numpy()
    )
    analyzer = torch.complex(batch["analyzer_re"], batch["analyzer_im"])
    calibration = batch["effective_calibration"]
    query = batch["query_counts"]
    rng = np.random.default_rng(6000 + n)
    probabilities = effective_probabilities(command, analyzer, calibration).double().cpu().numpy()
    counter_counts = torch.zeros_like(batch["counts"])
    for row, setting in batch["setting_mask"].nonzero().cpu().tolist():
        pi = probabilities[row, setting]
        # The owner already validated normalization in float32. Multinomial
        # samples require the same unit sum after changing numerical precision.
        draws = rng.multinomial(int(batch["shots"][row, setting]), pi / pi.sum())
        counter_counts[row, setting] = counter_counts.new_tensor(draws)
    counter = normalized_state(model(dict(batch, counts=counter_counts), N=n))
    sample_tensors = {
        "network": rho,
        "command": command,
        "train_prior": prior,
        "classical": classical,
        "command_generated_counts_network": counter,
    }
    risks = {
        name: state_risk(value, truth).cpu().tolist() for name, value in sample_tensors.items()
    }
    # Per-paid-query multinomial cross entropy without factorial term; model
    # differences equal NLL differences and are independent of count constants.
    query_nll = {
        name: (
            -(query * effective_probabilities(value, analyzer, calibration).log()).sum((-2, -1))
            / query.sum((-2, -1))
        )
        .cpu()
        .tolist()
        for name, value in sample_tensors.items()
    }
    primary = np.array(risks["network"])
    summary["state_risk_by_estimator"] = {k: paired_estimate(v, family) for k, v in risks.items()}
    summary["paired_risk_network_minus"] = {
        k: paired_estimate(primary - np.asarray(v), family)
        for k, v in risks.items()
        if k != "network"
    }
    summary["query_cross_entropy_per_read"] = {
        k: paired_estimate(v, family) for k, v in query_nll.items()
    }
    floor = rank_floor(truth, 4)
    summary["rank_four_hs_squared_floor"] = {
        "all": float(floor.mean()),
        "dephased": float(floor[batch["target_operational_noise"] == 3].mean()),
    }
    err = uncertainty_scores(output, truth)
    av, pv = output["W_amplitude_variance"], output["W_phase_variance"]
    valid = err["phase_valid"]
    # Risk ranking is frozen before looking at truth. Phase 0 is the declared
    # public reference. The truth-coherence mask is used only to score errors.
    u = public_uncertainty_score(output, batch["mode_mask"])
    order = u.argsort().cpu().numpy()
    summary["risk_coverage_by_predicted_variance"] = {
        str(fraction): float(primary[order[: max(1, round(fraction * len(order)))]].mean())
        for fraction in (0.25, 0.5, 0.75, 1.0)
    }
    summary["amplitude_squared_error_over_variance"] = float(
        (err["amplitude_residual"].square() / av).mean()
    )
    summary["phase_squared_error_over_variance"] = float(
        ((err["phase_residual"].square() / pv)[valid]).mean()
    )
    mask = batch["target_pair_mask"] & batch["target_operational_noise"].eq(5)[:, None, None]
    predicted = output["pair_entanglement_probs"][mask]
    actual = batch["target_pair"][mask]
    summary["partition_within_family"] = dict(
        rows=int(batch["target_operational_noise"].eq(5).sum()),
        network_brier=float((predicted - actual).square().mean()),
        train_prior_brier=float((pair_prior - actual).square().mean()),
    )
    from .projector_auxiliary_metrics import auxiliary_metrics

    summary["auxiliary_diagnostics"] = auxiliary_metrics(output, batch)
    samples.update(
        estimators=risks,
        family=family.tolist(),
        uncertainty=u.cpu().tolist(),
        rank_four_floor=floor.cpu().tolist(),
        query_cross_entropy_per_read=query_nll,
    )
    return summary, samples
