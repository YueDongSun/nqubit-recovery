"""Held-out module-specific diagnostics for the corrected inference study."""

from __future__ import annotations

import itertools
import numpy as np
import torch


PARTITION_SCORE_VERSION = "labeled_unordered_pairs_v2"


def _legal_partitions(n, device):
    """Enumerate each equivalence relation once as a restricted growth string."""
    choices = []
    for tail in itertools.product(range(n), repeat=n - 1):
        labels = (0, *tail)
        if all(labels[i] <= 1 + max(labels[:i]) for i in range(1, n)):
            values = torch.tensor(labels, device=device)
            choices.append(values[:, None] == values[None, :])
    return torch.stack(choices)


def _rank_auc(positive, negative):
    difference = positive[:, None] - negative[None, :]
    return float((difference.gt(0).float() + 0.5 * difference.eq(0)).mean())


def auxiliary_metrics(output, batch):
    diagnostics = {}
    if "idea_noise_gates" in output:
        g = output["idea_noise_gates"]
        diagnostics["router_fraction"] = g.mean(0).tolist()
        diagnostics["routing_entropy_nats"] = float(-torch.special.xlogy(g, g).sum(-1).mean())
    if "idea_csm_gram" in output:
        teacher, pred = batch["target_csm_gram"], output["idea_csm_gram"]
        teacher_eig, pred_eig = torch.linalg.eigvalsh(teacher.double()), torch.linalg.eigvalsh(
            pred.double()
        )
        diagnostics["mean_gram_eigenvalues"] = {
            "teacher": teacher_eig.mean(0).tolist(),
            "predicted": pred_eig.mean(0).tolist(),
        }
    if "target_phase_edge_weight" in batch:
        weight = batch["target_phase_edge_weight"]
        valid = batch["target_phase_edge_mask"]
        diagnostics["laplacian_edges"] = dict(
            mean_informative_edges=float(valid.sum(-1).float().mean()),
            available_topological_edges=weight.shape[-1],
            connected_fraction=float(valid.all(-1).float().mean()),
            mean_weight_by_edge=weight.mean(0).tolist(),
        )
    if "idea_rsm_fraction" in output:
        valid = batch["target_rsm_mask"]
        diagnostics["rsm_mse"] = float(
            (output["idea_rsm_fraction"][valid] - batch["target_rsm_fraction"][valid])
            .square()
            .mean()
        )
        truth, harm = batch["target_fd_utility"], batch["target_fd_harm"]
        prediction = output["idea_fd_logits"]
        risk = output["idea_fd_harm_logits"].sigmoid()
        diagnostics["fd_utility_rmse"] = float((prediction - truth).square().mean().sqrt())
        diagnostics["fd_harm_brier"] = float((risk - harm).square().mean())
        non_noop = batch["public_candidate_actions"].abs().sum((-2, -1)) > 0
        best = prediction.masked_fill(~non_noop, -torch.inf).argmax(-1, keepdim=True)
        order = risk.gather(1, best).squeeze(-1).argsort()
        curve = {}
        for coverage in (0.25, 0.5, 0.75, 1.0):
            use = order[: max(1, round(len(order) * coverage))]
            curve[str(coverage)] = dict(
                mean_conditional_gain=float(truth.gather(1, best)[use].mean()),
                harm_given_execution=float(harm.gather(1, best)[use].mean()),
            )
        diagnostics["counterfactual_nonnoop_coverage"] = curve
    prob = output["pair_entanglement_probs"]
    n = prob.shape[-1]
    if n <= 4:
        prob = 0.5 * (prob + prob.transpose(1, 2))
        truth = batch["target_partition"].bool()
        upper = torch.ones(n, n, device=prob.device, dtype=torch.bool).triu(1)
        legal = _legal_partitions(n, prob.device)
        # Squared-error score remains finite at exact zero or one.
        dist = (prob[:, None, upper] - legal[None, :, upper].float()).square().sum(-1)
        decoded = legal[dist.argmin(-1)]
        within = batch["target_operational_noise"] == 5
        labeled = batch["target_partition_mask"][:, upper].bool()
        complete = labeled.all(-1)
        exact = decoded[:, upper].eq(truth[:, upper]).all(-1)
        within_complete = within & complete
        pair_scores = prob[:, upper]
        pair_truth = truth[:, upper]
        pos = pair_scores[within[:, None] & labeled & pair_truth]
        neg = pair_scores[within[:, None] & labeled & ~pair_truth]
        diagnostics["legal_partition"] = dict(
            score_version=PARTITION_SCORE_VERSION,
            count=len(legal), complete_rows=int(complete.sum()),
            within_family_complete_rows=int(within_complete.sum()),
            exact_fraction=float(exact[complete].float().mean()) if complete.any() else None,
            within_family_exact_fraction=float(exact[within_complete].float().mean()) if within_complete.any() else None,
            within_family_pair_auc=_rank_auc(pos, neg) if len(pos) and len(neg) else None,
        )
        pred_h = output["hyperedge_entanglement_probs"]
        valid = batch["target_hyperedge_mask"]
        diagnostics["hyperedge_brier"] = float(
            (pred_h[valid] - batch["target_hyperedge"][valid]).square().mean()
        )
    return diagnostics
