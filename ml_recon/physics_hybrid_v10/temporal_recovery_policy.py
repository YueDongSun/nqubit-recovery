"""Native projector integration for the frozen classifier, feedback student and temporal hierarchy."""

from __future__ import annotations

import itertools

import torch
from torch import nn

from .effective_projector_observation import make_public_observation
from .feedback_n_student_csm import V10FeedbackNStudentCSM
from .trackb_inherited_hierarchy import (
    OrderedTrackBHierarchy,
    TrackBInheritedHierarchyConfig,
)
from .temporal_recovery_environment import feasible, state_vector


def legal_map(scores):
    """Exact maximum pair-label likelihood over legal partitions for N=3,4."""
    b, n, _ = scores.shape
    if n not in (3, 4):
        raise ValueError("exact partition enumeration is registered only for 3/4Q")
    ids = [
        a
        for a in itertools.product(range(n), repeat=n)
        if a[0] == 0 and all(a[j] <= 1 + max(a[:j]) for j in range(1, n))
    ]
    labels = torch.tensor(ids, device=scores.device)
    same = labels[:, None, :] == labels[:, :, None]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=scores.device), 1)
    # Cross entropy operates on logits to avoid singular logarithms.
    odds = scores[:, upper] @ same[:, upper].to(scores).T
    return same[odds.argmax(-1)].to(scores)


def raw_record(env, obs, executed):
    """Operation-local counts, settings, exposures, command and executed action at its own state."""
    counts = obs.counts.double()
    b, ops, _ = counts.shape
    centered_counts = (counts - obs.shots * env.nominal) / (
        obs.shots * env.nominal
    ).sqrt()
    features = [
        centered_counts[:, :, None, :].expand(-1, -1, env.n, -1),
        obs.analyzers.real[None, :, :, None].expand(b, -1, -1, -1),
        obs.analyzers.imag[None, :, :, None].expand(b, -1, -1, -1),
        torch.full(
            (b, ops, env.n, 1), float(torch.log1p(torch.tensor(obs.shots))) / 10
        ),
        obs.executed_command[:, None].expand(-1, ops, -1, -1),
        executed[:, None].expand(-1, ops, -1, -1),
    ]
    return torch.cat(features, -1).float()


def fisher_dispersion(env, obs):
    estimate = env.estimate(obs)
    prob, jac = env.geometry(estimate)
    fisher = obs.shots * jac.transpose(1, 2) @ (jac / prob.flatten(1).unsqueeze(-1))
    covariance = torch.linalg.inv(fisher + env.prior_precision)
    covariance = env.projector @ covariance @ env.projector
    est_pop = state_vector(estimate).abs().square()
    pop_jac = 2 * (torch.diag_embed(est_pop) - est_pop[:, :, None] * est_pop[:, None])
    pop_var = (pop_jac @ covariance[:, ::2, ::2] @ pop_jac.transpose(-1, -2)).diagonal(
        dim1=1, dim2=2
    )
    phase_cov = covariance[:, 1::2, 1::2]
    phase_var = (
        phase_cov.diagonal(dim1=1, dim2=2)
        + phase_cov[:, :1, 0]
        - 2 * phase_cov[:, :, 0]
    )
    var = torch.stack([pop_var, phase_var / torch.pi**2], -1).float()
    return estimate, fisher, est_pop, var


def current_context(env, obs, classifier, device):
    b, ops, _ = obs.counts.shape
    shots = torch.full((b, ops), obs.shots, dtype=torch.int64)
    command = state_vector(obs.executed_command)
    public = make_public_observation(
        counts=obs.counts,
        shots=shots,
        analyzers=obs.analyzers,
        prepared=command,
        calibration=torch.tensor(env.calibration).expand(b, -1),
        visible_mask=torch.ones_like(shots, dtype=torch.bool),
    )
    # Recovery target is the same uniform W for every next-preparation command.
    public["public_control_target"] = torch.full_like(
        public["public_control_target"], 0
    )
    public["public_control_target"][..., 0] = env.n**-0.5
    model_input = {k: v.to(device) for k, v in public.items()}
    with torch.no_grad():
        out = classifier(model_input, N=env.n, fine_identifiable=False)
    population = out["W_populations"].detach().cpu()
    re = out["W_coherence_re"].detach().cpu()[:, :, 0]
    im = out["W_coherence_im"].detach().cpu()[:, :, 0]
    estimate, fisher, est_pop, var = fisher_dispersion(env, obs)
    # A local model-based dispersion, not a calibrated confidence gate.
    strength = torch.hypot(re, im)
    semantics = {
        "denoised_measurement_tokens": torch.stack(
            [population, torch.atan2(im, re) / torch.pi], -1
        ),
        "covariance": var,
        "esm_mode_tokens": torch.stack([strength, torch.zeros_like(strength)], -1),
    }
    block = fisher.reshape(b, env.n, 2, env.n, 2).square().sum((2, 4)).sqrt()
    magnitude = block.diagonal(dim1=1, dim2=2).sqrt()
    edge = (block / (magnitude[:, :, None] * magnitude[:, None, :] + 1e-15)).float()
    d = fisher.diagonal(dim1=1, dim2=2).reshape(b, env.n, 2)
    target = torch.zeros(b, env.n, 2)
    target[..., 0] = 1 / env.n
    estimate = estimate.float()
    node = torch.cat(
        [
            d.log1p().float(),
            var,
            estimate,
            est_pop.float()[..., None],
            obs.executed_command.float(),
        ],
        -1,
    )
    glob = torch.stack(
        [
            d.sum((1, 2)).log1p(),
            block.mean((1, 2)).log1p(),
            var[..., 0].mean(1),
            (var[..., 1] * torch.pi**2).mean(1),
        ],
        -1,
    ).float()
    gate = torch.cat([glob, torch.tensor(env.calibration[:4]).expand(b, -1)], -1)
    inputs = {
        "noisy_measurement_tokens": estimate,
        "physical_mode_tokens": obs.executed_command.float(),
        "target_mode_tokens": target,
        "csm_node_context": node,
        "csm_edge_context": edge,
        "csm_global_context": glob,
        "gate_features": gate,
        "analytic_action": feasible(-estimate),
        "mode_mask": torch.ones(b, env.n, dtype=torch.bool),
    }
    # Include ESM only if the export carries independently qualified control-use
    # evidence. Baseline/factor-only heads are never decoded into a policy field.
    if getattr(classifier, "partition_control_qualified", False):
        partition = legal_map(out["pair_entanglement_logits"].detach().cpu())
        inputs.update(
            partition_current_context=partition,
            partition_target_context=torch.ones_like(partition),
        )
    return {**inputs, **semantics}


def prepare_prefix(env, prefix, current, indices=None):
    history, raw = [], []
    old_est = env.estimate(prefix.observations[0]).float()
    for before, after, action in zip(
        prefix.observations[:-1], prefix.observations[1:], prefix.actions
    ):
        new_est = env.estimate(after).float()
        history.append(
            torch.cat(
                [old_est, action.float(), new_est, torch.zeros_like(old_est[..., :1])],
                -1,
            )
        )
        raw.append(raw_record(env, before, action))
        old_est = new_est
    b, n = old_est.shape[:2]
    record = raw_record(env, prefix.observations[-1], torch.zeros_like(old_est))
    payload = {
        **current,
        "history_tokens": torch.stack(history, 1)
        if history
        else torch.empty(b, 0, n, 7),
        "raw_prefix": torch.stack([*raw, record], 1),
    }
    return {k: v[indices] if indices is not None else v for k, v in payload.items()}


FEEDBACK_CONFIG = dict(input_dim=6, gate_input_dim=8, width=24, heads=4, blocks=1)
HIERARCHY_CONFIG = dict(
    width=24, heads=4, temporal_blocks=1, spatial_blocks=1, residual_scale=0.5
)


class RecoveryController(nn.Module):
    def __init__(self, feedback=None):
        super().__init__()
        self.feedback = feedback or V10FeedbackNStudentCSM(**FEEDBACK_CONFIG)
        self.hierarchy = OrderedTrackBHierarchy(
            TrackBInheritedHierarchyConfig(**HIERARCHY_CONFIG), FEEDBACK_CONFIG["width"]
        )
        self.raw_encoder = nn.Sequential(nn.Linear(11, 24), nn.GELU(), nn.Linear(24, 6))
        self.raw_query = nn.Linear(6, 9, bias=False)

    def forward(self, payload, arm):
        inputs = {k: v for k, v in payload.items() if k != "raw_prefix"}
        history = inputs["history_tokens"]
        b, t, n, _ = history.shape
        inputs["history_mask"] = torch.ones(
            (b, t), dtype=torch.bool, device=history.device
        )
        if arm == "current":
            inputs["history_tokens"] = torch.zeros_like(history)
        temporal = dict(inputs)
        sem = {
            k: inputs[k]
            for k in ("denoised_measurement_tokens", "esm_mode_tokens", "covariance")
        }
        if arm in ("current", "summary", "ordered", "pooled", "shuffled"):
            raw = self.raw_encoder(payload["raw_prefix"]).mean(2)
            temporal["csm_node_context"] = inputs["csm_node_context"] + self.raw_query(
                raw[:, -1]
            )
        if arm in ("ordered", "pooled", "shuffled"):
            records = raw[:, :-1]
            if arm == "pooled" and t:
                records = records.mean(1, keepdim=True).expand(-1, t, -1, -1)
            if arm == "shuffled" and t >= 3:
                perm = torch.randperm(
                    t - 1, generator=torch.Generator().manual_seed(913 + t)
                ).tolist()
                records = records[:, [*perm, t - 1]]
            temporal["history_tokens"] = torch.cat(
                [records, torch.zeros_like(records[..., :1])], -1
            )
        elif arm not in ("zero", "current", "summary"):
            raise ValueError("unknown temporal controller arm")
        residual = None if arm == "zero" else self.hierarchy(temporal, sem)
        out = self.feedback(inputs, hierarchy_residual=residual)
        return feasible(torch.stack([out["delta_log_amp"], out["delta_phase"]], -1))
