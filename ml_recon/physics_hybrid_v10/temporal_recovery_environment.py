"""Independent simulated preparations, native observations and causal ten-decision rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from nqrec.physics.single_event_mc import (
    calibrate_source_weights,
    four_cell_probabilities,
    k1_event_probability,
    sample_counts,
)

DECISIONS = 10
WINDOW_READS = 7000
AMPLITUDE_BOUND = 0.35
PHASE_BOUND = 0.70
CALIBRATION = (0.025, 0.65, 0.002, 0.35, 0.002)
PRIOR_SD = (0.30, 0.40)
VELOCITY_SD = (0.05, 0.08)


def centered(x):
    return x - x.mean(dim=-2, keepdim=True)


def feasible(action, projector=None):
    """Same support, gauge, and scalar ray bound for executed proposals and teachers."""
    if projector is None:
        delta = centered(action)
    else:
        delta = (projector @ action.flatten(1).unsqueeze(-1)).reshape_as(action)
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite projected action")
    ratio = (delta.abs() / delta.new_tensor([AMPLITUDE_BOUND, PHASE_BOUND])).amax(
        (1, 2)
    )
    return delta / ratio.clamp_min(1)[:, None, None]


def state_vector(coordinates):
    c = torch.exp(torch.complex(coordinates[..., 0], coordinates[..., 1]))
    return c / torch.linalg.vector_norm(c, dim=-1, keepdim=True)


def fidelity(coordinates):
    c = state_vector(coordinates)
    return c.mean(-1).abs().square() * c.shape[-1]


def allocation(n, first_fraction=None):
    """Identical total and final diagnostic; all eleven windows retain distinct READs."""
    operations = 3 * n - 2
    if WINDOW_READS % operations:
        raise ValueError("registered N must divide the declared observation budget")
    ordinary = WINDOW_READS // operations
    result = np.full(DECISIONS + 1, ordinary, dtype=np.int64)
    if first_fraction is not None:
        first = int(round(DECISIONS * ordinary * first_fraction))
        rest = DECISIONS * ordinary - first
        result[0] = first
        result[1:DECISIONS] = rest // (DECISIONS - 1)
        result[1 : 1 + rest % (DECISIONS - 1)] += 1
    if np.any(result <= 0) or result.sum() * operations != WINDOW_READS * (
        DECISIONS + 1
    ):
        raise ValueError("invalid eleven-window allocation")
    return result


@dataclass(frozen=True)
class Observation:
    counts: torch.Tensor
    shots: int
    analyzers: torch.Tensor
    executed_command: torch.Tensor
    acquisition: int


class PreparationEnvironment:
    """Fixed coherent device bias; each update changes the *next* preparation command.

    The conditional K1 target is uniform. Classical Fisher information and likelihood
    use the same fixed public calibration and K2=0 family as data generation.
    """

    def __init__(self, n, *, calibration=CALIBRATION):
        if n not in (3, 4):
            raise ValueError("registered systems are 3Q and 4Q")
        self.n = n
        self.calibration = tuple(float(v) for v in calibration)
        if len(self.calibration) != 5:
            raise ValueError("expected five public operating parameters")
        dtype = torch.float64
        s, eta, bg, eta_i, self.bg = [
            torch.tensor(x, dtype=dtype) for x in self.calibration
        ]
        self.joint = calibrate_source_weights(
            s, eta, bg, double_probability=s.new_zeros(())
        )["joint"]
        basis = torch.eye(n, dtype=torch.complex128)
        pairs = [
            (basis[k] + rot * basis[k + 1]) / np.sqrt(2)
            for k in range(n - 1)
            for rot in (1.0, -1j)
        ]
        self.analyzers = torch.stack([*basis, *pairs])
        self.transfer = self.analyzers * eta_i.sqrt()
        self.projector = torch.kron(
            torch.eye(n, dtype=dtype) - 1 / n, torch.eye(2, dtype=dtype)
        )
        self.zero = torch.zeros(1, n, 2, dtype=dtype)
        self.prior_precision = torch.diag(
            torch.tensor(PRIOR_SD, dtype=dtype).repeat(n).pow(-2)
        )
        self.nominal, self.derivatives = self.geometry(self.zero)
        self.linear_fisher = self.derivatives.transpose(-1, -2) @ (
            self.derivatives / self.nominal.flatten(1).unsqueeze(-1)
        )

    def law(self, coords):
        c = state_vector(coords)
        rho = c.unsqueeze(-1) * c.conj().unsqueeze(-2)
        r1 = k1_event_probability(rho[:, None], self.transfer[None], self.bg)
        zero = torch.zeros_like(r1)
        responses = torch.stack((zero + self.bg, r1, zero), dim=-1)
        return four_cell_probabilities(
            self.joint, responses.unsqueeze(-1).expand(-1, -1, -1, 2)
        )

    def geometry(self, coords):
        """Finite differences of the actual owner with centered coordinates (no nuisance estimated)."""
        b, n, _ = coords.shape
        d, h = 2 * n, 1e-5
        perturb = self.projector.reshape(d, n, 2)[None] * h
        high = self.law((coords[:, None] + perturb).reshape(-1, n, 2)).reshape(b, d, -1)
        low = self.law((coords[:, None] - perturb).reshape(-1, n, 2)).reshape(b, d, -1)
        return self.law(coords), ((high - low) / (2 * h)).transpose(1, 2)

    def estimate(self, obs):
        """One Fisher-scoring step about the target, with the declared Gaussian prior."""
        residual = obs.counts.flatten(1).double() - obs.shots * self.nominal.flatten(1)
        score = (residual / self.nominal.flatten(1)) @ self.derivatives[0]
        precision = obs.shots * self.linear_fisher + self.prior_precision
        estimate = torch.linalg.solve(precision, score.unsqueeze(-1)).reshape(
            -1, self.n, 2
        )
        return centered(estimate)

    def acquire(self, coords, command, shots, seed, index):
        p = self.law(coords)
        counts = sample_counts(
            p,
            int(shots),
            np.random.default_rng(np.random.SeedSequence([seed, index, 81])),
        )
        return Observation(
            torch.from_numpy(counts), int(shots), self.analyzers, command.clone(), index
        )


@dataclass(frozen=True)
class Prefix:
    observations: tuple[Observation, ...]
    actions: tuple[torch.Tensor, ...]


def rollout(
    env: PreparationEnvironment,
    policy: Callable | None,
    *,
    episodes,
    seed,
    schedule,
    collect_training=False,
    privileged_teacher=False,
    secondary_one_step=False,
    initial_scale=1.0,
    constant_velocity=False,
    action_scale=1.0,
    record_proposals=False,
):
    """Policy receives only acquired counts/settings and past executable actions."""
    opportunities = 1 if secondary_one_step else DECISIONS
    if len(schedule) != opportunities + 1:
        raise ValueError("requires ten decisions and eleven acquisitions")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 17]))
    bias = centered(
        torch.from_numpy(rng.normal(size=(episodes, env.n, 2)))
        * torch.tensor(PRIOR_SD, dtype=torch.float64)
    )
    if not np.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("initial spread multiplier must be finite and positive")
    if not np.isfinite(action_scale) or not 0 < action_scale <= 1:
        raise ValueError("action multiplier must lie in (0,1]")
    if initial_scale != 1:
        bias = bias * initial_scale
    if constant_velocity:
        velocity = centered(torch.from_numpy(rng.normal(size=(episodes, env.n, 2)))
                            * torch.tensor(VELOCITY_SD, dtype=torch.float64))
    command = torch.zeros_like(bias)
    scores, observations, actions, training, proposals = [], [], [], [], []
    for t, per_operation in enumerate(schedule):
        coords = (bias + t * velocity if constant_velocity else bias) + command
        scores.append(fidelity(coords))
        observations.append(env.acquire(coords, command, per_operation, seed, t))
        if t == opportunities:
            break
        prefix = Prefix(tuple(observations), tuple(actions))
        # The oracle is computed only for labelled TRAIN/DEV or an explicit reference.
        if collect_training or privileged_teacher:
            # Local branch of angles, appropriate for the registered narrow Gaussian bias.
            phase = torch.atan2(torch.sin(coords[..., 1]), torch.cos(coords[..., 1]))
            target = feasible(-0.45 * torch.stack([coords[..., 0], phase], dim=-1))
        if collect_training:
            training.append((prefix, target))
        chosen = target if privileged_teacher else policy(prefix)
        proposal = chosen.double() * action_scale
        if record_proposals:
            proposals.append(proposal)
        action = feasible(proposal, env.projector)
        actions.append(action)
        command = command + action
    return {
        "scores": torch.stack(scores, dim=1),
        "actions": torch.stack(actions, dim=1),
        "observations": observations,
        "training": training,
        **({"proposals": torch.stack(proposals, dim=1)} if record_proposals else {}),
    }


def classical_policy(env, kind, gain=0.45):
    def decide(prefix):
        error = -env.estimate(prefix.observations[-1])
        if kind == "noop":
            return torch.zeros_like(error)
        if kind == "analytic":
            return gain * error
        if kind == "pid":
            old = [-env.estimate(obs) for obs in prefix.observations[:-1]]
            # Clamp the integral before multiplying by its gain (anti-windup).
            integral = torch.stack([*old, error]).sum(0).clamp(-0.5, 0.5)
            derivative = error - old[-1] if old else torch.zeros_like(error)
            return gain * error + 0.08 * integral + 0.025 * derivative
        raise ValueError("unknown public comparator")

    return decide
