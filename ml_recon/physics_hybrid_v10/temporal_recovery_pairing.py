"""Bind temporal paired comparisons to one independent episode axis shared across fitted seeds."""

from __future__ import annotations

import json
import numpy as np
import torch

from .temporal_recovery_environment import PRIOR_SD, centered, fidelity
from .temporal_recovery_evaluation import CONFIRM_SEED, TRAIN_SEEDS, paired_interval


def confirmation_axis(n, *, long_run, prefix=7_000_000_000_000):
    if n not in (3, 4):
        raise ValueError("unregistered temporal confirmation system")
    if long_run:
        sources = [(prefix + 1000 * n + i, 64) for i in range(0, 512, 64)]
    else:
        sources = [(CONFIRM_SEED + n, 128)]
    ids, initial = [], []
    for seed, count in sources:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 17]))
        x = centered(
            torch.from_numpy(rng.normal(size=(count, n, 2)))
            * torch.tensor(PRIOR_SD, dtype=torch.float64)
        )
        initial.extend(fidelity(x).tolist())
        ids.extend(f"n{n}_state_{seed}_{i}" for i in range(count))
    return np.asarray(ids), np.asarray(initial)


def check_endpoint_panel(panel, ids, initial):
    if len(ids) != len(set(ids.tolist())) or panel.shape != (len(ids), 11):
        raise ValueError(
            "confirmation requires unique episodes and eleven physical endpoints"
        )
    if not np.allclose(panel[:, 0], initial, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "confirmation panel ordering or initial states do not match independent IDs"
        )
    if not np.isfinite(panel).all():
        raise ValueError("confirmation endpoint contains a nonfinite value")


def paired_seed_panel(seed_rows, episode_ids, family_size):
    if not seed_rows or len(set(episode_ids.tolist())) != len(episode_ids):
        raise ValueError("confirmation episode IDs must be unique")
    if any(np.asarray(row).shape != (len(episode_ids),) for row in seed_rows.values()):
        raise ValueError("each fitted seed must cover the full paired episode axis")
    pooled = paired_interval(np.mean(list(seed_rows.values()), axis=0), family_size)
    return pooled, {
        str(seed): paired_interval(row, family_size) for seed, row in seed_rows.items()
    }


SHORT_ARMS = (
    "zero",
    "current",
    "summary",
    "ordered",
    "pooled",
    "shuffled",
    "learner",
    "ordered_one_common",
    "ordered_one_optimized",
    "learner_one_common",
    "learner_one_optimized",
)
LONG_ARMS = (
    *SHORT_ARMS,
    "noop",
    "analytic",
    "pid",
    "analytic_one_common",
    "analytic_one_optimized",
)


def registered_reports(results, artifacts, *, seeds, long_run):
    """Validate the entire registered grid before computing any confidence interval."""
    registered = tuple(TRAIN_SEEDS)
    if len(seeds) != len(registered) or set(seeds) != set(registered):
        raise ValueError("requires the complete three distinct registered model seeds")
    prefix = "long_seed" if long_run else "learned_seed"
    expected_files = {f"{prefix}_{s}.json" for s in registered}
    discovered = {path.name for path in results.glob(f"{prefix}_*.json")}
    if discovered != expected_files:
        raise ValueError(
            f"expected exactly the registered seed reports: missing={expected_files - discovered}, unexpected={discovered - expected_files}"
        )
    arms = LONG_ARMS if long_run else SHORT_ARMS
    reports = {}
    for seed in registered:
        with (results / f"{prefix}_{seed}.json").open(encoding="utf-8") as stream:
            report = json.load(stream)
        reported = report.get("training_seed")
        bound = report.get("binding", {}).get("seed")
        if (
            type(reported) is not int
            or reported != seed
            or type(bound) is not int
            or bound != seed
        ):
            raise ValueError(
                f"training_seed or binding.seed differs from registered filename seed {seed}"
            )
        systems = report.get("systems" if long_run else "arms", {})
        if set(systems) != {"3", "4"}:
            raise ValueError(
                f"incomplete or unexpected N panel for registered seed {seed}"
            )
        for n in (3, 4):
            reported_arms = (
                systems[str(n)].get("arms", {}) if long_run else systems[str(n)]
            )
            if set(reported_arms) != set(arms):
                raise ValueError(
                    f"seed {seed}, N={n}: missing arms {set(arms) - set(reported_arms)}; unexpected arms {set(reported_arms) - set(arms)}"
                )
            stem = ("long_" if long_run else "") + f"n{n}_"
            owned = artifacts / f"seed_{seed}"
            expected = {f"{stem}{arm}.npz" for arm in arms}
            found = {f.name for f in owned.glob(stem + "*.npz")}
            if found != expected:
                raise ValueError(
                    f"seed {seed}, N={n}: episode archive set differs; missing={expected - found}, unexpected={found - expected}"
                )
        reports[seed] = report
    return reports


def checked_archive(archive, arm_report, *, n, ids, initial):
    scores = archive["fidelity"]
    check_endpoint_panel(scores, ids, initial)
    count = len(ids)
    actions = archive["executed_actions"]
    native = archive["native_counts"]
    opportunities = archive["reads_per_setting"]
    settings = 3 * n - 2
    if actions.shape != (count, 10, n, 2) or not np.isfinite(actions).all():
        raise ValueError("invalid ten-action trajectory")
    if (
        native.shape != (count, 11, settings, 4)
        or not np.issubdtype(native.dtype, np.integer)
        or (native < 0).any()
        or opportunities.shape != (11,)
        or (opportunities <= 0).any()
        or not np.all(native.sum(-1) == opportunities[None, :, None])
        or opportunities.sum() * settings != 77000
        or opportunities[-1] * settings != 7000
    ):
        raise ValueError(
            "episode archive does not preserve its eleven native acquisitions and paid exposure"
        )
    curve = (scores - scores[:, :1]).mean(0)
    if (
        arm_report.get("episodes") != count
        or arm_report.get("acquisitions") != 11
        or arm_report.get("decisions") != 10
        or arm_report.get("paid_reads") != 77000
        or not np.allclose(
            arm_report.get("round_mean_gain", []), curve, rtol=1e-10, atol=1e-12
        )
        or not np.isclose(
            arm_report.get("terminal_gain", np.nan), curve[-1], rtol=1e-10, atol=1e-12
        )
    ):
        raise ValueError(
            "reported seed/arm summary and its per-episode evidence disagree"
        )
    return scores, np.abs(actions).max(axis=(2, 3)) > 1e-8
