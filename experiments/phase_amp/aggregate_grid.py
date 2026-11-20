"""Aggregate registered static amplitude/phase operating points from returned trajectory arrays."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .coordinates import error_sizes, fidelity, grid_coords

HERE = Path(__file__).resolve().parent
FAMILY = 400
DRAWS = 262_144
TAIL = 0.05 / (2 * FAMILY)
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
TOL = 1e-10
ACTIVE = 1e-8
POLICIES = ("noop", "analytic", "learned_one", "learned_k")


def exact(harm, total, tail=0.025):
    if not total:
        return [None, None]
    lo = 0.0 if not harm else float(beta.ppf(tail, harm, total - harm + 1))
    hi = 1.0 if harm == total else float(beta.ppf(1 - tail, harm + 1, total - harm))
    return [lo, hi]


def array_csv(path, rows, keys=()):
    names = list(dict.fromkeys([*keys, *(k for r in rows for k in r)]))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def cell_result(path, reg, meta, refs):
    result = json.loads((path / "result.json").read_bytes())
    n = meta["n"]
    size = reg["episodes_per_cell"]
    seeds = reg["seeds"]
    required = dict(
        study=reg["study"],
        source=reg["source"],
        n=n,
        condition=meta["condition"],
        amplitude_index=meta["a_index"],
        phase_index=meta["p_index"],
        amplitude_rms=meta["initial_amplitude_rms"],
        phase_rms_rad=meta["initial_phase_rms_rad"],
        k=reg["k_by_n"][str(n)],
        reads=reg["reads"],
        actual_per_policy_reads_per_episode=reg["reads"],
        episodes=size,
        seeds=seeds,
        role="independent_range",
    )
    if any(result[k] != v for k, v in required.items()):
        raise ValueError(f"{path}: registration or heldout result differs")
    diagnostics = []
    quantiles = []
    end = {}
    replay = 0.0
    with np.load(path / "episode_panels.npz", allow_pickle=False) as p:
        ids = p["episode_ids"]
        x = p["initial_coords"]
        a0, ph0 = error_sizes(x)
        if (
            x.shape != (size, n, 2)
            or ids.shape != (size,)
            or len(set(ids.tolist())) != size
            or not np.array_equal(p["seeds"], seeds)
            or int(p["paid_reads"]) != reg["reads"]
            or int(p["frozen_k"]) != required["k"]
        ):
            raise ValueError(f"{path}: wrong episode, seed, or exposure binding")
        if not np.allclose(
            a0, required["amplitude_rms"], rtol=1e-10, atol=1e-12
        ) or not np.allclose(ph0, required["phase_rms_rad"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{path}: realized coordinates differ")
        first = str(ids[0]).split("_")
        known = grid_coords(
            n, size, int(first[1]), required["amplitude_rms"], required["phase_rms_rad"]
        )
        if first[0] != f"n{n}" or not np.allclose(x, known, rtol=1e-11, atol=1e-12):
            raise ValueError(f"{path}: episode IDs do not reproduce paired directions")
        if n not in refs:
            refs[n] = ids.copy()
        if not np.array_equal(ids, refs[n]):
            raise ValueError(
                f"{path}: paired directions do not have the same episode IDs"
            )
        null = p["noop_fidelity"]
        active_k = None
        harm_k = None
        for arm in POLICIES:
            learned = arm.startswith("learned")
            saved = p[arm + "_fidelity"] if learned else p[arm + "_fidelity"][None]
            actions = (
                (p[arm + "_actions"] if learned else p[arm + "_actions"][None])
                if arm != "noop"
                else np.zeros((1, size, 10, n, 2))
            )
            count = len(seeds) if learned else 1
            schedule = np.asarray(p[arm + "_read_schedule"]).reshape(count, 11)
            if not (
                saved.shape == (count, size, 11)
                and actions.shape == (count, size, 10, n, 2)
                and np.isfinite(actions).all()
                and np.isfinite(saved).all()
            ):
                raise ValueError(f"{path}: wrong policy trajectory shape")
            if not np.all(schedule.sum(axis=1) * (3 * n - 2) == reg["reads"]):
                raise ValueError(f"{path}: unequal actual per-policy READs")
            path_coords = np.empty((count, size, 11, n, 2))
            path_coords[:, :, 0] = x
            path_coords[:, :, 1:] = x[None, :, None] + np.cumsum(actions, axis=2)
            err = float(np.abs(fidelity(path_coords) - saved).max())
            replay = max(replay, err)
            if err > 2e-10:
                raise ValueError(
                    f"{path}: reconstructed {arm} overlaps differ by {err}"
                )
            end[arm] = saved[:, :, -1]
            gain = end[arm] - null[None, :, -1]
            harmed = gain < -TOL
            act = np.max(abs(actions), axis=(-1, -2)) > ACTIVE
            acted = act.any(axis=-1)
            mistake = act & (np.diff(saved, axis=-1) < -TOL)
            terminal = path_coords[:, :, -1]
            amp, phi = error_sizes(terminal)
            seed_names = seeds if learned else ["shared"]
            mag = np.exp(
                terminal[..., 0] - terminal[..., 0].max(axis=-1, keepdims=True)
            )
            relative_mode = 100 * (
                np.sqrt(n) * mag / np.linalg.norm(mag, axis=-1, keepdims=True) - 1
            )
            ii, jj = np.triu_indices(n, 1)
            angles = terminal[..., 1]
            dp = angles[..., ii] - angles[..., jj]
            pair_angles = np.arctan2(np.sin(dp), np.cos(dp))
            for j, label in enumerate(seed_names):
                den = int(acted[j].sum())
                h = int(harmed[j].sum())
                he = int((harmed[j] & acted[j]).sum())
                slots = int(act[j].sum())
                lo, hi = exact(h, size)
                clo, chi = exact(he, den)
                diagnostics.append(
                    dict(
                        **meta,
                        policy=arm,
                        seed=label,
                        episodes=size,
                        executed_episodes=den,
                        execution_coverage=den / size,
                        harmed_episodes=h,
                        harm_rate=h / size,
                        harm_pointwise_lower=lo,
                        harm_pointwise_upper=hi,
                        harmed_executed_episodes=he,
                        harm_given_execution=he / den if den else None,
                        conditional_harm_pointwise_lower=clo,
                        conditional_harm_pointwise_upper=chi,
                        gain_given_execution=float(gain[j, acted[j]].mean())
                        if den
                        else None,
                        active_slots=slots,
                        total_slots=10 * size,
                        mistaken_active_slots=int(mistake[j].sum()),
                        mistake_given_active=float(mistake[j].sum() / slots)
                        if slots
                        else None,
                    )
                )
            strata = list(enumerate(seed_names))
            if learned:
                strata.append((None, "fixed_seed_mixture"))
            for idx, label in strata:
                a = amp.ravel() if idx is None else amp[idx]
                ph = phi.ravel() if idx is None else phi[idx]
                rates = (
                    relative_mode.ravel() if idx is None else relative_mode[idx].ravel()
                )
                angles_flat = (
                    pair_angles.ravel() if idx is None else pair_angles[idx].ravel()
                )
                for q, v, w, rate, ang, abs_ang in zip(
                    QUANTILES,
                    np.quantile(a, QUANTILES),
                    np.quantile(ph, QUANTILES),
                    np.quantile(rates, QUANTILES),
                    np.quantile(angles_flat, QUANTILES),
                    np.quantile(abs(angles_flat), QUANTILES),
                ):
                    quantiles.append(
                        dict(
                            **meta,
                            policy=arm,
                            seed=label,
                            quantile=q,
                            independent_episodes=size,
                            fixed_policy_count=count if idx is None else 1,
                            terminal_amplitude_rms=float(v),
                            terminal_phase_rms_rad=float(w),
                            terminal_phase_rms_deg=float(np.degrees(w)),
                            terminal_raw_pair_phase_rms_rad=float(
                                w * np.sqrt(2 * n / (n - 1))
                            ),
                            terminal_raw_pair_phase_rms_deg=float(
                                np.degrees(w) * np.sqrt(2 * n / (n - 1))
                            ),
                            terminal_mode_magnitude_deviation_from_ideal_percent=float(
                                rate
                            ),
                            terminal_signed_pair_difference_deg=float(np.degrees(ang)),
                            terminal_abs_pair_difference_deg=float(np.degrees(abs_ang)),
                        )
                    )
            if arm == "learned_k":
                active_k = acted
                harm_k = harmed
    harm_counts = harm_k.sum(axis=1)
    hc = np.asarray([exact(int(h), size, TAIL / len(seeds)) for h in harm_counts])
    lower, upper = hc.mean(axis=0)
    current = dict(
        **meta,
        status="scored",
        k=required["k"],
        dev_selected_k=reg["frozen_control"][f"{n}_{meta['condition']}"][
            "dev_selected_k"
        ],
        episodes=size,
        fixed_seed_count=len(seeds),
        reads=reg["reads"],
        harm=float(harm_k.mean()),
        harm_lower=float(lower),
        harm_upper=float(upper),
        zero_harm_upper_per_seed=float(exact(0, size, TAIL / len(seeds))[1]),
        episode_execution=float(active_k.mean()),
        executed_seed_episodes=int(active_k.sum()),
        harmed_executed_seed_episodes=int((active_k & harm_k).sum()),
        harm_given_execution=float((active_k & harm_k).sum() / active_k.sum())
        if active_k.any()
        else None,
        max_fidelity_replay_error=replay,
        initial_fidelity_mean=float(null[:, 0].mean()),
        terminal_fidelity_mean=float(end["learned_k"].mean()),
        q50_terminal_amplitude=float(np.median(amp)),
        q50_terminal_phase_deg=float(np.degrees(np.median(phi))),
    )
    gains = np.stack(
        [
            (end["learned_k"] - end[arm]).mean(axis=0)
            for arm in ("learned_one", "noop", "analytic")
        ],
        axis=-1,
    )
    return current, gains, diagnostics, quantiles


def paired_intervals(vectors, n, draws):
    """Each resample uses common episode weights over all tested points and fixed seeds at N."""
    data = np.concatenate(vectors, axis=1)
    episodes = data.shape[0]
    means = data.mean(axis=0)
    rng = np.random.default_rng([20261120471, n])
    sample = np.empty((draws, data.shape[1]))
    p = np.full(episodes, 1 / episodes)
    block = 2048
    for i in range(0, draws, block):
        length = min(block, draws - i)
        weights = rng.multinomial(episodes, p, size=length)
        sample[i : i + length] = weights.astype(np.float64) @ data / episodes
    return means.reshape(-1, 3), np.quantile(sample, [TAIL, 1 - TAIL], axis=0).reshape(
        2, -1, 3
    )


def classify(r):
    if r["a_index"] == 0 and r["p_index"] == 0:
        return (
            "hold_preserved"
            if r["harm_upper"] < 0.10 and r["k_minus_noop_lower"] > -0.005
            else "hold_failed"
            if r["harm_lower"] > 0.10 or r["k_minus_noop_upper"] < -0.005
            else "hold_unresolved"
        )
    if (
        r["k_minus_one_lower"] > 0.01
        and r["k_minus_noop_lower"] > 0
        and r["harm_upper"] < 0.10
    ):
        return "usable"
    if r["k_minus_noop_upper"] < 0 or r["harm_lower"] > 0.10:
        return "harmful"
    return "unresolved"


def aggregate(
    folder, draws=DRAWS, expected_study="phase_amp_control_range_v1_20261120"
):
    regfile = folder / "registration.json"
    reg = json.loads(regfile.read_bytes())
    refs = {}
    rows = []
    diagnostics = []
    quantiles = []
    if (
        reg["study"] != expected_study
        or reg["episodes_per_cell"] != 256
        or len(reg["amplitudes"])
        * len(reg["phases_rad"])
        * len(reg["calibration"])
        * len(reg["n"])
        != 100
    ):
        raise ValueError("Wrong registered grid")
    for n in reg["n"]:
        evaluated = []
        v = []
        for cond in reg["calibration"]:
            for ai, a in enumerate(reg["amplitudes"]):
                for pi, p in enumerate(reg["phases_rad"]):
                    meta = dict(
                        n=n,
                        condition=cond,
                        a_index=ai,
                        p_index=pi,
                        initial_amplitude_rms=a,
                        initial_phase_rms_rad=p,
                        initial_phase_rms_deg=float(np.degrees(p)),
                        initial_raw_pair_phase_rms_rad=float(
                            p * np.sqrt(2 * n / (n - 1))
                        ),
                        initial_raw_pair_phase_rms_deg=float(
                            np.degrees(p) * np.sqrt(2 * n / (n - 1))
                        ),
                    )
                    path = folder / f"n{n}" / cond / f"a{ai}_p{pi}"
                    if (
                        not (path / "result.json").is_file()
                        or not (path / "episode_panels.npz").is_file()
                    ):
                        rows.append(dict(**meta, status="pending", decision="pending"))
                        continue
                    cell, gain, action, terminal = cell_result(path, reg, meta, refs)
                    rows.append(cell)
                    evaluated.append(cell)
                    v.append(gain)
                    diagnostics.extend(action)
                    quantiles.extend(terminal)
        if v:
            means, limits = paired_intervals(v, n, draws)
            for i, row in enumerate(evaluated):
                for j, key in enumerate(
                    ("k_minus_one", "k_minus_noop", "k_minus_analytic")
                ):
                    row[key] = float(means[i, j])
                    row[key + "_lower"] = float(limits[0, i, j])
                    row[key + "_upper"] = float(limits[1, i, j])
                row["decision"] = classify(row)
    if set(refs) == {3, 4} and set(refs[3].tolist()) & set(refs[4].tolist()):
        raise ValueError("3/4-mode independent cohorts overlap")
    confirmed = sum(r["status"] == "scored" for r in rows)
    out = dict(
        study=reg["study"],
        registration_sha256=hashlib.sha256(regfile.read_bytes()).hexdigest(),
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        returned_cells=confirmed,
        total_cells=100,
        independent_directions_by_n={str(n): len(x) for n, x in refs.items()},
        fixed_seeds=reg["seeds"],
        simultaneous_family=FAMILY,
        bootstrap_draws=draws,
        per_tail_alpha=TAIL,
        uncertainty="Gains: Bonferroni percentile episode bootstrap, fixed seeds paired and averaged within episode. Harm: average per-seed Clopper–Pearson bounds with additional three-seed Bonferroni adjustment. All 400 endpoints retained in partial snapshots.",
        axes_and_actions="Grid amplitude and phase are across-mode RMS quantities. Per-mode action box bounds 0.35 log amplitude and 0.70 radians bound individual components per action, not the RMS or one-step attainability.",
        residual_range="RMS endpoint quantiles are over episodes (and the equally weighted fixed policy-seed mixture). Raw pairwise RMS undoes the N-dependent scale. Component quantiles marginalize modes or unordered pairs within episodes; magnitude is 100 times normalized mode amplitude divided by 1/sqrt(N) minus 100. Pair differences are wrapped. Fixed seeds/modes do not add independent episodes; no inter-bin interpolation.",
        decision_counts={
            key: sum(r["decision"] == key for r in rows)
            for key in sorted({r["decision"] for r in rows})
        },
        cells=rows,
    )
    return out, diagnostics, quantiles


def figures(folder, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    colors = {
        "usable": "#00785C",
        "harmful": "#B63E26",
        "unresolved": "#D7A424",
        "hold_preserved": "#3164A2",
        "hold_failed": "#792A70",
        "hold_unresolved": "#8F77AD",
        "pending": "#D8DFE2",
    }
    cells = summary["cells"]
    reg = json.loads((folder / "registration.json").read_bytes())
    cols = (
        ("decision", "Decision"),
        ("episode_execution", "Trajectories with an action"),
        ("q50_terminal_amplitude", "Final log-amplitude RMS, median"),
        ("q50_terminal_phase_deg", "Final scaled phase RMS (deg), median"),
    )
    fig, axes = plt.subplots(
        4, 4, figsize=(12.3, 10.7), layout="constrained", sharex=True, sharey=True
    )
    for i, (n, cond) in enumerate((n, c) for n in reg["n"] for c in reg["calibration"]):
        group = [r for r in cells if r["n"] == n and r["condition"] == cond]
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 4.5)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xticks(range(5), [f"{np.degrees(v):.0f}" for v in reg["phases_rad"]])
            ax.set_yticks(range(5), [str(v) for v in reg["amplitudes"]])
            if j == 0:
                ax.set_ylabel(f"N={n}, {cond}\nInitial log-amplitude RMS", fontsize=9)
            if i == 3:
                ax.set_xlabel("Initial scaled phase RMS (deg)", fontsize=9)
            measured = [r for r in group if r["status"] == "scored"]
            x = [r["p_index"] for r in measured]
            y = [r["a_index"] for r in measured]
            if key == "decision":
                ax.scatter(
                    [r["p_index"] for r in group],
                    [r["a_index"] for r in group],
                    c=[colors[r["decision"]] for r in group],
                    s=180,
                    marker="s",
                )
            elif measured:
                values = [r[key] for r in measured]
                mx = (
                    1.0
                    if key == "episode_execution"
                    else max(
                        [r[key] for r in cells if r["status"] == "scored"] + [1e-12]
                    )
                )
                norm = Normalize(0, mx)
                sc = ax.scatter(
                    x, y, c=values, cmap="viridis", norm=norm, s=180, marker="s"
                )
                if i == 3:
                    fig.colorbar(
                        sc, ax=axes[:, j], location="bottom", shrink=0.73, aspect=26
                    )
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor=c,
            markeredgecolor=c,
            label=k.replace("_", " "),
        )
        for k, c in colors.items()
        if any(r["decision"] == k for r in cells)
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=9
    )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(folder / f"operating_range.{ext}", dpi=165, bbox_inches="tight")
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    args = cli.parse_args()
    summary, action, quantiles = aggregate(args.root)
    tmp = args.root / "grid_result.tmp"
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.root / "grid_result.json")
    array_csv(args.root / "grid_cells.csv", summary["cells"])
    array_csv(
        args.root / "action_conditions.csv",
        action,
        ("n", "condition", "policy", "seed", "episodes", "executed_episodes"),
    )
    array_csv(
        args.root / "terminal_individual_quantiles.csv",
        quantiles,
        (
            "n",
            "condition",
            "policy",
            "seed",
            "quantile",
            "terminal_amplitude_rms",
            "terminal_phase_rms_rad",
            "terminal_phase_rms_deg",
        ),
    )
    figures(args.root, summary)
    print(summary["returned_cells"], summary["decision_counts"])


if __name__ == "__main__":
    main()
