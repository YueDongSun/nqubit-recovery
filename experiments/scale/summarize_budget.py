"""Aggregate each budget condition on independent parent states and plot the mixed-error scale curve."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

POLICIES = (
    "noop",
    "learned_plugin",
    "classical_fisher",
    "oracle_bounded",
    "learned_fd",
)
MODES = ("fixed", "per_setting", "mean_fisher")
LABEL = {
    "fixed": "Fixed total READ",
    "per_setting": "Equal READ per setting",
    "mean_fisher": "Equal mean Fisher information",
}
COLOR = dict(fixed="#283A75", per_setting="#138574", mean_fisher="#CB602F")


def read_vectors(path, name):
    with gzip.open(path, "rt") as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if r["policy"] == name),
            key=lambda r: int(r["parent_id"]),
        )
    if len(rows) != 128 or [int(r["parent_id"]) for r in rows] != list(range(128)):
        raise ValueError(f"unexpected parent set at {path}")
    fields = (
        "F0",
        "FT",
        "delta_F",
        "amp_rmsT",
        "phase_rmsT",
        "harm_vs_noop",
        "executed",
    )
    return {k: np.array([float(r[k]) for r in rows]) for k in fields}


def summarize():
    from scipy.stats import beta

    protocol = json.loads((OUT / "protocol.json").read_text())
    sizes = protocol["shared_true_sizes"]
    rows = []
    # Resample the parent population, not separately repeated observations from the same parent.
    rng = np.random.default_rng(20261120890127)
    weights = rng.multinomial(128, np.full(128, 1 / 128), size=10000) / 128

    def interval(v):
        samples = weights @ v
        return [float(x) for x in np.quantile(samples, [0.025, 0.975])]

    for n in sizes:
        for s, a, p in (
            ("a2_p0", 0.35, 0.0),
            ("a0_p2", 0.0, 0.7),
            ("a2_p2", 0.35, 0.7),
        ):
            base_name = f"n{n}_{s}_k1"
            for policy in POLICIES:
                fixed = read_vectors(ROOT / "rows" / (base_name + ".csv.gz"), policy)
                for mode in MODES:
                    design = protocol["budgets"][str(n)][mode]
                    values = (
                        fixed
                        if mode == "fixed"
                        else read_vectors(
                            OUT / "rows" / (base_name + "_" + mode + ".csv.gz"), policy
                        )
                    )
                    np.testing.assert_array_equal(values["F0"], fixed["F0"])
                    if n == 3:
                        np.testing.assert_allclose(
                            values["delta_F"], fixed["delta_F"], atol=2e-6, rtol=0
                        )
                    gain = values["delta_F"]
                    effect = gain - fixed["delta_F"]
                    harms = int((gain < -1e-12).sum())
                    hlow = (
                        0.0
                        if harms == 0
                        else float(beta.ppf(0.025, harms, 129 - harms))
                    )
                    hupp = (
                        1.0
                        if harms == 128
                        else float(beta.ppf(0.975, harms + 1, 128 - harms))
                    )
                    gl, gh = interval(gain)
                    el, eh = interval(effect)
                    rows.append(
                        dict(
                            n=n,
                            A=a,
                            P=p,
                            policy=policy,
                            budget_mode=mode,
                            parents=len(gain),
                            decision_reads=design["total_read_decision"],
                            terminal_reads=design["terminal"],
                            total_reads=design["total"],
                            mean_relative_fisher=design[
                                "joint_mean_per_relative_coordinate"
                            ],
                            mean_relative_amplitude_fisher=design[
                                "amplitude_mean_per_relative_coordinate"
                            ],
                            mean_relative_phase_fisher=design[
                                "phase_mean_per_relative_coordinate"
                            ],
                            mean_gain=float(gain.mean()),
                            gain_lower=gl,
                            gain_upper=gh,
                            gain_minus_fixed=float(effect.mean()),
                            comparison_lower=el,
                            comparison_upper=eh,
                            harm=harms / 128,
                            harm_lower=hlow,
                            harm_upper=hupp,
                            practical_harm_001=float((gain < -0.01).mean()),
                            terminal_amp_median=float(np.median(values["amp_rmsT"])),
                            terminal_phase_median_degrees=float(
                                np.degrees(np.median(values["phase_rmsT"]))
                            ),
                            terminal_phase_q10_degrees=float(
                                np.degrees(np.quantile(values["phase_rmsT"], 0.10))
                            ),
                            terminal_phase_q90_degrees=float(
                                np.degrees(np.quantile(values["phase_rmsT"], 0.90))
                            ),
                        )
                    )
    with (OUT / "budget_comparison.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return protocol, rows


def plot(reg, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    ns = reg["shared_true_sizes"]
    xx = np.arange(len(ns))
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.15), layout="constrained")
    for mode in MODES:
        line = dict(color=COLOR[mode], marker="o", lw=1.9, ms=4)
        total = [reg["budgets"][str(n)][mode]["total"] for n in ns]
        axs[0, 0].plot(xx, total, label=LABEL[mode], **line)
        for idx, key in enumerate(["learned_plugin", "classical_fisher"]):
            v = sorted(
                (
                    r
                    for r in rows
                    if r["A"] == 0.35
                    and r["P"] == 0.7
                    and r["policy"] == key
                    and r["budget_mode"] == mode
                ),
                key=lambda r: r["n"],
            )
            ax = axs[0, 1] if idx == 0 else axs[1, 0]
            y = np.array([r["mean_gain"] for r in v])
            ax.plot(xx, y, **line)
            ax.fill_between(
                xx,
                [r["gain_lower"] for r in v],
                [r["gain_upper"] for r in v],
                color=COLOR[mode],
                alpha=0.12,
                lw=0,
            )
            if idx == 0:
                h = np.array([r["harm"] for r in v])
                axh = axs[1, 1]
                axh.plot(xx, h, **line)
                axh.fill_between(
                    xx,
                    [r["harm_lower"] for r in v],
                    [r["harm_upper"] for r in v],
                    color=COLOR[mode],
                    alpha=0.09,
                    lw=0,
                )
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel("Total READ per preparation")
    axs[0, 0].legend(fontsize=8, loc="upper left", frameon=False)
    axs[0, 1].set_ylabel("Mean change in W overlap")
    axs[1, 0].set_ylabel("Mean change in W overlap")
    for ax in (axs[0, 1], axs[1, 0]):
        ax.axhline(0, color="0.5", lw=0.7)
    axs[1, 1].set_ylabel("Trajectories with any endpoint loss")
    axs[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax, title in zip(
        axs.ravel(),
        [
            "a  Acquisition cost",
            "b  Same frozen shared estimator",
            "c  Classical Fisher correction",
            "d  Frozen estimator harm rate",
        ],
    ):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(xx, ns)
        ax.set_xlabel("Mode count N")
        ax.grid(axis="y", alpha=0.17)
        ax.axvline(2.5, color="0.5", ls=":", lw=0.9)
    fig.suptitle(
        "Readout budget and one-step recovery across mode count",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"budget_scale.{suffix}", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=Path)
    args = parser.parse_args()
    ROOT = args.primary
    OUT = args.budget
    protocol, table = summarize()
    plot(protocol, table)
