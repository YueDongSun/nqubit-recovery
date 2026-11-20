"""Render the fixed-77,000 READ shared-checkpoint scaling curves and measured recovery map."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


SETS = ((0.35, 0.0), (0.0, 0.7), (0.35, 0.7))
LABELS = ("Amplitude only", "Phase only", "Amplitude and phase")
COLORS = ("#116D74", "#B24D25", "#394E87")


def read_table(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def export(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            ROOT / f"{name}.{ext}", dpi=195, bbox_inches="tight", pad_inches=0.12
        )
    plt.close(fig)


def main():
    records = read_table("operating_cells.csv")
    quant = read_table("terminal_uncertainty.csv")
    rows = {
        (
            int(r["n"]),
            int(r["k"]),
            float(r["amp_log_rms"]),
            float(r["phase_rad_rms"]),
            r["policy"],
        ): r
        for r in records
    }
    qs = {
        (
            int(r["n"]),
            int(r["k"]),
            float(r["amp_log_rms"]),
            float(r["phase_rad_rms"]),
            r["policy"],
            r["terminal_variable"],
        ): r
        for r in quant
    }

    def column(k, a, p, pol, var):
        return np.array([float(rows[n, k, a, p, pol][var]) for n in NS])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout="constrained")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS, labels=[str(n) for n in NS])
        ax.minorticks_off()
        ax.axvspan(2.8, 10.9, color="#EAF0EE", alpha=0.75, zorder=0)
        ax.axvline(np.sqrt(8 * 16), ls=":", lw=0.9, color="#64746B")
        ax.grid(axis="y", alpha=0.19)
        ax.set_xlim(2.8, 275)
        ax.set_xlabel("Number of modes N")
    gain = axes[0, 0]
    harm = axes[0, 1]
    gain.axhline(0, color="0.4", lw=0.9)
    for (a, p), name, color in zip(SETS, LABELS, COLORS):
        y = column(1, a, p, "learned_plugin", "gain_noop") * 100
        lower = column(1, a, p, "learned_plugin", "gain_noop_lower") * 100
        upper = column(1, a, p, "learned_plugin", "gain_noop_upper") * 100
        gain.errorbar(
            NS,
            y,
            yerr=[y - lower, upper - y],
            color=color,
            fmt="o-",
            capsize=2,
            lw=1.5,
            ms=4,
            label=name,
        )
    gain.set_title("A   One action at matched initial severity")
    gain.set_ylabel("Gain over NOOP (fidelity points)")
    gain.legend(fontsize=9, loc="best", frameon=False)
    for k, col, marker in ((1, COLORS[2], "o"), (4, "#8A3981", "s")):
        y = column(k, 0.35, 0.7, "learned_plugin", "harm_rate") * 100
        harm.plot(
            NS, y, color=col, marker=marker, lw=1.5, ms=4, label=f"K={k}, any loss"
        )
        ys = column(k, 0.35, 0.7, "learned_plugin", "harm_over_one_point_rate") * 100
        harm.plot(
            NS,
            ys,
            color=col,
            ls="--",
            marker=marker,
            markerfacecolor="white",
            ms=4,
            label=f"K={k}, loss >1 point",
        )
        for ax, feature, initial in (
            (axes[1, 0], "amp_log_rms", 0.35),
            (axes[1, 1], "phase_deg_rms", np.degrees(0.7)),
        ):
            quantiles = np.array(
                [
                    [
                        float(qs[n, k, 0.35, 0.7, "learned_plugin", feature][q])
                        for n in NS
                    ]
                    for q in ("q10", "median", "q90")
                ]
            )
            ax.fill_between(NS, quantiles[0], quantiles[2], color=col, alpha=0.16)
            ax.plot(
                NS, quantiles[1], color=col, marker=marker, ms=4, lw=1.5, label=f"K={k}"
            )
            if k == 1:
                ax.axhline(initial, color="0.35", ls=":", lw=1.1, label="Initial")
    harm.axhline(10, color="0.4", lw=0.9, ls=":")
    harm.set_ylim(-4, 104)
    harm.set_title("B   Terminal harm with both errors present")
    harm.set_ylabel("Parents harmed (%)")
    harm.legend(fontsize=8, frameon=False, ncol=2)
    axes[1, 0].set_title("C   Remaining amplitude error")
    axes[1, 0].set_ylabel("Centered log-amplitude RMS")
    axes[1, 1].set_title("D   Remaining phase error")
    axes[1, 1].set_ylabel("Scaled wrapped-pair phase RMS (degrees)")
    for ax in axes[1]:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "One frozen estimator from 3 to 256 modes  |  77,000 READ per trajectory",
        fontsize=15,
    )
    fig.supxlabel(
        "Shaded N values were seen in training. A shows simultaneous mean intervals; C–D show medians and the 10th–90th percentiles across 128 simulated parents.",
        fontsize=9,
    )
    export(fig, "shared_recovery_scale")
    labels = [
        "usable",
        "unresolved",
        "harmful",
        "hold_preserved",
        "hold_unresolved",
        "hold_failed",
    ]
    cmap = ListedColormap(
        ["#2A8275", "#E4BC60", "#B8463F", "#509DB8", "#C3BDDC", "#86567D"]
    )
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(11, 7.8),
        gridspec_kw={"height_ratios": [10, 3]},
        layout="constrained",
    )
    for panel, k in zip(ax, (1, 4)):
        ordered = sorted(
            {
                (a, p)
                for n, k0, a, p, pol in rows
                if k0 == k and pol == "learned_plugin"
            },
            key=lambda pair: (
                0
                if pair == (0, 0)
                else 1
                if pair[1] == 0
                else 2
                if pair[0] == 0
                else 3,
                *pair,
            ),
        )
        data = np.array(
            [
                [labels.index(rows[n, k, a, p, "learned_plugin"]["status"]) for n in NS]
                for a, p in ordered
            ]
        )
        panel.imshow(
            data,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(labels) - 0.5,
            aspect="auto",
            interpolation="none",
        )
        panel.set_xticks(range(len(NS)), labels=[str(n) for n in NS])
        panel.set_yticks(
            range(len(ordered)),
            labels=[f"{a:.2f}  /  {np.degrees(p):.1f}°" for a, p in ordered],
        )
        panel.set_xticks(np.arange(-0.5, len(NS), 1), minor=True)
        panel.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
        panel.grid(which="minor", color="white", lw=1)
        panel.tick_params(which="minor", length=0)
        panel.axvline(2.5, lw=2, color="#303030", ls=":")
        panel.set_ylabel("Amplitude RMS / phase RMS")
        panel.set_title(
            f"K={k}  |  {'one bounded correction vs NOOP' if k == 1 else 'four bounded corrections vs one and NOOP'}"
        )
    ax[-1].set_xlabel("Number of modes N")
    fig.legend(
        handles=[
            Patch(facecolor=c, label=s.replace("_", " "))
            for c, s in zip(cmap.colors, labels)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Recovery classifications at the tested amplitude and phase points", fontsize=15
    )
    export(fig, "shared_recovery_range_map")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    ROOT = parser.parse_args().root
    REG = json.loads((ROOT / "protocol.json").read_text())
    NS = np.asarray(REG["sizes"])
    main()
