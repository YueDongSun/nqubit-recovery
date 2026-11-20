"""Compare the three frozen capacities using exactly the same counts and simulated parents."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

CAPS = ("C-S", "C-L", "C-XXL")


def write(name, rows):
    with (ROOT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary():
    from scipy.stats import beta, t

    parents = REG["parents"]
    tables = []
    keys = []
    diff = []
    ids = np.arange(parents)
    for cell in REG["cells"]:
        samples = []
        for cap in CAPS:
            f = ROOT / "rows" / (cell["tag"] + "_" + cap.lower() + ".csv.gz")
            if not f.is_file():
                raise FileNotFoundError(f)
            with gzip.open(f, "rt", newline="") as stream:
                rows = sorted(csv.DictReader(stream), key=lambda r: int(r["parent_id"]))
            if len(rows) != parents or not np.array_equal(
                ids, [int(r["parent_id"]) for r in rows]
            ):
                raise ValueError("independent parents differ")
            if any(
                (
                    r["capacity"] != cap
                    or r["policy"] != "learned_plugin"
                    or int(r["N"]) != cell["N"]
                    or int(r["K"]) != 1
                    or float(r["A"]) != cell["A"]
                    or float(r["P"]) != cell["P"]
                    or int(r["seed"]) != REG["seed"]
                    or int(r["reads_decision"]) + int(r["reads_terminal"]) != 77000
                    or sum(int(r["terminal_" + z]) for z in ("00", "10", "01", "11"))
                    != 7000
                )
                for r in rows
            ):
                raise ValueError("unmatched capacity or budget")
            field = lambda k: np.array([float(r[k]) for r in rows])
            f0, ft = field("F0"), field("FT")
            g = ft - f0
            np.testing.assert_allclose(g, field("delta_F"), atol=2e-12)
            if samples:
                np.testing.assert_allclose(f0, samples[0][0], atol=2e-12)
            samples.append((f0, g))
            h = int((g < -1e-12).sum())
            err = float(t.ppf(0.975, parents - 1) * g.std(ddof=1) / parents**0.5)
            row = dict(
                n=cell["N"],
                a=cell["A"],
                p_rad=cell["P"],
                capacity=cap,
                parameters=REG["models"][cap]["parameters"],
                parents=parents,
                paid_reads=77000,
                mean_gain=g.mean(),
                lower=g.mean() - err,
                upper=g.mean() + err,
                mean_terminal=ft.mean(),
                any_harm=h / parents,
                any_harm_lower=beta.ppf(0.025, h, parents - h + 1) if h else 0,
                any_harm_upper=beta.ppf(0.975, h + 1, parents - h)
                if h < parents
                else 1,
                material_harm=(g < -0.01).mean(),
                material_harmed=int((g < -0.01).sum()),
            )
            for label, raw in (
                ("log_amplitude", field("amp_rmsT")),
                ("phase_deg", np.degrees(field("phase_rmsT"))),
            ):
                for k, v in zip(
                    ("q10", "median", "q90"), np.quantile(raw, [0.1, 0.5, 0.9])
                ):
                    row[label + "_" + k] = v
            tables.append(row)
        for j in (1, 2):
            keys.append(
                dict(
                    n=cell["N"],
                    a=cell["A"],
                    p_rad=cell["P"],
                    larger=CAPS[j],
                    smaller=CAPS[j - 1],
                    parents=parents,
                )
            )
            diff.append(samples[j][1] - samples[j - 1][1])
    v = np.array(diff, dtype=np.float64).T
    mean = v.mean(0)
    se = v.std(0, ddof=1) / parents**0.5
    use = se > 1e-14
    weights = (
        np.random.default_rng(202611202053).multinomial(
            parents, np.full(parents, 1 / parents), size=20000
        )
        / parents
    )
    cutoff = (
        np.quantile(
            np.max(np.abs(weights @ (v[:, use] - mean[use]) / se[use]), axis=1), 0.95
        )
        if use.any()
        else 0
    )
    paired = [
        dict(
            **k,
            bigger_minus_smaller_gain=float(mu),
            lower=float(mu - cutoff * ss),
            upper=float(mu + cutoff * ss),
        )
        for k, mu, ss in zip(keys, mean, se)
    ]
    write("capacity_curves.csv", tables)
    write("capacity_paired.csv", paired)
    (ROOT / "summary.json").write_text(
        json.dumps(
            dict(
                cells=len(REG["cells"]),
                capacity_cells=len(tables),
                paired_comparisons=len(paired),
                seeds=[REG["seed"]],
                parents=parents,
                steps=REG["updates"],
                sizes=REG["sizes"],
                interval="95% simultaneous across 48 paired gain contrasts by parent bootstrap maximum standardized shift; model mean and harm curves have separate pointwise intervals",
                critical=float(cutoff),
            ),
            indent=2,
        )
        + "\n"
    )


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (ROOT / "capacity_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    ns = REG["sizes"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), layout="constrained", sharex=True)
    plt.rcParams["svg.fonttype"] = "none"
    for j, (a, p, title) in enumerate(
        (
            (0.35, 0.0, "Amplitude error"),
            (0.0, 0.70, "Phase error"),
            (0.35, 0.70, "Combined error"),
        )
    ):
        for cap, color, marker in zip(
            CAPS, ("#127269", "#4163A8", "#AA4D2C"), ("o", "s", "^")
        ):
            ss = sorted(
                (
                    r
                    for r in rows
                    if r["capacity"] == cap
                    and float(r["a"]) == a
                    and float(r["p_rad"]) == p
                ),
                key=lambda r: int(r["n"]),
            )
            gain = np.array([float(r["mean_gain"]) * 100 for r in ss])
            lo = np.array([float(r["lower"]) * 100 for r in ss])
            hi = np.array([float(r["upper"]) * 100 for r in ss])
            harm = np.array([float(r["material_harm"]) * 100 for r in ss])
            ax[0, j].errorbar(
                ns,
                gain,
                yerr=[gain - lo, hi - gain],
                color=color,
                fmt=marker + "-",
                capsize=2,
                lw=1.4,
                label=f"{cap}  {float(ss[0]['parameters']) / 1e6:.1f}M",
            )
            ax[1, j].plot(ns, harm, color=color, marker=marker, lw=1.4)
        ax[0, j].set_title(title)
        ax[0, j].axhline(0, color=".4", lw=0.8)
        ax[1, j].set_ylim(-4, 104)
        ax[1, j].set_xlabel("Number of modes N")
        for axis in ax[:, j]:
            axis.set_xscale("log", base=2)
            axis.set_xticks(ns, labels=[str(n) for n in ns], fontsize=9)
            axis.axvspan(2.8, 11.25, color="#E6EDEB", zorder=0)
            axis.set_xlim(2.8, 278)
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    ax[0, 0].set_ylabel("Mean gain over NOOP (fidelity points)")
    ax[1, 0].set_ylabel("Loss >1 fidelity point (%)")
    ax[0, 0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(
        "Does increasing parameter count improve recovery at larger N?", fontsize=15
    )
    fig.supxlabel(
        "Same 32,768 optimizer updates, seed and four-cell counts. Shaded sizes were used in training. Bars give pointwise 95% uncertainty of the mean across 128 parents.",
        fontsize=9,
    )
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            ROOT / ("matched_capacity_recovery." + ext), dpi=180, bbox_inches="tight"
        )
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    ROOT = parser.parse_args().root
    REG = json.loads((ROOT / "protocol.json").read_text())
    summary()
    plot()
