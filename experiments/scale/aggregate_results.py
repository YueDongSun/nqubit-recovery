"""Pair terminal outcomes by parent across N and separate uncertainty of means from trajectories."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

DRAWS = 20000


def interval(events, total, tail):
    if total == 0:
        return None, None
    lo = 0.0 if events == 0 else float(beta.ppf(tail, events, total - events + 1))
    hi = (
        1.0
        if events == total
        else float(beta.ppf(1 - tail, events + 1, total - events))
    )
    return lo, hi


def save_csv(name, rows):
    with (ROOT / name).open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def inputs():
    cells = {}
    ids = np.arange(PARENTS)
    for spec in REG["cells"]:
        name = spec["tag"]
        f = ROOT / "rows" / (name + ".csv.gz")
        if not f.is_file():
            continue
        groups = defaultdict(list)
        with gzip.open(f, "rt", newline="") as fp:
            for row in csv.DictReader(fp):
                groups[row["policy"]].append(row)
        policies = set(REG["policies"]) - ({"learned_fd"} if spec["K"] > 1 else set())
        if set(groups) != policies:
            raise ValueError(f"incomplete policies at {name}")
        for policy, g in groups.items():
            g.sort(key=lambda x: int(x["parent_id"]))
            v = {k: np.array([float(r[k]) for r in g]) for k in g[0] if k != "policy"}
            for field in ("N", "K", "A", "P"):
                if not np.allclose(v[field], spec[field]):
                    raise ValueError((name, field))
            if (
                not np.array_equal(v["parent_id"], ids)
                or not (v["seed"] == REG["optimizer_seed"]).all()
            ):
                raise ValueError("unexpected parent or optimizer identity")
            if (
                not (v["reads_decision"] == REG["reads_decision"]).all()
                or not (v["reads_terminal"] == REG["reads_terminal"]).all()
            ):
                raise ValueError("budget mismatch")
            if not (
                sum(v["terminal_" + k] for k in ("00", "10", "01", "11"))
                == v["reads_terminal"]
            ).all():
                raise ValueError("terminal cells do not conserve READs")
            if not (
                np.allclose(v["amp_rms0"], spec["A"])
                and np.allclose(v["phase_rms0"], spec["P"])
            ):
                raise ValueError("different initial physical errors")
            cells[(spec["N"], spec["K"], spec["A"], spec["P"], policy)] = v
    return cells


def main():
    global ROOT, REG, PARENTS
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    ROOT = parser.parse_args().root
    REG = json.loads((ROOT / "protocol.json").read_text())
    PARENTS = REG["parents"]
    cells = inputs()
    if len(cells) != sum(5 if r["K"] == 1 else 4 for r in REG["cells"]):
        raise RuntimeError("complete registered grid required")
    estimates = {}
    families = defaultdict(list)
    for key, v in cells.items():
        n, k, a, p, policy = key
        no = cells[n, k, a, p, "noop"]
        np.testing.assert_allclose(v["F0"], no["F0"], atol=2e-12)
        d = v["FT"] - no["FT"]
        np.testing.assert_allclose(d, v["delta_F"], atol=2e-12)
        if not np.array_equal(d < -1e-12, v["harm_vs_noop"].astype(bool)):
            raise ValueError("reported harm is inconsistent")
        estimates[key, "noop"] = d
        families[policy].append((key, "noop"))
        if k > 1:
            estimates[key, "one"] = v["FT"] - cells[n, 1, a, p, policy]["FT"]
            families[policy].append((key, "one"))
    # One parent resampling per draw across all severities, sizes and horizons.
    # Split familywise error equally between the simultaneous means and binomial harms.
    rng = np.random.default_rng(202611201943719)
    weights = (
        rng.multinomial(PARENTS, np.full(PARENTS, 1 / PARENTS), size=DRAWS) / PARENTS
    )
    intervals = {}
    bounds = {}
    for policy, keys in families.items():
        values = np.column_stack([estimates[z] for z in keys])
        avg = values.mean(0)
        se = values.std(0, ddof=1) / np.sqrt(PARENTS)
        use = se > 1e-14
        if policy in ("learned_plugin", "learned_fd"):
            critical = (
                float(
                    np.quantile(
                        np.max(
                            np.abs((weights @ (values[:, use] - avg[use])) / se[use]),
                            axis=1,
                        ),
                        0.975,
                    )
                )
                if use.any()
                else 0
            )
            harms = sum(x[1] == "noop" for x in keys)
            bounds[policy] = dict(
                mean_endpoints=len(keys),
                harm_endpoints=harms,
                critical=critical,
                harm_tail=0.025 / (2 * harms),
                scope="simultaneous within policy",
            )
        else:
            critical = float(t.ppf(0.975, PARENTS - 1))
            bounds[policy] = dict(
                mean_endpoints=len(keys),
                critical=critical,
                harm_tail=0.025,
                scope="descriptive pointwise",
            )
        for key, value, width in zip(keys, avg, critical * se):
            intervals[key] = (float(value), float(value - width), float(value + width))
    summary = []
    quantiles = []
    for key, v in sorted(cells.items()):
        n, k, a, p, policy = key
        gain, lo, hi = intervals[key, "noop"]
        over1, lo1, hi1 = intervals.get((key, "one"), (None, None, None))
        harmed = v["harm_vs_noop"].astype(bool)
        executed = v["executed"].astype(bool)
        hl, hu = interval(int(harmed.sum()), PARENTS, bounds[policy]["harm_tail"])
        status = "descriptive"
        if policy in ("learned_plugin", "learned_fd"):
            if a == 0 and p == 0:
                status = (
                    "hold_preserved"
                    if hu < 0.1
                    else "hold_failed"
                    if hl > 0.1
                    else "hold_unresolved"
                )
            elif hi < 0 or hl > 0.1:
                status = "harmful"
            elif hu < 0.1 and (
                (k == 1 and lo > 0.01) or (k > 1 and lo > 0 and lo1 > 0.01)
            ):
                status = "usable"
            else:
                status = "unresolved"
        stub = dict(
            n=n,
            k=k,
            amp_log_rms=a,
            phase_rad_rms=p,
            phase_deg_rms=float(np.degrees(p)),
            policy=policy,
            parents=PARENTS,
            training_support="trained N"
            if n in REG["training_sizes"]
            else "size extrapolation",
        )
        counts = [int(v["terminal_" + nm].sum()) for nm in ("00", "10", "01", "11")]
        summary.append(
            dict(
                **stub,
                status=status,
                F0=float(v["F0"].mean()),
                FT=float(v["FT"].mean()),
                gain_noop=gain,
                gain_noop_lower=lo,
                gain_noop_upper=hi,
                gain_one=over1,
                gain_one_lower=lo1,
                gain_one_upper=hi1,
                harmed=int(harmed.sum()),
                harm_rate=float(harmed.mean()),
                harm_lower=hl,
                harm_upper=hu,
                harmed_over_one_point=int((estimates[key, "noop"] < -0.01).sum()),
                harm_over_one_point_rate=float((estimates[key, "noop"] < -0.01).mean()),
                executed=int(executed.sum()),
                coverage=float(executed.mean()),
                harm_given_execution=float(harmed[executed].mean())
                if executed.any()
                else None,
                fraction_saturated=float(v["saturated"].mean()),
                terminal_00=counts[0],
                terminal_10=counts[1],
                terminal_01=counts[2],
                terminal_11=counts[3],
                rows_file=f"rows/{next(x['tag'] for x in REG['cells'] if (x['N'], x['K'], x['A'], x['P']) == key[:4])}.csv.gz",
            )
        )
        for field, var in (
            ("amp_log_rms", v["amp_rmsT"]),
            ("phase_deg_rms", np.degrees(v["phase_rmsT"])),
            ("conditional_overlap", v["FT"]),
        ):
            lo95, q10, med, q90, hi95 = np.quantile(var, [0.025, 0.1, 0.5, 0.9, 0.975])
            quantiles.append(
                dict(
                    **stub,
                    terminal_variable=field,
                    q025=lo95,
                    q10=q10,
                    median=med,
                    q90=q90,
                    q975=hi95,
                )
            )
    save_csv("operating_cells.csv", summary)
    save_csv("terminal_uncertainty.csv", quantiles)
    result = dict(
        sizes=REG["sizes"],
        cells=len(REG["cells"]),
        policy_cells=len(summary),
        unique_parents=PARENTS,
        independent_optimizer_seeds=1,
        train_sizes=REG["training_sizes"],
        heldout_size_extrapolation=REG["extrapolation_sizes"],
        intervals=bounds,
        bootstrap_draws=DRAWS,
        individual_uncertainty="empirical terminal quantiles over 128 independent parent draws; not a confidence interval for the mean",
        tested_counts={
            str(n): {
                s: sum(
                    x["n"] == n
                    and x["policy"] == "learned_plugin"
                    and x["k"] == 1
                    and x["status"] == s
                    for x in summary
                )
                for s in (
                    "usable",
                    "unresolved",
                    "harmful",
                    "hold_preserved",
                    "hold_unresolved",
                    "hold_failed",
                )
            }
            for n in REG["sizes"]
        },
    )
    (ROOT / "range_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({k: result[k] for k in ("cells", "policy_cells", "tested_counts")})
    )


if __name__ == "__main__":
    main()
