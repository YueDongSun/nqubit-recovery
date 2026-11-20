"""Aggregate only the twelve B0 points specified before the primary grid results."""

from pathlib import Path
from collections import Counter
import json
import numpy as np

from . import aggregate_grid as agg


def main():
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--b2", type=Path, required=True)
    cli.add_argument("--b0", type=Path, required=True)
    args = cli.parse_args()
    folder, b0_root = args.b2, args.b0
    plan = json.loads((b0_root / "registration.json").read_text())
    expected = {
        (n, kind, a, p)
        for n in plan["n"]
        for kind, coords in plan["fixed_points"].items()
        for a, p in coords
    }
    summary, actions, quantiles = agg.aggregate(b0_root, expected_study=plan["study"])
    cells = [c for c in summary["cells"] if c["status"] == "scored"]
    observed = {(c["n"], c["condition"], c["a_index"], c["p_index"]) for c in cells}
    if observed != expected:
        raise ValueError("Returned B0 cells differ from the predeclared panel")
    differences = []
    for c in cells:
        sub = (
            Path("n" + str(c["n"]))
            / c["condition"]
            / f"a{c['a_index']}_p{c['p_index']}"
            / "episode_panels.npz"
        )
        with np.load(b0_root / sub) as x, np.load(folder / sub) as y:
            for k in ("episode_ids", "seeds", "initial_coords", "noop_fidelity"):
                if not np.array_equal(x[k], y[k]):
                    raise ValueError(f"Different paired {k}")
            delta = (
                y["learned_k_fidelity"][:, :, -1] - x["learned_k_fidelity"][:, :, -1]
            ).mean(axis=0)
            boot = (
                np.random.default_rng([20261120, c["n"], 983])
                .choice(delta, (10000, len(delta)), replace=True)
                .mean(axis=1)
            )
            lo, hi = np.quantile(boot, [0.025, 0.975])
            differences.append(
                dict(
                    n=c["n"],
                    condition=c["condition"],
                    a_index=c["a_index"],
                    p_index=c["p_index"],
                    paired_parent_directions=len(delta),
                    seed_count=len(x["seeds"]),
                    b2_minus_b0_terminal=float(delta.mean()),
                    pointwise_lower=float(lo),
                    pointwise_upper=float(hi),
                )
            )
    summary.update(
        total_cells=12,
        returned_cells=len(cells),
        cells=cells,
        decision_counts=dict(Counter(c["decision"] for c in cells)),
        comparison_role="secondary paired B0 replication at twelve preselected cells",
        between_model_contrasts="Descriptive paired state bootstrap, pointwise intervals. The operating decisions retain the more conservative 400-endpoint rule used for B2.",
    )
    (b0_root / "secondary_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    agg.array_csv(b0_root / "grid_cells.csv", cells)
    agg.array_csv(b0_root / "b2_minus_b0.csv", differences)
    agg.array_csv(b0_root / "action_conditions.csv", actions)
    agg.array_csv(b0_root / "terminal_individual_quantiles.csv", quantiles)
    print(summary["decision_counts"])


if __name__ == "__main__":
    main()
