"""Exact waiting-age and lifetime sensitivity under the registered 128-tick fusion instrument."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import single_event_fusion as fusion


def evaluate(registration, archived):
    kernel = fusion.optical_kernel()
    horizon = registration["horizon"]
    age = np.arange(horizon)
    summary, ages = [], []
    for source in registration["sources"]:
        p2, topology = source["p2"], source["topology"]
        posterior = fusion.herald_weights(p2)
        fresh = fusion.stored_components(posterior, 1.0, topology)
        for lifetime in registration["lifetimes_ticks"]:
            survival = np.ones(horizon) if lifetime is None else np.exp(-age / lifetime)
            stored = fusion.stored_components(posterior, survival, topology)
            click, single, overlap = np.einsum("ai,ijm,j->am", stored, kernel, fresh).T
            quality = np.divide(
                overlap, click, out=np.zeros_like(click), where=click > 0
            )
            for policy in ("synchronous", "buffered"):
                pair, paid = fusion.arrival_distribution(horizon, policy)
                mass = 1e6 * pair * click / paid
                for t in age[pair > 0]:
                    ages.append(
                        dict(
                            p2=p2,
                            topology=topology,
                            lifetime_ticks=lifetime,
                            policy=policy,
                            age=int(t),
                            clicks_per_million=float(mass[t]),
                            conditional_full_overlap=float(quality[t])
                            if click[t] > 0
                            else None,
                        )
                    )
                for cutoff in registration["overlap_thresholds"]:
                    eligible = (click > 0) & (overlap >= cutoff * click)
                    accepted = float(mass.sum())
                    useful = float(mass[eligible].sum())
                    eligible_ages = age[eligible & (pair > 0)]
                    row = dict(
                        p2=p2,
                        topology=topology,
                        lifetime_ticks=lifetime,
                        policy=policy,
                        threshold=cutoff,
                        accepted_per_million=accepted,
                        qualified_per_million=useful,
                        rejected_quality_fraction=(accepted - useful) / accepted
                        if accepted
                        else None,
                        last_qualified_age=int(eligible_ages[-1])
                        if len(eligible_ages)
                        else None,
                        conditional_full_overlap=float(pair @ overlap / (pair @ click)),
                        qualified_mean_age=float(
                            mass[eligible] @ age[eligible] / useful
                        )
                        if useful
                        else None,
                    )
                    if lifetime == 50 and cutoff == 0.7:
                        ref = next(
                            x
                            for x in archived["rows"]
                            if x["policy"] == policy
                            and x["lifetime"] == 50
                            and x["horizon"] == horizon
                            and x["p2"] == p2
                            and x["topology"] == topology
                        )
                        np.testing.assert_allclose(
                            [accepted, useful, row["conditional_full_overlap"]],
                            [
                                ref["click_per_million_writes"],
                                ref["qualified_click_per_million_writes"],
                                ref["full_w_overlap_given_click"],
                            ],
                            rtol=1e-10,
                            atol=1e-12,
                        )
                    summary.append(row)
    baseline = {
        (x["p2"], x["topology"], x["lifetime_ticks"], x["threshold"]): x
        for x in summary
        if x["policy"] == "synchronous"
    }
    for row in summary:
        sync = baseline[
            row["p2"], row["topology"], row["lifetime_ticks"], row["threshold"]
        ]["qualified_per_million"]
        row["qualified_minus_sync_per_million"] = row["qualified_per_million"] - sync
        row["qualified_ratio_to_sync"] = (
            row["qualified_per_million"] / sync if sync > 0 else None
        )
    return summary, ages


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reg = dict(
        horizon=128,
        lifetimes_ticks=[12.5, 25, 50, 100, 200, None],
        overlap_thresholds=[0.65, 0.70, 0.75],
        sources=[
            dict(p2=0.0, topology="local"),
            dict(p2=0.0004, topology="local"),
            dict(p2=0.0004, topology="cross"),
        ],
    )
    summary, ages = evaluate(reg, json.loads(args.archive.read_text()))
    for name, rows in (("lifetime_thresholds", summary), ("age_contributions", ages)):
        with (args.output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "result.json").write_text(
        json.dumps(
            dict(
                registration=reg,
                exact_conditions=len(summary),
                age_rows=len(ages),
                archive_matched_cases=6,
                summary=summary,
            ),
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
