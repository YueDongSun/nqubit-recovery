"""Locked paired state-risk contrasts for the new capacity, module and bank matrices."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

MAIN_SEEDS = (7301, 7302, 7303)
MODULE_SEEDS = (7301, 7302)
FAMILIES = {
    -1: "nominal",
    0: "phase",
    1: "amplitude",
    2: "amplitude_and_phase",
    3: "dephasing",
    5: "bipartition",
}
MODULES = ("noise_moe", "rho8", "esm", "csm", "laplacian", "fd", "all_no_fd")
SIZES = (3, 4, 8)


def contrast_plan():
    """Specify comparisons before looking at the available confirmation candidates."""
    plan = []

    def add(group, name, choice, seeds, *terms):
        plan.append(
            {"family": group, "name": name, "candidate": choice, "seeds": seeds, "terms": terms}
        )

    def run(cap, bank, arm="joint"):
        return f"{arm}_{cap}_{bank}_s{{seed}}"

    def mod(profile, bank_b=False):
        return f"module_{profile}_s{{seed}}" + ("_bank_b" if bank_b else "")

    def sub(group, name, choice, seeds, left, right):
        add(group, name, choice, seeds, (1, left, "network"), (-1, right, "network"))

    mechanisms = (
        ("real_counts_minus_command_replacement", "command_generated_counts_network"),
        ("network_minus_dev_selected_simple", "dev_baseline"),
    )
    for cap in ("cs", "cm", "cl"):
        for bank in (1024, 8192, 32768):
            model = run(cap, bank)
            stages = ["step_32768", "validation_selected"]
            if cap == "cs" and bank in (1024, 32768):
                stages.append("step_131072")
            for choice in stages:
                for label, other in mechanisms:
                    add(
                        "measurement_use",
                        f"{cap}_{bank}_{label}",
                        choice,
                        MAIN_SEEDS,
                        (1, model, "network"),
                        (-1, model, other),
                    )
    for bank in (1024, 8192, 32768):
        for cap, small in (("cm", "cs"), ("cl", "cm")):
            sub(
                "capacity",
                f"{cap}_minus_{small}_bank{bank}",
                "step_32768",
                MAIN_SEEDS,
                run(cap, bank),
                run(small, bank),
            )
    for cap in ("cs", "cm", "cl"):
        for bank, small in ((8192, 1024), (32768, 8192)):
            sub(
                "bank_size",
                f"bank{bank}_minus_{small}_{cap}",
                "step_32768",
                MAIN_SEEDS,
                run(cap, bank),
                run(cap, small),
            )
    for choice in ("step_32768", "step_131072", "validation_selected"):
        for arm, ref in (("joint", "state"), ("staged", "state"), ("joint", "staged")):
            sub(
                "training_objective",
                f"{arm}_minus_{ref}",
                choice,
                MAIN_SEEDS,
                run("cs", 32768, arm),
                run("cs", 32768, ref),
            )
    control = mod("noise_ce_control")
    for m in MODULES:
        sub("modules", f"{m}_minus_noise_ce", "step_8192", MODULE_SEEDS, mod(m), control)
    sub(
        "module_increments",
        "fd_plus_csm_minus_csm",
        "step_8192",
        MODULE_SEEDS,
        mod("fd"),
        mod("csm"),
    )
    sub(
        "module_increments",
        "full_joint_minus_all_no_fd",
        "step_8192",
        MODULE_SEEDS,
        run("cs", 8192),
        mod("all_no_fd"),
    )
    sub("modules", "full_joint_minus_noise_ce", "step_8192", MODULE_SEEDS, run("cs", 8192), control)
    sub(
        "bank_replication",
        "bank_b_csm_minus_noise_ce",
        "step_8192",
        MODULE_SEEDS,
        mod("csm", True),
        mod("noise_ce_control", True),
    )
    for m in ("csm", "noise_ce_control"):
        sub(
            "bank_replication",
            f"{m}_bank_b_minus_a",
            "step_8192",
            MODULE_SEEDS,
            mod(m, True),
            mod(m),
        )
    add(
        "bank_replication",
        "bank_b_minus_a_csm_effect",
        "step_8192",
        MODULE_SEEDS,
        (1, mod("csm", True), "network"),
        (-1, mod("noise_ce_control", True), "network"),
        (-1, mod("csm"), "network"),
        (1, control, "network"),
    )
    return plan


class _StateScores:
    """Cache only the per-state fields consumed by the planned comparisons."""

    def __init__(self, root, frozen):
        self.root = Path(root)
        self.frozen = frozen
        self.results = {}
        self.rows = {}
        self.reference = {}

    def get(self, run, label, n):
        entry = self.frozen["candidates"].get(run)
        if entry is None or label not in entry["choices"]:
            return None
        if run not in self.results:
            dest = self.root / run / "confirmation" / "result.json"
            if not dest.exists():
                return None
            data = json.loads(dest.read_text())
            identity = data["identity"]
            roster = self.frozen["roster"]
            if (
                identity.get("lane") != run
                or identity.get("source") != roster["source"]
                or identity.get("split_namespace") != roster["namespace"]
                or identity.get("weights") != entry["weights"]
            ):
                raise ValueError(
                    f"{run}: confirmation identity does not match frozen source, namespace and weights"
                )
            self.results[run] = {
                selector: {
                    size: {
                        "state_file": row["state_file"],
                        "updates": row["updates"],
                        "count_panel_identity": row["count_panel_identity"],
                        "dev_selected_baseline": row["metrics"]["dev_selected_baseline"],
                    }
                    for size, row in systems.items()
                }
                for selector, systems in data["candidates"].items()
            }
        step = entry["choices"][label]
        key = (run, step, n)
        if key in self.rows:
            return self.rows[key]
        record = self.results[run].get(label, {}).get(str(n))
        if record is None:
            return None
        if record["updates"] != step:
            raise ValueError(f"{run}: wrong candidate update")
        states = json.loads((self.root / run / "confirmation" / record["state_file"]).read_text())
        family = np.array(states["family"], dtype=np.int64)
        count = self.frozen["roster"]["states_per_n"]
        if family.shape != (count,) or set(np.unique(family)) != set(FAMILIES):
            raise ValueError(f"{run}/N{n}: wrong confirmation length or scientific family support")
        index = np.asarray(states.get("state_index", np.arange(count)))
        if not np.array_equal(index, np.arange(count)):
            raise ValueError(
                f"{run}/N{n}: independent state ordering differs from the stored cohort"
            )
        metrics = {k: np.asarray(v, dtype=np.float64) for k, v in states["estimators"].items()}
        if any(v.shape != family.shape or not np.isfinite(v).all() for v in metrics.values()):
            raise ValueError(f"{run}/N{n}: invalid per-state risk")
        panel = record["count_panel_identity"]
        if not panel:
            raise ValueError("missing observation panel identity")
        command = metrics["command"]
        if n not in self.reference:
            self.reference[n] = (family, panel, command)
        original, first_panel, first_command = self.reference[n]
        if not (
            panel == first_panel
            and np.array_equal(family, original)
            and np.allclose(command, first_command, atol=2e-6, rtol=1e-6)
        ):
            raise ValueError(
                f"N{n}: confirmation observations, original state risks or families are unpaired"
            )
        baseline = record["dev_selected_baseline"]
        if baseline not in ("classical", "train_prior", "command"):
            raise ValueError(f"unregistered DEV baseline: {baseline}")
        metrics["dev_baseline"] = metrics[baseline]
        self.rows[key] = (metrics, baseline, step)
        return self.rows[key]


def _seed_differences(scores, spec, n):
    differences = {}
    selections = {}
    missing = {}
    for seed in spec["seeds"]:
        parts = []
        details = []
        unavailable = []
        for weight, template, field in spec["terms"]:
            run = template.format(seed=seed)
            result = scores.get(run, spec["candidate"], n)
            if result is None:
                unavailable.append(run)
                continue
            arrays, baseline, step = result
            parts.append(weight * arrays[field])
            details.append(
                {
                    "run": run,
                    "step": step,
                    "coefficient": weight,
                    "risk": baseline if field == "dev_baseline" else field,
                    "mean_risk": float(arrays[field].mean()),
                }
            )
        if unavailable:
            missing[str(seed)] = sorted(set(unavailable))
        else:
            differences[str(seed)] = np.sum(parts, axis=0)
            selections[str(seed)] = details
    return differences, selections, missing


def _bootstrap_groups(values, family, *, draws, seed):
    """Paired, stratified nonparametric bootstrap. One vectorized column per contrast."""
    rng = np.random.default_rng(seed)
    overall = np.zeros((draws, values.shape[1]), np.float64)
    strata = {}
    for value in FAMILIES:
        group = values[family == value]
        n = len(group)
        means = np.empty_like(overall)
        for i in range(0, draws, 256):
            end = min(i + 256, draws)
            w = rng.multinomial(n, np.full(n, 1 / n), size=end - i).astype(np.float64)
            means[i:end] = (w @ group) / n
        overall += means * (n / len(values))
        tail = 0.025 / (len(SIZES) * len(FAMILIES))
        strata[value] = np.quantile(means, (0.025, 0.975, tail, 1 - tail), axis=0)
    return overall, strata


def scientific_contrasts(root, frozen, *, bootstrap_draws=20000):
    """Return JSON-ready fixed-seed comparisons from already-scored, frozen scientific rows."""
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    plan = contrast_plan()
    families = Counter(p["family"] for p in plan)
    counts = {k: v * len(SIZES) for k, v in families.items()}
    scores = _StateScores(root, frozen)
    records = []
    for n in SIZES:
        complete = []
        vectors = []
        for spec in plan:
            diffs, bindings, missing = _seed_differences(scores, spec, n)
            record = {
                "contrast": spec["name"],
                "candidate": spec["candidate"],
                "n": n,
                "multiplicity_family": spec["family"],
                "predeclared_family_size": counts[spec["family"]],
                "expected_seeds": list(spec["seeds"]),
                "seed_rows": bindings,
                "missing_by_seed": missing,
                "per_seed_mean": {seed: float(v.mean()) for seed, v in diffs.items()},
            }
            if missing:
                record["status"] = "incomplete_seed_panel" if diffs else "not_available"
            else:
                record["status"] = "complete_fixed_seed_panel"
                averaged = np.mean(list(diffs.values()), axis=0)
                record["n_states"] = len(averaged)
                record["mean_delta_risk"] = float(averaged.mean())
                record["by_family"] = {}
                family = scores.reference[n][0]
                for v, name in FAMILIES.items():
                    subset = family == v
                    record["by_family"][name] = {
                        "label": v,
                        "n_states": int(subset.sum()),
                        "mean_delta_risk": float(averaged[subset].mean()),
                        "per_seed_mean": {
                            seed: float(x[subset].mean()) for seed, x in diffs.items()
                        },
                    }
                complete.append(record)
                vectors.append(averaged)
            records.append(record)
        if not vectors:
            continue
        panel = np.column_stack(vectors)
        overall, strata = _bootstrap_groups(
            panel, scores.reference[n][0], draws=bootstrap_draws, seed=86113 + n
        )
        for j, record in enumerate(complete):
            tail = 0.025 / record["predeclared_family_size"]
            record["ci95_pointwise"] = np.quantile(overall[:, j], [0.025, 0.975]).tolist()
            record["ci95_simultaneous"] = np.quantile(overall[:, j], [tail, 1 - tail]).tolist()
            for value, name in FAMILIES.items():
                record["by_family"][name]["ci95_pointwise"] = strata[value][:2, j].tolist()
                record["by_family"][name]["ci95_within_contrast_across_N_and_six_families"] = (
                    strata[value][2:, j].tolist()
                )
    return {
        "schema": "matrix_scientific_contrasts_v1",
        "namespace": frozen["roster"]["namespace"],
        "endpoint": "paired squared Frobenius error of conditional K1 density; negative favors first term",
        "statistical_unit": "independent latent state; within-seed paired differences averaged across the full fixed fitted seed panel before stratified state resampling",
        "interval_scope": "overall two-sided Bonferroni by named contrast family across all planned sizes and candidates, regardless of missing fits; family-specific estimates have pointwise intervals and a separate 18-endpoint simultaneous interval for this fixed contrast across three sizes and six source families (not across all contrasts)",
        "incomplete_seed_policy": "available individual seed point estimates only; no pooled estimate or interval",
        "interpretation": "matched update counts do not match compute time; candidate-step and DEV-selected comparisons retain their actual fitted steps; bank interaction conditional on two declared banks",
        "bootstrap_draws": bootstrap_draws,
        "family_sizes": counts,
        "rows": records,
    }
