"""Register matrix checkpoints and pair their locked scores by independent state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .effective_inference_experiment import write_json
from .long_projector_training import matrix_fit, constructor


RUNGS = (4096, 8192, 32768, 131072)
SCHEMA = "corrected_projector_matrix_20261120_v1"


def manifest_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def register_once(path, record):
    """Publish a fully written immutable record, also under simultaneous retries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        tmp = Path(f.name)
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != record:
                raise ValueError(f"registered choices changed at {path}")
    finally:
        tmp.unlink()


def make_roster(matrices, namespace, source):
    runs = {}
    for matrix in matrices:
        with Path(matrix).open(encoding="utf8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            name = row["run_id"]
            if name in runs:
                raise ValueError(f"duplicate run {name}")
            spec = matrix_fit(Path(matrix), name)
            runs[name] = {"study": row["study"], "fit": spec,
                          "fixed_steps": [s for s in RUNGS if s <= spec["steps"]]}
    if not runs or not namespace or not source:
        raise ValueError("empty experiment scope")
    return dict(schema=SCHEMA, namespace=namespace, source=source, runs=runs,
                sizes=[3, 4, 8], states_per_n=768,
                baselines=["command", "train_prior", "classical", "command_generated_counts_network"],
                family="six declared generator families", optimization_selection="first minimum positive-step DEV risk")


def freeze_roster(root, roster, partial=False):
    """Read training receipts only. Complete all candidate decisions before scoring."""
    frozen, missing = {}, {}
    for name, registered in roster["runs"].items():
        home = root / name
        done = home / "training_done.json"
        spec = registered["fit"]
        if not done.exists():
            if not partial:
                raise FileNotFoundError(done)
            done = home / "progress.json"
            if not done.exists():
                missing[name] = "not started"
                continue
        outcome = json.loads(done.read_text())
        identity = json.loads((home / "registration.json").read_text())["identity"]
        spec = registered["fit"]
        if identity["lane"] != name or identity["source"] != roster["source"] or identity["split_namespace"] != roster["namespace"]:
            raise ValueError(f"unmatched run provenance for {name}")
        expected = (spec["seed"], spec["capacity"], spec["rows"], spec["lr"], spec["steps"], spec["arm"])
        observed = (identity["seed"], identity["constructor"]["profile"], identity["train_states_per_n"],
                    identity["learning_rate"], identity["fixed_updates"], identity["arm"])
        cfg = constructor(spec["arm"], "factor_gram_v2", spec["capacity"], spec["idea_profile"])
        if (observed != expected or identity["constructor"] != cfg
                or identity.get("staged_primary_updates") != spec["stage"]
                or spec.get("train_namespace") != identity.get("train_namespace")):
            raise ValueError(f"unmatched fit recipe for {name}")
        if int(outcome["step"]) != spec["steps"]:
            missing[name] = f"censored at {outcome['step']} of {spec['steps']} updates"
        choices = {f"step_{s}":s for s in registered["fixed_steps"] if s <= outcome["step"]}
        if not choices or not outcome.get("best_validation_step", 0) > 0:
            if not partial:
                raise ValueError(f"incomplete terminal or missing positive DEV selection: {name}")
            missing[name] = "terminal budget or positive DEV selection missing"
            continue
        choices["validation_selected"] = int(outcome["best_validation_step"])
        files = {}
        for s in sorted(set(choices.values())):
            path = home / f"snapshots/step_{s:09d}/core.pt"
            with path.open("rb") as f:
                files[str(s)] = hashlib.file_digest(f, "sha256").hexdigest()
        frozen[name] = dict(choices=choices, weights=files, identity=identity, training=outcome)
    if not frozen:
        raise ValueError("no completed candidates to confirm")
    return dict(schema=SCHEMA, roster=roster, candidates=frozen, missing=missing)


def summarize(root, frozen):
    """One row per frozen logical comparison, retaining shared checkpoint aliases."""
    results = []
    pair_identity = {}
    for name, entry in frozen["candidates"].items():
        dest = root / name / "confirmation"
        data = json.loads((dest / "result.json").read_text())
        if data["identity"].get("weights") != entry["weights"] or data["identity"].get("split_namespace") != frozen["roster"]["namespace"]:
            raise ValueError(f"score is not bound to frozen candidates for {name}")
        for label, step in entry["choices"].items():
            for n in frozen["roster"]["sizes"]:
                rec = data["candidates"][label][str(n)]
                if rec["updates"] != step:
                    raise ValueError("wrong selected score")
                pair = rec["count_panel_identity"]
                if pair_identity.setdefault(n, pair) != pair:
                    raise ValueError(f"unpaired observations at N={n}")
                states = json.loads((dest / rec["state_file"]).read_text())
                model = np.asarray(states["estimators"]["network"])
                families = np.asarray(states["family"])
                if len(model) != frozen["roster"]["states_per_n"] or len(families) != len(model):
                    raise ValueError("unexpected state count")
                for group in [None, *np.unique(families)]:
                    mask = np.ones(len(model), bool) if group is None else families == group
                    diffs = {key: float((model[mask]-np.asarray(states["estimators"][key])[mask]).mean())
                             for key in frozen["roster"]["baselines"]}
                    results.append(dict(run_id=name, candidate=label, n=n, step=step,
                                        family="all" if group is None else int(group),
                                        state_count=int(mask.sum()), risk=float(model[mask].mean()),
                                        paired_model_minus=diffs, state_file=str(dest / rec["state_file"])))
    from .matrix_scientific_contrasts import scientific_contrasts
    return dict(schema=SCHEMA, rows=results, missing=frozen["missing"],
                scientific_contrasts=scientific_contrasts(root, frozen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["register", "freeze", "seal-calibration", "aggregate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", nargs="+", type=Path)
    parser.add_argument("--split-namespace")
    parser.add_argument("--source")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    roster_file = args.root / "matrix_roster.json"
    frozen_file = args.root / "frozen_candidates.json"
    if args.stage == "register":
        register_once(roster_file, make_roster(args.matrix, args.split_namespace, args.source))
    elif args.stage == "freeze":
        register_once(frozen_file, freeze_roster(args.root, json.loads(roster_file.read_text()), args.allow_incomplete))
    elif args.stage == "seal-calibration":
        frozen = json.loads(frozen_file.read_text())
        digest = manifest_digest(frozen)
        for name, entry in frozen["candidates"].items():
            receipt = json.loads((args.root / name / "confirmation" / "calibration_complete.json").read_text())
            if receipt != {"candidate_manifest_sha256": digest, "weights": entry["weights"], "sizes": frozen["roster"]["sizes"]}:
                raise ValueError(f"CAL incomplete for {name}")
        register_once(args.root / "matrix_calibration_complete.json", {"candidate_manifest_sha256": digest})
    else:
        write_json(args.root / "matrix_scores.json", summarize(args.root, json.loads(frozen_file.read_text())))


if __name__ == "__main__":
    main()
