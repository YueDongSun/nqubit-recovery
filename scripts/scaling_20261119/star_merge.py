"""Collect matched, complete single-seed STAR fits for paired scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

import star_long as study


from terminal_contrasts import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--score-terminal", action="store_true",
                        help="Open one common evaluation panel only after validating all three fit-only terminals")
    parser.add_argument("--device", default="cpu", help="Scoring device for the joint terminal panel")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != 3:
        raise ValueError("Three distinct optimizer seeds are required")
    if not args.engineering and not args.score_terminal:
        parser.error("scientific confirmation requires an explicit --score-terminal request")
    role = "development" if args.engineering else "confirmation"
    result_file = "engineering_result.json" if args.engineering else "result.json"
    fit_file = ("engineering_" if args.engineering else "") + "fit_terminal.json"
    panel_file = "engineering_scores.npz" if args.engineering else "confirmation.npz"
    digest = study.file_sha(args.reference)
    common, models, panels, devices = None, {}, {}, {}
    for run, seed in zip(args.seed_runs, args.seeds):
        result = json.loads((run / (fit_file if args.score_terminal else result_file)).read_text())
        identity = result["identity"]
        weights = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if weights["identity"] != identity or json.loads((run / "registration.json").read_text()) != identity:
            raise ValueError(f"Registration, weights and terminal result disagree for {run}")
        if (identity["schema"] != "corrected_shared_star_fixed_budget_v1"
                or identity["seeds"] != [seed] or identity["evaluation_role"] != role
                or identity["parent_sha256"] != digest or tuple(identity["sizes"]) != study.SIZES
                or result["step_per_arm_seed"] != identity["updates"] or weights["step"] != identity["updates"]):
            raise ValueError(f"Incomplete or incompatible terminal run {run}")
        shared = {k: v for k, v in identity.items() if k not in ("seeds", "device", "parent")}
        if common is not None and common != shared:
            raise ValueError("The three runs differ in source, panel, initializer or update budget")
        common = shared
        if args.score_terminal and (identity.get("scoring") != "deferred_three_seed"
                                   or result.get("confirmation_scored") is not False):
            raise ValueError(f"Expected an unscored fit-only terminal: {run}")
        devices[str(seed)] = result.get("runtime_segments", identity.get("device"))
        expected = {study.model_key(arm, seed, n) for n in study.SIZES for arm in study.ARMS}
        if set(weights["models"]) != expected:
            raise ValueError(f"Missing or unexpected model exports for seed {seed}")
        models.update(weights["models"])
        if args.score_terminal:
            continue
        with np.load(run / panel_file, allow_pickle=False) as archive:
            required = {f"n{n}_visibility" for n in study.SIZES}
            for n in study.SIZES:
                for law in ("corrected", "archived"):
                    roles = ["prior", "frozen", *(study.model_key(a, seed, n) for a in study.ARMS)]
                    required.update(f"n{n}_{law}_{name}" for name in roles)
            if set(archive.files) != required:
                raise ValueError(f"Unmatched state-level panel keys for seed {seed}")
            for key in sorted(required):
                data = archive[key]
                if data.shape != (identity["rows"][role],) or not np.isfinite(data).all():
                    raise ValueError(f"Invalid state-level panel {key}")
                if key in panels:
                    # Common inference outputs can differ by FP64 rounding across GPUs.
                    np.testing.assert_allclose(data, panels[key], atol=1e-11, rtol=1e-10)
                else:
                    panels[key] = data
    identity = dict(common, schema="collected_shared_star_v1", seeds=args.seeds, devices=devices)
    args.output.mkdir(parents=True, exist_ok=True)
    with study.owner(args.output):
        dest = args.output / result_file
        if dest.exists():
            if json.loads(dest.read_text())["identity"] != identity:
                raise ValueError("Output already belongs to a different panel")
            return
        if args.score_terminal:
            # No scientific panel is generated until all three immutable fit records passed checks.
            torch.set_num_threads(2)
            device = torch.device(args.device)
            def load_core(state):
                core = study.Unified_star_density_net(hidden=identity["hidden"],
                                         rank_columns=identity["rank"], device=device)
                core.load_state_dict(state, strict=True)
                return core.eval()
            frozen = load_core(torch.load(args.reference, map_location="cpu", weights_only=True))
            fitted = {key: load_core(state) for key,state in models.items()}
            _, panels = study.evaluate(fitted, frozen, role, device, args.seeds, identity["split_seed"])
        study.atom_torch(args.output / "terminal_weights.pt",
                         dict(identity=identity, step=identity["updates"], models=models))
        panel = args.output / panel_file
        scratch = panel.with_suffix(".tmp.npz")
        np.savez_compressed(scratch, **panels)
        scratch.replace(panel)
        contrasts = analyze(panels, seeds=args.seeds)
        study.atom_json(dest, dict(identity=identity, step_per_arm_seed=identity["updates"],
                                   weights_file="terminal_weights.pt", panel_file=panel_file,
                                   total_optimizer_updates=identity["updates"]*len(args.seeds)*len(study.ARMS),
                                   paired_terminal_contrasts=contrasts, uncertainty_unit="independent latent state"))


if __name__ == "__main__":
    main()
