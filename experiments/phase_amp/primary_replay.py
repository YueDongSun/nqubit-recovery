"""Recover the initial latent coordinates from recorded seeds and validate them against saved trajectories."""

from pathlib import Path
import csv, json
import numpy as np
from scipy.stats import beta
from .coordinates import primary_initial, error_sizes, fidelity

HERE = Path(__file__).resolve().parent
ABINS = [0, 0.12, 0.25, 0.35, 0.70, 1.4, np.inf]
PBINS = [0, 0.20, 0.45, 0.70, 1.10, np.inf]


def ci(k, n):
    if not n:
        return None
    return [
        float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
    ]


def main():
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    ROOT = args.primary
    HERE = args.output
    HERE.mkdir(parents=True, exist_ok=True)
    per_cell = {}
    binned = []
    for p in sorted(ROOT.glob("n?/*/*/result.json")):
        d = json.loads(p.read_text())
        key = str(p.parent.relative_to(ROOT))
        n = d["cell"]["n"]
        k = d["decisions"]["learner"]["tested_k"]
        with np.load(p.parent / "episode_panels.npz", allow_pickle=False) as z:
            ids = z["episode_ids"]
            b, v = primary_initial(ids, n, d["cell"]["spread"], d["cell"]["drift"])
            a, ph = error_sizes(b)
            t = np.arange(11)
            latent = b[:, None] + v[:, None] * t[None, :, None, None]
            null = fidelity(latent)
            err = float(np.max(np.abs(null - z["noop_fidelity"])))
            if err > 2e-12:
                raise ValueError("NOOP reconstruction differs from saved fidelity")
            finals = []
            harm = []
            anyact = []
            mistakes = []
            active = []
            for seed in d["seeds"]:
                name = f"seed{seed}_learner_inferred_k{k}"
                act = z[name + "_executed"]
                raw_f = z[name + "_fidelity"]
                c = np.zeros_like(latent)
                c[:, 1:] = np.cumsum(act, axis=1)
                obs = fidelity(latent + c)
                err = max(err, float(np.max(np.abs(obs - raw_f))))
                if err > 2e-12:
                    raise ValueError(
                        "logged policy trace differs from reconstructed hidden state"
                    )
                active_slot = np.max(np.abs(act), axis=(2, 3)) > 1e-8
                counter = fidelity(latent[:, 1:] + c[:, :-1])
                mistakes.append((obs[:, 1:] < counter - 1e-10) & active_slot)
                harm.append(raw_f[:, -1] < null[:, -1] - 1e-10)
                anyact.append(active_slot.any(1))
                active.append(active_slot)
                finals.append(latent[:, -1] + c[:, -1])
            harm = np.stack(harm)
            acted = np.stack(anyact)
            active = np.stack(active)
            mistakes = np.stack(mistakes)
            fa, fp = error_sizes(np.stack(finals))
            qs = [0.025, 0.10, 0.50, 0.90, 0.975]
            per_cell[key] = dict(
                episodes=len(ids),
                fixed_seeds=d["seeds"],
                tested_k=k,
                max_abs_replay_error=err,
                quantiles=qs,
                initial_amp_rms=np.quantile(a, qs).tolist(),
                initial_phase_rms_rad=np.quantile(ph, qs).tolist(),
                initial_phase_rms_deg=np.degrees(np.quantile(ph, qs)).tolist(),
                initial_pairwise_phase_rms_rad=(
                    np.quantile(ph, qs) * np.sqrt(2 * n / (n - 1))
                ).tolist(),
                initial_pairwise_phase_rms_deg=np.degrees(
                    np.quantile(ph, qs) * np.sqrt(2 * n / (n - 1))
                ).tolist(),
                terminal_pairwise_phase_rms_rad_mixture=(
                    np.quantile(fp, qs) * np.sqrt(2 * n / (n - 1))
                ).tolist(),
                terminal_pairwise_phase_rms_deg_mixture=np.degrees(
                    np.quantile(fp, qs) * np.sqrt(2 * n / (n - 1))
                ).tolist(),
                terminal_amp_rms_mixture=np.quantile(fa, qs).tolist(),
                terminal_phase_rms_rad_mixture=np.quantile(fp, qs).tolist(),
                harm_counts_per_seed=harm.sum(1).tolist(),
                executed_episodes_per_seed=acted.sum(1).tolist(),
                harm_executed_per_seed=(harm & acted).sum(1).tolist(),
                mistaken_executed_actions_per_seed=mistakes.sum((1, 2)).tolist(),
                executed_actions_per_seed=active.sum((1, 2)).tolist(),
            )
            for ai in range(len(ABINS) - 1):
                for pi in range(len(PBINS) - 1):
                    mask = (
                        (a >= ABINS[ai])
                        & (a < ABINS[ai + 1])
                        & (ph >= PBINS[pi])
                        & (ph < PBINS[pi + 1])
                    )
                    for j, seed in enumerate(d["seeds"]):
                        num = int(np.sum(harm[j] & acted[j] & mask))
                        den = int(np.sum(acted[j] & mask))
                        interval = ci(num, den)
                        binned.append(
                            dict(
                                cell=key,
                                seed=seed,
                                amp_lo=ABINS[ai],
                                amp_hi=ABINS[ai + 1],
                                phase_lo=PBINS[pi],
                                phase_hi=PBINS[pi + 1],
                                episodes=int(mask.sum()),
                                executed_episodes=den,
                                harmed_executed=num,
                                harm_given_execution=num / den if den else None,
                                pointwise_lower=interval[0] if interval else None,
                                pointwise_upper=interval[1] if interval else None,
                            )
                        )
    (HERE / "primary_replay.json").write_text(
        json.dumps(
            {
                "source": "saved primary episode IDs, applied actions and fidelities; simulator latent coordinates separately regenerated and F0...F10 checked",
                "cells": per_cell,
            },
            indent=2,
        )
        + "\n"
    )
    with (HERE / "primary_binned_harm.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=binned[0])
        w.writeheader()
        w.writerows(binned)
    print(len(per_cell), max(v["max_abs_replay_error"] for v in per_cell.values()))


if __name__ == "__main__":
    main()
