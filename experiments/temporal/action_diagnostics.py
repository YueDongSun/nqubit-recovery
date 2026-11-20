"""Describe harms and bounded movements in returned Fig. 5 episodes, conditional on fixed policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


ROOT = Path(__file__).resolve().parent
ACTION_TOL = 1e-8
HARM_TOL = 1e-10
AMPLITUDE_BOUND = 0.35
PHASE_BOUND = 0.70


def summarize_actions(
    actions: np.ndarray, fidelity: np.ndarray, noop: np.ndarray
) -> dict:
    """One trajectory contributes once to its conditional harm denominator after any executed move."""
    episodes, windows, _, _ = actions.shape
    if fidelity.shape != noop.shape or fidelity.shape != (episodes, windows + 1):
        raise ValueError("inconsistent trajectory/action window counts")
    if not (
        np.isfinite(actions).all()
        and np.isfinite(fidelity).all()
        and np.isfinite(noop).all()
    ):
        raise ValueError("nonfinite trajectory")
    if not np.array_equal(fidelity[:, 0], noop[:, 0]):
        raise ValueError("NOOP and policy initial states are not paired")
    scale = np.array([AMPLITUDE_BOUND, PHASE_BOUND])
    scaled = actions / scale
    radius = np.max(np.abs(scaled), axis=(2, 3))
    if np.any(radius > 1 + 1e-7):
        raise ValueError("returned command exceeded a registered limit")
    moved = np.max(np.abs(actions), axis=(2, 3)) > ACTION_TOL
    acted = moved.any(axis=1)
    terminal_gain = fidelity[:, -1] - noop[:, -1]
    harms = terminal_gain < -HARM_TOL
    previous = np.zeros_like(scaled[:, 0])
    has_previous = np.zeros(episodes, dtype=bool)
    turns, reversed_turns = 0, 0
    for t in range(windows):
        active = moved[:, t]
        paired = active & has_previous
        product = (previous * scaled[:, t]).sum(axis=(1, 2))
        turns += int(paired.sum())
        reversed_turns += int(np.count_nonzero(paired & (product < 0)))
        previous[active] = scaled[active, t]
        has_previous |= active
    executed = int(acted.sum())
    active_slots = int(moved.sum())
    harmed_executed = int(np.count_nonzero(acted & harms))
    saturated = int(np.count_nonzero(moved & (radius >= 1.0 - 1e-7)))
    return dict(
        episodes=episodes,
        executed_episodes=executed,
        episode_execution_coverage=executed / episodes,
        harmed_episodes=int(harms.sum()),
        harm_vs_paired_noop=float(harms.mean()),
        harmed_executed_episodes=harmed_executed,
        harm_given_execution=fraction_if_observed(harmed_executed, executed),
        mean_gain_given_execution=float(terminal_gain[acted].mean())
        if executed
        else None,
        slots=moved.size,
        active_slots=active_slots,
        slot_execution_coverage=active_slots / moved.size,
        saturated_active_slots=saturated,
        saturation_given_active=fraction_if_observed(saturated, active_slots),
        consecutive_active_pairs=turns,
        reversed_active_pairs=reversed_turns,
        reversal_given_pair=fraction_if_observed(reversed_turns, turns),
    )


def fraction_if_observed(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def write_cell(home: Path) -> int:
    result = json.loads((home / "result.json").read_text())
    frozen = json.loads((home / "freeze.json").read_text())
    identity = frozen["identity"]
    if (
        result["identity"]["identity"] != identity
        or result["cell"] != frozen["plan"]["cell"]
    ):
        raise ValueError("Fig. 5 result/DEV binding changed")
    with np.load(home / "episode_panels.npz", allow_pickle=False) as panels:
        ids = panels["episode_ids"]
        if (
            len(np.unique(ids)) != identity["confirm_episodes"]
            or len(ids) != 512
            or panels["seeds"].tolist() != identity["shared_seed_panel"]
            or int(panels["paid_reads"]) != 77000
        ):
            raise ValueError(
                "unexpected independent episode, training-seed, or READ count"
            )
        rows = []
        for arm, published in sorted(result["arms"].items()):
            metrics = summarize_actions(
                panels[f"{arm}_executed"],
                panels[f"{arm}_fidelity"],
                panels["noop_fidelity"],
            )
            if (
                metrics["active_slots"] != published["n_active_slots"]
                or metrics["episodes"] != published["n_episodes"]
                or not np.isclose(
                    metrics["harm_vs_paired_noop"],
                    published["harm_vs_noop"],
                    rtol=0,
                    atol=1e-10,
                )
            ):
                raise ValueError(f"{arm} disagrees with the locked result")
            rows.append(
                {
                    "n": identity["n"],
                    "lane": identity["lane"],
                    "condition": identity["slot"],
                    "arm": arm,
                    **metrics,
                }
            )
    with NamedTemporaryFile(dir=home, mode="w", newline="", delete=False) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        dest = Path(fp.name)
    dest.replace(home / "action_diagnostics.csv")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    for n in (3, 4):
        for lane in ("b0", "b2"):
            for slot in ("nominal", "weak_static", "strong_drift"):
                cell = root / f"n{n}" / lane / slot
                if (cell / "result.json").is_file() and (
                    cell / "episode_panels.npz"
                ).is_file():
                    print(cell, write_cell(cell))
