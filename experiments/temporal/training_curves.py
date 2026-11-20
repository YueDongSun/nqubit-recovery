"""Export each realized feedback/control update block and its simulated training exposure."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
ARMS = ("feedback", "current", "summary", "ordered", "learner")
SEEDS = (11031, 11047, 11071)


def collect(root: Path) -> int:
    rows = []
    for lane in ("b0", "b2"):
        for seed in SEEDS:
            home = root / "fits" / lane / f"seed_{seed}"
            records = [
                json.loads(s) for s in (home / "blocks.jsonl").read_text().splitlines()
            ]
            # An interrupted append can leave an earlier attempt for one block. Keep the latest committed attempt.
            by_block = {(r["core_blocks"], r["adapter_blocks"]): r for r in records}
            previous = 0
            for (core, adapter), row in sorted(by_block.items()):
                step = core + adapter
                if (
                    step != previous + 1
                    or not 0 < core <= 128
                    or not 0 <= adapter <= 256
                    or (adapter and core != 128)
                ):
                    raise ValueError(
                        f"Missing or inconsistent block: {home}, {core}, {adapter}"
                    )
                previous = step
                expected = set(ARMS[1:] if adapter else ARMS[:1])
                loss = row["loss"]
                if set(loss) != expected or not all(
                    math.isfinite(v) for v in loss.values()
                ):
                    raise ValueError(f"Invalid logged action loss: {home}, {step}")
                rows.append(
                    dict(
                        lane=lane,
                        seed=seed,
                        phase="adapters" if adapter else "core",
                        block=step,
                        core_blocks=core,
                        adapter_blocks=adapter,
                        **{f"{arm}_block_mean_loss": loss.get(arm) for arm in ARMS},
                        **{
                            key: row[key]
                            for key in (
                                "core_actual_updates",
                                "updates_per_adapter",
                                "on_policy_blocks",
                                "teacher_stream_episodes",
                                "additional_on_policy_episodes",
                                "unique_training_reads",
                                "per_adapter_training_reads",
                                "core_training_reads",
                                "seconds",
                                "wall_seconds",
                            )
                        },
                    )
                )
    with NamedTemporaryFile(dir=root, mode="w", newline="", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(stream.name)
    tmp.replace(root / "training_curves.csv")
    return len(rows)


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    print(f"Returned {collect(cli.parse_args().root)} measured training-block rows.")
