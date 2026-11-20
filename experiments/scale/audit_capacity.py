"""Reconstruct capacity-matched one-step outcomes from the paired public counts and original actions."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np
from .audit_raw import error_size, fid


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    base, root = args.primary, args.root
    reg = json.loads((root / "protocol.json").read_text())
    total = 0
    max_error = 0.0
    for cell in reg["cells"]:
        tag = cell["tag"]
        with np.load(base / "arrays" / (tag + ".npz")) as panel:
            x = panel["initial_coords"]
            f0 = fid(x)
            a0, p0 = error_size(x)
            for tier in ("C-S", "C-L", "C-XXL"):
                suffix = tag + "_" + tier.lower()
                with gzip.open(root / "rows" / (suffix + ".csv.gz"), "rt") as stream:
                    rows = sorted(
                        csv.DictReader(stream), key=lambda r: int(r["parent_id"])
                    )
                if len(rows) != 128 or [int(r["parent_id"]) for r in rows] != list(
                    range(128)
                ):
                    raise ValueError("unpaired parent")
                if tier == "C-S":
                    action = panel["learned_plugin_actions"][:, 0]
                    terminal = panel["learned_plugin_terminal_counts"]
                else:
                    with np.load(root / "arrays" / (suffix + ".npz")) as material:
                        action = material["actions"]
                        terminal = material["terminal_counts"]
                np.testing.assert_allclose(
                    a0, [float(r["amp_rms0"]) for r in rows], atol=1e-12
                )
                np.testing.assert_allclose(
                    p0, [float(r["phase_rms0"]) for r in rows], atol=1e-12
                )
                np.testing.assert_allclose(
                    f0, [float(r["F0"]) for r in rows], atol=1e-12
                )
                ft = fid(x + action)
                error = np.max(abs(ft - np.array([float(r["FT"]) for r in rows])))
                if error > 1e-7:
                    raise ArithmeticError("terminal state does not match actual action")
                max_error = max(max_error, float(error))
                aa, pp = error_size(x + action)
                np.testing.assert_allclose(
                    aa, [float(r["amp_rmsT"]) for r in rows], atol=1.5e-7, rtol=0
                )
                np.testing.assert_allclose(
                    pp, [float(r["phase_rmsT"]) for r in rows], atol=1.5e-7, rtol=0
                )
                np.testing.assert_array_equal(
                    terminal.sum(1),
                    [
                        [int(r["terminal_" + k]) for k in ("00", "10", "01", "11")]
                        for r in rows
                    ],
                )
                assert np.all(terminal.sum((1, 2)) == 7000) and np.all(
                    panel["initial_counts"].sum((1, 2)) == 70000
                )
                total += len(rows)
    report = dict(
        verified=True,
        total_rows=total,
        paired_input_panels=len(reg["cells"]),
        capacities=3,
        maximum_reconstructed_overlap_error=max_error,
    )
    (root / "raw_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
