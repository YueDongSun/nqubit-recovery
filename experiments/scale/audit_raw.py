"""Reconstruct decision counts, executed actions and endpoint rows from the returned native arrays."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np


def error_size(x):
    amp = x[..., 0]
    p = x[..., 1]
    n = x.shape[-2]
    i, j = np.triu_indices(n, 1)
    wrapped = np.angle(np.exp(1j * (p[:, i] - p[:, j])))
    return np.sqrt(np.mean((amp - amp.mean(1, keepdims=True)) ** 2, axis=1)), np.sqrt(
        np.mean(wrapped**2, 1) * (n - 1) / (2 * n)
    )


def fid(x):
    v = np.exp(x[..., 0] + 1j * x[..., 1])
    return np.abs(np.sum(v, axis=1)) ** 2 / (
        x.shape[1] * np.sum(np.abs(v) ** 2, axis=1)
    )


def audit(folder, budget=False):
    spec = json.loads((folder / "protocol.json").read_text())
    cases = spec["cases" if budget else "cells"]
    maximum = 0.0
    records = 0
    terminal_reads = 0
    for item in cases:
        tag = item["tag"]
        k = item["K"]
        with gzip.open(folder / "rows" / (tag + ".csv.gz"), "rt") as stream:
            rows = list(csv.DictReader(stream))
        with np.load(folder / "arrays" / (tag + ".npz")) as arrays:
            x = arrays["initial_coords"]
            parents = arrays["parent_ids"]
            ex = arrays["decision_exposure"]
            end_ex = arrays["terminal_exposure"]
            np.testing.assert_array_equal(parents, np.arange(128))
            assert np.all(arrays["initial_counts"].sum(-1) == ex)
            aa, pp = error_size(x)
            np.testing.assert_allclose(aa, item["A"], atol=1e-12)
            np.testing.assert_allclose(pp, item["P"], atol=1e-12)
            for policy in sorted({r["policy"] for r in rows}):
                subset = sorted(
                    [r for r in rows if r["policy"] == policy],
                    key=lambda r: int(r["parent_id"]),
                )
                if [int(r["parent_id"]) for r in subset] != list(range(len(x))):
                    raise ValueError("unpaired parent row")
                actions = (
                    arrays[policy + "_action"][:, None]
                    if budget
                    else arrays[policy + "_actions"]
                )
                if not np.all(
                    np.max(np.abs(actions), axis=(0, 1, 2))
                    <= np.array([0.35, 0.70]) + 3e-7
                ):
                    raise ValueError("invalid action bound")
                state = x.copy()
                history = [fid(state)]
                for t in range(k):
                    if t > 0 and not np.all(
                        arrays[policy + "_decision_" + str(t)].sum(-1) == ex
                    ):
                        raise ValueError("incorrect decision count table")
                    state = state + actions[:, t]
                    history.append(fid(state))
                if not budget:
                    err = float(
                        abs(
                            np.stack(history, axis=1) - arrays[policy + "_fidelity"]
                        ).max()
                    )
                    maximum = max(maximum, err)
                    assert err < 1.5e-7
                end = arrays[policy + "_terminal_counts"]
                if not np.all(end.sum(-1) == end_ex):
                    raise ValueError("terminal exposures disagree")
                observed = np.array(
                    [
                        [int(r["terminal_" + c]) for c in ("00", "10", "01", "11")]
                        for r in subset
                    ]
                )
                np.testing.assert_array_equal(observed, end.sum(1))
                terminal_reads += int(observed.sum())
                np.testing.assert_allclose(
                    history[-1], [float(r["FT"]) for r in subset], rtol=0, atol=1.5e-7
                )
                aa, pp = error_size(state)
                np.testing.assert_allclose(
                    aa, [float(r["amp_rmsT"]) for r in subset], atol=1.5e-7, rtol=0
                )
                np.testing.assert_allclose(
                    pp, [float(r["phase_rmsT"]) for r in subset], atol=1.5e-7, rtol=0
                )
                records += len(subset)
    out = dict(
        verified_cells=len(cases),
        verified_parent_policy_rows=records,
        terminal_read_sum=terminal_reads,
        max_reconstructed_fidelity_error=maximum,
        action_storage_dtype="float32",
        verified=True,
    )
    (folder / "raw_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--budget", action="store_true")
    args = parser.parse_args()
    print(audit(args.root, args.budget), flush=True)
