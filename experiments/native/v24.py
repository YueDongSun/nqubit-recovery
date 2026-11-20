"""Primary real-data loader: the filtered V2.4 build.

Layout under the build root (for example data/v2_4/real_permanent_r2/):

    observations/real/real_{3qubit|4qubit}_{split}.h5
    index/index.sqlite

Run and repeat ids live only in the sqlite index. READY, manifest, and shard
sha checks are ignored. Every row in this build is usable; a false
row_valid_mask is an error, not a silent drop.

Validation splits are held out. Probe splits are the 4Q anti-drift anchors
(whole snapshots). Signed-probe rows use per-row perturbed operation
descriptors when those are stored; otherwise they are marked and excluded
from nominal-operation fits. global_multi is stored and is never read as alpha.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np

from nqrec.conventions import READ_AOD_PHASE_SIGN, require_phase_sign
from nqrec.physics.hardware_basis import hardware_basis

K_FOR = {"3qubit": 75, "4qubit": 256}
N_FOR = {"3qubit": 3, "4qubit": 4}
HELD_OUT_SPLITS = frozenset(
    {"validation", "validation_target_family", "probe_validation"}
)
ANCHOR_SPLITS = frozenset({"probe_train", "probe_validation"})
TRAINING_ELIGIBLE_SPLITS = frozenset({"train", "train_target_family", "probe_train"})
_REQUIRED_DATASETS = (
    "sample_id",
    "signal_counts",
    "idler_counts",
    "coin_counts",
    "read_count_scalar",
    "four_cell_counts",
    "exposure_known_mask",
    "target_amplitudes_raw",
    "target_phases_raw",
    "target_mask",
    "row_valid_mask",
)
_REQUIRED_META = (
    "split",
    "system_type",
    "N",
    "flat_order",
)


def reconstructed_descriptors(n_modes: int, flat_order: list[str]) -> list[dict]:
    """Operation descriptors from the J18 basis when the shard stores none.

    The shard labels must equal the reconstructed labels in order. The basis
    carries the legacy read-AOD convention, phase_sign = -1.
    """
    basis = hardware_basis(n_modes)
    if list(flat_order) != basis["labels"]:
        raise ValueError(
            "shard flat_order does not match the reconstructed V2.4 basis (J18)"
        )
    descriptors = []
    for beta, theta in zip(basis["beta"], basis["theta"]):
        cells = [int(j) for j in np.flatnonzero(beta > 0)]
        descriptors.append(
            {
                "cells": cells,
                "amplitudes": [float(beta[j]) for j in cells],
                "phases": [float(theta[j]) for j in cells],
                "phase_unit": "rad",
            }
        )
    return descriptors


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_index(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT sample_id, session_id, repeat_group_id, cell_id, split, "
            "system_type, N, observation_shard, observation_row FROM samples"
        ).fetchall()
    finally:
        connection.close()
    table = {}
    for record in rows:
        sample_id = _text(record[0])
        if sample_id in table:
            raise ValueError(f"duplicate sample_id {sample_id} in {path}")
        table[sample_id] = {
            "sample_id": sample_id,
            "run_id": _text(record[1]),
            "trajectory_id": _text(record[2]),
            "cell_id": _text(record[3]),
            "split": _text(record[4]),
            "system_type": _text(record[5]),
            "N": int(record[6]),
            "observation_shard": _text(record[7]),
            "observation_row": int(record[8]),
        }
    if not table:
        raise ValueError(f"index {path} has no samples")
    return table


def _dataset(handle, name: str) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"missing dataset {name}")
    return np.asarray(handle[name])


def _sample_ids(array) -> list[str]:
    return [_text(value) for value in array]


def _counts(array, n_rows: int, k: int, name: str) -> np.ndarray:
    if array.shape != (n_rows, k):
        raise ValueError(f"{name} shape {array.shape}, expected ({n_rows}, {k})")
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise ValueError(f"{name} is non-finite")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(array == np.floor(array)):
            raise ValueError(f"{name} is not integer-valued")
        array = array.astype(np.int64)
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int64)
    else:
        raise ValueError(f"{name} has dtype {array.dtype}")
    if np.any(array < 0):
        raise ValueError(f"{name} has a negative count")
    return array


def _check_cells(read, signal, idler, coin, four) -> None:
    expected = np.stack(
        [read[:, None] - signal - idler + coin, signal - coin, idler - coin, coin],
        axis=-1,
    )
    if np.any(expected < 0):
        raise ValueError(
            "negative cell count: need C <= S, C <= I, and READ - S - I + C >= 0"
        )
    if np.any(coin > signal) or np.any(coin > idler):
        raise ValueError("C exceeds S or I")
    if four.shape != expected.shape:
        raise ValueError(
            f"four_cell_counts shape {four.shape}, expected {expected.shape}"
        )
    if not np.array_equal(np.asarray(four, dtype=np.int64), expected.astype(np.int64)):
        raise ValueError("four_cell_counts is inconsistent with S, I, C, and R")
    if not np.all(expected.sum(axis=-1) == read[:, None]):
        raise AssertionError("four cells do not sum to READ")


def _targets(amplitudes, phases, mask, n_modes: int):
    if amplitudes.shape != phases.shape or amplitudes.shape != mask.shape:
        raise ValueError("target amplitudes, phases, and mask have different shapes")
    if amplitudes.shape[1] < n_modes:
        raise ValueError(
            f"target width {amplitudes.shape[1]} is shorter than N={n_modes}"
        )
    kept_a = []
    kept_p = []
    for row_a, row_p, row_m in zip(amplitudes, phases, mask):
        chosen = np.asarray(row_m, dtype=bool)
        if int(chosen.sum()) != n_modes:
            raise ValueError(
                f"target_mask selects {int(chosen.sum())} entries, expected {n_modes}"
            )
        kept_a.append(np.asarray(row_a, dtype=np.float64)[chosen])
        kept_p.append(np.asarray(row_p, dtype=np.float64)[chosen])
    return np.stack(kept_a), np.stack(kept_p)


def _anchor_fields(split: str, probe_amplitude, perturbed_descriptors):
    is_anchor = split in ANCHOR_SPLITS
    if not is_anchor:
        return {
            "is_anchor": False,
            "probe_kind": "",
            "exclude_from_nominal_fits": False,
            "operation_source": "file",
        }
    if probe_amplitude is None:
        return {
            "is_anchor": True,
            "probe_kind": "unspecified",
            "exclude_from_nominal_fits": True,
            "operation_source": "file",
        }
    if float(probe_amplitude) == 0.0:
        return {
            "is_anchor": True,
            "probe_kind": "zero_probe",
            "exclude_from_nominal_fits": False,
            "operation_source": "file",
        }
    if perturbed_descriptors is None:
        return {
            "is_anchor": True,
            "probe_kind": "signed_probe",
            "exclude_from_nominal_fits": True,
            "operation_source": "missing",
        }
    return {
        "is_anchor": True,
        "probe_kind": "signed_probe",
        "exclude_from_nominal_fits": False,
        "operation_source": "perturbed",
    }


def load_shard(path: str | Path, index: dict) -> list[dict]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "metadata_json" not in handle.attrs:
            raise ValueError(f"{path.name} is missing the metadata_json attribute")
        meta = json.loads(_text(handle.attrs["metadata_json"]))
        for key in _REQUIRED_META:
            if key not in meta:
                raise ValueError(f"{path.name} metadata_json missing {key}")
        system = meta["system_type"]
        if system not in K_FOR:
            raise ValueError(f"unknown system_type {system!r}")
        n_modes = int(meta["N"])
        if n_modes != N_FOR[system]:
            raise ValueError(f"{system} metadata N={n_modes}, expected {N_FOR[system]}")
        split = meta["split"]
        k = K_FOR[system]
        flat_order = meta["flat_order"]
        if "operation_descriptors" in meta:
            descriptors = meta["operation_descriptors"]
            phase_sign = require_phase_sign(meta["operation_descriptor_phase_sign"])
            basis_source = "file"
        else:
            descriptors = reconstructed_descriptors(n_modes, flat_order)
            phase_sign = READ_AOD_PHASE_SIGN
            basis_source = "reconstructed_j18"
        if len(flat_order) != k or len(descriptors) != k:
            raise ValueError(
                f"{path.name} flat_order or operation_descriptors length is not {k}"
            )
        for name in _REQUIRED_DATASETS:
            if name not in handle:
                raise ValueError(f"{path.name} missing dataset {name}")
        sample_ids = _sample_ids(_dataset(handle, "sample_id"))
        n_rows = len(sample_ids)
        signal = _counts(_dataset(handle, "signal_counts"), n_rows, k, "signal_counts")
        idler = _counts(_dataset(handle, "idler_counts"), n_rows, k, "idler_counts")
        coin = _counts(_dataset(handle, "coin_counts"), n_rows, k, "coin_counts")
        read = _counts(
            _dataset(handle, "read_count_scalar").reshape(n_rows, 1),
            n_rows,
            1,
            "read_count_scalar",
        )[:, 0]
        four = _dataset(handle, "four_cell_counts")
        if four.shape != (n_rows, k, 4):
            raise ValueError(
                f"four_cell_counts shape {four.shape}, expected ({n_rows}, {k}, 4)"
            )
        four = four.astype(np.int64)
        _check_cells(read, signal, idler, coin, four)
        if signal.shape[1] not in (75, 256):
            raise ValueError(f"K={signal.shape[1]} is not 75 or 256")
        valid = np.asarray(_dataset(handle, "row_valid_mask")).astype(bool)
        if valid.shape != (n_rows,) or not np.all(valid):
            raise ValueError(
                f"{path.name} has a row with row_valid_mask false; V2.4 rows are all usable"
            )
        exposure_known = np.asarray(_dataset(handle, "exposure_known_mask")).astype(
            bool
        )
        if exposure_known.shape != (n_rows,):
            raise ValueError("exposure_known_mask has the wrong shape")
        if "exposure_s" not in handle:
            if np.any(exposure_known):
                raise ValueError(
                    f"{path.name} marks exposure known but has no exposure_s dataset"
                )
            exposure = None
        else:
            exposure = np.asarray(handle["exposure_s"], dtype=np.float64)
            if exposure.shape != (n_rows,):
                raise ValueError("exposure_s has the wrong shape")
            if np.any(exposure_known & (~np.isfinite(exposure) | (exposure <= 0))):
                raise ValueError("a known exposure_s entry is not positive and finite")
        amplitudes = np.asarray(
            _dataset(handle, "target_amplitudes_raw"), dtype=np.float64
        )
        phases = np.asarray(_dataset(handle, "target_phases_raw"), dtype=np.float64)
        mask = np.asarray(_dataset(handle, "target_mask"))
        target_a, target_p = _targets(amplitudes, phases, mask, n_modes)
        probe = (
            np.asarray(handle["phase_probe_amplitude_rad"], dtype=np.float64)
            if "phase_probe_amplitude_rad" in handle
            else None
        )
        if probe is not None and probe.shape != (n_rows,):
            raise ValueError("phase_probe_amplitude_rad has the wrong shape")
        perturbed = None
        if "perturbed_operation_descriptors_json" in handle:
            perturbed = [
                _text(value)
                for value in handle["perturbed_operation_descriptors_json"][:]
            ]
            if len(perturbed) != n_rows:
                raise ValueError(
                    "perturbed operation descriptors do not match the row count"
                )
        global_multi = (
            np.asarray(handle["global_multi"]) if "global_multi" in handle else None
        )
        eligible = (
            np.asarray(handle["training_eligible_mask"]).astype(bool)
            if "training_eligible_mask" in handle
            else None
        )
        if eligible is not None and eligible.shape != (n_rows,):
            raise ValueError("training_eligible_mask has the wrong shape")
    rows = []
    for i, sample_id in enumerate(sample_ids):
        if sample_id not in index:
            raise ValueError(f"sample {sample_id} is missing from the index")
        record = index[sample_id]
        if record["observation_row"] != i:
            raise ValueError(
                f"sample {sample_id} observation_row {record['observation_row']} != {i}"
            )
        if (
            record["split"] != split
            or record["system_type"] != system
            or record["N"] != n_modes
        ):
            raise ValueError(f"index row for {sample_id} does not match the shard")
        # The real index stores the shard path relative to the build root.
        if Path(record["observation_shard"]).name != path.name:
            raise ValueError(
                f"index shard for {sample_id} is {record['observation_shard']}, file is {path.name}"
            )
        anchor = _anchor_fields(
            split,
            None if probe is None else float(probe[i]),
            None if perturbed is None else perturbed[i],
        )
        if eligible is not None:
            expected_eligible = split in TRAINING_ELIGIBLE_SPLITS
            if bool(eligible[i]) != expected_eligible:
                raise ValueError(
                    f"training_eligible_mask disagrees with split {split} on {sample_id}"
                )
        descriptors_row = descriptors
        if anchor["operation_source"] == "perturbed":
            parsed = json.loads(perturbed[i])
            if len(parsed) != k:
                raise ValueError(
                    f"perturbed descriptors for {sample_id} do not have length {k}"
                )
            descriptors_row = parsed
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "held_out": split in HELD_OUT_SPLITS,
                "system_type": system,
                "n_modes": n_modes,
                "k": k,
                "phase_sign": phase_sign,
                "flat_order": flat_order,
                "operation_descriptors": descriptors_row,
                "basis_source": basis_source,
                "run_id": record["run_id"],
                "trajectory_id": record["trajectory_id"],
                "cell_id": record["cell_id"],
                "signal": signal[i],
                "idler": idler[i],
                "coin": coin[i],
                "read_count": int(read[i]),
                "four_cell_counts": four[i],
                "exposure_unknown": not bool(exposure_known[i]),
                "exposure_s": None
                if (exposure is None or not bool(exposure_known[i]))
                else float(exposure[i]),
                "target_amplitudes": target_a[i],
                "target_phases": target_p[i],
                "global_multi": None if global_multi is None else global_multi[i],
                **anchor,
            }
        )
    return rows


def discover_shards(root: str | Path) -> list[Path]:
    root = Path(root)
    directory = root / "observations" / "real"
    if not directory.is_dir():
        raise FileNotFoundError(f"no observations/real under {root}")
    files = sorted(directory.glob("real_*.h5"))
    if not files:
        raise FileNotFoundError(f"no real_*.h5 under {directory}")
    return files


def load_v24(root: str | Path) -> list[dict]:
    root = Path(root)
    index = load_index(root / "index" / "index.sqlite")
    rows = []
    for path in discover_shards(root):
        rows.extend(load_shard(path, index))
    if not rows:
        raise ValueError(f"no rows under {root}")
    return rows


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Load a filtered V2.4 real-data build")
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    rows = load_v24(args.root)
    held = sum(row["held_out"] for row in rows)
    anchors = sum(row["is_anchor"] for row in rows)
    print(f"rows={len(rows)} held_out={held} anchors={anchors}")


if __name__ == "__main__":
    main()
