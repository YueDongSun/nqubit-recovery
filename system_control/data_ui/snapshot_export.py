"""NumPy snapshot export helpers for data UI measurements."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


DEFAULT_ARRAY_FIELDS = (
    "signal_bins",
    "idler_bins",
    "coin_bins",
    "multi2_bins",
    "multi3_bins",
    "multi4_bins",
    "multi2_cycle",
    "multi3_cycle",
    "multi4_cycle",
    "segment_values",
    "coin_idler_values",
    "signed_segment_values",
    "legacy_clipped_segment_values",
    "signal_bins_raw",
    "idler_bins_raw",
    "coin_bins_raw",
    "signal_hist",
    "idler_hist",
    "rho_raw",
    "rho_calibrated",
    "rho_posterior_mean",
    "phase_probe_tensor",
    "count_jacobian_label",
)


def save_numpy_snapshot(
    current_data: Optional[Dict],
    system_type: str,
    output_root: Optional[str] = None,
    array_fields: Iterable[str] = DEFAULT_ARRAY_FIELDS,
) -> Optional[Path]:
    """Save current data-ui arrays before clearing counters.

    Returns None when there is no current data, and creates no directory.
    """

    if not current_data:
        return None

    root = Path(output_root) if output_root else Path(__file__).resolve().parents[2] / "data"
    root.mkdir(parents=True, exist_ok=True)

    timestamp = float(current_data.get("timestamp", time.time()))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    snapshot_dir = _unique_dir(root / f"{stamp}_{system_type}")
    staging_dir = root / f".{snapshot_dir.name}.incomplete_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        saved_arrays = []
        for field in array_fields:
            if field not in current_data or current_data[field] is None:
                continue
            np.save(staging_dir / f"{field}.npy", np.asarray(current_data[field]))
            saved_arrays.append(field)

        metadata = {
            "system_type": system_type,
            "timestamp": timestamp,
            "saved_arrays": saved_arrays,
        }
        for key, value in current_data.items():
            if key in saved_arrays:
                continue
            converted = _json_value(value)
            if converted is not None:
                metadata[key] = converted

        # The versioned manifest complements metadata.json; the latter remains
        # for historical scripts and UI compatibility.  Both are written into
        # a hidden staging directory and exposed only by the final atomic rename.
        from system_control.experiment_control.data_protocol import (
            atomic_write_json,
            build_snapshot_manifest,
            validate_snapshot_files,
        )

        manifest = build_snapshot_manifest(
            snapshot_dir=staging_dir,
            system_type=system_type,
            current_data=current_data,
            array_fields=saved_arrays,
        )
        metadata["schema_version"] = manifest.schema_version
        metadata["sample_id"] = manifest.sample_id
        atomic_write_json(staging_dir / "metadata.json", metadata)
        atomic_write_json(staging_dir / "manifest.json", manifest.to_json_dict())
        validate_snapshot_files(staging_dir, manifest)
        staging_dir.replace(snapshot_dir)
    except Exception:
        # Keep the hidden directory as crash evidence.  It is never considered
        # a completed snapshot and a retry receives a fresh UUID.
        raise

    return snapshot_dir


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique snapshot directory under {path.parent}")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            converted = _json_value(item)
            if converted is not None:
                out.append(converted)
        return out
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            converted = _json_value(item)
            if converted is not None:
                out[str(key)] = converted
        return out
    return str(value)
