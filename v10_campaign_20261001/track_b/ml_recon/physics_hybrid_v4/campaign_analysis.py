"""Offline drift selection and paired-probe Jacobian/Fisher labels for V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from system_control.experiment_control.data_protocol import (
    CampaignManifestV1,
    atomic_write_json,
    load_snapshot_manifest,
)

from .drift import DRIFT_CANDIDATES, TEMPERATURE_MODES, fit_drift_candidates




def _completed_anchor_records(campaign: CampaignManifestV1) -> list[Dict[str, Any]]:
    records = []
    for ledger in campaign.states:
        if ledger.get("status") != "completed" or ledger.get("state_role") != "anchor":
            continue
        snapshot_path = ledger.get("snapshot_path")
        if not snapshot_path:
            continue
        records.append(
            {
                **ledger,
                "snapshot_path": Path(snapshot_path),
                "phase_probe": dict(ledger.get("phase_probe") or {}),
            }
        )
    return records


def _count_matrix(snapshot_path: Path) -> tuple[np.ndarray, float]:
    manifest = load_snapshot_manifest(snapshot_path)
    arrays = []
    for name in ("signal_bins", "idler_bins", "coin_bins"):
        record = manifest.arrays.get(name)
        if not record:
            raise ValueError(f"{snapshot_path} is missing {name}")
        arrays.append(
            np.asarray(
                np.load(snapshot_path / record["path"], allow_pickle=False),
                dtype=float,
            ).reshape(-1)
        )
    # Global/cycle multi-click totals remain in their own likelihood and are
    # never expanded into operation-local pseudo-counts.
    return np.stack(arrays, axis=-1), max(float(manifest.duration_s), 1e-6)


def _fit_campaign_drift(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    time_values = []
    phase_values = []
    temperature_values = []
    temperature_complete = True
    for record in records:
        manifest = load_snapshot_manifest(record["snapshot_path"])
        density = _load_density(record["snapshot_path"], manifest)
        if density is None:
            continue
        state = np.linalg.eigh(0.5 * (density + density.conj().T))[1][:, -1]
        phase = np.angle(state) - np.angle(state[0])
        phase_values.append(np.angle(np.exp(1j * phase)))
        time_values.append(float(manifest.ended_at))
        environment = dict(getattr(manifest, "environment", {}) or {})
        temperature = environment.get("temperature_c")
        if temperature is None:
            temperature_complete = False
            temperature_values.append(0.0)
        else:
            temperature_values.append(float(temperature))
    if len(time_values) < 8:
        return {
            "executed": False,
            "reason": "at least eight completed anchors with rho are required",
            "candidates": list(DRIFT_CANDIDATES),
        }
    order = np.argsort(time_values)
    time_array = np.asarray(time_values, dtype=float)[order]
    time_array -= time_array[0]
    phase_array = np.stack(phase_values, axis=0)[order]
    block_id = np.arange(time_array.size, dtype=int) // 4
    modes = TEMPERATURE_MODES if temperature_complete else ("time_only",)
    result = fit_drift_candidates(
        time_array,
        phase_array,
        block_id,
        temperature_c=np.asarray(temperature_values, dtype=float)[order],
        candidates=DRIFT_CANDIDATES,
        temperature_modes=modes,
        seed=20260717,
    )
    payload = result.to_json_dict()
    payload.update(
        {
            "executed": True,
            "temperature_complete": temperature_complete,
            "temperature_shuffled_negative_control_executed": temperature_complete,
            "precision": "float64",
        }
    )
    return payload


def _load_density(snapshot_path: Path, manifest) -> Optional[np.ndarray]:
    for key in ("posterior_mean", "calibrated", "raw"):
        relative = (manifest.reconstruction or {}).get(key)
        if relative and (snapshot_path / relative).exists():
            density = np.asarray(
                np.load(snapshot_path / relative, allow_pickle=False), dtype=complex
            )
            if density.shape[0] >= 16:
                indices = [8, 4, 2, 1]
                density = density[np.ix_(indices, indices)]
            return density[:4, :4]
    return None


def analyze_campaign(
    campaign_manifest_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Run drift selection plus anchor-joint six-direction identification."""
    campaign = CampaignManifestV1.load(campaign_manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _completed_anchor_records(campaign)
    drift_report = _fit_campaign_drift(records)
    atomic_write_json(output / "drift_selection.json", drift_report)
    from ml_recon.physics_hybrid_v5.campaign_probe_analysis import (
        analyze_joint_probe_records,
    )
    return analyze_joint_probe_records(campaign, output, records, drift_report=drift_report)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            analyze_campaign(args.campaign_manifest, args.output_dir),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
