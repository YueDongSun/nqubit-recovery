"""Count-quality filtering and 1:1 state→snapshot resolution for measured data.

Supports two ingestion models:

* 4Q campaign mode (authoritative): the campaign manifest ledger
  (`CampaignManifestV1`) determines which snapshot each state points at.
  Outlier states are re-resolved to a same-plan clean re-collection, or
  dropped entirely.
* 3Q plan mode (no manifest): the 1000-state automation plan is the state
  list; each state is assigned the cleanest same-plan snapshot among all its
  snapshots, or dropped entirely.

Outlier detection is identical in both modes: a snapshot is bad when
`signal_total > 10x median(signal_total)` (laser unlock / elevated counts) OR
`idler_total < median(idler_total) / 10` (idler collapse).  Bad snapshots are
pure noise and carry no usable signal; excluded states are dropped with
reliability weight 0 and are not identified in the data itself.

Nothing here writes to manifest.json or any ledger; source of truth stays in
the data.  The report is a derived, disposable artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from system_control.experiment_control.data_protocol import (
    CampaignManifestV1,
    atomic_write_json,
)

# Default robust thresholds relative to the referenced-snapshot medians.
SIGNAL_ELEVATION_FACTOR = 10.0  # signal_total > 10x median(signal_total)
IDLER_COLLAPSE_FACTOR = 10.0  # idler_total < median(idler_total) / 10


def _count_totals(snapshot_dir: Path) -> Dict[str, float]:
    """Return per-snapshot signal/idler/coin totals from the count arrays."""
    totals: Dict[str, float] = {}
    for channel in ("signal", "idler", "coin"):
        array = np.load(snapshot_dir / f"{channel}_bins.npy", allow_pickle=False)
        totals[channel] = float(np.asarray(array, dtype=float).sum())
    return totals


def _phase_close(
    left: Sequence[float],
    right: Sequence[float],
    tolerance_deg: float = 1e-3,
) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) < tolerance_deg for a, b in zip(left, right))


def _state_phases(snapshot_dir: Path) -> Optional[Sequence[float]]:
    """Read target phases_deg from a snapshot's legacy metadata."""
    metadata_path = snapshot_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = (metadata.get("feedback_context") or {}).get("target_state_params") or {}
    return target.get("phases_deg")


def _index_all_snapshots(
    data_root: Path,
    suffix: str,
    plan_by_name: Mapping[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Index every snapshot under data_root (matching suffix) by claimed state name."""
    snapshots: list[Dict[str, Any]] = []
    for directory in sorted(data_root.glob(f"*_{suffix}")):
        if not (directory / "metadata.json").exists():
            continue
        state_name = None
        phases = None
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            target = (metadata.get("feedback_context") or {}).get("target_state_params") or {}
            state_name = target.get("name")
            phases = target.get("phases_deg")
        if not state_name:
            continue
        plan_state = plan_by_name.get(state_name)
        plan_match = bool(
            plan_state and phases and _phase_close(phases, plan_state.get("phases_deg") or [])
        )
        totals = _count_totals(directory)
        snapshots.append(
            {
                "dir": directory,
                "name": directory.name,
                "state_name": state_name,
                "phases_deg": phases,
                "plan_match": plan_match,
                "signal_total": totals["signal"],
                "idler_total": totals["idler"],
                "coin_total": totals["coin"],
            }
        )
    return snapshots


def _detect_outliers(
    referenced: Sequence[Dict[str, Any]],
    *,
    signal_factor: float = SIGNAL_ELEVATION_FACTOR,
    idler_factor: float = IDLER_COLLAPSE_FACTOR,
) -> tuple[np.ndarray, float, float]:
    """Return (bad_mask, signal_threshold, idler_threshold) for referenced snapshots."""
    signal = np.asarray([item["signal_total"] for item in referenced], dtype=float)
    idler = np.asarray([item["idler_total"] for item in referenced], dtype=float)
    signal_median = float(np.median(signal))
    idler_median = float(np.median(idler))
    signal_threshold = signal_factor * signal_median
    idler_threshold = idler_median / idler_factor
    bad = (signal > signal_threshold) | (idler < idler_threshold)
    return bad, signal_threshold, idler_threshold


def _pick_replacement(
    candidate: Dict[str, Any],
    options: Sequence[Dict[str, Any]],
    *,
    idler_threshold: float,
    signal_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Choose a same-plan, non-referenced, clean re-collection for a bad state.

    A substitute must come from the *same* phase plan as the campaign state
    (`plan_match`): re-collections collected under a different phase plan (e.g.
    the +90-degree-shifted 50-state recovery plan) are a physically different
    target and must not be substituted.  Candidates must also pass the same
    idler-collapse and signal-elevation gates used for detection, so a
    substitute is never itself pure noise.  Best candidate is the one with the
    most idler signal (least likely a weak/partial exposure), breaking ties
    toward the later timestamp.
    """
    usable = [
        item
        for item in options
        if item["plan_match"]
        and not item["dir"].resolve() == candidate["dir"].resolve()
        and item["signal_total"] <= signal_threshold
        and item["idler_total"] >= idler_threshold
    ]
    if not usable:
        return None
    usable.sort(key=lambda item: (item["idler_total"], item["name"]))
    return usable[-1]


def _build_resolution(
    candidates: Sequence[Dict[str, Any]],
    by_state: Mapping[str, Sequence[Dict[str, Any]]],
    *,
    signal_threshold: float,
    idler_threshold: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Resolve each candidate state into good / substituted / excluded.

    Returns (states_report, resolved_snapshots).  `candidates` are the
    authoritative per-state snapshots (from a manifest ledger in campaign mode,
    or the best-candidate list in plan mode).  `by_state` holds every snapshot
    on disk grouped by state name, used to find substitutions.
    """
    states_report: list[Dict[str, Any]] = []
    resolved: list[Dict[str, Any]] = []
    for item in candidates:
        base = {
            "state_index": item.get("state_index"),
            "state_name": item["state_name"],
            "state_role": item.get("state_role"),
            "signal_total": item["signal_total"],
            "idler_total": item["idler_total"],
        }
        if item["signal_total"] <= signal_threshold and item["idler_total"] >= idler_threshold:
            states_report.append({**base, "status": "good", "snapshot_dir": item["name"]})
            resolved.append(
                {"state_index": base["state_index"], "snapshot_dir": item["name"]}
            )
            continue

        options = list(by_state.get(item["state_name"], []))
        replacement = _pick_replacement(
            item,
            options,
            idler_threshold=idler_threshold,
            signal_threshold=signal_threshold,
        )
        if replacement is not None:
            states_report.append(
                {
                    **base,
                    "status": "substituted",
                    "snapshot_dir": replacement["name"],
                    "original_dir": item["name"],
                    "replacement_signal_total": replacement["signal_total"],
                    "replacement_idler_total": replacement["idler_total"],
                }
            )
            resolved.append(
                {"state_index": base["state_index"], "snapshot_dir": replacement["name"]}
            )
        else:
            states_report.append(
                {**base, "status": "excluded", "snapshot_dir": None}
            )
    return states_report, resolved


def _plan_by_name(plan_path: str | Path) -> Dict[str, Dict[str, Any]]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    return {state.get("name"): state for state in plan.get("states", [])}


def resolve_campaign(
    campaign_manifest_path: str | Path,
    data_root: str | Path,
    plan_path: str | Path,
    *,
    signal_factor: float = SIGNAL_ELEVATION_FACTOR,
    idler_factor: float = IDLER_COLLAPSE_FACTOR,
) -> Dict[str, Any]:
    """Build the clean 1:1 state→snapshot mapping for a manifest-led campaign.

    `campaign_manifest_path` is the authoritative ledger (e.g. the 4Q run
    `campaign_manifest.json`): each state points at its original snapshot, and
    outliers are re-resolved to a same-plan clean re-collection or excluded.
    """
    campaign = CampaignManifestV1.load(campaign_manifest_path)
    data_root = Path(data_root)
    plan_by_name = _plan_by_name(plan_path)
    suffix = "4qubit"

    all_snapshots = _index_all_snapshots(data_root, suffix, plan_by_name)
    by_state: Dict[str, list[Dict[str, Any]]] = {}
    for item in all_snapshots:
        by_state.setdefault(item["state_name"], []).append(item)

    referenced = []
    for ledger in campaign.states:
        snapshot_path = ledger.get("snapshot_path")
        if not snapshot_path:
            continue
        dir_name = Path(str(snapshot_path)).name
        directory = data_root / dir_name
        if not directory.is_dir():
            continue
        totals = _count_totals(directory)
        phases = _state_phases(directory)
        plan_state = plan_by_name.get(ledger.get("name"))
        plan_match = bool(
            plan_state and phases and _phase_close(phases, plan_state.get("phases_deg") or [])
        )
        referenced.append(
            {
                "state_index": ledger.get("state_index"),
                "state_name": ledger.get("name"),
                "state_role": ledger.get("state_role"),
                "dir": directory,
                "name": dir_name,
                "phases_deg": phases,
                "plan_match": plan_match,
                "signal_total": totals["signal"],
                "idler_total": totals["idler"],
                "coin_total": totals["coin"],
            }
        )

    bad, signal_threshold, idler_threshold = _detect_outliers(
        referenced, signal_factor=signal_factor, idler_factor=idler_factor
    )
    states_report, resolved = _build_resolution(
        referenced,
        by_state,
        signal_threshold=signal_threshold,
        idler_threshold=idler_threshold,
    )

    summary = {
        "mode": "campaign",
        "campaign_id": campaign.campaign_id,
        "system_type": "4qubit",
        "state_count": len(campaign.states),
        "referenced_count": len(referenced),
        "good_count": sum(1 for item in states_report if item["status"] == "good"),
        "substituted_count": sum(1 for item in states_report if item["status"] == "substituted"),
        "excluded_count": sum(1 for item in states_report if item["status"] == "excluded"),
        "signal_median": float(np.median([item["signal_total"] for item in referenced])),
        "idler_median": float(np.median([item["idler_total"] for item in referenced])),
        "signal_threshold": signal_threshold,
        "idler_threshold": idler_threshold,
        "signal_elevation_factor": signal_factor,
        "idler_collapse_factor": idler_factor,
    }

    return {
        "summary": summary,
        "states": states_report,
        "resolved_snapshots": resolved,
    }


def resolve_plan(
    plan_path: str | Path,
    data_root: str | Path,
    *,
    suffix: str = "3qubit",
    signal_factor: float = SIGNAL_ELEVATION_FACTOR,
    idler_factor: float = IDLER_COLLAPSE_FACTOR,
) -> Dict[str, Any]:
    """Build the clean 1:1 state→snapshot mapping for plan-only data (e.g. 3Q).

    With no campaign manifest, the automation plan is the authoritative state
    list.  Every snapshot on disk that matches a plan state is a candidate; the
    cleanest same-plan snapshot is chosen per state.  A state with no clean
    same-plan snapshot is excluded.
    """
    data_root = Path(data_root)
    plan_by_name = _plan_by_name(plan_path)
    all_snapshots = _index_all_snapshots(data_root, suffix, plan_by_name)

    # Robust thresholds over ALL snapshots (median is outlier-immune), so a
    # per-state choice is never skewed by its own laser-unlock duplicates.
    all_signal = np.asarray([item["signal_total"] for item in all_snapshots], dtype=float)
    all_idler = np.asarray([item["idler_total"] for item in all_snapshots], dtype=float)
    signal_threshold = signal_factor * float(np.median(all_signal))
    idler_threshold = float(np.median(all_idler)) / idler_factor

    # Per-state candidate selection: the cleanest same-plan snapshot, if one
    # exists; otherwise the plan-matching snapshot that is least far from the
    # clean envelope (used only so detection can label it bad).
    states_in_order = [state for state in plan_by_name.values()]
    candidates: list[Dict[str, Any]] = []
    for plan_state in states_in_order:
        state_name = plan_state.get("name")
        options = [
            item for item in all_snapshots
            if item["state_name"] == state_name and item["plan_match"]
        ]
        if not options:
            # No same-plan snapshot exists for this state: nothing on disk
            # targets this state, so it cannot be resolved.
            continue
        clean = [
            item
            for item in options
            if item["signal_total"] <= signal_threshold
            and item["idler_total"] >= idler_threshold
        ]
        if clean:
            # Most trustworthy exposure among clean same-plan candidates.
            candidates.append(max(clean, key=lambda item: (item["idler_total"], item["name"])))
            continue
        # No clean same-plan snapshot: keep the same-plan candidate with the
        # highest idler so detection can classify it and seek a substitute.
        candidates.append(max(options, key=lambda item: (item["idler_total"], item["name"])))

    by_state: Dict[str, list[Dict[str, Any]]] = {}
    for item in all_snapshots:
        by_state.setdefault(item["state_name"], []).append(item)

    # Give each candidate a plan-derived index.
    for index, item in enumerate(candidates, start=1):
        item["state_index"] = index
    states_report, resolved = _build_resolution(
        candidates,
        by_state,
        signal_threshold=signal_threshold,
        idler_threshold=idler_threshold,
    )

    summary = {
        "mode": "plan",
        "system_type": suffix,
        "plan_state_count": len(states_in_order),
        "candidate_count": len(candidates),
        "good_count": sum(1 for item in states_report if item["status"] == "good"),
        "substituted_count": sum(1 for item in states_report if item["status"] == "substituted"),
        "excluded_count": sum(1 for item in states_report if item["status"] == "excluded"),
        "missing_count": len(states_in_order) - len(candidates),
        "signal_median": float(np.median(all_signal)),
        "idler_median": float(np.median(all_idler)),
        "signal_threshold": signal_threshold,
        "idler_threshold": idler_threshold,
        "signal_elevation_factor": signal_factor,
        "idler_collapse_factor": idler_factor,
    }

    return {
        "summary": summary,
        "states": states_report,
        "resolved_snapshots": resolved,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("campaign", "plan"),
        default="campaign",
        help="campaign=manifest ledger (4Q); plan=automation plan as state list (3Q)",
    )
    parser.add_argument(
        "--campaign-manifest",
        help="Path to the campaign manifest.json (campaign mode)",
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to the automation plan JSON defining state phases",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing the snapshot dirs (default: data)",
    )
    parser.add_argument(
        "--suffix",
        default="3qubit",
        help="Snapshot dir suffix for plan mode (default: 3qubit)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the resolution report (state_mapping.json)",
    )
    parser.add_argument(
        "--signal-factor",
        type=float,
        default=SIGNAL_ELEVATION_FACTOR,
        help="signal_total > factor x median(signal_total) marks an outlier",
    )
    parser.add_argument(
        "--idler-factor",
        type=float,
        default=IDLER_COLLAPSE_FACTOR,
        help="idler_total < median(idler_total) / factor marks an outlier",
    )
    args = parser.parse_args(argv)

    if args.mode == "campaign":
        if not args.campaign_manifest:
            parser.error("--campaign-manifest is required in campaign mode")
        report = resolve_campaign(
            args.campaign_manifest,
            args.data_root,
            args.plan,
            signal_factor=args.signal_factor,
            idler_factor=args.idler_factor,
        )
    else:
        report = resolve_plan(
            args.plan,
            args.data_root,
            suffix=args.suffix,
            signal_factor=args.signal_factor,
            idler_factor=args.idler_factor,
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "state_mapping.json", report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
