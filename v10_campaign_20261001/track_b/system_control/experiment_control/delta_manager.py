"""Additive ML delta matrix handling for feedback calibration."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from system_control.qubit_recon_files.system_specs import SYSTEM_SPECS as _QUBIT_SPECS

from .models import CalibrationDelta


SYSTEM_SPECS = {
    "3qubit": {
        "suffix": "",
        "freq_steps": _QUBIT_SPECS["3qubit"]["freq_steps"],
        "groups_per_row": _QUBIT_SPECS["3qubit"]["groups_per_row"],
        "cycles": 1,
    },
    "4qubit": {
        "suffix": "_4qubit",
        "freq_steps": _QUBIT_SPECS["4qubit"]["freq_steps"],
        "groups_per_row": _QUBIT_SPECS["4qubit"]["groups_per_row"],
        "cycles": _QUBIT_SPECS["4qubit"]["total_cycles"],
    },
}


class DeltaManager:
    """Persist and apply ML-generated additive delta matrices."""

    def __init__(self, matrix_dir: str):
        self.matrix_dir = Path(matrix_dir)
        self.delta_dir = self.matrix_dir / "ml_delta"
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_root = self.delta_dir / "candidates"

    def delta_path(
        self,
        system_type: str,
        channel: str,
        matrix_type: str,
        cycle_index: Optional[int] = None,
    ) -> Path:
        spec = SYSTEM_SPECS[system_type]
        suffix = spec["suffix"]
        if cycle_index is not None:
            return self.delta_dir / f"{channel}_{matrix_type}_delta_{cycle_index}{suffix}.txt"
        return self.delta_dir / f"{channel}_{matrix_type}_delta{suffix}.txt"

    def expected_shape(
        self,
        system_type: str,
        channel: str,
        cycle_index: Optional[int] = None,
    ) -> tuple[int, int]:
        spec = SYSTEM_SPECS[system_type]
        freq_steps = spec["freq_steps"]
        groups = spec["groups_per_row"]
        y_channels = {"W_CH2", "W_CH4", "R_CH2", "R_CH4"}
        if channel in y_channels and cycle_index is None:
            return (freq_steps, groups)
        return (freq_steps, freq_steps)

    def load_delta(
        self,
        system_type: str,
        channel: str,
        matrix_type: str,
        cycle_index: Optional[int] = None,
        strict: bool = False,
    ) -> np.ndarray:
        path = self.delta_path(system_type, channel, matrix_type, cycle_index)
        shape = self.expected_shape(system_type, channel, cycle_index)
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Required delta matrix does not exist: {path}")
            return np.zeros(shape, dtype=float)
        try:
            data = np.loadtxt(path)
            if data.shape == shape:
                return data.astype(float)
        except Exception as exc:
            if strict:
                raise ValueError(f"Cannot parse delta matrix {path}: {exc}") from exc
        if strict:
            raise ValueError(f"Delta matrix {path} has shape {getattr(data, 'shape', None)}; expected {shape}")
        return np.zeros(shape, dtype=float)

    def save_delta(
        self,
        system_type: str,
        delta: CalibrationDelta,
        cycle_index: Optional[int] = None,
        accumulate: bool = True,
    ) -> CalibrationDelta:
        path = self.delta_path(system_type, delta.target_channel, delta.matrix_type, cycle_index)
        shape = self.expected_shape(system_type, delta.target_channel, cycle_index)
        update = np.asarray(delta.clipped_delta(), dtype=float)
        if update.shape != shape:
            raise ValueError(f"Delta shape {update.shape} does not match expected {shape} for {path.name}")

        if not delta.accepted:
            delta.path = str(path)
            return delta

        current = self.load_delta(system_type, delta.target_channel, delta.matrix_type, cycle_index)
        output = current + update if accumulate else update
        np.savetxt(path, output, fmt="%.9f")
        delta.path = str(path)
        return delta

    def save_deltas(
        self,
        system_type: str,
        deltas: Iterable[CalibrationDelta],
        accumulate: bool = True,
    ) -> List[CalibrationDelta]:
        saved: List[CalibrationDelta] = []
        spec = SYSTEM_SPECS[system_type]
        for delta in deltas:
            if system_type == "4qubit" and delta.target_channel in {"W_CH1", "R_CH1"}:
                if delta.cycle_index_zero_based is None:
                    cycles = range(1, spec["cycles"] + 1)
                else:
                    zero_based = int(delta.cycle_index_zero_based)
                    if not 0 <= zero_based < spec["cycles"]:
                        raise ValueError(f"cycle_index_zero_based must be in [0, {spec['cycles']})")
                    cycles = (zero_based + 1,)
                for cycle in cycles:
                    saved.append(self.save_delta(system_type, delta, cycle, accumulate))
            else:
                saved.append(self.save_delta(system_type, delta, None, accumulate))
        return saved

    def stage_candidate(
        self,
        system_type: str,
        deltas: Iterable[CalibrationDelta],
        *,
        accumulate: bool = True,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Write an isolated, hash-addressed candidate without changing active deltas."""

        candidate_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        root = self.candidate_root / candidate_id
        values_dir = root / "values"
        previous_dir = root / "previous"
        values_dir.mkdir(parents=True, exist_ok=False)
        previous_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        staged: List[CalibrationDelta] = []
        for delta, cycle_index in self._expanded_deltas(system_type, deltas):
            active = self.delta_path(system_type, delta.target_channel, delta.matrix_type, cycle_index)
            shape = self.expected_shape(system_type, delta.target_channel, cycle_index)
            update = np.asarray(delta.clipped_delta(), dtype=float)
            if update.shape != shape:
                raise ValueError(f"Delta shape {update.shape} does not match expected {shape} for {active.name}")

            candidate_path = values_dir / active.name
            current = self.load_delta(
                system_type,
                delta.target_channel,
                delta.matrix_type,
                cycle_index,
                strict=active.exists(),
            )
            output = current + update if accumulate else update
            np.savetxt(candidate_path, output, fmt="%.9f")
            previous_exists = active.exists()
            if previous_exists:
                shutil.copy2(active, previous_dir / active.name)
            digest = self._sha256(candidate_path)
            entries.append({
                "target_channel": delta.target_channel,
                "matrix_type": delta.matrix_type,
                "cycle_index": cycle_index,
                "active_path": str(active),
                "candidate_path": str(candidate_path),
                "candidate_sha256": digest,
                "previous_exists": previous_exists,
                "previous_path": str(previous_dir / active.name) if previous_exists else None,
                "previous_sha256": self._sha256(active) if previous_exists else None,
                "accepted": bool(delta.accepted),
                "reason": delta.reason,
                "limit": float(delta.limit),
            })
            staged_delta = CalibrationDelta(
                delta.target_channel,
                delta.matrix_type,
                update,
                delta.limit,
                accepted=delta.accepted,
                reason=delta.reason,
                path=str(candidate_path),
                cycle_index_zero_based=delta.cycle_index_zero_based,
            )
            staged.append(staged_delta)

        manifest = {
            "schema": "DeltaCandidateV1",
            "candidate_id": candidate_id,
            "system_type": system_type,
            "created_at": time.time(),
            "status": "candidate",
            "human_approved": False,
            "entries": entries,
            "metadata": metadata or {},
        }
        self._write_manifest(root / "manifest.json", manifest)
        return {"candidate_id": candidate_id, "manifest_path": str(root / "manifest.json"), "deltas": staged}

    def promote_candidate(self, candidate_id: str, *, human_approved: bool) -> Dict:
        """Atomically promote a staged candidate after explicit human approval."""

        if not human_approved:
            raise PermissionError("candidate promotion requires explicit human approval")
        manifest_path = self.candidate_root / candidate_id / "manifest.json"
        manifest = self._load_manifest(manifest_path)
        if manifest.get("status") != "candidate":
            raise ValueError(f"candidate {candidate_id} has status {manifest.get('status')!r}, expected 'candidate'")
        promoted_paths = []
        for entry in manifest.get("entries", []):
            candidate = Path(entry["candidate_path"])
            active = Path(entry["active_path"])
            if self._sha256(candidate) != entry["candidate_sha256"]:
                raise ValueError(f"candidate hash mismatch: {candidate}")
            active.parent.mkdir(parents=True, exist_ok=True)
            tmp = active.with_name(f".{active.name}.{candidate_id}.tmp")
            shutil.copy2(candidate, tmp)
            os.replace(tmp, active)
            promoted_paths.append(str(active))
        manifest["status"] = "promoted"
        manifest["human_approved"] = True
        manifest["promoted_at"] = time.time()
        self._write_manifest(manifest_path, manifest)
        return {"candidate_id": candidate_id, "status": "promoted", "active_paths": promoted_paths}

    def rollback_candidate(self, candidate_id: str) -> Dict:
        """Restore the exact pre-promotion active files (or remove newly created ones)."""

        manifest_path = self.candidate_root / candidate_id / "manifest.json"
        manifest = self._load_manifest(manifest_path)
        if manifest.get("status") != "promoted":
            raise ValueError("only a promoted candidate can be rolled back")
        restored = []
        for entry in manifest.get("entries", []):
            active = Path(entry["active_path"])
            if entry.get("previous_exists"):
                previous = Path(entry["previous_path"])
                if self._sha256(previous) != entry.get("previous_sha256"):
                    raise ValueError(f"rollback hash mismatch: {previous}")
                tmp = active.with_name(f".{active.name}.{candidate_id}.rollback.tmp")
                shutil.copy2(previous, tmp)
                os.replace(tmp, active)
            elif active.exists():
                active.unlink()
            restored.append(str(active))
        manifest["status"] = "rolled_back"
        manifest["rolled_back_at"] = time.time()
        self._write_manifest(manifest_path, manifest)
        return {"candidate_id": candidate_id, "status": "rolled_back", "restored_paths": restored}

    def guard_amp_deltas(
        self,
        system_type: str,
        deltas: Iterable[CalibrationDelta],
        retrieval_rate: Optional[float],
        drop_threshold: float = 0.02,
    ) -> List[CalibrationDelta]:
        baseline, baseline_mad, history = self._rolling_efficiency_baseline(system_type, retrieval_rate)
        guarded = []
        drop_detected = (
            retrieval_rate is not None
            and baseline > 0
            and retrieval_rate < baseline * (1.0 - drop_threshold)
        )

        for delta in deltas:
            if drop_detected and delta.matrix_type == "amp":
                guarded.append(CalibrationDelta(
                    target_channel=delta.target_channel,
                    matrix_type=delta.matrix_type,
                    delta=np.zeros_like(delta.delta),
                    limit=delta.limit,
                    accepted=False,
                    reason=(
                        f"Rejected amp delta: retrieval_rate={retrieval_rate:.6g} "
                        f"below rolling baseline={baseline:.6g} (MAD={baseline_mad:.3g}) "
                        f"by more than {drop_threshold:.1%}"
                    ),
                ))
            else:
                guarded.append(delta)
        if retrieval_rate is not None and not drop_detected:
            self._update_efficiency_history(system_type, history, float(retrieval_rate))
        return guarded

    def write_snapshot_metadata(self, system_type: str, payload: Dict) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(payload.get("timestamp", time.time())))
        path = self.delta_dir / f"snapshot_{system_type}_{stamp}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def _rolling_efficiency_baseline(
        self,
        system_type: str,
        retrieval_rate: Optional[float],
    ) -> tuple[float, float, List[float]]:
        path = self.delta_dir / f"efficiency_baseline_{system_type}.json"
        history: List[float] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload.get("history"), list):
                    history = [float(value) for value in payload["history"] if np.isfinite(float(value))]
                elif "retrieval_rate" in payload:  # v1 compatibility
                    history = [float(payload["retrieval_rate"])]
            except Exception:
                history = []
        if not history and retrieval_rate is not None:
            history = [float(retrieval_rate)]
        baseline = float(np.median(history)) if history else 0.0
        mad = float(np.median(np.abs(np.asarray(history) - baseline))) if history else 0.0
        return baseline, mad, history

    def _get_or_set_efficiency_baseline(self, system_type: str, retrieval_rate: Optional[float]) -> float:
        """Compatibility shim returning the v2 rolling median."""
        baseline, _mad, history = self._rolling_efficiency_baseline(system_type, retrieval_rate)
        if retrieval_rate is not None:
            self._update_efficiency_history(system_type, history, float(retrieval_rate))
        return baseline

    def _update_efficiency_history(self, system_type: str, history: List[float], retrieval_rate: float) -> None:
        path = self.delta_dir / f"efficiency_baseline_{system_type}.json"
        if not history or not np.isclose(history[-1], retrieval_rate):
            history = [*history, retrieval_rate]
        history = history[-20:]
        baseline = float(np.median(history)) if history else 0.0
        mad = float(np.median(np.abs(np.asarray(history) - baseline))) if history else 0.0
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "schema": "RollingRetrievalBaselineV2",
                "history": history,
                "window": 20,
                "median": baseline,
                "mad": mad,
                "timestamp": time.time(),
            }, handle, indent=2)

    def _expanded_deltas(self, system_type: str, deltas: Iterable[CalibrationDelta]):
        spec = SYSTEM_SPECS[system_type]
        for delta in deltas:
            if not delta.accepted:
                continue
            if system_type == "4qubit" and delta.target_channel in {"W_CH1", "R_CH1"}:
                if delta.cycle_index_zero_based is None:
                    for cycle in range(1, spec["cycles"] + 1):
                        yield delta, cycle
                else:
                    zero_based = int(delta.cycle_index_zero_based)
                    if not 0 <= zero_based < spec["cycles"]:
                        raise ValueError(f"cycle_index_zero_based must be in [0, {spec['cycles']})")
                    yield delta, zero_based + 1
            else:
                yield delta, None

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_manifest(path: Path, payload: Dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def _load_manifest(path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
