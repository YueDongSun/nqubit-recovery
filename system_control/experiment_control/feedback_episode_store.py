"""Atomic persistence and strict resume validation for V6 feedback episodes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .feedback_episode import FeedbackEpisodeV1, FeedbackIterationV2


class FeedbackEpisodeStore:
    """Store each episode and iteration as independently recoverable JSON."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, episode: FeedbackEpisodeV1) -> Path:
        directory = self.episode_dir(episode.episode_id)
        if directory.exists():
            raise FileExistsError(f"feedback episode already exists: {episode.episode_id}")
        (directory / "iterations").mkdir(parents=True)
        self._atomic_write_json(directory / "episode.json", episode.to_json_dict())
        self._atomic_write_json(
            directory / "status_history.json",
            {
                "schema": "FeedbackEpisodeStatusHistoryV1",
                "episode_id": episode.episode_id,
                "events": [{"status": episode.status, "at": episode.created_at}],
            },
        )
        return directory

    def save(self, episode: FeedbackEpisodeV1) -> Path:
        path = self.episode_dir(episode.episode_id) / "episode.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        self._atomic_write_json(path, episode.to_json_dict())
        return path

    def load(self, episode_id: str) -> FeedbackEpisodeV1:
        path = self.episode_dir(episode_id) / "episode.json"
        return FeedbackEpisodeV1.from_json_dict(self._read_json(path))

    def resume(
        self,
        episode_id: str,
        *,
        target_hash: str,
        model_checkpoint_hash: str,
        detector_calibration_hash: str,
        bundle_store: Any,
    ) -> FeedbackEpisodeV1:
        episode = self.load(episode_id)
        expected = {
            "target_hash": str(target_hash),
            "model_checkpoint_hash": str(model_checkpoint_hash),
            "detector_calibration_hash": str(detector_calibration_hash),
        }
        for field_name, value in expected.items():
            if str(getattr(episode, field_name)) != value:
                raise ValueError(f"cannot resume episode with changed {field_name}")
        iterations = self.list_iterations(episode_id)
        if iterations and iterations[-1].iteration_index != episode.active_iteration:
            raise ValueError("episode active_iteration does not match durable iteration ledger")
        for iteration in iterations:
            if iteration.parent_accepted_iteration >= iteration.iteration_index:
                raise ValueError("episode contains an invalid accepted parent chain")
            parent = bundle_store.load_manifest(iteration.before_bundle_id)
            if parent.bundle_hash != iteration.before_bundle_hash:
                raise ValueError("episode parent bundle hash changed")
            if parent.target_hash != episode.target_hash:
                raise ValueError("episode parent bundle changed the target")
            if iteration.candidate_bundle_id is not None:
                candidate = bundle_store.load_manifest(iteration.candidate_bundle_id)
                if candidate.bundle_hash != iteration.candidate_bundle_hash:
                    raise ValueError("episode candidate bundle hash changed")
                if candidate.parent_bundle_id != iteration.before_bundle_id:
                    raise ValueError("rejected or unrelated bundle entered the parent chain")
        pointer = bundle_store.active_pointer()
        if episode.active_bundle_id and (
            pointer is None
            or pointer.get("bundle_id") != episode.active_bundle_id
            or pointer.get("bundle_hash") != episode.active_bundle_hash
        ):
            raise ValueError("active command pointer differs from the episode ledger")
        return episode

    def save_iteration(
        self, episode_id: str, iteration: FeedbackIterationV2,
    ) -> Path:
        self.load(episode_id)
        path = self._iteration_path(episode_id, iteration.iteration_index)
        if path.is_file():
            current = FeedbackIterationV2.from_json_dict(self._read_json(path))
            immutable_fields = (
                "iteration_index",
                "parent_accepted_iteration",
                "before_snapshot_id",
                "before_snapshot_hash",
                "before_bundle_id",
                "before_bundle_hash",
            )
            for field_name in immutable_fields:
                if getattr(current, field_name) != getattr(iteration, field_name):
                    raise ValueError(f"iteration immutable field changed: {field_name}")
        self._atomic_write_json(path, iteration.to_json_dict())
        return path

    def load_iteration(self, episode_id: str, iteration_index: int) -> FeedbackIterationV2:
        return FeedbackIterationV2.from_json_dict(
            self._read_json(self._iteration_path(episode_id, iteration_index))
        )

    def list_iterations(self, episode_id: str) -> list[FeedbackIterationV2]:
        directory = self.episode_dir(episode_id) / "iterations"
        if not directory.is_dir():
            return []
        return [
            FeedbackIterationV2.from_json_dict(self._read_json(path))
            for path in sorted(directory.glob("iteration_*.json"))
        ]

    def record_transition(
        self,
        episode_id: str,
        *,
        previous_status: str,
        next_status: str,
        details: Mapping[str, Any],
    ) -> None:
        path = self.episode_dir(episode_id) / "status_history.json"
        record = self._read_json(path)
        events = list(record.get("events") or [])
        events.append({
            "previous_status": str(previous_status),
            "status": str(next_status),
            "at": time.time(),
            "details": dict(details),
        })
        self._atomic_write_json(path, {**record, "events": events})

    def status_history(self, episode_id: str) -> Dict[str, Any]:
        return dict(self._read_json(self.episode_dir(episode_id) / "status_history.json"))

    @staticmethod
    def bundle_status_record(bundle_path: str | Path) -> Dict[str, Any]:
        return dict(FeedbackEpisodeStore._read_json(Path(bundle_path) / "bundle_status.json"))

    def episode_dir(self, episode_id: str) -> Path:
        if not str(episode_id).startswith("episode_"):
            raise ValueError("episode_id must start with episode_")
        path = (self.root_dir / str(episode_id)).resolve()
        if self.root_dir not in path.parents:
            raise ValueError("episode path escapes the configured root")
        return path

    def _iteration_path(self, episode_id: str, iteration_index: int) -> Path:
        if int(iteration_index) < 1 or int(iteration_index) > 10:
            raise ValueError("iteration_index must be in 1..10")
        return self.episode_dir(episode_id) / "iterations" / f"iteration_{int(iteration_index):03d}.json"

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8",
        )
        temporary.replace(path)


__all__ = ["FeedbackEpisodeStore"]
