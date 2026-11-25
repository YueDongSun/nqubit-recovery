"""Observation-only Track B history view for the bounded V2.4 pilot.

The generic V2.4 feedback loader intentionally assembles every feedback
target and computes the deployable analytic action.  The inverse-transformer
pilot consumes only the five history/measurement fields defined here, so this
view keeps the same source identity and HDF5 handle ownership while avoiding
truth-shard reads and derived action computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ml_recon.data_v2_4.loaders import V24TrainingDataset, _to_tensor


PILOT_TASK = "feedback"
PILOT_ARM = "A0_RAW"
PILOT_FAMILY = "feedback_history"
PILOT_DOMAIN = "global_pauli_v24"
PILOT_SPLITS = frozenset({"train", "validation"})
PILOT_HISTORY_ROUNDS = 10
PILOT_OBSERVATION_FIELDS = (
    "history_mask",
    "history_tokens",
    "mode_mask",
    "noisy_measurement_tokens",
    "target_mode_tokens",
)


class Pilot_history_dataset(V24TrainingDataset):
    """Narrow A0 raw Track B history reader with no truth-shard access."""

    def __init__(self, dataset_root: str | Path, *, build_id: str, cell_id: str,
                 split: str, task: str = PILOT_TASK, arm: str = PILOT_ARM,
                 derived_checkpoint_sha: str = "",
                 require_frozen_classifier_cache: bool = False) -> None:
        mode_count = self._mode_count_for_cell(cell_id)
        if task != PILOT_TASK:
            raise ValueError("Track B pilot history dataset requires feedback task")
        if arm != PILOT_ARM:
            raise ValueError("Track B pilot history dataset requires A0_RAW arm")
        if split not in PILOT_SPLITS:
            raise ValueError("Track B pilot history dataset requires train or validation split")
        super().__init__(
            dataset_root,
            build_id=build_id,
            cell_id=cell_id,
            split=split,
            task=task,
            arm=arm,
            derived_checkpoint_sha=derived_checkpoint_sha,
            require_frozen_classifier_cache=require_frozen_classifier_cache,
        )
        try:
            expected_cell = f"trackb_feedback_history_n{mode_count}"
            if self.cell_id != expected_cell:
                raise ValueError("Track B pilot cell identity mismatch")
            for row in self.rows:
                if (str(row.get("dataset_family")) != PILOT_FAMILY
                        or str(row.get("measurement_domain")) != PILOT_DOMAIN
                        or str(row.get("cell_id")) != expected_cell
                        or str(row.get("split")) != split
                        or type(row.get("N")) is not int or row["N"] != mode_count
                        or str(row.get("system_type")) != f"{mode_count}qubit"):
                    raise ValueError("Track B pilot row metadata contract mismatch")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _mode_count_for_cell(cell_id: str) -> int:
        prefix = "trackb_feedback_history_n"
        if not isinstance(cell_id, str) or not cell_id.startswith(prefix):
            raise ValueError("Track B pilot cell must be trackb_feedback_history_n3 or n4")
        suffix = cell_id[len(prefix):]
        if suffix not in ("3", "4"):
            raise ValueError("Track B pilot supports only n3 and n4 history cells")
        return int(suffix)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        relative = str(row["observation_shard"])
        values = self.handle_cache.read_row(
            self.root / relative,
            int(row["observation_row"]),
            PILOT_OBSERVATION_FIELDS,
            expected_sha=self.identity["shard_hashes"][relative],
        )
        self._validate_row_shapes(values, self._mode_count_for_cell(self.cell_id))
        return {
            **{key: _to_tensor(value) for key, value in values.items()},
            "sample_id": row["sample_id"],
        }

    @staticmethod
    def _validate_row_shapes(values: Dict[str, np.ndarray], mode_count: int) -> None:
        expected = set(PILOT_OBSERVATION_FIELDS)
        if set(values) != expected:
            raise ValueError("Track B pilot observation fields are incomplete")
        history = np.asarray(values["history_tokens"])
        history_mask = np.asarray(values["history_mask"])
        noisy = np.asarray(values["noisy_measurement_tokens"])
        target = np.asarray(values["target_mode_tokens"])
        mode_mask = np.asarray(values["mode_mask"])
        if (history.ndim != 3 or history.shape != (PILOT_HISTORY_ROUNDS, mode_count, 7)
                or history_mask.shape != (PILOT_HISTORY_ROUNDS,)
                or noisy.shape != (mode_count, 2)
                or target.shape != (mode_count, 2)
                or mode_mask.shape != (mode_count,)):
            raise ValueError("Track B pilot history observation shape mismatch")
        if history_mask.dtype != np.bool_ or mode_mask.dtype != np.bool_:
            raise ValueError("Track B pilot history masks must be boolean")


__all__ = [
    "PILOT_ARM",
    "PILOT_DOMAIN",
    "PILOT_FAMILY",
    "PILOT_HISTORY_ROUNDS",
    "PILOT_OBSERVATION_FIELDS",
    "PILOT_SPLITS",
    "Pilot_history_dataset",
]
