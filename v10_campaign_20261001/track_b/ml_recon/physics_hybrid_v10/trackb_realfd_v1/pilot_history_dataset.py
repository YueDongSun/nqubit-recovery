"""Observation-only Track B history view with bounded root admission.

The public constructor admits one immutable metadata/index snapshot before any
row selection. Internal paired split readers share the same admission while
owning separate HDF5 caches. Truth shards and analytic action are never read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from ml_recon.data_v2_4.admission_snapshot import read_stable_bytes
from ml_recon.data_v2_4.loaders import _to_tensor
from ml_recon.data_v2_4.shard_reader import V24Hdf5HandleCache
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V1,
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_CELL_IDS,
    TRACKB_TEACHER_POLICY_V2_SHA256,
)
from .pilot_dataset_admission import (
    MAX_PILOT_INDEX_BYTES,
    MAX_PILOT_MANIFEST_BYTES,
    Pilot_dataset_admission,
    admit_pilot_dataset,
)
from .pilot_observation_identity import (
    validate_observation_row_index,
    validate_stored_observation_identity,
)


PILOT_TASK = "feedback"
PILOT_ARM = "A0_RAW"
PILOT_FAMILY = "feedback_history"
PILOT_DOMAIN = "global_pauli_v24"
PILOT_SPLITS = frozenset({"train", "validation"})
PILOT_HISTORY_ROUNDS = 10
PILOT_OBSERVATION_FIELDS = (
    "history_mask", "history_tokens", "mode_mask",
    "noisy_measurement_tokens", "target_mode_tokens",
)


def _read_verified_snapshot(
    path: Path, *, expected_sha: str, label: str, max_bytes: int,
) -> bytes:
    """Compatibility helper for exact READY-bound regular-file snapshots."""
    return read_stable_bytes(
        path, label=f"Track B pilot {label}", expected_sha=expected_sha,
        max_bytes=max_bytes, reject_reparse=True,
    )


class Pilot_history_dataset(Dataset):
    """Narrow A0 history reader whose identity and rows share one admission."""

    def __init__(self, dataset_root: str | Path, *, build_id: str, cell_id: str,
                 split: str, task: str = PILOT_TASK, arm: str = PILOT_ARM,
                 derived_checkpoint_sha: str = "",
                 require_frozen_classifier_cache: bool = False,
                 history_policy: str = TRACKB_TEACHER_POLICY_V1,
                 history_policy_sha256: str = "") -> None:
        if task != PILOT_TASK or arm != PILOT_ARM:
            raise ValueError("Track B pilot history dataset requires feedback task and A0_RAW arm")
        if split not in PILOT_SPLITS:
            raise ValueError("Track B pilot history dataset requires train or validation split")
        if history_policy == TRACKB_TEACHER_POLICY_V1:
            if type(history_policy_sha256) is not str or history_policy_sha256:
                raise ValueError("Track B V1 reader cannot declare a policy SHA")
        elif history_policy == TRACKB_TEACHER_POLICY_V2:
            if history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256:
                raise ValueError("Track B V2 reader requires the registered history policy SHA")
        else:
            raise ValueError("Track B pilot history policy is unsupported")
        self._mode_count_for_cell(cell_id, history_policy)
        admission = admit_pilot_dataset(
            dataset_root, build_id=build_id, cell_id=cell_id,
            history_policy=history_policy, history_policy_sha256=history_policy_sha256,
        )
        self._initialize_from_admission(admission, split=split)

    @classmethod
    def _from_admission(cls, admission: Pilot_dataset_admission, *,
                        split: str) -> "Pilot_history_dataset":
        """Create a split view of the same factory-validated root snapshot."""
        if type(admission) is not Pilot_dataset_admission:
            raise TypeError("Track B reader requires a factory-created pilot admission")
        admission.require_valid()
        if split not in PILOT_SPLITS:
            raise ValueError("Track B pilot history dataset requires train or validation split")
        instance = cls.__new__(cls)
        instance._initialize_from_admission(admission, split=split)
        return instance

    def _initialize_from_admission(self, admission: Pilot_dataset_admission, *,
                                   split: str) -> None:
        if type(admission) is not Pilot_dataset_admission:
            raise TypeError("Track B reader requires a factory-created pilot admission")
        admission.require_valid()
        self.root = admission.root
        self.identity = admission.identity_dict()
        self.rows = [dict(row) for row in admission.rows_for_split(split)]
        self.sample_ids = tuple(str(row["sample_id"]) for row in self.rows)
        self.cell_id = admission.cell_id
        self.split = split
        self.task = PILOT_TASK
        self.arm = PILOT_ARM
        self.history_policy = admission.history_policy
        self.history_policy_sha256 = admission.history_policy_sha256
        self.validation = admission.validation_dict()
        self.classifier_checkpoint_sha = self.validation.get("classifier_checkpoint_sha", "")
        self.control_forward_sha = self.validation.get("control_forward_sha", "")
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=8)

    @staticmethod
    def _expected_cell(mode_count: int, history_policy: str) -> str:
        if history_policy == TRACKB_TEACHER_POLICY_V1:
            return f"trackb_feedback_history_n{mode_count}"
        if history_policy == TRACKB_TEACHER_POLICY_V2:
            return TRACKB_TEACHER_POLICY_V2_CELL_IDS[mode_count]
        raise ValueError("Track B pilot history policy is unsupported")

    @classmethod
    def _mode_count_for_cell(cls, cell_id: str, history_policy: str) -> int:
        if history_policy == TRACKB_TEACHER_POLICY_V2:
            matches = [mode for mode, candidate in TRACKB_TEACHER_POLICY_V2_CELL_IDS.items()
                       if candidate == cell_id]
            if len(matches) != 1:
                raise ValueError("Track B V2 pilot cell must be trackb_feedback_history_v2_n3 or n4")
            return matches[0]
        if history_policy != TRACKB_TEACHER_POLICY_V1:
            raise ValueError("Track B pilot history policy is unsupported")
        prefix = "trackb_feedback_history_n"
        if not isinstance(cell_id, str) or not cell_id.startswith(prefix):
            raise ValueError("Track B pilot cell must be trackb_feedback_history_n3 or n4")
        suffix = cell_id[len(prefix):]
        if suffix not in ("3", "4"):
            raise ValueError("Track B pilot supports only n3 and n4 history cells")
        return int(suffix)

    def __len__(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        self.handle_cache.close()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        relative = str(row["observation_shard"])
        row_index = validate_observation_row_index(row)
        values = self.handle_cache.read_row(
            self.root / relative, row_index,
            (*PILOT_OBSERVATION_FIELDS, "sample_id", "row_valid_mask"),
            expected_sha=self.identity["shard_hashes"][relative],
        )
        validate_stored_observation_identity(
            values.pop("sample_id"), values.pop("row_valid_mask"),
            sample_id=row["sample_id"],
        )
        self._validate_row_shapes(
            values, self._mode_count_for_cell(self.cell_id, self.history_policy))
        return {
            **{key: _to_tensor(value) for key, value in values.items()},
            "sample_id": row["sample_id"],
        }

    @staticmethod
    def _validate_row_shapes(values: Dict[str, np.ndarray], mode_count: int) -> None:
        if set(values) != set(PILOT_OBSERVATION_FIELDS):
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
    "MAX_PILOT_INDEX_BYTES", "MAX_PILOT_MANIFEST_BYTES", "PILOT_ARM",
    "PILOT_DOMAIN", "PILOT_FAMILY", "PILOT_HISTORY_ROUNDS",
    "PILOT_OBSERVATION_FIELDS", "PILOT_SPLITS", "Pilot_history_dataset",
]
