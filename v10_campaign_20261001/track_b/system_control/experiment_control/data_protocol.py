"""Versioned measurement and feedback-transition data contracts.

The raw NumPy arrays remain the source of truth.  These manifests make the
experimental context explicit so that missing calibration or schedule fields
cannot silently become numeric zeros during model training.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


SNAPSHOT_MANIFEST_V3_VERSION = 3
SNAPSHOT_MANIFEST_VERSION = 4
CAMPAIGN_MANIFEST_VERSION = 1
FEEDBACK_TRANSITION_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SnapshotManifestV3:
    sample_id: str
    system_type: str
    started_at: float
    ended_at: float
    duration_s: float
    arrays: Dict[str, Dict[str, Any]]
    target_state_params: Dict[str, Any] = field(default_factory=dict)
    control_state_params: Dict[str, Any] = field(default_factory=dict)
    target_state_hash: Optional[str] = None
    control_state_hash: Optional[str] = None
    plan_id: Optional[str] = None
    state_index: Optional[int] = None
    state_role: str = "unspecified"
    anchor_id: Optional[str] = None
    operation_registry_hash: Optional[str] = None
    matrix_snapshot_hash: Optional[str] = None
    calibration_snapshot_hash: Optional[str] = None
    model_schema_hash: Optional[str] = None
    command_table_id: Optional[str] = None
    command_table_hash: Optional[str] = None
    awg_state: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    count_summary: Dict[str, Any] = field(default_factory=dict)
    multi_click_observation: Dict[str, Any] = field(default_factory=dict)
    missing_value_masks: Dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    source: str = "data_ui"
    episode_id: Optional[str] = None
    iteration_index: Optional[int] = None
    parent_bundle_hash: Optional[str] = None
    active_bundle_hash: Optional[str] = None
    matrix_bundle_hash: Optional[str] = None
    waveform_hash: Optional[str] = None
    upload_receipt_hash: Optional[str] = None
    requested_action_hash: Optional[str] = None
    projected_action_hash: Optional[str] = None
    actual_action_hash: Optional[str] = None
    subwindow_metadata: Dict[str, Any] = field(default_factory=dict)
    feedback_backend: Optional[str] = None
    model_checkpoint_hash: Optional[str] = None
    prepared_artifact_id: Optional[str] = None
    schema_version: int = SNAPSHOT_MANIFEST_V3_VERSION

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        if self.schema_version != SNAPSHOT_MANIFEST_V3_VERSION:
            raise ValueError(
                f"SnapshotManifestV3 requires schema_version={SNAPSHOT_MANIFEST_V3_VERSION}"
            )
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if not self.sample_id:
            raise ValueError("sample_id is required")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "SnapshotManifestV3":
        return cls(**dict(payload))


@dataclass
class SnapshotManifestV4(SnapshotManifestV3):
    """Authoritative V4 acquisition manifest.

    V3 remains readable for one compatibility cycle.  V4 makes exposure,
    campaign/probe state, environmental missingness, and all hardware hashes
    first-class so the training adapter never has to invent them.
    """

    campaign_id: Optional[str] = None
    schedule: Dict[str, Any] = field(default_factory=dict)
    probe: Dict[str, Any] = field(default_factory=dict)
    acquisition: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    schema_hash: Optional[str] = None
    schema_version: int = SNAPSHOT_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        if self.schema_version != SNAPSHOT_MANIFEST_VERSION:
            raise ValueError(
                f"SnapshotManifestV4 requires schema_version={SNAPSHOT_MANIFEST_VERSION}"
            )
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if not self.sample_id:
            raise ValueError("sample_id is required")
        exposure = float(self.acquisition.get("exposure_s", self.duration_s) or 0.0)
        if abs(exposure - self.duration_s) > 1e-6:
            raise ValueError("V4 acquisition exposure_s must equal duration_s")

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "SnapshotManifestV4":
        return cls(**dict(payload))


@dataclass
class CampaignManifestV1:
    """Atomically updated resume ledger for one immutable acquisition plan."""

    campaign_id: str
    system_type: str
    plan_id: str
    measurement_duration_s: float
    state_count: int
    coverage_count: int
    zero_anchor_count: int
    paired_probe_anchor_count: int
    states: list[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active_probe: Optional[Dict[str, Any]] = None
    next_resume_state_index: int = 1
    schema_version: int = CAMPAIGN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.system_type != "4qubit":
            raise ValueError("CampaignManifestV1 is currently 4qubit-only")
        if self.schema_version != CAMPAIGN_MANIFEST_VERSION:
            raise ValueError("CampaignManifestV1 requires schema_version=1")
        if not self.campaign_id or not self.plan_id:
            raise ValueError("campaign_id and plan_id are required")
        if float(self.measurement_duration_s) != 1200.0:
            raise ValueError("the V4 4Q campaign window is fixed at 1200 seconds")
        if self.state_count != len(self.states):
            raise ValueError("state_count must match the state ledger length")
        if self.state_count == 1000 and (
            self.coverage_count,
            self.zero_anchor_count,
            self.paired_probe_anchor_count,
        ) != (940, 12, 48):
            raise ValueError("the 1000-state campaign must be 940 coverage + 12 zero + 48 probe")
        sample_ids = [
            str(item.get("sample_id"))
            for item in self.states
            if item.get("sample_id") not in (None, "")
        ]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("campaign sample_id values must be unique")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CampaignManifestV1":
        return cls(**dict(payload))

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        atomic_write_json(destination, self.to_json_dict())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CampaignManifestV1":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_complete(self) -> None:
        if self.active_probe is not None:
            raise ValueError("completed campaign must not retain an active probe")
        for state in self.states:
            if state.get("status") != "completed":
                raise ValueError(
                    f"campaign state {state.get('state_index')} is not completed"
                )
            if float(state.get("duration_s", 0.0)) != self.measurement_duration_s:
                raise ValueError("every completed campaign state must have the fixed exposure")
            if not state.get("sample_id") or not state.get("snapshot_path"):
                raise ValueError("completed campaign state is missing sample identity/path")
            hashes = dict(state.get("hashes") or {})
            if any(
                hashes.get(name) in (None, "")
                for name in ("schema", "registry", "calibration", "matrix", "model_schema", "awg")
            ):
                raise ValueError("completed campaign state is missing an authoritative hash")


def build_snapshot_manifest(
    *,
    snapshot_dir: Path,
    system_type: str,
    current_data: Mapping[str, Any],
    array_fields: Iterable[str],
) -> SnapshotManifestV3 | SnapshotManifestV4:
    ended_at = float(current_data.get("timestamp", time.time()) or time.time())
    context = dict(current_data.get("feedback_context") or {})
    episode_context = _json_mapping(
        current_data.get("feedback_episode")
        or context.get("feedback_episode")
        or {}
    )
    automation = dict(context.get("automation") or {})
    target = dict(context.get("target_state_params") or {})
    control = dict(
        current_data.get("control_state_params")
        or context.get("control_state_params")
        or {}
    )
    target_state_hash = _first_present(
        current_data, episode_context, context, target,
        keys=("target_state_hash", "target_hash"),
    )
    control_state_hash = _first_present(
        current_data, episode_context, context, control,
        keys=("control_state_hash", "control_hash"),
    )
    duration_s = float(
        current_data.get("duration_s")
        or context.get("collection_duration_s")
        or context.get("measurement_duration_s")
        or 0.0
    )
    started_at = float(current_data.get("started_at") or (ended_at - max(duration_s, 0.0)))
    sample_id = str(current_data.get("sample_id") or uuid.uuid4())

    arrays: Dict[str, Dict[str, Any]] = {}
    for field_name in array_fields:
        path = snapshot_dir / f"{field_name}.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays[field_name] = {
            "path": path.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha256_file(path),
        }

    quality_flags: list[str] = []
    required = ("signal_bins", "idler_bins", "coin_bins")
    for field_name in required:
        if field_name not in arrays:
            quality_flags.append(f"missing_required_array:{field_name}")
    for field_name in ("signal_bins_raw", "idler_bins_raw", "coin_bins_raw"):
        if field_name not in arrays:
            quality_flags.append(f"missing_raw_count_array:{field_name}")

    operation_hash = _first_present(
        current_data,
        context,
        keys=("operation_point_registry_hash", "operation_registry_hash"),
    )
    matrix_hash = _first_present(current_data, context, keys=("matrix_snapshot_hash",))
    calibration_hash = _first_present(current_data, context, keys=("calibration_snapshot_hash",))
    model_schema_hash = _first_present(current_data, context, keys=("model_schema_hash", "feature_schema_hash"))
    command_table_id = _first_present(
        current_data, episode_context, context, keys=("command_table_id",),
    )
    command_table_hash = _first_present(
        current_data, episode_context, context, keys=("command_table_hash",),
    )
    if (command_table_id is None) != (command_table_hash is None):
        quality_flags.append("incomplete_context:command_table_id_hash_pair")

    for label, value in (
        ("operation_registry_hash", operation_hash),
        ("matrix_snapshot_hash", matrix_hash),
        ("calibration_snapshot_hash", calibration_hash),
        ("model_schema_hash", model_schema_hash),
    ):
        if value is None:
            quality_flags.append(f"missing_context:{label}")

    state_index = automation.get("state_index")
    if state_index is not None:
        state_index = int(state_index)
    state_role = str(target.get("state_role") or automation.get("state_role") or "unspecified")
    anchor_id = target.get("anchor_id") or automation.get("anchor_id")
    plan_path = automation.get("source_plan_path") or automation.get("plan_path")
    plan_id = automation.get("plan_id")
    if plan_id is None and plan_path:
        plan_id = stable_json_hash({"path": str(plan_path)})[:16]

    reconstruction = {
        "raw": "rho_raw.npy" if "rho_raw" in arrays else None,
        "calibrated": "rho_calibrated.npy" if "rho_calibrated" in arrays else None,
        "posterior_mean": "rho_posterior_mean.npy" if "rho_posterior_mean" in arrays else None,
    }
    uncertainty = _json_mapping(current_data.get("posterior_uncertainty") or {})
    count_summary = {
        name: current_data.get(name)
        for name in ("multi2_count", "multi3_count", "multi4_count", "multi2_pairs")
        if current_data.get(name) is not None
    }
    multi_click_observation = _multi_click_scope_metadata(arrays, current_data)
    missing_masks = _json_mapping(current_data.get("missing_value_masks") or {})
    temperature_c = _first_present(
        current_data,
        context,
        keys=("temperature_c", "temperature"),
    )
    temperature_missing = temperature_c is None
    missing_masks.setdefault("temperature_c", temperature_missing)
    environment = {
        "temperature_c": None if temperature_missing else float(temperature_c),
        "temperature_missing": bool(temperature_missing),
    }
    environment.update(_json_mapping(current_data.get("environment") or context.get("environment") or {}))
    campaign_id = automation.get("campaign_id") or context.get("campaign_id")
    probe = _json_mapping(
        current_data.get("phase_probe")
        or context.get("phase_probe")
        or target.get("phase_probe")
        or {}
    )
    schedule = {
        key: value
        for key, value in {
            "schedule_index": target.get("schedule_index", state_index),
            "state_role": state_role,
            "anchor_id": anchor_id,
            "block_id": target.get("block_id"),
            "pair_split_key": target.get("pair_split_key"),
        }.items()
        if value is not None
    }
    acquisition = {
        "exposure_s": max(duration_s, 0.0),
        "write_count": _first_present(current_data, context, keys=("write_count",)),
        "read_count": _first_present(current_data, context, keys=("read_count",)),
        "gate_exposure": _first_present(current_data, context, keys=("gate_exposure",)),
        "window_policy": str(
            context.get("window_policy")
            or current_data.get("window_policy")
            or "fixed_duration"
        ),
        "fpga_protocol_version": current_data.get("fpga_protocol_version"),
        "corrected_count_arrays": [
            name for name in ("signal_bins", "idler_bins", "coin_bins") if name in arrays
        ],
        "raw_count_arrays": [
            name
            for name in ("signal_bins_raw", "idler_bins_raw", "coin_bins_raw")
            if name in arrays
        ],
    }
    schema_payload = {
        "schema_version": SNAPSHOT_MANIFEST_VERSION,
        "arrays": sorted(arrays),
        "system_type": system_type,
        "operation_registry_hash": operation_hash,
        "model_schema_hash": model_schema_hash,
        "command_table_id": command_table_id,
        "command_table_hash": command_table_hash,
        "target_state_hash": target_state_hash,
        "control_state_hash": control_state_hash,
    }

    feedback_backend = _first_present(
        current_data, episode_context, context, keys=("feedback_backend",),
    )
    episode_values = {
        name: _first_present(current_data, episode_context, context, keys=(name,))
        for name in (
            "episode_id",
            "iteration_index",
            "parent_bundle_hash",
            "active_bundle_hash",
            "matrix_bundle_hash",
            "waveform_hash",
            "upload_receipt_hash",
            "requested_action_hash",
            "projected_action_hash",
            "actual_action_hash",
            "model_checkpoint_hash",
            "prepared_artifact_id",
        )
    }
    subwindow_metadata = _json_mapping(
        current_data.get("subwindow_metadata")
        or episode_context.get("subwindow_metadata")
        or context.get("subwindow_metadata")
        or {}
    )
    schema_payload["feedback_backend"] = feedback_backend
    schema_payload["episode_provenance_fields"] = sorted(
        name for name, value in episode_values.items() if value is not None
    )

    base_kwargs = dict(
        sample_id=sample_id,
        system_type=system_type,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=max(duration_s, 0.0),
        arrays=arrays,
        target_state_params=target,
        control_state_params=control,
        target_state_hash=None if target_state_hash is None else str(target_state_hash),
        control_state_hash=None if control_state_hash is None else str(control_state_hash),
        plan_id=None if plan_id is None else str(plan_id),
        state_index=state_index,
        state_role=state_role,
        anchor_id=None if anchor_id is None else str(anchor_id),
        operation_registry_hash=None if operation_hash is None else str(operation_hash),
        matrix_snapshot_hash=None if matrix_hash is None else str(matrix_hash),
        calibration_snapshot_hash=None if calibration_hash is None else str(calibration_hash),
        model_schema_hash=None if model_schema_hash is None else str(model_schema_hash),
        command_table_id=None if command_table_id is None else str(command_table_id),
        command_table_hash=None if command_table_hash is None else str(command_table_hash),
        awg_state=_json_mapping(current_data.get("awg_state") or context.get("awg_state") or {}),
        reconstruction=reconstruction,
        uncertainty=uncertainty,
        multi_click_observation=_json_mapping(multi_click_observation),
        count_summary=_json_mapping(count_summary),
        missing_value_masks=missing_masks,
        quality_flags=quality_flags,
        source=str(current_data.get("source") or "data_ui"),
        episode_id=None if episode_values["episode_id"] is None else str(episode_values["episode_id"]),
        iteration_index=(
            None if episode_values["iteration_index"] is None
            else int(episode_values["iteration_index"])
        ),
        parent_bundle_hash=(
            None if episode_values["parent_bundle_hash"] is None
            else str(episode_values["parent_bundle_hash"])
        ),
        active_bundle_hash=(
            None if episode_values["active_bundle_hash"] is None
            else str(episode_values["active_bundle_hash"])
        ),
        matrix_bundle_hash=(
            None if episode_values["matrix_bundle_hash"] is None
            else str(episode_values["matrix_bundle_hash"])
        ),
        waveform_hash=(
            None if episode_values["waveform_hash"] is None
            else str(episode_values["waveform_hash"])
        ),
        upload_receipt_hash=(
            None if episode_values["upload_receipt_hash"] is None
            else str(episode_values["upload_receipt_hash"])
        ),
        requested_action_hash=(
            None if episode_values["requested_action_hash"] is None
            else str(episode_values["requested_action_hash"])
        ),
        projected_action_hash=(
            None if episode_values["projected_action_hash"] is None
            else str(episode_values["projected_action_hash"])
        ),
        actual_action_hash=(
            None if episode_values["actual_action_hash"] is None
            else str(episode_values["actual_action_hash"])
        ),
        subwindow_metadata=subwindow_metadata,
        feedback_backend=None if feedback_backend is None else str(feedback_backend),
        model_checkpoint_hash=(
            None if episode_values["model_checkpoint_hash"] is None
            else str(episode_values["model_checkpoint_hash"])
        ),
        prepared_artifact_id=(
            None if episode_values["prepared_artifact_id"] is None
            else str(episode_values["prepared_artifact_id"])
        ),
    )
    if duration_s <= 0.0 and campaign_id is None:
        base_kwargs["quality_flags"] = [
            *quality_flags,
            "v3_compatibility_missing_exposure",
        ]
        manifest: SnapshotManifestV3 | SnapshotManifestV4 = SnapshotManifestV3(**base_kwargs)
    else:
        manifest = SnapshotManifestV4(
            **base_kwargs,
            campaign_id=None if campaign_id is None else str(campaign_id),
            schedule=_json_mapping(schedule),
            probe=probe,
            acquisition=_json_mapping(acquisition),
            environment=_json_mapping(environment),
            schema_hash=stable_json_hash(schema_payload),
        )
    if manifest.feedback_backend == "v6_episode":
        validate_v6_episode_provenance_fields(manifest)
    return manifest


V6_EPISODE_REQUIRED_PROVENANCE_FIELDS = (
    "episode_id",
    "iteration_index",
    "target_state_hash",
    "control_state_hash",
    "parent_bundle_hash",
    "active_bundle_hash",
    "command_table_id",
    "command_table_hash",
    "matrix_bundle_hash",
    "waveform_hash",
    "upload_receipt_hash",
    "requested_action_hash",
    "projected_action_hash",
    "actual_action_hash",
    "model_checkpoint_hash",
    "prepared_artifact_id",
)


def prepared_artifact_snapshot_context(
    *,
    prepared_artifact: Any,
    bundle: Any,
    episode_id: str,
    iteration_index: int,
    subwindow_metadata: Mapping[str, Any],
    model_checkpoint_hash: str,
) -> Dict[str, Any]:
    """Build snapshot provenance only from the frozen prepared artifact/bundle."""

    if str(bundle.artifact_id) != str(prepared_artifact.artifact_id):
        raise ValueError("bundle does not bind the supplied PreparedSequenceArtifactV1")
    if str(bundle.bundle_hash) == "" or not bundle.upload_receipt_hash:
        raise ValueError("active bundle is missing upload receipt provenance")
    return {
        "feedback_backend": "v6_episode",
        "episode_id": str(episode_id),
        "iteration_index": int(iteration_index),
        "target_state_hash": str(prepared_artifact.target_hash),
        "control_state_hash": str(prepared_artifact.control_hash),
        "parent_bundle_hash": str(bundle.parent_bundle_hash),
        "active_bundle_hash": str(bundle.bundle_hash),
        "command_table_id": str(prepared_artifact.command_table_id),
        "command_table_hash": str(prepared_artifact.command_table_hash),
        "matrix_bundle_hash": str(prepared_artifact.matrix_hash),
        "waveform_hash": str(prepared_artifact.waveform_hash),
        "upload_receipt_hash": str(bundle.upload_receipt_hash),
        "requested_action_hash": str(prepared_artifact.requested_action_hash),
        "projected_action_hash": str(prepared_artifact.projected_action_hash),
        "actual_action_hash": str(prepared_artifact.actual_action_hash),
        "subwindow_metadata": dict(subwindow_metadata),
        "model_checkpoint_hash": str(model_checkpoint_hash),
        "prepared_artifact_id": str(prepared_artifact.artifact_id),
    }


def validate_v6_episode_provenance_fields(
    manifest: SnapshotManifestV3 | SnapshotManifestV4,
) -> None:
    """Fail closed when a strict V6 episode snapshot lacks authoritative facts."""

    missing = [
        name
        for name in V6_EPISODE_REQUIRED_PROVENANCE_FIELDS
        if getattr(manifest, name, None) in (None, "")
    ]
    if manifest.feedback_backend != "v6_episode":
        missing.append("feedback_backend")
    metadata = dict(manifest.subwindow_metadata or {})
    if not metadata:
        missing.append("subwindow_metadata")
    elif not bool(metadata.get("authoritative_complete", False)):
        missing.append("subwindow_metadata.authoritative_complete")
    else:
        expected = int(metadata.get("expected_subwindows", 0) or 0)
        completed = int(metadata.get("completed_subwindows", 0) or 0)
        window_s = float(metadata.get("measurement_window_s", 0.0) or 0.0)
        total_s = float(metadata.get("total_duration_s", 0.0) or 0.0)
        if expected <= 0 or completed != expected:
            missing.append("subwindow_metadata.completed_subwindows")
        if window_s <= 0.0 or abs(total_s - window_s) > 1e-6:
            missing.append("subwindow_metadata.total_duration_s")
        if abs(manifest.duration_s - window_s) > 1e-6:
            missing.append("duration_s")
    if missing:
        raise ValueError(
            "strict V6 episode snapshot is missing provenance: " + ", ".join(missing)
        )


def validate_v6_episode_snapshot(
    manifest: SnapshotManifestV3 | SnapshotManifestV4,
    *,
    prepared_artifact: Any,
    bundle: Any,
    episode_id: str,
    iteration_index: int,
    model_checkpoint_hash: str,
) -> None:
    """Bind a strict snapshot to the exact pre-upload artifact and receipt."""

    validate_v6_episode_provenance_fields(manifest)
    expected = prepared_artifact_snapshot_context(
        prepared_artifact=prepared_artifact,
        bundle=bundle,
        episode_id=episode_id,
        iteration_index=iteration_index,
        subwindow_metadata=manifest.subwindow_metadata,
        model_checkpoint_hash=model_checkpoint_hash,
    )
    for field_name, expected_value in expected.items():
        if field_name == "subwindow_metadata":
            continue
        actual_value = getattr(manifest, field_name)
        if str(actual_value) != str(expected_value):
            raise ValueError(f"V6 episode snapshot provenance mismatch: {field_name}")


def load_snapshot_manifest(snapshot_dir: str | Path) -> SnapshotManifestV3 | SnapshotManifestV4:
    """Load V4 or the one-cycle V3 compatibility contract."""

    directory = Path(snapshot_dir)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", SNAPSHOT_MANIFEST_V3_VERSION))
        if version == SNAPSHOT_MANIFEST_VERSION:
            return SnapshotManifestV4.from_json_dict(payload)
        if version == SNAPSHOT_MANIFEST_V3_VERSION:
            return SnapshotManifestV3.from_json_dict(payload)
        raise ValueError(f"unsupported snapshot manifest schema_version={version}")

    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No manifest.json or metadata.json under {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    system_type = str(metadata.get("system_type") or "")
    saved_arrays = list(metadata.get("saved_arrays") or [])
    manifest = build_snapshot_manifest(
        snapshot_dir=directory,
        system_type=system_type,
        current_data=metadata,
        array_fields=saved_arrays,
    )
    manifest.quality_flags.append("legacy_metadata_adapter")
    return manifest


def validate_snapshot_files(
    snapshot_dir: str | Path,
    manifest: SnapshotManifestV3 | SnapshotManifestV4 | None = None,
) -> SnapshotManifestV3 | SnapshotManifestV4:
    """Verify every array recorded by a snapshot manifest against disk.

    Loading the JSON contract alone is intentionally cheap.  Campaign commit
    calls this stronger check so a truncated or externally modified NPY file
    can never be promoted to an authoritative sample.
    """

    directory = Path(snapshot_dir)
    loaded = manifest or load_snapshot_manifest(directory)
    for field_name, record in loaded.arrays.items():
        relative_path = Path(str(record.get("path") or ""))
        if not relative_path.name or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid snapshot array path for {field_name}: {relative_path}")
        path = directory / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"snapshot array is missing: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != list(record.get("shape") or []):
            raise ValueError(f"snapshot array shape mismatch: {field_name}")
        if str(values.dtype) != str(record.get("dtype") or ""):
            raise ValueError(f"snapshot array dtype mismatch: {field_name}")
        if sha256_file(path) != str(record.get("sha256") or ""):
            raise ValueError(f"snapshot array hash mismatch: {field_name}")
    return loaded


@dataclass
class FeedbackTransitionV1:
    transition_id: str
    system_type: str
    before_sample_id: str
    after_sample_id: str
    action: Dict[str, Any]
    action_bounds: Dict[str, Any]
    prediction_uncertainty: Dict[str, Any]
    guard_result: Dict[str, Any]
    observed_outcome: Dict[str, Any]
    policy_id: str
    batch_index: int
    created_at: float = field(default_factory=time.time)
    rollback: Dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    human_approved: bool = False
    schema_version: int = FEEDBACK_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        if not self.before_sample_id or not self.after_sample_id:
            raise ValueError("before_sample_id and after_sample_id are required")
        if self.before_sample_id == self.after_sample_id:
            raise ValueError("before and after snapshots must be different")
        if self.schema_version != FEEDBACK_TRANSITION_VERSION:
            raise ValueError("FeedbackTransitionV1 requires schema_version=1")
        if self.batch_index < 1 or self.batch_index > 5:
            raise ValueError("batch_index must be in 1..5")
        if self.promoted and not self.human_approved:
            raise ValueError("a transition cannot be promoted without human approval")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "FeedbackTransitionV1":
        return cls(**dict(payload))

    def save(self, output_dir: str | Path) -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.transition_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)
        return path


def new_transition_id(system_type: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{system_type}_{uuid.uuid4().hex[:8]}"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _multi_click_scope_metadata(
    arrays: Mapping[str, Any], current_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe only multi-click fields that actually exist in this snapshot."""
    field_scopes: Dict[str, str] = {}
    for name in ("multi2_bins", "multi3_bins", "multi4_bins"):
        if name in arrays:
            field_scopes[name] = "point"
    for name in ("multi2_cycle", "multi3_cycle", "multi4_cycle"):
        if name in arrays:
            field_scopes[name] = "cycle"
    for name in ("multi2_pairs", "multi2_count", "multi3_count", "multi4_count"):
        if current_data.get(name) is not None:
            field_scopes[name] = "global"
    if not field_scopes:
        return {}
    scopes = set(field_scopes.values())
    if "point" in scopes:
        primary_scope = "point"
    elif "cycle" in scopes:
        primary_scope = "cycle"
    else:
        primary_scope = "global"
    return {
        "schema": "multi_click_observation", "version": 2,
        "scope": primary_scope, "field_scopes": field_scopes,
        "available_fields": sorted(field_scopes),
    }
def _first_present(*mappings: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def _json_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return json.loads(json.dumps(dict(value), default=str))
