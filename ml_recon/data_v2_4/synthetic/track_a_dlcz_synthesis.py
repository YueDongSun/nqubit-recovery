"""Source-bound Track A DLCZ event synthesis for V2.4 research fixtures.

This adapter deliberately owns no optical source law.  It invokes the Track A
event renderer copied from its reviewed source revision and retains the result's
provenance receipts.  Its public entry point accepts only the registered
research-only K=1 general-N schedule; legacy 75/256 schedules are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
from typing import Sequence

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.v10_recoverability_foundry.contracts.dlcz_event_mc import (
    DlczAllocationReceipt,
    DlczDomainContract,
)
from ml_recon.v10_recoverability_foundry.data.aod_operation_schedule import (
    Aod_operation_schedule,
    build_registered_synthetic_schedule,
)
from ml_recon.v10_recoverability_foundry.data.dlcz_event_renderer import (
    DlczEventRenderResult,
    build_dlcz_operation_parameters,
    render_dlcz_event_row,
)
from ml_recon.v10_recoverability_foundry.physics.dlcz_kernel_adapter import (
    kernel_provenance,
)


TRACK_A_SOURCE_COMMIT = "a7b5fae30524f2a91301d0b2e117228fe835bba6"
TRACK_A_SOURCE_TREE = "9e76a0c733f143b0f51c65cea1c6fc9162dd2af2"
TRACK_A_EVENT_RENDERER_SCHEMA = "V10DlczEventRendererV1"
TRACK_A_OBSERVATION_ADAPTER_SCHEMA = "V10DlczObservationAdapterV1"
TRACK_A_SYNTHESIS_SCHEMA = "V24TrackADlczSynthesisV1"
TRACK_A_DENSE_MODE_LIMIT = 16
TRACK_A_ALLOCATION_SCHEMA = "V10DeterministicAllocationReceiptV1"
TRACK_A_GLOBAL_MULTI_SCOPE = "NATIVE_CLOCK_GLOBAL_NOT_OPERATION_TARGET"


class _sample_key_rng:
    """Expose Track A named substreams from V2.4's deterministic sample key."""

    def __init__(self, sample_key) -> None:
        self.sample_key = sample_key

    def stream(self, primitive_name: str, substream_name: str) -> np.random.Generator:
        if not primitive_name or not substream_name:
            raise ValueError("Track A DLCZ primitive and substream names are required")
        digest = hashlib.sha256(
            f"{primitive_name}|{substream_name}".encode("utf-8")
        ).hexdigest()
        return self.sample_key.rng(f"track_a_dlcz:{digest}")

    def __getattr__(self, name: str):
        return getattr(self.sample_key, name)


@dataclass(frozen=True)
class TrackADlczSynthesisConfigV1:
    """Explicit unbound-hypothesis settings for research-only synthetic rows."""

    requested_mode_shot_budget: int = 144
    exposure_coordinate: float = 1.0
    write_chi: float = 0.04
    background_signal_rate: float = 0.001
    background_idler_rate: float = 0.001
    eta_signal: float = 0.45
    eta_retrieval: float = 0.35
    eta_idler: float = 0.45
    w_overlap: float = 0.90
    eta_optical_path: float = 0.80
    spinwave_coherence_visibility: float = 0.95

    def validate(self) -> None:
        if type(self.requested_mode_shot_budget) is not int or self.requested_mode_shot_budget < 1:
            raise ValueError("Track A DLCZ mode-shot budget must be a positive integer")
        if not 0.0 < float(self.exposure_coordinate):
            raise ValueError("Track A DLCZ exposure coordinate must be positive")
        if not 0.0 < float(self.write_chi) < 1.0:
            raise ValueError("Track A DLCZ write_chi must lie in (0, 1)")
        for name in (
            "background_signal_rate", "background_idler_rate", "eta_signal",
            "eta_retrieval", "eta_idler", "w_overlap", "eta_optical_path",
            "spinwave_coherence_visibility",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (name.startswith("eta_") and value > 1.0):
                raise ValueError(f"invalid Track A DLCZ parameter: {name}")


@dataclass(frozen=True)
class TrackADlczStorageMetadataV1:
    """One typed metadata contract shared by V2.4 shards and build manifests."""

    synthetic_abi: str
    measurement_domain: str
    schedule_family: str
    schedule_sha: str
    operation_count: int
    operation_roles_sha: str
    allocation_schema: str
    allocation_sha: str
    budget_unit: str
    global_multi_scope: str
    track_a_source_commit: str
    track_a_source_tree: str

    def validate(self) -> None:
        if self.synthetic_abi != TRACK_A_SYNTHESIS_SCHEMA or self.measurement_domain != "dlcz_operation_v24":
            raise ValueError("Track A DLCZ storage metadata has an invalid ABI/domain")
        if self.schedule_family != "W_COMPLETE_GENERAL_N" or type(self.operation_count) is not int or self.operation_count < 1:
            raise ValueError("Track A DLCZ storage metadata has an invalid schedule")
        if self.allocation_schema != TRACK_A_ALLOCATION_SCHEMA or self.budget_unit != "mode_shot":
            raise ValueError("Track A DLCZ storage metadata has an invalid allocation")
        if self.global_multi_scope != TRACK_A_GLOBAL_MULTI_SCOPE:
            raise ValueError("Track A DLCZ storage metadata has an invalid multi-click scope")
        if (self.track_a_source_commit, self.track_a_source_tree) != (TRACK_A_SOURCE_COMMIT, TRACK_A_SOURCE_TREE):
            raise ValueError("Track A DLCZ storage metadata source identity diverged")
        for value in (self.schedule_sha, self.operation_roles_sha, self.allocation_sha):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Track A DLCZ storage metadata requires SHA identities")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "synthetic_abi": self.synthetic_abi,
            "measurement_domain": self.measurement_domain,
            "schedule_id": self.schedule_family,
            "schedule_family": self.schedule_family,
            "schedule_sha": self.schedule_sha,
            "operation_count": self.operation_count,
            "operation_roles_sha": self.operation_roles_sha,
            "allocation_schema": self.allocation_schema,
            "allocation_sha": self.allocation_sha,
            "budget_unit": self.budget_unit,
            "global_multi_scope": self.global_multi_scope,
            "track_a_source_commit": self.track_a_source_commit,
            "track_a_source_tree": self.track_a_source_tree,
            "shot_mode": "mode_shot_allocation",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrackADlczStorageMetadataV1":
        expected = {
            "synthetic_abi", "measurement_domain", "schedule_family", "schedule_sha",
            "operation_count", "operation_roles_sha", "allocation_schema", "allocation_sha",
            "budget_unit", "global_multi_scope", "track_a_source_commit", "track_a_source_tree",
            "schedule_id", "shot_mode",
        }
        if set(value) != expected:
            raise ValueError("Track A DLCZ storage metadata field set diverged")
        if value["schedule_id"] != value["schedule_family"] or value["shot_mode"] != "mode_shot_allocation":
            raise ValueError("Track A DLCZ storage metadata compatibility fields diverged")
        core = {name: value[name] for name in expected - {"schedule_id", "shot_mode"}}
        result = cls(**core)
        result.validate()
        if result.to_dict() != value:
            raise ValueError("Track A DLCZ storage metadata is not canonical")
        return result

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


def _current_git_identity() -> tuple[str, str]:
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), text=True).strip()
    return commit, tree


def track_a_source_provenance() -> dict[str, object]:
    """Bind the exact imported source revision and current kernel file receipts."""

    kernel = kernel_provenance()
    kernel_sha = sha256_of(kernel)
    return {
        "schema": TRACK_A_SYNTHESIS_SCHEMA,
        "track_a_source_commit": TRACK_A_SOURCE_COMMIT,
        "track_a_source_tree": TRACK_A_SOURCE_TREE,
        "event_renderer_schema": TRACK_A_EVENT_RENDERER_SCHEMA,
        "observation_adapter_schema": TRACK_A_OBSERVATION_ADAPTER_SCHEMA,
        "kernel_provenance": kernel,
        "kernel_provenance_sha": kernel_sha,
        "current_source_identity": _current_git_identity(),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def build_track_a_general_n_schedule(n_mode: int) -> Aod_operation_schedule:
    """Build only the Track A registered K=1 research schedule, never 75/256."""

    if type(n_mode) is not int or not 3 <= n_mode <= TRACK_A_DENSE_MODE_LIMIT:
        raise ValueError("Track A DLCZ general-N synthesis supports N=3..16 only")
    return build_registered_synthetic_schedule("W_COMPLETE_GENERAL_N", n_mode)


def render_track_a_dlcz_row(
    sample_key,
    *,
    schedule: Aod_operation_schedule,
    mode_probability: Sequence[float] | None = None,
    config: TrackADlczSynthesisConfigV1 = TrackADlczSynthesisConfigV1(),
) -> DlczEventRenderResult:
    """Render one source-bound DLCZ row using a supplied registered schedule."""

    config.validate()
    schedule.validate()
    if schedule.family != "W_COMPLETE_GENERAL_N" or not 3 <= schedule.n_mode <= TRACK_A_DENSE_MODE_LIMIT:
        raise ValueError("Track A DLCZ public renderer accepts only registered N=3..16 general-N schedules")
    scope = schedule.semantic_scope
    if scope is None or scope.supported_excitation_sectors != (1,) or scope.is_experimental_75_256_parity:
        raise ValueError("Track A DLCZ public renderer requires the K=1 non-legacy schedule scope")
    canonical_schedule = build_track_a_general_n_schedule(schedule.n_mode)
    if schedule.sha256() != canonical_schedule.sha256():
        raise ValueError("Track A DLCZ public renderer requires the exact canonical general-N schedule")
    n_mode = schedule.n_mode
    probability = np.full(n_mode, 1.0 / n_mode, dtype=np.float64)
    if mode_probability is not None:
        probability = np.asarray(mode_probability, dtype=np.float64).reshape(-1)
    if probability.shape != (n_mode,) or np.any(probability < 0.0) or not np.isclose(probability.sum(), 1.0):
        raise ValueError("Track A DLCZ mode_probability must be normalized and schedule-aligned")
    allocation = DlczAllocationReceipt.from_requested(
        config.requested_mode_shot_budget,
        schedule.operation_ids,
        n_mode=n_mode,
        budget_unit="mode_shot",
        operation_registry_sha=schedule.sha256(),
    ).with_hash()
    operation_count = len(schedule.operation_ids)
    exposure = np.full(operation_count, config.exposure_coordinate, dtype=np.float64)
    lambda_pair = -np.log1p(-config.write_chi) / config.exposure_coordinate
    operation_parameters = build_dlcz_operation_parameters(
        schedule,
        allocation,
        exposure_s=exposure,
        lambda_pair=np.full(operation_count, lambda_pair),
        write_chi=np.full(operation_count, config.write_chi),
        background_signal_rate=np.full(operation_count, config.background_signal_rate),
        background_idler_rate=np.full(operation_count, config.background_idler_rate),
        eta_signal=np.full(operation_count, config.eta_signal),
        eta_retrieval=np.full(operation_count, config.eta_retrieval),
        eta_idler=np.full(operation_count, config.eta_idler),
        w_overlap=np.full(operation_count, config.w_overlap),
        eta_optical_path=np.full(operation_count, config.eta_optical_path),
        spinwave_coherence_visibility=np.full(operation_count, config.spinwave_coherence_visibility),
        parameter_status="HYPOTHESIS_UNBOUND",
    )
    result = render_dlcz_event_row(
        schedule=schedule,
        allocation=allocation,
        operation_parameters=operation_parameters,
        n_mode=n_mode,
        mode_probability=probability,
        keyed_rng=_sample_key_rng(sample_key),
        group_id=f"v24:{sample_key.sample_id}",
        domain=DlczDomainContract(status="HYPOTHESIS_UNBOUND"),
        arm="PAIR",
        retain_truth_ledgers=True,
    )
    result.validate(n_mode=n_mode)
    if result.metadata.get("kernel_provenance", {}).get("schema") != "V10DlczKernelAdapterV1":
        raise RuntimeError("Track A DLCZ renderer returned an unbound kernel receipt")
    return result


def render_track_a_dlcz_dataset_sample(
    sample_key,
    *,
    cell: dict | None = None,
) -> dict:
    """Project one Track A event row into V2.4 public/truth storage planes.

    The public plane contains only operation-window four-cell observations and
    aggregate multi-click counters.  Per-operation parameters and event ledgers
    remain in the privileged provenance payload rather than student features.
    """

    cell = {} if cell is None else dict(cell)
    n_mode = int(sample_key.N)
    schedule = build_track_a_general_n_schedule(n_mode)
    requested_budget = int(cell.get("requested_mode_shot_budget", 144))
    result = render_track_a_dlcz_row(
        sample_key,
        schedule=schedule,
        config=TrackADlczSynthesisConfigV1(requested_mode_shot_budget=requested_budget),
    )
    observations = result.observations
    four_cell_counts = np.asarray(
        [item.four_cell_counts for item in observations], dtype=np.uint32
    )
    repeats = np.asarray([item.repeats for item in observations], dtype=np.uint32)
    setting_mask = repeats > 0
    measurement_probability = np.divide(
        four_cell_counts,
        np.maximum(repeats[:, None], 1),
        dtype=np.float64,
    ).astype(np.float32)
    multi2_pairs = np.asarray(
        [item.global_multi2_pair_counts for item in observations], dtype=np.uint64
    ).sum(axis=0)
    multi_counts = np.asarray(
        [item.global_multi_click_counts for item in observations], dtype=np.uint64
    ).sum(axis=0)
    global_multi = np.asarray(
        (multi_counts[0], *multi2_pairs, multi_counts[1], multi_counts[2]),
        dtype=np.uint64,
    )
    provenance = {
        **track_a_source_provenance(),
        "event_renderer_metadata": result.metadata,
        "source_event_ledger_sha": [ledger.sha256() for ledger in result.source_ledgers],
        "readout_event_ledger_sha": [ledger.sha256() for ledger in result.readout_ledgers],
        "operation_schedule_sha": schedule.sha256(),
        "operation_ids": list(schedule.operation_ids),
        "legacy_real_75_256_claimed": False,
    }
    provenance_bytes = json.dumps(
        provenance, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    storage_metadata = TrackADlczStorageMetadataV1(
        synthetic_abi=TRACK_A_SYNTHESIS_SCHEMA,
        measurement_domain="dlcz_operation_v24",
        schedule_family=schedule.family,
        schedule_sha=schedule.sha256(),
        operation_count=len(schedule.operation_ids),
        operation_roles_sha=sha256_of(list(schedule.operation_roles)),
        allocation_schema=TRACK_A_ALLOCATION_SCHEMA,
        allocation_sha=result.allocation.full_allocation_sha,
        budget_unit=result.allocation.budget_unit,
        global_multi_scope=str(result.metadata["global_multi_scope"]),
        track_a_source_commit=TRACK_A_SOURCE_COMMIT,
        track_a_source_tree=TRACK_A_SOURCE_TREE,
    )
    observation = {
        "counts": four_cell_counts,
        "measurement_probability": measurement_probability,
        "basis_descriptors": np.concatenate(
            (schedule.amplitudes, schedule.phases), axis=1
        ).astype(np.float32),
        "shots": repeats,
        "setting_uncertainty": np.sqrt(
            np.maximum(measurement_probability * (1.0 - measurement_probability), 0.0)
            .mean(axis=1)
            / np.maximum(repeats, 1)
        ).astype(np.float32),
        "setting_mask": setting_mask,
        "global_multi": global_multi,
        "global_multi_mask": np.asarray(True, dtype=np.bool_),
        "environment": np.zeros(7, dtype=np.float32),
        "environment_mask": np.zeros(7, dtype=np.bool_),
    }
    truth = {
        "track_a_dlcz_provenance": np.asarray(provenance_bytes, dtype=f"S{len(provenance_bytes)}"),
    }
    identity = hashlib.sha256(provenance_bytes).hexdigest()
    return {
        "observation": observation,
        "truth": truth,
        "identities": {
            "latent_state_id": identity,
            "transition_id": result.metadata["crn_receipt_sha"],
            "crn_group_id": result.metadata["allocation_sha"],
        },
        "storage_metadata": storage_metadata,
    }


__all__ = [
    "TRACK_A_DENSE_MODE_LIMIT",
    "TrackADlczStorageMetadataV1",
    "TRACK_A_EVENT_RENDERER_SCHEMA",
    "TRACK_A_OBSERVATION_ADAPTER_SCHEMA",
    "TRACK_A_SOURCE_COMMIT",
    "TRACK_A_SOURCE_TREE",
    "TrackADlczSynthesisConfigV1",
    "build_track_a_general_n_schedule",
    "render_track_a_dlcz_row",
    "render_track_a_dlcz_dataset_sample",
    "track_a_source_provenance",
]
