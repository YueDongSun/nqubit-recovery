"""Occupancy-vector spin-wave propagation for synthetic event diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..contracts.dlcz_event_mc import DlczEvent, DlczOperationParameters
from .dlcz_multi_cell_spinwave import (
    DlczMultiCellSpinwaveChannel,
    SpinwaveProjection,
)


SPINWAVE_SCHEMA = "V10DlczSpinwaveEventStateV1"


# A one-body PSD storage/AOD projection does not define the two-particle
# channel.  In particular, it cannot determine whether the local and cross
# Fock components interfere, decohere, or have distinct retrieval loss.  Keep
# this boundary explicit until a registered two-particle apparatus forward map
# and observable are supplied.
TWO_PARTICLE_READOUT_DEFERRED = (
    "DEFER_NO_REGISTERED_TWO_PARTICLE_SPINWAVE_READOUT_PROPAGATOR"
)
TWO_PARTICLE_READOUT_NOT_APPLICABLE = "NOT_APPLICABLE_SINGLE_EXCITATION"
SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS = (
    "SAME_COLLECTIVE_MODE_LINEAR_OPTICS_FOCK_HYPOTHESIS"
)


@dataclass(frozen=True)
class SpinwaveEventState:
    event_id: str
    occupancy: tuple[int, ...]
    topology: str
    phase_rad: float
    wavevector: tuple[float, ...]
    storage_wait_s: float
    survival_probability: float
    dephasing_factor: float
    retrieval_overlap: float
    parameter_status: str
    survival_by_cell: tuple[float, ...] = ()
    dephasing_by_cell: tuple[float, ...] = ()
    evolution_kernel: str = "UNBOUND_PLACEHOLDER"
    selected_pre_splitter_probability: float = 1.0
    coherent_sector_kind: str = "LEGACY_OCCUPANCY_PLACEHOLDER"
    multi_particle_readout_status: str = TWO_PARTICLE_READOUT_NOT_APPLICABLE

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("spin-wave event identity is required")
        if self.topology not in {
            "SINGLE",
            "LOCAL_DOUBLE",
            "CROSS_DOUBLE",
            "MULTIPAIR",
            "NONE",
        }:
            raise ValueError("unregistered spin-wave topology")
        if any(type(value) is not int or value < 0 for value in self.occupancy):
            raise ValueError("spin-wave occupancy must be non-negative integers")
        for name in ("phase_rad", "storage_wait_s"):
            if (
                not math.isfinite(float(getattr(self, name)))
                or float(getattr(self, name)) < 0.0
                and name == "storage_wait_s"
            ):
                raise ValueError(f"{name} must be finite")
        for name in ("survival_probability", "dephasing_factor", "retrieval_overlap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        selected_pre_splitter = float(self.selected_pre_splitter_probability)
        if (
            not math.isfinite(selected_pre_splitter)
            or not 0.0 <= selected_pre_splitter <= 1.0
        ):
            raise ValueError("selected_pre_splitter_probability must lie in [0, 1]")
        if any(not math.isfinite(float(value)) for value in self.wavevector):
            raise ValueError("spin-wave wavevector must be finite")
        if self.survival_by_cell and (
            len(self.survival_by_cell) != len(self.occupancy)
            or any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.survival_by_cell
            )
        ):
            raise ValueError("per-cell survival factors must align with occupancy")
        if self.dephasing_by_cell and (
            len(self.dephasing_by_cell) != len(self.occupancy)
            or any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.dephasing_by_cell
            )
        ):
            raise ValueError("per-cell dephasing factors must align with occupancy")
        if self.evolution_kernel not in {
            "SINGLE_CELL_PLACEHOLDER",
            "LOCAL_DOUBLE_CLUSTER_PLACEHOLDER",
            "CROSS_CELL_PRODUCT_PLACEHOLDER",
            "MULTIPAIR_PLACEHOLDER",
            "NONE",
            "UNBOUND_PLACEHOLDER",
            "MULTI_CELL_PSD_AOD_UNBOUND",
        }:
            raise ValueError("unregistered spin-wave evolution kernel")
        if self.coherent_sector_kind not in {
            "LEGACY_OCCUPANCY_PLACEHOLDER",
            "SAME_COLLECTIVE_MODE_FOCK_HYPOTHESIS",
            "NONE",
        }:
            raise ValueError("unregistered coherent spin-wave sector kind")
        if self.multi_particle_readout_status not in {
            TWO_PARTICLE_READOUT_DEFERRED,
            TWO_PARTICLE_READOUT_NOT_APPLICABLE,
            SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS,
        }:
            raise ValueError("unregistered multi-particle readout status")
        if (
            sum(self.occupancy) >= 2
            and self.evolution_kernel == "MULTI_CELL_PSD_AOD_UNBOUND"
            and self.multi_particle_readout_status
            not in {
                TWO_PARTICLE_READOUT_DEFERRED,
                SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS,
            }
        ):
            raise ValueError(
                "multi-cell multi-particle states require a deferred readout status"
            )

    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(repr(self).encode()).hexdigest()


def _topology(event: DlczEvent) -> str:
    if event.mark == "SINGLE_CELL_SINGLE_EXCITATION":
        return "SINGLE"
    if event.mark == "LOCAL_DOUBLE":
        return "LOCAL_DOUBLE"
    if event.mark == "CROSS_DOUBLE":
        return "CROSS_DOUBLE"
    if event.mark == "MULTIPAIR_K_GE_3":
        return "MULTIPAIR"
    return "NONE"


def propagate_spinwave_event(
    event: DlczEvent,
    params: DlczOperationParameters,
    *,
    rng: np.random.Generator,
    multi_cell_channel: DlczMultiCellSpinwaveChannel | None = None,
    multi_cell_projection: SpinwaveProjection | None = None,
) -> SpinwaveEventState:
    """Propagate a source event while preserving local/cross topology.

    The current source has no apparatus-bound dephasing or nonlinear-phase
    receipt.  Consequently the default law is a transparent placeholder with
    explicit ``HYPOTHESIS_UNBOUND`` provenance; it cannot be promoted to a
    calibration claim.
    """

    params.validate()
    event.validate(n_mode=len(event.occupancy) if event.occupancy else None)
    topology = _topology(event)
    wait = float(params.storage_wait_s)
    lifetime = max(float(params.storage_lifetime_s), np.finfo(float).tiny)
    survival = float(math.exp(-wait / lifetime))
    survival_by_cell = (survival,) * len(event.occupancy)
    dephasing_by_cell = survival_by_cell
    evolution_kernel = {
        "SINGLE": "SINGLE_CELL_PLACEHOLDER",
        "LOCAL_DOUBLE": "LOCAL_DOUBLE_CLUSTER_PLACEHOLDER",
        "CROSS_DOUBLE": "CROSS_CELL_PRODUCT_PLACEHOLDER",
        "MULTIPAIR": "MULTIPAIR_PLACEHOLDER",
        "NONE": "NONE",
    }[topology]
    selected_pre_splitter_probability = 1.0
    retrieval_overlap = 1.0
    dephasing_factor = 1.0
    coherent_sector_kind = (
        "NONE" if topology == "NONE" else "LEGACY_OCCUPANCY_PLACEHOLDER"
    )
    multi_particle_readout_status = TWO_PARTICLE_READOUT_NOT_APPLICABLE
    if multi_cell_channel is not None:
        multi_cell_channel.validate()
        if multi_cell_channel.n_cell != len(event.occupancy):
            raise ValueError("multi-cell channel and event occupancy dimensions differ")
        projection = (
            multi_cell_channel.project_single_excitation()
            if multi_cell_projection is None
            else multi_cell_projection
        )
        if projection.storage_kernel.shape != (
            len(event.occupancy),
            len(event.occupancy),
        ):
            raise ValueError(
                "multi-cell projection and event occupancy dimensions differ"
            )
        storage_kernel = projection.storage_kernel
        survival_by_cell = tuple(float(value.real) for value in np.diag(storage_kernel))
        survival = float(np.mean(survival_by_cell)) if survival_by_cell else 1.0
        # This field remains a bounded one-body visibility summary.  Population
        # survival above is intentionally independent of this off-diagonal term.
        dephasing_by_cell = (float(multi_cell_channel.off_diagonal_visibility),) * len(
            event.occupancy
        )
        dephasing_factor = float(multi_cell_channel.off_diagonal_visibility)
        selected_pre_splitter_probability = projection.selected_pre_splitter_probability
        retrieval_overlap = projection.selected_mode_probability
        evolution_kernel = "MULTI_CELL_PSD_AOD_UNBOUND"
        # Occupancy remains truth-private sector bookkeeping.  It must not
        # collapse the collective write mode before the AOD projection.
        coherent_sector_kind = (
            "NONE" if topology == "NONE" else "SAME_COLLECTIVE_MODE_FOCK_HYPOTHESIS"
        )
        if sum(event.occupancy) >= 2:
            # For the declared same-collective-mode n-Fock hypothesis, a
            # passive lossy linear channel with selected one-photon
            # probability q has K~Binomial(n,q) selected photons.  This is
            # deliberately not a local/cross occupancy propagator: source
            # marks remain truth-private bookkeeping only.
            multi_particle_readout_status = SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS
    # A phase draw is retained per event.  Local and cross events remain
    # distinct even when their aggregate photon count is two.
    phase = float(event.phase_rad + rng.normal(scale=0.0))
    state = SpinwaveEventState(
        event_id=event.event_id,
        occupancy=event.occupancy,
        topology=topology,
        phase_rad=phase,
        wavevector=event.wavevector,
        storage_wait_s=wait,
        survival_probability=survival,
        dephasing_factor=dephasing_factor,
        retrieval_overlap=retrieval_overlap,
        # The event-level spin-wave law is still a topology-preserving
        # placeholder.  A registered detector/source receipt cannot also
        # register this separate propagation kernel.
        parameter_status="HYPOTHESIS_UNBOUND",
        survival_by_cell=survival_by_cell,
        dephasing_by_cell=dephasing_by_cell,
        evolution_kernel=evolution_kernel,
        selected_pre_splitter_probability=selected_pre_splitter_probability,
        coherent_sector_kind=coherent_sector_kind,
        multi_particle_readout_status=multi_particle_readout_status,
    )
    state.validate()
    return state


__all__ = [
    "SPINWAVE_SCHEMA",
    "TWO_PARTICLE_READOUT_DEFERRED",
    "TWO_PARTICLE_READOUT_NOT_APPLICABLE",
    "SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS",
    "SpinwaveEventState",
    "propagate_spinwave_event",
]
