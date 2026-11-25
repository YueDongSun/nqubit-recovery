"""Detached, source-visible supervision support for physical objectives.

Target availability is intentionally separate from the support on which an
objective is effective.  A perfect numerical fit still has effective support;
coverage is never inferred from a scalar loss value.
"""

from __future__ import annotations

from typing import Mapping

import torch


OBJECTIVE_SUPERVISION_SCHEMA = "V10PhysicalObjectiveSupervisionV1"
OBJECTIVE_SUPERVISION_SCHEMA_V2 = "V10PhysicalObjectiveSupervisionV2"
TARGET_AVAILABILITY_KEYS = (
    "target_mode_mask",
    "target_sector_mask",
    "target_measurement_mask",
)
EFFECTIVE_MASK_KEYS = (
    "state_rows",
    "covariance_rows",
    "sector_rows",
    "measurement_operations",
)
COVERAGE_UNIT_ROWS = "ROWS"
COVERAGE_UNIT_OPERATIONS = "OPERATIONS"
COVERAGE_UNIT_MODE_ELEMENTS = "MODE_ELEMENTS"


def _require_bool_tensor(value: object, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
        raise ValueError(f"{name} must be a strict bool tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(value.shape)}")
    return value


def _detached_bool(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().to(dtype=torch.bool)


def _validate_measurement_support(
    measurement_mask: torch.Tensor, batch: Mapping[str, object]
) -> None:
    operation_mask = batch.get("operation_mask")
    if operation_mask is None:
        return
    operation_mask = _require_bool_tensor(
        operation_mask,
        name="operation_mask",
        shape=tuple(measurement_mask.shape),
    )
    if operation_mask.device != measurement_mask.device:
        raise ValueError("operation_mask and target_measurement_mask device mismatch")
    if torch.any(measurement_mask & ~operation_mask):
        raise ValueError("target_measurement_mask exceeds available operations")


def build_physical_objective_supervision(
    batch: Mapping[str, object], belief_valid_mask: torch.Tensor
) -> dict[str, object]:
    """Build the one canonical physical-loss support packet for a batch."""
    mode_mask_raw = batch.get("target_mode_mask")
    if not isinstance(mode_mask_raw, torch.Tensor) or mode_mask_raw.ndim != 2:
        raise ValueError("target_mode_mask must be a rank-2 strict bool tensor")
    batch_size, n_mode = mode_mask_raw.shape
    mode_mask = _require_bool_tensor(
        mode_mask_raw, name="target_mode_mask", shape=(batch_size, n_mode)
    )
    sector_mask = _require_bool_tensor(
        batch.get("target_sector_mask"), name="target_sector_mask", shape=(batch_size,)
    )
    measurement_raw = batch.get("target_measurement_mask")
    if not isinstance(measurement_raw, torch.Tensor) or measurement_raw.ndim != 2:
        raise ValueError("target_measurement_mask must be a rank-2 strict bool tensor")
    measurement_mask = _require_bool_tensor(
        measurement_raw,
        name="target_measurement_mask",
        shape=(batch_size, measurement_raw.shape[1]),
    )
    belief_valid = _require_bool_tensor(
        belief_valid_mask, name="belief_valid_mask", shape=(batch_size,)
    )
    tensors = (mode_mask, sector_mask, measurement_mask, belief_valid)
    if any(value.device != mode_mask.device for value in tensors):
        raise ValueError("physical objective supervision tensors must share a device")
    _validate_measurement_support(measurement_mask, batch)
    state_rows = mode_mask.all(dim=-1)
    if batch.get("operation_input_recipe") == "operation_poisson_v1":
        operation_mask = _require_bool_tensor(batch.get("operation_mask"), name="operation_mask", shape=tuple(measurement_mask.shape))
        query_mask = _require_bool_tensor(batch.get("likelihood_target_mask"), name="likelihood_target_mask", shape=tuple(measurement_mask.shape))
        observed_mask = _require_bool_tensor(batch.get("observed_count_mask"), name="observed_count_mask", shape=tuple(measurement_mask.shape))
        intensity_mask = _require_bool_tensor(batch.get("target_effective_mu_o_mask"), name="target_effective_mu_o_mask", shape=tuple(measurement_mask.shape))
        rate_mask = _require_bool_tensor(batch.get("target_excitation_rate_o_mask"), name="target_excitation_rate_o_mask", shape=tuple(measurement_mask.shape))
        sector_operation_mask = _require_bool_tensor(batch.get("target_sector_probability_mask"), name="target_sector_probability_mask", shape=tuple(measurement_mask.shape))
        topology_mask = _require_bool_tensor(batch.get("target_k2_topology_mask"), name="target_k2_topology_mask", shape=tuple(measurement_mask.shape))
        if bool(topology_mask.any()):
            raise ValueError(
                "K2 topology targets are NOT_AVAILABLE without registered observables"
            )
        if (
            torch.any(query_mask & observed_mask)
            or not torch.equal(query_mask | observed_mask, operation_mask)
            or not torch.equal(measurement_mask, query_mask)
        ):
            raise ValueError(
                "query and observed masks must be a disjoint partition of operations"
            )
        packet = {
            "schema": OBJECTIVE_SUPERVISION_SCHEMA_V2,
            "target_availability": {"target_mode_mask": _detached_bool(mode_mask), "target_sector_mask": _detached_bool(sector_mask), "target_measurement_mask": _detached_bool(measurement_mask), "query_operations": _detached_bool(query_mask), "intensity_operations": _detached_bool(intensity_mask), "rate_operations": _detached_bool(rate_mask), "sector_operations": _detached_bool(sector_operation_mask), "topology_operations": _detached_bool(topology_mask)},
            "effective_masks": {"state_rows": _detached_bool(state_rows), "covariance_rows": _detached_bool(state_rows & belief_valid), "sector_rows": _detached_bool(sector_mask), "query_operations": _detached_bool(query_mask), "intensity_operations": _detached_bool(intensity_mask), "rate_operations": _detached_bool(rate_mask), "sector_operations": _detached_bool(sector_operation_mask), "topology_operations": _detached_bool(topology_mask)},
            "belief_valid_mask": _detached_bool(belief_valid),
        }
        return validate_physical_objective_supervision(packet, batch)
    packet = {
        "schema": OBJECTIVE_SUPERVISION_SCHEMA,
        "target_availability": {
            "target_mode_mask": _detached_bool(mode_mask),
            "target_sector_mask": _detached_bool(sector_mask),
            "target_measurement_mask": _detached_bool(measurement_mask),
        },
        "effective_masks": {
            "state_rows": _detached_bool(state_rows),
            "covariance_rows": _detached_bool(state_rows & belief_valid),
            "sector_rows": _detached_bool(sector_mask),
            "measurement_operations": _detached_bool(measurement_mask),
        },
        "belief_valid_mask": _detached_bool(belief_valid),
    }
    return validate_physical_objective_supervision(packet, batch)


def validate_physical_objective_supervision(
    packet: Mapping[str, object], batch: Mapping[str, object]
) -> dict[str, object]:
    if packet.get("schema") == OBJECTIVE_SUPERVISION_SCHEMA_V2:
        if set(packet) != {
            "schema",
            "target_availability",
            "effective_masks",
            "belief_valid_mask",
        }:
            raise ValueError("physical objective V2 packet field mismatch")
        availability = packet.get("target_availability")
        effective = packet.get("effective_masks")
        required = {"target_mode_mask", "target_sector_mask", "target_measurement_mask", "query_operations", "intensity_operations", "rate_operations", "sector_operations", "topology_operations"}
        if not isinstance(availability, Mapping) or not isinstance(effective, Mapping) or set(availability) != required or set(effective) != {"state_rows", "covariance_rows", "sector_rows", "query_operations", "intensity_operations", "rate_operations", "sector_operations", "topology_operations"}:
            raise ValueError("physical objective V2 mask contract is incomplete")
        mode_raw = batch.get("target_mode_mask")
        measurement_raw = batch.get("target_measurement_mask")
        if (
            not isinstance(mode_raw, torch.Tensor)
            or mode_raw.ndim != 2
            or not isinstance(measurement_raw, torch.Tensor)
            or measurement_raw.ndim != 2
        ):
            raise ValueError("physical objective V2 target masks have invalid rank")
        batch_size, n_mode = mode_raw.shape
        operation_shape = tuple(measurement_raw.shape)
        mode_mask = _require_bool_tensor(
            mode_raw, name="target_mode_mask", shape=(batch_size, n_mode)
        )
        sector_mask = _require_bool_tensor(
            batch.get("target_sector_mask"),
            name="target_sector_mask",
            shape=(batch_size,),
        )
        measurement_mask = _require_bool_tensor(
            measurement_raw,
            name="target_measurement_mask",
            shape=operation_shape,
        )
        operation_mask = _require_bool_tensor(
            batch.get("operation_mask"),
            name="operation_mask",
            shape=operation_shape,
        )
        query_mask = _require_bool_tensor(
            batch.get("likelihood_target_mask"),
            name="likelihood_target_mask",
            shape=operation_shape,
        )
        observed_mask = _require_bool_tensor(
            batch.get("observed_count_mask"),
            name="observed_count_mask",
            shape=operation_shape,
        )
        intensity_mask = _require_bool_tensor(
            batch.get("target_effective_mu_o_mask"),
            name="target_effective_mu_o_mask",
            shape=operation_shape,
        )
        rate_mask = _require_bool_tensor(
            batch.get("target_excitation_rate_o_mask"),
            name="target_excitation_rate_o_mask",
            shape=operation_shape,
        )
        sector_operation_mask = _require_bool_tensor(
            batch.get("target_sector_probability_mask"),
            name="target_sector_probability_mask",
            shape=operation_shape,
        )
        topology_mask = _require_bool_tensor(
            batch.get("target_k2_topology_mask"),
            name="target_k2_topology_mask",
            shape=operation_shape,
        )
        if bool(topology_mask.any()):
            raise ValueError(
                "K2 topology targets are NOT_AVAILABLE without registered observables"
            )
        belief_valid = _require_bool_tensor(
            packet.get("belief_valid_mask"),
            name="belief_valid_mask",
            shape=(batch_size,),
        )
        source_tensors = (
            mode_mask,
            sector_mask,
            measurement_mask,
            operation_mask,
            query_mask,
            observed_mask,
            intensity_mask,
            rate_mask,
            sector_operation_mask,
            topology_mask,
            belief_valid,
        )
        if any(value.device != mode_mask.device for value in source_tensors):
            raise ValueError("physical objective V2 masks must share one device")
        if (
            torch.any(query_mask & observed_mask)
            or not torch.equal(query_mask | observed_mask, operation_mask)
            or not torch.equal(measurement_mask, query_mask)
            or any(
                torch.any(value & ~operation_mask)
                for value in (intensity_mask, rate_mask, sector_operation_mask, topology_mask)
            )
        ):
            raise ValueError("physical objective V2 source mask relation is invalid")
        expected = {
            "target_mode_mask": mode_mask,
            "target_sector_mask": sector_mask,
            "target_measurement_mask": measurement_mask,
            "query_operations": query_mask,
            "intensity_operations": intensity_mask,
            "rate_operations": rate_mask,
            "sector_operations": sector_operation_mask,
            "topology_operations": topology_mask,
            "state_rows": mode_mask.all(dim=-1),
            "covariance_rows": mode_mask.all(dim=-1) & belief_valid,
            "sector_rows": sector_mask,
        }
        for name, value in availability.items():
            checked = _require_bool_tensor(
                value, name=name, shape=tuple(expected[name].shape)
            )
            if (
                checked.requires_grad
                or checked.grad_fn is not None
                or checked.device != mode_mask.device
                or not torch.equal(checked, expected[name])
            ):
                raise ValueError(f"physical objective V2 availability mismatch: {name}")
        for name, value in effective.items():
            checked = _require_bool_tensor(
                value, name=name, shape=tuple(expected[name].shape)
            )
            if (
                checked.requires_grad
                or checked.grad_fn is not None
                or checked.device != mode_mask.device
                or not torch.equal(checked, expected[name])
            ):
                raise ValueError(f"physical objective V2 effective mask mismatch: {name}")
        if belief_valid.requires_grad or belief_valid.grad_fn is not None:
            raise ValueError("physical objective V2 belief mask must be detached")
        return {
            "schema": OBJECTIVE_SUPERVISION_SCHEMA_V2,
            "target_availability": dict(availability),
            "effective_masks": dict(effective),
            "belief_valid_mask": belief_valid,
        }
    """Reject altered, non-detached, or shape-incompatible supervision support."""
    if set(packet) != {
        "schema", "target_availability", "effective_masks", "belief_valid_mask"
    } or packet.get("schema") != OBJECTIVE_SUPERVISION_SCHEMA:
        raise ValueError("physical objective supervision packet schema mismatch")
    availability = packet.get("target_availability")
    effective = packet.get("effective_masks")
    if not isinstance(availability, Mapping) or set(availability) != set(TARGET_AVAILABILITY_KEYS):
        raise ValueError("physical objective target availability keys mismatch")
    if not isinstance(effective, Mapping) or set(effective) != set(EFFECTIVE_MASK_KEYS):
        raise ValueError("physical objective effective mask keys mismatch")
    mode_raw = batch.get("target_mode_mask")
    if not isinstance(mode_raw, torch.Tensor) or mode_raw.ndim != 2:
        raise ValueError("target_mode_mask must be a rank-2 strict bool tensor")
    batch_size, n_mode = mode_raw.shape
    mode_mask = _require_bool_tensor(mode_raw, name="target_mode_mask", shape=(batch_size, n_mode))
    sector_mask = _require_bool_tensor(
        batch.get("target_sector_mask"), name="target_sector_mask", shape=(batch_size,)
    )
    measurement_raw = batch.get("target_measurement_mask")
    if not isinstance(measurement_raw, torch.Tensor) or measurement_raw.ndim != 2:
        raise ValueError("target_measurement_mask must be a rank-2 strict bool tensor")
    measurement_mask = _require_bool_tensor(
        measurement_raw, name="target_measurement_mask", shape=(batch_size, measurement_raw.shape[1])
    )
    belief_valid = _require_bool_tensor(
        packet.get("belief_valid_mask"), name="belief_valid_mask", shape=(batch_size,)
    )
    packet_tensors = (
        availability["target_mode_mask"],
        availability["target_sector_mask"],
        availability["target_measurement_mask"],
        effective["state_rows"],
        effective["covariance_rows"],
        effective["sector_rows"],
        effective["measurement_operations"],
        belief_valid,
    )
    if any(
        not isinstance(value, torch.Tensor) or value.device != mode_mask.device
        for value in packet_tensors
    ):
        raise ValueError("physical objective supervision packet device mismatch")
    if sector_mask.device != mode_mask.device or measurement_mask.device != mode_mask.device:
        raise ValueError("physical objective supervision tensors must share a device")
    _validate_measurement_support(measurement_mask, batch)
    expected = {
        "target_mode_mask": mode_mask,
        "target_sector_mask": sector_mask,
        "target_measurement_mask": measurement_mask,
        "state_rows": mode_mask.all(dim=-1),
        "covariance_rows": mode_mask.all(dim=-1) & belief_valid,
        "sector_rows": sector_mask,
        "measurement_operations": measurement_mask,
    }
    for name in TARGET_AVAILABILITY_KEYS:
        value = _require_bool_tensor(availability[name], name=name, shape=tuple(expected[name].shape))
        if value.requires_grad or value.grad_fn is not None or not torch.equal(value, expected[name]):
            raise ValueError(f"physical objective target availability mismatch: {name}")
    for name in EFFECTIVE_MASK_KEYS:
        value = _require_bool_tensor(effective[name], name=name, shape=tuple(expected[name].shape))
        if value.requires_grad or value.grad_fn is not None or not torch.equal(value, expected[name]):
            raise ValueError(f"physical objective effective mask mismatch: {name}")
    if belief_valid.requires_grad or belief_valid.grad_fn is not None:
        raise ValueError("belief_valid_mask must be detached")
    return {
        "schema": OBJECTIVE_SUPERVISION_SCHEMA,
        "target_availability": dict(availability),
        "effective_masks": dict(effective),
        "belief_valid_mask": belief_valid,
    }


__all__ = [
    "COVERAGE_UNIT_MODE_ELEMENTS",
    "COVERAGE_UNIT_OPERATIONS",
    "COVERAGE_UNIT_ROWS",
    "EFFECTIVE_MASK_KEYS",
    "OBJECTIVE_SUPERVISION_SCHEMA",
    "OBJECTIVE_SUPERVISION_SCHEMA_V2",
    "TARGET_AVAILABILITY_KEYS",
    "build_physical_objective_supervision",
    "validate_physical_objective_supervision",
]
