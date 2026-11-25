"""Hash-bound identity for a single CSM forward/Jacobian instance."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import sha256_of


FORWARD_INSTANCE_SCHEMA = "V10CSMForwardInstanceV1"


def object_sha(value: object) -> str:
    if isinstance(value, str) and len(value) == 64:
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return sha256_of(value)


def forward_callable_sha(forward: object) -> str:
    """Hash callable source for provenance; instance identity adds its inputs."""
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        source = repr(forward)
    return object_sha(source)


@dataclass(frozen=True)
class CSMForwardInstanceContract:
    """Identity of all state, intervention, schedule, and metric inputs to J."""

    sample_id: str
    current_state_sha: str
    operation_point_sha: str
    intervention_sha: str
    intervention_parameter_sha: str
    feedback_measurement_schedule_sha: str
    shot_contract_sha: str
    u0_sha: str
    manifold_sha: str
    forward_code_sha: str
    covariance_sha: str

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("forward instance requires a non-empty sample_id")
        for name, value in asdict(self).items():
            if name == "sample_id":
                continue
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {"schema": FORWARD_INSTANCE_SCHEMA, **asdict(self)}

    def sha256(self) -> str:
        return sha256_of(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "instance_sha": self.sha256()}


def build_forward_instance_contract(
    *,
    sample_id: str,
    current_state: object,
    operation_point: object,
    intervention: object,
    intervention_parameters: object,
    feedback_measurement_schedule: object,
    shot_contract: object,
    u0: object,
    manifold: object,
    forward_code: object,
    covariance: object,
) -> CSMForwardInstanceContract:
    """Build an instance contract from raw values or already-computed hashes."""
    return CSMForwardInstanceContract(
        sample_id=str(sample_id),
        current_state_sha=object_sha(current_state),
        operation_point_sha=object_sha(operation_point),
        intervention_sha=object_sha(intervention),
        intervention_parameter_sha=object_sha(intervention_parameters),
        feedback_measurement_schedule_sha=object_sha(feedback_measurement_schedule),
        shot_contract_sha=object_sha(shot_contract),
        u0_sha=object_sha(u0),
        manifold_sha=object_sha(manifold),
        forward_code_sha=object_sha(forward_code),
        covariance_sha=object_sha(covariance),
    )

RESIDUAL_INSTANCE_SCHEMA = "V10CSMResidualInstanceV1"


@dataclass(frozen=True)
class CSMResidualInstanceContract:
    """Target/residual identity kept separate from target-independent geometry."""

    geometry_forward_instance_sha: str
    target_sha: str
    residual_sha: str
    metric_contract_sha: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {"schema": RESIDUAL_INSTANCE_SCHEMA, **asdict(self)}

    def sha256(self) -> str:
        return sha256_of(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "residual_instance_sha": self.sha256()}


def build_residual_instance_contract(
    *,
    geometry_forward_instance: object,
    target: object,
    residual: object,
    metric_contract: object,
) -> CSMResidualInstanceContract:
    return CSMResidualInstanceContract(
        geometry_forward_instance_sha=object_sha(geometry_forward_instance),
        target_sha=object_sha(target),
        residual_sha=object_sha(residual),
        metric_contract_sha=object_sha(metric_contract),
    )




ForwardInstanceContract = CSMForwardInstanceContract


__all__ = [
    "CSMForwardInstanceContract", "CSMResidualInstanceContract",
    "FORWARD_INSTANCE_SCHEMA", "RESIDUAL_INSTANCE_SCHEMA",
    "ForwardInstanceContract",
    "build_forward_instance_contract", "build_residual_instance_contract",
    "forward_callable_sha",
    "object_sha",
]
