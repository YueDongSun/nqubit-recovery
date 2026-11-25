"""Hash-bound identity for a single CSM forward/Jacobian instance."""

from __future__ import annotations

import inspect
import hashlib
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from types import ModuleType
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
    """Bind callable code, defaults, captured state, and referenced constants."""
    function = forward.__func__ if inspect.ismethod(forward) else forward
    if not inspect.isfunction(function):
        raise ValueError("bound CSM forward must be an inspectable Python function")
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = function.__code__.co_code.hex()
    closure = inspect.getclosurevars(function)
    global_state = {}
    for name, value in sorted(closure.globals.items()):
        if inspect.isfunction(value):
            try:
                helper_source = inspect.getsource(value)
            except (OSError, TypeError):
                helper_source = value.__code__.co_code.hex()
            global_state[name] = {
                "function": value.__module__ + "." + value.__qualname__,
                "source_sha": object_sha(helper_source),
                "defaults_sha": runtime_value_sha(value.__defaults__),
            }
        else:
            global_state[name] = {"value_sha": runtime_value_sha(value)}
    return object_sha({
        "schema": "V10BoundForwardCallableV2",
        "source_sha": object_sha(source),
        "defaults_sha": runtime_value_sha(function.__defaults__),
        "keyword_defaults_sha": runtime_value_sha(function.__kwdefaults__),
        "closure_sha": runtime_value_sha(closure.nonlocals),
        "globals": global_state,
        "bound_self_sha": (runtime_value_sha(forward.__self__)
                           if inspect.ismethod(forward) else None),
    })


def runtime_value_sha(value: object) -> str:
    """Hash explicit numerical runtime state without pointer-based repr values."""
    def normalize(item: object) -> object:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not np.isfinite(item):
                raise ValueError("bound runtime state must be finite")
            return item
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject:
                raise ValueError("bound runtime arrays cannot have object dtype")
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError("bound runtime arrays must be finite")
            return {"dtype": array.dtype.str, "shape": list(array.shape),
                    "bytes_sha": hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(item, Enum):
            return {"enum": type(item).__qualname__, "value": normalize(item.value)}
        if isinstance(item, ModuleType):
            return {"module": item.__name__}
        if isinstance(item, type) or inspect.isbuiltin(item):
            return {"symbol": getattr(item, "__module__", "") + "."
                    + getattr(item, "__qualname__", getattr(item, "__name__", ""))}
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("bound runtime mappings require string keys")
            return {key: normalize(child) for key, child in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        if is_dataclass(item) and not isinstance(item, type):
            return {"dataclass": type(item).__module__ + "." + type(item).__qualname__,
                    "fields": {field.name: normalize(getattr(item, field.name))
                               for field in fields(item)}}
        if inspect.isfunction(item) or inspect.ismethod(item):
            return {"callable_sha": forward_callable_sha(item)}
        raise ValueError(f"unsupported mutable bound runtime state: {type(item).__name__}")

    return object_sha(normalize(value))


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
    "runtime_value_sha",
    "object_sha",
]
