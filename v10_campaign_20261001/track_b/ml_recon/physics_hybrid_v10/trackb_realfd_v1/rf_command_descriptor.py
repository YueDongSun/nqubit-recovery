"""Raw, source-bound RF command descriptor for Track B measurement modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .contracts import canonical_sha256, require_sha256


_VECTOR_FIELDS = (
    "raw_amplitude", "relative_phase_sin", "relative_phase_cos", "frequency_hz",
    "frequency_offset_hz", "rf_voltage_or_dac", "per_channel_power",
)
_SCALAR_FIELDS = ("total_power", "maximum_power", "amplitude_imbalance")
_AUXILIARY_VECTOR_FIELDS = ("quadratic_terms", "cross_amplitude_terms", "cross_phase_terms")


def _optional_vector(values: Sequence[object], expected: int, name: str) -> tuple[float | None, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise ValueError(f"{name} must preserve one value per ordered tone")
    result: list[float | None] = []
    for value in values:
        if value is None:
            result.append(None)
        else:
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} values must be finite or explicitly unavailable")
            result.append(converted)
    return tuple(result)


def _optional_scalar(value: object, name: str) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite or explicitly unavailable")
    return converted


@dataclass(frozen=True)
class RawRFCommandDescriptorV1:
    """Absolute RF values plus availability masks; missing values are never zero-filled."""

    ordered_tone_ids: tuple[str, ...]
    write_read_roles: tuple[str, ...]
    raw_amplitude: tuple[float | None, ...]
    relative_phase_sin: tuple[float | None, ...]
    relative_phase_cos: tuple[float | None, ...]
    frequency_hz: tuple[float | None, ...]
    frequency_offset_hz: tuple[float | None, ...]
    rf_voltage_or_dac: tuple[float | None, ...]
    per_channel_power: tuple[float | None, ...]
    total_power: float | None
    maximum_power: float | None
    amplitude_imbalance: float | None
    quadratic_terms: tuple[float | None, ...]
    cross_amplitude_terms: tuple[float | None, ...]
    cross_phase_terms: tuple[float | None, ...]
    cycle_index: int
    operation_family: str
    within_cycle_index: int
    current_operating_point_id: str | None
    command_table_sha: str
    rf_command_basis_sha: str
    field_availability_masks: tuple[tuple[str, tuple[bool, ...]], ...]
    schema: str = field(default="RawRFCommandDescriptorV1", init=False)

    def __post_init__(self) -> None:
        count = len(self.ordered_tone_ids)
        if count == 0 or len(set(self.ordered_tone_ids)) != count:
            raise ValueError("raw RF descriptor requires a unique ordered tone list")
        if any(type(value) is not str or not value for value in self.ordered_tone_ids):
            raise ValueError("tone identifiers must be nonempty strings")
        if len(self.write_read_roles) != count or any(value not in ("WRITE", "READ", "UNKNOWN") for value in self.write_read_roles):
            raise ValueError("each tone needs an explicit WRITE/READ/UNKNOWN role")
        for name in _VECTOR_FIELDS:
            object.__setattr__(self, name, _optional_vector(getattr(self, name), count, name))
        for name in _SCALAR_FIELDS:
            object.__setattr__(self, name, _optional_scalar(getattr(self, name), name))
        for name in _AUXILIARY_VECTOR_FIELDS:
            values = _optional_vector(getattr(self, name), len(getattr(self, name)), name)
            object.__setattr__(self, name, values)
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("cycle_index must be a nonnegative integer")
        if type(self.within_cycle_index) is not int or self.within_cycle_index < 0:
            raise ValueError("within_cycle_index must be a nonnegative integer")
        if type(self.operation_family) is not str or not self.operation_family:
            raise ValueError("operation_family must be explicit")
        if self.current_operating_point_id is not None and (
            type(self.current_operating_point_id) is not str or not self.current_operating_point_id
        ):
            raise ValueError("current operating-point identity must be explicit when available")
        require_sha256(self.command_table_sha, "command_table_sha")
        require_sha256(self.rf_command_basis_sha, "rf_command_basis_sha")
        masks = dict(self.field_availability_masks)
        expected_names = set(_VECTOR_FIELDS + _SCALAR_FIELDS + _AUXILIARY_VECTOR_FIELDS)
        if set(masks) != expected_names or len(masks) != len(self.field_availability_masks):
            raise ValueError("RF descriptor requires exact availability masks for every raw field")
        for name in _VECTOR_FIELDS:
            expected_mask = tuple(value is not None for value in getattr(self, name))
            if tuple(masks[name]) != expected_mask:
                raise ValueError(f"availability mask for {name} does not match source values")
        for name in _SCALAR_FIELDS:
            if tuple(masks[name]) != (getattr(self, name) is not None,):
                raise ValueError(f"availability mask for {name} does not match source values")
        for name in _AUXILIARY_VECTOR_FIELDS:
            expected_mask = tuple(value is not None for value in getattr(self, name))
            if tuple(masks[name]) != expected_mask:
                raise ValueError(f"availability mask for {name} does not match source values")
        for sine, cosine in zip(self.relative_phase_sin, self.relative_phase_cos, strict=True):
            if (sine is None) != (cosine is None):
                raise ValueError("relative phase sine/cosine availability must be paired")
            if sine is not None and not math.isclose(sine * sine + cosine * cosine, 1.0, abs_tol=2e-5):
                raise ValueError("relative phase sine/cosine must represent a unit circle")

    @classmethod
    def from_raw(
        cls,
        *,
        ordered_tone_ids: Sequence[str],
        write_read_roles: Sequence[str],
        raw_amplitude: Sequence[object],
        relative_phase_sin: Sequence[object],
        relative_phase_cos: Sequence[object],
        frequency_hz: Sequence[object],
        frequency_offset_hz: Sequence[object],
        rf_voltage_or_dac: Sequence[object],
        per_channel_power: Sequence[object],
        total_power: object,
        maximum_power: object,
        amplitude_imbalance: object,
        cycle_index: int,
        operation_family: str,
        within_cycle_index: int,
        command_table_sha: str,
        rf_command_basis_sha: str,
        quadratic_terms: Sequence[object] = (),
        cross_amplitude_terms: Sequence[object] = (),
        cross_phase_terms: Sequence[object] = (),
        current_operating_point_id: str | None = None,
    ) -> "RawRFCommandDescriptorV1":
        tone_count = len(tuple(ordered_tone_ids))
        vector_values = {
            "raw_amplitude": tuple(raw_amplitude),
            "relative_phase_sin": tuple(relative_phase_sin),
            "relative_phase_cos": tuple(relative_phase_cos),
            "frequency_hz": tuple(frequency_hz),
            "frequency_offset_hz": tuple(frequency_offset_hz),
            "rf_voltage_or_dac": tuple(rf_voltage_or_dac),
            "per_channel_power": tuple(per_channel_power),
        }
        masks = {
            name: tuple(value is not None for value in values)
            for name, values in vector_values.items()
        }
        masks.update({
            "total_power": (total_power is not None,),
            "maximum_power": (maximum_power is not None,),
            "amplitude_imbalance": (amplitude_imbalance is not None,),
        })
        auxiliary_values = {
            "quadratic_terms": tuple(quadratic_terms),
            "cross_amplitude_terms": tuple(cross_amplitude_terms),
            "cross_phase_terms": tuple(cross_phase_terms),
        }
        masks.update({
            name: tuple(value is not None for value in values)
            for name, values in auxiliary_values.items()
        })
        return cls(
            ordered_tone_ids=tuple(ordered_tone_ids),
            write_read_roles=tuple(write_read_roles),
            raw_amplitude=vector_values["raw_amplitude"],
            relative_phase_sin=vector_values["relative_phase_sin"],
            relative_phase_cos=vector_values["relative_phase_cos"],
            frequency_hz=vector_values["frequency_hz"],
            frequency_offset_hz=vector_values["frequency_offset_hz"],
            rf_voltage_or_dac=vector_values["rf_voltage_or_dac"],
            per_channel_power=vector_values["per_channel_power"],
            total_power=total_power,
            maximum_power=maximum_power,
            amplitude_imbalance=amplitude_imbalance,
            quadratic_terms=auxiliary_values["quadratic_terms"],
            cross_amplitude_terms=auxiliary_values["cross_amplitude_terms"],
            cross_phase_terms=auxiliary_values["cross_phase_terms"],
            cycle_index=cycle_index,
            operation_family=operation_family,
            within_cycle_index=within_cycle_index,
            current_operating_point_id=current_operating_point_id,
            command_table_sha=command_table_sha,
            rf_command_basis_sha=rf_command_basis_sha,
            field_availability_masks=tuple((name, tuple(mask)) for name, mask in sorted(masks.items())),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ordered_tone_ids": list(self.ordered_tone_ids),
            "write_read_roles": list(self.write_read_roles),
            **{name: list(getattr(self, name)) for name in _VECTOR_FIELDS},
            **{name: getattr(self, name) for name in _SCALAR_FIELDS},
            "quadratic_terms": list(self.quadratic_terms),
            "cross_amplitude_terms": list(self.cross_amplitude_terms),
            "cross_phase_terms": list(self.cross_phase_terms),
            "cycle_index": self.cycle_index,
            "operation_family": self.operation_family,
            "within_cycle_index": self.within_cycle_index,
            "current_operating_point_id": self.current_operating_point_id,
            "command_table_sha": self.command_table_sha,
            "rf_command_basis_sha": self.rf_command_basis_sha,
            "field_availability_masks": {name: list(mask) for name, mask in self.field_availability_masks},
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = ["RawRFCommandDescriptorV1"]
