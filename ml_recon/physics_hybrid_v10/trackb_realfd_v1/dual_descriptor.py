"""Typed pairing of projector geometry and absolute raw RF command descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import canonical_sha256
from .projector_descriptor import IdealProjectorDescriptorV2
from .rf_command_descriptor import RawRFCommandDescriptorV1


@dataclass(frozen=True)
class DualOperationDescriptorV1:
    """One operation with intentionally non-aliasing geometry and RF representations."""

    ideal_projector: IdealProjectorDescriptorV2
    raw_rf_command: RawRFCommandDescriptorV1
    schema: str = field(default="DualOperationDescriptorV1", init=False)

    def __post_init__(self) -> None:
        if self.ideal_projector.command_table_sha != self.raw_rf_command.command_table_sha:
            raise ValueError("projector and RF command must bind the same command table")
        if self.ideal_projector.cycle_index != self.raw_rf_command.cycle_index:
            raise ValueError("projector and RF command cycle indices disagree")
        if self.ideal_projector.within_cycle_index != self.raw_rf_command.within_cycle_index:
            raise ValueError("projector and RF command within-cycle indices disagree")
        if self.ideal_projector.operation_family != self.raw_rf_command.operation_family:
            raise ValueError("projector and RF command operation families disagree")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "projector_sha": self.ideal_projector.projector_sha,
            "raw_rf_command_sha": self.raw_rf_command.sha256(),
            "operation_registry_sha": self.ideal_projector.operation_registry_sha,
            "command_table_sha": self.ideal_projector.command_table_sha,
        })


__all__ = ["DualOperationDescriptorV1"]
