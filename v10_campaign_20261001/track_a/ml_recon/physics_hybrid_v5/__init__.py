"""Physics Hybrid V5 research contracts and preflight utilities.

V5 is a research-only namespace. It does not import the experiment runner,
write AWG matrices, or authorize hardware actions.
"""

from .contracts import (
    PhysicsHybridV5DataContract,
    PhysicsHybridV5ExperimentContract,
    validate_data_contract,
    validate_experiment_contract,
)
from .digital_twin import RegisteredCountDigitalTwin, RegisteredTwinParameters
from .command_contract import PhysicsHybridCommandTableV1
from .detector_click_model import MultiClickObservationV2
from .projective_action import ProjectiveAction

__all__ = [
    "PhysicsHybridV5DataContract",
    "PhysicsHybridV5ExperimentContract",
    "validate_data_contract",
    "validate_experiment_contract",
    "RegisteredCountDigitalTwin",
    "RegisteredTwinParameters",
    "PhysicsHybridCommandTableV1",
    "MultiClickObservationV2",
    "ProjectiveAction",
]
