"""Unified experiment orchestration for 3-qubit and 4-qubit feedback runs."""

from .awg_uploader import AwgUploader, FrontPanelAwgUploader, NoOpAwgUploader
from .data_source import MeasurementAccumulator, MockMeasurementSource
from .data_protocol import (
    CampaignManifestV1,
    SnapshotManifestV3,
    SnapshotManifestV4,
    load_snapshot_manifest,
)
from .data_ui_actions import DataUiPrepareResult, clear_start_data_ui, save_clear_start_data_ui
from .dataset_store import FeedbackDatasetStore
from .delta_manager import DeltaManager
from .fpga_programmer import (
    FpgaProgramResult,
    FpgaProgramTarget,
    find_quartus_pgm,
    fpga_bitstream_path,
    fpga_project_dir,
    program_fpga,
    resolve_fpga_target,
)
from .models import CalibrationDelta, ExperimentRunConfig, FeedbackPrediction, FeedbackSample, MeasurementSnapshot
from .sequence_service import SequenceService
from .ui_controllers import DataUiController, RealDataUiSource, SequenceUiController, UiActionResult
from .ui_readiness import ExperimentUiReadiness, build_experiment_readiness

__all__ = [
    "AwgUploader",
    "FrontPanelAwgUploader",
    "NoOpAwgUploader",
    "MeasurementAccumulator",
    "MockMeasurementSource",
    "CampaignManifestV1",
    "SnapshotManifestV3",
    "SnapshotManifestV4",
    "load_snapshot_manifest",
    "DataUiPrepareResult",
    "clear_start_data_ui",
    "save_clear_start_data_ui",
    "FeedbackDatasetStore",
    "DeltaManager",
    "FpgaProgramResult",
    "FpgaProgramTarget",
    "find_quartus_pgm",
    "fpga_bitstream_path",
    "fpga_project_dir",
    "program_fpga",
    "resolve_fpga_target",
    "CalibrationDelta",
    "ExperimentRunConfig",
    "FeedbackPrediction",
    "FeedbackSample",
    "MeasurementSnapshot",
    "ExperimentRunner",
    "ExperimentState",
    "SequenceService",
    "RealDataUiSource",
    "DataUiController",
    "UiActionResult",
    "SequenceUiController",
    "ExperimentUiReadiness",
    "build_experiment_readiness",
]


def __getattr__(name):
    if name in {"ExperimentRunner", "ExperimentState"}:
        from .runner import ExperimentRunner, ExperimentState
        return {"ExperimentRunner": ExperimentRunner, "ExperimentState": ExperimentState}[name]
    raise AttributeError(name)
