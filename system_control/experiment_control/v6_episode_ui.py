"""Default-hidden operator panel for guarded V6 feedback episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


V6_EPISODE_ACTIONS = (
    "start_shadow",
    "start_gated_episode",
    "approve_candidate",
    "reject_candidate",
    "roll_back",
    "stop_keep_best",
    "restore_baseline",
)


@dataclass(frozen=True)
class V6EpisodeUiStateV1:
    status: str = "DISABLED"
    desired_target: Mapping[str, Any] = field(default_factory=dict)
    current_applied_control: Mapping[str, Any] = field(default_factory=dict)
    iteration_index: int = 0
    maximum_iterations: int = 5
    current_bundle_id: str = "none"
    current_bundle_hash: str = "none"
    best_bundle_id: str = "none"
    best_bundle_hash: str = "none"
    measurement_elapsed_s: float = 0.0
    measurement_window_s: float = 900.0
    subwindow_index: int = 0
    subwindow_count: int = 6
    proposed_action: Mapping[str, Any] = field(default_factory=dict)
    projected_action: Mapping[str, Any] = field(default_factory=dict)
    applied_action: Mapping[str, Any] = field(default_factory=dict)
    cumulative_action: Tuple[float, ...] = ()
    amplitude_actionable_mask: Tuple[bool, ...] = ()
    phase_actionable_mask: Tuple[bool, ...] = ()
    fisher_block_condition: Any = None
    posterior_std: Any = None
    candidate_utility: Any = None
    cycle_consistency: Any = None
    retrieval: Any = None
    multi_click_guard: Any = None
    command_hash: str = "none"
    waveform_hash: str = "none"
    approval: Mapping[str, Any] = field(default_factory=dict)
    rollback: Mapping[str, Any] = field(default_factory=dict)
    stop_keep_best: Mapping[str, Any] = field(default_factory=dict)
    restore_baseline: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "V6EpisodeUiStateV1":
        fields = cls.__dataclass_fields__
        values = {name: payload[name] for name in fields if name in payload}
        for name in (
            "cumulative_action", "amplitude_actionable_mask", "phase_actionable_mask",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def bundle_summary(self) -> str:
        return (
            f"current={self.current_bundle_id} [{self.current_bundle_hash}] | "
            f"best={self.best_bundle_id} [{self.best_bundle_hash}]"
        )

    def diagnostic_summary(self) -> str:
        values = asdict(self)
        names = (
            "proposed_action",
            "projected_action",
            "applied_action",
            "cumulative_action",
            "amplitude_actionable_mask",
            "phase_actionable_mask",
            "fisher_block_condition",
            "posterior_std",
            "candidate_utility",
            "cycle_consistency",
            "retrieval",
            "multi_click_guard",
            "command_hash",
            "waveform_hash",
            "approval",
            "rollback",
            "stop_keep_best",
            "restore_baseline",
        )
        return "\n".join(f"{name}: {values[name]}" for name in names)


class V6EpisodePanel(QGroupBox):
    """Read-mostly panel whose seven actions are all explicit button events."""

    action_requested = pyqtSignal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("V6 Guarded Feedback Episode", parent)
        self._state = V6EpisodeUiStateV1()
        self.setVisible(False)
        layout = QVBoxLayout(self)

        settings_layout = QFormLayout()
        self.checkpoint_path_edit = QLineEdit()
        self.normalization_path_edit = QLineEdit()
        self.config_path_edit = QLineEdit("ml_recon/configs/physics_hybrid_v4.json")
        self.output_dir_edit = QLineEdit("ml_recon/reports/v6_feedback_episodes")
        settings_layout.addRow("Checkpoint", self.checkpoint_path_edit)
        settings_layout.addRow("Normalization", self.normalization_path_edit)
        settings_layout.addRow("Config", self.config_path_edit)
        settings_layout.addRow("Episode output", self.output_dir_edit)
        layout.addLayout(settings_layout)

        summary_layout = QGridLayout()
        self.status_label = QLabel("Status: DISABLED")
        self.iteration_label = QLabel("Iteration: 0 / 5")
        self.target_label = QLabel("Desired target: none")
        self.control_label = QLabel("Applied control: none")
        self.bundle_label = QLabel(self._state.bundle_summary())
        for label in (
            self.status_label,
            self.iteration_label,
            self.target_label,
            self.control_label,
            self.bundle_label,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self.status_label, 0, 0)
        summary_layout.addWidget(self.iteration_label, 0, 1)
        summary_layout.addWidget(self.target_label, 1, 0, 1, 2)
        summary_layout.addWidget(self.control_label, 2, 0, 1, 2)
        summary_layout.addWidget(self.bundle_label, 3, 0, 1, 2)
        layout.addLayout(summary_layout)

        self.measurement_progress = QProgressBar()
        self.measurement_progress.setRange(0, 900)
        self.measurement_progress.setFormat("900 s window: %v / %m s")
        self.subwindow_progress = QProgressBar()
        self.subwindow_progress.setRange(0, 6)
        self.subwindow_progress.setFormat("150 s subwindows: %v / %m")
        layout.addWidget(self.measurement_progress)
        layout.addWidget(self.subwindow_progress)

        self.diagnostics_text = QTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMaximumHeight(220)
        self.diagnostics_text.setPlainText(self._state.diagnostic_summary())
        layout.addWidget(self.diagnostics_text)

        button_layout = QGridLayout()
        button_specs = (
            ("start_shadow", "Start Shadow"),
            ("start_gated_episode", "Start Gated Episode"),
            ("approve_candidate", "Approve Candidate"),
            ("reject_candidate", "Reject Candidate"),
            ("roll_back", "Roll Back"),
            ("stop_keep_best", "Stop and Keep Best"),
            ("restore_baseline", "Restore Baseline"),
        )
        self.action_buttons: Dict[str, QPushButton] = {}
        for index, (action, label) in enumerate(button_specs):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, name=action: self.action_requested.emit(name))
            self.action_buttons[action] = button
            button_layout.addWidget(button, index // 4, index % 4)
        layout.addLayout(button_layout)
        self._update_button_states()

    @property
    def state(self) -> V6EpisodeUiStateV1:
        return self._state

    def settings(self) -> Dict[str, str]:
        return {
            "checkpoint_path": self.checkpoint_path_edit.text().strip(),
            "normalization_path": self.normalization_path_edit.text().strip(),
            "config_path": self.config_path_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
        }

    def update_state(self, value: V6EpisodeUiStateV1 | Mapping[str, Any]) -> None:
        self._state = value if isinstance(value, V6EpisodeUiStateV1) else V6EpisodeUiStateV1.from_mapping(value)
        state = self._state
        self.status_label.setText(f"Status: {state.status}")
        self.iteration_label.setText(
            f"Iteration: {state.iteration_index} / {state.maximum_iterations}"
        )
        self.target_label.setText(f"Desired target: {dict(state.desired_target)}")
        self.control_label.setText(f"Applied control: {dict(state.current_applied_control)}")
        self.bundle_label.setText(state.bundle_summary())
        self.measurement_progress.setRange(0, max(1, int(round(state.measurement_window_s))))
        self.measurement_progress.setValue(max(0, int(round(state.measurement_elapsed_s))))
        self.subwindow_progress.setRange(0, max(1, int(state.subwindow_count)))
        self.subwindow_progress.setValue(max(0, int(state.subwindow_index)))
        self.diagnostics_text.setPlainText(state.diagnostic_summary())
        self._update_button_states()

    def _update_button_states(self) -> None:
        status = self._state.status
        terminal = status in {"STOPPED_KEEP_BEST", "ABORTED_RESTORED_BASELINE", "COMPLETED"}
        self.action_buttons["start_shadow"].setEnabled(status in {"DISABLED", "IDLE"})
        self.action_buttons["start_gated_episode"].setEnabled(status in {"DISABLED", "IDLE"})
        self.action_buttons["approve_candidate"].setEnabled(status == "WAITING_OPERATOR_APPROVAL")
        self.action_buttons["reject_candidate"].setEnabled(
            status in {"CANDIDATE_GUARDED", "WAITING_OPERATOR_APPROVAL"}
        )
        self.action_buttons["roll_back"].setEnabled(
            status in {"CANDIDATE_ACTIVATED", "CANDIDATE_MEASURING", "OUTCOME_EVALUATED"}
        )
        self.action_buttons["stop_keep_best"].setEnabled(not terminal and status not in {"DISABLED", "IDLE"})
        self.action_buttons["restore_baseline"].setEnabled(not terminal and status not in {"DISABLED", "IDLE"})


__all__ = [
    "V6_EPISODE_ACTIONS",
    "V6EpisodePanel",
    "V6EpisodeUiStateV1",
]
