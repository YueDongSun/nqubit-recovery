"""Versioned bounded V2 runtime using the existing admitted training loop."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from .effective_observation_loss import Effective_observation_loss_config, effective_observation_objective
from .effective_observation_v2 import Effective_singleton_observation, validate_global_initialization_buffers
from .effective_family_context_v3 import (
    CHECKPOINT_SCHEMA as FAMILY_CHECKPOINT_SCHEMA,
    RECEIPT_SCHEMA as FAMILY_RECEIPT_SCHEMA,
    MODEL_CONFIG_SCHEMA as FAMILY_CONFIG_SCHEMA,
)
from .contracts import canonical_sha256
from .pilot_diagnostics import Pilot_diagnostic_config, Pilot_diagnostic_recorder, capture_pilot_diagnostics
from .training import TrackBLongTrainingAuthorizationV1, TrackBRealRuntimeConfigV1, TrackBRealRuntimeIdentityV1, TrackBRealTrainingRuntimeV1


class Effective_observation_runtime(TrackBRealTrainingRuntimeV1):
    """Train the effective READ candidate, never reinterpret V1 checkpoints.

The base loop retains its exact source/admission/row/session/query binding,
step budget, rollback, AdamW scheduler and RNG ownership. This class changes
only the model/loss/checkpoint contract. It is not a long-training launcher or
a grant of hardware authority. Full configuration preimages are now retained.
"""

    CHECKPOINT_SCHEMA = "TrackBEffectiveObservationCheckpointV2"
    RECEIPT_SCHEMA = "TrackBEffectiveObservationTrainingReceiptV2"
    MODEL_CONFIG_SCHEMA = "TrackBEffectiveSingletonConfigV2"
    LOSS_CONFIG_SCHEMA = "TrackBEffectiveSingletonLossV2"

    def __init__(
        self, *, model: Effective_singleton_observation,
        identity: TrackBRealRuntimeIdentityV1, config: TrackBRealRuntimeConfigV1,
        loss_config: Effective_observation_loss_config = Effective_observation_loss_config(),
        diagnostic_config: Pilot_diagnostic_config = Pilot_diagnostic_config(),
        device: str = "cpu",
    ) -> None:
        if type(model) is not Effective_singleton_observation:
            raise TypeError("effective runtime requires the explicitly versioned V2 model")
        if model.config.measurement_context is not None and device != "cpu":
            raise ValueError("family-context V3 currently requires CPU execution for independent replay")
        if config.precision != "fp32":
            raise ValueError("effective complex-PSD candidate currently requires FP32 precision")
        diagnostic_config.validate()
        validate_global_initialization_buffers(model)
        if model.config.global_initialization is not None:
            from .global_count_initialization import validate_global_initialization_binding

            validate_global_initialization_binding(model.config.global_initialization, identity)
        self._diagnostic_config = diagnostic_config
        super().__init__(model=model, identity=identity, config=config, loss_weights=loss_config, device=device)

    def _assert_runtime_transition_semantics(self) -> None:
        super()._assert_runtime_transition_semantics()
        validate_global_initialization_buffers(self._model)

    def _execution_state_sha(self) -> str:
        payload = {
            "runtime_state_sha": super()._execution_state_sha(),
            "diagnostic_config": asdict(self._diagnostic_config),
        }
        if self._model.config.measurement_context is not None:
            validate_global_initialization_buffers(self._model)
            payload["measurement_context_sha"] = self._model.family_context.context.sha256()
        return canonical_sha256(payload)

    def _compute_training_loss(self, prediction: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
        losses = effective_observation_objective(prediction, batch, config=self._loss_weights)
        update = self.global_step + 1
        if self._diagnostic_config.enabled and update in self._diagnostic_config.requested_updates(self._config.optimizer_steps):
            losses["diagnostic_snapshot"] = capture_pilot_diagnostics(
                self._model, prediction, losses, batch, loss_config=self._loss_weights,
                identity_sha=self._identity.sha256(), runtime_config_sha=self._config.sha256(), update_step=update,
            )
        return losses

    def _validate_batch_identity(self, raw_batch: Mapping[str, Any]) -> None:
        super()._validate_batch_identity(raw_batch)
        if self._model.config.read_trials_source == "native_scalar":
            from .native_read_trials import validate_native_read_conservation

            validate_native_read_conservation(raw_batch)
        if self._model.config.global_initialization is not None:
            rows = {row[0]: row for row in self._model.config.global_initialization.train_rows}
            for index, sample in enumerate(raw_batch["sample_ids"]):
                if rows[sample][3:] != (bool(raw_batch["global_multi_mask"][index]), raw_batch["global_multi_scope"][index]):
                    raise ValueError("global initialization masks/scopes differ from the admitted batch")

    def _checkpoint_payload(self, *, completed: bool) -> dict[str, Any]:
        validate_global_initialization_buffers(self._model)
        payload = super()._checkpoint_payload(completed=completed)
        payload["metadata"].update({
            "runtime_config": asdict(self._config),
            "loss_config": asdict(self._loss_weights),
            "diagnostic_config": asdict(self._diagnostic_config),
            "response_scope": "CONDITIONAL_ON_NATIVE_EFFECTIVE_READ",
            "idler_event": "STICKY_OR_OF_EXACT_SINGLETON_OVER_19_ACCEPTED_CLOCKS",
        })
        return payload

    def _validate_checkpoint_extension(self, metadata: Mapping[str, Any]) -> None:
        expected = {
            "runtime_config": asdict(self._config),
            "loss_config": asdict(self._loss_weights),
            "diagnostic_config": asdict(self._diagnostic_config),
            "response_scope": "CONDITIONAL_ON_NATIVE_EFFECTIVE_READ",
            "idler_event": "STICKY_OR_OF_EXACT_SINGLETON_OVER_19_ACCEPTED_CLOCKS",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("effective checkpoint configuration preimages or event semantics differ")

    def run(
        self, batch_iterator_factory: Callable[[int], Iterable[Mapping[str, Any]]], *,
        checkpoint_path: str | Path, monitor_path: str | Path,
        long_authorization: TrackBLongTrainingAuthorizationV1 | None = None,
        approval_receipt_path: str | Path | None = None,
        diagnostics_dir: str | Path | None = None,
    ) -> dict[str, object]:
        recorder = None
        if self._diagnostic_config.enabled:
            if diagnostics_dir is None:
                raise ValueError("enabled diagnostics require a run-owned trajectory directory")
            recorder = Pilot_diagnostic_recorder(
                Path(diagnostics_dir), identity_sha=self._identity.sha256(), runtime_config_sha=self._config.sha256(), config=self._diagnostic_config,
                optimizer_steps=self._config.optimizer_steps, restored_step=self.global_step,
            )
        elif diagnostics_dir is not None:
            raise ValueError("a diagnostic directory cannot silently use a disabled observer")
        receipt = super().run(
            batch_iterator_factory, checkpoint_path=checkpoint_path, monitor_path=monitor_path,
            long_authorization=long_authorization, approval_receipt_path=approval_receipt_path,
            step_observer=None if recorder is None else recorder.record,
        )
        return {
            **receipt,
            "diagnostic_config": asdict(self._diagnostic_config),
            "diagnostic_records": [] if recorder is None else recorder.records,
        }


class Family_context_observation_runtime(Effective_observation_runtime):
    """Reuse bounded training mechanics under a distinct family-context ABI."""

    CHECKPOINT_SCHEMA = FAMILY_CHECKPOINT_SCHEMA
    RECEIPT_SCHEMA = FAMILY_RECEIPT_SCHEMA
    MODEL_CONFIG_SCHEMA = FAMILY_CONFIG_SCHEMA
