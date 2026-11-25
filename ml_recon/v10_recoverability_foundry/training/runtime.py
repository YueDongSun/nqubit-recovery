"""Thin Track A adapter over the authoritative V10 training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ml_recon.physics_hybrid_v10.track_a_n8.training_runtime import (
    V10Trainer,
    V10TrainingConfig,
    V10TrainingIdentity,
    V10TrainingObserver,
)

from ..config import TrackAConfig


@dataclass(frozen=True)
class TrackARuntimeAdapter:
    """Bind foundry configuration without implementing a second trainer.

    Seed setup, AMP/GradScaler behavior, atomic checkpoint replacement, strict
    checkpoint identity, RNG restoration, and resume state remain owned by
    :class:`V10Trainer`. The foundry smoke uses only seed and runtime metadata;
    it does not create a checkpoint or claim that resume/AMP were exercised.
    """

    config: TrackAConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.config.validate()
        selected_device = torch.device(self.device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("Track A runtime supports cpu or cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

    @property
    def selected_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def amp_active(self) -> bool:
        return bool(
            self.config.training.amp_enabled
            and self.selected_device.type == "cuda"
        )

    def seed_all(self) -> None:
        """Delegate Python, NumPy, CPU, and CUDA seed setup to V10Trainer."""

        V10Trainer._seed_all(self.config.data.global_seed)

    def training_config(
        self,
        *,
        model_id: str,
        output_dir: str,
        run_id: str = "",
    ) -> V10TrainingConfig:
        """Translate Track A configuration into the registered V10 schema."""

        smoke_mode = self.config.training.run_mode == "smoke"
        result = V10TrainingConfig(
            model_id=model_id,
            output_dir=output_dir,
            track_id="track_a",
            run_id=run_id,
            run_mode=self.config.training.run_mode,
            global_seed=self.config.data.global_seed,
            seed_panel=(self.config.data.global_seed,),
            batch_size=self.config.training.batch_size,
            epochs=self.config.training.epochs,
            learning_rate=self.config.training.learning_rate,
            amp_enabled=self.config.training.amp_enabled,
            max_train_batches=2 if smoke_mode else None,
            max_validation_batches=1 if smoke_mode else None,
            training_authorized=self.config.training.training_authorized,
        )
        result.validate()
        return result

    def build_trainer(
        self,
        *,
        model: torch.nn.Module,
        identity: V10TrainingIdentity,
        output_dir: str,
        step_function: Callable[
            [torch.nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]],
        prediction_function: Callable[
            [torch.nn.Module, Dict[str, Any]], Any],
        batch_sampler_factory: Optional[
            Callable[[Dataset, int, bool], Any]
        ] = None,
        observers: Optional[Sequence[V10TrainingObserver]] = None,
        run_id: str = "",
    ) -> V10Trainer:
        """Construct the existing V10Trainer; execution remains caller-owned."""

        return V10Trainer(
            model=model,
            config=self.training_config(
                model_id=str(model.model_id), output_dir=output_dir, run_id=run_id),
            identity=identity,
            step_function=step_function,
            prediction_function=prediction_function,
            device=str(self.selected_device),
            batch_sampler_factory=batch_sampler_factory,
            observers=observers,
        )

    def contract_report(self) -> Dict[str, object]:
        runtime_config = self.training_config(
            model_id="C_N_FULL_V10", output_dir="<caller_bound>")
        return {
            "adapter_schema": "V10TrackARuntimeAdapterV1",
            "base_runtime": (
                "ml_recon.physics_hybrid_v10.track_a_n8.training_runtime.V10Trainer"),
            "checkpoint_schema": "V10CheckpointV1",
            "seed_owner": "V10Trainer._seed_all",
            "amp_owner": "V10Trainer.amp_active_and_GradScaler",
            "atomic_checkpoint_owner": "V10Trainer._save_checkpoint_os_replace",
            "resume_owner": "V10Trainer._load_checkpoint",
            "global_seed": runtime_config.global_seed,
            "run_mode": runtime_config.run_mode,
            "batch_size": runtime_config.batch_size,
            "epochs": runtime_config.epochs,
            "amp_requested": runtime_config.amp_enabled,
            "amp_active": self.amp_active,
            "training_authorized": runtime_config.training_authorized,
            "ten_k_training_authorized": self.config.training.ten_k_training_authorized,
            "research_only": True,
        }
