"""The unchanged numerical training loop for admitted Track B pilot rows.

Callers own data admission, TRAIN-only normalization, source checks and artifact
publication. This module only constructs the model and optimizes prepared rows;
its return value is engineering data, never execution or promotion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.utils.data import DataLoader

from .inverse_transformer import (
    RAW_GAIN_PARAMETERIZATION_V1,
    TrackBInverseTransformerConfigV1,
    TrackBInverseTransformerDependencyV1,
    TrackBInverseTransformerV1,
)
from .ten_round_guidance import TEN_ROUND_COUNT

if TYPE_CHECKING:
    from .inverse_transformer_pilot import TrackBTargetNormalizationV1, TrackBV24TenRoundPilotConfigV1


@dataclass(frozen=True)
class Pilot_training_dependencies:
    """Resolve legacy helper names at the caller so existing test seams survive."""

    model_config_for_target_normalization: Callable[..., TrackBInverseTransformerConfigV1]
    model_factory: Callable[..., TrackBInverseTransformerV1]
    round_inputs: Callable[..., Any]
    logical_action_loss_components: Callable[..., Any]
    assign_training_gradients: Callable[..., Any]
    checked_optimizer_step: Callable[..., float]
    evaluate_teacher_forced: Callable[..., Any]
    round_coverage_batch_schedule: Callable[..., Any]
    measure_dependency: Callable[..., TrackBInverseTransformerDependencyV1]
    random_sampling_kind: str
    coverage_sampling_kind: str


@dataclass(frozen=True)
class Pilot_initialized_model:
    model: TrackBInverseTransformerV1
    model_config: TrackBInverseTransformerConfigV1
    device: torch.device


@dataclass(frozen=True)
class Pilot_training_result:
    """Owned numerical outputs; the caller must separately verify and publish."""

    model: TrackBInverseTransformerV1
    model_config: TrackBInverseTransformerConfigV1
    completed_updates: int
    round_schedule: tuple[int, ...]
    batch_schedule: tuple[tuple[int, ...], ...] | None
    progress: tuple[dict[str, object], ...]
    evaluation: dict[str, object]
    guidance: tuple[dict[str, object], ...]
    dependency: TrackBInverseTransformerDependencyV1


def initialize_pilot_model(
    config: TrackBV24TenRoundPilotConfigV1,
    normalization: TrackBTargetNormalizationV1,
    dependencies: Pilot_training_dependencies,
) -> Pilot_initialized_model:
    """Preserve the legacy seed, construction and raw-gain bias ordering."""

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model_config = dependencies.model_config_for_target_normalization(config, normalization)
    model = dependencies.model_factory(model_config).to(device)
    if config.gain_parameterization == RAW_GAIN_PARAMETERIZATION_V1:
        with torch.no_grad():
            model.gain_head.bias.fill_(normalization.gain_center)
    return Pilot_initialized_model(model, model_config, device)


def train_pilot_core(
    initialized: Pilot_initialized_model,
    config: TrackBV24TenRoundPilotConfigV1,
    normalization: TrackBTargetNormalizationV1,
    train_dataset: Any,
    validation_dataset: Any,
    dependencies: Pilot_training_dependencies,
    *,
    on_progress: Callable[[dict[str, object]], None] | None = None,
) -> Pilot_training_result:
    """Run the original optimizer, batch schedule and teacher-forced readout."""

    model, model_config, device = initialized.model, initialized.model_config, initialized.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    batch_schedule: tuple[tuple[int, ...], ...] | None = None
    if config.training_sampling_kind == dependencies.random_sampling_kind:
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, generator=generator,
        )
        round_schedule = [index % TEN_ROUND_COUNT for index in range(config.optimizer_steps)]
        random.Random(config.seed).shuffle(round_schedule)
    elif config.training_sampling_kind == dependencies.coverage_sampling_kind:
        batch_schedule, round_schedule = dependencies.round_coverage_batch_schedule(
            dataset_size=len(train_dataset), batch_size=config.batch_size,
            optimizer_steps=config.optimizer_steps, seed=config.seed,
        )
        loader = DataLoader(train_dataset, batch_sampler=batch_schedule, num_workers=0)
    else:
        raise ValueError("Track B training sampling kind is unsupported")
    iterator = iter(loader)
    progress: list[dict[str, object]] = []
    completed_updates = 0
    model.train()
    for step in range(1, config.optimizer_steps + 1):
        if config.training_sampling_kind == dependencies.random_sampling_kind:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
        else:
            batch = next(iterator)
        round_index = round_schedule[step - 1]
        inputs, target_action, target_gain = dependencies.round_inputs(
            batch, round_index=round_index, device=device,
            source_view_kind=config.source_view_kind,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        action_loss, gain_loss, metrics = dependencies.logical_action_loss_components(
            output, target_action, target_gain, normalization, round_index=round_index,
        )
        loss = action_loss + config.gain_loss_weight * gain_loss
        metrics = {**metrics, "loss": float(loss.detach().cpu())}
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("ten-round inverse Transformer loss is non-finite")
        dependencies.assign_training_gradients(
            model, action_loss, gain_loss,
            gain_loss_weight=config.gain_loss_weight,
            gradient_composition_kind=config.gradient_composition_kind,
        )
        gradient_norm = dependencies.checked_optimizer_step(model, optimizer)
        completed_updates = step
        if step == 1 or step % 10 == 0 or step == config.optimizer_steps:
            record = {"optimizer_step": step, "round_index": round_index + 1, **metrics,
                      "gradient_norm_before_clip": gradient_norm}
            progress.append(record)
            if on_progress is not None:
                on_progress(record)
    evaluation, guidance = dependencies.evaluate_teacher_forced(
        model, validation_dataset, batch_size=config.batch_size, device=device,
        source_view_kind=config.source_view_kind,
    )
    first_batch = next(iter(DataLoader(
        validation_dataset, batch_size=min(config.batch_size, len(validation_dataset)),
    )))
    dependency_inputs, _, _ = dependencies.round_inputs(
        first_batch, round_index=TEN_ROUND_COUNT - 1, device=device,
        source_view_kind=config.source_view_kind,
    )
    dependency = dependencies.measure_dependency(model, dependency_inputs)
    return Pilot_training_result(
        model=model, model_config=model_config, completed_updates=completed_updates,
        round_schedule=tuple(round_schedule), batch_schedule=batch_schedule,
        progress=tuple(progress), evaluation=evaluation, guidance=tuple(guidance),
        dependency=dependency,
    )


__all__ = [
    "Pilot_training_dependencies", "Pilot_initialized_model", "Pilot_training_result",
    "initialize_pilot_model", "train_pilot_core",
]
