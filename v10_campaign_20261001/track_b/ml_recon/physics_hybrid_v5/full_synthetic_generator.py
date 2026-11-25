"""Full-forward synthetic curriculum with generated inverse-teacher labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class FullSyntheticSampleV1:
    counts: np.ndarray
    teacher_action: np.ndarray
    active_mechanism_mask: np.ndarray
    exact_physics_coefficients: np.ndarray
    exact_command_perturbations: np.ndarray
    exact_detector_parameters: np.ndarray
    exact_source_parameters: np.ndarray
    exact_drift_parameters: np.ndarray
    exact_aod_form_ids: np.ndarray
    curriculum_label: str
    source_id: str = "synthetic"


class FullForwardSyntheticGenerator:
    """Compose write/read/Born/source/detector/NB2 calls in one generator."""

    def __init__(
        self,
        *,
        write_forward: Callable[..., Mapping[str, Any]],
        read_forward: Callable[..., Mapping[str, Any]],
        born_forward: Callable[..., Mapping[str, Any]],
        source_forward: Callable[..., Any],
        detector_forward: Callable[..., Any],
        inverse_teacher: Any,
        dispersion: float = 0.02,
    ) -> None:
        self.write_forward = write_forward
        self.read_forward = read_forward
        self.born_forward = born_forward
        self.source_forward = source_forward
        self.detector_forward = detector_forward
        self.inverse_teacher = inverse_teacher
        self.dispersion = float(max(dispersion, 0.0))

    def generate(
        self,
        command: Any,
        latent: Mapping[str, Any],
        *,
        rng: np.random.Generator,
        curriculum_label: str,
    ) -> FullSyntheticSampleV1:
        write = self.write_forward(command, latent)
        read = self.read_forward(command, latent)
        born = self.born_forward(write, read, latent)
        source = self.source_forward(born, latent)
        detector = self.detector_forward(source, latent)
        mean = np.maximum(np.asarray(detector["expected_counts"], dtype=float), 1e-8)
        counts = sample_nb2(mean, self.dispersion, rng)
        teacher = self.inverse_teacher.solve(
            {"counts": counts, **dict(latent)}, command, target_counts=latent.get("target_counts")
        )
        teacher_action = np.asarray(teacher.action_coordinates, dtype=float)
        return FullSyntheticSampleV1(
            counts=counts,
            teacher_action=teacher_action,
            active_mechanism_mask=np.asarray(latent["active_mechanism_mask"], dtype=np.float32),
            exact_physics_coefficients=np.asarray(latent["exact_physics_coefficients"], dtype=np.float32),
            exact_command_perturbations=np.asarray(latent["exact_command_perturbations"], dtype=np.float32),
            exact_detector_parameters=np.asarray(latent["exact_detector_parameters"], dtype=np.float32),
            exact_source_parameters=np.asarray(latent["exact_source_parameters"], dtype=np.float32),
            exact_drift_parameters=np.asarray(latent["exact_drift_parameters"], dtype=np.float32),
            exact_aod_form_ids=np.asarray(latent["exact_aod_form_ids"], dtype=np.int64),
            curriculum_label=str(curriculum_label),
        )


def sample_nb2(mean: Any, dispersion: float, rng: np.random.Generator) -> np.ndarray:
    mu = np.maximum(np.asarray(mean, dtype=float), 0.0)
    alpha = float(max(dispersion, 0.0))
    if alpha <= 1e-12:
        return rng.poisson(mu).astype(np.int64)
    shape = 1.0 / alpha
    rate = rng.gamma(shape=shape, scale=mu * alpha)
    return rng.poisson(rate).astype(np.int64)


def paired_counterfactual(
    generator: FullForwardSyntheticGenerator,
    first_command: Any,
    second_command: Any,
    latent: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    curriculum_label: str,
) -> Dict[str, Any]:
    first = generator.generate(first_command, latent, rng=rng, curriculum_label=curriculum_label)
    second = generator.generate(second_command, latent, rng=rng, curriculum_label=curriculum_label)
    return {
        "first": first,
        "second": second,
        "pair_teacher_delta": second.teacher_action - first.teacher_action,
        "same_latent_error": True,
        "command_shuffle_teacher_correctness_available": True,
    }


__all__ = [
    "FullForwardSyntheticGenerator",
    "FullSyntheticSampleV1",
    "paired_counterfactual",
    "sample_nb2",
]
