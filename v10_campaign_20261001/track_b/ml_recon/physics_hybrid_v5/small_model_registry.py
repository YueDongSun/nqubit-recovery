"""Budget-validated 30k–3M parameter model registry for local screening."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


DEFAULT_SCREEN_CONFIG = Path("ml_recon/configs/v6_small_model_screen.json")


@dataclass(frozen=True)
class SmallModelSpecV1:
    model_id: str
    role: str
    target_parameters: int
    depth: int
    requires_exact_command_fraction: float = 0.0


def load_small_model_config(path: str | Path = DEFAULT_SCREEN_CONFIG) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "v6_small_model_screen" or int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported small-model screen config")
    if payload.get("promotion_ready") is not False or payload.get("research_only") is not True:
        raise ValueError("small-model screen must remain research-only and non-promotable")
    return payload


def model_specs(config: Mapping[str, Any]) -> Dict[str, SmallModelSpecV1]:
    result = {}
    for model_id, values in dict(config["models"]).items():
        result[model_id] = SmallModelSpecV1(
            model_id=model_id,
            role=str(values["role"]),
            target_parameters=int(values["target_parameters"]),
            depth=int(values.get("depth", 2)),
            requires_exact_command_fraction=float(values.get("requires_exact_command_fraction", 0.0)),
        )
    return result


if nn is not None:
    class BudgetedSmallModel(nn.Module):
        """Compact MLP plus an active coefficient bank to hit a declared budget."""

        def __init__(self, spec: SmallModelSpecV1, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.spec = spec
            depth = max(int(spec.depth), 1)
            width = max(8, int(math.sqrt(spec.target_parameters / max(depth, 1))))
            while _mlp_parameter_count(input_dim, output_dim, width, depth) > spec.target_parameters:
                width -= 1
                if width < 8:
                    raise ValueError(f"parameter target is too small for {spec.model_id}")
            layers = [nn.Linear(int(input_dim), width), nn.SiLU()]
            for _ in range(depth - 1):
                layers.extend((nn.Linear(width, width), nn.SiLU()))
            layers.append(nn.Linear(width, int(output_dim)))
            self.network = nn.Sequential(*layers)
            base_count = sum(parameter.numel() for parameter in self.network.parameters())
            bank_count = int(spec.target_parameters) - base_count
            self.coefficient_bank = nn.Parameter(torch.zeros(bank_count))
            self.output_dim = int(output_dim)

        def forward(self, features: Any) -> Any:
            output = self.network(features)
            if self.coefficient_bank.numel() > 0:
                # The bank is an active global residual basis, so every
                # declared parameter participates in optimization.
                bank_chunks = torch.tensor_split(self.coefficient_bank, self.output_dim)
                residual = torch.stack([
                    chunk.mean() if chunk.numel() else output.sum() * 0.0
                    for chunk in bank_chunks
                ])
                output = output + residual
            return output
else:  # pragma: no cover
    class BudgetedSmallModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("small models require torch")


def _mlp_parameter_count(input_dim: int, output_dim: int, width: int, depth: int) -> int:
    return (input_dim + 1) * width + max(depth - 1, 0) * (width + 1) * width + (width + 1) * output_dim


def parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def validate_parameter_budget(
    model: Any,
    spec: SmallModelSpecV1,
    *,
    tolerance: float = 0.10,
) -> int:
    actual = parameter_count(model)
    if actual > int(math.floor(spec.target_parameters * (1.0 + float(tolerance)))):
        raise ValueError(
            f"{spec.model_id} has {actual} parameters, exceeding budget {spec.target_parameters}"
        )
    return actual


def build_small_model(
    model_id: str,
    *,
    config: Mapping[str, Any],
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
) -> Any:
    specs = model_specs(config)
    if model_id not in specs:
        raise ValueError(f"unknown small model: {model_id}")
    spec = specs[model_id]
    input_dim = int(input_dim or config["input_dim"])
    output_dim = int(output_dim or (
        config["noise_output_dim"] if spec.role == "noise" else config["calibrator_output_dim"]
    ))
    model = BudgetedSmallModel(spec, input_dim, output_dim)
    validate_parameter_budget(
        model, spec, tolerance=float(config.get("parameter_budget_tolerance", 0.10))
    )
    return model


__all__ = [
    "BudgetedSmallModel",
    "DEFAULT_SCREEN_CONFIG",
    "SmallModelSpecV1",
    "build_small_model",
    "load_small_model_config",
    "model_specs",
    "parameter_count",
    "validate_parameter_budget",
]
