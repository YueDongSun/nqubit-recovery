"""Immutable provenance for the target-free PROFILED_CSM estimand."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Mapping

from .contracts import sha256_of
from .control_coordinate_layout import INTERLEAVED_CONTROL_LAYOUT_SHA


PROFILED_CSM_SCHEMA = "V10ProfiledCSMContractV2"
ANALYTIC_PROFILE_SOLVER = "orthogonal_projection_svd"
ANALYTIC_PROFILE_INITIALIZATION = "not_applicable_analytic_projection"
ANALYTIC_PROFILE_PENALTY = "none"


def _require_sha(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")


@dataclass(frozen=True)
class Profiled_csm_contract:
    """All non-target profile semantics, hashed as one immutable contract."""

    nuisance_fields: tuple[str, ...]
    nuisance_domain_bounds: tuple[tuple[float, float], ...]
    penalty_kind: str
    penalty_weight: float
    solver_identity: str
    solver_tolerance: float
    initialization: str
    calibration_source_sha: str
    forward_recipe_sha: str
    coordinate_layout_sha: str = INTERLEAVED_CONTROL_LAYOUT_SHA
    calibration_source_path: str = ""

    def __post_init__(self) -> None:
        nuisance_fields = tuple(self.nuisance_fields)
        domain_bounds = tuple(tuple(bounds) for bounds in self.nuisance_domain_bounds)
        object.__setattr__(self, "nuisance_fields", nuisance_fields)
        object.__setattr__(self, "nuisance_domain_bounds", domain_bounds)
        self.validate()

    def validate(self) -> None:
        if not self.nuisance_fields or len(set(self.nuisance_fields)) != len(self.nuisance_fields):
            raise ValueError("profile contract requires unique nuisance_fields")
        if len(self.nuisance_domain_bounds) != len(self.nuisance_fields):
            raise ValueError("profile contract requires one nuisance domain per field")
        for bounds in self.nuisance_domain_bounds:
            if len(bounds) != 2:
                raise ValueError("profile nuisance domains require lower/upper pairs")
            lower, upper = bounds
            if (
                any(type(value) not in (int, float) for value in (lower, upper))
                or not all(isfinite(value) for value in (lower, upper))
                or lower >= upper
            ):
                raise ValueError("profile nuisance domains must be finite ordered bounds")
        if self.solver_identity != ANALYTIC_PROFILE_SOLVER:
            raise ValueError("PROFILED_CSM only supports the declared analytic projection solver")
        if self.initialization != ANALYTIC_PROFILE_INITIALIZATION:
            raise ValueError("analytic profile projection does not accept an initialization")
        if (
            self.penalty_kind != ANALYTIC_PROFILE_PENALTY
            or type(self.penalty_weight) not in (int, float)
            or self.penalty_weight != 0.0
        ):
            raise ValueError("analytic profile projection has no penalty")
        if (
            type(self.solver_tolerance) not in (int, float)
            or not isfinite(self.solver_tolerance)
            or self.solver_tolerance <= 0.0
        ):
            raise ValueError("profile solver_tolerance must be finite and positive")
        _require_sha(self.calibration_source_sha, "calibration_source_sha")
        if not self.calibration_source_path:
            raise ValueError("profile contract requires a calibration source path")
        if (
            Path(self.calibration_source_path).is_absolute()
            or ".." in Path(self.calibration_source_path).parts
            or ":" in self.calibration_source_path
        ):
            raise ValueError("profile calibration source path must be relative and contained")
        _require_sha(self.forward_recipe_sha, "forward_recipe_sha")
        if self.coordinate_layout_sha != INTERLEAVED_CONTROL_LAYOUT_SHA:
            raise ValueError("profile contract must bind the canonical control layout")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        result["schema"] = PROFILED_CSM_SCHEMA
        return result

    def sha256(self) -> str:
        return sha256_of(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Profiled_csm_contract":
        expected = {item.name for item in fields(cls)} | {"schema"}
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PROFILED_CSM_SCHEMA
        ):
            raise ValueError("profile contract fields do not match the V2 schema")
        payload = {name: value[name] for name in expected - {"schema"}}
        for name in ("nuisance_fields", "nuisance_domain_bounds"):
            if not isinstance(payload[name], (list, tuple)):
                raise ValueError(f"profile contract {name} must be an array")
        payload["nuisance_fields"] = tuple(payload["nuisance_fields"])
        payload["nuisance_domain_bounds"] = tuple(
            tuple(bounds) for bounds in payload["nuisance_domain_bounds"]
        )
        result = cls(**payload)
        result.validate()
        return result

    def verify_calibration_source(self, root: str | Path) -> Path:
        """Rehash the immutable calibration source beneath its admission root."""
        self.validate()
        if not self.calibration_source_path:
            raise ValueError("PROFILED_CSM requires a file-backed calibration source path")
        root_path = Path(root).resolve(strict=True)
        source_path = (root_path / self.calibration_source_path).resolve(strict=True)
        if not source_path.is_file() or not source_path.is_relative_to(root_path):
            raise ValueError("profile calibration source path escapes its admission root")
        import hashlib

        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha != self.calibration_source_sha:
            raise ValueError("profile calibration source bytes changed or differ from admission")
        return source_path


__all__ = [
    "ANALYTIC_PROFILE_INITIALIZATION",
    "ANALYTIC_PROFILE_PENALTY",
    "ANALYTIC_PROFILE_SOLVER",
    "PROFILED_CSM_SCHEMA",
    "Profiled_csm_contract",
]
