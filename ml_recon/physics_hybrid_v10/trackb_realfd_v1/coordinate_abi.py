"""Explicit Track B logical/mode/RF coordinate ABI.

The ABI uses grouped relative-log-amplitude then relative-phase coordinates.
Conversions are written in terms of named axes and a registered Helmert basis;
they never infer an order from a reshape, a mapping, or dataclass field order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_git_oid,
    require_sha256,
)


TRACKB_GROUPED_LAYOUT = "TRACKB_GROUPED_LOGAMP_RELATIVE_PHASE_V1"
TRACKB_GAUGE_RULE = "HELMERT_ZERO_SUM_RELATIVE_GAUGE_V1"
TRACKB_PHASE_CONVENTION = "RADIANS_WRAP_MINUS_PI_OPEN_PI_V1"
MATERIALIZER_GROUPED_LAYOUT = "GLOBAL_W_PROJECTIVE_AITCHISON_LOG_PROBABILITY_V1"
LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE = 2.0


def system_mode_count(system_type: str) -> int:
    modes = {"3qubit": 3, "4qubit": 4}
    try:
        return modes[system_type]
    except KeyError as error:
        raise ValueError("Track B supports only 3qubit and 4qubit") from error


def logical_dimension(N: int) -> int:
    if type(N) is not int or N < 2:
        raise ValueError("N must be an integer at least two")
    return 2 * N - 2


def coordinate_layout_payload(N: int) -> dict[str, object]:
    logical_dimension(N)
    return {
        "schema": "TrackBCoordinateLayoutV1",
        "coordinate_layout": TRACKB_GROUPED_LAYOUT,
        "N": N,
        "amplitude_coordinates": [f"relative_log_amplitude_{index}" for index in range(N - 1)],
        "phase_coordinates": [f"relative_phase_rad_{index}" for index in range(N - 1)],
        "grouping": "all_amplitude_then_all_phase",
        "gauge_rule": TRACKB_GAUGE_RULE,
        "phase_convention": TRACKB_PHASE_CONVENTION,
    }


def coordinate_layout_sha(N: int) -> str:
    return canonical_sha256(coordinate_layout_payload(N))


def gauge_rule_sha(N: int) -> str:
    return canonical_sha256({"schema": "TrackBGaugeRuleV1", "N": N, "rule": TRACKB_GAUGE_RULE})


def phase_convention_sha(N: int) -> str:
    return canonical_sha256({
        "schema": "TrackBPhaseConventionV1", "N": N, "convention": TRACKB_PHASE_CONVENTION,
    })


def logical_action_payload_sha(
    system_type: str, amplitude_coordinates: Sequence[object], phase_coordinates: Sequence[object],
) -> str:
    """Canonical identity shared by proposal, RF delta, and AWG materialization."""
    N = system_mode_count(system_type)
    amplitude = _finite_vector(amplitude_coordinates, N - 1, "amplitude_coordinates")
    phase = _finite_vector(phase_coordinates, N - 1, "phase_coordinates")
    return canonical_sha256({
        "schema": "TrackBLogicalActionPayloadV1",
        "system_type": system_type,
        "amplitude_coordinates": list(amplitude),
        "phase_coordinates": list(phase),
        "coordinate_layout_sha": coordinate_layout_sha(N),
    })


def manifold_sha(N: int) -> str:
    return canonical_sha256({
        "schema": "TrackBWLogAmplitudeRelativePhaseManifoldV1",
        "N": N,
        "logical_dimension": logical_dimension(N),
        "gauge_rule_sha": gauge_rule_sha(N),
        "phase_convention_sha": phase_convention_sha(N),
    })


def materializer_coordinate_layout_sha(N: int) -> str:
    logical_dimension(N)
    return canonical_sha256({
        "schema": "TrackBMaterializerCoordinateLayoutV1",
        "system_type": "3qubit" if N == 3 else "4qubit" if N == 4 else "unsupported",
        "coordinate_layout": MATERIALIZER_GROUPED_LAYOUT,
        "amplitude_chart": "HELMERT_AITCHISON_LOG_PROBABILITY",
        "phase_chart": "HELMERT_RELATIVE_PHASE_RADIANS",
        "coordinate_count": N - 1,
    })


def materializer_coordinate_adapter_sha(N: int) -> str:
    logical_dimension(N)
    return canonical_sha256({
        "schema": "TrackBLogAmplitudeToMaterializerLogProbabilityV1",
        "system_type": "3qubit" if N == 3 else "4qubit" if N == 4 else "unsupported",
        "logical_amplitude_chart": "HELMERT_RELATIVE_LOG_AMPLITUDE",
        "materializer_amplitude_chart": "HELMERT_AITCHISON_LOG_PROBABILITY",
        "log_amplitude_to_log_probability_scale": LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE,
        "phase_chart": "HELMERT_RELATIVE_PHASE_RADIANS",
        "coordinate_count": N - 1,
    })


def _finite_vector(values: Sequence[object], expected: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must be an ordered sequence")
    if any(type(value) is bool or not isinstance(value, Real) for value in values):
        raise ValueError(f"{name} must contain numeric values, never booleans or strings")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if len(result) != expected or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain exactly {expected} finite values")
    return result


def grouped_to_interleaved(
    amplitude_coordinates: Sequence[object], phase_coordinates: Sequence[object], N: int,
) -> tuple[float, ...]:
    """Convert named grouped coordinates to an explicit ``a0,p0,a1,p1,...`` sequence."""
    expected = N - 1
    amplitude = _finite_vector(amplitude_coordinates, expected, "amplitude_coordinates")
    phase = _finite_vector(phase_coordinates, expected, "phase_coordinates")
    return tuple(component for pair in zip(amplitude, phase, strict=True) for component in pair)


def interleaved_to_grouped(values: Sequence[object], N: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert an explicitly declared interleaved sequence without shape inference."""
    vector = _finite_vector(values, logical_dimension(N), "interleaved_coordinates")
    return tuple(vector[0::2]), tuple(vector[1::2])


def helmert_basis(N: int) -> np.ndarray:
    """Return the registered robust zero-sum basis, never a fixed mode-zero gauge."""
    logical_dimension(N)
    basis = np.zeros((N, N - 1), dtype=np.float64)
    for column in range(N - 1):
        normalizer = math.sqrt((column + 1) * (column + 2))
        basis[:column + 1, column] = 1.0 / normalizer
        basis[column + 1, column] = -(column + 1) / normalizer
    return basis


def wrap_phase(value: Sequence[object] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("phase values must be finite")
    return np.arctan2(np.sin(array), np.cos(array))


def shortest_phase_difference(left: Sequence[object], right: Sequence[object]) -> tuple[float, ...]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("phase vectors must have matching finite shapes")
    return tuple(float(value) for value in wrap_phase(a - b))


def grouped_to_mode_action(
    amplitude_coordinates: Sequence[object], phase_coordinates: Sequence[object], N: int,
) -> tuple[tuple[float, float], ...]:
    amplitude = np.asarray(_finite_vector(amplitude_coordinates, N - 1, "amplitude_coordinates"))
    phase = np.asarray(_finite_vector(phase_coordinates, N - 1, "phase_coordinates"))
    basis = helmert_basis(N)
    mode_amplitude = amplitude @ basis.T
    mode_phase = phase @ basis.T
    return tuple((float(mode_amplitude[index]), float(mode_phase[index])) for index in range(N))


def mode_to_grouped_action(mode_coordinates: Sequence[Sequence[object]], N: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(mode_coordinates, (tuple, list, np.ndarray)) or len(mode_coordinates) != N:
        raise ValueError("mode action must be an explicitly ordered [N,2] sequence")
    pairs = tuple(tuple(float(value) for value in pair) for pair in mode_coordinates)
    if any(len(pair) != 2 or not all(math.isfinite(value) for value in pair) for pair in pairs):
        raise ValueError("mode action must contain finite [amplitude, phase] pairs")
    array = np.asarray(pairs, dtype=np.float64)
    # A non-zero mean would reintroduce global brightness/phase gauges.  Never
    # silently choose a reference mode, particularly not a near-zero mode.
    if not np.allclose(array.mean(axis=0), 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("mode action violates the registered zero-sum gauge")
    basis = helmert_basis(N)
    return tuple(array[:, 0] @ basis), tuple(array[:, 1] @ basis)


@dataclass(frozen=True)
class IdealLogicalActionV1:
    """Source-bound ideal action in the canonical Track B grouped coordinate ABI."""

    system_type: str
    N: int
    amplitude_coordinates: tuple[float, ...]
    phase_coordinates: tuple[float, ...]
    interleaved_coordinates: tuple[float, ...]
    coordinate_layout: str
    coordinate_layout_sha: str
    manifold_sha: str
    gauge_rule: str
    gauge_rule_sha: str
    phase_convention: str
    phase_convention_sha: str
    provider_type: str
    provider_source_commit: str
    provider_source_tree: str
    provider_checkpoint_sha: str | None
    provider_config_sha: str
    analytic_action_application_count: int = 1
    research_only: bool = True
    schema: str = field(default="IdealLogicalActionV1", init=False)

    def __post_init__(self) -> None:
        if self.N != system_mode_count(self.system_type):
            raise ValueError("system_type and N disagree")
        expected = self.N - 1
        amplitude = _finite_vector(self.amplitude_coordinates, expected, "amplitude_coordinates")
        phase = _finite_vector(self.phase_coordinates, expected, "phase_coordinates")
        interleaved = _finite_vector(self.interleaved_coordinates, logical_dimension(self.N), "interleaved_coordinates")
        if interleaved != grouped_to_interleaved(amplitude, phase, self.N):
            raise ValueError("interleaved coordinates do not exactly match named grouped coordinates")
        if self.coordinate_layout != TRACKB_GROUPED_LAYOUT:
            raise ValueError("Track B action requires its registered grouped coordinate layout")
        if self.coordinate_layout_sha != coordinate_layout_sha(self.N):
            raise ValueError("coordinate layout SHA does not match N")
        if self.manifold_sha != manifold_sha(self.N):
            raise ValueError("manifold SHA does not match N")
        if self.gauge_rule != TRACKB_GAUGE_RULE or self.gauge_rule_sha != gauge_rule_sha(self.N):
            raise ValueError("gauge rule is not the registered robust convention")
        if self.phase_convention != TRACKB_PHASE_CONVENTION or self.phase_convention_sha != phase_convention_sha(self.N):
            raise ValueError("phase convention is not registered")
        if type(self.provider_type) is not str or not self.provider_type:
            raise ValueError("provider_type must be explicit")
        require_git_oid(self.provider_source_commit, "provider_source_commit")
        require_git_oid(self.provider_source_tree, "provider_source_tree")
        if self.provider_checkpoint_sha is not None:
            require_sha256(self.provider_checkpoint_sha, "provider_checkpoint_sha")
        require_sha256(self.provider_config_sha, "provider_config_sha")
        if self.analytic_action_application_count != 1:
            raise ValueError("an analytic or Track A ideal action must be applied exactly once")
        if self.research_only is not True:
            raise ValueError("ideal actions remain research-only")
        object.__setattr__(self, "amplitude_coordinates", amplitude)
        object.__setattr__(self, "phase_coordinates", phase)
        object.__setattr__(self, "interleaved_coordinates", interleaved)

    @classmethod
    def from_grouped(
        cls,
        *,
        system_type: str,
        amplitude_coordinates: Sequence[object],
        phase_coordinates: Sequence[object],
        provider_type: str,
        provider_source_commit: str,
        provider_source_tree: str,
        provider_config_sha: str,
        provider_checkpoint_sha: str | None = None,
    ) -> "IdealLogicalActionV1":
        N = system_mode_count(system_type)
        amplitude = _finite_vector(amplitude_coordinates, N - 1, "amplitude_coordinates")
        phase = _finite_vector(phase_coordinates, N - 1, "phase_coordinates")
        return cls(
            system_type=system_type,
            N=N,
            amplitude_coordinates=amplitude,
            phase_coordinates=phase,
            interleaved_coordinates=grouped_to_interleaved(amplitude, phase, N),
            coordinate_layout=TRACKB_GROUPED_LAYOUT,
            coordinate_layout_sha=coordinate_layout_sha(N),
            manifold_sha=manifold_sha(N),
            gauge_rule=TRACKB_GAUGE_RULE,
            gauge_rule_sha=gauge_rule_sha(N),
            phase_convention=TRACKB_PHASE_CONVENTION,
            phase_convention_sha=phase_convention_sha(N),
            provider_type=provider_type,
            provider_source_commit=provider_source_commit,
            provider_source_tree=provider_source_tree,
            provider_checkpoint_sha=provider_checkpoint_sha,
            provider_config_sha=provider_config_sha,
        )

    @property
    def grouped_coordinates(self) -> tuple[float, ...]:
        return self.amplitude_coordinates + self.phase_coordinates

    @property
    def logical_dimension(self) -> int:
        return logical_dimension(self.N)

    @property
    def track_a_interleaved_coordinates(self) -> tuple[float, ...]:
        """Explicit Track-A adapter spelling; ordering is declared, never guessed."""
        return self.interleaved_coordinates

    @classmethod
    def from_track_a_interleaved(
        cls,
        *,
        system_type: str,
        interleaved_coordinates: Sequence[object],
        provider_type: str,
        provider_source_commit: str,
        provider_source_tree: str,
        provider_config_sha: str,
        provider_checkpoint_sha: str | None = None,
    ) -> "IdealLogicalActionV1":
        N = system_mode_count(system_type)
        amplitude, phase = interleaved_to_grouped(interleaved_coordinates, N)
        return cls.from_grouped(
            system_type=system_type,
            amplitude_coordinates=amplitude,
            phase_coordinates=phase,
            provider_type=provider_type,
            provider_source_commit=provider_source_commit,
            provider_source_tree=provider_source_tree,
            provider_config_sha=provider_config_sha,
            provider_checkpoint_sha=provider_checkpoint_sha,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "N": self.N,
            "logical_dimension": self.logical_dimension,
            "amplitude_coordinates": list(self.amplitude_coordinates),
            "phase_coordinates": list(self.phase_coordinates),
            "interleaved_coordinates": list(self.interleaved_coordinates),
            "coordinate_layout": self.coordinate_layout,
            "coordinate_layout_sha": self.coordinate_layout_sha,
            "manifold_sha": self.manifold_sha,
            "gauge_rule": self.gauge_rule,
            "gauge_rule_sha": self.gauge_rule_sha,
            "phase_convention": self.phase_convention,
            "phase_convention_sha": self.phase_convention_sha,
            "provider_type": self.provider_type,
            "provider_source_commit": self.provider_source_commit,
            "provider_source_tree": self.provider_source_tree,
            "provider_checkpoint_sha": self.provider_checkpoint_sha,
            "provider_config_sha": self.provider_config_sha,
            "analytic_action_application_count": self.analytic_action_application_count,
            "research_only": True,
        }

    def mode_action(self) -> "ModeActionV1":
        return ModeActionV1.from_logical_action(self)

    def to_global_w_projective_action(self) -> dict[str, object]:
        """Adapt explicitly to the existing materializer JSON ABI; no import is needed.

        The materializer chart is Helmert ``log(a**2)``, whereas Track B owns
        Helmert ``log(a)``.  The factor is serialized with a versioned adapter
        identity so a same-shaped legacy action cannot be silently reinterpreted.
        """
        return {
            "schema": "GlobalWProjectiveActionV1",
            "system_type": self.system_type,
            "amplitude_coordinates": [
                LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE * value
                for value in self.amplitude_coordinates
            ],
            "phase_coordinates": list(self.phase_coordinates),
            "brightness_delta": 0.0,
            "brightness_enabled": False,
            # Match the established materializer representation of disabled
            # optional action channels.  ``0.0`` is an explicit no-op, not a
            # missing or inferred command.
            "read_action": 0.0,
            "local_residual": 0.0,
            "cycle_residual": 0.0,
            "research_only": True,
            "coordinate_layout_sha": materializer_coordinate_layout_sha(self.N),
            "coordinate_adapter_sha": materializer_coordinate_adapter_sha(self.N),
        }

    @classmethod
    def from_global_w_projective_action(
        cls,
        action: Mapping[str, object],
        *,
        provider_type: str,
        provider_source_commit: str,
        provider_source_tree: str,
        provider_config_sha: str,
        provider_checkpoint_sha: str | None = None,
    ) -> "IdealLogicalActionV1":
        if not isinstance(action, Mapping) or action.get("schema") != "GlobalWProjectiveActionV1":
            raise ValueError("expected GlobalWProjectiveActionV1 mapping")
        N = system_mode_count(action.get("system_type"))
        if (
            action.get("coordinate_layout_sha") != materializer_coordinate_layout_sha(N)
            or action.get("coordinate_adapter_sha") != materializer_coordinate_adapter_sha(N)
        ):
            raise ValueError(
                "GlobalWProjectiveActionV1 requires the registered "
                "materializer coordinate layout and Track B adapter SHA"
            )
        if action.get("brightness_enabled", False) is not False or float(action.get("brightness_delta", 0.0)) != 0.0:
            raise ValueError("Track B ABI does not admit brightness actions")
        for key in ("read_action", "local_residual", "cycle_residual"):
            value = action.get(key, 0.0)
            if value is not None and float(value) != 0.0:
                raise ValueError(f"Track B ABI does not admit {key}")
        return cls.from_grouped(
            system_type=action.get("system_type"),
            amplitude_coordinates=tuple(
                float(value) / LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE
                for value in action.get("amplitude_coordinates", ())
            ),
            phase_coordinates=action.get("phase_coordinates", ()),
            provider_type=provider_type,
            provider_source_commit=provider_source_commit,
            provider_source_tree=provider_source_tree,
            provider_config_sha=provider_config_sha,
            provider_checkpoint_sha=provider_checkpoint_sha,
        )

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "system_type": self.system_type,
            "amplitude_coordinates": list(self.amplitude_coordinates),
            "phase_coordinates": list(self.phase_coordinates),
            "coordinate_layout_sha": self.coordinate_layout_sha,
            "manifold_sha": self.manifold_sha,
            "gauge_rule_sha": self.gauge_rule_sha,
            "phase_convention_sha": self.phase_convention_sha,
            "provider_type": self.provider_type,
            "provider_source_commit": self.provider_source_commit,
            "provider_source_tree": self.provider_source_tree,
            "provider_checkpoint_sha": self.provider_checkpoint_sha,
            "provider_config_sha": self.provider_config_sha,
            "analytic_action_application_count": self.analytic_action_application_count,
        })


@dataclass(frozen=True)
class RecoverableLogicalActionV1:
    """A target-specific logical action constrained to registered CSM support.

    Direct construction is retained only for parsing older contract fixtures.
    A proposal-eligible action must be created with ``from_projection`` so its
    coordinates and basis identity prove the actual projection took place.
    """

    ideal_action: IdealLogicalActionV1
    amplitude_coordinates: tuple[float, ...]
    phase_coordinates: tuple[float, ...]
    support_sha: str
    damping: float
    source_action_sha: str
    reason_code: str
    projection_basis_sha: str | None = None
    projection_residual_norm: float | None = None
    projection_verified: bool = False
    schema: str = field(default="RecoverableLogicalActionV1", init=False)

    def __post_init__(self) -> None:
        N = self.ideal_action.N
        amplitude = _finite_vector(self.amplitude_coordinates, N - 1, "amplitude_coordinates")
        phase = _finite_vector(self.phase_coordinates, N - 1, "phase_coordinates")
        require_sha256(self.support_sha, "support_sha")
        require_sha256(self.source_action_sha, "source_action_sha")
        if self.source_action_sha != self.ideal_action.sha256():
            raise ValueError("recoverable action must bind its exact ideal action")
        if not math.isfinite(float(self.damping)) or not 0.0 <= float(self.damping) <= 1.0:
            raise ValueError("damping must be finite in [0,1]")
        if type(self.reason_code) is not str or not self.reason_code:
            raise ValueError("recoverable action requires a reason code")
        if type(self.projection_verified) is not bool:
            raise TypeError("projection_verified must be an exact boolean")
        if self.projection_verified:
            if self.projection_basis_sha is None or self.projection_residual_norm is None:
                raise ValueError("verified recoverable action requires a basis identity and residual")
            require_sha256(self.projection_basis_sha, "projection_basis_sha")
            if (
                not math.isfinite(float(self.projection_residual_norm))
                or float(self.projection_residual_norm) < 0.0
            ):
                raise ValueError("projection residual norm must be finite and nonnegative")
        elif self.projection_basis_sha is not None or self.projection_residual_norm is not None:
            raise ValueError("unverified recoverable action cannot carry partial projection metadata")
        object.__setattr__(self, "amplitude_coordinates", amplitude)
        object.__setattr__(self, "phase_coordinates", phase)

    @classmethod
    def from_ideal(
        cls, ideal_action: IdealLogicalActionV1, *, support_sha: str, damping: float = 1.0,
        reason_code: str = "RECOVERABLE_SUPPORT_PRESENT",
    ) -> "RecoverableLogicalActionV1":
        factor = float(damping)
        return cls(
            ideal_action=ideal_action,
            amplitude_coordinates=tuple(factor * value for value in ideal_action.amplitude_coordinates),
            phase_coordinates=tuple(factor * value for value in ideal_action.phase_coordinates),
            support_sha=support_sha,
            damping=factor,
            source_action_sha=ideal_action.sha256(),
            reason_code=reason_code,
        )

    @classmethod
    def from_projection(
        cls,
        ideal_action: IdealLogicalActionV1,
        *,
        rsm: Any,
        registered_basis: Any,
        damping: float = 1.0,
    ) -> "RecoverableLogicalActionV1":
        """Project ``u_ideal`` through the registered CSM basis before damping.

        ``registered_basis`` is intentionally an object rather than a raw
        array: it must provide both ``support_basis`` and ``sha256()``.  The
        target RSM is then checked to bind that exact CSM identity.  This keeps
        support geometry target-independent while preventing an arbitrary
        caller-supplied basis from being labelled recoverable.
        """
        try:
            rsm_sha = rsm.sha256()
            csm_sha = registered_basis.sha256()
            rsm_csm_sha = rsm.csm_sha
            basis = np.asarray(registered_basis.support_basis, dtype=np.float64)
            reason_code = rsm.reason_code
        except AttributeError as error:
            raise TypeError("from_projection requires registered RSM and CSM support objects") from error
        if rsm_csm_sha != csm_sha:
            raise ValueError("RSM does not bind the supplied registered CSM support")
        dimension = ideal_action.logical_dimension
        if (
            basis.ndim != 2 or basis.shape[0] != dimension
            or not np.all(np.isfinite(basis))
            or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1.0e-8, rtol=0.0)
        ):
            raise ValueError("registered CSM basis is not a finite orthonormal logical basis")
        factor = float(damping)
        if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError("projection damping must lie in [0,1]")
        ideal = np.asarray(ideal_action.grouped_coordinates, dtype=np.float64)
        projected = basis @ (basis.T @ ideal) if basis.shape[1] else np.zeros_like(ideal)
        residual = float(np.linalg.norm(ideal - projected))
        damped = factor * projected
        split = ideal_action.N - 1
        return cls(
            ideal_action=ideal_action,
            amplitude_coordinates=tuple(float(value) for value in damped[:split]),
            phase_coordinates=tuple(float(value) for value in damped[split:]),
            support_sha=rsm_sha,
            damping=factor,
            source_action_sha=ideal_action.sha256(),
            reason_code=reason_code,
            projection_basis_sha=csm_sha,
            projection_residual_norm=residual,
            projection_verified=True,
        )

    def verify_projection(self, *, rsm: Any, csm: Any, tolerance: float = 1.0e-8) -> None:
        """Recompute the projection proof against the live registered objects.

        ``projection_verified`` is a diagnostic flag, not an authority bit.  A
        candidate is eligible only when the stored coordinates and residual
        reproduce ``V V^T u_ideal`` for the exact CSM/RSM identities supplied at
        the gate boundary.
        """
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ValueError("projection verification tolerance must be finite and nonnegative")
        if not self.projection_verified:
            raise ValueError("recoverable action projection is not verified")
        try:
            csm_sha = csm.sha256()
            rsm_sha = rsm.sha256()
            rsm_csm_sha = rsm.csm_sha
            basis = np.asarray(csm.support_basis, dtype=np.float64)
        except AttributeError as error:
            raise TypeError("projection verification requires registered CSM/RSM objects") from error
        if self.projection_basis_sha != csm_sha or self.support_sha != rsm_sha:
            raise ValueError("recoverable action projection identities do not match CSM/RSM")
        if rsm_csm_sha != csm_sha:
            raise ValueError("RSM and CSM projection identities disagree")
        ideal = np.asarray(self.ideal_action.grouped_coordinates, dtype=np.float64)
        stored = np.asarray(self.grouped_coordinates, dtype=np.float64)
        if (
            basis.ndim != 2 or basis.shape[0] != ideal.size
            or not np.all(np.isfinite(basis))
            or basis.shape[1] == 0
            or not np.allclose(
                basis.T @ basis, np.eye(basis.shape[1]), atol=1.0e-8, rtol=0.0,
            )
        ):
            raise ValueError("registered CSM projection basis is invalid")
        projected = basis @ (basis.T @ ideal)
        expected = float(self.damping) * projected
        scale = max(float(np.linalg.norm(ideal)), 1.0)
        if not np.allclose(stored, expected, atol=float(tolerance) * scale, rtol=0.0):
            raise ValueError("recoverable action coordinates do not match the registered projection")
        residual = float(np.linalg.norm(ideal - projected))
        if self.projection_residual_norm is None or not math.isclose(
            float(self.projection_residual_norm), residual,
            rel_tol=0.0, abs_tol=float(tolerance) * scale,
        ):
            raise ValueError("recoverable action projection residual does not match the registered basis")
        if self.reason_code != rsm.reason_code:
            raise ValueError("recoverable action projection reason does not match RSM")

    def scaled(self, factor: float) -> "RecoverableLogicalActionV1":
        """Return a collinear action while retaining its registered projection proof."""
        value = float(factor)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("recoverable action scale must lie in [0,1]")
        return RecoverableLogicalActionV1(
            ideal_action=self.ideal_action,
            amplitude_coordinates=tuple(value * item for item in self.amplitude_coordinates),
            phase_coordinates=tuple(value * item for item in self.phase_coordinates),
            support_sha=self.support_sha,
            damping=value * self.damping,
            source_action_sha=self.source_action_sha,
            reason_code=self.reason_code,
            projection_basis_sha=self.projection_basis_sha,
            projection_residual_norm=self.projection_residual_norm,
            projection_verified=self.projection_verified,
        )

    @property
    def grouped_coordinates(self) -> tuple[float, ...]:
        return self.amplitude_coordinates + self.phase_coordinates

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "source_action_sha": self.source_action_sha,
            "support_sha": self.support_sha,
            "damping": self.damping,
            "amplitude_coordinates": list(self.amplitude_coordinates),
            "phase_coordinates": list(self.phase_coordinates),
            "reason_code": self.reason_code,
            "projection_basis_sha": self.projection_basis_sha,
            "projection_residual_norm": self.projection_residual_norm,
            "projection_verified": self.projection_verified,
        })


@dataclass(frozen=True)
class ModeActionV1:
    """Gauge-fixed ``[N,2]`` mode coordinates with an explicit source action binding."""

    system_type: str
    N: int
    mode_coordinates: tuple[tuple[float, float], ...]
    coordinate_layout_sha: str
    source_logical_action_sha: str
    schema: str = field(default="ModeActionV1", init=False)

    def __post_init__(self) -> None:
        if self.N != system_mode_count(self.system_type):
            raise ValueError("mode action system_type and N disagree")
        if self.coordinate_layout_sha != coordinate_layout_sha(self.N):
            raise ValueError("mode action coordinate layout SHA is not registered")
        require_sha256(self.source_logical_action_sha, "source_logical_action_sha")
        mode_to_grouped_action(self.mode_coordinates, self.N)

    @classmethod
    def from_logical_action(cls, action: IdealLogicalActionV1) -> "ModeActionV1":
        return cls(
            system_type=action.system_type,
            N=action.N,
            mode_coordinates=grouped_to_mode_action(
                action.amplitude_coordinates, action.phase_coordinates, action.N),
            coordinate_layout_sha=action.coordinate_layout_sha,
            source_logical_action_sha=action.sha256(),
        )

    def to_grouped(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return mode_to_grouped_action(self.mode_coordinates, self.N)

    @property
    def logical_dimension(self) -> int:
        return logical_dimension(self.N)


@dataclass(frozen=True)
class RFCommandDeltaV1:
    """Research RF delta descriptor.  It is a proposal, not an uploaded command.

    Each position belongs to the registered RF-coordinate ordering.  Amplitude,
    phase, and ``other`` components are kept separate so an adapter cannot
    accidentally reinterpret a phase/frequency/offset coordinate as amplitude.
    ``other_delta`` is optional only for source compatibility with already
    serialized amplitude/phase-only proposals; omitted values normalize to an
    explicit all-zero vector and never stand in for an unknown command.
    """

    system_type: str
    ordered_tone_ids: tuple[str, ...]
    amplitude_delta: tuple[float, ...]
    phase_delta_rad: tuple[float, ...]
    rf_coordinate_layout_sha: str
    source_logical_action_sha: str
    other_delta: tuple[float, ...] | None = None
    logical_action_payload_sha: str | None = None
    hardware_action_authorized: bool = False
    schema: str = field(default="RFCommandDeltaV1", init=False)

    def __post_init__(self) -> None:
        system_mode_count(self.system_type)
        if not self.ordered_tone_ids or len(set(self.ordered_tone_ids)) != len(self.ordered_tone_ids):
            raise ValueError("RF delta requires a unique ordered tone list")
        if any(type(value) is not str or not value for value in self.ordered_tone_ids):
            raise ValueError("tone identifiers must be nonempty strings")
        amplitude = _finite_vector(
            self.amplitude_delta, len(self.ordered_tone_ids), "amplitude_delta",
        )
        phase = _finite_vector(
            self.phase_delta_rad, len(self.ordered_tone_ids), "phase_delta_rad",
        )
        other = (
            tuple(0.0 for _ in self.ordered_tone_ids)
            if self.other_delta is None
            else _finite_vector(
                self.other_delta, len(self.ordered_tone_ids), "other_delta",
            )
        )
        require_sha256(self.rf_coordinate_layout_sha, "rf_coordinate_layout_sha")
        require_sha256(self.source_logical_action_sha, "source_logical_action_sha")
        if self.logical_action_payload_sha is not None:
            require_sha256(self.logical_action_payload_sha, "logical_action_payload_sha")
        if self.hardware_action_authorized is not False:
            raise ValueError("a Track B RF proposal cannot self-authorize hardware action")
        object.__setattr__(self, "amplitude_delta", amplitude)
        object.__setattr__(self, "phase_delta_rad", phase)
        object.__setattr__(self, "other_delta", other)

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "ordered_tone_ids": list(self.ordered_tone_ids),
            "amplitude_delta": list(self.amplitude_delta),
            "phase_delta_rad": list(self.phase_delta_rad),
            "other_delta": list(self.other_delta or ()),
            "rf_coordinate_layout_sha": self.rf_coordinate_layout_sha,
            "source_logical_action_sha": self.source_logical_action_sha,
            "logical_action_payload_sha": self.logical_action_payload_sha,
            "hardware_action_authorized": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RFCommandDeltaV1":
        expected = {
            "schema", "system_type", "ordered_tone_ids", "amplitude_delta",
            "phase_delta_rad", "other_delta", "rf_coordinate_layout_sha",
            "source_logical_action_sha", "logical_action_payload_sha",
            "hardware_action_authorized",
        }
        if set(payload) != expected or payload.get("schema") != "RFCommandDeltaV1":
            raise ValueError("RF command delta schema is invalid")
        return cls(
            system_type=payload["system_type"],
            ordered_tone_ids=tuple(payload["ordered_tone_ids"]),
            amplitude_delta=tuple(payload["amplitude_delta"]),
            phase_delta_rad=tuple(payload["phase_delta_rad"]),
            rf_coordinate_layout_sha=payload["rf_coordinate_layout_sha"],
            source_logical_action_sha=payload["source_logical_action_sha"],
            other_delta=tuple(payload.get("other_delta", ())),
            logical_action_payload_sha=(
                None if payload.get("logical_action_payload_sha") is None
                else str(payload["logical_action_payload_sha"])
            ),
            hardware_action_authorized=False,
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "RFCommandDeltaV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="RFCommandDeltaV1",
            exact_fields=frozenset({
                "schema", "system_type", "ordered_tone_ids", "amplitude_delta",
                "phase_delta_rad", "other_delta", "rf_coordinate_layout_sha",
                "source_logical_action_sha", "logical_action_payload_sha",
                "hardware_action_authorized",
            }),
        )
        command = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, command)
        return command


@dataclass(frozen=True)
class RFMaterializationPreviewV1:
    """Pure diagnostic preview returned before an external materializer is invoked."""

    requested_rf_delta: RFCommandDeltaV1
    support_leakage_norm: float
    requested_to_materialized_logical_error: float
    direction_cosine: float
    bounds_valid: bool
    phase_wrap_valid: bool
    actual_loaded_discrepancy: float | None
    accepted_for_external_review: bool
    preview_only: bool = True
    hardware_action_authorized: bool = False
    mapping_core_sha: str | None = None
    schema: str = field(default="RFMaterializationPreviewV1", init=False)

    def __post_init__(self) -> None:
        numeric = (
            self.support_leakage_norm, self.requested_to_materialized_logical_error,
            self.direction_cosine,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("RF preview metrics must be finite")
        if any(float(value) < 0.0 for value in numeric[:2]):
            raise ValueError("RF preview leakage/error metrics must be nonnegative")
        if not -1.0 <= float(self.direction_cosine) <= 1.0:
            raise ValueError("direction cosine must lie in [-1,1]")
        if self.actual_loaded_discrepancy is not None and (
            not math.isfinite(float(self.actual_loaded_discrepancy))
            or float(self.actual_loaded_discrepancy) < 0.0
        ):
            raise ValueError("actual-loaded discrepancy must be finite and nonnegative when available")
        for name in ("bounds_valid", "phase_wrap_valid", "accepted_for_external_review"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"RF preview {name} must be an exact boolean")
        if self.preview_only is not True or self.hardware_action_authorized is not False:
            raise ValueError("Track B previews are diagnostic and never hardware-authorized")
        if self.mapping_core_sha is not None:
            require_sha256(self.mapping_core_sha, "mapping_core_sha")

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "requested_rf_delta_sha": self.requested_rf_delta.sha256(),
            "support_leakage_norm": self.support_leakage_norm,
            "requested_to_materialized_logical_error": self.requested_to_materialized_logical_error,
            "direction_cosine": self.direction_cosine,
            "bounds_valid": self.bounds_valid,
            "phase_wrap_valid": self.phase_wrap_valid,
            "actual_loaded_discrepancy": self.actual_loaded_discrepancy,
            "accepted_for_external_review": self.accepted_for_external_review,
            "preview_only": True,
            "hardware_action_authorized": False,
            "mapping_core_sha": self.mapping_core_sha,
        })


__all__ = [
    "IdealLogicalActionV1", "ModeActionV1", "RFCommandDeltaV1", "RFMaterializationPreviewV1",
    "RecoverableLogicalActionV1", "TRACKB_GAUGE_RULE", "TRACKB_GROUPED_LAYOUT",
    "TRACKB_PHASE_CONVENTION", "coordinate_layout_payload", "coordinate_layout_sha",
    "gauge_rule_sha", "grouped_to_interleaved", "grouped_to_mode_action", "helmert_basis",
    "interleaved_to_grouped", "logical_dimension", "manifold_sha", "mode_to_grouped_action",
    "phase_convention_sha", "shortest_phase_difference", "system_mode_count", "wrap_phase",
    "LOG_AMPLITUDE_TO_LOG_PROBABILITY_SCALE", "MATERIALIZER_GROUPED_LAYOUT",
    "materializer_coordinate_adapter_sha", "materializer_coordinate_layout_sha",
]
