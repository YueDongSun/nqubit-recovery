"""Typed contracts for the DLCZ event-level Monte Carlo diagnostic.

The event simulator is deliberately separate from the formal Track A
operation-Poisson forward.  These contracts make the source mark, occupancy,
observation, provenance and budget boundaries explicit.  They do not grant
calibration, Gate, training or hardware authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from ml_recon.physics_hybrid_v10.evidence_snapshot import load_json_snapshot


DLCZ_EVENT_SCHEMA = "V10DlczEventV1"
DLCZ_OPERATION_SCHEMA = "V10DlczOperationParametersV1"
DLCZ_ALLOCATION_SCHEMA = "V10DeterministicAllocationReceiptV1"
DLCZ_DOMAIN_SCHEMA = "V10DlczDomainContractV1"
DLCZ_CALIBRATION_SCHEMA = "V10DlczCalibrationDrawV1"
G2_MIN = 1.0
G2_MAX = 10.0

EVENT_MARKS = (
    "NO_SOURCE_EVENT",
    "SINGLE_CELL_SINGLE_EXCITATION",
    "LOCAL_DOUBLE",
    "CROSS_DOUBLE",
    "MULTIPAIR_K_GE_3",
    "CLASSICAL_SHARED_SHOCK",
    "UNCORRELATED_SIGNAL_BACKGROUND",
    "UNCORRELATED_IDLER_BACKGROUND",
    "SIGNAL_DARK",
    "IDLER_DARK",
)
EVENT_CHANNELS = ("source", "signal", "idler", "background", "dark")
DEAD_TIME_MODES = ("none", "paralyzable", "non_paralyzable")
SOURCE_LAWS = ("DLCZ_GEOMETRIC_PER_SHOT_V8",)
EXPOSURE_COORDINATES = ("NORMALIZED_GATE_COORDINATE", "SI_SECONDS")
DOMAIN_STATUSES = (
    "REGISTERED",
    "HYPOTHESIS_UNBOUND",
    "OUT_OF_SCOPE",
    "NOT_AVAILABLE",
)
_REGISTERED_TOKEN = object()


def _finite(
    value: object, name: str, *, lower: float | None = None, upper: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return result


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _require_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    field_name: str,
    expected: Mapping[str, Any],
) -> None:
    """Require a receipt to bind exactly the values it authorizes.

    A status string alone must never upgrade caller-supplied hypothesis values
    to ``REGISTERED``.  Comparing canonical payload hashes keeps tuple/list
    representation differences out of the authorization decision while still
    rejecting omitted, extra, or altered values.
    """

    payload = receipt.get(field_name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"verified receipt must contain {field_name}")
    if set(payload) != set(expected):
        raise ValueError(f"verified receipt {field_name} fields do not match")
    if _hash_payload(dict(payload)) != _hash_payload(dict(expected)):
        raise ValueError(f"verified receipt {field_name} values do not match")


@dataclass(frozen=True)
class DlczEvent:
    """One source, optical, dark or detected event.

    ``occupancy`` is the latent cell occupancy vector.  It is retained in the
    truth ledger and never copied into the observation plane.  A background or
    dark event has a zero occupancy vector and can coexist with a source event
    in the same opportunity.
    """

    event_id: str
    opportunity_id: str
    operation_id: str
    mark: str
    channel: str
    cell_indices: tuple[int, ...] = ()
    occupancy: tuple[int, ...] = ()
    time_s: float = 0.0
    phase_rad: float = 0.0
    wavevector: tuple[float, ...] = ()
    detected: bool = False
    detector_id: str | None = None

    def validate(self, *, n_mode: int | None = None) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.event_id, self.opportunity_id, self.operation_id)
        ):
            raise ValueError("event identities must be non-empty strings")
        if self.mark not in EVENT_MARKS:
            raise ValueError(f"unregistered event mark: {self.mark}")
        if self.channel not in EVENT_CHANNELS:
            raise ValueError(f"unregistered event channel: {self.channel}")
        if type(self.detected) is not bool:
            raise ValueError("detected must be a strict Boolean")
        if self.detector_id is not None and (
            not isinstance(self.detector_id, str) or not self.detector_id
        ):
            raise ValueError("detector_id must be a non-empty string when present")
        if not isinstance(self.cell_indices, tuple) or any(
            type(index) is not int or index < 0 for index in self.cell_indices
        ):
            raise ValueError("cell_indices must be a tuple of non-negative integers")
        if tuple(sorted(set(self.cell_indices))) != self.cell_indices:
            raise ValueError("cell_indices must be sorted and unique")
        if not isinstance(self.occupancy, tuple) or any(
            type(value) is not int or value < 0 for value in self.occupancy
        ):
            raise ValueError("occupancy must be a tuple of non-negative integers")
        if n_mode is not None and len(self.occupancy) != n_mode:
            raise ValueError("occupancy length must match n_mode")
        if self.occupancy and any(
            index >= len(self.occupancy) for index in self.cell_indices
        ):
            raise ValueError("cell_indices exceed occupancy dimension")
        if not math.isfinite(float(self.time_s)) or float(self.time_s) < 0.0:
            raise ValueError("event time must be finite and non-negative")
        if not math.isfinite(float(self.phase_rad)):
            raise ValueError("event phase must be finite")
        if any(not math.isfinite(float(value)) for value in self.wavevector):
            raise ValueError("wavevector must contain finite values")
        if self.mark == "SIGNAL_DARK" and self.channel not in {"signal", "dark"}:
            raise ValueError("signal dark marks must stay on the signal channel")
        if self.mark == "IDLER_DARK" and self.channel not in {"idler", "dark"}:
            raise ValueError("idler dark marks must stay on the idler channel")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return _hash_payload(self.to_dict())


@dataclass(frozen=True)
class DlczOperationParameters:
    """Operation-point-bound rates and detector parameters.

    ``exposure_s`` is retained for API compatibility, but its unit is defined by
    ``exposure_coordinate``.  In the default synthetic diagnostic it is a
    normalized gate coordinate, not elapsed SI seconds.  Every ``*_rate`` is
    per that same coordinate, so its product with ``exposure_s`` is a mean
    background/dark count per write shot.  A real-rate claim must instead use
    ``SI_SECONDS`` and a verified apparatus receipt.

    ``write_chi`` is the V8 geometric non-vacuum activation probability per
    shot.  ``lambda_pair`` is a legacy field name for the derived non-vacuum
    activation hazard per exposure coordinate, ``-log(1-write_chi)/exposure``;
    it is neither the exact-one-pair probability nor the mean excitation.
    ``repeats`` is the fixed number of opportunities used by the four-cell
    contract.  The default status is explicitly hypothesis/unbound because no
    apparatus receipt is present in this source.
    """

    operation_id: str
    exposure_s: float
    repeats: int
    lambda_pair: float
    write_chi: float
    background_signal_rate: float
    background_idler_rate: float
    eta_signal: float
    eta_retrieval: float
    eta_idler: float
    w_overlap: float
    dark_signal_rate: float = 0.0
    dark_idler_rate: float = 0.0
    local_double_fraction: float = 0.0
    cross_double_fraction: float = 0.0
    classical_shared_rate: float = 0.0
    storage_wait_s: float = 0.0
    storage_lifetime_s: float = 1.0
    eta_optical_path: float = 1.0
    spinwave_coherence_visibility: float = 1.0
    source_law: str = "DLCZ_GEOMETRIC_PER_SHOT_V8"
    exposure_coordinate: str = "NORMALIZED_GATE_COORDINATE"
    parameter_status: str = "HYPOTHESIS_UNBOUND"
    parameter_receipt_sha: str = ""
    _registration_token: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_verified_receipt(
        cls,
        receipt_path: str,
        expected_sha: str,
        **kwargs: Any,
    ) -> "DlczOperationParameters":
        snapshot, receipt = load_json_snapshot(
            receipt_path,
            "DLCZ operation calibration receipt",
            expected_sha=expected_sha,
            maximum_bytes=4 * 1024 * 1024,
        )
        if (
            receipt.get("status") != "REGISTERED"
            or receipt.get("receipt_kind") != "APPARATUS_CALIBRATION"
        ):
            raise ValueError(
                "REGISTERED operation parameters require a verified apparatus receipt"
            )
        forbidden = {
            "parameter_status",
            "parameter_receipt_sha",
            "_registration_token",
        }
        if forbidden & set(kwargs):
            raise ValueError("receipt constructor owns parameter registration fields")
        unregistered = cls(**kwargs)
        unregistered.validate()
        expected_payload = {
            "operation_id": unregistered.operation_id,
            "exposure_s": unregistered.exposure_s,
            "repeats": unregistered.repeats,
            "lambda_pair": unregistered.lambda_pair,
            "write_chi": unregistered.write_chi,
            "background_signal_rate": unregistered.background_signal_rate,
            "background_idler_rate": unregistered.background_idler_rate,
            "eta_signal": unregistered.eta_signal,
            "eta_retrieval": unregistered.eta_retrieval,
            "eta_idler": unregistered.eta_idler,
            "w_overlap": unregistered.w_overlap,
            "dark_signal_rate": unregistered.dark_signal_rate,
            "dark_idler_rate": unregistered.dark_idler_rate,
            "local_double_fraction": unregistered.local_double_fraction,
            "cross_double_fraction": unregistered.cross_double_fraction,
            "classical_shared_rate": unregistered.classical_shared_rate,
            "storage_wait_s": unregistered.storage_wait_s,
            "storage_lifetime_s": unregistered.storage_lifetime_s,
            "eta_optical_path": unregistered.eta_optical_path,
            "spinwave_coherence_visibility": unregistered.spinwave_coherence_visibility,
            "source_law": unregistered.source_law,
            "exposure_coordinate": unregistered.exposure_coordinate,
        }
        _require_receipt_payload(
            receipt,
            field_name="operation_parameters",
            expected=expected_payload,
        )
        registered = cls(
            **kwargs,
            parameter_status="REGISTERED",
            parameter_receipt_sha=snapshot.sha256,
            _registration_token=_REGISTERED_TOKEN,
        )
        registered.validate()
        return registered

    def validate(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id is required")
        if type(self.repeats) is not int or self.repeats < 0:
            raise ValueError("repeats must be a non-negative integer")
        _finite(self.exposure_s, "exposure_s", lower=0.0)
        for name in (
            "lambda_pair",
            "background_signal_rate",
            "background_idler_rate",
            "dark_signal_rate",
            "dark_idler_rate",
            "classical_shared_rate",
        ):
            _finite(getattr(self, name), name, lower=0.0)
        _finite(self.write_chi, "write_chi", lower=0.0, upper=0.25)
        for name in ("eta_signal", "eta_retrieval", "eta_idler", "w_overlap"):
            _finite(getattr(self, name), name, lower=0.0, upper=1.0)
        for name in ("local_double_fraction", "cross_double_fraction"):
            _finite(getattr(self, name), name, lower=0.0, upper=1.0)
        if self.local_double_fraction + self.cross_double_fraction > 1.0 + 1e-12:
            raise ValueError("local and cross double fractions cannot exceed one")
        _finite(self.storage_wait_s, "storage_wait_s", lower=0.0)
        _finite(self.storage_lifetime_s, "storage_lifetime_s", lower=0.0)
        _finite(self.eta_optical_path, "eta_optical_path", lower=0.0, upper=1.0)
        _finite(
            self.spinwave_coherence_visibility,
            "spinwave_coherence_visibility",
            lower=0.0,
            upper=1.0,
        )
        if self.source_law not in SOURCE_LAWS:
            raise ValueError("unregistered DLCZ source law")
        if self.exposure_coordinate not in EXPOSURE_COORDINATES:
            raise ValueError("unregistered exposure coordinate")
        if self.source_law == "DLCZ_GEOMETRIC_PER_SHOT_V8":
            if self.exposure_s == 0.0:
                if self.write_chi != 0.0 or self.lambda_pair != 0.0:
                    raise ValueError(
                        "zero exposure requires zero geometric write activation"
                    )
            else:
                expected_rate = -math.log1p(-self.write_chi) / self.exposure_s
                if not math.isclose(
                    self.lambda_pair,
                    expected_rate,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        "lambda_pair must derive from write_chi and exposure for the V8 per-shot geometric law"
                    )
        if self.parameter_status not in DOMAIN_STATUSES:
            raise ValueError("unregistered parameter status")
        if self.parameter_status == "REGISTERED":
            if self.exposure_coordinate != "SI_SECONDS":
                raise ValueError(
                    "REGISTERED real-rate calibration requires an SI_SECONDS exposure coordinate"
                )
            if (
                self._registration_token is not _REGISTERED_TOKEN
                or len(self.parameter_receipt_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.parameter_receipt_sha
                )
            ):
                raise ValueError(
                    "REGISTERED operation parameters require a verified receipt constructor"
                )
        elif self.parameter_receipt_sha or self._registration_token is not None:
            raise ValueError(
                "unregistered operation parameters cannot carry a receipt token"
            )
        if (
            self.local_double_fraction or self.cross_double_fraction
        ) and not math.isclose(
            self.local_double_fraction + self.cross_double_fraction, 1.0, abs_tol=1e-12
        ):
            raise ValueError(
                "local/cross double fractions must be both zero or sum to one"
            )

    @property
    def effective_idler_efficiency(self) -> float:
        self.validate()
        return self.eta_retrieval * self.eta_idler * self.w_overlap

    @property
    def write_activation_rate(self) -> float:
        """Return the legacy-named non-vacuum activation hazard.

        This is ``lambda_pair`` only for compatibility.  Use the explicit
        probability properties below when reasoning about geometric sectors.
        """

        self.validate()
        return self.lambda_pair

    @property
    def nonvacuum_activation_probability(self) -> float:
        """Return ``P(n>=1)=chi`` for one V8 geometric write shot."""

        self.validate()
        return self.write_chi

    @property
    def exact_one_pair_probability(self) -> float:
        """Return ``P(n=1)=(1-chi)chi`` for one V8 geometric write shot."""

        self.validate()
        return (1.0 - self.write_chi) * self.write_chi

    @property
    def mean_excitation_per_shot(self) -> float:
        """Return ``E[n]=chi/(1-chi)`` for one V8 geometric write shot."""

        self.validate()
        return self.write_chi / (1.0 - self.write_chi)

    @property
    def background_signal_mean_per_shot(self) -> float:
        """Return the signal-background Poisson mean for one write shot."""

        self.validate()
        return self.exposure_s * self.background_signal_rate

    @property
    def background_idler_mean_per_shot(self) -> float:
        """Return the idler-background Poisson mean for one write shot."""

        self.validate()
        return self.exposure_s * self.background_idler_rate

    @property
    def dark_signal_mean_per_shot(self) -> float:
        """Return the signal-dark Poisson mean for one write shot."""

        self.validate()
        return self.exposure_s * self.dark_signal_rate

    @property
    def dark_idler_mean_per_shot(self) -> float:
        """Return the idler-dark Poisson mean for one write shot."""

        self.validate()
        return self.exposure_s * self.dark_idler_rate

    @property
    def effective_classical_shared_rate(self) -> float:
        self.validate()
        if self.classical_shared_rate == 0.0:
            return self.lambda_pair
        if not math.isclose(
            self.classical_shared_rate, self.lambda_pair, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("classical shared rate must be derived from lambda_pair")
        return self.classical_shared_rate

    def sha256(self) -> str:
        self.validate()
        payload = {
            "schema": DLCZ_OPERATION_SCHEMA,
            "operation_id": self.operation_id,
            "exposure_s": self.exposure_s,
            "repeats": self.repeats,
            "lambda_pair": self.lambda_pair,
            "write_chi": self.write_chi,
            "background_signal_rate": self.background_signal_rate,
            "background_idler_rate": self.background_idler_rate,
            "eta_signal": self.eta_signal,
            "eta_retrieval": self.eta_retrieval,
            "eta_idler": self.eta_idler,
            "w_overlap": self.w_overlap,
            "dark_signal_rate": self.dark_signal_rate,
            "dark_idler_rate": self.dark_idler_rate,
            "local_double_fraction": self.local_double_fraction,
            "cross_double_fraction": self.cross_double_fraction,
            "classical_shared_rate": self.classical_shared_rate,
            "storage_wait_s": self.storage_wait_s,
            "storage_lifetime_s": self.storage_lifetime_s,
            "eta_optical_path": self.eta_optical_path,
            "spinwave_coherence_visibility": self.spinwave_coherence_visibility,
            "source_law": self.source_law,
            "exposure_coordinate": self.exposure_coordinate,
            "parameter_status": self.parameter_status,
            "parameter_receipt_sha": self.parameter_receipt_sha,
        }
        return _hash_payload(payload)


@dataclass(frozen=True)
class DlczDomainContract:
    """Finite-window and g2 support contract."""

    g2_min: float = G2_MIN
    g2_max: float = G2_MAX
    denominator_floor: float = 1.0e-12
    gate_window_s: float = 190.0e-9
    dead_time_s: float = 0.0
    dead_time_mode: str = "none"
    external_association_gate_s: float = 200.0e-9
    read_clock_period_s: float = 10.0e-9
    read_clock_count: int = 19
    double_source_same_clock_probability: float = 1.0
    splitter_probability_by_spad: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    spad_efficiency_by_detector: tuple[float, ...] = ()
    status: str = "HYPOTHESIS_UNBOUND"

    def validate(self) -> None:
        _finite(self.g2_min, "g2_min", lower=0.0)
        _finite(self.g2_max, "g2_max", lower=self.g2_min)
        if self.g2_min != G2_MIN or self.g2_max != G2_MAX:
            raise ValueError("the synthetic g2 domain is registered as [1, 10]")
        _finite(self.denominator_floor, "denominator_floor", lower=0.0)
        _finite(self.gate_window_s, "gate_window_s", lower=0.0)
        _finite(self.dead_time_s, "dead_time_s", lower=0.0)
        _finite(
            self.external_association_gate_s,
            "external_association_gate_s",
            lower=0.0,
        )
        _finite(self.read_clock_period_s, "read_clock_period_s", lower=0.0)
        if type(self.read_clock_count) is not int or self.read_clock_count < 1:
            raise ValueError("read_clock_count must be a positive integer")
        _finite(
            self.double_source_same_clock_probability,
            "double_source_same_clock_probability",
            lower=0.0,
            upper=1.0,
        )
        if (
            len(self.splitter_probability_by_spad) != 4
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in self.splitter_probability_by_spad
            )
            or not math.isclose(
                float(sum(self.splitter_probability_by_spad)), 1.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError("splitter probabilities must be a four-SPAD simplex")
        if self.spad_efficiency_by_detector and (
            len(self.spad_efficiency_by_detector) != 4
            or any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.spad_efficiency_by_detector
            )
        ):
            raise ValueError("SPAD efficiencies must be empty or four probabilities")
        if self.dead_time_mode not in DEAD_TIME_MODES:
            raise ValueError("unregistered dead-time mode")
        if self.status not in DOMAIN_STATUSES:
            raise ValueError("unregistered domain status")

    def sha256(self) -> str:
        self.validate()
        return _hash_payload({"schema": DLCZ_DOMAIN_SCHEMA, **asdict(self)})

    @property
    def read_clock_support_s(self) -> float:
        self.validate()
        return float(self.read_clock_count) * float(self.read_clock_period_s)

    def spad_efficiencies(self, fallback_efficiency: float) -> tuple[float, ...]:
        self.validate()
        fallback = _finite(
            fallback_efficiency,
            "fallback_efficiency",
            lower=0.0,
            upper=1.0,
        )
        if self.spad_efficiency_by_detector:
            return tuple(float(value) for value in self.spad_efficiency_by_detector)
        return (fallback,) * 4


@dataclass(frozen=True)
class DlczCalibrationDraw:
    """Operation/session calibration draw used by the event renderer.

    Zero-error defaults are an explicit unbound hypothesis, not a claim that
    an apparatus is perfectly calibrated.
    """

    amplitude_error: tuple[float, ...]
    phase_error: tuple[float, ...]
    mode_overlap: tuple[float, ...]
    crosstalk: tuple[float, ...]
    status: str = "HYPOTHESIS_UNBOUND"
    receipt_sha: str = ""
    _registration_token: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_verified_receipt(
        cls,
        receipt_path: str,
        expected_sha: str,
        *,
        amplitude_error: Sequence[float],
        phase_error: Sequence[float],
        mode_overlap: Sequence[float],
        crosstalk: Sequence[float],
    ) -> "DlczCalibrationDraw":
        snapshot, receipt = load_json_snapshot(
            receipt_path,
            "DLCZ AOD calibration receipt",
            expected_sha=expected_sha,
            maximum_bytes=4 * 1024 * 1024,
        )
        if (
            receipt.get("status") != "REGISTERED"
            or receipt.get("receipt_kind") != "APPARATUS_CALIBRATION"
        ):
            raise ValueError(
                "REGISTERED calibration requires a verified apparatus receipt"
            )
        expected_payload = {
            "amplitude_error": tuple(float(value) for value in amplitude_error),
            "phase_error": tuple(float(value) for value in phase_error),
            "mode_overlap": tuple(float(value) for value in mode_overlap),
            "crosstalk": tuple(float(value) for value in crosstalk),
        }
        _require_receipt_payload(
            receipt,
            field_name="calibration_draw",
            expected=expected_payload,
        )
        return cls(
            expected_payload["amplitude_error"],
            expected_payload["phase_error"],
            expected_payload["mode_overlap"],
            expected_payload["crosstalk"],
            "REGISTERED",
            snapshot.sha256,
            _REGISTERED_TOKEN,
        )

    def validate(self) -> None:
        lengths = {
            len(self.amplitude_error),
            len(self.phase_error),
            len(self.mode_overlap),
            len(self.crosstalk),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
            raise ValueError(
                "calibration draw vectors must be aligned with at least three modes"
            )
        for name in ("amplitude_error", "phase_error", "mode_overlap", "crosstalk"):
            values = getattr(self, name)
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain finite values")
        if any(value < 0.0 or value > 1.0 for value in self.mode_overlap):
            raise ValueError("mode_overlap must lie in [0,1]")
        if any(value < 0.0 or value >= 1.0 for value in self.crosstalk):
            raise ValueError("crosstalk must lie in [0,1)")
        if self.status not in DOMAIN_STATUSES:
            raise ValueError("unregistered calibration status")
        if self.status == "REGISTERED":
            if (
                self._registration_token is not _REGISTERED_TOKEN
                or len(self.receipt_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.receipt_sha
                )
            ):
                raise ValueError(
                    "REGISTERED calibration requires a verified receipt constructor"
                )
        elif self.receipt_sha or self._registration_token is not None:
            raise ValueError("unregistered calibration cannot carry a receipt token")

    def sha256(self) -> str:
        self.validate()
        return _hash_payload(
            {
                "schema": DLCZ_CALIBRATION_SCHEMA,
                "amplitude_error": self.amplitude_error,
                "phase_error": self.phase_error,
                "mode_overlap": self.mode_overlap,
                "crosstalk": self.crosstalk,
                "status": self.status,
                "receipt_sha": self.receipt_sha,
            }
        )


@dataclass(frozen=True)
class DlczAllocationReceipt:
    """Deterministic quotient+remainder operation allocation."""

    requested_budget: int
    budget_unit: str
    operation_ids: tuple[str, ...]
    allocation_policy: str
    quotient: int
    remainder: int
    remainder_owner_ids: tuple[str, ...]
    repeats_by_operation: tuple[int, ...]
    b_trial_realized: int
    b_mode_shot_realized: int
    unspent: int
    operation_registry_sha: str
    n_mode: int = 1
    full_allocation_sha: str = ""

    @classmethod
    def from_requested(
        cls,
        requested_budget: int,
        operation_ids: Sequence[str],
        *,
        n_mode: int,
        budget_unit: str = "mode_shot",
        allocation_policy: str = "REGISTERED_OPERATION_ORDER",
        operation_registry_sha: str = "",
    ) -> "DlczAllocationReceipt":
        if type(requested_budget) is not int or requested_budget < 0:
            raise ValueError("requested_budget must be a non-negative integer")
        ids = tuple(operation_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("operation_ids must be unique and non-empty")
        if type(n_mode) is not int or n_mode < 1:
            raise ValueError("n_mode must be a positive integer")
        if budget_unit not in {"trial", "mode_shot"}:
            raise ValueError("budget_unit must be trial or mode_shot")
        if allocation_policy != "REGISTERED_OPERATION_ORDER":
            raise ValueError("only the registered-order allocator is implemented")
        operation_count = len(ids)
        requested_trials = (
            requested_budget if budget_unit == "trial" else requested_budget // n_mode
        )
        quotient, remainder = divmod(requested_trials, operation_count)
        repeats = tuple(
            quotient + int(index < remainder) for index in range(operation_count)
        )
        realized_trials = sum(repeats)
        realized_mode_shots = n_mode * realized_trials
        return cls(
            requested_budget=requested_budget,
            budget_unit=budget_unit,
            operation_ids=ids,
            allocation_policy=allocation_policy,
            quotient=quotient,
            remainder=remainder,
            remainder_owner_ids=ids[:remainder],
            repeats_by_operation=repeats,
            b_trial_realized=realized_trials,
            b_mode_shot_realized=realized_mode_shots,
            unspent=requested_budget
            - (realized_trials if budget_unit == "trial" else realized_mode_shots),
            operation_registry_sha=operation_registry_sha,
            n_mode=n_mode,
        )

    def validate(self) -> None:
        if type(self.requested_budget) is not int or self.requested_budget < 0:
            raise ValueError("requested_budget must be a non-negative integer")
        if self.budget_unit not in {"trial", "mode_shot"}:
            raise ValueError("invalid budget unit")
        if (
            not isinstance(self.operation_ids, tuple)
            or not self.operation_ids
            or len(set(self.operation_ids)) != len(self.operation_ids)
        ):
            raise ValueError("operation_ids must be unique")
        if self.allocation_policy != "REGISTERED_OPERATION_ORDER":
            raise ValueError("unregistered allocation policy")
        if type(self.n_mode) is not int or self.n_mode < 1:
            raise ValueError("n_mode must be a positive integer")
        if (
            type(self.quotient) is not int
            or type(self.remainder) is not int
            or self.quotient < 0
            or not 0 <= self.remainder < len(self.operation_ids)
        ):
            raise ValueError("invalid quotient/remainder")
        if self.remainder_owner_ids != self.operation_ids[: self.remainder]:
            raise ValueError("remainder owners must be the registered prefix")
        if len(self.repeats_by_operation) != len(self.operation_ids) or any(
            type(value) is not int or value < 0 for value in self.repeats_by_operation
        ):
            raise ValueError("repeats_by_operation must be non-negative integers")
        expected = tuple(
            self.quotient + int(index < self.remainder)
            for index in range(len(self.operation_ids))
        )
        if self.repeats_by_operation != expected:
            raise ValueError("allocation does not match quotient+remainder policy")
        if self.b_trial_realized != sum(self.repeats_by_operation):
            raise ValueError("B_trial does not match realized repeats")
        if self.b_mode_shot_realized != self.n_mode * self.b_trial_realized:
            raise ValueError("B_mode_shot does not match N_mode * B_trial")
        expected_unspent = self.requested_budget - (
            self.b_trial_realized
            if self.budget_unit == "trial"
            else self.b_mode_shot_realized
        )
        if self.unspent != expected_unspent or self.unspent < 0:
            raise ValueError("realized budget fields must be non-negative")
        if self.full_allocation_sha:
            if self.full_allocation_sha != self.compute_full_allocation_sha():
                raise ValueError("full allocation SHA mismatch")

    def compute_full_allocation_sha(self) -> str:
        self.validate_without_full_hash()
        return _hash_payload(
            {
                "schema": DLCZ_ALLOCATION_SCHEMA,
                "requested_budget": self.requested_budget,
                "budget_unit": self.budget_unit,
                "operation_ids": self.operation_ids,
                "allocation_policy": self.allocation_policy,
                "quotient": self.quotient,
                "remainder": self.remainder,
                "remainder_owner_ids": self.remainder_owner_ids,
                "repeats_by_operation": self.repeats_by_operation,
                "b_trial_realized": self.b_trial_realized,
                "b_mode_shot_realized": self.b_mode_shot_realized,
                "unspent": self.unspent,
                "operation_registry_sha": self.operation_registry_sha,
                "n_mode": self.n_mode,
            }
        )

    def validate_without_full_hash(self) -> None:
        if type(self.requested_budget) is not int or self.requested_budget < 0:
            raise ValueError("requested_budget must be a non-negative integer")
        if self.budget_unit not in {"trial", "mode_shot"}:
            raise ValueError("invalid budget unit")
        if self.allocation_policy != "REGISTERED_OPERATION_ORDER":
            raise ValueError("unregistered allocation policy")
        if type(self.n_mode) is not int or self.n_mode < 1:
            raise ValueError("n_mode must be a positive integer")
        if not self.operation_ids or len(set(self.operation_ids)) != len(
            self.operation_ids
        ):
            raise ValueError("operation_ids must be unique")
        if self.remainder_owner_ids != self.operation_ids[: self.remainder]:
            raise ValueError("remainder owners must be the registered prefix")
        expected = tuple(
            self.quotient + int(index < self.remainder)
            for index in range(len(self.operation_ids))
        )
        if self.repeats_by_operation != expected or self.b_trial_realized != sum(
            expected
        ):
            raise ValueError("allocation does not match quotient+remainder policy")
        if self.b_mode_shot_realized != self.n_mode * self.b_trial_realized:
            raise ValueError("B_mode_shot does not match N_mode * B_trial")

    def with_hash(self) -> "DlczAllocationReceipt":
        self.validate_without_full_hash()
        return DlczAllocationReceipt(
            **{
                **asdict(self),
                "full_allocation_sha": self.compute_full_allocation_sha(),
            }
        )

    def sha256(self) -> str:
        self.validate()
        return _hash_payload({"schema": DLCZ_ALLOCATION_SCHEMA, **asdict(self)})


def source_event_mark_for_occupancy(occupancy: Sequence[int]) -> str:
    """Classify a non-empty occupancy vector without collapsing topology."""
    values = tuple(occupancy)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("occupancy must be non-negative")
    total = sum(values)
    if total == 0:
        return "NO_SOURCE_EVENT"
    if total == 1:
        return "SINGLE_CELL_SINGLE_EXCITATION"
    if total == 2:
        occupied = [value for value in values if value]
        return "LOCAL_DOUBLE" if 2 in occupied else "CROSS_DOUBLE"
    return "MULTIPAIR_K_GE_3"


def contract_sha(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible contract payload for provenance binding."""
    return _hash_payload(value)
