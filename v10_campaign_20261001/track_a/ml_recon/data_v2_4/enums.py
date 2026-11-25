"""Closed vocabularies for V2.4 authority, visibility, and identifiability."""

from __future__ import annotations

from enum import Enum


class Authority(str, Enum):
    EXACT_BY_CONSTRUCTION = "EXACT_BY_CONSTRUCTION"
    MEASURED_OBSERVATION = "MEASURED_OBSERVATION"
    DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
    WEAK_ESTIMATE = "WEAK_ESTIMATE"
    UNAVAILABLE = "UNAVAILABLE"


class Visibility(str, Enum):
    DEPLOYABLE_INPUT = "DEPLOYABLE_INPUT"
    DERIVED_DEPLOYABLE_INPUT = "DERIVED_DEPLOYABLE_INPUT"
    PRIVILEGED_TEACHER_ONLY = "PRIVILEGED_TEACHER_ONLY"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    PROVENANCE_ONLY = "PROVENANCE_ONLY"


class Identifiability(str, Enum):
    IDENTIFIABLE_UNDER_THIS_VIEW = "IDENTIFIABLE_UNDER_THIS_VIEW"
    EQUIVALENCE_CLASS_ONLY = "EQUIVALENCE_CLASS_ONLY"
    NOT_AUDITED = "NOT_AUDITED"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"


class StorageTier(str, Enum):
    CORE = "CORE"
    EXTENDED = "EXTENDED"
    AUDIT = "AUDIT"


class Provenance(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class MeasurementDomain(str, Enum):
    GLOBAL_PAULI = "global_pauli_v24"
    DLCZ_OPERATION = "dlcz_operation_v24"


DATASET_FAMILIES = (
    "small_n_family",
    "general_n_structure",
    "w_control",
    "feedback_one_step",
    "feedback_history",
    "feedback_noop",
    "mechanism_shift",
    "ood_detector_drift",
    "locked_ood_composite",
    "negative_control",
)

N_PANEL = (3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256)
