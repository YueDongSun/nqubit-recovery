"""Versioned public measurement metadata for family-context response models.

The context binds only admitted public operation metadata.  It does not carry
row identifiers, target counts, RF values, or learned operation embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .contracts import canonical_sha256, require_sha256


MEASUREMENT_CONTEXT_SCHEMA = "TrackBEffectiveMeasurementContextV1"
REGISTERED_MAXIMUM_FAMILY_COEFFICIENT = 4.0
REGISTERED_MAXIMUM_FAMILY_CORRECTION_L2 = 8.0
COEFFICIENT_INTERPRETATION = "BOUNDED_ORTHONORMAL_FAMILY_CONTRAST_COEFFICIENT"
CORRECTION_INTERPRETATION = "VISIBLE_CENTERED_PER_ROW_BRANCH_FAMILY_CORRECTION_WITH_L2_BOUND"


@dataclass(frozen=True)
class Effective_measurement_context:
    """Canonical operation-to-family metadata committed by one admission."""

    admission_manifest_sha: str
    operation_metadata_sha: str
    family_vocabulary: tuple[str, ...]
    operation_family_mapping: tuple[tuple[str, str], ...]
    maximum_family_coefficient: float = REGISTERED_MAXIMUM_FAMILY_COEFFICIENT
    maximum_family_correction_l2: float = REGISTERED_MAXIMUM_FAMILY_CORRECTION_L2
    coefficient_interpretation: str = COEFFICIENT_INTERPRETATION
    correction_interpretation: str = CORRECTION_INTERPRETATION
    schema: str = MEASUREMENT_CONTEXT_SCHEMA

    def validate(self) -> None:
        require_sha256(self.admission_manifest_sha, "measurement_context_admission_manifest_sha")
        require_sha256(self.operation_metadata_sha, "measurement_context_operation_metadata_sha")
        if self.schema != MEASUREMENT_CONTEXT_SCHEMA:
            raise ValueError("effective measurement context schema is invalid")
        if self.coefficient_interpretation != COEFFICIENT_INTERPRETATION \
                or self.correction_interpretation != CORRECTION_INTERPRETATION:
            raise ValueError("effective measurement context interpretation is invalid")
        if self.maximum_family_coefficient != REGISTERED_MAXIMUM_FAMILY_COEFFICIENT \
                or self.maximum_family_correction_l2 != REGISTERED_MAXIMUM_FAMILY_CORRECTION_L2 \
                or not all(math.isfinite(value) for value in (
                    self.maximum_family_coefficient, self.maximum_family_correction_l2,
                )):
            raise ValueError("effective measurement context bounds are not registered")
        vocabulary = self.family_vocabulary
        if type(vocabulary) is not tuple or len(vocabulary) < 2 \
                or any(type(value) is not str or not value for value in vocabulary) \
                or vocabulary != tuple(sorted(set(vocabulary))):
            raise ValueError("family vocabulary must be a sorted unique tuple with at least two families")
        mapping = self.operation_family_mapping
        if type(mapping) is not tuple or not mapping or mapping != tuple(sorted(mapping)):
            raise ValueError("operation-family mapping must be a nonempty canonical tuple")
        operation_ids = []
        mapped_families = []
        for entry in mapping:
            if type(entry) is not tuple or len(entry) != 2 \
                    or any(type(value) is not str or not value for value in entry):
                raise ValueError("operation-family entries must contain nonempty string pairs")
            operation_ids.append(entry[0])
            mapped_families.append(entry[1])
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("operation-family mapping repeats an operation ID")
        if set(mapped_families) != set(vocabulary):
            raise ValueError("operation-family mapping and vocabulary differ")

    @classmethod
    def from_operation_metadata(
        cls, *, admission_manifest_sha: str, operation_metadata_sha: str,
        operation_ids: Sequence[str], operation_families: Sequence[str],
    ) -> "Effective_measurement_context":
        if len(operation_ids) != len(operation_families) or not operation_ids:
            raise ValueError("operation IDs and families must have one nonempty shared length")
        if any(type(value) is not str or not value for value in (*operation_ids, *operation_families)):
            raise ValueError("operation metadata must contain nonempty strings without coercion")
        mapping = tuple(sorted(zip(operation_ids, operation_families, strict=True)))
        result = cls(
            admission_manifest_sha=admission_manifest_sha,
            operation_metadata_sha=operation_metadata_sha,
            family_vocabulary=tuple(sorted({family for _, family in mapping})),
            operation_family_mapping=mapping,
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "admission_manifest_sha": self.admission_manifest_sha,
            "operation_metadata_sha": self.operation_metadata_sha,
            "family_vocabulary": list(self.family_vocabulary),
            "operation_family_mapping": [list(value) for value in self.operation_family_mapping],
            "maximum_family_coefficient": self.maximum_family_coefficient,
            "maximum_family_correction_l2": self.maximum_family_correction_l2,
            "coefficient_interpretation": self.coefficient_interpretation,
            "correction_interpretation": self.correction_interpretation,
            "schema": self.schema,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Effective_measurement_context":
        expected = {
            "admission_manifest_sha", "operation_metadata_sha", "family_vocabulary",
            "operation_family_mapping", "maximum_family_coefficient",
            "maximum_family_correction_l2", "coefficient_interpretation",
            "correction_interpretation", "schema",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("effective measurement context fields must match exactly")
        vocabulary = payload["family_vocabulary"]
        mapping = payload["operation_family_mapping"]
        if not isinstance(vocabulary, (list, tuple)) or not isinstance(mapping, (list, tuple)):
            raise ValueError("effective measurement context collections are invalid")
        result = cls(
            admission_manifest_sha=payload["admission_manifest_sha"],
            operation_metadata_sha=payload["operation_metadata_sha"],
            family_vocabulary=tuple(vocabulary),
            operation_family_mapping=tuple(
                tuple(value) if isinstance(value, (list, tuple)) else value for value in mapping
            ),
            maximum_family_coefficient=payload["maximum_family_coefficient"],
            maximum_family_correction_l2=payload["maximum_family_correction_l2"],
            coefficient_interpretation=payload["coefficient_interpretation"],
            correction_interpretation=payload["correction_interpretation"],
            schema=payload["schema"],
        )
        result.validate()
        if result.to_dict() != dict(payload):
            raise ValueError("effective measurement context changes under typed parsing")
        return result

    def batch_family_indices(
        self, batch: Mapping[str, Any], *, operation_count: int,
    ) -> tuple[int, ...]:
        """Validate public batch metadata and return permutation-aware family indices."""
        self.validate()
        if batch.get("admission_manifest_sha") != self.admission_manifest_sha:
            raise ValueError("batch admission identity differs from the measurement context")
        if "operation_metadata_sha" in batch \
                and batch.get("operation_metadata_sha") != self.operation_metadata_sha:
            raise ValueError("batch operation metadata identity differs from the measurement context")
        operation_ids = batch.get("operation_ids")
        operation_families = batch.get("operation_families")
        if type(operation_ids) is not tuple or type(operation_families) is not tuple \
                or len(operation_ids) != operation_count or len(operation_families) != operation_count \
                or any(type(value) is not str or not value for value in operation_ids + operation_families) \
                or len(set(operation_ids)) != operation_count:
            raise ValueError("batch operation-family metadata is incomplete")
        expected_mapping = dict(self.operation_family_mapping)
        if set(operation_ids) != set(expected_mapping) \
                or any(expected_mapping[operation_id] != family for operation_id, family in zip(
                    operation_ids, operation_families, strict=True,
                )):
            raise ValueError("batch operation ID-to-family correspondence differs from the measurement context")
        family_indices = {family: index for index, family in enumerate(self.family_vocabulary)}
        return tuple(family_indices[family] for family in operation_families)
