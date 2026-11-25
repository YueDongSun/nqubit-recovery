"""Single source of truth for V2.4 contract and materialization identities."""

from __future__ import annotations

import hashlib
import json


SOURCE_CONTRACT_VERSION = "2.3"
DATASET_CONTRACT_VERSION = "2.4.0"
CONTRACT_KIND = "NQubit Dataset Contract V2.4.0"
GENERATOR_REVISION = "data_v2_4_generator_r9_physics_closure_v2"
REAL_CONVERTER_REVISION = "data_v2_4_real_converter_r2"
MATERIALIZATION_PROFILE_VERSION = "v24_materialization_v1"
NOISE_INTERVENTION_REVISION = "v24_noise_intervention_r2"


def _identity_sha(kind: str, revision: str) -> str:
    payload = json.dumps(
        {kind: revision}, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


GENERATOR_SHA = _identity_sha("generator", GENERATOR_REVISION)
REAL_CONVERTER_SHA = _identity_sha("converter", REAL_CONVERTER_REVISION)
MATERIALIZATION_PROFILE_SCHEMA_SHA = _identity_sha(
    "materialization_profile", MATERIALIZATION_PROFILE_VERSION)
NOISE_INTERVENTION_SHA = _identity_sha(
    "noise_intervention", NOISE_INTERVENTION_REVISION)
