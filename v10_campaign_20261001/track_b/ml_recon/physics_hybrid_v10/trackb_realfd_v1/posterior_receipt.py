"""Source-bound producer receipt for Track B posterior payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from typing import Any

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .forward_draw_set import BoundTrackBForwardDrawSetV1
from .observation_plate import (
    TrainableMaskedObservationPlateConfigV1,
    TrainableMaskedObservationPlateV1,
)
from .effective_observation_v2 import Effective_singleton_observation
from .posterior import TrackBPosteriorV1
from .source_artifacts import (
    ACTIONFUL_SOURCE_ARTIFACT_NAMES,
    TrackBActionfulSourceArtifactBundleV1,
)
from .posterior_producer import (
    EXECUTION_RECEIPT_SCHEMA,
    REGISTERED_POSTERIOR_OUTPUT_SCHEMA,
    REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME,
    REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH,
)
from .datasets import TrackBRealDataAdmissionManifestV1, TrackBRealFoldAdmissionV1
from .training import (
    TrackBRealRuntimeIdentityV1,
    TrackBRealBatchPlanV1,
    TrackBRealTrainingRuntimeV1,
    trackb_model_state_sha,
)


def _load_native_trackb_observation_plate(
    checkpoint_bytes: bytes,
    *,
    expected_source_commit: str | None = None,
    expected_source_tree: str | None = None,
    expected_admission_manifest_sha: str | None = None,
) -> tuple[Any, TrackBRealRuntimeIdentityV1]:
    """Load a complete research-only Track B observation checkpoint strictly."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("actionful posterior receipt requires a weights-only checkpoint reader") from error
    checkpoint_payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True,
    )
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("posterior producer checkpoint is not a native Track B mapping")
    metadata = checkpoint_payload.get("metadata")
    model_state = checkpoint_payload.get("model_state_dict")
    effective_checkpoint_schemas = {
        "TrackBEffectiveObservationCheckpointV2",
        "TrackBEffectiveFamilyContextCheckpointV3",
    }
    if not isinstance(metadata, dict) \
            or metadata.get("schema") not in {
                TrackBRealTrainingRuntimeV1.CHECKPOINT_SCHEMA,
                *effective_checkpoint_schemas,
            } \
            or not isinstance(model_state, dict) or not model_state:
        raise ValueError("posterior producer checkpoint is not a complete native Track B checkpoint")

    # Effective V2/V3 checkpoints use the newer typed reader because their
    # configuration and terminal-state contracts are wider than the legacy
    # observation plate.  Keep the legacy path below unchanged for existing
    # callers, while making the effective path reuse one strict authority.
    if metadata.get("schema") in effective_checkpoint_schemas:
        from .effective_checkpoint_reader import load_effective_checkpoint_bytes

        file_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        identity_sha = metadata.get("identity_sha")
        if type(identity_sha) is not str:
            raise ValueError("effective checkpoint identity SHA is missing")
        source_commit = metadata.get("identity", {}).get("source_commit") \
            if isinstance(metadata.get("identity"), dict) else None
        source_tree = metadata.get("identity", {}).get("source_tree") \
            if isinstance(metadata.get("identity"), dict) else None
        if type(source_commit) is not str or type(source_tree) is not str:
            raise ValueError("effective checkpoint source identity is missing")
        replay = load_effective_checkpoint_bytes(
            checkpoint_bytes,
            expected_file_sha=file_sha,
            expected_identity_sha=identity_sha,
            expected_source_commit=expected_source_commit or source_commit,
            expected_source_tree=expected_source_tree or source_tree,
        )
        if expected_admission_manifest_sha is not None \
                and replay.identity.admission_manifest_sha != expected_admission_manifest_sha:
            raise ValueError("effective checkpoint identity admission manifest is invalid")
        return replay.final_model, replay.identity
    try:
        model_config = TrainableMaskedObservationPlateConfigV1.from_dict(
            metadata.get("model_config", {}),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("posterior producer checkpoint model config is invalid") from error
    if metadata.get("model_config_sha") != model_config.sha256():
        raise ValueError("posterior producer checkpoint model config SHA is invalid")
    try:
        identity = TrackBRealRuntimeIdentityV1.from_dict(metadata.get("identity", {}))
    except (TypeError, ValueError) as error:
        raise ValueError("posterior producer checkpoint runtime identity is invalid") from error
    if metadata.get("identity_sha") != identity.sha256() \
            or identity.system_type != model_config.system_type \
            or identity.model_config_sha != model_config.sha256():
        raise ValueError("posterior producer checkpoint identity does not bind its model config")
    if expected_source_commit is not None and identity.source_commit != expected_source_commit:
        raise ValueError("posterior producer checkpoint identity source commit is invalid")
    if expected_source_tree is not None and identity.source_tree != expected_source_tree:
        raise ValueError("posterior producer checkpoint identity source tree is invalid")
    if expected_admission_manifest_sha is not None \
            and identity.admission_manifest_sha != expected_admission_manifest_sha:
        raise ValueError("posterior producer checkpoint identity admission manifest is invalid")
    if metadata.get("research_only") is not True \
            or metadata.get("hardware_action_authorized") is not False:
        raise ValueError("posterior producer checkpoint safety authority is invalid")
    model = TrainableMaskedObservationPlateV1(model_config)
    expected_state = model.state_dict()
    if set(model_state) != set(expected_state) or any(
        not isinstance(model_state[name], torch.Tensor)
        or model_state[name].shape != expected_state[name].shape
        or model_state[name].dtype != expected_state[name].dtype
        or (
            (model_state[name].is_floating_point() or model_state[name].is_complex())
            and not bool(torch.all(torch.isfinite(model_state[name])).item())
        )
        for name in expected_state
    ):
        raise ValueError("posterior producer checkpoint model state does not match its native model")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError("posterior producer checkpoint strict model load failed") from error
    if metadata.get("model_state_sha") != trackb_model_state_sha(model):
        raise ValueError("posterior producer checkpoint model-state SHA is invalid")
    model.eval()
    return model, identity


def _validate_checkpoint_session_contract(
    checkpoint_model: Any,
    checkpoint_identity: TrackBRealRuntimeIdentityV1,
    fold_admission: TrackBRealFoldAdmissionV1,
) -> None:
    """Validate the session ABI that is actually represented by a checkpoint.

    Legacy observation plates own a bounded session embedding vocabulary.  The
    Effective V2/V3 models do not: their session provenance is instead bound
    to the exact admitted rows in ``TrackBRealRuntimeIdentityV1``.  Do not
    invent a vocabulary size for the latter merely because both formats carry
    per-sample session bindings.
    """
    admitted_bindings = fold_admission.session_index_bindings
    checkpoint_bindings = checkpoint_identity.session_index_bindings
    if type(checkpoint_model) is TrainableMaskedObservationPlateV1:
        if admitted_bindings != checkpoint_bindings:
            raise ValueError("posterior producer selected-fold session binding differs from checkpoint identity")
        if any(
            session_index >= checkpoint_model.config.session_count
            for _sample_id, session_index in admitted_bindings
        ):
            raise ValueError("posterior producer selected-fold session binding exceeds model vocabulary")
        return
    if type(checkpoint_model) is Effective_singleton_observation:
        if admitted_bindings != checkpoint_bindings:
            raise ValueError("effective checkpoint session bindings do not match admitted row identity")
        return
    raise TypeError("posterior producer checkpoint has an unsupported session ABI")


@dataclass(frozen=True)
class BoundTrackBPosteriorReceiptV1:
    """Bind an exact posterior numeric payload to its source producer inputs.

    A raw :class:`TrackBPosteriorV1` remains useful for diagnostics, but an
    actionful proposal path must consume this receipt so model/data/row/batch
    provenance is present at the same boundary as the posterior moments.
    """

    posterior: TrackBPosteriorV1
    posterior_payload_sha: str
    source_commit: str
    source_tree: str
    model_checkpoint_sha: str
    model_output_bundle_sha: str
    data_manifest_sha: str
    row_identity_sha: str
    batch_plan_sha: str
    forward_draw_set_sha: str
    forward_draw_ids_sha: str
    producer_source_sha: str
    producer_source_git_blob_oid: str
    producer_config_sha: str
    calibration_coverage_sha: str
    producer_manifest_sha: str
    source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1
    source_artifact_bundle_sha: str
    forward_source_binding_sha: str
    schema: str = field(default="BoundTrackBPosteriorReceiptV1", init=False)

    def __post_init__(self) -> None:
        if type(self.posterior) is not TrackBPosteriorV1:
            raise TypeError("posterior receipt requires the exact TrackBPosteriorV1 type")
        self.posterior.__post_init__()
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "posterior_payload_sha", "model_checkpoint_sha", "model_output_bundle_sha",
            "data_manifest_sha", "row_identity_sha", "batch_plan_sha", "forward_draw_set_sha",
            "forward_draw_ids_sha", "producer_source_sha", "producer_config_sha",
            "calibration_coverage_sha",
            "producer_manifest_sha",
            "source_artifact_bundle_sha", "forward_source_binding_sha",
        ):
            require_sha256(getattr(self, name), name)
        require_git_oid(self.producer_source_git_blob_oid, "producer_source_git_blob_oid")
        if type(self.source_artifact_bundle) is not TrackBActionfulSourceArtifactBundleV1:
            raise TypeError("posterior receipt requires an exact file-backed source artifact bundle")
        if self.source_artifact_bundle.source_commit != self.source_commit \
                or self.source_artifact_bundle.source_tree != self.source_tree:
            raise ValueError("posterior receipt source bundle commit/tree differs from the receipt")
        if self.source_artifact_bundle.actionful_binding_sha256 != self.source_artifact_bundle_sha:
            raise ValueError("posterior receipt source artifact bundle changed after admission")
        expected_artifact_shas = {
            "posterior_payload": self.posterior_payload_sha,
            "model_checkpoint": self.model_checkpoint_sha,
            "model_output_bundle": self.model_output_bundle_sha,
            "data_manifest": self.data_manifest_sha,
            "row_identity": self.row_identity_sha,
            "batch_plan": self.batch_plan_sha,
            "producer_source": self.producer_source_sha,
            "producer_config": self.producer_config_sha,
            "calibration_coverage": self.calibration_coverage_sha,
            "producer_manifest": self.producer_manifest_sha,
        }
        self.source_artifact_bundle.assert_current(ACTIONFUL_SOURCE_ARTIFACT_NAMES)
        if any(
            self.source_artifact_bundle.artifact_sha256(name) != expected_sha
            for name, expected_sha in expected_artifact_shas.items()
        ):
            raise ValueError("posterior receipt fields do not match the rehashed source artifacts")
        admission_manifest = TrackBRealDataAdmissionManifestV1.from_canonical_json_bytes(
            self.source_artifact_bundle.read_bytes("data_manifest"),
        )
        if admission_manifest.sha256() != self.data_manifest_sha:
            raise ValueError("posterior producer data manifest SHA does not match its native payload")
        checkpoint_model, checkpoint_identity = _load_native_trackb_observation_plate(
            self.source_artifact_bundle.read_bytes("model_checkpoint"),
            expected_source_commit=self.source_commit,
            expected_source_tree=self.source_tree,
            expected_admission_manifest_sha=self.data_manifest_sha,
        )
        if checkpoint_model.config.system_type != self.posterior.system_type \
                or admission_manifest.system_type != self.posterior.system_type:
            raise ValueError("posterior producer checkpoint, data manifest, and posterior systems must align")
        fold_admission = TrackBRealFoldAdmissionV1.from_canonical_json_bytes(
            self.source_artifact_bundle.read_bytes("row_identity"),
        )
        batch_plan = TrackBRealBatchPlanV1.from_canonical_json_bytes(
            self.source_artifact_bundle.read_bytes("batch_plan"),
        )
        if fold_admission.admission_manifest_sha != admission_manifest.sha256() \
                or fold_admission.outer_fold_sha != checkpoint_identity.outer_fold_sha \
                or fold_admission.selected_row_identity_sha != checkpoint_identity.row_identity_sha \
                or fold_admission.selected_sample_ids != checkpoint_identity.allowed_sample_ids \
                or fold_admission.selected_group_ids != checkpoint_identity.allowed_group_ids \
                or fold_admission.selected_row_identity_records != checkpoint_identity.allowed_row_identities:
            raise ValueError("posterior producer selected-fold admission does not match checkpoint identity")
        _validate_checkpoint_session_contract(
            checkpoint_model,
            checkpoint_identity,
            fold_admission,
        )
        if batch_plan.sample_ids != checkpoint_identity.batch_plan_sample_ids \
                or batch_plan.batch_size != checkpoint_identity.batch_plan_batch_size \
                or batch_plan.view_seed != checkpoint_identity.batch_plan_view_seed \
                or batch_plan.query_fraction != checkpoint_identity.batch_plan_query_fraction \
                or batch_plan.semantic_sha256() != checkpoint_identity.batch_plan_sha:
            raise ValueError("posterior producer batch plan does not match checkpoint identity")
        if self.source_artifact_bundle.read_bytes("posterior_payload") \
                != canonical_json_bytes(self.posterior.to_dict()):
            raise ValueError("posterior receipt payload file does not match the live posterior")
        producer_source = self.source_artifact_bundle.artifact("producer_source")
        if producer_source.git_blob_oid != self.producer_source_git_blob_oid:
            raise ValueError("posterior receipt producer source does not match its tracked Git blob")
        config = self.source_artifact_bundle.read_canonical_json_object("producer_config")
        expected_config_fields = {
            "schema", "producer_relative_path", "producer_qualified_name",
            "producer_recipe_sha", "output_schema",
        }
        if set(config) != expected_config_fields \
                or config.get("schema") != "TrackBPosteriorProducerConfigV1" \
                or config.get("producer_relative_path") != REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH \
                or config.get("producer_qualified_name") != REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME \
                or config.get("output_schema") != REGISTERED_POSTERIOR_OUTPUT_SCHEMA:
            raise ValueError("posterior producer configuration is not the registered source contract")
        expected_recipe_sha = canonical_sha256({
            "schema": "TrackBPosteriorProducerRecipeV1",
            "producer_relative_path": REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH,
            "producer_qualified_name": REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME,
            "output_schema": REGISTERED_POSTERIOR_OUTPUT_SCHEMA,
        })
        if config.get("producer_recipe_sha") != expected_recipe_sha:
            raise ValueError("posterior producer configuration recipe identity is invalid")
        output = self.source_artifact_bundle.read_canonical_json_object("model_output_bundle")
        expected_output = {
            "schema": REGISTERED_POSTERIOR_OUTPUT_SCHEMA,
            "posterior_payload_sha": self.posterior_payload_sha,
            "model_checkpoint_sha": self.model_checkpoint_sha,
            "data_manifest_sha": self.data_manifest_sha,
            "row_identity_sha": self.row_identity_sha,
            "batch_plan_sha": self.batch_plan_sha,
            "calibration_coverage_sha": self.calibration_coverage_sha,
            "producer_config_sha": self.producer_config_sha,
        }
        if output != expected_output:
            raise ValueError("posterior producer output does not bind its consumed artifacts")
        manifest = self.source_artifact_bundle.read_canonical_json_object("producer_manifest")
        expected_manifest_fields = {
            "schema", "source_commit", "source_tree", "producer_relative_path",
            "producer_qualified_name", "producer_recipe_sha", "output_schema",
            "producer_source_git_blob_oid", "artifact_shas",
        }
        expected_artifact_shas = {
            name: self.source_artifact_bundle.artifact_sha256(name)
            for name in ACTIONFUL_SOURCE_ARTIFACT_NAMES
            if name != "producer_manifest"
        }
        if set(manifest) != expected_manifest_fields \
                or manifest.get("schema") != EXECUTION_RECEIPT_SCHEMA \
                or manifest.get("source_commit") != self.source_commit \
                or manifest.get("source_tree") != self.source_tree \
                or manifest.get("producer_relative_path") != REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH \
                or manifest.get("producer_qualified_name") != REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME \
                or manifest.get("producer_recipe_sha") != expected_recipe_sha \
                or manifest.get("output_schema") != REGISTERED_POSTERIOR_OUTPUT_SCHEMA \
                or manifest.get("producer_source_git_blob_oid") != self.producer_source_git_blob_oid \
                or manifest.get("artifact_shas") != expected_artifact_shas:
            raise ValueError("posterior producer execution receipt does not cross-bind source artifacts")
        if self.posterior_payload_sha != self.posterior.sha256():
            raise ValueError("posterior receipt numeric payload SHA does not match the posterior")
        if self.posterior.forward_draw_set_sha != self.forward_draw_set_sha \
                or self.posterior.forward_draw_ids_sha != self.forward_draw_ids_sha:
            raise ValueError("posterior receipt draw identities do not match the posterior")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "posterior": {
                "payload_sha": self.posterior_payload_sha,
                "system_type": self.posterior.system_type,
                "forward_sha": self.posterior.forward_sha,
                "covariance_sha": self.posterior.covariance_sha,
                "forward_draw_set_sha": self.posterior.forward_draw_set_sha,
                "forward_draw_ids_sha": self.posterior.forward_draw_ids_sha,
            },
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "model_checkpoint_sha": self.model_checkpoint_sha,
            "model_output_bundle_sha": self.model_output_bundle_sha,
            "data_manifest_sha": self.data_manifest_sha,
            "row_identity_sha": self.row_identity_sha,
            "batch_plan_sha": self.batch_plan_sha,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
            "producer_source_sha": self.producer_source_sha,
            "producer_source_git_blob_oid": self.producer_source_git_blob_oid,
            "producer_config_sha": self.producer_config_sha,
            "calibration_coverage_sha": self.calibration_coverage_sha,
            "producer_manifest_sha": self.producer_manifest_sha,
            "source_artifact_bundle_sha": self.source_artifact_bundle_sha,
            "forward_source_binding_sha": self.forward_source_binding_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_source_artifacts(
        cls,
        *,
        posterior: TrackBPosteriorV1,
        forward_draw_set: BoundTrackBForwardDrawSetV1,
        source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1,
    ) -> "BoundTrackBPosteriorReceiptV1":
        if type(posterior) is not TrackBPosteriorV1 \
                or type(forward_draw_set) is not BoundTrackBForwardDrawSetV1:
            raise TypeError("posterior source receipt requires exact posterior and forward draw types")
        forward_draw_set.assert_actionful_source_bound()
        if forward_draw_set.forward_source_binding_sha is None:
            raise ValueError("file-backed forward draw set has no source binding SHA")
        source_artifact_bundle.assert_current(ACTIONFUL_SOURCE_ARTIFACT_NAMES)
        if source_artifact_bundle.forward_binding_sha256 != forward_draw_set.forward_source_binding_sha:
            raise ValueError("posterior source bundle does not bind the supplied forward draw producer")
        if source_artifact_bundle.read_bytes("posterior_payload") != canonical_json_bytes(posterior.to_dict()):
            raise ValueError("posterior source payload file does not match the supplied posterior")
        return cls(
            posterior=posterior,
            posterior_payload_sha=source_artifact_bundle.artifact_sha256("posterior_payload"),
            source_commit=source_artifact_bundle.source_commit,
            source_tree=source_artifact_bundle.source_tree,
            model_checkpoint_sha=source_artifact_bundle.artifact_sha256("model_checkpoint"),
            model_output_bundle_sha=source_artifact_bundle.artifact_sha256("model_output_bundle"),
            data_manifest_sha=source_artifact_bundle.artifact_sha256("data_manifest"),
            row_identity_sha=source_artifact_bundle.artifact_sha256("row_identity"),
            batch_plan_sha=source_artifact_bundle.artifact_sha256("batch_plan"),
            forward_draw_set_sha=forward_draw_set.sha256(),
            forward_draw_ids_sha=canonical_sha256({
                "schema": "TrackBForwardDrawIDsV1",
                "draw_ids": list(forward_draw_set.draw_ids),
            }),
            producer_source_sha=source_artifact_bundle.artifact_sha256("producer_source"),
            producer_source_git_blob_oid=(
                source_artifact_bundle.artifact("producer_source").git_blob_oid or ""
            ),
            producer_config_sha=source_artifact_bundle.artifact_sha256("producer_config"),
            calibration_coverage_sha=source_artifact_bundle.artifact_sha256("calibration_coverage"),
            producer_manifest_sha=source_artifact_bundle.artifact_sha256("producer_manifest"),
            source_artifact_bundle=source_artifact_bundle,
            source_artifact_bundle_sha=source_artifact_bundle.actionful_binding_sha256,
            forward_source_binding_sha=forward_draw_set.forward_source_binding_sha,
        )

    def verify_for(
        self, *, posterior: TrackBPosteriorV1, forward_draw_set: BoundTrackBForwardDrawSetV1,
        forward_draw_set_sha: str, forward_draw_ids_sha: str, source_commit: str, source_tree: str,
    ) -> None:
        self.__post_init__()
        if type(posterior) is not TrackBPosteriorV1:
            raise TypeError("posterior receipt verification requires the exact posterior type")
        if type(forward_draw_set) is not BoundTrackBForwardDrawSetV1:
            raise TypeError("posterior receipt verification requires the exact forward draw set type")
        forward_draw_set.assert_actionful_source_bound()
        if forward_draw_set.forward_source_binding_sha != self.forward_source_binding_sha \
                or self.source_artifact_bundle.forward_binding_sha256 != self.forward_source_binding_sha:
            raise ValueError("posterior receipt source bundle does not bind the live forward producer")
        if self.posterior_payload_sha != posterior.sha256() \
                or self.forward_draw_set_sha != forward_draw_set_sha \
                or self.forward_draw_ids_sha != forward_draw_ids_sha \
                or self.source_commit != source_commit or self.source_tree != source_tree:
            raise ValueError("posterior receipt does not bind the live proposal inputs")


__all__ = ["BoundTrackBPosteriorReceiptV1"]
