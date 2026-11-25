"""Exact-artifact rollback records; live rollback remains unavailable here."""

from __future__ import annotations

from .schemas import (
    V10CampaignAuthority,
    V10MaterializedArtifact,
    V10RollbackRecord,
    V10UploadReceipt,
)


def rollback_simulation(
    *,
    authority: V10CampaignAuthority,
    harmful_artifact: V10MaterializedArtifact,
    restored_artifact: V10MaterializedArtifact,
    restored_receipt: V10UploadReceipt,
) -> V10RollbackRecord:
    authority.validate()
    if not authority.simulate_only or not harmful_artifact.simulate_only \
            or not restored_artifact.simulate_only or not restored_receipt.simulate_only:
        raise PermissionError("this V10 rollback path is simulation-only")
    if restored_receipt.artifact_sha != restored_artifact.sha256():
        raise ValueError("rollback receipt does not bind the restored artifact")
    return V10RollbackRecord(
        rollback_id=f"rollback_{harmful_artifact.artifact_id}",
        campaign_id=authority.campaign_id,
        harmful_artifact_sha=harmful_artifact.sha256(),
        restored_artifact_sha=restored_artifact.sha256(),
        restored_waveform_hashes=dict(restored_artifact.waveform_hashes),
        receipt_sha=restored_receipt.sha256(),
        simulate_only=True,
        verified=True,
    )
