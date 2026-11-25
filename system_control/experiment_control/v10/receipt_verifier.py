"""Hash verifier for simulated or externally supplied V10 upload receipts."""

from __future__ import annotations

from typing import Dict

from .schemas import V10MaterializedArtifact, V10UploadReceipt


def verify_receipt(artifact: V10MaterializedArtifact,
                   receipt: V10UploadReceipt) -> Dict[str, object]:
    reasons = []
    if receipt.artifact_sha != artifact.sha256():
        reasons.append("artifact_hash_mismatch")
    if receipt.campaign_id != artifact.campaign_id:
        reasons.append("campaign_id_mismatch")
    if receipt.simulate_only != artifact.simulate_only:
        reasons.append("simulation_scope_mismatch")
    if receipt.uploaded_waveform_hashes != artifact.waveform_hashes:
        reasons.append("waveform_hash_mismatch")
    if not receipt.verified:
        reasons.append("receipt_not_verified")
    return {"passed": not reasons, "reasons": tuple(reasons),
            "upload_authority_inferred": False}
