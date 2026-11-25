"""Fail-closed V10 system-control bridge.

This package owns campaign authority. The research model package never imports it
and never receives upload, FPGA, or matrix-write authority.
"""

from .schemas import (
    V10ActionProposal,
    V10CampaignAuthority,
    V10MaterializedArtifact,
    V10ModelPackage,
    V10UploadReceipt,
)

__all__ = [
    "V10ActionProposal",
    "V10CampaignAuthority",
    "V10MaterializedArtifact",
    "V10ModelPackage",
    "V10UploadReceipt",
]
