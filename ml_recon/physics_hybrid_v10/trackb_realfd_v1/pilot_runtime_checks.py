"""Fail-closed checks for the bounded Track B pilot runtime.

The source identity is deliberately derived from this module's repository,
never from the caller's current working directory.  Optimizer updates are
similarly conservative: every present gradient is checked before clipping or
publishing a step, and parameters are checked again after the single update.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import torch

from .contracts import require_git_oid


# Keep the repository anchor and optimizer safety policy visible and
# centralized.  ``parents[3]`` is the Track B decoder worktree containing this
# module (trackb_realfd_v1 -> physics_hybrid_v10 -> ml_recon -> repository).
PILOT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MAX_GRADIENT_NORM = 1.0


def _run_git(*arguments: str) -> str:
    """Run a read-only Git identity/status command in the pilot repository."""

    completed = subprocess.run(
        ["git", "-C", str(PILOT_REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def pilot_source_identity() -> tuple[str, str]:
    """Return ``(commit_sha, tree_sha)`` for a strictly clean pilot source.

    Both tracked modifications and every untracked path are rejected.  This
    prevents a runtime receipt from claiming an identity that omits local
    source changes.  Generated output must therefore be written outside the
    repository (or the source must be committed before identity capture).
    """

    status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "pilot source repository is not clean; refusing to capture identity: "
            + status
        )
    commit_sha = _run_git("rev-parse", "HEAD")
    require_git_oid(commit_sha, "pilot_source_commit")
    tree_sha = _run_git("rev-parse", f"{commit_sha}^{{tree}}")
    require_git_oid(tree_sha, "pilot_source_tree")
    return commit_sha, tree_sha


def require_unchanged_pilot_source(expected: tuple[str, str]) -> None:
    """Require the current clean source identity to equal ``expected``."""

    if not isinstance(expected, tuple) or len(expected) != 2:
        raise TypeError("expected pilot identity must be a (commit_sha, tree_sha) tuple")
    actual = pilot_source_identity()
    if actual != expected:
        raise RuntimeError(
            "pilot source identity changed: "
            f"expected {expected!r}, observed {actual!r}"
        )


def checked_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Validate gradients, perform one clipped optimizer step, and return its norm.

    ``None`` gradients are intentionally preserved for unused parameters.  Any
    nonfinite present gradient or norm aborts before ``optimizer.step``.  A
    nonfinite parameter after the step raises after the optimizer has run; the
    caller must treat that result as unpublished and discard the candidate.
    """

    parameters = tuple(model.parameters())
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite gradient detected before optimizer step")

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=MAX_GRADIENT_NORM,
        error_if_nonfinite=True,
    )
    gradient_norm_value = float(gradient_norm.detach().cpu())
    if not math.isfinite(gradient_norm_value):
        raise FloatingPointError("nonfinite gradient norm")

    optimizer.step()

    for parameter in parameters:
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError("nonfinite parameter after optimizer step")
    return gradient_norm_value


__all__ = [
    "MAX_GRADIENT_NORM",
    "PILOT_REPOSITORY_ROOT",
    "checked_optimizer_step",
    "pilot_source_identity",
    "require_unchanged_pilot_source",
]
