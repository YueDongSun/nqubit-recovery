"""Model path helpers shared by the experiment UI and runners."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


LEGACY_FEEDBACK_MODEL_PATH = "ml_recon/models/feedback_model.pt"


def default_feedback_model_path(system_type: str, repo_root: Optional[Path] = None) -> Path:
    """Return the system-specific JSON feedback model path used by current training."""

    if system_type not in {"3qubit", "4qubit"}:
        raise ValueError("system_type must be '3qubit' or '4qubit'")
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "ml_recon" / "models" / f"feedback_model_{system_type}.json"


def resolve_feedback_model_path(
    system_type: str,
    configured_path: Optional[str],
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve a feedback model path, replacing the historical .pt default."""

    root = repo_root or Path(__file__).resolve().parents[2]
    text = str(configured_path or "").strip()
    if not text or _normalise_path_text(text) == _normalise_path_text(LEGACY_FEEDBACK_MODEL_PATH):
        return default_feedback_model_path(system_type, root)
    path = Path(text)
    return path if path.is_absolute() else root / path


def display_feedback_model_path(system_type: str, configured_path: Optional[str] = None) -> str:
    """Return a repo-relative path suitable for editable UI fields."""

    path = resolve_feedback_model_path(system_type, configured_path)
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _normalise_path_text(path_text: str) -> str:
    return str(Path(path_text.replace("\\", "/"))).replace("\\", "/").lower()


# ---------- classifier model paths ---------------------------------------


def default_classifier_model_path(system_type: str, repo_root: Optional[Path] = None) -> Path:
    """System-specific path for the pretrained state-family classifier."""

    if system_type not in {"3qubit", "4qubit"}:
        raise ValueError("system_type must be '3qubit' or '4qubit'")
    suffix = "3q" if system_type == "3qubit" else "4q"
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "ml_recon" / "models" / f"state_family_classifier_{suffix}.pt"


def resolve_classifier_model_path(
    system_type: str,
    configured_path: Optional[str],
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve a classifier model path, defaulting to the system-specific .pt."""

    root = repo_root or Path(__file__).resolve().parents[2]
    text = str(configured_path or "").strip()
    if not text:
        return default_classifier_model_path(system_type, root)
    path = Path(text)
    return path if path.is_absolute() else root / path


def display_classifier_model_path(system_type: str, configured_path: Optional[str] = None) -> str:
    """Repo-relative classifier model path for UI fields."""

    path = resolve_classifier_model_path(system_type, configured_path)
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
