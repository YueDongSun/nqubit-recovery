"""Hardware-bound validation for generated AWG sequence directories."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def validate_sequence_dir_for_hardware(
    sequence_dir: str | Path,
    system_type: str,
    *,
    excitation_mode: int = 1,
    strict: bool = False,
):
    """Run upload-bound safety checks for generated W/R sequence files.

    Strict mode treats missing or Fourier-shifted W_CH1/R_CH1 files as a hard
    stop before any AWG front-panel command is issued. Coherence notes are
    returned to the caller for logging, but only actual frequency issues block
    upload.
    """

    if not strict:
        return None

    path = Path(sequence_dir)
    if not path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {path}")

    recon_dir = Path(__file__).resolve().parents[1] / "qubit_recon_files"
    if str(recon_dir) not in sys.path:
        sys.path.insert(0, str(recon_dir))

    try:
        from fourier_sequence_guard import (
            analyze_sequence_channels,
            build_default_config,
            format_fourier_report,
            load_generated_channels,
        )

        config = build_default_config(system_type, str(path), int(excitation_mode or 1))
        channels = load_generated_channels(str(path), ("W_CH1", "R_CH1"))
        report = analyze_sequence_channels(
            system_type,
            channels,
            config,
            channels_to_check=("W_CH1", "R_CH1"),
            max_issues=20,
        )
    except Exception as exc:
        raise RuntimeError(
            "Strict hardware mode could not validate generated W_CH1/R_CH1 "
            f"waveforms in {path}: {exc}"
        ) from exc

    if report.has_frequency_issues:
        raise RuntimeError(
            "Strict hardware mode blocked AWG upload.\n"
            + format_fourier_report(report, max_issues=8)
        )
    return report


def hardware_validation_summary(report: Optional[object]) -> str:
    """Return a short log line for a successful upload-bound validation."""

    if report is None:
        return "Strict hardware mode: validation skipped."
    checked = getattr(report, "checked_segments", 0)
    skipped = getattr(report, "skipped_segments", 0)
    issues = len(getattr(report, "issues", []) or [])
    return (
        "Strict hardware Fourier guard OK: "
        f"checked={checked}, skipped={skipped}, frequency_issues={issues}"
    )
