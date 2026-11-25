#!/usr/bin/env python3
"""Fourier guard for W_CH1/R_CH1 multi-tone sequence windows.

The AWG sample rate is fixed at 1 GHz and the AOD frequency grid is normally
2.5 MHz. This checker looks at generated analog samples on that exact grid and
warns when the strongest Fourier peaks no longer match the intended frequency
set for a simultaneous multi-tone window.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from system_specs import SYSTEM_SPECS

_SPEC_3Q = SYSTEM_SPECS["3qubit"]
_SPEC_4Q = SYSTEM_SPECS["4qubit"]

DEFAULT_SAMPLING_RATE = _SPEC_3Q["sampling_rate"]
DEFAULT_FREQ_STEP_MHZ = _SPEC_3Q["freq_step_mhz"]
DEFAULT_CHANNELS = ("W_CH1", "R_CH1")

FREQ_STEPS_3Q = _SPEC_3Q["freq_steps"]
GROUPS_PER_ROW_3Q = _SPEC_3Q["groups_per_row"]
GROUP_SIZE_3Q = _SPEC_3Q["group_size"]
TOTAL_OPERATIONS_3Q = _SPEC_3Q["total_operations"]
OP_DURATION_NS_3Q = _SPEC_3Q["op_duration_ns"]

FREQ_STEPS_4Q = _SPEC_4Q["freq_steps"]
GROUPS_PER_ROW_4Q = _SPEC_4Q["groups_per_row"]
GROUP_SIZE_4Q = _SPEC_4Q["group_size"]
OPS_PER_CYCLE_4Q = _SPEC_4Q["ops_per_cycle"]
OP_DURATION_NS_4Q = _SPEC_4Q["op_duration_ns"]
CYCLE_ACTIVE_NS_4Q = _SPEC_4Q["cycle_active_ns"]
CYCLE_GAP_NS_4Q = _SPEC_4Q["cycle_gap_ns"]
TOTAL_CYCLES_4Q = _SPEC_4Q["total_cycles"]


@dataclass
class FourierIssue:
    channel: str
    scope: str
    start_ns: int
    duration_ns: int
    expected_mhz: List[float]
    observed_top_mhz: List[float]
    missing_mhz: List[float]
    unexpected_mhz: List[float]
    strongest_unexpected_mhz: Optional[float]
    strongest_unexpected_ratio: float
    min_expected_ratio: float

    def format(self) -> str:
        missing = _format_freqs(self.missing_mhz) if self.missing_mhz else "none"
        unexpected = _format_freqs(self.unexpected_mhz) if self.unexpected_mhz else "none"
        strongest = (
            f"{self.strongest_unexpected_mhz:.1f} MHz"
            if self.strongest_unexpected_mhz is not None
            else "none"
        )
        return (
            f"{self.channel} {self.scope}: expected {_format_freqs(sorted(self.expected_mhz))} MHz, "
            f"observed top {_format_freqs(sorted(self.observed_top_mhz))} MHz; "
            f"missing={missing}, unexpected={unexpected}, "
            f"strongest unexpected={strongest} "
            f"({self.strongest_unexpected_ratio:.2f}x median expected)"
        )


@dataclass
class FourierReport:
    system_type: str
    checked_segments: int
    skipped_segments: int
    issues: List[FourierIssue]
    coherence_warnings: List[str]

    @property
    def has_warnings(self) -> bool:
        return bool(self.issues or self.coherence_warnings)

    @property
    def has_frequency_issues(self) -> bool:
        return bool(self.issues)

    def warning_lines(self, max_issues: int = 12) -> List[str]:
        lines = []
        for issue in self.issues[:max_issues]:
            lines.append(f"Fourier guard warning: {issue.format()}")
        remaining = len(self.issues) - max_issues
        if remaining > 0:
            lines.append(f"Fourier guard warning: {remaining} more affected windows omitted")
        return lines

    def note_lines(self) -> List[str]:
        return [f"Fourier guard note: {line}" for line in self.coherence_warnings]

    def to_dict(self) -> Dict[str, object]:
        return {
            "system_type": self.system_type,
            "checked_segments": self.checked_segments,
            "skipped_segments": self.skipped_segments,
            "coherence_warnings": list(self.coherence_warnings),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class SegmentSpec:
    channel: str
    scope: str
    source_start_ns: int
    data_start_ns: int
    duration_ns: int
    expected_mhz: List[float]
    intended_amplitudes: List[float]


_BASIS_CACHE: Dict[Tuple[int, Tuple[float, ...], float], np.ndarray] = {}


def analyze_frequency_window(
    samples: np.ndarray,
    expected_mhz: Sequence[float],
    intended_amplitudes: Optional[Sequence[float]] = None,
    *,
    channel: str = "channel",
    scope: str = "window",
    start_ns: int = 0,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    freq_step_mhz: float = DEFAULT_FREQ_STEP_MHZ,
    guard_steps: int = 2,
    missing_ratio: float = 0.35,
    unexpected_ratio: float = 0.50,
    sample_model: str = "zoh",
) -> Optional[FourierIssue]:
    """Return a FourierIssue when an expected multi-tone window looks shifted.

    Frequencies are evaluated on the exact 2.5 MHz grid rather than on the
    nearest FFT bin. This mirrors the failure mode where a neighboring grid
    component, for example 82.5 or 95 MHz, becomes stronger than an intended
    component such as 87.5 or 90 MHz.
    """

    expected = _unique_sorted(expected_mhz)
    if len(expected) < 2 or len(samples) < 8:
        return None

    segment = np.asarray(samples, dtype=float)
    if not np.any(np.isfinite(segment)):
        return None
    segment = np.nan_to_num(segment, copy=False)
    segment = segment - float(np.mean(segment))
    if float(np.max(np.abs(segment))) <= 0.0:
        return None

    intended = np.asarray(
        intended_amplitudes if intended_amplitudes is not None else np.ones(len(expected)),
        dtype=float,
    )
    if intended.size != len(expected):
        intended = np.ones(len(expected), dtype=float)
    intended = np.maximum(np.abs(intended), 1e-12)

    grid = _analysis_grid(expected, freq_step_mhz, guard_steps)
    amplitudes = _projection_amplitudes(segment, grid, sampling_rate, sample_model)
    amp_by_freq = {_freq_key(freq): float(amp) for freq, amp in zip(grid, amplitudes)}
    expected_keys = {_freq_key(freq) for freq in expected}
    expected_amps = np.asarray([amp_by_freq.get(_freq_key(freq), 0.0) for freq in expected])
    if not np.any(expected_amps > 0.0):
        return None

    normalized_expected = expected_amps / intended
    median_expected = float(np.median(expected_amps))
    median_normalized = float(np.median(normalized_expected))
    median_expected = max(median_expected, 1e-12)
    median_normalized = max(median_normalized, 1e-12)

    missing = [
        expected[idx]
        for idx, value in enumerate(normalized_expected)
        if value < missing_ratio * median_normalized
    ]

    outside = [
        (float(freq), float(amp))
        for freq, amp in zip(grid, amplitudes)
        if _freq_key(freq) not in expected_keys
    ]
    strongest_unexpected = max(outside, key=lambda item: item[1], default=(None, 0.0))
    strongest_unexpected_ratio = float(strongest_unexpected[1] / median_expected)

    top_count = len(expected)
    top_indices = np.argsort(amplitudes)[-top_count:][::-1]
    observed_top = [float(grid[idx]) for idx in top_indices]
    unexpected_top = [
        freq
        for freq in observed_top
        if _freq_key(freq) not in expected_keys
        and amp_by_freq.get(_freq_key(freq), 0.0) >= unexpected_ratio * median_expected
    ]

    if not missing and not unexpected_top:
        return None

    min_expected_ratio = float(np.min(normalized_expected) / median_normalized)
    return FourierIssue(
        channel=channel,
        scope=scope,
        start_ns=int(start_ns),
        duration_ns=int(len(samples)),
        expected_mhz=[float(freq) for freq in expected],
        observed_top_mhz=observed_top,
        missing_mhz=[float(freq) for freq in missing],
        unexpected_mhz=[float(freq) for freq in unexpected_top],
        strongest_unexpected_mhz=(
            float(strongest_unexpected[0]) if strongest_unexpected[0] is not None else None
        ),
        strongest_unexpected_ratio=strongest_unexpected_ratio,
        min_expected_ratio=min_expected_ratio,
    )


def analyze_sequence_channels(
    system_type: str,
    channels: Dict[str, np.ndarray],
    config,
    *,
    channels_to_check: Sequence[str] = DEFAULT_CHANNELS,
    sampling_rate: Optional[float] = None,
    freq_step_mhz: Optional[float] = None,
    guard_steps: int = 2,
    analysis_window_ns: Optional[int] = None,
    sample_model: str = "zoh",
    max_issues: Optional[int] = None,
) -> FourierReport:
    system_type = _normalize_system_type(system_type)
    sampling_rate = float(sampling_rate or getattr(config, "sampling_rate", DEFAULT_SAMPLING_RATE))
    freq_step_mhz = float(freq_step_mhz or getattr(config, "freq_step", DEFAULT_FREQ_STEP_MHZ))

    if system_type == "3qubit":
        segments = _segments_3q(config, channels, channels_to_check)
    else:
        segments = _segments_4q(config, channels, channels_to_check)

    issues: List[FourierIssue] = []
    checked = 0
    skipped = 0
    coherence_warnings = _coherence_warnings(segments, freq_step_mhz)

    for segment in segments:
        data = channels.get(segment.channel)
        if data is None:
            skipped += 1
            continue
        analog = _analog_column(data)
        start = int(segment.data_start_ns)
        duration = int(segment.duration_ns)
        if analysis_window_ns is not None:
            duration = min(duration, int(analysis_window_ns))
        end = int(segment.data_start_ns + duration)
        if start < 0 or end > len(analog) or end <= start:
            skipped += 1
            continue
        checked += 1
        issue = analyze_frequency_window(
            analog[start:end],
            segment.expected_mhz,
            segment.intended_amplitudes,
            channel=segment.channel,
            scope=segment.scope,
            start_ns=start,
            sampling_rate=sampling_rate,
            freq_step_mhz=freq_step_mhz,
            guard_steps=guard_steps,
            sample_model=sample_model,
        )
        if issue is not None:
            issues.append(issue)
            if max_issues is not None and len(issues) >= max_issues:
                break

    return FourierReport(
        system_type=system_type,
        checked_segments=checked,
        skipped_segments=skipped,
        issues=issues,
        coherence_warnings=coherence_warnings,
    )


def run_and_attach_fourier_guard(
    system_type: str,
    channels: Dict[str, np.ndarray],
    config,
    *,
    max_issue_lines: int = 12,
) -> FourierReport:
    try:
        report = analyze_sequence_channels(system_type, channels, config)
    except Exception as exc:
        report = FourierReport(
            system_type=str(system_type),
            checked_segments=0,
            skipped_segments=0,
            issues=[],
            coherence_warnings=[f"Fourier guard failed: {exc}"],
        )
    config.fourier_guard_report = report
    config.fourier_has_frequency_issues = report.has_frequency_issues
    config.fourier_warnings = report.warning_lines(max_issue_lines)
    config.fourier_notes = report.note_lines()
    if report.has_frequency_issues:
        for line in config.fourier_warnings:
            print(line)
    else:
        print(
            f"Fourier guard OK: checked {report.checked_segments} "
            f"{system_type} W_CH1/R_CH1 multi-tone windows"
        )
    return report


def format_fourier_report(report: FourierReport, max_issues: int = 12) -> str:
    header = (
        f"Fourier guard report ({report.system_type}): "
        f"checked={report.checked_segments}, skipped={report.skipped_segments}, "
        f"issues={len(report.issues)}"
    )
    lines = [header]
    if report.has_frequency_issues:
        lines.extend(report.warning_lines(max_issues))
    else:
        lines.append("Fourier guard OK: W_CH1/R_CH1 peaks match intended frequency sets.")
    lines.extend(report.note_lines())
    return "\n".join(lines)


def load_generated_channels(
    output_dir: str,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    prefix: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    loaded: Dict[str, np.ndarray] = {}
    for channel in channels:
        path = _find_channel_file(output_dir, channel, prefix)
        loaded[channel] = np.loadtxt(path, delimiter=",", skiprows=1)
    return loaded


def build_default_config(system_type: str, output_dir: Optional[str], excitation_mode: int):
    system_type = _normalize_system_type(system_type)
    if system_type == "3qubit":
        from entangle_sequence import EntangleConfig

        config = EntangleConfig()
    else:
        from entangle_sequence_4qubit import EntangleConfig4Qubit

        config = EntangleConfig4Qubit()
    if output_dir:
        config.output_dir = output_dir
    config.w_ch1_excitation_mode = excitation_mode
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check generated 3qubit/4qubit W_CH1 and R_CH1 waveforms on the 2.5 MHz Fourier grid."
    )
    parser.add_argument("--system", choices=["3qubit", "4qubit"], required=True)
    parser.add_argument("--output-dir", default=None, help="Directory containing generated *_T*us.txt files")
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--prefix", default=None, help="Optional file prefix such as T235us or T1034us")
    parser.add_argument("--excitation-mode", type=int, default=1)
    parser.add_argument("--freq-step", type=float, default=DEFAULT_FREQ_STEP_MHZ)
    parser.add_argument("--guard-steps", type=int, default=2)
    parser.add_argument(
        "--analysis-window-ns",
        type=int,
        default=None,
        help="Optional sub-window length. 400 ns is one coherent 2.5 MHz period at 1 GS/s.",
    )
    parser.add_argument(
        "--sample-model",
        choices=["zoh", "sampled"],
        default="zoh",
        help="zoh treats each AWG sample as constant for 1 ns; sampled uses point samples.",
    )
    parser.add_argument("--max-issues", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args(argv)

    config = build_default_config(args.system, args.output_dir, args.excitation_mode)
    output_dir = args.output_dir or config.output_dir
    channels = load_generated_channels(output_dir, args.channels, args.prefix)
    report = analyze_sequence_channels(
        args.system,
        channels,
        config,
        channels_to_check=args.channels,
        freq_step_mhz=args.freq_step,
        guard_steps=args.guard_steps,
        analysis_window_ns=args.analysis_window_ns,
        sample_model=args.sample_model,
        max_issues=args.max_issues,
    )

    print(format_fourier_report(report, args.max_issues))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)

    return 1 if args.fail_on_warning and report.has_warnings else 0


def _segments_3q(config, channels: Dict[str, np.ndarray], channels_to_check: Sequence[str]) -> List[SegmentSpec]:
    frequencies = np.linspace(float(config.freq_start), float(config.freq_end), int(config.freq_steps))
    segments: List[SegmentSpec] = []
    for op_index in range(TOTAL_OPERATIONS_3Q):
        row = op_index // GROUPS_PER_ROW_3Q
        group = op_index % GROUPS_PER_ROW_3Q
        col_start = group * GROUP_SIZE_3Q
        cols = [col_start + offset for offset in range(GROUP_SIZE_3Q)]
        op_start = op_index * OP_DURATION_NS_3Q

        if "W_CH1" in channels_to_check and "W_CH1" in channels:
            mode = int(getattr(config, "w_ch1_excitation_mode", 1))
            if mode == 1:
                _append_segment_3q(segments, config, frequencies, "W_CH1", row, cols, op_start, OP_DURATION_NS_3Q, "op")
            elif mode == 2:
                _append_segment_3q(segments, config, frequencies, "W_CH1", row, cols[:2], op_start, 2000, "op first2")
            elif mode == 3:
                for offset, col in enumerate(cols):
                    _append_segment_3q(
                        segments,
                        config,
                        frequencies,
                        "W_CH1",
                        row,
                        [col],
                        op_start + offset * 1000,
                        1000,
                        f"op part{offset + 1}",
                    )

        if "R_CH1" in channels_to_check and "R_CH1" in channels:
            _append_segment_3q(segments, config, frequencies, "R_CH1", row, cols, op_start, OP_DURATION_NS_3Q, "op")
    return segments


def _segments_4q(config, channels: Dict[str, np.ndarray], channels_to_check: Sequence[str]) -> List[SegmentSpec]:
    frequencies = np.linspace(float(config.freq_start), float(config.freq_end), int(config.freq_steps))
    segments: List[SegmentSpec] = []
    for cycle_index in range(TOTAL_CYCLES_4Q):
        cycle_start = cycle_index * (CYCLE_ACTIVE_NS_4Q + CYCLE_GAP_NS_4Q)
        for op_index in range(OPS_PER_CYCLE_4Q):
            row = op_index // GROUPS_PER_ROW_4Q
            group = op_index % GROUPS_PER_ROW_4Q
            col_start = group * GROUP_SIZE_4Q
            cols = [col_start + offset for offset in range(GROUP_SIZE_4Q)]
            op_start = cycle_start + op_index * OP_DURATION_NS_4Q

            if "W_CH1" in channels_to_check and "W_CH1" in channels:
                mode = int(getattr(config, "w_ch1_excitation_mode", 1))
                if mode == 1:
                    _append_segment_4q(
                        segments,
                        config,
                        frequencies,
                        "W_CH1",
                        cycle_index,
                        row,
                        cols,
                        op_start,
                        OP_DURATION_NS_4Q,
                        "op",
                    )
                elif mode == 2:
                    _append_segment_4q(
                        segments,
                        config,
                        frequencies,
                        "W_CH1",
                        cycle_index,
                        row,
                        cols[:3],
                        op_start,
                        3000,
                        "op first3",
                    )
                elif mode == 3:
                    _append_segment_4q(
                        segments,
                        config,
                        frequencies,
                        "W_CH1",
                        cycle_index,
                        row,
                        cols[:2],
                        op_start,
                        2000,
                        "op first2",
                    )

            if "R_CH1" in channels_to_check and "R_CH1" in channels:
                _append_segment_4q(
                    segments,
                    config,
                    frequencies,
                    "R_CH1",
                    cycle_index,
                    row,
                    cols,
                    op_start,
                    OP_DURATION_NS_4Q,
                    "op",
                )
    return segments


def _append_segment_3q(
    segments: List[SegmentSpec],
    config,
    frequencies: np.ndarray,
    channel: str,
    row: int,
    cols: Sequence[int],
    source_start: int,
    duration: int,
    label: str,
) -> None:
    amp_matrix = config.amp_matrices.get(channel)
    if amp_matrix is None:
        return
    active_cols, amplitudes = _active_columns(amp_matrix, row, cols)
    if len(active_cols) < 2:
        return
    delay = int(config.channel_analog_delays.get(channel, 0))
    group = active_cols[0] // GROUP_SIZE_3Q
    segments.append(
        SegmentSpec(
            channel=channel,
            scope=f"row={row} group={group} {label}",
            source_start_ns=int(source_start),
            data_start_ns=int(source_start + delay),
            duration_ns=int(duration),
            expected_mhz=[float(frequencies[col]) for col in active_cols],
            intended_amplitudes=amplitudes,
        )
    )


def _append_segment_4q(
    segments: List[SegmentSpec],
    config,
    frequencies: np.ndarray,
    channel: str,
    cycle_index: int,
    row: int,
    cols: Sequence[int],
    source_start: int,
    duration: int,
    label: str,
) -> None:
    if channel == "W_CH1":
        amp_matrix = config.w_ch1_cycle_amp_matrices[cycle_index]
    else:
        amp_matrix = config.cycle_amp_matrices["R_CH1"][cycle_index]
    active_cols, amplitudes = _active_columns(amp_matrix, row, cols)
    if len(active_cols) < 2:
        return
    delay = int(config.channel_analog_delays.get(channel, 0))
    group = active_cols[0] // GROUP_SIZE_4Q
    segments.append(
        SegmentSpec(
            channel=channel,
            scope=f"cycle={cycle_index + 1} row={row} group={group} {label}",
            source_start_ns=int(source_start),
            data_start_ns=int(source_start + delay),
            duration_ns=int(duration),
            expected_mhz=[float(frequencies[col]) for col in active_cols],
            intended_amplitudes=amplitudes,
        )
    )


def _active_columns(amp_matrix: np.ndarray, row: int, cols: Sequence[int], floor: float = 1e-9) -> Tuple[List[int], List[float]]:
    active_cols: List[int] = []
    amplitudes: List[float] = []
    for col in cols:
        amp = float(amp_matrix[row, col])
        if abs(amp) > floor:
            active_cols.append(int(col))
            amplitudes.append(abs(amp))
    return active_cols, amplitudes


def _coherence_warnings(segments: Sequence[SegmentSpec], freq_step_mhz: float) -> List[str]:
    beat_period_ns = 1000.0 / float(freq_step_mhz)
    seen = set()
    warnings: List[str] = []
    for segment in segments:
        periods = segment.duration_ns / beat_period_ns
        if abs(periods - round(periods)) <= 1e-9:
            continue
        key = (segment.channel, segment.duration_ns, len(segment.expected_mhz))
        if key in seen:
            continue
        seen.add(key)
        warnings.append(
            f"{segment.channel} has {segment.duration_ns} ns windows with "
            f"{len(segment.expected_mhz)} active tones "
            f"({periods:.2f} periods of {freq_step_mhz:.1f} MHz spacing); "
            "Fourier amplitudes on the 2.5 MHz grid can become phase-sensitive."
        )
    return warnings


def _projection_amplitudes(
    samples: np.ndarray,
    grid_mhz: np.ndarray,
    sampling_rate: float,
    sample_model: str = "zoh",
) -> np.ndarray:
    n = len(samples)
    key = (n, tuple(float(f"{freq:.9f}") for freq in grid_mhz), float(sampling_rate))
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        sample_index = np.arange(n, dtype=float)
        basis = np.exp(
            -1j
            * 2.0
            * np.pi
            * (np.asarray(grid_mhz, dtype=float)[:, None] * 1e6)
            * sample_index[None, :]
            / float(sampling_rate)
        )
        _BASIS_CACHE[key] = basis
    amplitudes = (2.0 / max(n, 1)) * np.abs(basis @ samples)
    if sample_model == "zoh":
        hold_response = np.abs(np.sinc((np.asarray(grid_mhz, dtype=float) * 1e6) / float(sampling_rate)))
        return amplitudes * hold_response
    if sample_model == "sampled":
        return amplitudes
    raise ValueError("sample_model must be 'zoh' or 'sampled'")


def _analysis_grid(expected_mhz: Sequence[float], freq_step_mhz: float, guard_steps: int) -> np.ndarray:
    min_index = math.floor(min(expected_mhz) / freq_step_mhz) - guard_steps
    max_index = math.ceil(max(expected_mhz) / freq_step_mhz) + guard_steps
    return np.arange(min_index, max_index + 1, dtype=float) * freq_step_mhz


def _analog_column(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        return data.astype(float)
    return data[:, 0].astype(float)


def _find_channel_file(output_dir: str, channel: str, prefix: Optional[str]) -> str:
    pattern = f"{channel}_{prefix}.txt" if prefix else f"{channel}_T*us.txt"
    matches = glob.glob(os.path.join(output_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No generated file found for {channel} in {output_dir!r}")
    return max(matches, key=os.path.getmtime)


def _normalize_system_type(system_type: str) -> str:
    value = system_type.lower().replace("_", "")
    if value in {"3q", "3qubit", "3qubits"}:
        return "3qubit"
    if value in {"4q", "4qubit", "4qubits"}:
        return "4qubit"
    raise ValueError("system_type must be '3qubit' or '4qubit'")


def _unique_sorted(freqs: Iterable[float]) -> List[float]:
    return sorted({_freq_key(freq) for freq in freqs})


def _freq_key(freq: float) -> float:
    return round(float(freq), 6)


def _format_freqs(freqs: Sequence[float]) -> str:
    return "/".join(f"{float(freq):.1f}" for freq in freqs)


if __name__ == "__main__":
    raise SystemExit(main())
