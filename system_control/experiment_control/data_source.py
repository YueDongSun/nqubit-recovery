"""Reusable measurement aggregation for serial or mock data sources."""

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import numpy as np

from .models import MeasurementSnapshot


class MeasurementAccumulator:
    """Aggregate repeated data_ui frames into one calibration window."""

    def __init__(self, system_type: str, window_s: float):
        if system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        self.system_type = system_type
        self.window_s = float(window_s)
        self.num_cells = 75 if system_type == "3qubit" else 256
        self.reset()

    def reset(self) -> None:
        self.started_at = time.time()
        self.last_timestamp = self.started_at
        self.signal_bins = np.zeros(self.num_cells, dtype=np.float64)
        self.idler_bins = np.zeros(self.num_cells, dtype=np.float64)
        self.coin_bins = np.zeros(self.num_cells, dtype=np.float64)
        self.segment_values = np.zeros(self.num_cells, dtype=np.float64)
        self.last_raw_payload: Dict = {}
        self.multi_click_counts = {"multi2": None, "multi3": None, "multi4": None}
        self.multi2_pair_counts: Dict[str, int] = {}
        self.read_count = 0
        self.write_count = 0
        self.counter_diagnostics = {}
        self.frames = 0

    def add_frame(self, frame: Dict) -> None:
        cumulative = frame.get("counter_semantics") == "cumulative_since_start"
        signal = self._array(frame.get("display_signal_bins", frame.get("signal_bins")))
        idler = self._array(frame.get("display_idler_bins", frame.get("idler_bins")))
        coin = self._array(frame.get("display_coin_bins", frame.get("coin_bins")))
        segment = self._array(frame.get("segment_values"))
        if cumulative:
            # Hardware frames are repeated views of monotonically increasing
            # FPGA counters. Summing them would integrate the same counts many
            # times. The display_* fields are already Start-baseline corrected.
            self.signal_bins = signal
            self.idler_bins = idler
            self.coin_bins = coin
            self.segment_values = segment
        else:
            # Mock and explicitly incremental sources keep the historical
            # additive aggregation contract.
            self.signal_bins += signal
            self.idler_bins += idler
            self.coin_bins += coin
            self.segment_values += segment
        self.last_raw_payload = {
            key: frame[key] for key in (
                "signal_bins_raw", "idler_bins_raw", "coin_bins_raw",
                "signal_hist", "idler_hist", "signal_hist_raw", "idler_hist_raw",
                "fpga_protocol_version", "counter_semantics",
                "signal_total", "idler_total", "coin_total",
            ) if key in frame
        }
        for key in ("multi2", "multi3", "multi4"):
            value = frame.get(f"{key}_count")
            if value is not None:
                self.multi_click_counts[key] = max(int(value), self.multi_click_counts[key] or 0)
        if frame.get("multi2_pairs"):
            self.multi2_pair_counts = {str(key): int(value) for key, value in frame["multi2_pairs"].items()}
        self.read_count = max(
            self.read_count,
            int(frame.get("display_read_count", frame.get("read_count", 0))),
        )
        self.write_count = max(
            self.write_count,
            int(frame.get("display_write_count", frame.get("write_count", 0))),
        )
        self.counter_diagnostics = {
            key: frame[key]
            for key in (
                "read_count_raw",
                "write_count_raw",
                "counter_decode_layout",
                "counter_decode_warning",
            )
            if key in frame
        }
        self.last_timestamp = float(frame.get("timestamp", time.time()))
        self.frames += 1

    def ready(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return self.frames > 0 and (now - self.started_at) >= self.window_s

    def snapshot(self) -> MeasurementSnapshot:
        duration = max(0.0, self.last_timestamp - self.started_at)
        cumulative = self.last_raw_payload.get("counter_semantics") == "cumulative_since_start"
        segment = self.segment_values.copy() if cumulative else self.segment_values / max(self.frames, 1)
        coin_sum = float(np.sum(self.coin_bins))
        signal_sum = float(np.sum(self.signal_bins))
        retrieval_rate = coin_sum / max(signal_sum, 1.0)
        coin_idler_values = np.zeros(self.num_cells, dtype=np.float64)
        np.divide(
            self.coin_bins,
            self.idler_bins,
            out=coin_idler_values,
            where=self.idler_bins > 0.0,
        )
        return MeasurementSnapshot(
            system_type=self.system_type,
            signal_bins=self.signal_bins.copy(),
            idler_bins=self.idler_bins.copy(),
            coin_bins=self.coin_bins.copy(),
            segment_values=segment,
            read_count=self.read_count,
            write_count=self.write_count,
            timestamp=self.last_timestamp,
            duration_s=duration,
            retrieval_rate=retrieval_rate,
            signal_bins_raw=_optional_array(self.last_raw_payload.get("signal_bins_raw")),
            idler_bins_raw=_optional_array(self.last_raw_payload.get("idler_bins_raw")),
            coin_bins_raw=_optional_array(self.last_raw_payload.get("coin_bins_raw")),
            signal_bins_corrected=self.signal_bins.copy(),
            idler_bins_corrected=self.idler_bins.copy(),
            coin_bins_corrected=self.coin_bins.copy(),
            signed_segment_values=segment.copy(),
            legacy_clipped_segment_values=np.maximum(segment, 0.0),
            coin_idler_values=coin_idler_values,
            signal_timing_histograms=_optional_array(self.last_raw_payload.get("signal_hist")),
            idler_timing_histograms=_optional_array(self.last_raw_payload.get("idler_hist")),
            multi2_count=self.multi_click_counts["multi2"],
            multi3_count=self.multi_click_counts["multi3"],
            multi4_count=self.multi_click_counts["multi4"],
            multi2_pair_counts=dict(self.multi2_pair_counts),
            signal_total=_optional_int(self.last_raw_payload.get("signal_total"), int(signal_sum)),
            idler_total=_optional_int(self.last_raw_payload.get("idler_total"), int(np.sum(self.idler_bins))),
            coin_total=_optional_int(self.last_raw_payload.get("coin_total"), int(coin_sum)),
            fpga_protocol_version=self.last_raw_payload.get("fpga_protocol_version", "legacy_counter_frame_v1"),
            feature_schema_version=2,
            missing_value_masks={
                "multi2_pairs": not bool(self.multi2_pair_counts),
                "multi3_count": self.multi_click_counts["multi3"] is None,
                "multi4_count": self.multi_click_counts["multi4"] is None,
                "timing_histograms": "signal_hist" not in self.last_raw_payload,
            },
            diagnostics={
                "frames": self.frames, **self.counter_diagnostics,
                "counter_semantics": self.last_raw_payload.get(
                    "counter_semantics", "incremental_frames"
                ),
                **{f"{key}_count": value for key, value in self.multi_click_counts.items() if value is not None},
                "multi2_pairs": dict(self.multi2_pair_counts),
            },
        )

    def _array(self, values: Optional[Iterable]) -> np.ndarray:
        if values is None:
            return np.zeros(self.num_cells, dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < self.num_cells:
            padded = np.zeros(self.num_cells, dtype=np.float64)
            padded[:arr.size] = arr
            return padded
        return arr[:self.num_cells]


def _optional_array(value):
    return None if value is None else np.asarray(value).copy()


def _optional_int(value, fallback: int) -> int:
    return fallback if value is None else int(value)


class MockMeasurementSource:
    """Deterministic-enough mock source for dry-run UI and tests."""

    def __init__(self, system_type: str, seed: int = 1234):
        self.system_type = system_type
        self.num_cells = 75 if system_type == "3qubit" else 256
        self.rng = np.random.default_rng(seed)
        self.read_count = 0
        self.write_count = 0

    def read_frame(self) -> Dict:
        n = self.num_cells
        base = 120.0 if self.system_type == "3qubit" else 180.0
        envelope = 1.0 + 0.08 * np.sin(np.linspace(0.0, np.pi, n))
        signal = self.rng.poisson(base * envelope).astype(np.int64)
        idler = self.rng.poisson(base * 0.45, size=n).astype(np.int64)
        coin = self.rng.poisson(np.maximum(signal * 0.08, 1.0)).astype(np.int64)
        read_inc = int(self.rng.integers(500, 900))
        write_inc = int(self.rng.integers(1, 5))
        self.read_count += read_inc
        self.write_count += write_inc
        segment = (coin / np.maximum(signal, 1)) - (idler / max(self.read_count, 1))
        multi2 = int(self.rng.integers(10, 50))
        multi3 = int(self.rng.integers(1, 10))
        multi4 = int(self.rng.integers(0, 3))
        multi2_pairs = {
            "12": int(self.rng.integers(2, 15)),
            "13": int(self.rng.integers(2, 15)),
            "14": int(self.rng.integers(2, 15)),
            "23": int(self.rng.integers(2, 15)),
            "24": int(self.rng.integers(2, 15)),
            "34": int(self.rng.integers(2, 15)),
        }
        return {
            "signal_bins": signal,
            "idler_bins": idler,
            "coin_bins": coin,
            "segment_values": segment,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "multi2_count": multi2,
            "multi3_count": multi3,
            "multi4_count": multi4,
            "multi2_pairs": multi2_pairs,
            "multi_click": multi2 + multi3 + multi4,
            "signal_interval": int(np.sum(signal)),
            "idler_interval": int(np.sum(idler)),
            "coin_interval": int(np.sum(coin)),
            "timestamp": time.time(),
        }
