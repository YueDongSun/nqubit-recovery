"""Serial frame protocol helpers for the FPGA data UIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterDecode:
    read_count: int
    write_count: int
    raw_read_count: int
    raw_write_count: int
    layout: str
    warning: str = ""


@dataclass(frozen=True)
class FrameLayout:
    """Byte layout for one FPGA counter frame."""

    system_type: str
    frame_bytes: int
    num_cells: int
    counter_decoder: str
    pair_offset: int
    read_offset: int
    read_size: int
    write_offset: int
    write_size: int
    signal_total_offset: int
    idler_total_offset: int
    coin_total_offset: int
    signal_base: int
    signal_stride: int
    idler_base: int
    idler_stride: int
    coin_base: int
    coin_stride: int
    signal_cell_base: int
    signal_cell_stride: int
    idler_cell_base: int
    idler_cell_stride: int
    scalar_size: int = 4
    cell_hist_bins: int = 10

    @property
    def multi2_offset(self) -> int:
        return 0

    @property
    def multi3_offset(self) -> int:
        return 4

    @property
    def multi4_offset(self) -> int:
        return 8

    @property
    def pair_count_bytes(self) -> int:
        return len(MULTI2_PAIR_NAMES) * 4

    @property
    def pair_end(self) -> int:
        return self.pair_offset + self.pair_count_bytes

    @property
    def final_offset(self) -> int:
        return self.idler_cell_base + self.cell_hist_bins * self.idler_cell_stride


MULTI2_PAIR_NAMES = ("12", "13", "14", "23", "24", "34")

# Prepared-only W4 protocol extension.  It is intentionally not assigned to
# LAYOUT_4Q.frame_bytes, so deployed firmware and acquisition timing remain
# untouched until an approved protocol-version rollout.
FOUR_Q_POINT_COUNTER_EXTENSION_VERSION = "4q_point_counters_v2"
FOUR_Q_POINT_COUNTER_EXTENSION_BYTES = 4 * (3 + len(MULTI2_PAIR_NAMES))


LAYOUT_3Q = FrameLayout(
    system_type="3qubit",
    frame_bytes=1044,
    num_cells=75,
    counter_decoder="decode_3qubit_counter_words",
    pair_offset=12,
    read_offset=36,
    read_size=8,
    write_offset=44,
    write_size=8,
    signal_total_offset=52,
    idler_total_offset=56,
    coin_total_offset=60,
    signal_base=64,
    signal_stride=4,
    idler_base=364,
    idler_stride=4,
    coin_base=664,
    coin_stride=4,
    signal_cell_base=964,
    signal_cell_stride=4,
    idler_cell_base=1004,
    idler_cell_stride=4,
)


LAYOUT_4Q = FrameLayout(
    system_type="4qubit",
    frame_bytes=1358,
    num_cells=256,
    counter_decoder="decode_4qubit_counter_words",
    pair_offset=12,
    read_offset=36,
    read_size=5,
    write_offset=41,
    write_size=5,
    signal_total_offset=46,
    idler_total_offset=50,
    coin_total_offset=54,
    signal_base=58,
    signal_stride=2,
    idler_base=570,
    idler_stride=2,
    coin_base=1082,
    coin_stride=1,
    signal_cell_base=1338,
    signal_cell_stride=1,
    idler_cell_base=1348,
    idler_cell_stride=1,
)


def decode_multi2_pair_counts(raw_frame: bytes, offset: int = 12) -> dict[str, int]:
    """Decode six 32-bit 2-click pair counters from a serial frame."""

    required = offset + len(MULTI2_PAIR_NAMES) * 4
    if len(raw_frame) < required:
        raise ValueError(f"frame too short for multi2 pair counters: need {required} bytes")
    return {
        name: int.from_bytes(raw_frame[offset + index * 4:offset + (index + 1) * 4], "big")
        for index, name in enumerate(MULTI2_PAIR_NAMES)
    }


def decode_multi_click_counts(raw_frame: bytes, layout: FrameLayout) -> dict[str, int]:
    """Decode the leading 2/3/4-click counters for either FPGA frame.

    Keeping this in the protocol module prevents the 3Q and 4Q UIs from
    drifting to different offsets or byte order.  The deployed firmware sends
    three consecutive unsigned 32-bit big-endian counters before the pair
    counters.
    """

    required = layout.multi4_offset + 4
    if len(raw_frame) < required:
        raise ValueError(f"frame too short for multi-click counters: need {required} bytes")
    multi2 = int.from_bytes(raw_frame[layout.multi2_offset:layout.multi2_offset + 4], "big")
    multi3 = int.from_bytes(raw_frame[layout.multi3_offset:layout.multi3_offset + 4], "big")
    multi4 = int.from_bytes(raw_frame[layout.multi4_offset:layout.multi4_offset + 4], "big")
    return {
        "multi2_count": multi2,
        "multi3_count": multi3,
        "multi4_count": multi4,
        "multi_click": multi2 + multi3 + multi4,
    }


def decode_4qubit_cycle_counter_extension(raw_extension: bytes) -> dict[str, object]:
    """Decode one optional appended 4Q per-cycle counter block.

    Frame acquisition does not request this block yet.  When explicitly
    enabled with firmware approval, append four blocks after the legacy frame
    and call this decoder per cycle; global legacy counters remain authoritative
    compatibility fields.
    """
    if len(raw_extension) < FOUR_Q_POINT_COUNTER_EXTENSION_BYTES:
        raise ValueError(f"4Q v2 extension needs {FOUR_Q_POINT_COUNTER_EXTENSION_BYTES} bytes")
    names = ("multi2", "multi3", "multi4", *MULTI2_PAIR_NAMES)
    values = {
        name: int.from_bytes(raw_extension[index * 4:(index + 1) * 4], "big")
        for index, name in enumerate(names)
    }
    return {
        "protocol_version": FOUR_Q_POINT_COUNTER_EXTENSION_VERSION,
        "multi2_count": values.pop("multi2"), "multi3_count": values.pop("multi3"), "multi4_count": values.pop("multi4"),
        "multi2_pairs": values,
    }


def decode_3qubit_counter_words(raw_read_count: int, raw_write_count: int) -> CounterDecode:
    """Decode 3Q read/write counters, including the observed shifted layout.

    The intended 3Q protocol has two big-endian 64-bit counters at offsets 12
    and 20. Real hardware observed on 2026-06-25 emitted the effective 32-bit
    counter value in the upper bytes instead:

      read  = count << 32
      write = count << 40 plus bytes from following scalar fields

    That produces the characteristic false ratio ``write/read == 256``. When
    that fingerprint is present, report the corrected counter values while
    preserving the raw words for diagnostics.
    """

    raw_read = int(raw_read_count)
    raw_write = int(raw_write_count)
    if raw_read <= 0:
        return CounterDecode(0, 0 if raw_write <= 0 else raw_write, raw_read, raw_write, "normal_u64_be")

    ratio = raw_write / max(raw_read, 1)
    shifted_read = raw_read >> 32
    shifted_write = raw_write >> 40
    shifted_counts_plausible = (
        shifted_read > 0
        and shifted_write > 0
        and abs(shifted_write - shifted_read) <= max(4, int(max(shifted_read, shifted_write) * 0.10))
    )
    shifted_fingerprint = (
        (raw_read & 0xFFFFFFFF) == 0
        and (raw_write & 0xFFFF) == 0
        and 128.0 <= ratio <= 512.0
        and shifted_counts_plausible
    )
    if shifted_fingerprint:
        return CounterDecode(
            read_count=shifted_read,
            write_count=shifted_write,
            raw_read_count=raw_read,
            raw_write_count=raw_write,
            layout="shifted_counter_words",
            warning=(
                "Detected shifted 3Q FPGA counter words; corrected read_count=raw_read>>32 "
                "and write_count=raw_write>>40."
            ),
        )
    return CounterDecode(raw_read, raw_write, raw_read, raw_write, "normal_u64_be")


def decode_4qubit_counter_words(raw_read_count: int, raw_write_count: int) -> CounterDecode:
    """Decode 4Q 40-bit counters, including the observed low-byte prefix.

    The intended 4Q protocol has two big-endian 40-bit counters at offsets 12
    and 17. Hardware observed on 2026-06-26 emitted the counter low byte once
    at the start of each 5-byte word and again at the end, e.g. count 10 was:

      0a 00 00 00 0a

    The lower 32 bits still contain the usable counter for normal experiment
    durations. Preserve the raw word and report the corrected value when this
    fingerprint is present.
    """

    raw_read = int(raw_read_count)
    raw_write = int(raw_write_count)

    def _decode_word(raw: int) -> tuple[int, bool]:
        if raw <= 0:
            return 0 if raw <= 0 else raw, False
        top_byte = (raw >> 32) & 0xFF
        low_byte = raw & 0xFF
        corrected = raw & 0xFFFFFFFF
        if raw >= (1 << 32) and corrected > 0 and top_byte == low_byte:
            return corrected, True
        return raw, False

    read_count, read_fixed = _decode_word(raw_read)
    write_count, write_fixed = _decode_word(raw_write)
    if read_fixed or write_fixed:
        return CounterDecode(
            read_count=read_count,
            write_count=write_count,
            raw_read_count=raw_read,
            raw_write_count=raw_write,
            layout="prefixed_low_byte_40bit",
            warning=(
                "Detected 4Q FPGA counter words with duplicated low byte prefix; "
                "corrected counters from the lower 32 bits."
            ),
        )
    return CounterDecode(raw_read, raw_write, raw_read, raw_write, "normal_u40_be")
