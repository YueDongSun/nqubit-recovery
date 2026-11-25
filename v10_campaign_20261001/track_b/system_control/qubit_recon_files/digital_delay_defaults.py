"""Shared digital-delay defaults for entangle sequence generation."""

W_CH1_DIGITAL_DELAY_BASE_NS = 9420
R_CH1_DIGITAL_DELAY_BASE_NS = 10050
DIGITAL_CHANNEL_STEP_NS = 50
DEFAULT_DIGITAL_DELAY_OFFSET_NS = 1500
MIN_DIGITAL_DELAY_OFFSET_NS = -W_CH1_DIGITAL_DELAY_BASE_NS
MAX_DIGITAL_DELAY_NS = 100000
MAX_DIGITAL_DELAY_OFFSET_NS = MAX_DIGITAL_DELAY_NS - (
    R_CH1_DIGITAL_DELAY_BASE_NS + 3 * DIGITAL_CHANNEL_STEP_NS
)


def digital_delay_values(offset_ns=DEFAULT_DIGITAL_DELAY_OFFSET_NS):
    """Return all channel Digital1 delays after applying a shared offset."""
    offset = int(offset_ns)
    return channel_digital_delay_values(
        W_CH1_DIGITAL_DELAY_BASE_NS + offset,
        R_CH1_DIGITAL_DELAY_BASE_NS + offset,
    )


def channel_digital_delay_values(w_ch1_delay_ns, r_ch1_delay_ns):
    """Return W/R CH1..CH4 delays with 50 ns spacing inside each group."""
    w_base = int(w_ch1_delay_ns)
    r_base = int(r_ch1_delay_ns)
    delays = {}
    for index in range(4):
        suffix = index + 1
        step = index * DIGITAL_CHANNEL_STEP_NS
        delays[f"W_CH{suffix}"] = w_base + step
        delays[f"R_CH{suffix}"] = r_base + step
    if min(delays.values()) < 0:
        raise ValueError("digital delay offset makes a channel delay negative")
    if max(delays.values()) > MAX_DIGITAL_DELAY_NS:
        raise ValueError("digital delay offset makes a channel delay exceed 100000 ns")
    return delays
