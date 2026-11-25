import os

# Force the non-GUI Agg backend BEFORE importing pyplot.
# This script is called from a QThread worker in the experiment UI;
# matplotlib's Qt/Tk backends require the main thread and would warn
# (or crash) if initialized here. Agg only writes PNGs via savefig()
# and makes plt.show() a no-op.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import Optional, Dict


def _maybe_show_plot():
    """Pop the current figure only when explicitly opted in.

    The main experiment UI drives this script on a worker thread; the
    default behaviour is to save PNG only. Set the environment variable
    AWG_SHOW_PLOTS=1 (true/yes/on also accepted) to pop interactive
    windows when running standalone on the main thread.
    """
    flag = os.environ.get("AWG_SHOW_PLOTS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        plt.show()


class AODAWGConfig:
    """AOD AWG configuration class"""

    # attribute annotations to satisfy static type checkers
        
    digital_width_read: int
    digital_width2_write: int
    digital_width2_read: int
    digital1_repeat: int
    sampling_rate: float
    total_duration: Optional[int]
    channel_digital_delays: Dict[str, int]
    channel_digital2_delays: Dict[str, int]
    channel_digital_widths: Dict[str, int]
    channel_digital2_widths: Dict[str, int]
    channel_analog_delays: Dict[str, int]
    # Optional user-provided physical delays (absolute hardware arrival times) per channel.
    # If provided, generator will compute channel_analog_delays = reference - physical_delay
    # where reference = max(physical_delays.values()). This makes the channel(s) with the
    # largest physical delay start at sequence time 0, and other channels are offset so
    # their effective physical timing matches the provided values.
    channel_physical_delays: Optional[Dict[str, int]]
    correction_matrix: Optional[np.ndarray]
    phase_vector: Optional[np.ndarray]
    output_dir: str
    analog_active_until: Optional[int]

    def __init__(self):
        # Public parameters
        self.freq_start = 85          # MHz
        self.freq_end = 122.5         # MHz
        self.freq_steps = 16            # steps
        self.point_duration = 1000      # ns
        self.digital_delay = 0       # ns
        # Second digital output defaults (by default mirror the first digital timing)
        self.digital_delay2 = 0         # ns (digital2)
        self.digital_width = 170        # ns
        # Per-type widths: default write/read widths (can be tuned independently)
        self.digital_width2 = 0       # ns (digital2)
        self.digital_width_write =130
        self.digital_width_read = 260
        self.digital_width2_write = 0
        self.digital_width2_read = 0
        # Per-channel width maps: W channels get write width, R channels get read width
        self.channel_digital_widths = {
            'W_CH1': self.digital_width_write,
            'W_CH2': self.digital_width_write,
            'W_CH3': self.digital_width_write,
            'W_CH4': self.digital_width_write,
            'R_CH1': self.digital_width_read,
            'R_CH2': self.digital_width_read,
            'R_CH3': self.digital_width_read,
            'R_CH4': self.digital_width_read,
        }
        self.channel_digital2_widths = {
            'W_CH1': self.digital_width2_write,
            'W_CH2': self.digital_width2_write,
            'W_CH3': self.digital_width2_write,
            'W_CH4': self.digital_width2_write,
            'R_CH1': self.digital_width2_read,
            'R_CH2': self.digital_width2_read,
            'R_CH3': self.digital_width2_read,
            'R_CH4': self.digital_width2_read,
        }
        # Digital1 repetition period (ns). If set, digital1 pulses every `digital1_repeat` ns after its start.
        self.digital1_repeat = 1000     # ns (1 us)
        self.sampling_rate = 1e9        # Hz

        # Total duration parameter (if None, calculated automatically)
        # If None, total_duration will be computed as max_delay + (freq_steps**2 * point_duration)
        self.total_duration = None      # ns - if set, overrides calculated duration
        adjustable_delay = 0
        W_CH_tunable = 9420
        R_CH_tunable = 10050
        
        # Channel digital delays
        self.channel_digital_delays = {
            'W_CH1': W_CH_tunable+0,    # ns
            'W_CH2': W_CH_tunable+50,    # ns
            'W_CH3': W_CH_tunable+2*50,   # ns
            'W_CH4': W_CH_tunable+3*50,   # ns
            'R_CH1': R_CH_tunable,   # ns
            'R_CH2': R_CH_tunable+50,   # ns
            'R_CH3': R_CH_tunable+2*50,   # ns
            'R_CH4': R_CH_tunable+3*50,   # ns
        }
        # Channel digital2 delays (independent offsets for digital2). By default mirror channel_digital_delays
        self.channel_digital2_delays = dict(self.channel_digital_delays)

        # Channel analog delays
        self.channel_analog_delays = {
            'W_CH1': 3970+adjustable_delay,    #ns
            'W_CH2': 4800+adjustable_delay,    # ns
            'W_CH3': 1750+adjustable_delay,    # ns
            'W_CH4': adjustable_delay,    # ns
            'R_CH1': 1950+adjustable_delay,    # ns
            'R_CH2': adjustable_delay,    # ns
            'R_CH3': 800+adjustable_delay,    # ns
            'R_CH4': 5100+adjustable_delay,    # ns
        }

        # Optional: absolute physical delays per channel (ns). If you provide these,
        # the generator will treat them as the actual hardware arrival times and
        # compute `channel_analog_delays` so that the channel with the largest
        # physical delay starts at sequence time 0. Example:
        # {'W_CH1':4100,'W_CH2':4100,'W_CH3':9200,'W_CH4':9200}
        # This will make W_CH3/4 start immediately (analog delay 0) and
        # W_CH1/2 delayed by 9200-4100=5100 ns relative to sequence start.
        self.channel_physical_delays = None

        # Amplitude correction coefficients per frequency point (length will be set at generate time)
        # If None, will default to ones(total_points)
        self.correction_matrix = None

        # Phase offsets (radians) per frequency point for Y channels only. If None will default to zeros.
        self.phase_vector = None

        # Output directory
        self.output_dir = os.path.join("AWG_Sequence", "256_awg_sequences")

        # Analog active period (ns). If set, analog is silent after this time from sequence start.
        # Example: 256000 for 256 us
        self.analog_active_until = None

def generate_awg_sequence(config=None):
    """Generate AWG sequence"""
    
    if config is None:
        config = AODAWGConfig()
    
    # Calculate parameters
    total_points = config.freq_steps * config.freq_steps
    
    # Calculate base duration from points
    base_duration = total_points * config.point_duration
    
    # Ensure correction_matrix and phase_vector have correct lengths
    # Initialize arrays on locals to avoid IDE/static-checker warnings about None indexing
    if config.correction_matrix is None:
        n = config.freq_steps
        factors = np.zeros((n, n), dtype=float)
        for row in range(n):
            for col in range(n):
                i = row + 1
                j = col + 1
                factors[row, col] = (80-2.5*i - 2.5*j)**(1.18)/65
                # --- Hot-corner flattening fit (focused) ---
                # Only the diagonal band around (i, j) = (12, 13) is hot;
                # cells along either edge (high i, low j) or (low i, high j)
                # are NOT hot. Use a tight 2D Gaussian (sigma_i=2.5,
                # sigma_j=1.8) centred at the empirical peak. Cells far
                # from the peak keep fit_damp ~ 1.0 (no change); cells near
                # the peak get fit_damp ~ 0.01-0.05 (strong attenuation that,
                # combined with the consumer's 0.35 floor clamp at line
                # 327/462, flattens the hot band).
                _di = (i - 12.0) / 2.5
                _dj = (j - 13.0) / 1.8
                _hot = np.exp(-(_di * _di + _dj * _dj))
                fit_damp = 1.0 / (1.0 + 80.0 * _hot)
                factors[row, col] *= fit_damp
        correction_vec = factors.flatten()
        max_val = np.max(correction_vec) if correction_vec.size > 0 else 0.0
        if max_val > 0:
            correction_vec = correction_vec / max_val
    else:
        corr_arr = np.asarray(config.correction_matrix)
        if corr_arr.size != total_points:
            corr_arr = np.resize(corr_arr, total_points)
        correction_vec = corr_arr

    if config.phase_vector is None:
        n = config.freq_steps
        phase_mat = np.zeros((n, n), dtype=float)

        # tunable parameters
        alpha = 0.35   # rad per step (x)
        beta  = 0.27   # rad per step (y)
        phi0  = np.pi / 3

        i0 = (n + 1) / 2
        j0 = (n + 1) / 2

        for row in range(n):
            for col in range(n):
                i = row + 1
                j = col + 1
                phase_mat[row, col] = (
                    phi0
                    + alpha * (i - i0)
                    + beta  * (j - j0)
                )

        phase_vec = phase_mat.flatten()

    else:
        ph_arr = np.asarray(config.phase_vector)
        if ph_arr.size != total_points:
            ph_arr = np.resize(ph_arr, total_points)
        phase_vec = ph_arr

    
    # Use total_duration if provided, otherwise compute as max_delay + base_duration
    # (ensures buffers cover the maximum channel delays)
    # If user provided absolute physical delays, convert them to per-channel
    # analog delays referenced to the maximum physical delay. This lets the user
    # specify hardware arrival times (physical delays) and have the generator
    # shift channels so the channel with the largest physical delay starts
    # at sequence time 0.
    phys = getattr(config, 'channel_physical_delays', None) or {}
    if phys:
        # Ensure keys cover the channels; fallback to 0 for missing keys
        reference = int(max(phys.values()))
        # Compute channel analog delays as offset from the reference
        computed_analog_delays = {}
        for ch in config.channel_analog_delays.keys():
            p = int(phys.get(ch, 0))
            computed_analog_delays[ch] = int(max(0, reference - p))
        # Replace config.channel_analog_delays for downstream logic
        config.channel_analog_delays = computed_analog_delays
        # Align digital pulses to the same physical reference by default
        config.digital_delay = int(reference)

    # First compute max delays (we need channel maps available)
    # digital delay for each channel = config.digital_delay + channel offset
    max_dig1 = max(config.digital_delay + v for v in config.channel_digital_delays.values())
    # include digital2 base delay + channel offsets as well
    max_dig2 = max(config.digital_delay2 + v for v in config.channel_digital2_delays.values())
    max_digital_delay = max(max_dig1, max_dig2)
    max_analog_delay = max(config.channel_analog_delays.values())
    max_delay = max(max_digital_delay, max_analog_delay)

    if config.total_duration is not None:
        total_duration = config.total_duration
        print(f"Using custom total duration: {total_duration} ns ({total_duration/1000} us)")
        if total_duration < base_duration:
            print(f"Warning: Custom total duration {total_duration}ns is less than required {base_duration}ns")
            print("Some points may be truncated")
    else:
        total_duration = max_delay + base_duration
        print(f"Using computed total duration = max_delay({max_delay}) + base_duration({base_duration}) = {total_duration} ns ({total_duration/1000} us)")
    
    print(f"Max digital delay: {max_digital_delay} ns")
    print(f"Max analog delay: {max_analog_delay} ns")
    
    # Create output directory
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Generate frequency sequence
    frequencies = np.linspace(config.freq_start, config.freq_end, config.freq_steps)
    
    # Create X,Y frequency grid
    fx_grid, fy_grid = np.meshgrid(frequencies, frequencies)
    fx = fx_grid.flatten()
    fy = fy_grid.flatten()
    
    print(f"Freq range: {config.freq_start} - {config.freq_end} MHz")
    print(f"Steps: {config.freq_steps}")
    print(f"Point duration: {config.point_duration} ns")
    print(f"Digital1 delay: {config.digital_delay} ns")
    print(f"Digital1 width (W/R): {config.digital_width_write}/{config.digital_width_read} ns")
    print(f"Digital2 delay: {config.digital_delay2} ns")
    print(f"Digital2 width (W/R): {config.digital_width2_write}/{config.digital_width2_read} ns")
    analog_until = getattr(config, 'analog_active_until', None)
    if analog_until is not None:
        print(f"Analog active until: {analog_until} ns ({analog_until/1000} us)")
    
    # Initialize channels data with buffer for delays
    buffer_size = total_duration + max_delay + config.point_duration
    # Each channel stores: [analog_value, digital1, digital2]
    channels = {
        'W_CH1': np.zeros((buffer_size, 3)),
        'W_CH2': np.zeros((buffer_size, 3)),
        'W_CH3': np.zeros((buffer_size, 3)),
        'W_CH4': np.zeros((buffer_size, 3)),
        'R_CH1': np.zeros((buffer_size, 3)),
        'R_CH2': np.zeros((buffer_size, 3)),
        'R_CH3': np.zeros((buffer_size, 3)),
        'R_CH4': np.zeros((buffer_size, 3))
    }
    
    # Create scan sequence
    odd_points = list(range(1, total_points + 1, 2))
    even_points = list(range(2, total_points + 1, 2))
    scan_sequence = odd_points + even_points
    
    print(f"Total points: {total_points}")
    print(f"Scan sequence: {len(odd_points)} odd + {len(even_points)} even")
    
    # Generate modulation coefficients
    modulation_coeffs = {}
    for ch_name in channels.keys():
        modulation_coeffs[ch_name] = np.ones(total_points)
    
    # Generate sequence data
    for seq_idx, point_num in enumerate(scan_sequence):
        # Actual point index
        actual_idx = point_num - 1
        
        # Start position
        time_start = seq_idx * config.point_duration
        
        # Check if this point fits within total duration
        if time_start >= total_duration:
            break  # Skip points beyond total duration

        # Get frequencies
        x_freq = fx[actual_idx]  # MHz
        y_freq = fy[actual_idx]  # MHz

        # Get correction coefficient (from local array) and apply floor compensation.
        # The fit_damp inside the hot diagonal band compresses the cell to ~0.005-0.015
        # (well below useful amplitude); without a floor the hot spot would be effectively
        # muted. Apply a uniform floor of 0.1 across all cells so:
        #   - hot cells (raw ~0.002-0.006 post-fit, post-normalization) are lifted to 0.1,
        #   - other cells with low correction values are also lifted to 0.1,
        #   - cells above 0.1 are unchanged.
        correction_coeff = correction_vec[actual_idx]
        if correction_coeff < 0.1:
            correction_coeff = 0.1
        

        # If this point is on a resonant frequency (90/100/110/120 MHz),
        # set the per-channel modulation amplitude for all W channels (main scan) to 2.5
        #if (
            #abs(x_freq - 90.0) < 1e-6 or abs(x_freq - 100.0) < 1e-6 or
            #abs(x_freq - 110.0) < 1e-6 or abs(x_freq - 120.0) < 1e-6 or
            #abs(y_freq - 90.0) < 1e-6 or abs(y_freq - 100.0) < 1e-6 or
            #abs(y_freq - 110.0) < 1e-6 or abs(y_freq - 120.0) < 1e-6
        #):
            #for mch in modulation_coeffs.keys():
                #if mch.startswith('W_'):
                    #modulation_coeffs[mch][actual_idx] = 2.5

        # Generate data for this point
        for ns in range(config.point_duration):
            global_idx = time_start + ns
            
            # Check if we're within total duration
            if global_idx >= total_duration:
                break
                
            # Process each channel
            for ch_name in channels.keys():
                # Calculate delays (relative to point start)
                dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
                ana_delay = config.channel_analog_delays[ch_name]
                
                # Analog time with delay
                analog_time_idx = global_idx + ana_delay
                
                # Generate RF signal
                t = analog_time_idx / config.sampling_rate

                # Base analog value
                if ch_name in ['W_CH1', 'W_CH3', 'R_CH1', 'R_CH3']:
                    # X frequency channels (no phase offset)
                    analog_val = modulation_coeffs[ch_name][actual_idx] * np.sin(2 * np.pi * x_freq * 1e6 * t)
                else:
                    # Y frequency channels: apply per-point initial phase (from local phase_vec)
                    phase = phase_vec[actual_idx]
                    analog_val = modulation_coeffs[ch_name][actual_idx] * np.sin(2 * np.pi * y_freq * 1e6 * t + phase)

                # Apply amplitude correction: move factor to W_CH1 and apply compensation floor
                if ch_name == 'W_CH1':
                    analog_val *= correction_coeff
                # Resonant-frequency amplitude is applied via `modulation_coeffs` (set earlier),
                # so do not overwrite the per-sample `analog_val` here.
                
                # digital signals are scheduled in a later pass (per-point absolute scheduling)

                # Apply analog silence: if analog_active_until is set and we've passed it (in analog time), force analog_val=0
                analog_until = getattr(config, 'analog_active_until', None)
                if analog_until is not None and analog_time_idx >= analog_until:
                    analog_val = 0.0

                # Store data (analog, digital1, digital2) at the analog_time index so delays take effect
                if 0 <= analog_time_idx < buffer_size:
                    channels[ch_name][int(analog_time_idx), 0] = analog_val
                    # leave digital columns for now (will be set in scheduling pass)
    
    # Trim channels to specified total duration
    # Before trimming, schedule digital pulses (one pulse per point) at absolute times
    for ch_name in channels.keys():
        # clear digital columns
        channels[ch_name][:, 1] = 0
        channels[ch_name][:, 2] = 0

    for seq_idx, point_num in enumerate(scan_sequence):
        point_start = seq_idx * config.point_duration
        for ch_name in channels.keys():
            # digital1
            dig1_start = point_start + config.digital_delay + config.channel_digital_delays[ch_name]
            # per-channel width (W channels use digital_width_write, R channels use digital_width_read)
            dig1_width = config.channel_digital_widths.get(ch_name, config.digital_width)
            dig1_end = dig1_start + dig1_width
            s = int(max(0, dig1_start))
            e = int(min(buffer_size, dig1_end))
            if s < e:
                channels[ch_name][s:e, 1] = 1

            # digital2
            dig2_channel_offset = config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
            dig2_start = point_start + config.digital_delay2 + dig2_channel_offset
            dig2_width = config.channel_digital2_widths.get(ch_name, config.digital_width2)
            dig2_end = dig2_start + dig2_width
            s2 = int(max(0, dig2_start))
            e2 = int(min(buffer_size, dig2_end))
            if s2 < e2:
                channels[ch_name][s2:e2, 2] = 1

    for ch_name in channels.keys():
        channels[ch_name] = channels[ch_name][:total_duration]
    
    # Generate filename prefix with common parameters
    total_duration_us = total_duration / 1000
    common_prefix = f"T{total_duration_us:.0f}us_D{config.digital_delay}ns_W{config.digital_width}ns"
    
    # Save to files with descriptive names
    for ch_name, data in channels.items():
        # Get channel-specific delays
        total_dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
        total_dig_delay2 = config.digital_delay2 + config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
        ana_delay = config.channel_analog_delays[ch_name]

        # Create filename with parameters
        filename = f"AWG_{ch_name}_{common_prefix}_dD{total_dig_delay}ns_dD2{total_dig_delay2}ns_aD{ana_delay}ns.txt"
        filepath = os.path.join(config.output_dir, filename)

        # Save data with CSV header (analog,digital1,digital2). Use explicit delimiter and no comment char for header.
        np.savetxt(filepath, data, fmt=['%.7f','%d','%d'], delimiter=',', header='analog,digital1,digital2', comments='')
        print(f"Channel {ch_name} saved: {filename}")
    
    # Generate plots with descriptive name
    plot_signals(channels, config, common_prefix)
    
    linear_sequence = list(range(1, total_points + 1))

    # Initialize channels for linear sequence
    channels_lin = {ch: np.zeros((buffer_size, 3)) for ch in channels.keys()}

    # modulation coefficients (reuse)
    modulation_coeffs_lin = {ch: np.ones(total_points) for ch in channels_lin.keys()}

    # Generate data for linear sequence
    for seq_idx, point_num in enumerate(linear_sequence):
        actual_idx = point_num - 1
        time_start = seq_idx * config.point_duration
        if time_start >= total_duration:
            break
        x_freq = fx[actual_idx]
        y_freq = fy[actual_idx]
        # Same uniform floor=0.1 as the scan path: hot cells (raw ~0.002-0.006
        # post-fit, post-normalization) are lifted to 0.1; other low-value
        # cells are also lifted to 0.1; cells above 0.1 are unchanged.
        correction_coeff = correction_vec[actual_idx]
        if correction_coeff < 0.1:
            correction_coeff = 0.1
        
        if (
            abs(x_freq - 90.0) < 1e-6 or abs(x_freq - 100.0) < 1e-6 or
            abs(x_freq - 110.0) < 1e-6 or abs(x_freq - 120.0) < 1e-6 or
            abs(y_freq - 90.0) < 1e-6 or abs(y_freq - 100.0) < 1e-6 or
            abs(y_freq - 110.0) < 1e-6 or abs(y_freq - 120.0) < 1e-6
        ):
            for mch in modulation_coeffs_lin.keys():
                if mch.startswith('W_'):
                    modulation_coeffs_lin[mch][actual_idx] = 0.95 * modulation_coeffs_lin[mch][actual_idx]

        for ns in range(config.point_duration):
            global_idx = time_start + ns
            if global_idx >= total_duration:
                break
            for ch_name in channels_lin.keys():
                dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
                ana_delay = config.channel_analog_delays[ch_name]
                analog_time_idx = global_idx + ana_delay
                t = analog_time_idx / config.sampling_rate

                if ch_name in ['W_CH1', 'W_CH3', 'R_CH1', 'R_CH3']:
                    analog_val = modulation_coeffs_lin[ch_name][actual_idx] * np.sin(2 * np.pi * x_freq * 1e6 * t)
                else:
                    phase = phase_vec[actual_idx]
                    analog_val = modulation_coeffs_lin[ch_name][actual_idx] * np.sin(2 * np.pi * y_freq * 1e6 * t + phase)

                if ch_name == 'W_CH1':
                    analog_val *= correction_coeff

                analog_until = getattr(config, 'analog_active_until', None)
                if analog_until is not None and analog_time_idx >= analog_until:
                    analog_val = 0.0

                if 0 <= analog_time_idx < buffer_size:
                    channels_lin[ch_name][int(analog_time_idx), 0] = analog_val

    # Schedule digital pulses for linear sequence
    for ch_name in channels_lin.keys():
        channels_lin[ch_name][:, 1] = 0
        channels_lin[ch_name][:, 2] = 0

    for seq_idx, point_num in enumerate(linear_sequence):
        point_start = seq_idx * config.point_duration
        for ch_name in channels_lin.keys():
            dig1_start = point_start + config.digital_delay + config.channel_digital_delays[ch_name]
            dig1_width = config.channel_digital_widths.get(ch_name, config.digital_width)
            dig1_end = dig1_start + dig1_width
            s = int(max(0, dig1_start))
            e = int(min(buffer_size, dig1_end))
            if s < e:
                channels_lin[ch_name][s:e, 1] = 1

            dig2_channel_offset = config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
            dig2_start = point_start + config.digital_delay2 + dig2_channel_offset
            dig2_width = config.channel_digital2_widths.get(ch_name, config.digital_width2)
            dig2_end = dig2_start + dig2_width
            s2 = int(max(0, dig2_start))
            e2 = int(min(buffer_size, dig2_end))
            if s2 < e2:
                channels_lin[ch_name][s2:e2, 2] = 1

    for ch_name in channels_lin.keys():
        channels_lin[ch_name] = channels_lin[ch_name][:total_duration]

    # Save linear sequence files (with '_linear' suffix)
    for ch_name, data in channels_lin.items():
        total_dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
        total_dig_delay2 = config.digital_delay2 + config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
        ana_delay = config.channel_analog_delays[ch_name]
        filename = f"AWG_{ch_name}_{common_prefix}_linear_dD{total_dig_delay}ns_dD2{total_dig_delay2}ns_aD{ana_delay}ns.txt"
        filepath = os.path.join(config.output_dir, filename)
        np.savetxt(filepath, data, fmt=['%.7f','%d','%d'], delimiter=',', header='analog,digital1,digital2', comments='')
        print(f"Channel {ch_name} saved (linear): {filename}")

    # Plot linear sequence
    plot_signals(channels_lin, config, common_prefix + '_linear')

    # Generate spiral order indices for an n x n grid (1-based point numbers)
    def make_spiral_sequence(n):
        # start from center for odd n; for even n behavior will pick the lower-center
        cx = (n - 1) // 2
        cy = (n - 1) // 2
        seq_coords = [(cx, cy)]
        step = 1
        # directions: right, down, left, up (clockwise spiral)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = cx, cy
        d = 0
        while len(seq_coords) < n * n:
            # move 'step' times in current direction
            for _ in range(2):  # each step value repeats twice (for rectangular spiral)
                dx, dy = dirs[d % 4]
                for _ in range(step):
                    x += dx
                    y += dy
                    if 0 <= x < n and 0 <= y < n:
                        seq_coords.append((x, y))
                        if len(seq_coords) >= n * n:
                            break
                d += 1
                if len(seq_coords) >= n * n:
                    break
            step += 1
        # convert (row,col) to 1-based point numbers assuming row-major flatten
        seq = [r * n + c + 1 for (r, c) in seq_coords]
        return seq

    spiral_sequence = make_spiral_sequence(config.freq_steps)
    print(f"Spiral sequence generated: {len(spiral_sequence)} points")

    # Initialize channels for spiral sequence
    channels_spiral = {ch: np.zeros((buffer_size, 3)) for ch in channels.keys()}
    modulation_coeffs_spiral = {ch: np.ones(total_points) for ch in channels_spiral.keys()}

    # Generate data for spiral sequence (same logic as linear/regular sequences)
    for seq_idx, point_num in enumerate(spiral_sequence):
        actual_idx = point_num - 1
        time_start = seq_idx * config.point_duration
        if time_start >= total_duration:
            break
        x_freq = fx[actual_idx]
        y_freq = fy[actual_idx]
        correction_coeff = correction_vec[actual_idx]
        # Same uniform floor=0.1 as scan/linear sequences above (see comment at line ~347).
        if correction_coeff < 0.1:
            correction_coeff = 0.1

        for ns in range(config.point_duration):
            global_idx = time_start + ns
            if global_idx >= total_duration:
                break
            for ch_name in channels_spiral.keys():
                dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
                ana_delay = config.channel_analog_delays[ch_name]
                analog_time_idx = global_idx + ana_delay
                t = analog_time_idx / config.sampling_rate

                if ch_name in ['W_CH1', 'W_CH3', 'R_CH1', 'R_CH3']:
                    analog_val = modulation_coeffs_spiral[ch_name][actual_idx] * np.sin(2 * np.pi * x_freq * 1e6 * t)
                else:
                    phase = phase_vec[actual_idx]
                    analog_val = modulation_coeffs_spiral[ch_name][actual_idx] * np.sin(2 * np.pi * y_freq * 1e6 * t + phase)

                if ch_name == 'W_CH1':
                    analog_val *= correction_coeff

                analog_until = getattr(config, 'analog_active_until', None)
                if analog_until is not None and analog_time_idx >= analog_until:
                    analog_val = 0.0

                if 0 <= analog_time_idx < buffer_size:
                    channels_spiral[ch_name][int(analog_time_idx), 0] = analog_val

    # Schedule digital pulses for spiral sequence
    for ch_name in channels_spiral.keys():
        channels_spiral[ch_name][:, 1] = 0
        channels_spiral[ch_name][:, 2] = 0

    for seq_idx, point_num in enumerate(spiral_sequence):
        point_start = seq_idx * config.point_duration
        for ch_name in channels_spiral.keys():
            dig1_start = point_start + config.digital_delay + config.channel_digital_delays[ch_name]
            dig1_width = config.channel_digital_widths.get(ch_name, config.digital_width)
            dig1_end = dig1_start + dig1_width
            s = int(max(0, dig1_start))
            e = int(min(buffer_size, dig1_end))
            if s < e:
                channels_spiral[ch_name][s:e, 1] = 1

            dig2_channel_offset = config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
            dig2_start = point_start + config.digital_delay2 + dig2_channel_offset
            dig2_width = config.channel_digital2_widths.get(ch_name, config.digital_width2)
            dig2_end = dig2_start + dig2_width
            s2 = int(max(0, dig2_start))
            e2 = int(min(buffer_size, dig2_end))
            if s2 < e2:
                channels_spiral[ch_name][s2:e2, 2] = 1

    for ch_name in channels_spiral.keys():
        channels_spiral[ch_name] = channels_spiral[ch_name][:total_duration]

    # Save spiral sequence files (with '_spiral' suffix)
    for ch_name, data in channels_spiral.items():
        total_dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
        total_dig_delay2 = config.digital_delay2 + config.channel_digital2_delays.get(ch_name, config.channel_digital_delays[ch_name])
        ana_delay = config.channel_analog_delays[ch_name]
        filename = f"AWG_{ch_name}_{common_prefix}_spiral_dD{total_dig_delay}ns_dD2{total_dig_delay2}ns_aD{ana_delay}ns.txt"
        filepath = os.path.join(config.output_dir, filename)
        np.savetxt(filepath, data, fmt=['%.7f','%d','%d'], delimiter=',', header='analog,digital1,digital2', comments='')
        print(f"Channel {ch_name} saved (spiral): {filename}")

    # Plot spiral sequence
    plot_signals(channels_spiral, config, common_prefix + '_spiral')

    return channels, config

def plot_signals(channels, config, common_prefix=""):
    """Generate signal plots"""
    
    display_points = min(50000, len(channels['W_CH1']))
    
    plt.figure(figsize=(20, 16))
    
    for idx, (ch_name, data) in enumerate(channels.items()):
        plt.subplot(4, 2, idx + 1)
        
        display_data = data[:display_points]
        time_ns = np.arange(len(display_data))

        analog_signal = display_data[:, 0]
        digital1_signal = display_data[:, 1]
        digital2_signal = display_data[:, 2] if display_data.shape[1] > 2 else np.zeros_like(digital1_signal)

        plt.plot(time_ns, analog_signal, 'b-', label='Analog', linewidth=0.5, alpha=0.7)
        plt.plot(time_ns, digital1_signal * 1.2 * np.max(analog_signal), 'r-', label='Digital1', linewidth=1)
        plt.plot(time_ns, digital2_signal * 1.4 * np.max(analog_signal), 'g-', label='Digital2', linewidth=1)

        total_dig_delay = config.digital_delay + config.channel_digital_delays[ch_name]
        total_dig_delay2 = config.digital_delay2 + config.channel_digital_delays[ch_name]
        ana_delay = config.channel_analog_delays[ch_name]
        plt.title(f'{ch_name}\nDig1: {total_dig_delay}ns, Dig2: {total_dig_delay2}ns, Ana: {ana_delay}ns')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)

        y_max = max(np.max(analog_signal), 1.4) if len(analog_signal) > 0 else 1.0
        plt.ylim(-y_max * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    
    # Create descriptive plot filename
    if common_prefix:
        plot_filename = f"AWG_Signals_{common_prefix}.png"
    else:
        total_duration_us = len(channels['W_CH1']) / 1000
        plot_filename = f"AWG_Signals_T{total_duration_us:.0f}us_D{config.digital_delay}ns_W{config.digital_width}ns.png"
    
    plot_filepath = os.path.join(config.output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_filename}")
    _maybe_show_plot()

def create_custom_config(base_config: Optional[AODAWGConfig] = None):
    """Create a custom configuration derived from AODAWGConfig defaults.

    To allow changing defaults in `AODAWGConfig` to take effect, this function
    now accepts an optional `base_config`. If none provided, it creates a
    fresh `AODAWGConfig()` and only overrides the specific fields the user
    likely intends to change for the "custom" case. This avoids silently
    hardcoding values (like freq_start/freq_end/freq_steps) that would
    otherwise ignore class-level changes.
    """
    config = base_config if base_config is not None else AODAWGConfig()

    # Apply only the custom changes (leave freq range/steps to class defaults)
    # Timing: keep 1 us point duration unless the class default was changed
    config.point_duration = 1000

    # Example custom digital timings (these are intentional overrides)
    config.digital_delay = 10000
    config.digital_width = 250
    config.digital_delay2 = 10000
    config.digital_width2 = 250

    # propagate per-type widths to channel maps
    config.digital_width_write = config.digital_width
    config.digital_width_read = config.digital_width
    config.digital_width2_write = config.digital_width2
    config.digital_width2_read = config.digital_width2
    config.channel_digital_widths = {ch: config.digital_width_write if ch.startswith('W_') else config.digital_width_read
                                    for ch in config.channel_digital_delays.keys()}
    config.channel_digital2_widths = {ch: config.digital_width2_write if ch.startswith('W_') else config.digital_width2_read
                                     for ch in config.channel_digital2_delays.keys()}

    # Total duration override and analog active period (intentional)
    config.total_duration = 266000  # ns (266 us)
    config.analog_active_until = 256000

    # Example per-channel delays for this custom config (intentional)
    config.channel_digital_delays = {
        'W_CH1': 0, 'W_CH2': 50, 'W_CH3': 100, 'W_CH4': 150,
        'R_CH1': 25, 'R_CH2': 75, 'R_CH3': 125, 'R_CH4': 175,
    }

    config.channel_analog_delays = {
        'W_CH1': 0, 'W_CH2': 20, 'W_CH3': 40, 'W_CH4': 60,
        'R_CH1': 10, 'R_CH2': 30, 'R_CH3': 50, 'R_CH4': 70,
    }

    # Correction matrix: default to ones so amplitudes are not altered
    total_pts = config.freq_steps * config.freq_steps
    config.correction_matrix = np.ones(total_pts)

    # Keep digital2 delays mirrored by default
    config.channel_digital2_delays = dict(config.channel_digital_delays)

    config.output_dir = os.path.join("AWG_Sequence", "custom_awg_sequences_256")

    return config

# Main execution
if __name__ == "__main__":
    print("Starting AOD AWG generation...")
    
    # Option 1: Default config
    # By default use the built-in AODAWGConfig with 16 points from 85 to 122.5 MHz.
    awg_sequences, config = generate_awg_sequence()

    # If you want the custom configuration, uncomment these lines instead:
    # custom_config = create_custom_config()
    # awg_sequences, config = generate_awg_sequence(custom_config)
    
    print("\nGeneration completed!")
    print(f"Data length: {len(awg_sequences['W_CH1'])} points ({len(awg_sequences['W_CH1'])/1000} us)")
    print("\nChannel delays:")
    for ch_name in awg_sequences.keys():
        total_dig = config.digital_delay + config.channel_digital_delays[ch_name]
        ana_delay = config.channel_analog_delays[ch_name]
        print(f"  {ch_name}: Dig={total_dig}ns, Ana={ana_delay}ns")
