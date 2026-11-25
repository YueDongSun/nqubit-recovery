"""
AWG5014C Python Controller - Converted from MATLAB codes

This module provides Python equivalents for the archived MATLAB AWG control
scripts in the script_old/AWGcontrol_port/codes/ directory.

Main functions:
- gen_wave(y, samples): Send waveform to AWG5014C
- sine_wave(harmonic, amplitude): Generate and send single sine wave
- sine_waves(harmonics, amplitudes, phases): Generate and send multiple sine waves
- get_image(camera, exposure): Capture image from camera (placeholder)
- sum_intensity(camera, exposure): Calculate total image intensity
- calculate_light_intensity(points, radius, image): Calculate intensity at specified points

Dependencies:
    pip install pyvisa numpy

Note: Camera functions are placeholders - implement based on your camera library.
"""

import numpy as np
import pyvisa
from typing import Any, Tuple, Optional, List

from awg5014c_transport import open_instrument_resource


class AWG5014CController:
    """Controller for Tektronix AWG5014C arbitrary waveform generator."""

    def __init__(self, visa_address: str = '192.168.2.122::4011', buffer_size: int = 2048):
        """
        Initialize AWG5014C controller.

        Args:
            visa_address: VISA address of the AWG (default: 192.168.2.122::4011)
            buffer_size: Buffer size for VISA communication (default: 2048)
        """
        self.visa_address = visa_address
        self.buffer_size = buffer_size
        self.rm: Optional[pyvisa.ResourceManager] = None
        self.awg: Optional[Any] = None

    def connect(self) -> None:
        """Establish connection to the AWG5014C."""
        self.awg = open_instrument_resource(self.visa_address)
        self.awg.read_termination = '\n'
        print(f"Connected to AWG5014C at {self.visa_address}")

    def disconnect(self) -> None:
        """Close the connection to the AWG."""
        if self.awg:
            self.awg.close()
        if self.rm:
            self.rm.close()
        self.awg = None
        self.rm = None
        print("Disconnected from AWG5014C")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def set_marker_voltage(self, high: float = 2.5, low: float = 0.0) -> None:
        """Set marker output HIGH/LOW voltage level on every source & marker.

        Marker voltage is global per AWG (per source, not per channel), and each
        source has two markers (M1, M2). Valid range is [-2.7, +2.7] V for both
        levels with HIGH > LOW and (HIGH - LOW) <= 5.4 V. AWG5014C defaults are
        HIGH=1.0 V / LOW=0.0 V; we override to 2.5 V / 0.0 V for the W-state
        experiment so digital pulses drive downstream hardware at full scale.

        Args:
            high: Marker HIGH level in volts (default 2.5).
            low:  Marker LOW level in volts (default 0.0).
        """
        if self.awg is None:
            raise ConnectionError("Not connected to AWG. Call connect() first.")
        if high <= low:
            raise ValueError(
                f"marker HIGH ({high} V) must be > LOW ({low} V)"
            )
        if abs(high) > 2.7 or abs(low) > 2.7:
            raise ValueError(
                f"marker voltages must be in [-2.7, +2.7] V (got HIGH={high}, LOW={low})"
            )
        if (high - low) > 5.4:
            raise ValueError(
                f"marker swing (HIGH-LOW={high - low} V) exceeds 5.4 V maximum"
            )
        for src in range(1, 5):
            for mk in (1, 2):
                self.awg.write(
                    f"SOURce{src}:MARKer{mk}:VOLTage:LEVel:IMMediate:HIGH {high}"
                )
                self.awg.write(
                    f"SOURce{src}:MARKer{mk}:VOLTage:LEVel:IMMediate:LOW {low}"
                )
        print(
            f"Marker voltage set on {self.visa_address}: HIGH={high} V, LOW={low} V "
            f"(4 sources x 2 markers = 8 outputs)"
        )

    def gen_wave(self, y: np.ndarray, samples: int, channel: int = 1) -> None:
        """
        Send a waveform to the AWG5014C.

        Equivalent to MATLAB gen_wave.m

        Args:
            y: Waveform data array (should be in range [-1, 1])
            samples: Number of samples
            channel: Output channel (1-4, default: 1)
        """
        if self.awg is None:
            raise ConnectionError("Not connected to AWG. Call connect() first.")

        # Ensure data is float32 and in valid range
        y = np.array(y, dtype=np.float32)
        y = np.clip(y, -1.0, 1.0)

        # Marker configuration (same as MATLAB)
        m1 = np.zeros(samples, dtype=np.uint8)
        m1[2:12] = 1
        m1 = np.left_shift(m1.astype(bool), 6).astype(np.uint8)

        m2 = np.zeros(samples, dtype=np.uint8)
        m2[2::2] = 1
        m2 = np.left_shift(m2.astype(bool), 7).astype(np.uint8)

        m = m1 + m2

        # Build binary block (5 bytes per sample: 4 for float32 + 1 for marker)
        binblock = np.zeros(samples * 5, dtype=np.uint8)
        for k in range(samples):
            y_bytes = np.frombuffer(np.float32(y[k]).tobytes(), dtype=np.uint8)
            binblock[k*5:k*5+4] = y_bytes
            binblock[k*5+4] = m[k]

        # Build binary block header (IEEE 488.2 definite-length block)
        num_bytes = len(binblock)
        bytes_str = str(num_bytes)
        header = f'#{len(bytes_str)}{bytes_str}'

        # Delete old waveform
        try:
            self.awg.write('wlist:waveform:delete "newwave1"')
        except pyvisa.Error:
            pass

        # Create new waveform destination
        self.awg.write(f':wlist:waveform:new "newwave1",{samples},real')

        # Write waveform data
        cmd = f':wlist:waveform:data "newwave1",0,{samples},{header}'
        cmd_bytes = cmd.encode('ascii')
        total_bytes = len(cmd_bytes) + num_bytes
        old_termination = self.awg.write_termination
        termination = (old_termination or '\n').encode('ascii')

        if total_bytes <= self.buffer_size:
            # Send in one chunk
            self.awg.write_raw(cmd_bytes + binblock.tobytes() + termination)
        else:
            # Send in chunks
            self.awg.write_termination = ''
            try:
                for i in range(0, len(cmd_bytes), self.buffer_size):
                    self.awg.write_raw(cmd_bytes[i:i+self.buffer_size])
                waveform_bytes = binblock.tobytes()
                for i in range(0, num_bytes, self.buffer_size):
                    self.awg.write_raw(waveform_bytes[i:i+self.buffer_size])
                self.awg.write_raw(termination)
            finally:
                self.awg.write_termination = old_termination

        # Source channel and enable output
        self.awg.write(f'source{channel}:waveform "newwave1"')
        self.awg.write(f'OUTPUT{channel}:STATE ON')
        self.awg.write('AWGControl:RUN:IMMediate')

        print(f"Waveform sent to channel {channel}, AWG running")

    def run(self) -> None:
        """Start waveform generation."""
        if self.awg is None:
            raise ConnectionError("Not connected to AWG.")
        self.awg.write('AWGControl:RUN:IMMediate')
        print("AWG running")

    def stop(self) -> None:
        """Stop waveform generation."""
        if self.awg is None:
            raise ConnectionError("Not connected to AWG.")
        self.awg.write('AWGControl:STOP:IMMediate')
        print("AWG stopped")

    def set_output(self, channel: int, state: bool) -> None:
        """Turn output on/off for a channel."""
        if self.awg is None:
            raise ConnectionError("Not connected to AWG.")
        state_str = 'ON' if state else 'OFF'
        self.awg.write(f'OUTPUT{channel}:STATE {state_str}')


# ============================================================================
# Waveform Generation Functions
# ============================================================================

def sine_wave(controller: AWG5014CController, harmonic: float, amplitude: float,
              samples: int = 1000, cycles: int = 2, channel: int = 1) -> None:
    """
    Generate and send a single sine wave to AWG.

    Equivalent to MATLAB sine_wave.m

    Args:
        controller: AWG5014CController instance
        harmonic: Harmonic number (multiplier of base frequency)
        amplitude: Waveform amplitude (clipped to 1.5 max)
        samples: Number of samples (default: 1000, base freq = 1MHz)
        cycles: Number of cycles (default: 2)
        channel: Output channel (default: 1)
    """
    # Clamp amplitude
    if amplitude > 1.5:
        amplitude = 1.5

    # Generate time array
    step = (2 * np.pi) / samples
    cycles_rad = cycles * 2 * np.pi
    t = np.arange(0, cycles_rad, step)

    # Generate sine wave
    y = amplitude * np.sin(harmonic * t)
    y = y.astype(np.float32)

    # Send to AWG
    controller.gen_wave(y, len(y), channel)


def sine_waves(controller: AWG5014CController, harmonics: np.ndarray,
               amplitudes: np.ndarray, phases: np.ndarray,
               samples: int = 1000, cycles: int = 100, channel: int = 1) -> None:
    """
    Generate and send multiple sine waves (superposition) to AWG.

    Equivalent to MATLAB sine_waves.m

    Args:
        controller: AWG5014CController instance
        harmonics: Array of harmonic numbers
        amplitudes: Array of amplitudes (will be normalized if sum of squares > 16)
        phases: Array of phases in radians
        samples: Number of samples (default: 1000)
        cycles: Number of cycles, max 10000 (default: 100)
        channel: Output channel (default: 1)
    """
    # Normalize amplitudes if too large (MATLAB: if sum(amplitude .* amplitude) > 16.0)
    total_power = np.sum(amplitudes ** 2)
    if total_power > 16.0:
        amplitudes = 4 * amplitudes / np.sqrt(total_power)
        print(f"Amplitudes normalized (total power was {total_power:.2f})")

    # Generate time array
    step = (2 * np.pi) / samples
    cycles_rad = cycles * 2 * np.pi
    t = np.arange(0, cycles_rad, step)

    # Generate superposition of sine waves
    # MATLAB: y(i) = sum(amplitude .* cos((harmonic * t(i)) + phi))
    y = np.zeros_like(t, dtype=np.float32)
    for i in range(len(t)):
        y[i] = np.sum(amplitudes * np.cos(harmonics * t[i] + phases))

    # Send to AWG
    total_samples = len(y)
    controller.gen_wave(y, total_samples, channel)


# ============================================================================
# Image Capture and Intensity Calculation Functions
# ============================================================================

def get_image(camera, exposure_time: float) -> np.ndarray:
    """
    Capture an image from the camera.

    Equivalent to MATLAB getImage.m

    NOTE: This is a placeholder function. You need to implement camera capture
    based on your specific camera library (e.g., pycamera, opencv, basler pylon, etc.).

    Args:
        camera: Camera object (type depends on your camera library)
        exposure_time: Exposure time in seconds

    Returns:
        2D numpy array representing the image
    """
    # TODO: Implement camera capture based on your camera library
    # Example for hypothetical camera:
    # camera.ExposureTime = exposure_time
    # import time
    # time.sleep(2)
    # image = camera.snapshot()
    # return np.array(image, dtype=np.uint16)

    raise NotImplementedError(
        "Camera capture not implemented. Implement based on your camera library.\n"
        "Common libraries: pycamera, opencv-python, basler pylon, IDS peak, etc."
    )


def sum_intensity(camera, exposure_time: float) -> float:
    """
    Calculate total intensity of an image.

    Equivalent to MATLAB sumIntensity.m

    Args:
        camera: Camera object
        exposure_time: Exposure time in seconds

    Returns:
        Total intensity sum
    """
    image = get_image(camera, exposure_time)
    return float(np.sum(image))


def calculate_light_intensity(points: np.ndarray, radius: int,
                              image: np.ndarray) -> np.ndarray:
    """
    Calculate light intensity at specified points with Manhattan distance radius.

    Equivalent to MATLAB calculateLightIntensity.m

    Args:
        points: Nx2 array of (x, y) coordinates
        radius: Manhattan distance radius for neighborhood
        image: 2D image array

    Returns:
        Array of intensities at each point
    """
    rows, cols = image.shape
    intensities = np.zeros(len(points))

    for i, (x, y) in enumerate(points):
        x = int(round(x))
        y = int(round(y))

        intensity_sum = 0

        # Check if point is within image boundaries
        if 0 < x <= cols and 0 < y <= rows:
            # Sum intensity in Manhattan distance radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 < nx <= cols and 0 < ny <= rows:
                        intensity_sum += image[ny - 1, nx - 1]  # 0-indexed

        intensities[i] = intensity_sum

    return intensities


# ============================================================================
# Optimization Functions
# ============================================================================

def this_harmonics(controller: AWG5014CController, camera, harmonics: np.ndarray,
                   amplitudes: np.ndarray, phases: np.ndarray,
                   exposure_time: float, points: np.ndarray) -> np.ndarray:
    """
    Send harmonics and measure light intensities at points.

    Equivalent to MATLAB this_harmonics.m

    Args:
        controller: AWG5014CController instance
        camera: Camera object
        harmonics: Array of harmonic numbers
        amplitudes: Array of amplitudes
        phases: Array of phases
        exposure_time: Exposure time in seconds
        points: Nx2 array of (x, y) coordinates

    Returns:
        Array of light intensities at each point
    """
    # Send waveform to AWG
    sine_waves(controller, harmonics, amplitudes, phases)

    # Capture image
    image = get_image(camera, exposure_time)

    # Calculate intensities
    return calculate_light_intensity(points, 10, image)


def trigger_volt(controller: AWG5014CController, camera, harmonics: np.ndarray,
                 phases: np.ndarray, goal_lightness: np.ndarray,
                 exposure_time: float, points: np.ndarray, init_amp: float,
                 n: int, gi: float) -> np.ndarray:
    """
    Optimize amplitudes to achieve target light intensities.

    Equivalent to MATLAB trigger_volts.m

    Args:
        controller: AWG5014CController instance
        camera: Camera object
        harmonics: Array of harmonic numbers
        phases: Array of phases
        goal_lightness: Target light intensities
        exposure_time: Exposure time in seconds
        points: Nx2 array of (x, y) coordinates
        init_amp: Initial amplitude
        n: Number of harmonics
        gi: Step size for optimization

    Returns:
        Optimized voltage (amplitudes)
    """
    def reduced_lightness(amplitudes):
        li = this_harmonics(controller, camera, harmonics, amplitudes,
                           phases, exposure_time, points)
        print(f"Amplitudes: {amplitudes}")
        print(f"Lightness: {li}")
        return li

    # This would call an optimization function
    # voltage = find_linear(@reduced_lightness, n, 1, init_amp, 0.1, gi, 5, goal_lightness)
    # For now, return initial amplitude as placeholder
    print("trigger_volt: Optimization not fully implemented")
    return np.full(n, init_amp)


# ============================================================================
# Example Usage
# ============================================================================

def example_single_sine():
    """Example: Generate and send a single sine wave."""
    with AWG5014CController() as awg:
        sine_wave(awg, harmonic=93, amplitude=1.0, samples=1000, cycles=2, channel=1)


def example_multiple_sines():
    """Example: Generate and send multiple sine waves."""
    with AWG5014CController() as awg:
        harmonics = np.array([1, 2, 3, 4, 5])
        amplitudes = np.array([0.5, 0.3, 0.2, 0.1, 0.1])
        phases = np.array([0, 0, 0, 0, 0])
        sine_waves(awg, harmonics, amplitudes, phases, samples=1000, cycles=100, channel=1)


def example_custom_waveform():
    """Example: Send custom waveform data."""
    with AWG5014CController() as awg:
        # Create custom waveform
        samples = 1000
        t = np.linspace(0, 4*np.pi, samples)
        y = np.sin(t) + 0.5*np.sin(3*t + np.pi/4)
        awg.gen_wave(y, samples, channel=1)


if __name__ == "__main__":
    print("AWG5014C Python Controller")
    print("=" * 50)
    print("\nThis module provides Python equivalents for MATLAB AWG control scripts.")
    print("\nTo use:")
    print("  from awg5014c_controller import AWG5014CController, sine_wave, sine_waves")
    print("\nExample:")
    print("  with AWG5014CController('192.168.2.122::4011') as awg:")
    print("      sine_wave(awg, harmonic=93, amplitude=1.0)")
    print("\nNote: Camera functions are placeholders - implement based on your camera.")
    print("\nRequirements: pip install pyvisa numpy")
