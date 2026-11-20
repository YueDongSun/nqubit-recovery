"""Write the archived E5 device configuration for an external calibration histogram."""

import argparse
import json
from pathlib import Path

DEVICE = {'efficiency': {'intrinsic': 0.15,
                'aod': 0.7,
                'cavity': 0.7,
                'detector': 0.7,
                'measured_mean_idler': 0.04,
                'eta_signal': 0.2},
 'signal_rate': 0.01,
 'g_si_nominal': 10.0,
 'background': None,
 'background_idler': None,
 'storage_time_us': 40.0,
 'n_max': 8,
 'phase_sign': -1,
 'points_per_round': 256,
 'uniform_idler_map': False,
 'priors': {'log_a_std': 0.7,
            'nb2_alpha': [0.3, 1.5],
            'thermal_dlcz_signal_rate': [0.015, 0.04],
            'multimode_nbar': [0.01, 0.04],
            'e1_sigma': [0.0, 0.05],
            'e2_jitter': [0.0, 0.2],
            'e3_step': [0.0, 0.1],
            'e3_corr_length': [1.0, 5.0],
            'e4_tau_us': [10.0, 60.0],
            'e4_tau_spread': 0.2,
            'e5_lambda': [0.0, 1.2],
            'e6_background': [0.00068, 0.014],
            'e1_strong_sigma': [0.5, 3.0]}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, help="CSV containing Signal Absolute and Idler Absolute rows")
    parser.add_argument("--output", type=Path, default=Path("nqrec/configs/round6_device.json"))
    args = parser.parse_args()
    calibration = args.calibration.resolve(strict=True)
    config = {**DEVICE, "calibration_csv": str(calibration)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
