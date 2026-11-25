# Experiment control core

This directory is the closed-loop experiment control software of the
DLCZ-type cold-atom W-state experiment: sequence generation, AWG upload,
serial data acquisition orchestration, density-matrix reconstruction, and
the optional ML feedback step.

```
IDLE -> SEQUENCE_READY -> WAITING_AWG -> DATA_UI_OPEN -> COLLECTING ->
RECONSTRUCTING -> INFERRING_DELTA -> GUARDING -> APPLYING_DELTA ->
REGENERATING_SEQUENCE -> ...
```

Any failed step stops the loop; the runner never advances after a failure.

## Layout

| Path | Purpose |
|------|---------|
| `run_experiment_ui.py` | Unified PyQt entry point |
| `experiment_control/` | Runner state machine, UI, AWG uploader, data source, delta manager, feedback bridges (v6-v9), fail-closed V10 campaign authority (`v10/`) |
| `qubit_recon_files/` | Sequence generation, phase calculation, Fourier guard, density reconstruction, matrix I/O, `system_specs.py` |
| `data_ui/` | Serial acquisition UIs (3Q/4Q), ML diagnosis, snapshot export |
| `cali_phase/`, `cali_retrival_rate/` | Phase / retrieval-rate calibration utilities |

The accompanying packages `../AWGcontrol_port/` (PyVISA/WebSocket AWG
transport), `../VerilogHDL/` (FPGA histogram firmware) and the `../ml_recon`
support modules complete the control loop.

## Running

Python 3.10+ with `PyQt5 numpy matplotlib pyserial pyvisa pyvisa-py
websocket-client websockets`, plus optional `torch scikit-learn`.

```sh
python system_control/run_experiment_ui.py
```

The AWG endpoints accept a plain IP, `ip::port`, or a full VISA resource
string. Bench instrument defaults (the AWG5014C pair and the serial port)
are pre-filled for the documented 3Q/4Q wiring; every one of them is
overridable in the UI or on the command line.

Physical constants (3Q 75-operation 15x15 grid, 4Q 256-operation 16x16
grid, 1 GHz sampling, 2.5 MHz tone spacing) live in
`qubit_recon_files/system_specs.py` — read them there rather than restating
them.

## Data flow

1. Choose a 3Q/4Q W target (amplitudes + phases) -> phase/amplitude matrices.
2. Generate AWG waveform text files (`analog,digital1,digital2` columns).
3. `fourier_sequence_guard.py` validates the tones on the 1 GHz sample grid.
4. Upload to the Tektronix AWG5014C pair (PyVISA or the WebSocket bridge).
5. Collect FPGA histogram counts over serial.
6. Reconstruct the density matrix (calibrated path by default for
   feedback/automation).
7. Optional ML feedback writes additive deltas; the next cycle overlays
   them via `matrix_utils.apply_additive_delta()`.

Versioned calibration files (`qubit_recon_files/phase_Cali*.txt`,
`qubit_recon_files/matrix/retrieval_efficiency_*.txt`) ship with the code;
everything else under `matrix/`, `ml_delta/`, `AWG_Sequence/`, `data/` and
`outputs/` is generated at run time.
