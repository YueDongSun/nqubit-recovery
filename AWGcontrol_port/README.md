# AWG instrument control port

PyVISA and WebSocket transports for the Tektronix AWG5014C arbitrary
waveform generators of the W-state experiment.

| File | Purpose |
|------|---------|
| `awg5014c_controller.py` | High-level controller: waveforms, sequences, run control |
| `awg5014c_transport.py` | VISA / WebSocket resource abstraction |
| `pyvisa_websocket_server.py` | Bridge exposing a bench AWG to the LAN over WebSocket |
| `set_front_3qubit.py` / `set_front_4qubit.py` | Front-panel equivalents: load the full W/R channel layout |
| `test_pyvisa_link.py` / `test_websocket_link.py` | Non-destructive link smoke tests |

Non-destructive checks:

```sh
python test_pyvisa_link.py --visa-library "@py" --list-only
python test_websocket_link.py --url ws://<ip>:8765/
```

Instrument addresses appear as defaults (`192.168.2.122::4011` and
`192.168.2.110::3992`, the two bench AWGs); pass any address as an
argument. The WebSocket bridge binds `0.0.0.0:8765` by default — run it on
a bench network or pass `--host` to restrict it.

`MATLAB_TO_PYTHON_REFERENCE.txt` documents the command-by-command
equivalence with the laboratory's previous MATLAB control scripts for the
same instrument.
