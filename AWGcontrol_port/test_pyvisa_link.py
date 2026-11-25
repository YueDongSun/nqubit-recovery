"""
PyVISA AWG5014C link and optional txt-upload smoke test.

Default behavior is intentionally non-destructive: list VISA resources, open the
requested AWG resources, query *IDN?, clear status, and disconnect. Use
--txt/--channel to upload one sequence txt, and add --run only when it is safe to
start the AWG output.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import struct
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_AWG1 = "192.168.2.122::4011"
DEFAULT_AWG2 = "192.168.2.110::3992"
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_SAMPLE_RATE = 1e9


def _import_pyvisa():
    try:
        import pyvisa  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pyvisa is not installed in this Python environment. Install it with "
            "`pip install pyvisa pyvisa-py` or run with the environment used by the AWG UI."
        ) from exc
    return pyvisa


def _candidate_resources(address: str) -> list[str]:
    address = address.strip()
    if not address.upper().startswith("TCPIP"):
        if "::" in address:
            parts = address.split("::")
            if len(parts) == 2 and parts[1].isdigit():
                try:
                    ipaddress.ip_address(parts[0])
                    return [
                        f"TCPIP0::{parts[0]}::{parts[1]}::SOCKET",
                        f"TCPIP::{parts[0]}::{parts[1]}::SOCKET",
                    ]
                except ValueError:
                    pass
        elif address.count(":") == 1:
            host, port = address.rsplit(":", 1)
            if port.isdigit():
                try:
                    ipaddress.ip_address(host)
                    return [
                        f"TCPIP0::{host}::{port}::SOCKET",
                        f"TCPIP::{host}::{port}::SOCKET",
                    ]
                except ValueError:
                    pass

    if "::" in address:
        return [address]

    try:
        ipaddress.ip_address(address)
    except ValueError:
        return [address]

    return [
        f"TCPIP0::{address}::inst0::INSTR",
        f"TCPIP0::{address}::INSTR",
        f"TCPIP::{address}::INSTR",
        address,
    ]


def _open_resource(resource_manager, address: str, timeout_ms: int):
    last_error: Optional[BaseException] = None
    for resource_name in _candidate_resources(address):
        try:
            instrument = resource_manager.open_resource(resource_name)
            instrument.timeout = timeout_ms
            instrument.write_termination = "\n"
            instrument.read_termination = "\n"
            return resource_name, instrument
        except Exception as exc:  # pyvisa backends raise backend-specific errors
            last_error = exc
    raise RuntimeError(f"Could not open {address!r}: {last_error}")


def _safe_query(instrument, command: str) -> str:
    try:
        return instrument.query(command).strip()
    except Exception as exc:
        return f"<query failed: {exc}>"


def _load_sequence_txt(path: Path, max_points: Optional[int]) -> tuple[list[float], list[int]]:
    analog: list[float] = []
    markers: list[int] = []

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["analog", "digital1", "digital2"]:
            raise ValueError(f"{path} must have header: analog,digital1,digital2")

        for row_index, row in enumerate(reader, start=2):
            if max_points is not None and len(analog) >= max_points:
                break
            if len(row) != 3:
                raise ValueError(f"{path}:{row_index} expected 3 columns, got {len(row)}")
            analog_value = max(-1.0, min(1.0, float(row[0])))
            digital1 = int(row[1])
            digital2 = int(row[2])
            analog.append(analog_value)
            markers.append((digital1 << 6) | (digital2 << 7))

    if not analog:
        raise ValueError(f"{path} did not contain any data rows")
    return analog, markers


def _binary_block(payload: bytes) -> bytes:
    byte_count = str(len(payload)).encode("ascii")
    return b"#" + str(len(byte_count)).encode("ascii") + byte_count + payload


def _upload_txt_waveform(
    instrument,
    txt_path: Path,
    waveform_name: str,
    channel: int,
    sample_rate: float,
    max_points: Optional[int],
) -> int:
    analog, markers = _load_sequence_txt(txt_path, max_points)
    sample_count = len(analog)

    payload = bytearray(sample_count * 5)
    offset = 0
    for analog_value, marker_value in zip(analog, markers):
        payload[offset : offset + 5] = struct.pack("<fB", analog_value, marker_value)
        offset += 5

    instrument.write("*CLS")
    instrument.write(f"SOURce1:FREQuency {sample_rate}")
    instrument.write(f'WLIST:WAVEFORM:DELETE "{waveform_name}"')
    instrument.write(f':WLIST:WAVEFORM:NEW "{waveform_name}", {sample_count}, REAL')

    command = f':WLIST:WAVEFORM:DATA "{waveform_name}",0,{sample_count},'.encode("ascii")
    old_termination = instrument.write_termination
    instrument.write_termination = ""
    try:
        instrument.write_raw(command + _binary_block(bytes(payload)))
        instrument.write_raw((old_termination or "\n").encode("ascii"))
    finally:
        instrument.write_termination = old_termination

    instrument.write(f'SOURCE{channel}:WAVEFORM "{waveform_name}"')
    return sample_count


def _test_one_address(resource_manager, address: str, args) -> bool:
    print(f"\n[{address}] opening...")
    resource_name, instrument = _open_resource(resource_manager, address, args.timeout_ms)
    print(f"[{address}] opened as {resource_name}")

    try:
        idn = _safe_query(instrument, "*IDN?")
        print(f"[{address}] *IDN? -> {idn}")

        instrument.write("*CLS")
        opc = _safe_query(instrument, "*OPC?")
        print(f"[{address}] *OPC? -> {opc}")

        if args.txt:
            sample_count = _upload_txt_waveform(
                instrument,
                args.txt,
                args.waveform_name,
                args.channel,
                args.sample_rate,
                args.max_points,
            )
            print(
                f"[{address}] uploaded {sample_count} points from {args.txt} "
                f"to waveform {args.waveform_name!r}, channel {args.channel}"
            )
            if args.enable_output:
                instrument.write(f"OUTPUT{args.channel}:STATE ON")
                print(f"[{address}] OUTPUT{args.channel}:STATE ON")
            if args.run:
                instrument.write("AWGControl:RMODe TRIGgered")
                instrument.write("AWGControl:RUN:IMMediate")
                print(f"[{address}] AWGControl:RUN:IMMediate sent")
            else:
                print(f"[{address}] upload complete; AWGControl:RUN not sent")

        return True
    finally:
        instrument.close()
        print(f"[{address}] closed")


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test PyVISA connectivity to AWG5014C instruments and optionally upload one txt waveform."
    )
    parser.add_argument(
        "--address",
        action="append",
        help="AWG IP address or VISA resource. Can be repeated. Defaults to the two front-panel AWGs.",
    )
    parser.add_argument(
        "--visa-library",
        default=None,
        help="Optional VISA backend, for example '@py' or a vendor VISA DLL path.",
    )
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--txt", type=Path, help="Optional analog,digital1,digital2 txt file to upload.")
    parser.add_argument("--channel", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--waveform-name", default="PYVISA_LINK_TEST")
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Limit uploaded txt points for smoke tests. Use 0 to upload the full file.",
    )
    parser.add_argument("--enable-output", action="store_true", help="Turn the selected output channel on after upload.")
    parser.add_argument("--run", action="store_true", help="Set triggered run mode and send AWGControl:RUN after upload.")
    parser.add_argument("--list-only", action="store_true", help="Only list VISA resources; do not open AWGs.")
    args = parser.parse_args(list(argv))

    if args.txt and not args.txt.exists():
        parser.error(f"--txt file does not exist: {args.txt}")
    if args.max_points == 0:
        args.max_points = None
    if args.run and not args.txt:
        parser.error("--run requires --txt so the test has an assigned waveform")
    if args.enable_output and not args.txt:
        parser.error("--enable-output requires --txt")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    pyvisa = _import_pyvisa()

    addresses = args.address or [DEFAULT_AWG1, DEFAULT_AWG2]
    resource_manager = pyvisa.ResourceManager(args.visa_library) if args.visa_library else pyvisa.ResourceManager()

    try:
        resources = resource_manager.list_resources()
        print("Detected VISA resources:")
        if resources:
            for resource in resources:
                print(f"  {resource}")
        else:
            print("  <none reported by backend>")

        if args.list_only:
            return 0

        ok = True
        for address in addresses:
            try:
                ok = _test_one_address(resource_manager, address, args) and ok
            except Exception as exc:
                ok = False
                print(f"[{address}] FAILED: {exc}")
        return 0 if ok else 1
    finally:
        resource_manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
