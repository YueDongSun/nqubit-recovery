"""
PyVISA-over-WebSocket AWG link smoke test.

This script is non-destructive by default: it connects to a websocket bridge
that forwards PyVISA operations, sends *IDN?, optionally *OPC?, and closes. Use
it for the AWG/control box that only exposes a websocket endpoint.

Example:
    python AWGcontrol_port/test_websocket_link.py --url ws://192.168.2.130:5025/
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

from awg5014c_transport import PyvisaWebSocketResource


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PyVISA-over-WebSocket connectivity.")
    parser.add_argument("--url", required=True, help="WebSocket endpoint, e.g. ws://192.168.2.130:5025/")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--idn-command", default="*IDN?")
    parser.add_argument("--skip-opc", action="store_true", help="Do not send *OPC? after *IDN?.")
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Extra query command to send after the basic checks. Can be repeated.",
    )
    return parser.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    resource = PyvisaWebSocketResource(args.url, timeout_ms=args.timeout_ms)
    print(f"[websocket] opening {args.url}")
    resource.connect()
    try:
        print(f"[websocket] {args.idn_command} -> {resource.query(args.idn_command)}")
        if not args.skip_opc:
            print(f"[websocket] *OPC? -> {resource.query('*OPC?')}")
        for command in args.command:
            print(f"[websocket] {command} -> {resource.query(command)}")
        return 0
    finally:
        resource.close()
        print("[websocket] closed")


if __name__ == "__main__":
    raise SystemExit(main())
