"""Expose a local PyVISA resource through a websocket bridge.

Run this on the machine that can open the instrument with PyVISA. The main UI
can then use the websocket URL as an AWG endpoint, while the actual instrument
I/O still happens through PyVISA on this server side.

Example:
    python AWGcontrol_port/pyvisa_websocket_server.py --resource 192.168.2.122::4011 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from typing import Any, Iterable, Optional


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge websocket clients to a local PyVISA resource.")
    parser.add_argument("--resource", required=True, help="PyVISA resource opened by the server.")
    parser.add_argument("--visa-library", default=None, help="Optional VISA backend, e.g. '@py'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    return parser.parse_args(list(argv))


def _open_instrument(args):
    from awg5014c_transport import open_instrument_resource

    instrument = open_instrument_resource(
        args.resource,
        timeout_ms=args.timeout_ms,
        visa_library=args.visa_library,
    )
    instrument.read_termination = "\n"
    return instrument


def _handle_operation(instrument, payload: dict[str, Any]) -> dict[str, Any]:
    op = payload.get("op")
    if "write_termination" in payload:
        instrument.write_termination = payload["write_termination"]
    if "read_termination" in payload:
        instrument.read_termination = payload["read_termination"]

    if op == "query":
        return {"ok": True, "response": instrument.query(str(payload["command"]))}
    if op == "write":
        instrument.write(str(payload["command"]))
        return {"ok": True, "result": None}
    if op == "write_raw":
        data = base64.b64decode(payload["data_b64"])
        instrument.write_raw(data)
        return {"ok": True, "result": None}
    return {"ok": False, "error": f"Unsupported operation: {op!r}"}


async def _serve(args) -> None:
    import websockets

    instrument = _open_instrument(args)
    lock = asyncio.Lock()

    async def handler(websocket):
        async for message in websocket:
            try:
                payload = json.loads(message)
                async with lock:
                    response = _handle_operation(instrument, payload)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            await websocket.send(json.dumps(response))

    try:
        async with websockets.serve(handler, args.host, args.port, max_size=None):
            print(f"PyVISA websocket bridge listening on ws://{args.host}:{args.port}/")
            print(f"Serving VISA resource: {args.resource}")
            await asyncio.Future()
    finally:
        instrument.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
