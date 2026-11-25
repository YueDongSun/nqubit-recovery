"""Transport helpers for AWG5014C PyVISA-style links.

Addresses beginning with ws:// or wss:// use a PyVISA-over-WebSocket bridge:
the local upload code still sees a resource with write/query/write_raw methods,
and each operation is forwarded as a websocket JSON request. All other
addresses use PyVISA, with plain IP addresses expanded to common TCPIP VISA
resource names.
"""

from __future__ import annotations

import base64
import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Optional


def is_websocket_address(address: str) -> bool:
    value = address.strip().lower()
    return value.startswith("ws://") or value.startswith("wss://")


def _parse_ip_port_shorthand(address: str) -> Optional[tuple[str, str]]:
    value = address.strip()
    if value.upper().startswith("TCPIP"):
        return None

    if "::" in value:
        parts = value.split("::")
        if len(parts) != 2:
            return None
        host, port = parts
    elif value.count(":") == 1:
        host, port = value.rsplit(":", 1)
    else:
        return None

    host = host.strip()
    port = port.strip()
    if not port.isdigit():
        return None

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return None
    return host, port


def candidate_visa_resources(address: str) -> list[str]:
    address = address.strip()
    ip_port = _parse_ip_port_shorthand(address)
    if ip_port is not None:
        host, port = ip_port
        return [
            f"TCPIP0::{host}::{port}::SOCKET",
            f"TCPIP::{host}::{port}::SOCKET",
        ]

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


def open_instrument_resource(address: str, timeout_ms: int = 10000, visa_library: Optional[str] = None):
    if is_websocket_address(address):
        return PyvisaWebSocketResource(address, timeout_ms=timeout_ms).connect()
    return VisaScpiResource(address, timeout_ms=timeout_ms, visa_library=visa_library).connect()


@dataclass
class VisaScpiResource:
    address: str
    timeout_ms: int = 10000
    visa_library: Optional[str] = None

    def __post_init__(self):
        self.write_termination = "\n"
        self.read_termination = "\n"
        self.timeout = self.timeout_ms
        self.resource_name = None
        self._rm = None
        self._instrument = None

    def connect(self):
        import pyvisa

        self._rm = pyvisa.ResourceManager(self.visa_library) if self.visa_library else pyvisa.ResourceManager()
        last_error = None
        for resource_name in candidate_visa_resources(self.address):
            try:
                self._instrument = self._rm.open_resource(resource_name)
                self.resource_name = resource_name
                self._instrument.timeout = self.timeout_ms
                self._instrument.write_termination = self.write_termination
                self._instrument.read_termination = self.read_termination
                return self
            except Exception as exc:
                last_error = exc
        self.close()
        raise RuntimeError(f"Could not open VISA resource {self.address!r}: {last_error}")

    def write(self, command):
        if isinstance(command, bytes):
            return self.write_raw(command)
        self._sync_terminations()
        return self._instrument.write(command)

    def write_raw(self, payload: bytes):
        self._sync_terminations()
        return self._instrument.write_raw(payload)

    def query(self, command: str) -> str:
        self._sync_terminations()
        return self._instrument.query(command)

    def close(self):
        if self._instrument is not None:
            self._instrument.close()
            self._instrument = None
        if self._rm is not None:
            self._rm.close()
            self._rm = None

    def _sync_terminations(self):
        if self._instrument is None:
            raise ConnectionError("VISA resource is not connected")
        self._instrument.write_termination = self.write_termination
        self._instrument.read_termination = self.read_termination
        self._instrument.timeout = self.timeout


@dataclass
class PyvisaWebSocketResource:
    url: str
    timeout_ms: int = 10000

    def __post_init__(self):
        self.write_termination = "\n"
        self.read_termination = "\n"
        self.timeout = self.timeout_ms
        self._socket = None

    def connect(self):
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError(
                "websocket-client is not installed. Install it with `pip install websocket-client`."
            ) from exc

        self._socket = websocket.create_connection(self.url, timeout=self.timeout_ms / 1000.0)
        return self

    def write(self, command):
        if isinstance(command, bytes):
            return self.write_raw(command)
        return self._request(
            {
                "op": "write",
                "command": str(command),
                "write_termination": self.write_termination,
            }
        )

    def write_raw(self, payload: bytes):
        return self._request(
            {
                "op": "write_raw",
                "data_b64": base64.b64encode(payload).decode("ascii"),
            }
        )

    def query(self, command: str) -> str:
        response = self._request(
            {
                "op": "query",
                "command": str(command),
                "write_termination": self.write_termination,
                "read_termination": self.read_termination,
            }
        )
        return "" if response is None else str(response).strip()

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _ensure_connected(self):
        if self._socket is None:
            raise ConnectionError("WebSocket resource is not connected")

    def _request(self, payload: dict[str, Any]):
        self._ensure_connected()
        self._socket.send(json.dumps(payload))
        raw_response = self._socket.recv()
        if isinstance(raw_response, bytes):
            raw_response = raw_response.decode("utf-8", errors="replace")
        try:
            response = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            return raw_response
        if response.get("ok") is False:
            raise RuntimeError(response.get("error") or "remote PyVISA websocket operation failed")
        for key in ("result", "response", "data"):
            if key in response:
                return response[key]
        return None


# Backward-compatible alias for callers/tests that imported the first draft name.
WebSocketScpiResource = PyvisaWebSocketResource
