"""Minimal ZMQ control channel for metrics/warm commands.

Falls back to a no-op stub when pyzmq is unavailable. Intended for
lightweight worker/control messaging without touching the streaming path.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    import zmq
except Exception:  # pragma: no cover
    zmq = None


class _NoopControl:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("pyzmq not installed; control channel disabled")

    def start(self) -> None:  # pragma: no cover - noop
        return


class ZMQControlServer:
    """Simple REP server handling JSON commands.

    Supported commands:
      - {"cmd": "metrics"}
      - {"cmd": "warm", "tokens": int}
    """

    def __init__(self, endpoint: str, on_request: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.endpoint = endpoint
        self.on_request = on_request

    def start(self) -> None:
        if zmq is None:
            _NoopControl().start()
            return
        thread = threading.Thread(target=self._serve, daemon=True)
        thread.start()
        logger.info("ZMQ control server listening on %s", self.endpoint)

    def _serve(self) -> None:
        assert zmq is not None
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        socket.bind(self.endpoint)
        while True:
            try:
                msg = socket.recv()
                try:
                    req = json.loads(msg.decode("utf-8"))
                except Exception:
                    socket.send_json({"error": "invalid_json"})
                    continue
                try:
                    resp = self.on_request(req)
                except Exception as exc:  # pragma: no cover - guard
                    logger.warning("Control handler error: %s", exc)
                    resp = {"error": str(exc)}
                socket.send_json(resp)
            except Exception as exc:  # pragma: no cover - keep alive
                logger.debug("Control server loop error: %s", exc)


def start_control_server(
    endpoint: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> None:
    server = ZMQControlServer(endpoint=endpoint, on_request=handler) if zmq else _NoopControl()
    server.start()


def send_control_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if zmq is None:
        raise RuntimeError("pyzmq not installed")
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(endpoint)
    socket.send_json(payload)
    return socket.recv_json()
