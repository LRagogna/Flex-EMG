"""
WebSocket broadcast server.

Downstream clients (e.g. a web dashboard, desktop app, or Jupyter notebook)
connect here to receive real-time processed EMG data.

Each message broadcast to clients is newline-delimited JSON:
    {
        "ts":       1713456789.123,   // Unix timestamp (seconds, float)
        "raw":      1.65,             // raw ADC → voltage (V)
        "filtered": 0.042             // after bandpass + notch (V)
    }
"""

import asyncio
import json
import logging

import websockets
import websockets.exceptions

import config

logger = logging.getLogger(__name__)


class BroadcastServer:
    def __init__(self) -> None:
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

    # ── WebSocket handler ──────────────────────────────────────────────────────

    async def _handler(
        self, websocket: websockets.WebSocketServerProtocol
    ) -> None:
        async with self._lock:
            self._clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info("Client connected: %s  (total: %d)", client_addr, len(self._clients))
        try:
            # Keep the connection alive; we only send, never receive data here.
            await websocket.wait_closed()
        finally:
            async with self._lock:
                self._clients.discard(websocket)
            logger.info(
                "Client disconnected: %s  (total: %d)",
                client_addr,
                len(self._clients),
            )

    # ── Broadcast ──────────────────────────────────────────────────────────────

    async def broadcast(
        self,
        ts: float,
        raw_voltage: float,
        filtered: float,
        gesture: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """
        Send one processed sample to every connected client.

        Optional gesture / confidence fields are included only when a trained
        classifier is running (main.py --classify or via test.py prediction mode).
        JSON schema:
            {"ts": float, "raw": float, "filtered": float,
             "gesture": str|null, "confidence": float|null}
        """
        if not self._clients:
            return

        payload_dict: dict = {"ts": round(ts, 6), "raw": raw_voltage, "filtered": filtered}
        if gesture is not None:
            payload_dict["gesture"]    = gesture
            payload_dict["confidence"] = round(confidence, 4) if confidence is not None else None
        payload = json.dumps(payload_dict)

        async with self._lock:
            targets = list(self._clients)

        # Fan-out; silently drop stale connections.
        results = await asyncio.gather(
            *[_safe_send(ws, payload) for ws in targets],
            return_exceptions=True,
        )
        # Clean up any sockets that errored.
        dead = {ws for ws, r in zip(targets, results) if isinstance(r, Exception)}
        if dead:
            async with self._lock:
                self._clients -= dead

    # ── Server lifecycle ───────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start the WebSocket server and run until cancelled."""
        async with websockets.serve(
            self._handler,
            config.SERVER_HOST,
            config.SERVER_PORT,
        ):
            logger.info(
                "EMG WebSocket server listening on ws://%s:%d",
                config.SERVER_HOST,
                config.SERVER_PORT,
            )
            await asyncio.Future()  # run forever


async def _safe_send(ws: websockets.WebSocketServerProtocol, payload: str) -> None:
    try:
        await ws.send(payload)
    except websockets.exceptions.ConnectionClosed:
        pass
