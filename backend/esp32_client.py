"""
ESP32 connection layer.

Supports two transport modes (set ESP32_MODE in config.py):
  • "websocket" – ESP32 acts as a WebSocket server; this module is the client.
  • "serial"    – ESP32 streams data over USB-serial (UART).

In both modes the module reads raw ADC integers sent by the ESP32 and puts
them on an asyncio.Queue for the pipeline to consume.

Expected data format from the ESP32 (one per sample, newline-delimited JSON):
    {"v": 2048}
or plain integer text:
    2048

A bare integer is parsed as the ADC value.  JSON {"v": N} is also accepted.
"""

import asyncio
import json
import logging
import time
from typing import Callable

import config

logger = logging.getLogger(__name__)


# ── WebSocket client ───────────────────────────────────────────────────────────

async def _ws_reader(
    queue: asyncio.Queue,
    on_connect: Callable | None = None,
    on_disconnect: Callable | None = None,
) -> None:
    """Connect to the ESP32 WebSocket server and push samples onto *queue*."""
    import websockets  # imported here so serial-only users don't need it

    uri = config.ESP32_WS_URI
    while True:
        try:
            logger.info("Connecting to ESP32 at %s …", uri)
            async with websockets.connect(uri) as ws:
                logger.info("Connected to ESP32 WebSocket.")
                if on_connect:
                    on_connect()
                async for message in ws:
                    sample = _parse_message(message)
                    if sample is not None:
                        await queue.put({"adc": sample, "ts": time.time()})
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            logger.warning("ESP32 WebSocket error: %s", exc)
            if on_disconnect:
                on_disconnect()
        except asyncio.CancelledError:
            return

        logger.info(
            "Reconnecting in %.1f s …", config.ESP32_WS_RECONNECT_DELAY
        )
        await asyncio.sleep(config.ESP32_WS_RECONNECT_DELAY)


# ── Serial client ──────────────────────────────────────────────────────────────

async def _serial_reader(
    queue: asyncio.Queue,
    on_connect: Callable | None = None,
    on_disconnect: Callable | None = None,
) -> None:
    """Read newline-delimited samples from the ESP32 over UART/USB-serial."""
    import serial  # imported here so WebSocket-only users don't need it
    import serial.tools.list_ports

    port = config.ESP32_SERIAL_PORT
    baud = config.ESP32_BAUD_RATE

    while True:
        try:
            logger.info("Opening serial port %s @ %d baud …", port, baud)
            with serial.Serial(
                port, baud, timeout=config.ESP32_SERIAL_TIMEOUT
            ) as ser:
                logger.info("Serial port open.")
                if on_connect:
                    on_connect()
                while True:
                    # readline is blocking; run in executor to stay async-safe
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, ser.readline
                    )
                    if not line:
                        continue
                    sample = _parse_message(line.decode("utf-8", errors="ignore"))
                    if sample is not None:
                        await queue.put({"adc": sample, "ts": time.time()})
        except serial.SerialException as exc:
            logger.warning("Serial error: %s", exc)
            if on_disconnect:
                on_disconnect()
        except asyncio.CancelledError:
            return

        await asyncio.sleep(config.ESP32_WS_RECONNECT_DELAY)


# ── Message parser ─────────────────────────────────────────────────────────────

def _parse_message(raw: str) -> int | None:
    """
    Parse a message from the ESP32 into a raw ADC integer.

    Accepts:
        "2048\n"          → 2048
        '{"v": 2048}\n'   → 2048
    Returns None if the message cannot be parsed.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "v" in obj:
            return int(obj["v"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    logger.debug("Unrecognised ESP32 message: %r", text)
    return None


# ── Public factory ─────────────────────────────────────────────────────────────

def create_esp32_reader(
    queue: asyncio.Queue,
    on_connect: Callable | None = None,
    on_disconnect: Callable | None = None,
) -> asyncio.coroutines:
    """
    Return the correct coroutine for the configured ESP32_MODE.

    Usage::

        task = asyncio.create_task(
            create_esp32_reader(queue, on_connect=cb)
        )
    """
    mode = config.ESP32_MODE.lower()
    if mode == "websocket":
        return _ws_reader(queue, on_connect, on_disconnect)
    elif mode == "serial":
        return _serial_reader(queue, on_connect, on_disconnect)
    else:
        raise ValueError(
            f"Unknown ESP32_MODE '{config.ESP32_MODE}'. "
            "Choose 'websocket' or 'serial'."
        )
