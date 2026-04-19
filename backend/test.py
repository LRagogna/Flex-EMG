"""
FlexEMG – all-in-one test script.

Runs the mock ESP32, backend pipeline, and live viewer in a single terminal.
No hardware required.

    python test.py           # live matplotlib plot  (default)
    python test.py --text    # scrolling text output (no display needed)

──────────────────────────────────────────────────────────────────────────────
Simulation keys  (same in both modes)
──────────────────────────────────────────────────────────────────────────────

  Movement simulation  (no training effect):
    W  –  Fingers Up      A  –  Wrist Left
    S  –  Squeeze         D  –  Wrist Right
    Space / any other  –  Release

  Training  (records labelled windows AND activates the matching mock signal):
    Hold  1  –  record  Fist Squeeze        (activates high-amplitude mock signal)
    Hold  2  –  record  Wrist Flexion Up    (activates medium-amplitude mock signal)

  Classifier actions:
    T  –  Train the classifier on collected windows
    P  –  Toggle live prediction on / off
    C  –  Clear all training data and reset the model

  Plot mode  : hold a key; release to return to rest.
  Text mode  : each keypress triggers a 1.5 s burst.

──────────────────────────────────────────────────────────────────────────────
When the real PCB is ready
──────────────────────────────────────────────────────────────────────────────
The classifier (classifier.py) is hardware-agnostic.  To train on real EMG:

    python main.py --train      # prompted recording session with the ESP32
    python main.py --classify   # live prediction in the production backend
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import websockets
import websockets.exceptions

# ── Patch config BEFORE importing backend modules ─────────────────────────────
import config
config.ESP32_WS_URI = "ws://localhost:8081"
config.ESP32_MODE   = "websocket"

from classifier import CLASSES, MIN_SAMPLES_PER_CLASS, STEP_SAMPLES, WINDOW_SAMPLES, EMGClassifier  # noqa: E402
from esp32_client import create_esp32_reader   # noqa: E402
from main import pipeline                      # noqa: E402
from server import BroadcastServer             # noqa: E402
from signal_processor import SignalProcessor   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-16s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


# ─────────────────────────────────────────────────────────────────────────────
# Movement definitions  (mock signal shapes)
# ─────────────────────────────────────────────────────────────────────────────

MOVEMENTS: dict[str, dict] = {
    "idle": {"name": "Rest",            "amplitude": 0.04, "freq_lo":  20, "freq_hi": 100},
    "w":    {"name": "Fingers Up",      "amplitude": 0.32, "freq_lo":  30, "freq_hi": 160},
    "s":    {"name": "Squeeze",         "amplitude": 0.58, "freq_lo":  20, "freq_hi": 350},
    "a":    {"name": "Wrist Left",      "amplitude": 0.26, "freq_lo":  25, "freq_hi": 140},
    "d":    {"name": "Wrist Right",     "amplitude": 0.30, "freq_lo":  25, "freq_hi": 180},
}

# The two training classes mapped to the movement keys that best simulate them.
# When the user holds 1 or 2, the mock uses this signal AND the classifier
# collects labelled windows.
TRAIN_KEY_TO_MOVEMENT = {
    "1": "s",   # Fist Squeeze    → high-amplitude broadband signal
    "2": "w",   # Wrist Flexion Up → medium-amplitude mid-frequency signal
}
TRAIN_KEY_TO_CLASS = {"1": 0, "2": 1}   # maps to CLASSES: {0: "Fist Squeeze", 1: "Wrist Flexion Up"}

ACTIVE_MOVEMENT_KEYS = {"w", "a", "s", "d"}
ALL_ACTIVE_KEYS      = ACTIVE_MOVEMENT_KEYS | set(TRAIN_KEY_TO_MOVEMENT)

TEXT_BURST_SECS = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Shared movement state  (asyncio thread reads; keyboard / UI thread writes)
# ─────────────────────────────────────────────────────────────────────────────

_mv_lock     = threading.Lock()
_movement_key: str   = "idle"
_burst_end:   float  = 0.0


def get_movement() -> str:
    with _mv_lock:
        global _movement_key, _burst_end
        if _burst_end and time.monotonic() > _burst_end:
            _movement_key = "idle"
            _burst_end    = 0.0
        return _movement_key


def set_movement(key: str, burst_secs: float = 0.0) -> None:
    with _mv_lock:
        global _movement_key, _burst_end
        _movement_key = key
        _burst_end    = (time.monotonic() + burst_secs) if burst_secs > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Classification state  (all in main thread; only _train_class needs a lock
#                         because the keyboard thread also writes it)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationState:
    """
    Manages the sliding window buffer, training data collection, and inference.

    The public interface is called from the main (viewer) thread.  The only
    exception is set_train_class() / release_train_class(), which are also
    called from the keyboard thread — protected by _lock.
    """

    def __init__(self) -> None:
        self.clf            = EMGClassifier()
        self._window        = deque(maxlen=WINDOW_SAMPLES)
        self._step          = 0
        self.predict_on     = False
        self.last_pred: tuple[int, float] | None = None   # (label, confidence)
        self.train_result:  dict | None = None             # from clf.train()
        self.status_msg:    str  = ""                      # e.g. error text
        self._lock          = threading.Lock()
        self._train_class:  int | None = None
        self._train_burst_end: float   = 0.0

    # Called from any thread ──────────────────────────────────────────────────

    def set_train_class(self, label: int, burst_secs: float = 0.0) -> None:
        with self._lock:
            self._train_class      = label
            self._train_burst_end  = (time.monotonic() + burst_secs) if burst_secs > 0 else 0.0

    def release_train_class(self) -> None:
        with self._lock:
            self._train_class      = None
            self._train_burst_end  = 0.0

    def get_train_class(self) -> int | None:
        with self._lock:
            if self._train_burst_end and time.monotonic() > self._train_burst_end:
                self._train_class     = None
                self._train_burst_end = 0.0
            return self._train_class

    # Called from main thread only ────────────────────────────────────────────

    def push_sample(self, filtered: float) -> None:
        """Feed one filtered sample; collect training data or predict if ready."""
        self._window.append(filtered)
        self._step += 1

        if len(self._window) < WINDOW_SAMPLES:
            return
        if self._step % STEP_SAMPLES != 0:
            return

        window = np.array(self._window)
        train_class = self.get_train_class()

        if train_class is not None:
            self.clf.add_training_sample(window, train_class)
        elif self.predict_on and self.clf.is_trained:
            label, conf  = self.clf.predict(window)
            self.last_pred = (label, conf)

    def train(self) -> None:
        try:
            self.train_result = self.clf.train()
            self.status_msg   = (
                f"Trained  accuracy={self.train_result['accuracy']:.1%}  "
                f"({self.train_result['n_samples']} windows)"
            )
            logger.info("[classifier]  %s", self.status_msg)
        except (RuntimeError, ValueError) as exc:
            self.status_msg = f"Train failed: {exc}"
            logger.warning("[classifier]  %s", self.status_msg)

    def save(self) -> None:
        try:
            self.clf.save()
            self.status_msg = "Model saved to model.pkl"
            logger.info("[classifier]  %s", self.status_msg)
        except Exception as exc:
            self.status_msg = f"Save failed: {exc}"

    def clear(self) -> None:
        self.clf.clear_training_data()
        self.last_pred    = None
        self.train_result = None
        self.status_msg   = "Training data cleared."
        logger.info("[classifier]  Training data cleared.")

    @property
    def counts(self) -> dict[int, int]:
        return self.clf.training_counts

    @property
    def is_trained(self) -> bool:
        return self.clf.is_trained

    @property
    def accuracy(self) -> float | None:
        return self.clf.train_accuracy


# Single shared instance used by both keyboard handlers and the viewer.
clf_state = ClassificationState()


# ─────────────────────────────────────────────────────────────────────────────
# EMG signal generator  (movement-aware, smooth envelope)
# ─────────────────────────────────────────────────────────────────────────────

class EMGGenerator:
    _ALPHA_ATTACK = 0.05
    _ALPHA_DECAY  = 0.003

    def __init__(self, seed: int = 42) -> None:
        from scipy.signal import butter, sosfilt_zi

        self._rng     = np.random.default_rng(seed)
        self._dt      = 1.0 / config.SAMPLE_RATE
        self._t       = 0.0
        self._adc_max = float((1 << config.ADC_RESOLUTION) - 1)
        self._filters: dict[str, dict] = {}

        nyq = config.SAMPLE_RATE / 2.0
        for key, params in MOVEMENTS.items():
            lo  = params["freq_lo"] / nyq
            hi  = min(params["freq_hi"], config.SAMPLE_RATE / 2.0 - 1.0) / nyq
            sos = butter(4, [lo, hi], btype="bandpass", output="sos")
            self._filters[key] = {"sos": sos, "zi": sosfilt_zi(sos) * 0.0}

        self._envelope: float = MOVEMENTS["idle"]["amplitude"]
        self._last_key: str   = "idle"

    def next_sample(self) -> int:
        from scipy.signal import sosfilt

        key        = get_movement()
        target_amp = MOVEMENTS[key]["amplitude"]

        alpha            = self._ALPHA_ATTACK if target_amp > self._envelope else self._ALPHA_DECAY
        self._envelope  += alpha * (target_amp - self._envelope)
        self._last_key   = key

        f = self._filters[key]
        shaped, f["zi"] = sosfilt(f["sos"], [self._rng.normal(0, 1.0)], zi=f["zi"])
        muscle = float(shaped[0]) * self._envelope

        interference = 0.6 * np.sin(2.0 * np.pi * 60.0 * self._t)
        voltage      = max(0.0, min(config.ADC_VREF, 1.65 + muscle + interference))
        self._t     += self._dt
        return int(round(voltage / config.ADC_VREF * self._adc_max))


# ─────────────────────────────────────────────────────────────────────────────
# Mock ESP32
# ─────────────────────────────────────────────────────────────────────────────

async def _mock_esp32_handler(websocket) -> None:
    logger.info("[mock-esp32]  backend connected")
    gen      = EMGGenerator()
    interval = 1.0 / config.SAMPLE_RATE
    try:
        while True:
            await websocket.send(json.dumps({"v": gen.next_sample()}))
            await asyncio.sleep(interval)
    except websockets.exceptions.ConnectionClosed:
        logger.info("[mock-esp32]  backend disconnected")


async def _run_mock_esp32() -> None:
    port = int(config.ESP32_WS_URI.rsplit(":", 1)[-1])
    async with websockets.serve(_mock_esp32_handler, "localhost", port):
        logger.info("[mock-esp32]  listening on ws://localhost:%d", port)
        await asyncio.Future()


# ─────────────────────────────────────────────────────────────────────────────
# Backend pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _run_backend() -> None:
    raw_queue        = asyncio.Queue(maxsize=4096)
    processor        = SignalProcessor()
    broadcast_server = BroadcastServer()

    def on_connect():
        processor.reset()
        logger.info("[backend]  ESP32 connected – filter state reset")

    await asyncio.gather(
        create_esp32_reader(raw_queue, on_connect),
        pipeline(raw_queue, processor, broadcast_server),
        broadcast_server.serve(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data collector
# ─────────────────────────────────────────────────────────────────────────────

async def _collect(out: queue.Queue) -> None:
    uri = f"ws://localhost:{config.SERVER_PORT}"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                logger.info("[viewer]  connected to backend at %s", uri)
                async for msg in ws:
                    out.put(json.loads(msg))
        except Exception as exc:
            logger.debug("[viewer]  %s – retrying …", exc)
            await asyncio.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Background asyncio thread
# ─────────────────────────────────────────────────────────────────────────────

async def _async_main(out: queue.Queue) -> None:
    mock_task = asyncio.create_task(_run_mock_esp32(), name="mock-esp32")
    await _wait_for_ws(config.ESP32_WS_URI, timeout=5.0)

    backend_task = asyncio.create_task(_run_backend(), name="backend")
    await _wait_for_ws(f"ws://localhost:{config.SERVER_PORT}", timeout=5.0)

    collector_task = asyncio.create_task(_collect(out), name="collector")
    await asyncio.gather(mock_task, backend_task, collector_task)


async def _wait_for_ws(uri: str, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with websockets.connect(uri):
                return
        except Exception:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"WebSocket server at {uri} did not open within {timeout}s")


def _start_background(out: queue.Queue) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_main(out))
    except Exception as exc:
        logger.error("Background thread error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard listener  (text mode)
# ─────────────────────────────────────────────────────────────────────────────

# Flags the keyboard thread sets; the viewer loop consumes them.
_action_lock  = threading.Lock()
_pending_train = False
_pending_clear = False
_pending_save  = False


def _set_action(name: str) -> None:
    global _pending_train, _pending_clear, _pending_save
    with _action_lock:
        if name == "train": _pending_train = True
        if name == "clear": _pending_clear = True
        if name == "save":  _pending_save  = True


def _consume_actions() -> list[str]:
    global _pending_train, _pending_clear, _pending_save
    actions = []
    with _action_lock:
        if _pending_train: actions.append("train"); _pending_train = False
        if _pending_clear: actions.append("clear"); _pending_clear = False
        if _pending_save:  actions.append("save");  _pending_save  = False
    return actions


def _start_keyboard_thread() -> bool:
    """
    Start a background thread that reads single keystrokes via termios.
    Returns True on success, False if stdin is not a TTY or termios is absent.
    """
    try:
        import termios
        import select as _select
    except ImportError:
        return False

    if not sys.stdin.isatty():
        return False

    def _reader():
        import termios as _t

        fd  = sys.stdin.fileno()
        old = _t.tcgetattr(fd)
        new = list(old)
        new[3] = new[3] & ~(_t.ECHO | _t.ICANON)
        new[6][_t.VMIN]  = 0
        new[6][_t.VTIME] = 0
        try:
            _t.tcsetattr(fd, _t.TCSANOW, new)
            while True:
                r, _, _ = _select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = os.read(fd, 1).decode("utf-8", errors="ignore").lower()
                if not ch:
                    continue

                if ch in TRAIN_KEY_TO_MOVEMENT:
                    mv    = TRAIN_KEY_TO_MOVEMENT[ch]
                    label = TRAIN_KEY_TO_CLASS[ch]
                    set_movement(mv, burst_secs=TEXT_BURST_SECS)
                    clf_state.set_train_class(label, burst_secs=TEXT_BURST_SECS)
                elif ch in ACTIVE_MOVEMENT_KEYS:
                    set_movement(ch, burst_secs=TEXT_BURST_SECS)
                elif ch == "t":
                    _set_action("train")
                elif ch == "p":
                    clf_state.predict_on = not clf_state.predict_on
                elif ch == "c":
                    clf_state.clear()
                elif ch == "q" or ch == "\x03":
                    os.kill(os.getpid(), __import__("signal").SIGINT)
                    break
                else:
                    set_movement("idle")
        finally:
            _t.tcsetattr(fd, _t.TCSADRAIN, old)

    t = threading.Thread(target=_reader, daemon=True, name="keyboard")
    t.start()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _progress_bar(n: int, total: int = MIN_SAMPLES_PER_CLASS, width: int = 16) -> str:
    """Return a text progress bar, e.g. '[████████░░░░░░░░] 8/15'."""
    filled = min(width, int(n / total * width)) if total > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {n}/{total}"


def _clf_status_lines() -> list[str]:
    """Return 4 human-readable status lines for the classification panel."""
    counts   = clf_state.counts
    trained  = clf_state.is_trained
    pred     = clf_state.last_pred
    pred_on  = clf_state.predict_on

    lines = [
        f"  {CLASSES[0]:<20} {_progress_bar(counts[0])}",
        f"  {CLASSES[1]:<20} {_progress_bar(counts[1])}",
        "",
        f"  Model: {'trained  accuracy={:.1%}'.format(clf_state.accuracy) if trained else 'not trained'}",
    ]

    if pred_on and trained and pred is not None:
        label, conf = pred
        lines.append(f"  Prediction: {CLASSES[label]}  ({conf:.0%})")
    elif pred_on and not trained:
        lines.append("  Prediction: train the model first (press T)")
    else:
        lines.append("  Prediction: off  (press P to enable)")

    if clf_state.status_msg:
        lines.append(f"  {clf_state.status_msg}")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Text viewer
# ─────────────────────────────────────────────────────────────────────────────

def run_text(out: queue.Queue) -> None:
    has_keyboard = _start_keyboard_thread()

    WINDOW   = config.SAMPLE_RATE
    raw_win  = deque(maxlen=WINDOW)
    filt_win = deque(maxlen=WINDOW)

    print("\nFlexEMG  –  live signal + gesture classifier  (Ctrl-C to stop)")
    if has_keyboard:
        print("  Signal keys:  W=Fingers Up  S=Squeeze  A=Wrist Left  D=Wrist Right")
        print("  Train  keys:  1=Fist Squeeze (record)  2=Wrist Flexion Up (record)")
        print("  Actions:      T=Train  P=Toggle predict  C=Clear data\n")
    else:
        print("  (keyboard unavailable – stdin is not a TTY)\n")

    hdr = (f"{'Time':>7}  {'Movement':<17}  {'Train':^7}  "
           f"{'Raw RMS':>9}  {'Filt RMS':>9}  {'Reduction':>10}  Prediction")
    print(hdr)
    print("─" * len(hdr))

    start = None
    n     = 0

    try:
        while True:
            # Consume any pending actions set by the keyboard thread.
            for action in _consume_actions():
                if action == "train":
                    clf_state.train()
                    if clf_state.is_trained:
                        clf_state.save()

            try:
                d = out.get(timeout=0.1)
            except queue.Empty:
                continue

            if start is None:
                start = d["ts"]
            n += 1

            clf_state.push_sample(d["filtered"])
            raw_win.append(d["raw"] - 1.65)
            filt_win.append(d["filtered"])

            if n % WINDOW != 0:
                continue

            elapsed  = d["ts"] - start
            raw_rms  = float(np.sqrt(np.mean(np.array(raw_win) ** 2)))
            filt_rms = float(np.sqrt(np.mean(np.array(filt_win) ** 2)))
            db       = 20.0 * np.log10(raw_rms / filt_rms) if filt_rms > 1e-9 else float("inf")

            mv_name   = MOVEMENTS[get_movement()]["name"]
            tc        = clf_state.get_train_class()
            train_tag = CLASSES[tc][:7] if tc is not None else "  --   "

            pred_str = "--"
            if clf_state.predict_on and clf_state.last_pred is not None:
                lbl, conf = clf_state.last_pred
                pred_str  = f"{CLASSES[lbl]} ({conf:.0%})"

            print(
                f"{elapsed:>7.1f}  {mv_name:<17}  {train_tag:^7}  "
                f"{raw_rms:>9.5f}  {filt_rms:>9.5f}  {db:>9.1f} dB  {pred_str}"
            )
    except KeyboardInterrupt:
        pass

    print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib viewer  (3 panels: raw | filtered | classifier status)
# ─────────────────────────────────────────────────────────────────────────────

def run_plot(out: queue.Queue) -> None:
    import matplotlib
    for backend in ("MacOSX", "TkAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("matplotlib not installed – falling back to text output.")
        run_text(out)
        return

    N        = int(2.0 * config.SAMPLE_RATE)
    raw_buf  = deque([0.0] * N, maxlen=N)
    filt_buf = deque([0.0] * N, maxlen=N)
    t_buf    = deque([0.0] * N, maxlen=N)

    BG    = "#0d1117"
    PAN   = "#161b22"
    RED   = "#ff6b6b"
    GRN   = "#56d364"
    YEL   = "#ffa657"
    BLU   = "#79c0ff"
    GRAY  = "#8b949e"
    WHT   = "#e6edf3"
    DGRAY = "#21262d"

    fig, (ax_raw, ax_filt, ax_clf) = plt.subplots(
        3, 1,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 3, 2]},
    )
    fig.patch.set_facecolor(BG)

    # ── Signal panels ─────────────────────────────────────────────────────────
    line_raw,  = ax_raw.plot([], [], color=RED, linewidth=0.8,
                              label="Raw  (DC removed)")
    line_filt, = ax_filt.plot([], [], color=GRN, linewidth=0.8,
                               label="Filtered  (20–500 Hz bandpass + 60 Hz notch)")

    for ax, title in [
        (ax_raw,  "RAW signal  —  60 Hz interference visible"),
        (ax_filt, "FILTERED  —  60 Hz removed, EMG band preserved"),
    ]:
        ax.set_facecolor(PAN)
        ax.set_ylim(-2.2, 2.2)
        ax.set_ylabel("Voltage (V)", color=GRAY)
        ax.set_title(title, fontsize=9, color=GRAY, loc="left")
        ax.tick_params(colors=GRAY)
        ax.grid(True, alpha=0.15, color=GRAY)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3, labelcolor=GRAY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Movement label (inside ax_raw so blit can own it)
    txt_movement = ax_raw.text(
        0.5, 0.92, f"Movement:  {MOVEMENTS['idle']['name']}",
        ha="center", va="top", fontsize=11, fontweight="bold", color=WHT,
        transform=ax_raw.transAxes, zorder=5,
    )

    ax_filt.set_xlabel("Time (s)", color=GRAY)

    # ── Classifier panel ──────────────────────────────────────────────────────
    ax_clf.set_facecolor(DGRAY)
    ax_clf.set_xlim(0, 1)
    ax_clf.set_ylim(0, 1)
    ax_clf.set_xticks([])
    ax_clf.set_yticks([])
    for spine in ax_clf.spines.values():
        spine.set_edgecolor("#30363d")

    # Static keybinding hint
    ax_clf.text(
        0.5, 0.97,
        "Train: hold 1 = Fist Squeeze   hold 2 = Wrist Flexion Up"
        "   |   T = Train   P = Predict   C = Clear",
        ha="center", va="top", fontsize=8, color=GRAY,
        transform=ax_clf.transAxes,
    )
    ax_clf.axhline(0.85, color="#30363d", linewidth=0.8)

    # Class 0 progress  (left column)
    txt_c0_label = ax_clf.text(0.03, 0.75, CLASSES[0],
        ha="left", va="center", fontsize=9, color=RED,
        transform=ax_clf.transAxes, fontweight="bold")
    txt_c0_bar   = ax_clf.text(0.03, 0.62, _progress_bar(0),
        ha="left", va="center", fontsize=9, color=RED,
        transform=ax_clf.transAxes, fontfamily="monospace")

    # Class 1 progress  (left column)
    txt_c1_label = ax_clf.text(0.03, 0.48, CLASSES[1],
        ha="left", va="center", fontsize=9, color=GRN,
        transform=ax_clf.transAxes, fontweight="bold")
    txt_c1_bar   = ax_clf.text(0.03, 0.35, _progress_bar(0),
        ha="left", va="center", fontsize=9, color=GRN,
        transform=ax_clf.transAxes, fontfamily="monospace")

    # Model accuracy  (left column, bottom)
    txt_accuracy = ax_clf.text(0.03, 0.15, "Model: not trained",
        ha="left", va="center", fontsize=9, color=GRAY,
        transform=ax_clf.transAxes)

    # Vertical divider
    ax_clf.axvline(0.55, color="#30363d", linewidth=0.8)

    # Prediction label  (right column)
    txt_pred_header = ax_clf.text(0.58, 0.75, "PREDICTION",
        ha="left", va="center", fontsize=8, color=GRAY,
        transform=ax_clf.transAxes, fontweight="bold")
    txt_pred_gesture = ax_clf.text(0.58, 0.52, "--",
        ha="left", va="center", fontsize=16, color=WHT,
        transform=ax_clf.transAxes, fontweight="bold")
    txt_pred_conf = ax_clf.text(0.58, 0.28, "",
        ha="left", va="center", fontsize=10, color=BLU,
        transform=ax_clf.transAxes)
    txt_status = ax_clf.text(0.58, 0.10, "",
        ha="left", va="center", fontsize=7.5, color=YEL,
        transform=ax_clf.transAxes)

    plt.tight_layout()

    # ── Key events ────────────────────────────────────────────────────────────
    def _on_key_press(event):
        key = (event.key or "").lower()

        if key in TRAIN_KEY_TO_MOVEMENT:
            mv    = TRAIN_KEY_TO_MOVEMENT[key]
            label = TRAIN_KEY_TO_CLASS[key]
            set_movement(mv)
            clf_state.set_train_class(label)
            txt_movement.set_text(f"Recording:  {CLASSES[label]}  [TRAINING]")
            txt_movement.set_color(YEL)

        elif key in ACTIVE_MOVEMENT_KEYS:
            set_movement(key)
            txt_movement.set_text(f"Movement:  {MOVEMENTS[key]['name']}  [ACTIVE]")
            txt_movement.set_color(YEL)

        elif key == "t":
            clf_state.train()
            if clf_state.is_trained:
                clf_state.save()

        elif key == "p":
            clf_state.predict_on = not clf_state.predict_on

        elif key == "c":
            clf_state.clear()

    def _on_key_release(event):
        key = (event.key or "").lower()

        if key in TRAIN_KEY_TO_MOVEMENT:
            clf_state.release_train_class()
            set_movement("idle")
            txt_movement.set_text(f"Movement:  {MOVEMENTS['idle']['name']}")
            txt_movement.set_color(WHT)

        elif key in ACTIVE_MOVEMENT_KEYS and key == get_movement():
            set_movement("idle")
            txt_movement.set_text(f"Movement:  {MOVEMENTS['idle']['name']}")
            txt_movement.set_color(WHT)

    fig.canvas.mpl_connect("key_press_event",   _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    # ── Animation ─────────────────────────────────────────────────────────────
    def _update(_frame):
        # Drain the sample queue
        drained = 0
        while drained < 300:
            try:
                d = out.get_nowait()
            except queue.Empty:
                break
            raw_buf.append(d["raw"] - 1.65)
            filt_buf.append(d["filtered"])
            t_buf.append(d["ts"])
            clf_state.push_sample(d["filtered"])
            drained += 1

        # Update signal plots
        if t_buf:
            t0    = t_buf[0]
            t_arr = [t - t0 for t in t_buf]
            line_raw.set_data(t_arr,  list(raw_buf))
            line_filt.set_data(t_arr, list(filt_buf))
            ax_raw.set_xlim(t_arr[0], t_arr[-1] + 0.001)
            ax_filt.set_xlim(t_arr[0], t_arr[-1] + 0.001)

        # Update classifier panel
        counts = clf_state.counts
        txt_c0_bar.set_text(_progress_bar(counts[0]))
        txt_c1_bar.set_text(_progress_bar(counts[1]))

        if clf_state.is_trained:
            txt_accuracy.set_text(
                f"Model trained  accuracy={clf_state.accuracy:.1%}"
            )
            txt_accuracy.set_color(GRN)
        else:
            txt_accuracy.set_text("Model: not trained  (press T)")
            txt_accuracy.set_color(GRAY)

        if clf_state.predict_on and clf_state.is_trained:
            if clf_state.last_pred is not None:
                lbl, conf = clf_state.last_pred
                color = RED if lbl == 0 else GRN
                txt_pred_gesture.set_text(CLASSES[lbl])
                txt_pred_gesture.set_color(color)
                txt_pred_conf.set_text(f"{conf:.0%} confidence")
            else:
                txt_pred_gesture.set_text("waiting…")
                txt_pred_gesture.set_color(GRAY)
                txt_pred_conf.set_text("")
        elif not clf_state.predict_on:
            txt_pred_gesture.set_text("off")
            txt_pred_gesture.set_color(GRAY)
            txt_pred_conf.set_text("press P to enable")
        else:
            txt_pred_gesture.set_text("train first")
            txt_pred_gesture.set_color(GRAY)
            txt_pred_conf.set_text("")

        txt_status.set_text(clf_state.status_msg)

        # Highlight the bar for the class currently being recorded
        tc = clf_state.get_train_class()
        txt_c0_bar.set_color(YEL if tc == 0 else RED)
        txt_c1_bar.set_color(YEL if tc == 1 else GRN)

        return (line_raw, line_filt, txt_movement,
                txt_c0_bar, txt_c1_bar, txt_accuracy,
                txt_pred_gesture, txt_pred_conf, txt_status)

    ani = animation.FuncAnimation(   # noqa: F841
        fig, _update, interval=50, blit=True, cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FlexEMG all-in-one test")
    parser.add_argument("--text", action="store_true",
                        help="Text output instead of a plot")
    args = parser.parse_args()

    out: queue.Queue = queue.Queue(maxsize=16_000)

    bg = threading.Thread(target=_start_background, args=(out,), daemon=True)
    bg.start()
    time.sleep(1.5)

    print()
    print("=" * 62)
    print("  FlexEMG test  –  all services running in this process")
    print(f"  Mock ESP32  →  ws://localhost:{int(config.ESP32_WS_URI.rsplit(':', 1)[-1])}")
    print(f"  Backend     →  ws://localhost:{config.SERVER_PORT}")
    print("  Gesture classes:")
    for k, v in CLASSES.items():
        mk = [tk for tk, c in TRAIN_KEY_TO_CLASS.items() if c == k][0]
        print(f"    Key {mk}  →  {v}")
    print("=" * 62)

    if args.text:
        run_text(out)
    else:
        run_plot(out)


if __name__ == "__main__":
    main()
