"""
FlexEMG backend – entry point.

Normal operation (real ESP32):
    python main.py

With live gesture classification (requires a saved model.pkl):
    python main.py --classify

Interactive training session (collect labelled data from the real ESP32
and train the classifier; model is saved to model.pkl when done):
    python main.py --train

Press Ctrl-C to stop.
"""

import argparse
import asyncio
import logging
import signal
import sys
from collections import deque
from pathlib import Path

import numpy as np

import config
from classifier import CLASSES, WINDOW_SAMPLES, STEP_SAMPLES, EMGClassifier
from esp32_client import create_esp32_reader
from signal_processor import SignalProcessor
from server import BroadcastServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Pipeline task ─────────────────────────────────────────────────────────────

async def pipeline(
    queue: asyncio.Queue,
    processor: SignalProcessor,
    server: BroadcastServer,
    classifier: EMGClassifier | None = None,
) -> None:
    """
    Drain the sample queue, filter each sample, and broadcast.

    When *classifier* is provided (--classify mode), a sliding window of
    filtered samples is maintained and a gesture prediction is attached to
    every STEP_SAMPLES-th broadcast frame.
    """
    sample_count  = 0
    log_interval  = config.SAMPLE_RATE      # log once per second

    # Sliding window for the classifier (only used when classifier is not None)
    clf_window: deque[float] = deque(maxlen=WINDOW_SAMPLES)
    clf_step:   int          = 0
    last_gesture: str | None        = None
    last_confidence: float | None   = None

    while True:
        item = await queue.get()
        result = processor.process_sample(item["adc"])

        gesture    = None
        confidence = None

        if classifier is not None:
            clf_window.append(result["filtered"])
            clf_step += 1
            if len(clf_window) == WINDOW_SAMPLES and clf_step % STEP_SAMPLES == 0:
                label, conf   = classifier.predict(np.array(clf_window))
                last_gesture    = CLASSES[label]
                last_confidence = conf
            gesture    = last_gesture
            confidence = last_confidence

        await server.broadcast(
            ts=item["ts"],
            raw_voltage=result["raw_voltage"],
            filtered=result["filtered"],
            gesture=gesture,
            confidence=confidence,
        )

        sample_count += 1
        if sample_count % log_interval == 0:
            pred_str = f"  gesture={last_gesture} ({last_confidence:.0%})" if last_gesture else ""
            logger.info(
                "Processed %d samples  |  raw=%.4f V  filtered=%.4f V%s",
                sample_count,
                result["raw_voltage"],
                result["filtered"],
                pred_str,
            )


# ── Interactive training session (real hardware) ──────────────────────────────

async def training_session(
    queue: asyncio.Queue,
    processor: SignalProcessor,
    duration_secs: float = 10.0,
) -> EMGClassifier:
    """
    Collect labelled training data from the real ESP32 and train the model.

    The user is prompted to perform each movement for *duration_secs* seconds.
    A brief rest period is recorded between movements so the classifier can
    distinguish activation from baseline.
    """
    from classifier import MIN_SAMPLES_PER_CLASS

    clf        = EMGClassifier()
    clf_window: deque[float] = deque(maxlen=WINDOW_SAMPLES)
    clf_step   = 0

    async def _collect_class(label: int, name: str) -> int:
        nonlocal clf_step
        n_windows  = 0
        n_samples  = 0
        deadline   = asyncio.get_event_loop().time() + duration_secs
        logger.info("Recording '%s' for %.0f s …", name, duration_secs)

        while asyncio.get_event_loop().time() < deadline:
            item = await queue.get()
            result = processor.process_sample(item["adc"])
            clf_window.append(result["filtered"])
            clf_step  += 1
            n_samples += 1

            if len(clf_window) == WINDOW_SAMPLES and clf_step % STEP_SAMPLES == 0:
                clf.add_training_sample(np.array(clf_window), label)
                n_windows += 1

        logger.info("  Collected %d windows for '%s'.", n_windows, name)
        return n_windows

    for label, name in CLASSES.items():
        input(f"\n  >> Ready to record '{name}'.  Perform the movement and press Enter …")
        await _collect_class(label, name)
        input("  >> Rest.  Press Enter when ready for the next movement …")

    logger.info("Training classifier …")
    result = clf.train()
    logger.info(
        "Training complete  |  accuracy=%.1f%%  samples=%d  counts=%s",
        result["accuracy"] * 100,
        result["n_samples"],
        {CLASSES[k]: v for k, v in result["counts"].items()},
    )
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("FlexEMG Backend")
    logger.info("  Sample rate  : %d Hz", config.SAMPLE_RATE)
    logger.info("  Bandpass     : %.0f – %.0f Hz", config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ)
    logger.info("  Notch        : %.0f Hz  (Q=%.1f)", config.NOTCH_FREQ_HZ, config.NOTCH_Q)
    logger.info("  ESP32 mode   : %s", config.ESP32_MODE)
    logger.info("  Server       : ws://%s:%d", config.SERVER_HOST, config.SERVER_PORT)
    logger.info("=" * 60)

    raw_queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    processor        = SignalProcessor()
    broadcast_server = BroadcastServer()

    def on_esp32_connect():
        processor.reset()
        logger.info("ESP32 connected – filter state reset.")

    def on_esp32_disconnect():
        logger.warning("ESP32 disconnected.")

    # ── Training mode ─────────────────────────────────────────────────────────
    if args.train:
        logger.info("TRAINING MODE – connect to ESP32 and follow the prompts.")
        esp32_task = asyncio.create_task(
            create_esp32_reader(raw_queue, on_esp32_connect, on_esp32_disconnect),
            name="esp32-reader",
        )
        # Wait until the ESP32 has connected and we have some data
        await asyncio.sleep(2.0)
        clf = await training_session(raw_queue, processor)
        clf.save()
        logger.info("Model saved to %s", EMGClassifier.DEFAULT_MODEL_PATH
                    if hasattr(EMGClassifier, "DEFAULT_MODEL_PATH") else "model.pkl")
        esp32_task.cancel()
        return

    # ── Classification mode ───────────────────────────────────────────────────
    clf: EMGClassifier | None = None
    if args.classify:
        try:
            clf = EMGClassifier.load()
            logger.info(
                "Loaded classifier  (training accuracy=%.1f%%)",
                (clf.train_accuracy or 0) * 100,
            )
            logger.info("Gestures: %s", list(CLASSES.values()))
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    # ── Normal (or classify) operation ────────────────────────────────────────
    esp32_task = asyncio.create_task(
        create_esp32_reader(raw_queue, on_esp32_connect, on_esp32_disconnect),
        name="esp32-reader",
    )
    pipeline_task = asyncio.create_task(
        pipeline(raw_queue, processor, broadcast_server, classifier=clf),
        name="pipeline",
    )
    server_task = asyncio.create_task(
        broadcast_server.serve(),
        name="broadcast-server",
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, esp32_task, pipeline_task, server_task)

    await asyncio.gather(esp32_task, pipeline_task, server_task)


def _shutdown(*tasks: asyncio.Task) -> None:
    logger.info("Shutting down …")
    for t in tasks:
        t.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlexEMG backend")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--train", action="store_true",
        help="Interactive training session: collect labelled EMG from the ESP32 "
             "and save a gesture classifier to model.pkl",
    )
    group.add_argument(
        "--classify", action="store_true",
        help="Load model.pkl and broadcast real-time gesture predictions "
             "alongside the filtered signal",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except asyncio.CancelledError:
        pass
    logger.info("Stopped.")
