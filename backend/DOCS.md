# FlexEMG Backend – Technical Reference

This document covers every module, class, and function in the backend.
It is written so that someone unfamiliar with the code can understand what
each piece does, how to configure it, and how the parts fit together.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Data flow](#2-data-flow)
3. [config.py – Global settings](#3-configpy--global-settings)
4. [signal_processor.py – EMG filter chain](#4-signal_processorpy--emg-filter-chain)
5. [esp32_client.py – ESP32 connection](#5-esp32_clientpy--esp32-connection)
6. [server.py – WebSocket broadcast server](#6-serverpy--websocket-broadcast-server)
7. [classifier.py – Gesture recognition](#7-classifierpy--gesture-recognition)
8. [main.py – Production entry point](#8-mainpy--production-entry-point)
9. [test.py – Simulation and training UI](#9-testpy--simulation-and-training-ui)
10. [esp32_firmware/emg_sender.ino – Arduino reference](#10-esp32_firmwareemg_senderino--arduino-reference)
11. [Workflow: simulation → real PCB](#11-workflow-simulation--real-pcb)

---

## 1. Project Overview

The backend receives raw 12-bit ADC samples from an ESP32 that is connected
to a forearm EMG circuit (0–3.3 V output). It applies a two-stage digital
filter to clean the signal, classifies the result into one of two gestures
using a trained machine-learning model, and streams everything in real time
over a WebSocket server to any connected client.

```
ESP32 (ADC)
    │  raw ADC counts  {"v": 2048}
    ▼
esp32_client.py   ← connects to ESP32 (WiFi WebSocket or USB serial)
    │  {adc: int, ts: float}
    ▼
signal_processor.py  ← bandpass 20–500 Hz  +  60 Hz notch
    │  {raw_voltage: float, filtered: float}
    ▼
classifier.py  (optional)  ← Hudgins features → LDA
    │  gesture: str, confidence: float
    ▼
server.py  ← WebSocket broadcast
    │  JSON frame to every connected client
    ▼
Dashboard / test.py viewer / any WebSocket client
```

---

## 2. Data Flow

### Sample journey, step by step

| Step | Where | What happens |
|------|--------|-------------|
| 1 | ESP32 firmware | ADC samples the forearm signal at 2000 Hz and sends `{"v": 2048}` over WebSocket or serial |
| 2 | `esp32_client.py` | Receives the message, parses it, timestamps it, and puts `{adc: 2048, ts: 1713456789.1}` on an asyncio queue |
| 3 | `signal_processor.py` | Converts the ADC count to volts, runs it through the bandpass filter (20–500 Hz), then the 60 Hz notch filter |
| 4 | `classifier.py` (optional) | Every 200 ms a new analysis window is ready; six Hudgins features are extracted and passed to the LDA model, which returns a gesture label and confidence |
| 5 | `server.py` | Broadcasts a JSON frame to all connected clients: `{"ts": …, "raw": …, "filtered": …, "gesture": …, "confidence": …}` |

### Broadcast frame format

```json
{
  "ts":         1713456789.123,
  "raw":        1.6504,
  "filtered":   0.0421,
  "gesture":    "Fist Squeeze",
  "confidence": 0.9712
}
```

`gesture` and `confidence` are `null` when the classifier is not running.

---

## 3. `config.py` – Global Settings

Single source of truth for every tunable parameter. Edit this file to
match your hardware before running the backend.

### Signal parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | `2000` Hz | Must match the ESP32 firmware's ADC rate. 2000 Hz gives a Nyquist frequency of 1000 Hz, safely above the 500 Hz EMG upper limit. |
| `ADC_RESOLUTION` | `12` bits | ESP32 ADC is 12-bit, so values range 0–4095. |
| `ADC_VREF` | `3.3` V | Reference voltage of the ADC. All raw counts are scaled against this. |

### Filter parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `BANDPASS_LOW_HZ` | `20.0` | Lower cutoff of the EMG bandpass filter. Removes slow drift and DC. |
| `BANDPASS_HIGH_HZ` | `500.0` | Upper cutoff. Attenuates high-frequency noise above the EMG band. |
| `BANDPASS_ORDER` | `4` | Butterworth filter order. Higher = steeper roll-off but more group delay. |
| `NOTCH_FREQ_HZ` | `60.0` | Frequency to notch out (power-line interference). Change to 50.0 in Europe. |
| `NOTCH_Q` | `30.0` | Quality factor of the notch. Higher Q = narrower notch. The current value achieves −254 dB at exactly 60 Hz. |

### ESP32 connection

| Variable | Default | Description |
|----------|---------|-------------|
| `ESP32_MODE` | `"websocket"` | Transport: `"websocket"` (WiFi) or `"serial"` (USB/UART). |
| `ESP32_WS_URI` | `"ws://192.168.1.100:81"` | IP and port of the ESP32 WebSocket server. Change this to match your ESP32's IP address on your network. |
| `ESP32_WS_RECONNECT_DELAY` | `3.0` s | How long to wait before attempting to reconnect after a dropped connection. |
| `ESP32_SERIAL_PORT` | `"/dev/tty.usbserial-0001"` | Serial port path (macOS). Windows uses `"COM3"`, Linux uses `"/dev/ttyUSB0"`. |
| `ESP32_BAUD_RATE` | `115200` | Must match `Serial.begin(115200)` in the Arduino sketch. |

### Backend server

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `"0.0.0.0"` | Listen on all network interfaces. Change to `"127.0.0.1"` to restrict to localhost. |
| `SERVER_PORT` | `8765` | Port that downstream clients (dashboards, notebooks) connect to. |

---

## 4. `signal_processor.py` – EMG Filter Chain

### What it does

Converts raw 12-bit ADC values to volts and applies a two-stage causal
digital filter in real time:

1. **Bandpass filter** (4th-order Butterworth, 20–500 Hz) – keeps only the
   physiological EMG frequency range and removes DC offset.
2. **Notch filter** (IIR, 60 Hz, Q = 30) – eliminates power-line interference.

Both filters are implemented as **second-order sections (SOS)** which are
numerically stable even at high orders. Running filter state (`zi`) is kept
between calls so each sample is processed causally with no look-ahead.

### Class: `SignalProcessor`

#### `__init__(...)`

Designs both filters using `scipy.signal` and initialises their state
vectors to mid-scale (1.65 V) so transients settle within a few milliseconds
rather than tens of milliseconds.

Parameters (all optional; defaults come from `config.py`):

| Parameter | Description |
|-----------|-------------|
| `sample_rate` | ADC sample rate in Hz |
| `bandpass_low` | Lower bandpass cutoff in Hz |
| `bandpass_high` | Upper bandpass cutoff in Hz |
| `bandpass_order` | Butterworth filter order |
| `notch_freq` | Notch center frequency in Hz |
| `notch_q` | Notch quality factor |
| `adc_resolution` | ADC bit depth (12 for ESP32) |
| `adc_vref` | ADC reference voltage in volts |

#### `adc_to_voltage(raw_adc: int) → float`

Converts a raw 12-bit ADC count (0–4095) to volts (0.0–3.3 V).

```python
voltage = raw_adc / 4095.0 * 3.3
```

#### `process_sample(raw_adc: int) → dict`

Processes **one sample** through the full filter chain. This is the
primary method used by the real-time pipeline.

Returns:
```python
{
    "raw_voltage": 1.6504,   # ADC count → volts, no filtering
    "filtered":    0.0421    # after bandpass + notch
}
```

Calling this in a tight loop is safe and efficient because `sosfilt` with
stored state (`zi`) is O(1) per sample.

#### `process_batch(raw_adc_array: list[int]) → dict`

Processes a list of samples in one vectorised NumPy call. More efficient
than calling `process_sample` in a Python loop when you have a large batch.

Returns:
```python
{
    "raw_voltage": np.ndarray,   # shape (N,)
    "filtered":    np.ndarray    # shape (N,)
}
```

#### `reset()`

Resets both filter state vectors to mid-scale. Call this whenever the
data stream restarts (e.g. after the ESP32 reconnects) to avoid a
transient spike from stale state.

---

## 5. `esp32_client.py` – ESP32 Connection

### What it does

Establishes and maintains the connection to the ESP32. It runs as an
asyncio coroutine so it never blocks the rest of the pipeline. If the
connection drops it waits `ESP32_WS_RECONNECT_DELAY` seconds and
automatically reconnects.

Each received message is parsed and placed on an asyncio queue as:
```python
{"adc": 2048, "ts": 1713456789.123}
```

### Supported message formats from the ESP32

```
2048          ← plain integer (simplest)
{"v": 2048}   ← JSON with key "v" (used by the reference Arduino sketch)
```

Anything else is silently ignored and logged at DEBUG level.

### `create_esp32_reader(queue, on_connect, on_disconnect) → coroutine`

The only public function. Returns the correct coroutine based on
`config.ESP32_MODE`:

- `"websocket"` → `_ws_reader` (connects to the ESP32's WebSocket server)
- `"serial"` → `_serial_reader` (reads from USB/UART)

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `queue` | `asyncio.Queue` where parsed samples are deposited |
| `on_connect` | Optional callback called each time a connection is established. The pipeline uses this to call `processor.reset()`. |
| `on_disconnect` | Optional callback called when the connection is lost |

**Usage in the pipeline:**
```python
task = asyncio.create_task(
    create_esp32_reader(raw_queue, on_connect=cb)
)
```

---

## 6. `server.py` – WebSocket Broadcast Server

### What it does

Maintains a set of connected WebSocket clients and fans out every processed
sample to all of them simultaneously. Clients connect and just listen;
the server never reads from them.

### Class: `BroadcastServer`

#### `broadcast(ts, raw_voltage, filtered, gesture=None, confidence=None)`

Serialises one sample frame to JSON and sends it to every connected client.
Stale connections that have already closed are silently removed from the
client set.

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ts` | `float` | Unix timestamp of the sample |
| `raw_voltage` | `float` | Raw ADC reading in volts |
| `filtered` | `float` | Filtered EMG voltage in volts |
| `gesture` | `str \| None` | Predicted gesture name (only when classifier is running) |
| `confidence` | `float \| None` | Prediction confidence 0–1 |

#### `serve()`

Starts the WebSocket server on `SERVER_HOST:SERVER_PORT` and runs
indefinitely. Called as an asyncio task.

### Connecting a client

Any WebSocket client can subscribe by connecting to
`ws://<host>:8765` (or whatever `SERVER_PORT` is set to).

Example with the Python `websockets` library:
```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        async for message in ws:
            frame = json.loads(message)
            print(frame["filtered"], frame.get("gesture"))

asyncio.run(main())
```

---

## 7. `classifier.py` – Gesture Recognition

### What it does

Trains and runs a two-class EMG gesture recogniser. The two classes are:

| Label | Gesture |
|-------|---------|
| `0` | **Fist Squeeze** — strong, broad-spectrum forearm activation |
| `1` | **Wrist Flexion Up** — lighter, moderate-frequency dorsal activation |

The pipeline is:
1. Collect 200 ms windows of filtered EMG (400 samples at 2000 Hz).
2. Extract 6 Hudgins time-domain features from each window.
3. Feed features into a `StandardScaler → LinearDiscriminantAnalysis` sklearn pipeline.

### Key constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CLASSES` | `{0: "Fist Squeeze", 1: "Wrist Flexion Up"}` | Label-to-name mapping |
| `WINDOW_SAMPLES` | 400 | Samples per analysis window (200 ms at 2 kHz) |
| `STEP_SAMPLES` | 200 | Samples between consecutive windows (100 ms, 50% overlap) |
| `MIN_SAMPLES_PER_CLASS` | 15 | Minimum windows required per class before training is allowed |
| `DEFAULT_MODEL_PATH` | `model.pkl` | Default save/load location |

### Hudgins feature set (6 features)

These are the standard time-domain features for real-time EMG pattern
recognition, proven in prosthetics research since 1993.

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| **MAV** | `mean(|x|)` | Overall signal amplitude / muscle effort |
| **RMS** | `sqrt(mean(x²))` | Signal power |
| **WL** | `sum(|x[i] - x[i-1]|)` | Waveform complexity / firing rate |
| **VAR** | `var(x)` | Signal variability |
| **ZC** | count of zero crossings | Frequency content estimate |
| **SSC** | count of slope sign changes | Frequency content, noise-robust |

### Class: `EMGClassifier`

#### `extract_features(window: np.ndarray) → np.ndarray`

Static method. Takes a 1-D array of `WINDOW_SAMPLES` filtered voltage
values and returns a feature vector of shape `(6,)`.

This is the same method used internally during both training and inference,
so it is the exact thing to call if you want to inspect what the model sees.

#### `add_training_sample(window, class_label)`

Adds one labelled window to the internal training buffer.
- `window` — 1-D array of `WINDOW_SAMPLES` filtered voltage samples
- `class_label` — `0` (Fist Squeeze) or `1` (Wrist Flexion Up)

Nothing is computed here; features are extracted immediately but the model
is not updated until `train()` is called.

#### `train() → dict`

Fits the StandardScaler + LDA pipeline on all collected windows.

Raises `ValueError` if any class has fewer than `MIN_SAMPLES_PER_CLASS` windows.

Returns a dict:
```python
{
    "accuracy":  0.975,    # training-set accuracy (0–1)
    "n_samples": 40,       # total windows used
    "counts":    {0: 20, 1: 20}
}
```

**Note:** Training accuracy is measured on the same data used to fit the
model (in-sample). It will be high by design. When the PCB is ready, collect
data across multiple sessions to get a better estimate of true generalisation.

#### `predict(window: np.ndarray) → tuple[int, float]`

Classifies one 200 ms window. Returns `(class_label, confidence)` where
`confidence` is the probability assigned to the predicted class by the LDA
model.

Raises `RuntimeError` if `train()` has not been called yet.

#### `save(path="model.pkl")`

Serialises the trained pipeline to disk using pickle.
Raises `RuntimeError` if the model has not been trained.

#### `load(path="model.pkl") → EMGClassifier` (class method)

Loads a previously saved model. Raises `FileNotFoundError` if the file
does not exist.

#### `clear_training_data()`

Discards all collected windows and resets the model to untrained state.
Use this to start a fresh training session without restarting the process.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_trained` | `bool` | Whether the model has been fitted |
| `train_accuracy` | `float \| None` | In-sample accuracy from the last `train()` call |
| `training_counts` | `dict[int, int]` | Windows collected per class so far |

---

## 8. `main.py` – Production Entry Point

### Normal operation

```bash
python main.py
```

Connects to the real ESP32, filters the signal, and streams processed data
to `ws://0.0.0.0:8765`. No classification is performed unless `--classify`
is passed.

### `--train` flag — interactive training session

```bash
python main.py --train
```

Runs a guided data collection session on the **real ESP32**. The terminal
prompts you to perform each gesture and records labelled windows from the
live filtered signal. When both classes have enough data, the model is
trained and saved to `model.pkl`.

**Step-by-step session:**

```
>> Ready to record 'Fist Squeeze'. Perform the movement and press Enter …
   [hold fist squeeze for 10 seconds]
>> Rest. Press Enter when ready for the next movement …
   [relax, then press Enter]
>> Ready to record 'Wrist Flexion Up'. Perform the movement and press Enter …
   [hold wrist flexion up for 10 seconds]
>> Rest. Press Enter when ready for the next movement …
   Training classifier …
   Training complete | accuracy=97.5% | samples=160 | counts={Fist Squeeze: 80, Wrist Flexion Up: 80}
   Model saved to model.pkl
```

The ESP32 must already be connected and streaming before this command is run.

### `--classify` flag — live gesture prediction

```bash
python main.py --classify
```

Loads `model.pkl` and attaches the classifier to the pipeline. Every
`STEP_SAMPLES` (100 ms) a prediction is made and included in the broadcast
frame:

```json
{"ts": 1.0, "raw": 1.65, "filtered": 0.042, "gesture": "Fist Squeeze", "confidence": 0.97}
```

Exits with an error if `model.pkl` does not exist. Run `--train` first.

### `--train` and `--classify` are mutually exclusive

You cannot pass both flags at once. Train first, then run with `--classify`.

### `pipeline(queue, processor, server, classifier=None)` — async function

The core data-processing coroutine. Continuously drains samples from the
queue, calls `processor.process_sample()`, optionally runs the classifier,
and broadcasts the result.

When `classifier` is not `None`, a rolling deque of `WINDOW_SAMPLES` filtered
values is maintained. Every `STEP_SAMPLES`-th call triggers a prediction,
and the result is attached to the next broadcast frame.

This function is also **imported by `test.py`** so both the simulation and
production paths use the same filtering code.

---

## 9. `test.py` – Simulation and Training UI

### Purpose

Runs the entire system — mock ESP32, backend pipeline, and live viewer — in
a single terminal with no hardware required. This is the primary tool for:

- Verifying the filter chain works before the PCB arrives
- Training the gesture classifier on simulated signals
- Testing prediction logic

### Running

```bash
python test.py           # live matplotlib plot (default)
python test.py --text    # scrolling table output (no display needed)
```

---

### Keyboard controls

The same keys work in both plot and text mode.

#### Movement simulation (no training effect)

| Key | Movement | Signal character |
|-----|----------|-----------------|
| `W` | Fingers Up | Medium amplitude, 30–160 Hz |
| `S` | Squeeze | High amplitude, 20–350 Hz |
| `A` | Wrist Left | Medium-low amplitude, 25–140 Hz |
| `D` | Wrist Right | Medium amplitude, 25–180 Hz |
| Space / any other | Release → Rest | Near-zero amplitude |

**Plot mode:** hold the key to keep the muscle active; release to relax.
The filtered signal amplitude rises with a ~30 ms attack and falls with
a ~500 ms decay, matching real muscle activation timing.

**Text mode:** each press triggers a 1.5-second burst and then
automatically returns to Rest (terminal input cannot detect key releases).

#### Training keys

| Key | Class recorded | Mock signal activated |
|-----|---------------|----------------------|
| Hold `1` | **Fist Squeeze** (class 0) | High-amplitude broadband signal (same as `S`) |
| Hold `2` | **Wrist Flexion Up** (class 1) | Medium-amplitude mid-frequency signal (same as `W`) |

While a training key is held, every completed 200 ms window of filtered
EMG is automatically labelled and added to the classifier's training
buffer. The progress bars in the classifier panel (or the Train column in
text mode) update in real time.

#### Classifier actions

| Key | Action |
|-----|--------|
| `T` | Train the classifier on all collected windows. If training succeeds, the model is automatically saved to `model.pkl`. |
| `P` | Toggle live prediction on/off. Predictions appear in the classifier panel or the Prediction column. |
| `C` | Clear all training data and reset the model to untrained state. |

---

### Plot mode — three panels

#### Top panel: Raw signal
Shows the ADC reading converted to volts with DC removed for display.
The regular 60 Hz oscillation from power-line interference is visible at rest.

#### Middle panel: Filtered signal
Shows the same signal after the bandpass and notch filters. The 60 Hz wave
is absent. When a movement key is held, the signal amplitude increases in
proportion to the movement's configured amplitude.

#### Bottom panel: Classifier status

Left column — training progress:
```
Fist Squeeze          [████████░░░░░░░░] 8/15
Wrist Flexion Up      [░░░░░░░░░░░░░░░░] 0/15

Model: not trained  (press T)
```

The bar for the class currently being recorded turns yellow. Once 15 windows
are collected for both classes and `T` is pressed, the bar changes to green
and accuracy is shown.

Right column — live prediction:
```
PREDICTION
Fist Squeeze
97% confidence
```

The gesture name is coloured red for Fist Squeeze and green for Wrist
Flexion Up.

---

### Text mode — live table

One summary row is printed per second:

```
  Time  Movement            Train     Raw RMS   Filt RMS   Reduction  Prediction
─────────────────────────────────────────────────────────────────────────────────
    1.2  Rest                 --       0.42368    0.01110       31.7 dB  --
    2.5  Squeeze           FistSq     0.53177    0.33322        4.1 dB  --
    4.9  Rest                 --       0.42315    0.01047       32.1 dB  Fist Squeeze (97%)
```

Columns:

| Column | Description |
|--------|-------------|
| Time | Elapsed seconds since first sample |
| Movement | Currently active mock movement |
| Train | Class being recorded (`FistSq` / `WristF` / `--`) |
| Raw RMS | RMS of the last 2000 raw samples (DC-removed). High value = 60 Hz interference present. |
| Filt RMS | RMS of the filtered signal. Tracks the actual muscle amplitude. |
| Reduction | `20 × log10(Raw RMS / Filt RMS)` in dB. ~32 dB at rest confirms the notch is working. |
| Prediction | Gesture name and confidence when prediction mode is on. |

---

### Internal components

#### `MOVEMENTS` dict

Defines how the mock ESP32 generator shapes its output for each key.
Each entry specifies:
- `amplitude` — target peak voltage of the muscle noise component (V)
- `freq_lo / freq_hi` — bandpass range that shapes the noise for this muscle group (Hz)

To add a new simulated movement, add an entry here and assign it a key.

#### `EMGGenerator` class

Produces one integer ADC sample per call for the mock ESP32.

The signal is composed of:
1. **1.65 V DC bias** — matches the mid-rail of the real 0–3.3 V circuit
2. **Gaussian noise** shaped through the movement's bandpass filter, scaled
   by a smoothed amplitude envelope
3. **0.6 V 60 Hz sine wave** — constant power-line interference

The amplitude envelope uses a first-order IIR smoother so activations
rise quickly (~30 ms) and relax slowly (~500 ms), matching real muscle timing.

#### `ClassificationState` class

Coordinates the sliding window buffer, training, and inference. This is the
glue between the incoming sample stream and the classifier.

Key methods:

| Method | Description |
|--------|-------------|
| `push_sample(filtered)` | Feed one filtered voltage sample. When the window is full and enough samples have stepped by, either records a training window or makes a prediction depending on current state. |
| `set_train_class(label, burst_secs)` | Called by keyboard handler when a training key is pressed. Sets the label that will be applied to the next windows. `burst_secs` makes it auto-expire (text mode). |
| `release_train_class()` | Called on key release (plot mode). Stops recording. |
| `train()` | Calls `clf.train()` and saves the result to `model.pkl` on success. Updates `status_msg` with the outcome. |
| `clear()` | Resets everything: clears training buffer, last prediction, status message. |
| `predict_on` | Boolean flag toggled by the `P` key. When `True` and the model is trained, predictions are generated every `STEP_SAMPLES`. |
| `last_pred` | `(label, confidence)` tuple of the most recent prediction, or `None`. |
| `counts` | Dict of `{class_label: n_windows}` collected so far. |

#### Thread safety

The mock signal generator runs in a background asyncio thread and reads
`_movement_key` via `get_movement()`. The keyboard handler (main thread or
keyboard daemon thread) writes it via `set_movement()`. Both are protected
by `_mv_lock`.

`ClassificationState._train_class` is similarly protected by an internal
lock because the keyboard daemon thread writes it while the viewer loop
(main thread) reads it in `push_sample()`. Everything else in
`ClassificationState` is only touched by the main thread.

---

## 10. `esp32_firmware/emg_sender.ino` – Arduino Reference

A reference Arduino sketch for the ESP32 that:
- Reads GPIO 34 (ADC1, input-only pin, no pull-up) at 2000 Hz
- Configures 12-bit resolution with 11 dB attenuation (full 0–3.3 V range)
- Sends each sample as `{"v": <adc_count>}` over WiFi WebSocket

Libraries required (Arduino Library Manager):
- **ArduinoWebsockets** by Gil Maimon
- **ArduinoJson** by Benoit Blanchon

Before flashing, edit these lines to match your network and backend:
```cpp
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
const char* BACKEND_HOST  = "192.168.1.50";   // IP of the machine running main.py
const uint16_t BACKEND_PORT = 8765;
```

---

## 11. Workflow: Simulation → Real PCB

### Phase 1 — Simulation (no hardware)

```bash
cd backend
pip install -r requirements.txt
python test.py
```

1. Hold `1` for several seconds to collect Fist Squeeze training data.
2. Hold `2` for several seconds to collect Wrist Flexion Up training data.
3. Watch the progress bars reach 15/15 for each class.
4. Press `T` to train. The model is saved to `model.pkl`.
5. Press `P` to enable prediction. Hold `1` or `2` and verify the label
   shown in the classifier panel matches the movement.

### Phase 2 — Real PCB arrives

1. Flash `esp32_firmware/emg_sender.ino` to your ESP32.
2. Update `config.py`:
   ```python
   ESP32_MODE    = "websocket"
   ESP32_WS_URI  = "ws://<your-esp32-ip>:8765"
   ```
3. Collect real training data:
   ```bash
   python main.py --train
   ```
   Follow the prompts: hold each movement for 10 seconds when asked.
   The model is saved to `model.pkl` when done.

4. Run the backend with live classification:
   ```bash
   python main.py --classify
   ```
   Every WebSocket frame now includes `"gesture"` and `"confidence"` fields.

5. To retrain at any time (e.g. after placing electrodes differently),
   delete `model.pkl` and run `python main.py --train` again.

### Switching between simulation and real hardware

The `classifier.py` API is identical in both cases. The only thing that
changes is where the filtered samples come from:

| Source | How samples arrive at the classifier |
|--------|--------------------------------------|
| Simulation (`test.py`) | `ClassificationState.push_sample()` is called in the viewer loop as frames arrive from the mock ESP32 |
| Real hardware (`main.py`) | The `pipeline()` coroutine maintains a sliding deque and calls `classifier.predict()` every `STEP_SAMPLES` |
