# ── Signal parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE: int = 2000          # Hz  (must match ESP32 ADC sample rate)
                                 # 2000 Hz gives Nyquist = 1000 Hz, safely above
                                 # the 500 Hz EMG upper band limit.
ADC_RESOLUTION: int = 12         # bits (ESP32 ADC is 12-bit → 0–4095)
ADC_VREF: float = 3.3            # V

# ── Bandpass filter  (keep EMG band 20–500 Hz) ────────────────────────────────
BANDPASS_LOW_HZ: float = 20.0
BANDPASS_HIGH_HZ: float = 500.0
BANDPASS_ORDER: int = 4          # Butterworth SOS order

# ── Notch filter  (reject 60 Hz power-line noise) ─────────────────────────────
NOTCH_FREQ_HZ: float = 60.0
NOTCH_Q: float = 30.0            # quality factor – higher = narrower notch

# ── ESP32 connection ──────────────────────────────────────────────────────────
# Set ESP32_MODE to "websocket" (WiFi) or "serial" (USB/UART).
ESP32_MODE: str = "websocket"

#   WebSocket mode – the ESP32 acts as a WS server; this backend is the client.
ESP32_WS_URI: str = "ws://192.168.1.100:81"
ESP32_WS_RECONNECT_DELAY: float = 3.0   # seconds between reconnect attempts

#   Serial mode
ESP32_SERIAL_PORT: str = "/dev/tty.usbserial-0001"  # macOS; adjust for your OS
ESP32_BAUD_RATE: int = 115200
ESP32_SERIAL_TIMEOUT: float = 1.0

# ── Backend WebSocket server (downstream clients connect here) ────────────────
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8765
