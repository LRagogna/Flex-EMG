/*
 * FlexEMG – ESP32 firmware (reference)
 *
 * Reads the EMG signal from GPIO 34 (ADC1_CH6) at 1000 Hz and streams
 * raw 12-bit ADC values to the Python backend via WebSocket.
 *
 * Data format sent:  {"v": 2048}\n
 *
 * Libraries required (install via Arduino Library Manager):
 *   • ArduinoWebsockets  by Gil Maimon  (v0.5.x)
 *   • ArduinoJson        by Benoit Blanchon  (v6.x)
 *
 * Board: "ESP32 Dev Module" (or your specific variant)
 *
 * IMPORTANT: GPIO 34 is input-only (no internal pull-up).
 *            Keep the EMG signal within 0–3.3 V at all times.
 */

#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>

using namespace websockets;

// ── Configuration ────────────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// Backend Python host and port (SERVER_PORT in config.py)
// The ESP32 connects TO the backend, which acts as the WS server.
// If you want the ESP32 to be the WS server instead, swap the roles and
// update ESP32_MODE = "websocket" + ESP32_WS_URI in config.py accordingly.
const char* BACKEND_HOST  = "192.168.1.50";   // ← IP of the machine running main.py
const uint16_t BACKEND_PORT = 8765;

const int   EMG_PIN        = 34;   // ADC1_CH6 – input-only pin
const int   SAMPLE_RATE_HZ = 2000; // must match config.SAMPLE_RATE
const int   SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ;  // 1000 µs

// ── Globals ───────────────────────────────────────────────────────────────────
WebsocketsClient wsClient;
bool wsConnected = false;

unsigned long lastSampleUs = 0;

// ── WiFi helpers ──────────────────────────────────────────────────────────────
void connectWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected. IP: ");
    Serial.println(WiFi.localIP());
}

// ── WebSocket helpers ─────────────────────────────────────────────────────────
void connectWebSocket() {
    Serial.println("Connecting to backend WebSocket…");
    wsConnected = wsClient.connect(BACKEND_HOST, BACKEND_PORT, "/");
    if (wsConnected) {
        Serial.println("WebSocket connected.");
    } else {
        Serial.println("WebSocket connection failed – will retry.");
    }
}

// ── Arduino setup / loop ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // Configure ADC: 12-bit, 0–3.3 V (attenuation 11 dB covers full range)
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    pinMode(EMG_PIN, INPUT);

    connectWiFi();

    wsClient.onEvent([](WebsocketsEvent event, String data) {
        if (event == WebsocketsEvent::ConnectionOpened) {
            wsConnected = true;
        } else if (event == WebsocketsEvent::ConnectionClosed) {
            wsConnected = false;
            Serial.println("WebSocket closed.");
        }
    });

    connectWebSocket();
}

void loop() {
    // Maintain WebSocket connection
    if (!wsConnected) {
        delay(3000);
        connectWebSocket();
        return;
    }
    wsClient.poll();

    // Sample at exactly SAMPLE_RATE_HZ using micros() timer
    unsigned long now = micros();
    if (now - lastSampleUs < SAMPLE_INTERVAL_US) {
        return;  // not time yet
    }
    lastSampleUs = now;

    int adcValue = analogRead(EMG_PIN);   // 0–4095

    // Build JSON payload
    StaticJsonDocument<32> doc;
    doc["v"] = adcValue;
    char buf[32];
    serializeJson(doc, buf);

    wsClient.send(buf);
}
