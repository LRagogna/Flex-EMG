"""
Real-time EMG signal processor.

Filter chain (applied in order per sample):
  1. Bandpass  20–500 Hz  – keeps the physiological EMG band
  2. Notch     60 Hz      – removes power-line interference

Both filters are implemented as second-order sections (SOS) so they stay
numerically stable even at high orders.  State vectors (zi) are maintained
between calls so the filter is causal and works sample-by-sample.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt_zi, sosfilt, tf2sos

import config


class SignalProcessor:
    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        bandpass_low: float = config.BANDPASS_LOW_HZ,
        bandpass_high: float = config.BANDPASS_HIGH_HZ,
        bandpass_order: int = config.BANDPASS_ORDER,
        notch_freq: float = config.NOTCH_FREQ_HZ,
        notch_q: float = config.NOTCH_Q,
        adc_resolution: int = config.ADC_RESOLUTION,
        adc_vref: float = config.ADC_VREF,
    ) -> None:
        self.sample_rate = sample_rate
        self.adc_resolution = adc_resolution
        self.adc_vref = adc_vref
        self._adc_max = float((1 << adc_resolution) - 1)

        # ── Design bandpass (Butterworth SOS) ─────────────────────────────────
        nyq = sample_rate / 2.0
        if bandpass_high >= nyq:
            raise ValueError(
                f"bandpass_high ({bandpass_high} Hz) must be < Nyquist "
                f"({nyq} Hz) for sample_rate={sample_rate} Hz"
            )
        self._sos_bp = butter(
            bandpass_order,
            [bandpass_low / nyq, bandpass_high / nyq],
            btype="bandpass",
            output="sos",
        )

        # ── Design notch (IIR → convert to SOS) ───────────────────────────────
        b_notch, a_notch = iirnotch(notch_freq, notch_q, fs=sample_rate)
        self._sos_notch = tf2sos(b_notch, a_notch)

        # ── Initialise filter states (mid-scale so transients settle fast) ────
        mid_voltage = adc_vref / 2.0
        self._zi_bp = sosfilt_zi(self._sos_bp) * mid_voltage
        self._zi_notch = sosfilt_zi(self._sos_notch) * mid_voltage

    # ── Public API ─────────────────────────────────────────────────────────────

    def adc_to_voltage(self, raw_adc: int) -> float:
        """Convert a 12-bit ADC count to volts (0 – 3.3 V)."""
        return float(raw_adc) / self._adc_max * self.adc_vref

    def process_sample(self, raw_adc: int) -> dict:
        """
        Process one ADC sample through the filter chain.

        Parameters
        ----------
        raw_adc : int
            Raw 12-bit ADC value from the ESP32 (0–4095).

        Returns
        -------
        dict with keys:
            raw_voltage  – ADC count converted to volts, no filtering
            filtered     – volts after bandpass + notch
        """
        voltage = self.adc_to_voltage(raw_adc)

        # Step 1 – bandpass (20–500 Hz)
        out_bp, self._zi_bp = sosfilt(
            self._sos_bp, [voltage], zi=self._zi_bp
        )
        # Step 2 – notch (60 Hz)
        out_notch, self._zi_notch = sosfilt(
            self._sos_notch, out_bp, zi=self._zi_notch
        )

        return {
            "raw_voltage": round(voltage, 6),
            "filtered": round(float(out_notch[0]), 6),
        }

    def process_batch(self, raw_adc_array: list[int]) -> dict:
        """
        Process a batch of ADC samples (more efficient than one-by-one).

        Returns a dict with numpy arrays:
            raw_voltage  – shape (N,)
            filtered     – shape (N,)
        """
        voltages = np.array(raw_adc_array, dtype=float) / self._adc_max * self.adc_vref

        out_bp, self._zi_bp = sosfilt(self._sos_bp, voltages, zi=self._zi_bp)
        out_notch, self._zi_notch = sosfilt(self._sos_notch, out_bp, zi=self._zi_notch)

        return {
            "raw_voltage": voltages,
            "filtered": out_notch,
        }

    def reset(self) -> None:
        """Reset filter state (call if the data stream restarts)."""
        mid_voltage = self.adc_vref / 2.0
        self._zi_bp = sosfilt_zi(self._sos_bp) * mid_voltage
        self._zi_notch = sosfilt_zi(self._sos_notch) * mid_voltage
