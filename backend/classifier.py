"""
EMG gesture classifier for FlexEMG.

Two gesture classes:
    0  –  Fist Squeeze        (strong, broad-spectrum forearm activation)
    1  –  Wrist Flexion Up    (lighter, moderate-frequency dorsal activation)

Feature extraction uses the Hudgins time-domain feature set, which is the
standard for real-time EMG pattern recognition in prosthetics research:
    MAV  – Mean Absolute Value
    RMS  – Root Mean Square
    WL   – Waveform Length
    VAR  – Variance
    ZC   – Zero Crossings  (with dead-band to reject noise floor)
    SSC  – Slope Sign Changes

The classifier is a scikit-learn Pipeline:
    StandardScaler  →  LinearDiscriminantAnalysis (LDA)

LDA is the standard choice for EMG: fast, works well with small datasets
(15–50 windows per class is sufficient), and gives a probability output.

──────────────────────────────────────────────────────────────────────────────
Hardware-agnostic API:  this module only accepts numpy arrays of filtered
voltage samples.  It does not care whether they come from the mock ESP32
in test.py or the real PCB going through main.py.  The training and
inference calls are identical in both cases.

Simulation workflow (test.py):
    - Hold key 1 to collect Fist Squeeze windows
    - Hold key 2 to collect Wrist Flexion Up windows
    - Press T to train
    - Press P to start live prediction

Real-hardware workflow (once PCB is ready):
    python main.py --train      # prompts user to perform each movement
    python main.py --classify   # loads model.pkl and predicts in real-time
──────────────────────────────────────────────────────────────────────────────
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

# ── Constants ──────────────────────────────────────────────────────────────────

CLASSES: dict[int, str] = {
    0: "Fist Squeeze",
    1: "Wrist Flexion Up",
}

# 200 ms analysis window at the configured sample rate.
# Standard for real-time EMG: short enough for low latency, long enough for
# stable frequency content.
WINDOW_SAMPLES: int = int(0.200 * config.SAMPLE_RATE)   # 400 samples @ 2 kHz

# Step between consecutive windows (50 % overlap).
STEP_SAMPLES: int = int(0.100 * config.SAMPLE_RATE)     # 200 samples @ 2 kHz

# Minimum windows required per class before training is allowed.
MIN_SAMPLES_PER_CLASS: int = 15

DEFAULT_MODEL_PATH: Path = Path(__file__).parent / "model.pkl"


# ── Classifier ─────────────────────────────────────────────────────────────────

class EMGClassifier:
    """
    Two-class EMG gesture recogniser.

    Typical usage
    -------------
    Training:
        clf = EMGClassifier()
        for window, label in training_data:
            clf.add_training_sample(window, label)   # label: 0 or 1
        result = clf.train()
        print(result["accuracy"])
        clf.save()

    Inference:
        clf = EMGClassifier.load()
        label, confidence = clf.predict(window)
    """

    def __init__(self) -> None:
        # StandardScaler normalises each feature to zero mean / unit variance,
        # which is important because EMG amplitude varies between subjects and
        # electrode placements.  LDA then finds the linear boundary.
        self._pipeline: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis()),
        ])
        self._X: list[np.ndarray] = []
        self._y: list[int]        = []
        self._is_trained: bool    = False
        self._train_accuracy: float | None = None

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_features(window: np.ndarray) -> np.ndarray:
        """
        Compute the 6-element Hudgins feature vector for one EMG window.

        Parameters
        ----------
        window : 1-D float array, length = WINDOW_SAMPLES
            Filtered EMG voltage values (V).

        Returns
        -------
        features : shape (6,)  –  [MAV, RMS, WL, VAR, ZC, SSC]
        """
        mav = float(np.mean(np.abs(window)))
        rms = float(np.sqrt(np.mean(window ** 2)))
        wl  = float(np.sum(np.abs(np.diff(window))))
        var = float(np.var(window))

        # Zero crossings: only count transitions that cross a small dead-band
        # so that baseline noise doesn't inflate the count on real hardware.
        thresh = 0.005   # V
        zc = int(np.sum(
            ((window[:-1] >= thresh)  & (window[1:] <= -thresh)) |
            ((window[:-1] <= -thresh) & (window[1:] >= thresh))
        ))

        # Slope sign changes: number of times the signal slope reverses.
        d   = np.diff(window)
        ssc = int(np.sum(
            ((d[:-1] > 0) & (d[1:] < 0)) |
            ((d[:-1] < 0) & (d[1:] > 0))
        ))

        return np.array([mav, rms, wl, var, float(zc), float(ssc)], dtype=float)

    # ── Training data ─────────────────────────────────────────────────────────

    def add_training_sample(self, window: np.ndarray, class_label: int) -> None:
        """
        Add one labelled EMG window to the training buffer.

        Parameters
        ----------
        window      : 1-D float array, length = WINDOW_SAMPLES
        class_label : 0 = Fist Squeeze, 1 = Wrist Flexion Up
        """
        if class_label not in CLASSES:
            raise ValueError(f"class_label must be one of {list(CLASSES)}, got {class_label}")
        self._X.append(self.extract_features(window))
        self._y.append(class_label)

    def clear_training_data(self) -> None:
        """Discard all collected training samples and reset the model."""
        self._X.clear()
        self._y.clear()
        self._is_trained     = False
        self._train_accuracy = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Fit the LDA pipeline on the current training buffer.

        Raises
        ------
        RuntimeError  if no training data has been collected.
        ValueError    if any class has fewer than MIN_SAMPLES_PER_CLASS windows.

        Returns
        -------
        dict:
            accuracy   – training-set accuracy  (0 – 1)
            n_samples  – total windows used
            counts     – {class_label: n_windows_for_that_class}
        """
        if not self._X:
            raise RuntimeError("No training data collected.  Hold key 1 or 2 to record samples.")

        X = np.array(self._X)
        y = np.array(self._y)

        counts = {label: int(np.sum(y == label)) for label in CLASSES}
        for label, n in counts.items():
            if n < MIN_SAMPLES_PER_CLASS:
                raise ValueError(
                    f"Not enough samples for '{CLASSES[label]}': "
                    f"have {n}, need at least {MIN_SAMPLES_PER_CLASS}."
                )

        self._pipeline.fit(X, y)
        self._train_accuracy = float(accuracy_score(y, self._pipeline.predict(X)))
        self._is_trained     = True

        return {
            "accuracy":  self._train_accuracy,
            "n_samples": int(len(y)),
            "counts":    counts,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, window: np.ndarray) -> tuple[int, float]:
        """
        Classify one EMG window.

        Returns
        -------
        (class_label, confidence)
            class_label : 0 or 1
            confidence  : probability of the predicted class (0 – 1)

        Raises
        ------
        RuntimeError if the classifier has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError("Call train() before predict().")
        features = self.extract_features(window).reshape(1, -1)
        label    = int(self._pipeline.predict(features)[0])
        proba    = float(np.max(self._pipeline.predict_proba(features)))
        return label, proba

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Serialize the trained model to *path* (default: model.pkl)."""
        if not self._is_trained:
            raise RuntimeError("Train the model before saving.")
        with open(path, "wb") as fh:
            pickle.dump({
                "pipeline":       self._pipeline,
                "is_trained":     self._is_trained,
                "train_accuracy": self._train_accuracy,
            }, fh)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "EMGClassifier":
        """
        Deserialize a saved model from *path*.

        Raises FileNotFoundError if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at '{path}'.  "
                "Run 'python main.py --train' or train in test.py first."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        inst                 = cls()
        inst._pipeline       = data["pipeline"]
        inst._is_trained     = data["is_trained"]
        inst._train_accuracy = data["train_accuracy"]
        return inst

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def train_accuracy(self) -> float | None:
        """Training-set accuracy, or None if not yet trained."""
        return self._train_accuracy

    @property
    def training_counts(self) -> dict[int, int]:
        """Number of collected windows per class label."""
        y = np.array(self._y) if self._y else np.array([], dtype=int)
        return {label: int(np.sum(y == label)) if len(y) > 0 else 0 for label in CLASSES}
