"""Deterministic preprocessing shared by training and inference."""
from __future__ import annotations

import io
import numpy as np
from scipy import signal
from scipy.io import wavfile

TARGET_AUDIO_RATE = 16_000
AUDIO_SECONDS = 8
AUDIO_FEATURE_VERSION = "welch-logbands-v1"
VIBRATION_FEATURE_VERSION = "triaxial-time-normalized-psd-v1"


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        raise ValueError("Signal contains no finite samples")
    return values


def decode_wav(payload: bytes) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(io.BytesIO(payload))
    if data.dtype.kind in "iu":
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = data.astype(np.float64) / scale
    else:
        data = data.astype(np.float64)
    data = _finite(data)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return int(rate), data[:, 0]


def preprocess_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float waveform at 16 kHz, center-trimmed or zero-padded to 8 s."""
    if sample_rate < 1:
        raise ValueError("Audio sample rate is invalid")
    x = _finite(samples).mean(axis=1)
    if sample_rate != TARGET_AUDIO_RATE:
        divisor = np.gcd(sample_rate, TARGET_AUDIO_RATE)
        x = signal.resample_poly(x, TARGET_AUDIO_RATE // divisor, sample_rate // divisor)
    length = TARGET_AUDIO_RATE * AUDIO_SECONDS
    if len(x) > length:
        start = (len(x) - length) // 2
        x = x[start:start + length]
    elif len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    x = x - np.mean(x)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def audio_features(samples: np.ndarray, sample_rate: int = TARGET_AUDIO_RATE) -> np.ndarray:
    """Fixed 8-second mono preprocessing and log spectral/time features."""
    x = preprocess_audio(samples, sample_rate)
    freqs, psd = signal.welch(x, fs=TARGET_AUDIO_RATE, nperseg=4096,
                              noverlap=2048, scaling="spectrum")
    edges = np.geomspace(30, TARGET_AUDIO_RATE / 2, 25)
    band_power = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        band_power.append(float(np.mean(psd[mask])) if np.any(mask) else 0.0)
    rms = float(np.sqrt(np.mean(x * x)))
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    peak = float(freqs[int(np.argmax(psd))])
    power = psd / (psd.sum() + 1e-12)
    entropy = float(-np.sum(power * np.log(power + 1e-12)))
    return np.asarray([rms, zcr, centroid, peak, entropy,
                       *np.log10(np.asarray(band_power) + 1e-12)], dtype=np.float32)


def preprocess_vibration(samples: np.ndarray) -> np.ndarray:
    """Clean triaxial acceleration, subtract DC/gravity and retain its sample rate."""
    x = _finite(samples)
    if x.shape[1] < 3:
        raise ValueError("Three acceleration axes are required")
    x = x[:, :3]
    x = x - np.median(x, axis=0, keepdims=True)
    return x.astype(np.float32)


def vibration_features(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Triaxial statistics plus PSD in Nyquist-normalized bands."""
    if sample_rate <= 0:
        raise ValueError("Vibration sample rate is invalid")
    x = preprocess_vibration(samples)
    if len(x) < 16:
        raise ValueError("Not enough vibration samples")
    features: list[float] = []
    for axis in range(3):
        y = x[:, axis].astype(np.float64)
        rms = np.sqrt(np.mean(y * y))
        peak = np.max(np.abs(y))
        features.extend([float(np.std(y)), float(rms), float(peak),
                         float(np.ptp(y)), float(peak / (rms + 1e-12)),
                         float(np.mean(((y - y.mean()) / (y.std() + 1e-12)) ** 4))])
        f, p = signal.welch(y, fs=sample_rate,
                            nperseg=min(1024, len(y)), scaling="spectrum")
        normalized = f / (sample_rate / 2)
        edges = np.linspace(0, 1, 9)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (normalized >= low) & (normalized < high)
            features.append(float(np.log10(np.mean(p[mask]) + 1e-15)) if np.any(mask) else -15.0)
    return np.asarray(features, dtype=np.float32)


def vibration_feature_names() -> list[str]:
    names = []
    for axis in ("x", "y", "z"):
        names.extend([f"{axis}_{n}" for n in ("std", "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis")])
        names.extend(f"{axis}_log_psd_band_{i}" for i in range(8))
    return names
