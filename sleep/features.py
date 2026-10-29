"""
features.py — band-power features per 30s epoch (delta/theta/alpha/
sigma/beta), computed from a plain 1D numpy array + sampling rate.
Pure numpy/scipy (no mne dependency) so this is testable on synthetic
signals without the real Sleep-EDF data (see sleep/README.md).
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 15.0),
    'beta': (15.0, 30.0),
}

DEFAULT_EPOCH_SECONDS = 30.0


def band_power(signal: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    """Power in `band` (Hz), integrated from the Welch PSD of `signal`."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return 0.0

    nperseg = min(len(signal), max(2, int(sfreq * 2)))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)

    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def epoch_band_powers(signal: np.ndarray, sfreq: float, epoch_seconds: float = DEFAULT_EPOCH_SECONDS) -> list[dict]:
    """
    Splits `signal` into non-overlapping `epoch_seconds`-long windows and
    returns one {'delta':p, 'theta':p, 'alpha':p, 'sigma':p, 'beta':p}
    dict per epoch. Trailing samples that don't fill a full epoch are
    dropped.
    """
    signal = np.asarray(signal, dtype=float)
    samples_per_epoch = int(sfreq * epoch_seconds)
    if samples_per_epoch <= 0:
        raise ValueError('sfreq * epoch_seconds must be positive')

    n_epochs = len(signal) // samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        window = signal[i * samples_per_epoch:(i + 1) * samples_per_epoch]
        epochs.append({band: band_power(window, sfreq, freq_range) for band, freq_range in BANDS.items()})
    return epochs
