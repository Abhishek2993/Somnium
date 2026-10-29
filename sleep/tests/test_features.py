import numpy as np

from features import BANDS, band_power, epoch_band_powers

SFREQ = 100.0


def _sine(freq: float, duration_seconds: float, sfreq: float = SFREQ) -> np.ndarray:
    t = np.arange(0, duration_seconds, 1 / sfreq)
    return np.sin(2 * np.pi * freq * t)


def test_band_power_empty_or_tiny_signal_is_zero():
    assert band_power(np.array([]), SFREQ, BANDS['delta']) == 0.0
    assert band_power(np.array([1.0]), SFREQ, BANDS['delta']) == 0.0


def test_band_power_delta_sine_scores_highest_in_delta():
    signal = _sine(2.0, duration_seconds=10)  # 2 Hz is in delta (0.5-4)
    powers = {band: band_power(signal, SFREQ, freq_range) for band, freq_range in BANDS.items()}
    assert max(powers, key=powers.get) == 'delta'


def test_band_power_alpha_sine_scores_highest_in_alpha():
    signal = _sine(10.0, duration_seconds=10)  # 10 Hz is in alpha (8-12)
    powers = {band: band_power(signal, SFREQ, freq_range) for band, freq_range in BANDS.items()}
    assert max(powers, key=powers.get) == 'alpha'


def test_band_power_beta_sine_scores_highest_in_beta():
    signal = _sine(20.0, duration_seconds=10)  # 20 Hz is in beta (15-30)
    powers = {band: band_power(signal, SFREQ, freq_range) for band, freq_range in BANDS.items()}
    assert max(powers, key=powers.get) == 'beta'


def test_epoch_band_powers_splits_into_correct_number_of_epochs():
    signal = _sine(2.0, duration_seconds=90)  # 90s / 30s epochs = 3 epochs
    epochs = epoch_band_powers(signal, SFREQ, epoch_seconds=30.0)
    assert len(epochs) == 3
    assert all(set(e.keys()) == set(BANDS.keys()) for e in epochs)


def test_epoch_band_powers_drops_trailing_partial_epoch():
    signal = _sine(2.0, duration_seconds=65)  # 65s -> 2 full 30s epochs, 5s dropped
    epochs = epoch_band_powers(signal, SFREQ, epoch_seconds=30.0)
    assert len(epochs) == 2


def test_epoch_band_powers_raises_for_non_positive_epoch_size():
    try:
        epoch_band_powers(np.zeros(100), SFREQ, epoch_seconds=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_epoch_band_powers_each_epoch_reflects_its_own_frequency():
    # Concatenate a delta-dominant epoch and an alpha-dominant epoch.
    delta_epoch = _sine(2.0, duration_seconds=30)
    alpha_epoch = _sine(10.0, duration_seconds=30)
    signal = np.concatenate([delta_epoch, alpha_epoch])

    epochs = epoch_band_powers(signal, SFREQ, epoch_seconds=30.0)
    assert len(epochs) == 2
    assert max(epochs[0], key=epochs[0].get) == 'delta'
    assert max(epochs[1], key=epochs[1].get) == 'alpha'
