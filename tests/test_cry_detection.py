import numpy as np
from src.cry_detection import preoprocess_audio, spectral_flatness, estimate_f0_autocorr, detect_rhythm, compute_centroid

def test_preprocess_audio():
    data = np.array([0, 1, 2, 3, 4])
    processed = preoprocess_audio(data, alpha=0.9)
    expected = np.array([0, 1, 1.9, 2.71, 3.439])
    np.testing.assert_almost_equal(processed, expected, decimal=2)

def test_spectral_flatness():
    frame = np.array([0, 1, 0, 1])
    flatness = spectral_flatness(frame)
    assert flatness >= 0  # Flatness should be non-negative

def test_estimate_f0_autocorr():
    frame = np.array([0, 1, 0, -1, 0, 1, 0, -1])
    sr = 8000
    f0 = estimate_f0_autocorr(frame, sr)
    assert f0 > 0  # F0 should be positive for a periodic signal

def test_detect_rhythm():
    energy_seq = np.array([0.1, 0.2, 0.1, 0.3, 0.1])
    sr = 8000
    hop_size = 512
    is_rhythmic = detect_rhythm(energy_seq, sr, hop_size)
    assert isinstance(is_rhythmic, bool)  # Should return a boolean

def test_compute_centroid():
    frame = np.array([0, 1, 0, 1])
    sr = 8000
    centroid = compute_centroid(frame, sr)
    assert centroid >= 0  # Centroid frequency should be non-negative