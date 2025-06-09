
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

class CryDetector:
    def __init__(self, filepath):
        """
        Initialize the detector with the path to an audio file.
        """
        self.filepath = filepath
        self.sr = None
        self.data = None
        self.energy = []
        self.zcr = []
        self.f0 = []
        self.flatness = []
        self.centroid = []
        self.cry_flags = []
        self.times = None
        self.frame_size = None
        self.hop_size = None
        self.num_frames = None

    def load_audio(self):
        """
        Loads the audio file, converts to mono if needed, and normalizes amplitude.
        """
        data, sr = librosa.load(self.filepath)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data / np.max(np.abs(data))
        self.data = data
        self.sr = sr

    @staticmethod
    def spectral_flatness(frame):
        """
        Computes the spectral flatness of a frame.
        """
        mag = np.abs(np.fft.rfft(frame))
        mag = mag + 1e-10
        geometric_mean = np.exp(np.mean(np.log(mag)))
        arithmetic_mean = np.mean(mag)
        return geometric_mean / arithmetic_mean

    @staticmethod
    def estimate_f0_autocorr(frame, sr):
        """
        Estimates the fundamental frequency (F0) using autocorrelation.
        """
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        d = np.diff(corr)
        try:
            start = np.where(d > 0)[0][0]
            peak = np.argmax(corr[start:]) + start
            return sr / peak if peak > 0 else 0
        except Exception:
            return 0

    @staticmethod
    def detect_rhythm(energy_seq, sr, hop_size):
        """
        Detects rhythmic periodicity in the energy sequence (to filter out music).
        """
        autocorr = np.correlate(energy_seq - np.mean(energy_seq), energy_seq - np.mean(energy_seq), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        threshold_lag = int(0.4 * sr / hop_size)
        peak = np.argmax(autocorr[threshold_lag:]) + threshold_lag
        periodicity_strength = autocorr[peak] / np.max(autocorr)
        return periodicity_strength > 0.6

    @staticmethod
    def compute_centroid(frame, sr):
        """
        Computes the spectral centroid of a frame.
        """
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1 / sr)
        return np.sum(freqs * spectrum) / (np.sum(spectrum))

    def extract_features(self):
        """
        Extracts short-time features (energy, ZCR, F0, flatness, centroid) from the audio.
        """
        self.frame_size = int(0.032 * self.sr)
        self.hop_size = int(0.016 * self.sr)
        self.num_frames = int((len(self.data) - self.frame_size) / self.hop_size)
        for i in range(self.num_frames):
            start = i * self.hop_size
            frame = self.data[start:start + self.frame_size]
            win = frame * np.hanning(len(frame))
            self.energy.append(np.sum(win ** 2))
            self.zcr.append(((win[:-1] * win[1:]) < 0).sum() / len(win))
            self.f0.append(self.estimate_f0_autocorr(win, self.sr))
            self.flatness.append(self.spectral_flatness(win))
            self.centroid.append(int(self.compute_centroid(frame, self.sr)))
        self.energy = np.array(self.energy)
        self.zcr = np.array(self.zcr)
        self.f0 = np.array(self.f0)
        self.flatness = np.array(self.flatness)
        self.centroid = np.array(self.centroid)
        self.times = np.arange(self.num_frames) * self.hop_size / self.sr

    def detect_cry(self):
        """
        Applies rule-based logic to detect cry segments in the audio.
        """
        is_rhythmic = self.detect_rhythm(self.energy, self.sr, self.hop_size)
        for i in range(self.num_frames):
            rule = (
                self.energy[i] > 0.01 and
                self.zcr[i] > 0.1 and
                300 <= self.f0[i] <= 600 and
                self.flatness[i] < 0.6 and
                not is_rhythmic and
                self.centroid[i] > 1000
            )
            self.cry_flags.append(1 if rule else 0)
        self.cry_flags = np.convolve(self.cry_flags, np.ones(3)/3, mode='same')
        self.cry_flags = (self.cry_flags > 0.6).astype(int)

    def display(self, file_name=None):
        """
        Plots the extracted features and detected cry segments.
        """
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Cry Detection for {file_name if file_name else self.filepath}")
        plt.subplot(4, 1, 1)
        plt.plot(self.times, self.energy, label='Energy')
        plt.title("Short-Time Energy")
        plt.subplot(4, 1, 2)
        plt.plot(self.times, self.f0, label='F0 (Hz)', color='green')
        plt.title("Fundamental Frequency")
        plt.subplot(4, 1, 3)
        plt.plot(self.times, self.zcr, label='zcr', color='red')
        plt.title("zcr")
        plt.subplot(4, 1, 4)
        plt.plot(self.times, self.cry_flags, label='Cry Detected', color='purple')
        plt.title("Cry Detected")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"outcome/{file_name}.png" if file_name else "cry_detection_plot.png")

    def run(self, display=True, file_name=None):
        """
        Runs the full cry detection pipeline.
        Args:
            display (bool): If True, show the plots.
            file_name (str): Optional name for saving/displaying plots.
        """
        self.load_audio()
        self.extract_features()
        self.detect_cry()
        if display:
            self.display(file_name)

# Example usage:
# file_path = "../data/NoCry-Noise-NoMusic/nc_n_nm1.ogg"
# detector = CryDetector(file_path)
# detector.run(display=True)