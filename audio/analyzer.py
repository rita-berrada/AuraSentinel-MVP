import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from audio.anonymizer import VoiceAnonymizer


class AudioTensionAnalyzer:
    """
    Computes an audio tension score [0.0, 1.0] from ambient sound chunks.

    Pipeline:
      1. Anonymize raw audio (pitch shift, formant shift) — no voiceprint retained
      2. Extract behavioral features from the anonymized signal
      3. Compute spectral flux to detect sudden sound changes
      4. Combine features into a weighted tension score

    Processing runs in 0.5–2 second batches for low-latency alerting.
    All processing happens on the edge device (Jetson Nano).
    """

    def __init__(self):
        self.anonymizer = VoiceAnonymizer()

    def analyze_chunk(self, raw_audio: np.ndarray, sr: int = 16000) -> float:
        """
        Analyze a short audio chunk and return a tension score.

        Args:
            raw_audio:  1-D float32 audio signal (0.5–2 seconds recommended).
            sr:         Sample rate in Hz.

        Returns:
            Tension score in [0.0, 1.0].
        """
        # Step 1: Anonymize before any processing or logging
        anon_audio = self.anonymizer.anonymize(raw_audio, sr)

        # Step 2: Behavioral features (safe — no biometric data)
        features = self.anonymizer.extract_behavioral_features(anon_audio, sr)

        # Step 3: Spectral flux — sudden loudness/energy changes signal disturbance
        if LIBROSA_AVAILABLE:
            spec = np.abs(librosa.stft(anon_audio.astype(float)))
            flux = float(np.mean(np.diff(spec, axis=1) ** 2))
            spectral_score = float(np.clip(flux * 100, 0, 1))
        else:
            spectral_score = 0.0

        # Step 4: Weighted combination
        score = (
            features["rms_energy"]    * 0.35 +
            features["pitch_variance"] * 0.30 +
            features["speech_rate"]    * 0.15 +
            spectral_score             * 0.20
        )
        return float(np.clip(score, 0.0, 1.0))
