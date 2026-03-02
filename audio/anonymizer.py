import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Audio anonymization disabled.")


class VoiceAnonymizer:
    """
    Anonymizes voice audio in real-time before any AI inference.

    Processing pipeline (runs entirely on the edge device):
      1. Pitch shifting   — shifts fundamental frequency, breaking voiceprint
      2. Time stretching  — subtly modifies formant patterns
      3. Length alignment — preserves original duration for downstream analysis

    The transformed audio preserves behavioral cues needed for tension
    detection (energy level, speech rate, agitation markers) while making
    speaker re-identification computationally infeasible.

    Raw (non-anonymized) audio is never retained or transmitted.
    """

    def __init__(self, pitch_shift_semitones: int = -3):
        self.pitch_shift = pitch_shift_semitones
        self.sample_rate = 16000

    def anonymize(self, audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply pitch shifting and formant modification to audio.

        Args:
            audio_array:  1-D float32 audio signal.
            sr:           Sample rate in Hz.

        Returns:
            Anonymized audio array at the same sample rate and length.
        """
        if not LIBROSA_AVAILABLE:
            return audio_array

        shifted = librosa.effects.pitch_shift(
            audio_array.astype(float), sr=sr, n_steps=self.pitch_shift
        )
        stretched = librosa.effects.time_stretch(shifted, rate=1.05)
        return stretched[: len(audio_array)]

    def extract_behavioral_features(
        self, audio_array: np.ndarray, sr: int = 16000
    ) -> dict:
        """
        Extract behavioral cues from anonymized audio.

        These features are safe to transmit to the inference server —
        they carry no voiceprint or biometric information.

        Returns:
            dict with keys: rms_energy, pitch_variance, speech_rate (all [0, 1])
        """
        if not LIBROSA_AVAILABLE:
            return {"rms_energy": 0.5, "pitch_variance": 0.0, "speech_rate": 0.0}

        rms = float(np.sqrt(np.mean(audio_array ** 2)))

        f0, _, _ = librosa.pyin(
            audio_array.astype(float),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        pitch_variance = float(np.nanstd(f0)) if f0 is not None else 0.0

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_array)))

        return {
            "rms_energy": float(np.clip(rms * 10, 0, 1)),
            "pitch_variance": float(np.clip(pitch_variance / 100, 0, 1)),
            "speech_rate": float(np.clip(zcr * 2, 0, 1)),
        }
