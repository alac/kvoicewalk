from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import soundfile as sf

class FitnessScorer:
    def __init__(self, target_path: str):
        self.encoder = VoiceEncoder()
        # We only need the embed for the target now, not the raw spectral features
        self.target_wav = preprocess_wav(target_path, source_sr=24000)
        self.target_embed = self.encoder.embed_utterance(self.target_wav)

    def hybrid_similarity(self, audio: NDArray[np.float32], audio2: NDArray[np.float32], target_similarity: float):
        """
        Combines Target Similarity, Self-Similarity (Stability), and Audio Quality.
        """
        # 1. Calculate Stability (Self-Similarity)
        # Compare the embedding of the main audio vs the secondary (rotated) text audio
        self_sim = self.self_similarity(audio, audio2)

        # 2. Calculate Quality (Heuristics)
        # Penalize silence, clipping, or weird artifacts
        quality_score = self.assess_quality(audio)

        # 3. Weighted Score
        # Target Match: 40% | Stability: 30% | Quality: 30%
        # We multiply by 100 to keep it on a 0-100 scale
        score = (target_similarity * 40.0) + (self_sim * 30.0) + (quality_score * 30.0)

        return {
            "score": score,
            "target_similarity": target_similarity,
            "self_similarity": self_sim,
            "quality_score": quality_score
        }

    def target_similarity(self, audio: NDArray[np.float32]) -> float:
        # Preprocess and embed the generated audio
        try:
            audio_wav = preprocess_wav(audio, source_sr=24000)
            audio_embed = self.encoder.embed_utterance(audio_wav)
            similarity = np.inner(audio_embed, self.target_embed)
            return float(similarity)
        except Exception as e:
            print(f"Error in target_similarity: {e}")
            return 0.0

    def self_similarity(self, audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        """
        Checks if the voice identity remains constant across two different sentences.
        """
        try:
            wav1 = preprocess_wav(audio1, source_sr=24000)
            embed1 = self.encoder.embed_utterance(wav1)

            wav2 = preprocess_wav(audio2, source_sr=24000)
            embed2 = self.encoder.embed_utterance(wav2)
            
            return float(np.inner(embed1, embed2))
        except Exception as e:
            print(f"Error in self_similarity: {e}")
            return 0.0

    def assess_quality(self, audio: NDArray[np.float32]) -> float:
        """
        Fast heuristics to determine if the audio is technically flawed.
        Returns a score between 0.0 (Unusable) and 1.0 (Good Technical Quality).
        """
        score = 1.0
        
        # 1. Length Check
        # If audio is too short (< 0.5s), it's likely a generation failure
        if len(audio) < 24000 * 0.5:
            return 0.0

        # 2. RMS (Volume) Check - Penalize Silence
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01: 
            return 0.0 # Too quiet / silent
        
        # 3. Clipping Check - Penalize loud distortion
        # If a significant portion of the audio hits the ceiling (1.0 or -1.0)
        peak = np.max(np.abs(audio))
        if peak >= 0.99:
            score -= 0.2 # 20% penalty for clipping

        # 4. DC Offset Check
        # Large mean implies the signal is off-center
        if abs(np.mean(audio)) > 0.1:
            score -= 0.2

        return max(0.0, score)
