import numpy as np
import soundfile as sf
import librosa
import traceback
from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder

class FitnessScorer:
    # =========================================================================
    #  WEIGHTS CONFIGURATION
    #  Set a weight to 0.0 to disable the calculation (saves time).
    #  Adjust these to tune the "Fullness" and "Identity".
    # =========================================================================
    WEIGHTS = {
        # --- 1. IDENTITY (Who is speaking?) ---
        "resemblyzer_cosine":   40.0,  # The heavy hitter for identity

        # --- 2. TIMBRE (The "Body" & "Fullness") ---
        # MFCCs capture the shape of the vocal tract. 
        "mfcc_mean_dist":       15.0,  # General tone match
        "mfcc_std_dist":        5.0,   # Expressiveness match
        
        # --- 3. SPECTRAL SHAPE (Brightness vs Dark/Full) ---
        "spectral_centroid":    5.0,   # Brightness (Center of mass)
        "spectral_bandwidth":   5.0,   # Width of frequency band
        "spectral_rolloff":     10.0,  # Bass/Treble balance (Crucial for "depth")
        "spectral_contrast":    5.0,   # Peak/Valley distinction (clarity)
        "spectral_flatness":    5.0,   # Noise-like vs Tone-like

        # --- 4. PITCH & INTONATION (Prosody) ---
        # Note: F0 estimation is slow. Set to 0.0 if generation is too slow.
        "f0_mean_error":        0.0, # 5.0,   # Average pitch match
        "f0_std_error":         0.0, # 2.0,   # Dynamic range of pitch
        
        # --- 5. QUALITY & TEXTURE ---
        "hnr_ratio":            10.0,  # Harmonic-to-Noise (Cleanliness vs Breathiness)
        "zero_crossing_rate":   0.0,   # (Optional) Breathiness/Fricatives
        
        # --- 6. TECHNICAL HYGIENE (Sanity Checks) ---
        "clipping_penalty":     20.0,  # Huge penalty for distortion
        "silence_penalty":      20.0,  # Huge penalty for silence
        "dc_offset_penalty":    5.0,   # Signal centering
        "rms_level_diff":       5.0,   # Volume matching
    }

    def __init__(self, target_path: str):
        print(f"Loading Fitness Scorer target: {target_path}")
        self.encoder = VoiceEncoder()
        
        # Load Raw Audio (for Librosa)
        # Librosa prefers 22050 usually, but we stick to 24000 to match Kokoro
        self.sr = 24000 
        self.target_audio, _ = sf.read(target_path, dtype="float32")
        
        # Handle stereo
        if len(self.target_audio.shape) > 1:
            self.target_audio = np.mean(self.target_audio, axis=1)

        # 1. Resemblyzer Setup
        self.target_wav_res = preprocess_wav(target_path, source_sr=self.sr)
        self.target_embed = self.encoder.embed_utterance(self.target_wav_res)

        # 2. Pre-calculate Target Metrics (The "Gold Standard")
        print("Computing target voice statistics...")
        self.target_stats = self._analyze_audio(self.target_audio)
        
        print(f"Target Stats Loaded:")
        print(f" - Centroid: {self.target_stats['mean_centroid']:.0f} Hz")
        print(f" - Pitch (F0): {self.target_stats['mean_f0']:.0f} Hz")
        print(f" - Rolloff (Fullness): {self.target_stats['mean_rolloff']:.0f} Hz")

    def _analyze_audio(self, audio: np.ndarray) -> dict:
        """
        Extracts all relevant features from an audio chunk.
        Returns a dictionary of stats.
        """
        stats = {}
        
        # Safety: tiny epsilon to prevent div by zero
        if len(audio) < 512: 
            return self._get_empty_stats()
            
        y = audio
        sr = self.sr

        # --- SPECTRAL FEATURES ---
        # Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        stats['mean_centroid'] = np.mean(cent)
        
        # Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        stats['mean_bandwidth'] = np.mean(bw)
        
        # Rolloff (85% energy point - good for "depth")
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        stats['mean_rolloff'] = np.mean(roll)
        
        # Contrast & Flatness
        stats['mean_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        stats['mean_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))

        # --- TIMBRE (MFCCs) ---
        # 20 coefficients is standard for voice
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        stats['mfcc_mean'] = np.mean(mfcc, axis=1) # Shape (20,)
        stats['mfcc_std'] = np.std(mfcc, axis=1)   # Shape (20,)

        # --- PITCH (F0) ---
        # Using Yin implementation if available (faster), else pyin
        # F0 estimation is heavy. Skip if weight is 0.
        if self.WEIGHTS["f0_mean_error"] > 0 or self.WEIGHTS["f0_std_error"] > 0:
            try:
                # Yin is faster than pyin
                f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
                # Filter out NaNs (unvoiced)
                f0 = f0[~np.isnan(f0)]
                if len(f0) > 0:
                    stats['mean_f0'] = np.mean(f0)
                    stats['std_f0'] = np.std(f0)
                else:
                    stats['mean_f0'] = 0.0
                    stats['std_f0'] = 0.0
            except:
                stats['mean_f0'] = 0.0
                stats['std_f0'] = 0.0
        else:
            stats['mean_f0'] = 0.0
            stats['std_f0'] = 0.0

        # --- QUALITY ---
        # HNR (Harmonic to Noise)
        try:
            harmonic = librosa.effects.harmonic(y)
            # RMS of harmonic / RMS of total (Approx HNR proxy for speed)
            h_rms = np.sqrt(np.mean(harmonic**2))
            t_rms = np.sqrt(np.mean(y**2))
            stats['hnr'] = h_rms / (t_rms + 1e-9)
        except:
            stats['hnr'] = 0.0

        stats['rms'] = np.sqrt(np.mean(y**2))
        stats['max_amp'] = np.max(np.abs(y))
        stats['dc_offset'] = np.mean(y)

        return stats

    def _get_empty_stats(self):
        return {
            'mean_centroid': 0, 'mean_bandwidth': 0, 'mean_rolloff': 0,
            'mean_contrast': 0, 'mean_flatness': 0, 
            'mfcc_mean': np.zeros(20), 'mfcc_std': np.zeros(20),
            'mean_f0': 0, 'std_f0': 0, 'hnr': 0,
            'rms': 0, 'max_amp': 0, 'dc_offset': 0
        }

    def hybrid_similarity(self, audio: NDArray[np.float32], audio2: NDArray[np.float32], target_similarity: float):
        """
        The main scoring entry point.
        """
        # 1. Identity Check (Resemblyzer) - Already passed as target_similarity
        # But we also want self-similarity (audio1 vs audio2) to ensure stability
        self_sim = self.self_similarity(audio, audio2)
        
        # 2. Analyze the generated audio
        gen_stats = self._analyze_audio(audio)
        
        # 3. Calculate Component Scores
        scores = {}
        total_score = 0.0
        total_possible_weight = 0.0

        for metric, weight in self.WEIGHTS.items():
            if weight <= 0: continue
            
            val = 0.0
            
            # --- IDENTITY ---
            if metric == "resemblyzer_cosine":
                # Average of target match and stability
                val = (target_similarity + self_sim) / 2.0
                # Scale: Resemblyzer usually 0.7-0.9. Map 0.5->0, 0.9->1
                val = max(0.0, (val - 0.5) * 2.5)
                val = min(1.0, val)

            # --- SPECTRAL ---
            elif metric in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
                key = f"mean_{metric.split('_')[1]}"
                val = self._score_scalar_diff(gen_stats[key], self.target_stats[key], tolerance=0.15)
            
            elif metric == "spectral_contrast":
                val = self._score_scalar_diff(gen_stats['mean_contrast'], self.target_stats['mean_contrast'], tolerance=0.2)
            
            elif metric == "spectral_flatness":
                 val = self._score_scalar_diff(gen_stats['mean_flatness'], self.target_stats['mean_flatness'], tolerance=0.2)

            # --- TIMBRE (MFCC) ---
            elif metric == "mfcc_mean_dist":
                # Euclidean distance between vectors
                dist = np.linalg.norm(gen_stats['mfcc_mean'] - self.target_stats['mfcc_mean'])
                # Dist usually 10-50. Convert to 0-1 score.
                # Smaller distance = Higher score.
                val = max(0.0, 1.0 - (dist / 30.0))
            
            elif metric == "mfcc_std_dist":
                dist = np.linalg.norm(gen_stats['mfcc_std'] - self.target_stats['mfcc_std'])
                val = max(0.0, 1.0 - (dist / 15.0))

            # --- PITCH ---
            elif metric == "f0_mean_error":
                if self.target_stats['mean_f0'] > 0:
                    val = self._score_scalar_diff(gen_stats['mean_f0'], self.target_stats['mean_f0'], tolerance=0.1)
                else: 
                    val = 1.0 # Ignore if target unvoiced
            
            elif metric == "f0_std_error":
                if self.target_stats['std_f0'] > 0:
                    val = self._score_scalar_diff(gen_stats['std_f0'], self.target_stats['std_f0'], tolerance=0.2)
                else:
                    val = 1.0

            # --- QUALITY ---
            elif metric == "hnr_ratio":
                val = self._score_scalar_diff(gen_stats['hnr'], self.target_stats['hnr'], tolerance=0.1)

            # --- TECHNICAL ---
            elif metric == "clipping_penalty":
                if gen_stats['max_amp'] > 0.99: val = 0.0
                else: val = 1.0
            
            elif metric == "silence_penalty":
                if gen_stats['rms'] < 0.01: val = 0.0
                else: val = 1.0

            elif metric == "dc_offset_penalty":
                if abs(gen_stats['dc_offset']) > 0.05: val = 0.0
                else: val = 1.0
            
            elif metric == "rms_level_diff":
                val = self._score_scalar_diff(gen_stats['rms'], self.target_stats['rms'], tolerance=0.3)

            # Accumulate
            scores[metric] = val
            total_score += val * weight
            total_possible_weight += weight

        # Final Normalization
        final_score = (total_score / total_possible_weight) * 100.0 if total_possible_weight > 0 else 0.0

        return {
            "score": final_score,
            "target_similarity": target_similarity,
            "self_similarity": self_sim,
            # Map new metrics to old keys to prevent breaking kvoicewalk prints
            "tone_score": scores.get("spectral_rolloff", 0.0), 
            "quality_score": scores.get("hnr_ratio", 0.0),
            "cents": f"{gen_stats['mean_centroid']:.0f}/{self.target_stats['mean_centroid']:.0f}",
            "debug_metrics": scores # Full breakdown for debugging
        }

    def _score_scalar_diff(self, current, target, tolerance=0.1) -> float:
        """
        Returns 1.0 if current == target.
        Drops to 0.5 if difference is 'tolerance' % (e.g. 10%).
        Drops to 0.0 if difference is 2 * tolerance.
        """
        if target == 0: return 0.0 if current != 0 else 1.0
        
        diff_ratio = abs(current - target) / abs(target)
        
        # Curve: 1 - (diff / (2*tol))
        # If diff is 0.1 and tol is 0.1, score is 0.5
        score = 1.0 - (diff_ratio / (2.0 * tolerance))
        return max(0.0, min(1.0, score))

    def target_similarity(self, audio: NDArray[np.float32]) -> float:
        try:
            audio_wav = preprocess_wav(audio, source_sr=24000)
            audio_embed = self.encoder.embed_utterance(audio_wav)
            similarity = np.inner(audio_embed, self.target_embed)
            return float(similarity)
        except Exception as e:
            # print(f"Error in target_similarity: {e}")
            return 0.0

    def self_similarity(self, audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        try:
            wav1 = preprocess_wav(audio1, source_sr=24000)
            embed1 = self.encoder.embed_utterance(wav1)
            wav2 = preprocess_wav(audio2, source_sr=24000)
            embed2 = self.encoder.embed_utterance(wav2)
            return float(np.inner(embed1, embed2))
        except Exception:
            return 0.0