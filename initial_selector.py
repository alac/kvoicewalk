from fitness_scorer import FitnessScorer
from speech_generator import SpeechGenerator
import numpy as np
import torch
import os

class InitialSelector:
    def __init__(self, target_path: str, target_text: str, other_text: str, voice_folder: str = "./voices") -> None:
        self.fitness_scorer = FitnessScorer(target_path)
        self.speech_generator = SpeechGenerator()
        voices = []
        if os.path.exists(voice_folder):
            for filename in os.listdir(voice_folder):
                if filename.endswith('.pt'):
                    file_path = os.path.join(voice_folder, filename)
                    try:
                        voice = torch.load(file_path)
                        voices.append({
                            'name': filename,
                            'voice': voice
                        })
                    except:
                        print(f"Failed to load {filename}")
        
        self.voices = voices
        self.target_text = target_text
        self.other_text = other_text

    def top_performer_start(self, population_limit: int) -> list[torch.Tensor]:
        """Simple top performer search to find best voices to use in random walk"""
        print(f"Evaluating {len(self.voices)} initial voices...")
        for voice in self.voices:
            audio = self.speech_generator.generate_audio(self.target_text, voice["voice"])
            audio2 = self.speech_generator.generate_audio(self.other_text, voice["voice"])
            target_similarity = self.fitness_scorer.target_similarity(audio)
            results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity)
            
            print(f'{voice["name"]:<30} Target:{results["target_similarity"]:.3f} Self:{results["self_similarity"]:.3f} Qual:{results["quality_score"]:.2f} Score:{results["score"]:.2f}')
            voice["results"] = results

        voices = sorted(self.voices, key=lambda x: x["results"]["score"], reverse=True)
        voices = voices[:population_limit]
        
        print("\nTop Performers Selected:")
        for voice in voices:
            print(f'{voice["name"]:<30} Score:{voice["results"]["score"]:.2f}')

        tensors = [voice["voice"] for voice in voices]
        return tensors

    def interpolate_search(self, population_limit: int) -> list[torch.Tensor]:
        """Finds an initial population of voices more optimal because of interpolated features"""
        # Initial Evaluation
        for voice in self.voices:
            audio = self.speech_generator.generate_audio(self.target_text, voice["voice"])
            audio2 = self.speech_generator.generate_audio(self.other_text, voice["voice"])
            target_similarity = self.fitness_scorer.target_similarity(audio)
            results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity)
            voice["results"] = results

        voices = sorted(self.voices, key=lambda x: x["results"]["score"], reverse=True)
        voices = voices[:population_limit]

        print("\nTop Performers (Before Interpolation):")
        for voice in voices:
            print(f'{voice["name"]:<20} Score:{voice["results"]["score"]:.2f}')

        res = {}
        print("\nInterpolating Best Voices...")
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                # Search range: slightly extrapolate beyond the two voices
                for iter in np.arange(-0.5, 1.5 + 0.01, 0.2): # Reduced range/step for speed
                    voice_tensor = interpolate(voices[i]["voice"], voices[j]["voice"], iter)
                    
                    audio = self.speech_generator.generate_audio(self.target_text, voice_tensor)
                    audio2 = self.speech_generator.generate_audio(self.other_text, voice_tensor)
                    
                    target_similarity = self.fitness_scorer.target_similarity(audio)
                    results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity)
                    
                    print(f'{i:<2} {j:<2} {iter:<4.2f} Target:{results.get("target_similarity", 0):.3f} Self:{results.get("self_similarity", 0):.3f} Qual:{results.get("quality_score", 0):.2f} Score:{results.get("score", 0):.2f}')

                    # Logic to keep the best interpolation for each pair
                    # (Simplified for readability)
                    if results["score"] > 75.0: # Only save decent ones
                        key = f"{i}_{j}_{iter:.2f}"
                        res[key] = (voice_tensor, iter, voices[i]["name"], voices[j]["name"], results)

        interpolated_voices: list[torch.Tensor] = []
        if not os.path.exists("./interpolated"):
            os.makedirs("./interpolated")
            
        # Sort found voices by score
        sorted_res = sorted(res.values(), key=lambda x: x[4]["score"], reverse=True)
        
        print("\nSaving Best Interpolations:")
        for value in sorted_res[:population_limit]:
            fname = f"{value[2]}_{value[3]}_{value[1]:.2f}.pt"
            print(f'Saving {fname} | Score: {value[4]["score"]:.3f}')
            torch.save(value[0], f"./interpolated/{fname}")
            interpolated_voices.append(value[0])

        return interpolated_voices

def interpolate(voice1, voice2, alpha):
    return (1 - alpha) * voice1 + alpha * voice2
