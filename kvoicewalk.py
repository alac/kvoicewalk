from typing import Any, List
from fitness_scorer import FitnessScorer
from initial_selector import InitialSelector
from speech_generator import SpeechGenerator
from voice_generator import VoiceGenerator
import random
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch
import os


TENSOR_SHAPE_DEBUG_FLAG = False
VALIDATION_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "I am trying to optimize this voice vector.",
    "Technological progress has merely provided us with more efficient means.",
    "It was the best of times, it was the worst of times."
]
CONTROL_SENTENCE = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."


class KVoiceWalk:
    def __init__(self,target_audio: str,target_text: str,other_text:str,voice_folder:str,interpolate_start: bool,population_limit: int, starting_voice: str) -> None:
        self.target_text = target_text
        self.other_text = other_text
        self.initial_selector = InitialSelector(target_audio,target_text,other_text,voice_folder=voice_folder)
        voices: list[torch.Tensor] = []
        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)
        self.speech_generator = SpeechGenerator()
        self.fitness_scorer = FitnessScorer(target_audio)
        self.voice_generator = VoiceGenerator(voices,starting_voice)
        # Either the mean or the supplied voice tensor
        self.starting_voice = self.voice_generator.starting_voice

    def genetic_algorithm(self,
                          generations: int,
                          population_size: int,
                          initial_mutation_rate: float,
                          crossover_rate: float,
                          elitism_count: int,
                          crossover_type: str = 'blend',  # blend or single_point
                          diversity_weight: float = 0.0,
                          min_mutation_rate: float = 0.01,
                          max_mutation_rate: float = 0.5,
                          stagnation_threshold: float = 0.01):
        if not os.path.exists("./out"):
            os.makedirs("./out")

        # Initialize population
        population = [self.voice_generator.generate_voice(self.starting_voice, diversity=random.uniform(0.1, 0.5)) for _ in range(population_size -1)]
        population.append(self.starting_voice)

        # --- Adaptive Mutation Init ---
        mutation_rate = initial_mutation_rate

        # PARAMETERS
        FRESH_BLOOD_RATE = 0.1  # Replace bottom 10% with random voices every gen
        MIN_MUTATION_STRENGTH = 0.02 # Never let mutation drop below this

        for generation in tqdm(range(generations), desc="Computing generations..."):
            # Calculate a progress factor (0.0 to 1.0) inside the generation loop
            progress = generation / generations

            # Define dynamic weights
            # Start Identity at 5.0, ramp up to 30.0
            # Start Timbre (MFCC) at 20.0, ramp down to 10.0
            current_weights = self.fitness_scorer.WEIGHTS.copy()
            current_weights["resemblyzer_cosine"] = 5.0 + (25.0 * progress) 
            current_weights["mfcc_mean_dist"] = 20.0 - (10.0 * progress)

            # Score population, including diversity calculation
            scores = [self.score_voice(voice, population=population, diversity_weight=diversity_weight, override_weights=current_weights)
                      for voice in tqdm(population, desc=f"Generation {generation+1}/{generations}")]

            # Sort population by final score
            # Sort population
            scored_population = sorted(zip(scores, population), key=lambda x: x[0]["score"], reverse=True)
        
            # Print stats
            best = scored_population[0][0]
            tqdm.write(f'Gen {generation+1}: Best Score={best["score"]:.2f} | ID:{best["target_similarity"]:.2f} Tone:{best["tone_score"]:.2f} Cents:{best["cents"]}')

            # ELITISM (Keep top 10%)
            new_population = [x[1] for x in scored_population[:int(population_size * 0.1)]]

            # BREEDING LOOP
            # We fill up to 90% of population size
            target_count = int(population_size * (1 - FRESH_BLOOD_RATE))
            
            while len(new_population) < target_count:
                parent1 = self.tournament_selection(scored_population)
                parent2 = self.tournament_selection(scored_population)

                child1, child2 = self.voice_generator.crossover(parent1, parent2, method=crossover_type)

                # DYNAMIC MUTATION
                # Increase mutation if score is stagnating
                current_mutation = max(MIN_MUTATION_STRENGTH, mutation_rate)
                
                if random.random() < current_mutation:
                    child1 = self.voice_generator.mutate(child1, mutation_strength=current_mutation)
                if random.random() < current_mutation:
                    child2 = self.voice_generator.mutate(child2, mutation_strength=current_mutation)
                
                new_population.extend([child1, child2])
            
            # FRESH BLOOD INJECTION
            # Fill the remaining slots with purely random noise voices
            # This forces the GA to look at completely new areas of the vector space
            while len(new_population) < population_size:
                # Generate a fresh random voice based on the GLOBAL mean/std
                fresh_voice = self.voice_generator.generate_voice(None, diversity=1.0) 
                new_population.append(fresh_voice)

            population = new_population

            # Save best voice of the generation for review
            best_voice = scored_population[0][1]
            best_score_info = scored_population[0][0]
            torch.save(best_voice, f'out/gen_{generation+1}_best.pt')
            sf.write(f'out/gen_{generation+1}_target.wav', best_score_info["audio"], 24000)

            control_audio = self.speech_generator.generate_audio(CONTROL_SENTENCE, best_voice, speed=1.0)
            sf.write(f'out/gen_{generation+1}_control.wav', control_audio, 24000)

    def tournament_selection(self, scored_population, k=5):
        """Selects a parent by choosing the best out of a small random sample."""
        sample_size = min(k, len(scored_population))
        tournament_entrants = random.sample(scored_population, sample_size)
        winner = max(tournament_entrants, key=lambda x: x[0]["score"])
        return winner[1]

    def score_voice(self, voice: torch.Tensor, min_similarity: float = 0.0, population: List[torch.Tensor] = [], diversity_weight: float = 0.0, override_weights=None) -> dict[str, Any]:
        global TENSOR_SHAPE_DEBUG_FLAG # Use the global flag to ensure this runs only once
        
        # 1. Generate Main Target
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        
        results: dict[str, Any] = {'audio': audio}
        
        # 2. Generate Validation (Random Sentence)
        # This ensures the voice is stable regardless of what it says
        random_text = random.choice(VALIDATION_SENTENCES) 
        
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(random_text, voice)
            # Pass the random text audio to your scorer
            hybrid_results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity, override_weights=override_weights)
            results.update(hybrid_results)
            raw_score = hybrid_results.get("score", 0.0)
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        results["raw_score"] = raw_score

        # --- Diversity Penalty Calculation ---
        if diversity_weight > 0 and len(population) > 1:
            # --- START DEBUGGING BLOCK ---
            if TENSOR_SHAPE_DEBUG_FLAG:
                print("\n--- KVoiceWalk Debugging ---")
                print(f"Shape of 'voice' tensor: {voice.shape}")
                # Find a different tensor in the population to compare against
                other_sample = next((p for p in population if not torch.equal(voice, p)), None)
                if other_sample is not None:
                    print(f"Shape of 'other' tensor sample: {other_sample.shape}")
                    try:
                        # Ensure both tensors are 2D for the comparison
                        sim_result = torch.nn.functional.cosine_similarity(voice.unsqueeze(0), other_sample.unsqueeze(0), dim=-1)
                        print(f"Shape of cosine_similarity result: {sim_result.shape}")
                        if sim_result.numel() > 1:
                            print("!!! WARNING: cosine_similarity is returning a multi-element tensor.")
                        else:
                            print("--> Cosine similarity result is a scalar tensor, as expected.")
                    except Exception as e:
                        print(f"XXX Error during sample similarity calculation: {e}")
                else:
                    print("Could not find a different tensor to compare for debugging.")
                print("--------------------------\n")
                TENSOR_SHAPE_DEBUG_FLAG = False # Turn off the flag after the first run
            # --- END DEBUGGING BLOCK ---

            similarities = torch.tensor([torch.nn.functional.cosine_similarity(voice.flatten(), other.flatten(), dim=0) for other in population if not torch.equal(voice, other)])

            if len(similarities) > 0:
                avg_similarity = torch.mean(similarities).item()
                # Normalize similarity to a 0-1 range where 1 is highly similar
                avg_similarity_normalized = (avg_similarity + 1) / 2
                # The penalty is a percentage of the raw score
                penalty = avg_similarity_normalized * diversity_weight
                final_score = raw_score * (1 - penalty)
            else:
                final_score = raw_score
        else:
            final_score = raw_score

        results["score"] = final_score
        return results