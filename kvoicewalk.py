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

# --- Add this line at the top of the file ---
TENSOR_SHAPE_DEBUG_FLAG = False

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
        population = [self.voice_generator.generate_voice(self.starting_voice, diversity=random.uniform(0.1, 0.5)) for _ in range(population_size)]

        # --- Adaptive Mutation Init ---
        mutation_rate = initial_mutation_rate
        last_best_score = 0.0

        for generation in range(generations):
            # Score population, including diversity calculation
            scores = [self.score_voice(voice, population=population, diversity_weight=diversity_weight) for voice in tqdm(population, desc=f"Generation {generation+1}/{generations}")]

            # Sort population by final score
            scored_population = sorted(zip(scores, population), key=lambda x: x[0]["score"], reverse=True)

            # --- Adaptive Mutation Logic ---
            current_best_score = scored_population[0][0]["score"]
            score_improvement = current_best_score - last_best_score
            if generation > 0 and score_improvement < stagnation_threshold:
                mutation_rate *= 1.2  # Increase mutation rate due to stagnation
            else:
                mutation_rate *= 0.95  # Decrease mutation rate towards minimum

            # Clamp the mutation rate to defined bounds
            mutation_rate = max(min_mutation_rate, min(mutation_rate, max_mutation_rate))
            last_best_score = current_best_score

            tqdm.write(f'Gen {generation+1}: Best Score={current_best_score:.2f} (Raw: {scored_population[0][0]["raw_score"]:.2f}), Mutation Rate={mutation_rate:.4f}')

            new_population = []

            # Elitism: carry over the best individuals
            for i in range(elitism_count):
                new_population.append(scored_population[i][1])

            # Selection, Crossover, and Mutation
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(scored_population)
                parent2 = self.tournament_selection(scored_population)

                # --- Crossover Logic ---
                if random.random() < crossover_rate:
                    if crossover_type == 'blend':
                        child1, child2 = self.voice_generator.blend_crossover(parent1, parent2)
                    else:  # Default to single_point
                        child1, child2 = self.voice_generator.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                # --- Mutation Logic ---
                if random.random() < mutation_rate:
                    child1 = self.voice_generator.mutate(child1, mutation_strength=mutation_rate)
                if random.random() < mutation_rate:
                    child2 = self.voice_generator.mutate(child2, mutation_strength=mutation_rate)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

            # Save best voice of the generation for review
            best_voice = scored_population[0][1]
            best_score_info = scored_population[0][0]
            torch.save(best_voice, f'out/gen_{generation+1}_best_voice_{best_score_info["score"]:.2f}.pt')
            sf.write(f'out/gen_{generation+1}_best_voice_{best_score_info["score"]:.2f}.wav', best_score_info["audio"], 24000)

    def tournament_selection(self, scored_population, k=5):
        """Selects a parent by choosing the best out of a small random sample."""
        sample_size = min(k, len(scored_population))
        tournament_entrants = random.sample(scored_population, sample_size)
        winner = max(tournament_entrants, key=lambda x: x[0]["score"])
        return winner[1]

    def score_voice(self, voice: torch.Tensor, min_similarity: float = 0.0, population: List[torch.Tensor] = [], diversity_weight: float = 0.0) -> dict[str, Any]:
        """Scores a voice based on similarity, with an optional penalty for being too similar to the rest of the population."""
        global TENSOR_SHAPE_DEBUG_FLAG # Use the global flag to ensure this runs only once

        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str, Any] = {'audio': audio}
        raw_score = 0.0

        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            hybrid_results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity)
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