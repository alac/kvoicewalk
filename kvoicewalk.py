from typing import Any
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

    def genetic_algorithm(self, generations: int, population_size: int, mutation_rate: float, crossover_rate: float, elitism_count: int):
        if not os.path.exists("./out"):
            os.makedirs("./out")

        # Initialize population
        population = [self.voice_generator.generate_voice(self.starting_voice, diversity=random.uniform(0.1, 0.5)) for _ in range(population_size)]
        
        for generation in range(generations):
            # Score population
            scores = [self.score_voice(voice) for voice in tqdm(population, desc=f"Generation {generation+1}/{generations}")]
            
            # Sort population by score
            scored_population = sorted(zip(scores, population), key=lambda x: x[0]["score"], reverse=True)
            
            tqdm.write(f'Generation {generation+1} Best Score: {scored_population[0][0]["score"]:.2f}')

            new_population = []

            # Elitism
            for i in range(elitism_count):
                new_population.append(scored_population[i][1])

            # Selection, Crossover, and Mutation
            while len(new_population) < population_size:
                # Tournament Selection
                parent1 = self.tournament_selection(scored_population)
                parent2 = self.tournament_selection(scored_population)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self.voice_generator.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                if random.random() < mutation_rate:
                    child1 = self.voice_generator.mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self.voice_generator.mutate(child2)
                
                new_population.extend([child1, child2])

            population = new_population[:population_size]

            # Save best voice of the generation
            best_voice = scored_population[0][1]
            best_score = scored_population[0][0]
            torch.save(best_voice, f'out/gen_{generation+1}_best_voice_{best_score["score"]:.2f}.pt')
            sf.write(f'out/gen_{generation+1}_best_voice_{best_score["score"]:.2f}.wav', self.score_voice(best_voice)['audio'], 24000)

    def tournament_selection(self, scored_population, k=5):
        tournament_entrants = random.sample(scored_population, k)
        winner = max(tournament_entrants, key=lambda x: x[0]["score"])
        return winner[1]

    def score_voice(self,voice: torch.Tensor,min_similarity: float = 0.0) -> dict[str,Any]:
        """Using a harmonic mean calculation to provide a score for the voice in similarity"""
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str,Any] = {
            'audio': audio
        }
        # Bail early and save the compute if the similarity sucks
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(audio,audio2,target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        return results
