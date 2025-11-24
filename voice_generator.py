import os
import torch
import random

class VoiceGenerator:
    def __init__(self, voices: list[torch.Tensor], starting_voice: str | None):
        self.voices = voices

        self.stacked = torch.stack(voices,dim=0)
        self.mean = self.stacked.mean(dim=0)
        self.std = self.stacked.std(dim=0)
        self.min = self.stacked.min(dim=0)[0]
        self.max = self.stacked.max(dim=0)[0]

        if starting_voice:
            self.starting_voice = torch.load(starting_voice)
        else:
            self.starting_voice = self.mean

    def generate_voice(self,base_tensor: torch.Tensor | None,diversity: float = 1.0, device: str = "cpu", clip: bool = False):
        """Generate a new voice tensor based on the base_tensor and diversity.

        Args:
            base_tensor (torch.Tensor | None): The base tensor to generate the new voice from.
            diversity (float, optional): The diversity of the new voice. Defaults to 1.0.
            device (str, optional): The device to generate the new voice on. Defaults to "cpu".
            clip (bool, optional): Whether to clip the new voice to the min and max values. Defaults to False.

        Returns:
            torch.Tensor: The new voice tensor.
        """
        if base_tensor is None:
            base_tensor = self.mean.to(device)
        else:
            base_tensor = base_tensor.clone().to(device)

         # Generate random noise with same shape
        noise = torch.randn_like(base_tensor, device=device)

        # Scale noise by standard deviation and the noise_scale factor
        scaled_noise = noise * self.std.to(device) * diversity

        # Add scaled noise to base tensor
        new_tensor = base_tensor + scaled_noise

        if clip:
            new_tensor = torch.clamp(new_tensor, self.min, self.max)

        return new_tensor

    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor):
        # FORCE BLEND. Single point crossover destroys latent vector coherence.
        # We mix 30-70% of one parent with the other.
        alpha = random.uniform(0.3, 0.7)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def blend_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, alpha: float = 0.5):
        """Performs a blend crossover between two parent tensors."""
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def mutate(self, voice: torch.Tensor, mutation_strength: float = 0.05):
        # Use the standard deviation of THIS voice vector, not the global dataset.
        # This scales the mutation to fit the current vector's range.
        scale = voice.std().item()
        mutation = torch.randn_like(voice) * scale * mutation_strength
        return voice + mutation
