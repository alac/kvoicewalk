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
        """Perform a single point crossover between two parent tensors."""
        point = random.randint(1, parent1.size(0) - 1)
        child1 = torch.cat((parent1[:point], parent2[point:]), dim=0)
        child2 = torch.cat((parent2[:point], parent1[point:]), dim=0)
        return child1, child2

    def blend_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, alpha: float = 0.5):
        """Performs a blend crossover between two parent tensors."""
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def mutate(self, voice: torch.Tensor, mutation_strength: float = 0.05):
        """Apply mutation to a voice tensor."""
        mutation = torch.randn_like(voice) * self.std.to(voice.device) * mutation_strength
        return voice + mutation
