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

    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, method: str = 'slerp'):
        """
        Combines two parents into two children.
        Methods:
            'blend': Linear interpolation.
            'slerp': Spherical Linear Interpolation (better for high-dim vectors).
        """
        # 1. Determine Alpha (The Mix Ratio)
        # OLD: alpha = random.uniform(0.3, 0.7) -> Causes convergence to mean
        # NEW: Extrapolative range. Allow going slightly "past" the parents.
        # Range [-0.2, 1.2] allows exploring outside the box defined by parents.
        alpha = random.uniform(-0.2, 1.2) 

        if method == 'slerp':
            # SLERP handles the curve, but we still use our extrapolative alpha
            child1 = self.slerp(parent1, parent2, alpha)
            child2 = self.slerp(parent1, parent2, 1.0 - alpha)
        else:
            # Linear Extrapolation
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

        # CRITICAL: Extrapolation can push values too high/low, causing audio artifacts.
        # Clamp children to the global min/max observed in your init.
        child1 = torch.clamp(child1, self.min.to(child1.device), self.max.to(child1.device))
        child2 = torch.clamp(child2, self.min.to(child2.device), self.max.to(child2.device))

        return child1, child2

    def slerp(self, p1: torch.Tensor, p2: torch.Tensor, t: float, dot_threshold: float = 0.9995):
        """
        Spherical Linear Interpolation.
        Preserves the "magnitude" (energy) of the voice better than linear blending.
        """
        # 1. Compute properties
        p1_norm = torch.norm(p1)
        p2_norm = torch.norm(p2)
        
        # Normalize vectors for angle calculation
        p1_u = p1 / p1_norm
        p2_u = p2 / p2_norm

        # 2. Calculate Dot Product (Cosine of angle)
        dot = torch.sum(p1_u * p2_u)

        # 3. Handle Parallel Vectors (Standard Linear Interpolation fallback)
        if torch.abs(dot) > dot_threshold:
            result = (1.0 - t) * p1 + t * p2
            return result

        # 4. Calculate Angle (Omega)
        # Clamp for numerical stability (acos requires -1 to 1)
        dot = torch.clamp(dot, -1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        # 5. Calculate Interpolation
        # Note: We interpolate the Magnitude separately from the Direction
        # to handle cases where parents have very different loudness/intensities.
        
        # Directional interpolation
        scale0 = torch.sin((1.0 - t) * omega) / sin_omega
        scale1 = torch.sin(t * omega) / sin_omega
        direction = scale0 * p1_u + scale1 * p2_u
        
        # Magnitude interpolation (Linear)
        mag = (1.0 - t) * p1_norm + t * p2_norm
        
        return direction * mag

    def blend_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, alpha: float = 0.5):
        """Performs a blend crossover between two parent tensors."""
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def mutate(self, voice: torch.Tensor, mutation_strength: float = 0.05, sparse_prob: float = 0.2):
        """
        Applies mutation to only a percentage of the dimensions (sparse_prob).
        This allows distinct traits to change without destroying the whole voice structure.
        """
        # Get the standard deviation of this specific voice for scaling
        scale = voice.std().item()
        
        # Create a mask: 1 where we mutate, 0 where we keep original
        # Bernoulli distribution creates a binary mask
        mask = torch.bernoulli(torch.full_like(voice, sparse_prob)).bool()
        
        # Generate noise
        # We increase strength slightly because we are applying it less often
        noise = torch.randn_like(voice) * scale * (mutation_strength * 3.0)
        
        # Apply noise only where mask is True
        mutated_voice = voice.clone()
        mutated_voice[mask] += noise[mask]
        
        return mutated_voice
