"""Sampling strategies for camera pose solvers.
"""
import torch


def entropy_based_sampling(entropy: torch.Tensor, num_sampled_points: int):
    """Get the correspondence map sampler based on entropy."""
    # Normalize scores between 0 and 1
    weights = -entropy
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    # Prioritize sampling from "peaky" correspondence maps
    cmap_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, num_sampled_points
    )
    return cmap_sampler


def uniform_sampling(entropy: torch.Tensor, num_sampled_points: int):
    """Uniform cmap sampling"""
    weights = torch.ones_like(entropy)
    cmap_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, num_sampled_points
    )
    return cmap_sampler