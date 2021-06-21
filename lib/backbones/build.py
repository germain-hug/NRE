"""NRE Convolutional Models Assembly.
"""
import logging
from typing import Dict

import torch
import torch.nn as nn

from .convolutional import CoarseModel, FineModel

convolutional_layers = {
    "Conv2d_1a_3x3": 32,
    "Conv2d_2a_3x3": 32,
    "Conv2d_2b_3x3": 64,
    "Conv2d_3b_1x1": 80,
    "Conv2d_4a_3x3": 192,
    "Mixed_5b": 256,
    "Mixed_5c": 288,
    "Mixed_5d": 288,
    "Mixed_6a": 768,
    "Mixed_6b": 768,
    "Mixed_6c": 768,
    "Mixed_6d": 768,
    "Mixed_6e": 768,
    "Mixed_7a": 1280,
}


def build_model(cfg: Dict):
    """Build the main model, consisting of a CNN + adap layers."""
    # Initialize CNN Backbones
    if cfg.level == "coarse":
        second_maxpool = (not hasattr(cfg, "no_maxpool")) or not cfg.no_maxpool
        layers, output_dimension = build_coarse_model(
            cfg.extraction_layer, second_maxpool
        )
    elif cfg.level == "fine":
        layers, output_dimension = build_fine_model(cfg.extraction_layer)
    else:
        raise KeyError
    layers = layers + [
        nn.Conv2d(
            output_dimension,
            output_dimension,
            kernel_size=1,
            bias=False,
        ),
        nn.BatchNorm2d(output_dimension),
    ]
    model = nn.Sequential(*layers)
    return model, output_dimension


def build_coarse_model(extraction_layer: str, second_maxpool: bool = True):
    """Assemble the coarse model up to the provided extraction layer.
    Args:
        * extraction_layer: The name of the feature extraction layer.
    Returns:
        * truncated_model: The list of the coarse model layers.
        * output_dim: The output features channel size.
    """
    assert extraction_layer in convolutional_layers, "Wrong layer name"
    base_model = CoarseModel()
    truncated_model = []
    for layer in convolutional_layers:
        truncated_model.append(base_model.__getattr__(layer))
        if layer == "Conv2d_2b_3x3" or (layer == "Conv2d_4a_3x3" and second_maxpool):
            truncated_model.append(
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        if layer == extraction_layer:
            break
    output_dim = convolutional_layers[extraction_layer]
    return truncated_model, output_dim


def build_fine_model(extraction_layer: str):
    """Assemble the fine model up to the provided extraction layer.
    Args:
        * extraction_layer: The name of the feature extraction layer.
    Returns:
        * truncated_model: The list of the coarse model layers.
        * output_dim: The output features channel size.
    """
    assert extraction_layer in convolutional_layers, "Wrong layer name"
    base_model = FineModel()
    truncated_model = []
    for layer in convolutional_layers:
        model_layer = base_model.__getattr__(layer)
        # Pad conv layers to preserve resolution
        if hasattr(model_layer, "conv"):
            padding = model_layer.conv.padding
            kernel_size = model_layer.conv.kernel_size
            if padding[0] == 0 and kernel_size[0] > 1:
                truncated_model.append(torch.nn.ZeroPad2d(2))
        truncated_model.append(model_layer)
        if layer == "Mixed_5d" or layer == extraction_layer:
            break
    output_dim = convolutional_layers[extraction_layer]
    return truncated_model, output_dim
