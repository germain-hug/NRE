"""Compute the input / output theoretical shape of a
given tensor based on the network architecture.
"""
import torch

from typing import Tuple, List
from ..backbones.inception import BasicConv2d, InceptionB, InceptionD


def compute_input_shape(model_layers: List, output_shape: Tuple[int]):
    """Compute the input shape of a tensor based on its output shape."""
    assert len(output_shape) == 2
    input_shape = output_shape
    for layer in model_layers:
        # Zero-padding layers
        if isinstance(layer, torch.nn.ZeroPad2d):
            input_shape[0] += 2 * layer.padding[0]
            input_shape[1] += 2 * layer.padding[1]

        # Conv and MaxPool Layers
        elif isinstance(layer, (torch.nn.Conv2d, BasicConv2d, torch.nn.MaxPool2d)):
            if isinstance(layer, BasicConv2d):
                layer = layer.conv
            to_tuple = lambda x: (x, x) if isinstance(x, int) else x
            kernel_size = to_tuple(layer.kernel_size)
            padding = to_tuple(layer.padding)
            stride = to_tuple(layer.stride)
            dilation = to_tuple(layer.dilation)
            input_shape[0] = (
                stride[0] * (input_shape[0] - 1)
                + kernel_size[0]
                - 2 * padding[0]
                + (kernel_size[0] - 1) * (dilation[0] - 1)
            )
            input_shape[1] = (
                stride[1] * (input_shape[1] - 1)
                + kernel_size[1]
                - 2 * padding[1]
                + (kernel_size[1] - 1) * (dilation[1] - 1)
            )

        # InceptionB and coarse InceptionD modules have specific MaxPooling modules
        elif isinstance(layer, InceptionB) or (
            isinstance(layer, InceptionD) and layer.version == "coarse"
        ):
            input_shape[0] = 2 * (input_shape[0] - 1) + 3 - 2
            input_shape[1] = 2 * (input_shape[1] - 1) + 3 - 2

    return input_shape
