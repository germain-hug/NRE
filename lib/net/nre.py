"""Tools to compute and manipulate correlation maps.
"""

import math
import torch

from ..tools.decorators import to_minibatch
from ..tools.tensor_ops import extract_local_features, offset_and_sparsify
from ..tools.math import log


@to_minibatch(expected_outputs=1)
def compute_correlation_maps(
    source_descriptors: torch.Tensor,
    target_features: torch.Tensor,
):
    """Compute dense correspondence maps.
    Args:
        * source_descriptors: The interpolated source keypoint descriptors.
        * target_features: The dense feature map of the target image.
    Returns:
        * correlation_map: The dense correlation map.
    """
    correlation_map = correlate(source_descriptors, target_features)
    return correlation_map


@to_minibatch(expected_outputs=2)
def compute_local_nre_maps(
    source_descriptors: torch.Tensor,
    target_features: torch.Tensor,
    prior_target_keypoints: torch.Tensor,
    norm_coarse: torch.Tensor,
    window_size: int,
):
    """Compute dense local correspondence maps.
    Args:
        * source_descriptors: The interpolated source keypoint descriptors.
        * target_features: The dense feature map of the target image.
        * prior_target_keypoints: The prior keypoint locations,
            in target feature space.
        * norm_coarse: The norm_coarse normalizing factors.
        * window_size: The size of the window around the prior keypoint locations,
            in target feature space.
    Returns:
        * maps: The sparse correspondence maps
        * num_non_zero: The number of non-zero values per maps
    """
    assert len(source_descriptors) == len(prior_target_keypoints)
    assert len(norm_coarse) == len(prior_target_keypoints)
    local_target_features, patch_origins = extract_local_features(
        target_features, prior_target_keypoints, window_size
    )[:2]
    maps = correlate_local(source_descriptors, local_target_features)
    maps = softmax(maps) * norm_coarse[..., None, None] / window_size
    maps = truncate(log(maps).neg_(), target_features)
    maps, num_non_zero = offset_and_sparsify(
        maps, patch_origins, list(target_features.shape[-2:])
    )
    return maps, num_non_zero


@torch.jit.script
def correlate_local(sparse_descriptors: torch.Tensor, dense_descriptors: torch.Tensor):
    """Compute local correlation maps.
    Args:
        * sparse_features: Sparse descriptors of size [N x C]
        * dense_features: Batch of local descriptor maps of size [B x C x H x W]
    Returns:
        * local_cmaps: The concatenated local correspondence maps, of size [B x H x W]
    """
    batch, channels, height, width = dense_descriptors.shape
    local_cmaps = torch.bmm(
        sparse_descriptors.unsqueeze(1), dense_descriptors.view(batch, channels, -1)
    ).view(batch, height, width)
    return local_cmaps


def cardinal_omega(features: torch.Tensor):
    """Compute Card(Omega) based on the target image feature size."""
    return features.shape[-1] * features.shape[-2] + 1


def truncate(log_cmaps: torch.Tensor, target_features: torch.Tensor):
    """Truncate correspondence maps at log(Card(Omega))"""
    log_cmaps[log_cmaps >= math.log(cardinal_omega(target_features))] = 0.0
    return log_cmaps


def relative_aspect_ratio(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """Compute the relative aspect ratio between two tensors.
    Args:
        * input_tensor: The [..., Hi, Wi] input tensor.
        * output_tensor: The [..., Hj, Wj] output tensor.
    Returns:
        * ratio: The [2 x 1] relative aspect ratio.
    """
    i_height, i_width = input_tensor.shape[-2:]
    o_height, o_width = output_tensor.shape[-2:]
    return torch.tensor(
        [
            [
                float(o_width) / float(i_width),
                float(o_height) / float(i_height),
            ]
        ],
        dtype=torch.float32,
    )


def downsample_keypoints(
    keypoints: torch.Tensor, image: torch.Tensor, features: torch.Tensor
):
    """Downsample keypoints from image-space to feature-space.
    Args:
        * keypoints: The [N x 2] keypoints in image space, in the (x, y)
            order where x points right and y points down.
        * image: The image tensor.
        * features: The feature tensor.
    """
    return keypoints.float() * relative_aspect_ratio(image, features).to(
        keypoints.device
    )


def interpolate(dense_descriptors: torch.Tensor, keypoints: torch.Tensor):
    """Bilinearly interpolate sparse descriptors.
    Args:
        * dense_descriptors: The dense descriptor maps, of size [1 x C x H x W]
        * keypoints: The [N x 2] keypoint coordinates,
    Returns:
        * sparse_descriptors: The [N x C] sparse descriptors.
    """
    batch, channels, height, width = dense_descriptors.shape
    assert batch == 1
    scale = torch.tensor([width - 1, height - 1]).to(keypoints)
    keypoints = (keypoints / scale) * 2 - 1
    keypoints = keypoints.clamp(min=-2, max=2)
    sparse_descriptors = torch.nn.functional.grid_sample(
        dense_descriptors, keypoints[None, :, None], mode="bilinear", align_corners=True
    )
    return sparse_descriptors.view(channels, -1).transpose(-1, -2)


@torch.jit.script
def correlate(sparse_descriptors: torch.Tensor, dense_descriptors: torch.Tensor):
    """Compute dense correlation maps for every sparse descriptor.
    Args:
        * sparse_descriptors: A tensor of size [N x C]
        * dense_descriptors: A tensor of size [C x H x W]
    Returns:
        * correlation_map: A dense, unnormalized similarity map of size [N x H x W]
    """
    channels, height, width = dense_descriptors.shape[-3:]
    correlation_map = sparse_descriptors @ dense_descriptors.reshape(channels, -1)
    return correlation_map.reshape(-1, height, width).contiguous()


@torch.jit.script
def softmax(correlation_maps: torch.Tensor):
    """Applies a 2D spatial softmax operation on the dense correlation maps.
    Args:
        * correlation_maps: The batch of the dense correlation maps, of size [N x H x W]
    Returns:
        * The correspondence maps.
    """
    batch, height, width = correlation_maps.shape
    return correlation_maps.view(batch, -1).softmax(dim=1).view(batch, height, width)
