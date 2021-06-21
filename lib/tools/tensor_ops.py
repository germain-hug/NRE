"""Misc. tensor operations.
"""
from typing import List

import numpy as np
import torch


def image_to_pil(tensor: torch.Tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.transpose(0, 1).transpose(1, 2)
    return tensor.transpose(1, 2, 0)


def to_cpu(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def to_numpy(x: torch.Tensor):
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return to_cpu(x).numpy()
    return x


def to_device(x: torch.Tensor, device: torch.device):
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_device(i, device) for i in x]
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        return x
    return x.to(device)


def to_dtype(x: torch.Tensor, dtype: torch.dtype):
    if isinstance(x, dict):
        return {k: to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_dtype(i, dtype) for i in x]
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    if not isinstance(x, torch.Tensor):
        return x
    return x.to(dtype)


@torch.jit.script
def argmax_2D(cmaps: torch.Tensor):
    """Retrieve 2D argmax coordinates from dense cost maps.
    Args:
        cmaps: A [N x H x W] tensor.
    Returns:
        argmax_2d: The [N x 2] 2D argmax coordinates.
    """
    channels, _, width = cmaps.shape
    _, argmax_1d = cmaps.view(channels, -1).max(1)
    x_indices = (argmax_1d % width).view(-1, 1)
    y_indices = (argmax_1d // width).view(-1, 1)
    return torch.cat((x_indices, y_indices), dim=1)


def distances_2D(x: torch.Tensor, y: torch.Tensor):
    """Compute 2D distances between two coordinate tensors."""
    if isinstance(x, np.ndarray):
        return np.sqrt(np.linalg.norm(x - y, ord=2, axis=1))
    else:
        return (x - y).norm(p=2, dim=1).sqrt()


def extract_local_features(
    feature_map: torch.Tensor, window_centers: torch.Tensor, window_size: int
):
    """Extract local sub feature maps centered around window_centers of size window_size.
    Args:
        * feature_map: The map to extract features from, of size [1 x C x H x W].
        * window_centers: The subwindows centers, of size [N x 2], in (x, y) order.
        * window_size: Sub-window size.
    Returns:
        * submaps: The tensor of local feature maps, of size [1 x C x ws x ws].
        * origins: The (x, y) coordinates of the upper-left corner of the patches.
        * offsets: The (x, y) coordinates of the window center in the actually extracted window.
    """
    # Prepare output tensors
    assert len(feature_map) == 1
    num_windows = len(window_centers)
    submaps = []

    # Pad feature map by window_size / 2 to obtain concatenable square patches
    padding_margin = padding_size(window_size)
    padded_feature_map = pad_feature_map(feature_map, padding_margin)
    window_centers += padding_margin

    # Extract local feature maps
    origins = torch.zeros(
        (num_windows, 2), device=window_centers.device, dtype=torch.long
    )
    for i, center in enumerate(window_centers):

        # Compute the subwindow coordinates
        x, y = int(center[0]), int(center[1])
        x_min = x - window_size // 2
        x_max = x + window_size // 2 + 1
        y_min = y - window_size // 2
        y_max = y + window_size // 2 + 1
        submaps.append(padded_feature_map[0, :, y_min:y_max, x_min:x_max])

        # Store upper-left window coordinates in the padded map
        origins[i, 0] = x_min
        origins[i, 1] = y_min

    submaps = torch.stack(submaps)

    return submaps, origins


def offset_and_sparsify(maps: torch.Tensor, origins: torch.Tensor, target_shape: List):
    """Assemble sparse maps from the batch of local patches.
    Args:
        * maps: The [N x H x W] dense tensor of local maps.
        * origins: The [N x 2] local windows origins.
        * target_shape: The target shape of the full image tensor.
    Returns:
        * full_maps: The assembled sparse map of size target_shape.
        * num_non_zero: The number of non-zero value per map.
    """
    # Assemble empty target_shape tensor
    batch, height, width = maps.shape
    window_size = height
    padding = padding_size(window_size)
    full_maps = torch.zeros(
        (
            batch,
            target_shape[0] + 2 * padding,
            target_shape[1] + 2 * padding,
        ),
        dtype=maps.dtype,
        device=maps.device,
    )

    # Fill with local patches
    for i, p in enumerate(origins):
        full_maps[i, p[1] : p[1] + height, p[0] : p[0] + width] = maps[i]
    full_maps = full_maps[:, : -2 * padding, : -2 * padding]

    # Sparsify
    num_non_zero = (full_maps != 0.0).sum(dim=(1, 2))
    full_maps = full_maps.to_sparse().coalesce()
    return full_maps, num_non_zero


def padding_size(window_size: int):
    padding_margin = int(window_size) // 2
    return padding_margin


def pad_feature_map(feature_map: torch.Tensor, padding_size: int):
    """Zero-pad feature map by a constant padding margin.
    Args:
        feature_map: The map to extract features from.
        padding_size: The padding size.
    """
    return torch.nn.ConstantPad2d(padding_size, 0.0)(feature_map)
