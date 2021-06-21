"""Methods to handle sparse NRE maps.
"""

import math
from typing import Dict

import torch

from ..tools.camera import project_keypoints


def sparsify_dense_nre_maps(nre_maps: torch.Tensor, cardinal_omega: float):
    """Sparsify dense NRE maps.
    Args:
        * nre_maps: The [N x H x W] dense NRE maps.
        * cardinal_omega: Card(Omega).
    Returns:
        * sparse_nre_maps: The sparse NRE maps.
        * num_non_zero: The number of non-zero values per map.
    """
    uniform_mask = nre_maps >= math.log(cardinal_omega)
    nre_maps[uniform_mask] = 0.0
    num_non_zero = (~uniform_mask).sum((1, 2))
    return nre_maps.to_sparse(), num_non_zero


@torch.jit.script
def gaussian_weighted_distance(distances: torch.Tensor, sigma_squared: float):
    """Apply a gaussian kernel to a tensor of 2D distances."""
    factor = 1.0 / float(2.0 * math.pi * sigma_squared)
    exp_numerator = -(torch.norm(distances, dim=1) ** 2.0)
    exp_denominator = float(2 * sigma_squared)
    gaussian_kernel = factor * torch.exp(exp_numerator / exp_denominator)
    return gaussian_kernel


@torch.jit.script
def compute_kernel_distances(
    sparse_indices: torch.Tensor,
    num_non_zero: torch.Tensor,
    keypoints: torch.Tensor,
    sigma: float,
):
    """Compute the gaussian-weighted distances between every keypoint
    and the non-zero coordinates in NRE maps.
    Args:
        * sparse_indices: The tensor of non-zero sparse indices.
        * num_non_zero: The number of non-zero values per sparse NRE map.
        * keypoints: The 2D keypoint coordinates.
        * sigma: The gaussian kernel parameter.
    """
    keypoints = keypoints.repeat_interleave(num_non_zero, dim=0)
    keypoint_differences = sparse_indices - keypoints
    kernel_distances = gaussian_weighted_distance(keypoint_differences, sigma ** 2)
    return kernel_distances, keypoint_differences


@torch.jit.script
def sparse_cost(
    sparse_values: torch.Tensor,
    sparse_indices: torch.Tensor,
    num_non_zero: torch.Tensor,
    placeholder: torch.Tensor,
    sum_indices: torch.Tensor,
    keypoints: torch.Tensor,
    log_cardinal: float,
    sigma: float,
):
    """Compute the sum of gaussian-weighted distances over
    non-zero coordinates in sparse NRE maps.
    Args:
        * sparse_values: The sparse NRE maps values.
        * sparse_indices: The sparse NRE maps indices.
        * num_non_zero: The tensor containing the number of non-zero values per map.
        * placeholder: A placeholder tensor to contain weighted kernel.
        * sum_indices: The indices used to compute the weighted kernel.
        * keypoints: The coordinates of the reprojected keypoints.
        * log_cardinal: The log of the cost maps cardinal.
        * sigma: The gaussian kernel parameter.
    Returns:
        * cost: The tensor of summed cost values per NRE map.
    """
    kernel_distances = compute_kernel_distances(
        sparse_indices, num_non_zero, keypoints, sigma
    )[0]
    cost = -kernel_distances * (log_cardinal - sparse_values)
    cost = placeholder.clone().scatter_add_(0, sum_indices, cost.to(placeholder))
    return cost


@torch.jit.script
def compute_t_qn_and_r_qn(
    sparse_values: torch.Tensor,
    sparse_indices: torch.Tensor,
    num_non_zero: torch.Tensor,
    keypoints: torch.Tensor,
    log_cardinal: float,
    sigma: float,
):
    """Compute 't_qn'."""
    kernel_distances, keypoint_differences = compute_kernel_distances(
        sparse_indices, num_non_zero, keypoints, sigma
    )
    t_qn = kernel_distances * (log_cardinal - sparse_values) / sigma ** 2
    return t_qn, keypoint_differences


@torch.jit.script
def compute_cost(
    sparse_values: torch.Tensor,
    sparse_indices: torch.Tensor,
    num_non_zero: torch.Tensor,
    placeholder: torch.Tensor,
    sum_indices: torch.Tensor,
    p3D: torch.Tensor,
    camera: Dict[str, torch.Tensor],
    log_cardinal: float,
    sigma: float,
    minibatch_size: int,
):
    """Compute the NRE costs at the current camera pose estimate (using minibatches).
    Args:
        * sparse_values: The sparse values of the [N x H x W] NRE maps.
        * sparse_indices: The indices of the sparse values.
        * num_non_zero: The tensor containing the number of non-zero values per map.
        * placeholder: An empty tensor used for an efficient computation of the weighted values.
        * sum_indices: The number of non-zero values per cost map.
        * p3D: The [N x 3] 3D keypoint world coordinates.
        * camera: Dictionary containing the initial rotation matrix and translation vector,
            as well as the intrinsics matrix and distortion coefficients.
        * log_cardinal: The log of the cost map cardinal.
        * sigma: The gaussian kernel sigma parameter.
        * minibatch_size: The size of the minibatch.
    Returns:
        * cost: The [N x 1] cost terms.
        * projected_keypoints: The [N x 2] projected keypoint coordinates at the current pose.
    """
    N = len(p3D)
    projected_keypoints, positive_z = project_keypoints(
        p3D, camera["R"], camera["t"], camera["K"], camera["dist_coeffs"]
    )
    cost = torch.zeros((N), dtype=p3D.dtype, device=p3D.device)

    start_idx, end_idx = 0, minibatch_size
    while start_idx < N:
        sub_num_non_zero = num_non_zero[start_idx:end_idx]
        if sub_num_non_zero.sum() > 0:

            # Fetch minibatch inputs
            sub_placeholder = placeholder[start_idx:end_idx]
            sparse_start_idx = torch.sum(num_non_zero[:start_idx])
            sparse_end_idx = sparse_start_idx + sub_num_non_zero.sum()
            sub_sum_indices = (
                sum_indices[sparse_start_idx:sparse_end_idx]
                - sum_indices[sparse_start_idx]
            )

            # NRE cost for current minibatch
            cost[start_idx:end_idx] = sparse_cost(
                sparse_values=sparse_values[sparse_start_idx:sparse_end_idx],
                sparse_indices=sparse_indices[sparse_start_idx:sparse_end_idx],
                num_non_zero=sub_num_non_zero,
                placeholder=sub_placeholder,
                sum_indices=sub_sum_indices,
                keypoints=projected_keypoints[start_idx:end_idx],
                log_cardinal=log_cardinal,
                sigma=sigma,
            )

        # Update minibatch indices
        start_idx += minibatch_size
        end_idx = min(start_idx + minibatch_size, N)

    # For negative z, set value zero
    cost[~positive_z] = 0.0

    return cost.view(N), projected_keypoints
