"""Solve for delta.
"""

from copy import deepcopy
from typing import Dict

import torch

from .sparse import compute_t_qn_and_r_qn


def skew_symmetric(v: torch.Tensor):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))
    return M


def so3exp_map(w: torch.Tensor, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def update_camera(camera: Dict[str, torch.Tensor], delta: torch.Tensor):
    """Update the camera pose given a delta in SO3."""
    new_camera = deepcopy(camera)
    new_camera["R"] = so3exp_map(delta[:3].view(1, 3))[0] @ new_camera["R"]
    new_camera["t"] = camera["t"] + delta[3:].view(3)
    return new_camera


def solve_delta(
    J: torch.Tensor,
    lambda_val: float,
    sigma_val: float,
    eye: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    placeholder: torch.Tensor,
    num_non_zero: torch.Tensor,
    sum_indices: torch.Tensor,
    projected_keypoints: torch.Tensor,
    log_cardinal: float,
    minibatch_size: int,
):
    """Solve for the camera pose increment delta."""
    chunk_start, chunk_end = 0, minibatch_size
    num_points = len(J)
    lh_term = torch.zeros(
        (6, 6), dtype=sparse_values.dtype, device=sparse_values.device
    )
    rh_term = torch.zeros((6, 1), dtype=lh_term.dtype, device=lh_term.device)
    while chunk_start < num_points:
        sub_J = J[chunk_start:chunk_end]
        sub_num_non_zero = num_non_zero[chunk_start:chunk_end]
        if sub_num_non_zero.sum() > 0:

            # Fetch minibatch inputs
            sub_placeholder = placeholder[chunk_start:chunk_end]
            sparse_start_idx = torch.sum(num_non_zero[:chunk_start])
            sparse_end_idx = sparse_start_idx + sub_num_non_zero.sum()
            sub_sum_indices = (
                sum_indices[sparse_start_idx:sparse_end_idx]
                - sum_indices[sparse_start_idx]
            )

            # Compute t_qn, r_qn for the current minibatch
            sub_t_qn, sub_r_qn = compute_t_qn_and_r_qn(
                sparse_values=sparse_values[sparse_start_idx:sparse_end_idx],
                sparse_indices=sparse_indices[sparse_start_idx:sparse_end_idx],
                num_non_zero=sub_num_non_zero,
                keypoints=projected_keypoints[chunk_start:chunk_end],
                log_cardinal=log_cardinal,
                sigma=sigma_val,
            )

            # Assemble left-hand term
            J_transpose_J = torch.bmm(sub_J.transpose(1, 2), sub_J)  # [N x 6 x 6]
            multiplicative_term = sub_placeholder.clone().scatter_add_(
                0, sub_sum_indices, sub_t_qn
            )
            lh_term += (
                J_transpose_J * multiplicative_term.unsqueeze(-1).unsqueeze(-1)
            ).sum(
                0
            )  # [6 x 6]

            # Assemble right-hand term
            interleaved_Jt = sub_J.transpose(1, 2).repeat_interleave(
                sub_num_non_zero, dim=0
            )  # [M x 6 x 2]
            J_transpose_r_qn = torch.bmm(
                interleaved_Jt, sub_r_qn.unsqueeze(-1)
            )  # [M x 6 x 1]
            rh_term += (J_transpose_r_qn * sub_t_qn.unsqueeze(-1).unsqueeze(-1)).sum(
                0
            )  # [6 x 1]

        # Update chunk indices
        chunk_start += minibatch_size
        chunk_end = min(chunk_start + minibatch_size, num_points)

    lh_term += eye * lambda_val
    delta = torch.mm(torch.inverse(lh_term), rh_term)
    return delta
