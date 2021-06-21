"""Solve for the camera pose using P3P.
"""
from math import log

import cv2
import numpy as np
import torch

from .sampling import entropy_based_sampling
from .costs import reprojection_cost
from ..tools.camera import project_keypoints


@torch.no_grad()
def run_p3p(
    cost_maps: torch.Tensor,
    argmax2D: torch.Tensor,
    entropy: torch.Tensor,
    top_N: torch.Tensor,
    p3D: torch.Tensor,
    K: torch.Tensor,
    dist_coeffs: torch.Tensor,
    cardinal_omega: float,
    n_iter: int,
):
    """Run P3P from dense correspondence maps, by sampling triplets of argmax.

    Args:
        cost_maps: The [N x H x W] dense cost maps.
        argmax2D: The [N x 2] pre-computed argmax coordinates.
        entropy: The [N x 1] correspondence maps entropy values.
        top_N: The [N x 1] binary mask of best top-k keypoints to use.
        p3D: The [N x 3] landmark coordinates.
        K: The [3 x 3] target image K matrix.
        dist_coeffs: The target image distortion coefficients array.
        cardinal_omega: The value used for padding.
        n_iter: The number of iterations.
    Returns:
        model: The estimated camera model.
    """
    assert len(p3D) == len(cost_maps)
    padding = log(float(cardinal_omega))

    # Pre-sample cost_maps based on their entropy
    cmap_sampler = entropy_based_sampling(entropy[top_N], n_iter)
    random_indices = torch.stack(
        [torch.tensor(list(cmap_sampler)) for _ in range(3)], dim=1
    )

    # Prepare P3P inputs
    best_argmax2D = np.ascontiguousarray(argmax2D[top_N].cpu()[:, None]).astype(
        np.float32
    )

    # Begin iterating
    K_npy = K.float().cpu().numpy()
    dist_npy = dist_coeffs.cpu().numpy()
    best_p3D = p3D[top_N].float()
    best_p3D_npy = best_p3D.cpu().numpy()
    best_cost_maps = cost_maps[top_N]
    best_cost, best_pose, best_iter = None, None, 0

    for i in range(n_iter):

        # Sample an argmax triplet
        random_idx = random_indices[i]

        # Perform P3P from 2D-3D matches
        retval, rvecs, tvecs = cv2.solveP3P(
            objectPoints=best_p3D_npy[random_idx],
            imagePoints=best_argmax2D[random_idx],
            cameraMatrix=K_npy,
            distCoeffs=dist_npy,
            flags=cv2.SOLVEPNP_P3P,
        )

        # Compute the cost for the given candidates
        if retval:
            for rvec, tvec in zip(rvecs, tvecs):
                R = torch.from_numpy(cv2.Rodrigues(rvec)[0]).to(best_p3D)
                t = torch.from_numpy(tvec).to(best_p3D).squeeze(-1)
                cost = compute_cost(
                    best_cost_maps, best_p3D, R, t, K, dist_coeffs, padding
                )
                # Update model
                if best_cost is None or cost < best_cost:
                    best_iter = i
                    best_cost = cost
                    best_pose = [rvec, tvec]

    if best_pose is None:
        return {"best_iter": "Failed."}
    return {
        "rvec": best_pose[0].astype(np.float32),
        "tvec": np.reshape(best_pose[1].astype(np.float32), (3,)),
        "cost": best_cost,
        "best_iter": best_iter,
    }


@torch.jit.script
def compute_cost(
    cost_maps: torch.Tensor,
    p3D: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    padding_value: float,
):
    """Compute the reprojection cost for a given camera pose."""
    p2D, visible = project_keypoints(p3D, R, t, K, dist)
    score = reprojection_cost(cost_maps, p2D, padding_value, visible)
    return score
