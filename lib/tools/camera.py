from typing import Tuple

import numpy as np
import torch


def to_homogeneous(points: torch.Tensor):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = torch.ones_like(points[..., :1])
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points: torch.Tensor, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


@torch.jit.script
def undistort_keypoints(keypoints: torch.Tensor, params: torch.Tensor):
    """Undistort keypoints using radial and decentering coefficients.
    Args:
        * keypoints: The image-space 2D keypoint coordinates, of size [N x 2].
        * params: The (k1, k2) radial and (p1, p2) distortion coefficients,
            of size [4 x 1].
    """
    k1 = params[0]
    k2 = params[1]
    p12 = params[2:][None].squeeze(-1)
    r2 = torch.sum(keypoints ** 2, -1, keepdim=True)
    uv = torch.prod(keypoints, -1, keepdim=True)
    radial = k1[None] * r2 + k2[None] * r2 ** 2
    undistorted_keypoints = (
        keypoints * (1 + radial)
        + 2 * p12 * uv
        + p12.flip(-1) * (r2 + 2 * keypoints ** 2)
    )
    return undistorted_keypoints


@torch.jit.script
def project_keypoints(
    p3D: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K: torch.Tensor,
    distortion: torch.Tensor,
    eps: float = 1e-4,
):
    """Project keypoints in from the 3D world to the image space.
    Args:
        * p3D: The 3D keypoint coordinates, of size [N x 3].
        * R: The [3 x 3] rotation matrix R (camera-to-world).
        * t: The [3 x 1] translation vector (camera-to-world).
        * K: The [3 x 3] intrinsics matrix.
        * distortion: The [4 x 1] distortion coefficients.
    Returns:
        * keypoints: The 2D keypoint coordinates in image-space.
        * visible: The mask of keypoints with a positive z-depth.
    """
    camera_p3D = p3D @ R.t() + t
    visible = camera_p3D[..., -1] >= eps
    camera_p3D[..., -1].clamp_(min=eps)
    keypoints = from_homogeneous(camera_p3D)
    keypoints = undistort_keypoints(keypoints, distortion)
    keypoints = keypoints @ K[:2, :2] + K[:2, 2]
    return keypoints, visible


def inbound_keypoints(
    keypoints: torch.Tensor,
    image_size: torch.Tensor,
    pad_l: int = 0,
    pad_r: int = 0,
    pad_u: int = 0,
    pad_b: int = 0,
):
    """Return the mask of keypoints that lie within the image borders."""
    if isinstance(keypoints, np.ndarray):
        return np.all(
            (keypoints >= np.array([pad_l, pad_u]))
            & (keypoints <= (image_size - 1 - np.array([pad_r, pad_b]))),
            -1,
        )
    device = keypoints.device
    dtype = keypoints.dtype
    return torch.all(
        (keypoints >= torch.tensor([pad_l, pad_u], device=device, dtype=dtype))
        & (
            keypoints
            <= (
                image_size
                - 1
                - torch.tensor([pad_r, pad_b], device=device, dtype=dtype)
            )
        ),
        -1,
    )


def scale_intrinsics(K: np.ndarray, scales: Tuple[float]):
    """Rescale linear calibration matrix K."""
    if isinstance(scales, torch.Tensor):
        scales = list(scales.view(2))
    if isinstance(scales, (int, float)):
        scales = (scales, scales)
    T = np.diag(np.r_[scales, [1.0]])
    T[[0, 1], [2, 2]] = (np.array(scales) - 1) / 2
    if isinstance(K, torch.Tensor):
        T = torch.from_numpy(T).to(K)
    else:
        T = T.astype(K.dtype)
    return T @ K
