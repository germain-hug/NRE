import math
from typing import Dict

import torch


def pose_error(R0: torch.Tensor, t0: torch.Tensor, R1: torch.Tensor, t1: torch.Tensor):
    """Compute the rotation and translation error.
    Args:
        * R0: The [3 x 3] first rotation matrix.
        * t0: The [3] first translation vector.
        * R1: The [3 x 3] second rotation matrix.
        * t1: The [3] second translation vector.
    Returns:
        * The rotation (in degrees) and translation error.
    """
    dt = torch.norm(t0.ravel() - t1.ravel(), p=2, dim=-1)
    trace = torch.diagonal(R0.transpose(-1, -2) @ R1, dim1=-1, dim2=-2).sum(-1)
    cos = torch.clamp((trace - 1) / 2, -1, 1)
    dr = torch.acos(cos).abs() / math.pi * 180.0
    return dr, dt


def display_pose_error(cam0: Dict, cam1: Dict):
    """Display the pose error between two cameras."""
    try:
        dr, dt = pose_error(cam0["R"], cam0["t"], cam1["R"], cam1["t"])
        error = f"dr={dr:.2f}° dt={dt:.2f}m"
        return error
    except KeyError:
        print("Unexpected camera format.")
