"""Methods to compute Jacobian matrices.
"""
import torch
from typing import Dict


@torch.jit.script
def distortion_derivative(ux: torch.Tensor, uy: torch.Tensor, k: torch.Tensor):
    """Compute the radial distortion derivative term.
    Args:
        ux: The [N x 1] x coordinates of the points in the camera plane.
        uy: The [N x 1] y coordinates of the points in the camera plane.
        k: The radial distortion value.
    Returns:
        The [N x 3 x 3] derivative matrix.
    """
    derivative = torch.zeros((len(ux), 3, 3), dtype=ux.dtype, device=ux.device)
    derivative[:, 0, 0] = 1 + k * (3 * ux ** 2 + uy ** 2)
    derivative[:, 1, 1] = 1 + k * (3 * uy ** 2 + ux ** 2)
    derivative[:, 0, 1] = 2 * k * ux * uy
    derivative[:, 1, 0] = 2 * k * ux * uy
    derivative[:, 2, 2] = 1.0
    return derivative


@torch.jit.script
def projection_derivatives(
    p3D: torch.Tensor,
    intrinsics: torch.Tensor,
    radial_distortion: torch.Tensor,
):
    """Compute the derivatives of the projection function w.r.t the 3D points.
    Args:
        p3D: The [N x 3] 3D points in camera coordinates.
        intrinsics: The [3 x 3] intrinsics matrix.
        radial_distortion: The radial distortion value.
    Return:
        term: The projection derivative matrix for each point, of size [N x 3 x 3]
    """
    N = len(p3D)

    # World-to-camera projection
    derivative = torch.zeros((N, 3, 3), dtype=p3D.dtype, device=p3D.device)
    u_x = p3D[:, 0]
    u_y = p3D[:, 1]
    u_z = p3D[:, 2]
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 1.0
    derivative[:, 0, 2] = -u_x / u_z
    derivative[:, 1, 2] = -u_y / u_z
    derivative /= u_z[..., None, None]

    # Radial distortion
    radial_distortion_derivative = distortion_derivative(
        u_x / u_z, u_y / u_z, radial_distortion
    )
    derivative = torch.bmm(radial_distortion_derivative, derivative)

    # Camera to image plane
    derivative = torch.bmm(intrinsics[None].repeat((N, 1, 1)), derivative)
    return derivative


@torch.jit.script
def pose_derivative(
    id_: torch.Tensor,
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    G_z: torch.Tensor,
    rotated_points: torch.Tensor,
):
    """Compute the projection Jacobian right hand term.
    Return:
        term: The [N x 3 x 6] right-hand term for each points
    """
    N = len(rotated_points)
    rotated_points = rotated_points.unsqueeze(-1)
    x_1 = torch.bmm(G_x.repeat((N, 1, 1)), rotated_points)
    x_2 = torch.bmm(G_y.repeat((N, 1, 1)), rotated_points)
    x_3 = torch.bmm(G_z.repeat((N, 1, 1)), rotated_points)
    id_ = id_.repeat((N, 1, 1))
    return torch.cat([x_1, x_2, x_3, id_], dim=-1)


@torch.jit.script
def compute_jacobian(
    p3D: torch.Tensor,
    camera: Dict[str, torch.Tensor],
    id_: torch.Tensor,
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    G_z: torch.Tensor,
):
    """Compute the camera reprojection Jacobian matrix.
    Args:
        p3D: The [N x 3] 3D keypoint world coordinates.
        * camera: Dictionary containing the initial rotation matrix and translation vector,
            as well as the intrinsics matrix and distortion coefficients.
    Returns:
        jacobian: The [N x 2 x 6] jacobian matrix.
    """
    # Compute 3D points in camera coordinates
    rotated_points = (camera["R"] @ p3D.t()).t()
    camera_landmarks = rotated_points + camera["t"].view(1, 3)

    # Compute the Jacobian
    proj_derivatives = projection_derivatives(
        camera_landmarks, camera["K"], camera["dist_coeffs"][0]
    )
    p_derivative = pose_derivative(id_, G_x, G_y, G_z, rotated_points)
    projection_jacobian = torch.bmm(proj_derivatives, p_derivative)
    assert projection_jacobian.shape[-1] == 6
    assert projection_jacobian.shape[-2] == 3
    return projection_jacobian[:, :2, :]
