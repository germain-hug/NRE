"""Run GNC-IRLS over sparse NRE maps.
"""
import logging
from math import log
from typing import Dict

import numpy as np
import torch

from ..tools.io import color
from .jacobians import compute_jacobian
from .pose import solve_delta, update_camera
from .scheduling import get_sigmas, rho_test
from .sparse import compute_cost


class GNC_IRLS:
    """GNC-IRLS class."""

    def __init__(
        self,
        lambda_min: float,
        lambda_max: float,
        device: torch.device,
        dtype: torch.dtype,
        minibatch_size: int,
    ):
        """Initialize helper class for GNC-IRLS.
        Args:
            * lambda_min: The minimum damping factor value.
            * lambda_max: The maximum damping factor value.
            * device: The device to operate on.
            * dtype: The variables dtype.
            * minibatch_size: The size of the minibatch (when computing reprojection costs).
        """
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.device = device
        self.dtype = dtype
        self.minibatch_size = minibatch_size
        self._G_x = self.G_x()
        self._G_y = self.G_y()
        self._G_z = self.G_z()
        self._id = torch.eye((3), dtype=dtype, device=device)

    def G_x(self):
        G_x = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        G_x[1, 2] = -1.0
        G_x[2, 1] = 1.0
        return G_x

    def G_y(self):
        G_y = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        G_y[0, 2] = 1.0
        G_y[2, 0] = -1.0
        return G_y

    def G_z(self):
        G_z = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        G_z[0, 1] = -1.0
        G_z[1, 0] = 1.0
        return G_z

    def prepare_sparse_inputs(
        self, sparse_nre: torch.sparse_coo_tensor, num_non_zero: torch.Tensor
    ):
        """Prepare input tensors to perform GNC-IRLS efficiently."""
        # Extract values and indices
        sparse_values = sparse_nre._values().data
        sparse_indices = sparse_nre._indices().data[-2:, :][[1, 0], :].t()

        # Placeholder tensor for efficient sparse operations
        num_non_zero_list = list(num_non_zero.cpu().numpy())
        placeholder = torch.zeros(
            len(num_non_zero_list), device=self.device, dtype=self.dtype
        )

        # Cumulative sum of non-zero values in sparse maps
        sum_indices = torch.cat(
            [
                torch.tensor([i] * s, device=self.device, dtype=torch.long)
                for i, s in enumerate(num_non_zero_list)
            ]
        )
        return {
            "sparse_values": sparse_values,
            "sparse_indices": sparse_indices,
            "placeholder": placeholder,
            "sum_indices": sum_indices,
            "num_non_zero": num_non_zero,
        }

    def run(
        self,
        p3D: torch.tensor,
        camera: Dict,
        sparse_nre: torch.sparse_coo_tensor,
        num_non_zero: torch.Tensor,
        cardinal_omega: float,
        eta: float,
        sigma_min: float,
        sigma_max: float,
        lambda_val: float = 1e-5,
        verbose=True,
    ):
        """Run GNC-IRLS over sparse NRE maps.

        Args:
            * p3D: The [N x 3] 3D keypoint world coordinates.
            * camera: Dictionary containing the initial rotation matrix and translation vector,
                as well as the intrinsics matrix and distortion coefficients.
            * sparse_nre: The sparse NRE maps.
            * num_non_zero: The tensor containing the number of non-zero values per map.
            * cardinal_omega: The cost maps cardinal.
            * sigmas: The scheduled sigma values.
            * eta: The eta ratio value.
            * sigma_min: The minimum gaussian kernel size.
            * sigma_max: The maximum gaussian kernel size.
            * lambda_val: The initial damping factor.
        Returns:
            camera: The updated camera.
            n_iter: The number of iterations the optimization ran for.
            n_updates: The number of updates applied the camera.
        """
        torch.cuda.empty_cache()
        log_cardinal = log(float(cardinal_omega))
        eye = torch.eye(6, 6, dtype=self.dtype, device=self.device)
        sigmas = get_sigmas(sigma_min, sigma_max)
        logging.info(color(f"| > Sigmas: {sigmas}"))

        # Shared parameters for sparse computation
        shared_params = self.prepare_sparse_inputs(sparse_nre, num_non_zero)
        cost_params = {
            "p3D": p3D,
            "log_cardinal": log_cardinal,
            "minibatch_size": self.minibatch_size,
        }

        # Compute initial error value and cost map gradients
        sigma_idx, sigma_val = 1, sigmas[0]
        n_iter, n_updates = 0, 0
        all_cameras = [[camera, sigma_val]]
        cost, projected_keypoints = compute_cost(
            camera=camera, sigma=sigma_val, **shared_params, **cost_params
        )
        total_cost = cost.sum()

        # Compute Jacobian matrix and reprojected keypoints for current camera
        J = compute_jacobian(
            p3D, camera, self._id, self._G_x, self._G_y, self._G_z
        )  # [N x 2 x 6]

        # Stopping criterion on lambda_max and sigma_min
        while lambda_val < self.lambda_max and sigma_idx < len(sigmas):

            # Solve for delta
            delta = solve_delta(
                J=J,
                lambda_val=lambda_val,
                sigma_val=sigma_val,
                eye=eye,
                projected_keypoints=projected_keypoints,
                log_cardinal=log_cardinal,
                minibatch_size=self.minibatch_size,
                **shared_params,
            )

            # Compute new error with the updated camera
            updated_camera = update_camera(camera, delta)
            new_cost, new_projected_keypoint = compute_cost(
                camera=updated_camera, sigma=sigma_val, **shared_params, **cost_params
            )
            new_total_cost = new_cost.sum()

            # Logging
            iteration_accepted = new_total_cost < total_cost
            if verbose and iteration_accepted:
                logging.info(
                    color(
                        "| > Iter = {:>3}, sigma = {:.1f}, lambda = {:.2e}, delta_norm = {:.2e}, cost = {:.2e}, accepted = {:}".format(
                            n_iter,
                            sigma_val,
                            lambda_val,
                            np.linalg.norm(delta.cpu().numpy().ravel()),
                            new_total_cost,
                            iteration_accepted,
                        )
                    )
                )

            if iteration_accepted:
                rho_delta = rho_test(cost, new_cost, total_cost, new_total_cost)
                if rho_delta <= eta or total_cost - new_total_cost < 1e-5:
                    sigma_val = sigmas[sigma_idx]
                    sigma_idx = min(sigma_idx + 1, len(sigmas) - 1)
                else:
                    lambda_val = max(lambda_val / 2.0, self.lambda_min)

                camera = updated_camera
                all_cameras.append([camera, sigma_val])
                J = compute_jacobian(
                    p3D, camera, self._id, self._G_x, self._G_y, self._G_z
                )
                cost = new_cost
                total_cost = new_total_cost
                projected_keypoints = new_projected_keypoint
                n_updates += 1
            else:
                lambda_val = 2.0 * lambda_val
            n_iter += 1

        return {
            "camera": camera,
            "n_iter": n_iter,
            "n_updates": n_updates,
            "cost": cost.sum(),
            "exit_sigma": sigmas[sigma_idx - 1],
            "all_cameras": all_cameras,
        }
