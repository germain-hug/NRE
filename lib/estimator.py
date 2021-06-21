"""The NRE-based camera pose estimator.
"""
import logging
import time
from copy import deepcopy
from typing import Dict

import cv2
import torch

from .net.net import Net
from .optim.gnc_irls import GNC_IRLS
from .optim.sparse import sparsify_dense_nre_maps
from .solvers import p3p
from .tools.camera import inbound_keypoints, project_keypoints, scale_intrinsics
from .tools.image import read_image
from .tools.io import color, load_config
from .tools.math import log
from .tools.metrics import display_pose_error
from .tools.tensor_ops import to_device, to_dtype
from .tools.tensor_shape import compute_input_shape

torch.set_grad_enabled(False)


class NREEstimator:
    """The NRE-based pose estimator class, to run P3P + GNC-IRLS camera pose estimation and refinement."""

    def __init__(
        self,
        coarse_checkpoint: str,
        fine_checkpoint: str,
        config_file: str,
        nn_device: torch.device = None,
        p3p_device: torch.device = None,
        irls_device: torch.device = None,
    ):
        """Initialize the NRE-based estimator.
        Args:
            * coarse_checkpoint: The path to the coarse checkpoint model.
            * fine_checkpoint: The path to the fine checkpoint model.
            * config_file: The path to the yaml config file.
            * nn_device: The pytorch device to perform neural network computation on.
            * p3p_device: The pytorch device to perform P3P on.
            * irls_device: The pytorch device to perform IRLS on.
        """
        # Setup torch device
        if None in [nn_device, p3p_device, irls_device]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if nn_device is None:
                nn_device = device
            if p3p_device is None:
                p3p_device = device
            if irls_device is None:
                irls_device = device

        # Load models
        self.coarse_network = (
            Net.load_from_checkpoint(coarse_checkpoint).to(nn_device).eval()
        )
        if fine_checkpoint is not None:
            self.fine_network = (
                Net.load_from_checkpoint(fine_checkpoint).to(nn_device).eval()
            )
        else:
            self.fine_network = None

        # Load config
        self.cfg = load_config(config_file)
        if self.cfg.window_size % 16 != 0 or self.cfg.window_size < 16:
            raise ValueError("Window size should be a multiple of 16.")
        self.set_max_image_sizes()
        self.nn_device = nn_device
        self.p3p_device = p3p_device
        self.irls_device = irls_device
        self.gnc_irls = GNC_IRLS(
            lambda_min=1e-8,
            lambda_max=1e7,
            device=irls_device,
            dtype=torch.float32,
            minibatch_size=self.cfg.gnc_minibatch,
        )

    @torch.no_grad()
    def localize(
        self,
        source_image: str,
        target_image: str,
        p3D: torch.Tensor,
        source_P: torch.Tensor,
        source_K: torch.Tensor,
        source_dist: torch.Tensor,
        target_K: torch.Tensor,
        target_dist: torch.Tensor,
        gt_pose: torch.Tensor = None,
        use_re: bool = False,
    ):
        """Given an image pair and 3D point coordinates, predict the target
        image camera pose.
        Args:
            * source_image: The path to the source image.
            * target_image: The path to the target image.
            * p3D: The [N x 3] 3D keypoint coordinates.
            * source_P: The [4 x 4] source camera pose.
            * source_K: The [3 x 3] source calibration matrix.
            * source_dist: The [4 x 1] source dist coefficients.
            * target_K: The [3 x 3] target calibration matrix.
            * target_dist: The [4 x 1] target dist coefficients.
            * gt_pose: (Optional) Provide the ground truth target pose
                to obtain its coarse reprojection error.
            * use_re: If True, maps are made of a single value at the
                argmax location.
        Returns:
            * target_P: The [4 x 4] estimated pose.
        """
        # Read and resize input images for the coarse model
        source_tensor, source_coarse_K = read_image(
            source_image,
            source_K,
            self.max_coarse_image_size,
        )
        target_tensor, target_coarse_K = read_image(
            target_image,
            target_K,
            self.max_coarse_image_size,
        )

        # Move tensors to GPU
        source_tensor = to_device(source_tensor, self.nn_device).float()[None]
        target_tensor = to_device(target_tensor, self.nn_device).float()[None]
        (
            p3D,
            source_P,
            source_coarse_K,
            source_dist,
            target_coarse_K,
            target_dist,
        ) = to_device(
            [p3D, source_P, source_coarse_K, source_dist, target_coarse_K, target_dist],
            self.p3p_device,
        )

        # Compute source 2D keypoints in the coarse image space
        source_coarse_p2D, visible = project_keypoints(
            p3D,
            source_P[:3, :3],
            source_P[:3, 3],
            source_coarse_K,
            source_dist,
        )

        # Compute dense coarse NRE maps
        start_time = time.perf_counter()
        nre_dict = self.coarse_network(
            {
                "source_image_tensor": source_tensor,
                "target_image_tensor": target_tensor,
                "source_keypoints": source_coarse_p2D[visible].to(self.nn_device),
            },
            target_device=self.p3p_device,
            compute_argmax=True,
            compute_entropy=True,
        )
        logging.info(
            color(
                "| Coarse forward took {:.1f}ms".format(
                    1e3 * (time.perf_counter() - start_time)
                )
            )
        )

        # (Optional) Use the standard reprojection error RE
        if use_re:
            nre_maps = torch.full_like(
                nre_dict["nre_maps"], log(nre_dict["cardinal_omega"])
            )
            for i, kp in enumerate(nre_dict["argmax_2D"]):
                nre_maps[i, kp[1], kp[0]] = 1e-3
            nre_dict["nre_maps"] = nre_maps

        # Rescale target intrinsics
        target_coarse_nre_K = scale_intrinsics(
            target_coarse_K, nre_dict["target_im2feat"]
        )

        # Perform coarse P3P
        logging.info(
            color(
                f"| Running P3P for {self.cfg.n_iter_p3p} iterations on {len(p3D[visible])} keypoints"
            )
        )
        coarse_pose_P3P = self.P3P(
            nre_dict, p3D[visible], target_coarse_nre_K, target_dist
        )

        # Compute normalization factor in the coarse map
        norm_coarse = self.compute_norm_coarse(
            nre_dict,
            p3D[visible],
            coarse_pose_P3P,
            target_coarse_nre_K,
            target_dist,
        )

        # (Optional) Skip GNC step
        if self.cfg.skip_gnc:
            output = {
                "P3P": {
                    "pose": {
                        "R": cv2.Rodrigues(coarse_pose_P3P["rvec"])[0],
                        "t": coarse_pose_P3P["tvec"],
                    }
                },
            }
            return output

        # (Optional) When ground-truth is available, compute cost at GT pose
        gt_cost = None
        if gt_pose is not None:
            gt_cost = self.compute_cost(
                nre_dict,
                p3D[visible],
                to_device(gt_pose, self.p3p_device),
                target_coarse_nre_K,
                target_dist,
            )

        # Sparsify coarse NRE maps + move to cpu
        nre_dict["sparse_nre"], nre_dict["num_non_zero"] = to_device(
            sparsify_dense_nre_maps(nre_dict["nre_maps"], nre_dict["cardinal_omega"]),
            self.irls_device,
        )
        del nre_dict["nre_maps"]
        torch.cuda.empty_cache()

        sigma_max = self.cfg.p3p_sigma_max
        sigma_min = self.cfg.p3p_sigma_min
        init_camera = to_device(
            {
                "R": cv2.Rodrigues(coarse_pose_P3P["rvec"])[0],
                "t": coarse_pose_P3P["tvec"],
                "K": target_coarse_nre_K,
                "dist_coeffs": target_dist,
            },
            self.irls_device,
        )

        # Perform coarse GNC-IRLS
        logging.info(
            color(
                f"| Running Coarse GNC-IRLS"
                f"[sigma_max={sigma_max}"
                f"/sigma_min={sigma_min}]"
            )
        )
        coarse_pose_IRLS = self.GNC_IRLS(
            nre_dict,
            p3D[visible].to(self.irls_device),
            init_camera,
            sigma_min,
            sigma_max,
        )

        # Cleanup coarse predictions
        del nre_dict, source_tensor, target_tensor
        torch.cuda.empty_cache()

        # Skip fine network
        output = {
            "P3P": {"pose": init_camera, "cost": coarse_pose_P3P["cost"]},
            "coarse_irls": {
                "pose": to_dtype(coarse_pose_IRLS["camera"], torch.float32),
                "cost": coarse_pose_IRLS["cost"],
            },
            "gt_cost": gt_cost,
        }
        if self.fine_network is None:
            return output

        # Read images for the fine model
        source_tensor, source_fine_K = read_image(
            source_image,
            source_K,
            self.cfg.max_fine_imsize,
        )
        target_tensor, target_fine_K = read_image(
            target_image,
            target_K,
            self.cfg.max_fine_imsize,
        )

        # Move to NN device
        source_tensor = to_device(source_tensor, self.nn_device).float()[None]
        target_tensor = to_device(target_tensor, self.nn_device).float()[None]
        p3D = to_device(p3D, self.irls_device)
        source_P = to_device(source_P, self.irls_device)
        source_fine_K = to_device(source_fine_K, self.irls_device)
        target_fine_K = to_device(target_fine_K, self.irls_device)
        source_dist = to_device(source_dist, self.irls_device)

        # Compute source 2D keypoints in the fine image space
        source_fine_p2D, visible = project_keypoints(
            p3D,
            source_P[:3, :3],
            source_P[:3, 3],
            source_fine_K,
            source_dist,
        )

        # Compute the prior target 2D keypoint locations
        prior_target_keypoints = project_keypoints(
            p3D,
            to_dtype(coarse_pose_IRLS["camera"]["R"], p3D.dtype),
            to_dtype(coarse_pose_IRLS["camera"]["t"], p3D.dtype),
            target_fine_K,
            target_dist,
        )[0]

        # Only keep prior keypoints with valid reprojections
        target_imsize = torch.tensor(
            target_tensor.shape[-2:][::-1],
            device=prior_target_keypoints.device,
            dtype=prior_target_keypoints.dtype,
        )
        borders = [self.cfg.window_size // 2] * 4
        visible &= inbound_keypoints(prior_target_keypoints, target_imsize, *borders)

        if visible.sum() == 0:
            logging.warning("No in-plane keypoints for fine GNC-IRLS.")
            return output

        # Compute sparse fine cost maps
        start_time = time.perf_counter()
        nre_dict = self.fine_network(
            {
                "source_image_tensor": source_tensor,
                "target_image_tensor": target_tensor,
                "source_keypoints": source_fine_p2D[visible].to(self.nn_device),
                "norm_coarse": norm_coarse[visible],
            },
            target_device=self.irls_device,
            sparsify=True,
            prior_keypoints=prior_target_keypoints[visible],
            window_size=self.cfg.window_size,
        )
        logging.info(
            color(
                "| Fine forward took {:.1f}ms".format(
                    1e3 * (time.perf_counter() - start_time)
                )
            )
        )
        target_fine_nre_K = scale_intrinsics(target_fine_K, nre_dict["target_im2feat"])

        # Perform fine GNC-IRLS
        logging.info(
            color(
                f"| Running Fine GNC-IRLS"
                f"[sigma_max={self.cfg.fine_sigma_max}"
                f"/sigma_min={self.cfg.fine_sigma_min}]"
            )
        )
        fine_irls_init_camera = deepcopy(coarse_pose_IRLS["camera"])
        fine_irls_init_camera["K"] = target_fine_nre_K
        fine_pose_IRLS = self.GNC_IRLS(
            nre_dict,
            p3D[visible].to(self.irls_device),
            fine_irls_init_camera,
            self.cfg.fine_sigma_min,
            self.cfg.fine_sigma_max,
        )

        output["fine_irls"] = {
            "pose": to_dtype(fine_pose_IRLS["camera"], torch.float32),
            "cost": fine_pose_IRLS["cost"],
        }
        return output

    def P3P(
        self,
        nre_dict: Dict,
        p3D: torch.Tensor,
        target_K: torch.Tensor,
        target_dist: torch.Tensor,
    ):
        """Perform coarse P3P camera pose estimation."""
        start_time = time.perf_counter()

        # Retain top-N keypoints based on map peak
        if self.cfg.top_n_p3p < 1.0:
            N = len(nre_dict["nre_maps"])
            sorted_idx = (
                nre_dict["nre_maps"].view(N, -1).min(dim=1).values.argsort(dim=0)
            )
            top_N = sorted_idx[: int(N * self.cfg.top_n_p3p)]
        else:
            top_N = torch.ones((len(p3D)), dtype=torch.bool, device=self.p3p_device)

        # Run P3P
        pose_dict = p3p.run_p3p(
            cost_maps=nre_dict["nre_maps"],
            argmax2D=nre_dict["argmax_2D"],
            entropy=nre_dict["entropy"],
            top_N=top_N,
            p3D=p3D,
            K=target_K,
            dist_coeffs=target_dist,
            cardinal_omega=nre_dict["cardinal_omega"],
            n_iter=self.cfg.n_iter_p3p,
        )
        logging.info(color(f"| > Best P3P iteration: {pose_dict['best_iter']}"))
        logging.info(
            color(
                "| > P3P took {:.1f}ms".format(1e3 * (time.perf_counter() - start_time))
            )
        )
        return pose_dict

    def GNC_IRLS(
        self,
        nre_dict: Dict[str, torch.Tensor],
        p3D: torch.Tensor,
        init_camera: Dict[str, torch.Tensor],
        sigma_min: float,
        sigma_max: float,
    ):
        """Perform GNC-IRLS."""
        start_time = time.perf_counter()
        pose_dict = self.gnc_irls.run(
            p3D=to_dtype(p3D, self.gnc_irls.dtype),
            camera=to_dtype(init_camera, self.gnc_irls.dtype),
            sparse_nre=to_dtype(nre_dict["sparse_nre"], self.gnc_irls.dtype),
            num_non_zero=nre_dict["num_non_zero"],
            cardinal_omega=nre_dict["cardinal_omega"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            eta=0.1,
            lambda_val=1e-8,
            verbose=True,
        )
        logging.info(
            color(
                f"| > Exit sigma={pose_dict['exit_sigma']} "
                f"/ {pose_dict['n_updates']} updates"
                f"/ {pose_dict['n_iter']} iterations"
            )
        )
        logging.info(
            color(
                f"| > Pose delta: {display_pose_error(pose_dict['camera'], to_dtype(init_camera, self.gnc_irls.dtype))}"
            )
        )
        logging.info(
            color(
                "| > GNC-IRLS took {:.1f}ms".format(
                    1e3 * (time.perf_counter() - start_time)
                )
            )
        )
        return pose_dict

    def compute_cost(
        self,
        nre_dict: torch.Tensor,
        p3D: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        dist_coeffs: torch.Tensor,
    ):
        """Compute the NRE cost at a given pose using coarse NRE maps."""
        return p3p.compute_cost(
            nre_dict["nre_maps"],
            p3D,
            pose[:3, :3],
            pose[:3, 3],
            K,
            dist_coeffs,
            log(nre_dict["cardinal_omega"]),
        )

    def set_max_image_sizes(self):
        """Pre-compute the maximum image sizes."""
        size = self.coarse_network.image_cfg["resize"]
        self.max_coarse_image_size = max(
            compute_input_shape(self.coarse_network.cnn_layers, [size, size])
        )

    def compute_norm_coarse(
        self,
        nre_dict: Dict,
        p3D: torch.Tensor,
        coarse_pose_P3P,
        target_coarse_nre_K,
        target_dist,
    ):
        """Compute the sum of neighbouring values in a correspondence map."""
        p2D = project_keypoints(
            p3D,
            torch.from_numpy(cv2.Rodrigues(coarse_pose_P3P["rvec"])[0]).to(p3D.device),
            torch.from_numpy(coarse_pose_P3P["tvec"]).to(p3D.device),
            target_coarse_nre_K,
            target_dist,
        )[0]
        norm_coarse = torch.zeros((len(p2D)), device=nre_dict["cmaps"].device)
        p = self.cfg.window_size // 16
        for i, (cmap, kp) in enumerate(zip(nre_dict["cmaps"], p2D.round())):
            x, y = int(kp[0].item()), int(kp[1].item())
            norm_coarse[i] = cmap[y - p : y + p, x - p : x + p].sum()
        return norm_coarse
