"""The NRE convolutional model.
"""
import logging
from typing import Dict

import pytorch_lightning as pl
import torch

from ..backbones.build import build_model
from ..tools.math import entropy, log
from ..tools.tensor_ops import argmax_2D
from .nre import (
    cardinal_omega,
    compute_correlation_maps,
    compute_local_nre_maps,
    downsample_keypoints,
    interpolate,
    relative_aspect_ratio,
    softmax,
)


class Net(pl.LightningModule):
    """The feature extraction model."""

    def __init__(self, **kwargs: Dict):
        """Initialize the network."""
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.cnn, self.descriptor_dimension = build_model(self.hparams)
        logging.info(
            f"Initialized {self.hparams.level} model "
            f"with output dimension {self.descriptor_dimension}"
        )

    def compute_cmaps(
        self,
        data: Dict,
        return_presoftmax: bool = False,
    ):
        """Compute dense correspondence maps.
        Args:
            * data: Dictionary containing input data. Expected keys include
                source_image_tensor, target_image_tensor and source_keypoints.
            * return_presoftmax: If True, pre-softmax maps are returned
                (e.g. for visualization purposes)
        Returns:
            * cmaps: The [N x H x W] correspondence maps tensor.
            * target_im2feat: The target image-to-feature downsampling ratio.
            * cardinal_omega: Card(Omega) of the maps.
        """
        # Compute dense feature maps
        (
            dense_source_features,
            dense_target_features,
            source_keypoints,
        ) = self.compute_dense_feature(data)

        # Compute the target feature-to-image upsampling ratio
        target_im2feat = relative_aspect_ratio(
            data["target_image_tensor"], dense_target_features
        )

        # Interpolate sparse source descriptors
        sparse_source_features = interpolate(dense_source_features, source_keypoints)

        # Compute dense correlation maps
        correlation_maps = compute_correlation_maps(
            source_descriptors=sparse_source_features,
            target_features=dense_target_features,
            num_points=len(source_keypoints),
            minibatch_size=self.hparams.minibatch_size,
        )

        # Normalize to obtain correspondence maps
        correspondence_maps = softmax(correlation_maps)
        card_omega = cardinal_omega(dense_target_features)

        output = {
            "cmaps": correspondence_maps.squeeze(0),
            "target_im2feat": target_im2feat,
            "cardinal_omega": card_omega,
        }
        if return_presoftmax:
            output["correlation_maps"] = correlation_maps
        return output

    def compute_sparse_local_nre(self, data: Dict):
        """Compute sparse and local NRE maps.
        Args:
            * data: Dictionary containing input data. Expected keys include
                source_image_tensor, target_image_tensor and source_keypoints.
        Returns:
            * sparse_nre: Sparse NRE maps computed around interest regions.
            * num_non_zero: Tensor containing the number of non-zero values per map.
            * target_im2feat: The target image-to-feature downsampling ratio.
            * cardinal_omega: Card(Omega) of the maps.
        """
        assert len(data["prior_keypoints"]) == len(data["source_keypoints"])
        # Compute dense feature maps
        (
            dense_source_features,
            dense_target_features,
            source_keypoints,
        ) = self.compute_dense_feature(data)

        # Compute the target feature-to-image upsampling ratio
        target_im2feat = relative_aspect_ratio(
            data["target_image_tensor"], dense_target_features
        )

        # Interpolate sparse source descriptors
        sparse_source_features = interpolate(dense_source_features, source_keypoints)

        # Compute prior keypoints in feature coordinate space
        prior_keypoints = downsample_keypoints(
            keypoints=data["prior_keypoints"],
            image=data["target_image_tensor"],
            features=dense_target_features,
        )

        # Compute sparse local NRE maps
        nre_maps, num_non_zero = compute_local_nre_maps(
            source_descriptors=sparse_source_features,
            target_features=dense_target_features,
            prior_target_keypoints=prior_keypoints,
            norm_coarse=data["norm_coarse"],
            window_size=data["window_size"],
            num_points=len(source_keypoints),
            minibatch_size=self.hparams.minibatch_size,
        )

        return {
            "sparse_nre": nre_maps,
            "num_non_zero": num_non_zero,
            "target_im2feat": target_im2feat,
            "cardinal_omega": cardinal_omega(dense_target_features),
        }

    def compute_dense_feature(self, data: Dict):
        """Compute dense feature maps."""
        # Compute dense CNN features
        dense_source_features = self.cnn(data["source_image_tensor"])
        dense_target_features = self.cnn(data["target_image_tensor"])

        # Compute source keypoints in feature coordinate space
        source_keypoints = downsample_keypoints(
            keypoints=data["source_keypoints"].squeeze_(0),
            image=data["source_image_tensor"],
            features=dense_source_features,
        )
        return dense_source_features, dense_target_features, source_keypoints

    def forward(
        self,
        data: Dict,
        compute_entropy: bool = False,
        compute_argmax: bool = False,
        sparsify: bool = False,
        prior_keypoints: torch.Tensor = None,
        window_size: int = None,
        target_device: torch.device = "cpu",
    ):
        """Compute NRE loss maps.
        Args:
            * data: Input dictionary containing the source and target image tensors,
                as well as the source 2D keypoint coordinates (in image space).
            * compute_entropy: If True, the correspondence maps entropy are computed.
            * compute_argmax: If True, the 2D argmax locations of the correspondence
                maps are computed (e.g. to run coarse P3P).
            * sparsify: If True, the NRE maps are converted to sparse tensors for a
                lighter memory footprint (e.g. to run fine IRLS).
            * prior_keypoints: If provided, NRE maps are only computed in the vicinity
                of prior keypoint coordinates (e.g. to compute fine local NRE maps).
                NB: Prior keypoints should lie within the target image plane.
            * window_size: The size of the local NRE maps (in feature space).
            * target_device: The target device to move the dense NRE loss maps to.
        """
        if sparsify:
            assert prior_keypoints is not None
            assert window_size is not None
            data["prior_keypoints"] = prior_keypoints
            data["window_size"] = window_size
            output = self.compute_sparse_local_nre(data)
            output["sparse_nre"] = output["sparse_nre"].to(target_device)
            output["num_non_zero"] = output["num_non_zero"].to(target_device)
        else:
            output = self.compute_cmaps(data)

        if compute_entropy:
            output["entropy"] = entropy(log(output["cmaps"]))
        if compute_argmax:
            output["argmax_2D"] = argmax_2D(output["cmaps"])
        if not sparsify:
            output["cmaps"] = output["cmaps"].to(target_device)
            output["nre_maps"] = (
                log(output["cmaps"]).neg().clamp(max=log(output["cardinal_omega"]))
            )
        return output

    @property
    def cnn_layers(self):
        return list(self.cnn)

    @property
    def image_cfg(self):
        return self.hparams.image
