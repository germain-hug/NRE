# Neural Reprojection Error: Merging Feature Learning and Camera Pose Estimation 

This is the official repository for our paper [Neural Reprojection Error: Merging Feature Learning and Camera Pose Estimation ](https://arxiv.org/abs/2103.07153), to appear in CVPR 2021. Code to will be released prior to the conference.

<p align="center">
  <a href="https://arxiv.org/abs/2103.07153"><img src="images/teaser.pdf" width="60%"/></a>
</p>

## Abstract

Absolute camera pose estimation is usually addressed by sequentially solving two distinct subproblems: First a feature matching problem that seeks to establish putative 2D-3D correspondences, and then a Perspective-n-Point problem that minimizes, with respect to the camera pose, the sum of so-called Reprojection Errors (RE). We argue that generating putative 2D-3D correspondences 1) leads to an important loss of information that needs to be compensated as far as possible, within RE, through the choice of a robust loss and the tuning of its hyperparameters and 2) may lead to an RE that conveys erroneous data to the pose estimator. In this paper, we introduce the Neural Reprojection Error (NRE) as a substitute for RE. NRE allows to rethink the camera pose estimation problem by merging it with the feature learning problem, hence leveraging richer information than 2D-3D correspondences and eliminating the need for choosing a robust loss and its hyperparameters. Thus NRE can be used as training loss to learn image descriptors tailored for pose estimation. We also propose a coarse-to-fine optimization method able to very efficiently minimize a sum of NRE terms with respect to the camera pose. We experimentally demonstrate that NRE is a good substitute for RE as it significantly improves both the robustness and the accuracy of the camera pose estimate while being computationally and memory highly efficient. From a broader point of view, we believe this new way of merging deep learning and 3D geometry may be useful in other computer vision applications.

## BibTex

Please consider citing our work:

```
@inproceedings{germain2021NRE,
  author    = {Hugo Germain and
               Vincent Lepetit and
               Guillaume Bourmaud},
  title     = {Neural Reprojection Error: Merging Feature Learning and Camera Pose Estimation},
  booktitle = {CVPR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.07153}
}
```

