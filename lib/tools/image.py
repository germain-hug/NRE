"""Image pre-processing tools.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .camera import scale_intrinsics


def read_image(
    path: str,
    K: torch.Tensor,
    max_image_size: int = None,
):
    """Read and resize an image.
    Adapted from PixLoc (author: Paul-Edouard Sarlin) https://psarlin.com/pixloc/
    Args:
        * path: The absolute path to the image.
        * K: The intrinsics matrix (to be rescaled along with the image).
        * image_cfg: The image config dict.
        * max_image_size: (Optional) The maximum image size.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    if max_image_size is not None:
        scales = (1, 1)
        if max(*image.shape[:2]) > max_image_size:
            image, scales = resize(image, max_image_size, fn=max)
        if scales != (1, 1):
            K = scale_intrinsics(K, scales)
    image = normalize(image)
    return image, K


def normalize(image):
    """Normalize the image tensor and reorder the dimensions."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image / 255.0)
    return image


def resize(image, size, fn=None):
    """Resize an image to a fixed size, or according to max or min edge.
    Adapted from PixLoc (author: Paul-Edouard Sarlin) https://psarlin.com/pixloc/
    """
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    return cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR), scale
