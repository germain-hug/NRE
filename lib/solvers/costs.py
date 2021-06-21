"""Compute reprojection costs.
"""
import torch


@torch.jit.script
def interpolate_costs(cost_maps: torch.Tensor, p2D: torch.Tensor):
    """Interpolate cost values in dense cost maps.
    Args:
        * cost_maps: The dense [N x H x W] cost maps.
        * p2D: The [N x 2] keypoint coordinates.
    Returns:
        * costs: The [N] tensor of bilinearly interpolated costs.
    """
    assert len(cost_maps) == len(p2D), "Dimensions do not match"
    height, width = cost_maps.shape[-2:]
    scale = torch.tensor([width - 1, height - 1], dtype=p2D.dtype, device=p2D.device)
    p2D = ((p2D / scale) * 2 - 1).clamp(min=-2, max=2)
    costs = torch.nn.functional.grid_sample(
        cost_maps[:, None],
        p2D[:, None, None],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return costs.view(len(p2D))


@torch.jit.script
def reprojection_cost(
    cost_maps: torch.Tensor,
    p2D: torch.Tensor,
    padding: float,
    mask: torch.Tensor,
):
    """Compute the reprojection cost.
    Args:
        cost_maps: The [N x H x W] cost maps.
        p2D: The keypoint coordinates.
        padding: The values used for out-of-bound padding.
        mask: The [N x 1] mask of valid projections.
    Returns:
        costs: The sum of reprojection costs.
    """
    assert len(cost_maps) == len(p2D)
    costs = torch.full((len(p2D),), padding, dtype=p2D.dtype, device=p2D.device)
    costs[mask] = interpolate_costs(cost_maps[mask], p2D[mask])
    return costs.sum()
