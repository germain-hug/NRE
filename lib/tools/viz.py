from typing import Dict

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.graph_objs.scatter import Marker
from plotly.subplots import make_subplots

from .camera import inbound_keypoints, project_keypoints
from .metrics import display_pose_error
from .tensor_ops import distances_2D, to_device

DPI = 100
GT_COLOR = "#03A63C"
P3P_COLOR = "#D94711"
COARSE_IRLS_COLOR = "#F28241"
FINE_IRLS_COLOR = "#F2B872"
BG_COLOR = "#F2F2F2"
POINT_SIZE = 4


def display_localization(
    input_data: Dict,
    pose: Dict,
    gt_pose: torch.Tensor,
):
    """Display reprojected keypoints after localization in a HTML file.
    Args:
        * input_data: The input dictionary returned by the dataloader.
        * pose: The dictionary of predicted poses.
        * gt_pose: The ground truth camera pose.
    """
    # Fetch images
    source_image_pil = cv2.imread(str(input_data["source_image"]))[..., ::-1]
    target_image_pil = cv2.imread(str(input_data["target_image"]))[..., ::-1]
    imshow_params = {
        "color_continuous_scale": "gray",
        "binary_compression_level": 9,
        "binary_format": "jpg",
    }

    # Fetch input data
    cpu = torch.device("cpu")
    p3D = to_device(input_data["p3D"], cpu)
    random_idx = np.random.choice(np.arange(len(p3D)), 100)
    p3D = p3D[random_idx]
    K = to_device(input_data["target_K"], cpu)
    dist = to_device(input_data["target_dist"], cpu)

    # Ground-truth, P3P, and IRLS poses
    gt_pose_dict = to_device({"R": gt_pose[:3, :3], "t": gt_pose[:3, 3]}, cpu)
    if "P3P" in pose:
        pose["P3P"] = to_device(pose["P3P"], cpu)
    pose["coarse_irls"] = to_device(pose["coarse_irls"], cpu)
    pose["fine_irls"] = to_device(pose["fine_irls"], cpu)

    # Ground truth keypoint reprojections
    source_gt_p2D = project_keypoints(
        p3D,
        torch.from_numpy(input_data["source_P"][:3, :3]),
        torch.from_numpy(input_data["source_P"][:3, 3]),
        torch.from_numpy(input_data["source_K"]),
        torch.from_numpy(input_data["source_dist"]),
    )[0].numpy()
    target_gt_p2D, positive_z = project_keypoints(
        p3D, gt_pose_dict["R"], gt_pose_dict["t"], K, dist
    )
    positive_z = positive_z.numpy().astype(np.bool)
    target_gt_p2D = target_gt_p2D.numpy()[positive_z]

    # P3P keypoint reprojections
    if "P3P" in pose:
        target_p3p_p2D = project_keypoints(
            p3D, pose["P3P"]["pose"]["R"], pose["P3P"]["pose"]["t"], K, dist
        )[0].numpy()[positive_z]

    # Coarse GNC-IRLS keypoint reprojections
    target_coarse_irls_p2D = project_keypoints(
        p3D, pose["coarse_irls"]["pose"]["R"], pose["coarse_irls"]["pose"]["t"], K, dist
    )[0].numpy()[positive_z]

    # Fine GNC-IRLS keypoint reprojections
    target_fine_irls_p2D = project_keypoints(
        p3D, pose["fine_irls"]["pose"]["R"], pose["fine_irls"]["pose"]["t"], K, dist
    )[0].numpy()[positive_z]

    # Compute 2D errors for histograms
    p3p_errors = distances_2D(target_gt_p2D, target_p3p_p2D)
    coarse_IRLS_errors = distances_2D(target_gt_p2D, target_coarse_irls_p2D)
    fine_IRLS_errors = distances_2D(target_gt_p2D, target_fine_irls_p2D)

    # In-plane reprojection filtering
    image_size = np.array(np.array(target_image_pil).shape[:2][::-1])
    source_image_size = np.array(np.array(source_image_pil).shape[:2][::-1])
    mask_gt = inbound_keypoints(target_gt_p2D, image_size)
    mask_gt_source = inbound_keypoints(source_gt_p2D, source_image_size)
    mask_p3p = inbound_keypoints(target_p3p_p2D, image_size)
    mask_coarse_gnc_irls = inbound_keypoints(target_coarse_irls_p2D, image_size)
    mask_fine_gnc_irls = inbound_keypoints(target_fine_irls_p2D, image_size)

    # Compute pose errors
    error_P3P = display_pose_error(pose["P3P"]["pose"], gt_pose_dict)
    error_coarse_IRLS = display_pose_error(pose["coarse_irls"]["pose"], gt_pose_dict)
    error_fine_IRLS = display_pose_error(pose["fine_irls"]["pose"], gt_pose_dict)

    # Build figure
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=("Source Image", "Target Image", "Reprojection Errors"),
    )
    fig.add_trace(px.imshow(source_image_pil, **imshow_params).data[0], 1, 1)
    fig.add_trace(px.imshow(target_image_pil, **imshow_params).data[0], 1, 2)

    title = ["Pose errors:"]
    title.append("{:<18}".format("P3P") + f": {error_P3P}\n")
    title.extend(
        [
            "{:<18}".format("+ Coarse GNC-IRLS") + f": {error_coarse_IRLS}",
            "{:<18}".format("+ Fine GNC-IRLS") + f": {error_fine_IRLS}",
        ]
    )

    # Source GT
    fig.add_trace(
        go.Scatter(
            x=source_gt_p2D[mask_gt_source, 0],
            y=source_gt_p2D[mask_gt_source, 1],
            marker=Marker(color=GT_COLOR, size=POINT_SIZE),
            mode="markers",
            name="Ground-Truth",
            showlegend=False,
        ),
        1,
        1,
    )

    # GT
    fig.add_trace(
        go.Scatter(
            x=target_gt_p2D[mask_gt, 0],
            y=target_gt_p2D[mask_gt, 1],
            marker=Marker(color=GT_COLOR, size=POINT_SIZE),
            mode="markers",
            name="Ground-Truth",
        ),
        1,
        2,
    )

    # P3P
    fig.add_trace(
        go.Scatter(
            x=target_p3p_p2D[mask_p3p, 0],
            y=target_p3p_p2D[mask_p3p, 1],
            marker=Marker(color=P3P_COLOR, size=POINT_SIZE),
            mode="markers",
            name="P3P",
        ),
        1,
        2,
    )

    # GNC-IRLS Coarse
    fig.add_trace(
        go.Scatter(
            x=target_coarse_irls_p2D[mask_coarse_gnc_irls, 0],
            y=target_coarse_irls_p2D[mask_coarse_gnc_irls, 1],
            marker=Marker(color=COARSE_IRLS_COLOR, size=POINT_SIZE),
            mode="markers",
            name="Coarse GNC-IRLS",
        ),
        1,
        2,
    )

    # GNC-IRLS Fine
    fig.add_trace(
        go.Scatter(
            x=target_fine_irls_p2D[mask_fine_gnc_irls, 0],
            y=target_fine_irls_p2D[mask_fine_gnc_irls, 1],
            marker=Marker(color=FINE_IRLS_COLOR, size=POINT_SIZE),
            mode="markers",
            name="Fine GNC-IRLS",
        ),
        1,
        2,
    )

    # Reprojection errors histograms
    fig.add_trace(
        go.Box(
            x=p3p_errors,
            marker_color=P3P_COLOR,
            name="P3P",
            showlegend=False,
            boxpoints=False,
            boxmean=True,
            notched=True,
        ),
        2,
        1,
    )
    fig.add_trace(
        go.Box(
            x=coarse_IRLS_errors,
            marker_color=COARSE_IRLS_COLOR,
            name="Coarse GNC-IRLS",
            showlegend=False,
            notched=True,
            boxpoints=False,
            boxmean=True,
        ),
        2,
        1,
    )
    fig.add_trace(
        go.Box(
            x=fine_IRLS_errors,
            marker_color=FINE_IRLS_COLOR,
            name="Fine GNC-IRLS",
            showlegend=False,
            notched=True,
            boxpoints=False,
            boxmean=True,
        ),
        2,
        1,
    )

    fig.update_layout(
        yaxis=dict(domain=[0.3, 1.0]),
        yaxis2=dict(domain=[0.3, 1.0]),
        yaxis3=dict(domain=[0.0, 0.2]),
        title=dict(
            text="<br>".join(title),
            x=0.05,
            y=0.15,
            font=dict(family="monospace", size=16),
            xanchor="left",
        ),
        margin=dict(l=50, r=50, t=50, b=150),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.04, font=dict(size=14)),
        bargap=0.2,
        bargroupgap=0.1,
        width=900,
        height=650,
        plot_bgcolor=BG_COLOR,
    )
    fig.update_traces(opacity=0.75, row=2, col=1)
    fig.update_shapes(dict(xref="x3", yref="y3"))
    fig.layout.annotations[2].update(y=0.2)

    border = 0
    fig.update_yaxes(
        range=[image_size[1] + border, -border], visible=False, row=1, col=2
    )
    fig.update_xaxes(
        range=[-border, image_size[0] + border], visible=False, row=1, col=2
    )
    fig.update_yaxes(
        range=[source_image_size[1] + border, -border], visible=False, row=1, col=1
    )
    fig.update_xaxes(
        range=[-border, source_image_size[0] + border], visible=False, row=1, col=1
    )
    return fig
