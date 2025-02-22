"""Yolo loss implementation."""

import torch
from torch import Tensor


def loss_yolo(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Compute the Yolo loss.

    Both tensors are expected to have the shape (BS x HG x WG x B x (5+C)):
    - y[..., 0] = x: the x position of the center of the box relative to the grid.
    - y[..., 1] = y: the y position of the center of the box relative to the grid.
    - y[..., 2] = w: the width of the box relative to the grid.
    - y[..., 3] = h: the height of the box relative to the grid.
    - y[..., 4] = P: the "objectness" score of the box (i.e. the probability that the
        box contains an object).
    - y[..., 5:] = C: the conditional probability vector of the box belonging to each
        class.

    Args:
        ground_truth (Tensor): The ground truth tensor.
        predicted (Tensor): The predicted tensor.

    Returns:
        Tensor: The loss tensor.
    """
    # First, we split the predicted tensors to retrieve the coordinates and the
    # conditional probabilities.
    predicted_xy = predicted[..., :2]  # (BS x HG x WG x B x 2)
    predicted_wh = predicted[..., 2:4]  # (BS x HG x WG x B x 2)
    predicted_objectness = predicted[..., 4]  # (BS x HG x WG x B)
    predicted_probabilities = predicted[..., 5:]  # (BS x HG x WG x B x C)

    # Same for the ground truth tensors.
    true_xy = ground_truth[..., :2]  # (BS x HG x WG x B x 2)
    true_wh = ground_truth[..., 2:4]  # (BS x HG x WG x B x 2)
    true_objectness = ground_truth[..., 4]  # (BS x HG x WG x B)
    true_probabilities = ground_truth[..., 5:]  # (BS x HG x WG x B x C)

    # Position error: a simple square error between the predicted and ground truth
    # positions.
    diff_xy = torch.square(predicted_xy - true_xy)
    diff_xy = torch.sum(diff_xy, dim=-1)
    diff_xy = diff_xy * true_objectness

    # Dimension error: a square error between the square roots of the predicted and
    # ground truth dimensions.
    diff_wh = torch.square(torch.sqrt(predicted_wh) - torch.sqrt(true_wh))
    diff_wh = torch.sum(diff_wh, dim=-1)
    diff_wh = diff_wh * true_objectness

    # The following operations consists in determining the IOU between the predicted and
    # ground truth boxes. First, we compute the position of top-left and bottom-right
    # corners of the predicted boxes.
    predicted_x0_y0 = predicted_xy - predicted_wh / 2
    predicted_x1_y1 = predicted_xy + predicted_wh / 2

    # Same goes for the ground truth boxes.
    true_x0_y0 = true_xy - true_wh / 2
    true_x1_y1 = true_xy + true_wh / 2

    # Then we compute the coordinates of the intersection between the predicted and
    # ground truth boxes.
    intersection_x0_y0 = torch.maximum(predicted_x0_y0, true_x0_y0)
    intersection_x1_y1 = torch.minimum(predicted_x1_y1, true_x1_y1)

    # Using the coordinates, we can deduce the dimensions of the intersection.
    # If the intersection is empty, at least one of the dimension will be negative. By
    # setting it to zero, the intersection area will be zero.
    intersection_wh = intersection_x1_y1 - intersection_x0_y0
    intersection_wh = torch.maximum(intersection_wh, 0.)

    # Then, we compute the intersection area between the predicted and ground truth
    # boxes.
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

    # To compute the IOU we also need to compute the union area.
    predicted_area = predicted_wh[..., 0] * predicted_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    union_area = predicted_area + true_area - intersection_area

    # Finally, we compute the IOU between the predicted and ground truth boxes.
    iou_scores = intersection_area / union_area

    # Objectness error: a square error between the predicted objectness and the IOU
    # between the boxes. This means that the objectness will tend to quantify the
    # quality of the predicted box.
    diff_objectness = torch.square(predicted_objectness - iou_scores)
    diff_objectness = diff_objectness * true_objectness

    # No objectness error: if the predicted box does not contain an object, the
    # objectness should tend to zero.
    diff_no_object = torch.square(predicted_objectness)
    diff_no_object = diff_no_object * (1 - true_objectness)

    # Classification error: a square error between the predicted and ground truth
    # conditional probabilities. Note that any kind of classification loss can be used
    # such as the binary cross-entropy.
    diff_classification = torch.square(predicted_probabilities - true_probabilities)
    diff_classification = torch.sum(diff_classification, dim=-1)
    diff_classification = diff_classification * true_objectness

    # The total loss is the weighted sum of all the previously computed errors.
    diff = (
        5 * diff_xy +
        5 * diff_wh +
        diff_objectness +
        diff_no_object +
        diff_classification
    )

    diff = torch.sum(diff, dim=(1, 2, 3))

    return diff
