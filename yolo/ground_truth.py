"""Module containing function to generate ground truth tensors."""

import numpy as np
import torch
from torch import Tensor

from yolo.dataset import YoloAnnotation


def _find_best_prior(boxes: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Find the best prior for each box.

    The best prior is the one with the smallest IOU distance with the bounding box.

    Args:
        boxes (np.ndarray): The bounding boxes in the YOLOv2 format.
        priors (np.ndarray): The priors of the YOLOv2 model.

    Returns:
        np.ndarray: A vector of indices of the best prior for each bounding box.
    """
    boxes_wh = boxes[..., 2:]
    boxes_wh = np.expand_dims(boxes_wh, axis=1)

    # As we use the IOU distance, we need to compute the intersection and union areas
    # between the boxes and the priors. The IOU distance is computed as if the two boxes
    # are centered between each other.
    intersection_wh = np.minimum(boxes_wh, priors)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    prior_area = priors[..., 0] * priors[..., 1]
    union_area = boxes_area + prior_area - intersection_area
    iou_scores = intersection_area / union_area

    # The selected prior for each bounding box is the one with the highest IOU score.
    best_prior = np.argmax(iou_scores, axis=-1)

    return best_prior


def _generate_single_ground_truth_tensor(
    annotations: list[YoloAnnotation],
    priors: np.ndarray,
    input_size: tuple[int, int],
    grid_size: tuple[int, int],
    num_classes: int
) -> Tensor:
    """Generate a ground truth output tensor.

    The generated tensor have the shape (HG x WG x B x (5+C)):
    - y[..., 0] = x: the x position of the center of the box relative to the grid.
    - y[..., 1] = y: the y position of the center of the box relative to the grid.
    - y[..., 2] = w: the width of the box relative to the grid.
    - y[..., 3] = h: the height of the box relative to the grid.
    - y[..., 4] = P: the "objectness" score of the box (i.e. the probability that the
        box contains an object).
    - y[..., 5:] = C: the conditional probability vector of the box belonging to each
        class.

    Args:
        annotations (list[YoloAnnotations]): List of ground truth annotations.
        priors (np.ndarray): Model priors.
        input_size (tuple[int, int]): Size of the input image (hw format).
        grid_size (tuple[int, int]): Size of the output grid (hw format).
        num_classes (int): Number of class.

    Returns:
        Tensor: Ground truth tensor.
    """
    # Note: the model outputs boxes that use xywh centered format.
    boxes = np.array([
        (
            annotation["box"][0] + annotation["box"][2] / 2,
            annotation["box"][1] + annotation["box"][3] / 2 ,
            annotation["box"][2],
            annotation["box"][3]
        )
        for annotation in annotations
    ], dtype=np.float32)

    # We initialize the ground truth tensor with zeros. Note that this tensor is
    # essentially a sparse tensor, where only the cells corresponding to the bounding
    # boxes are non-zero. This means that by default, the true "objectness" is set to
    # zero for every cell that doesn't contain a box and therefor, only the "no
    # objectness" loss will be applied to these cells. The other cells should have a
    # true "objectness" of 1 and in this case, all the other losses will be applied.
    # This tensor can be seen as a 3D grid of size HG x WG x B, where each cell
    # corresponds to a box (shape: (HG x WG x B x (5+C))).
    y_true = np.zeros((*grid_size, len(priors), 5 + num_classes))

    # If there are no bounding boxes, on the input image, we return an empty ground
    # truth tensor.
    if len(boxes) == 0:
        return torch.from_numpy(y_true)

    best_priors = _find_best_prior(boxes, priors)

    # First, we convert the bounding boxes from the standard format to the YOLOv2 format
    # and the class labels to one-hot vectors.
    scale_h, scale_w = input_size[0] / grid_size[0], input_size[1] / grid_size[1]
    boxes_scaled = boxes / [scale_w, scale_h, scale_w, scale_h]

    labels = [annotation["label"] for annotation in annotations]
    labels_one_hot = np.zeros((len(labels), num_classes))
    labels_one_hot[np.arange(len(labels)), labels] = 1

    # The bounding boxes should be placed at their corresponding grid cell by rounding
    # down the coordinates.
    grid_coordinates = np.floor(boxes_scaled[..., :2]).astype(np.uint32)
    grid_x = grid_coordinates[..., 0]
    grid_y = grid_coordinates[..., 1]

    # We construct the expected bounding box output using the previously converted
    # bounding box, the one-hot vector, and we set the true "objectness" to 1 as the
    # cell actually contains a box (shape: (NB x (5+C))).
    expected_values = np.concatenate([
        boxes_scaled,
        np.ones((len(boxes), 1)),
        labels_one_hot
    ], axis=-1)

    y_true[grid_y, grid_x, best_priors] = expected_values

    return torch.from_numpy(y_true)


def generate_ground_truth_tensors(
    annotations: list[list[YoloAnnotation]],
    priors: np.ndarray,
    input_size: tuple[int, int],
    grid_size: tuple[int, int],
    num_classes: int
) -> Tensor:
    """Generate a batch of ground truth output tensors.

    The generated tensor have the shape (BS x HG x WG x B x (5+C)):
    - y[..., 0] = x: the x position of the center of the box relative to the grid.
    - y[..., 1] = y: the y position of the center of the box relative to the grid.
    - y[..., 2] = w: the width of the box relative to the grid.
    - y[..., 3] = h: the height of the box relative to the grid.
    - y[..., 4] = P: the "objectness" score of the box (i.e. the probability that the
        box contains an object).
    - y[..., 5:] = C: the conditional probability vector of the box belonging to each
        class.

    Args:
        annotations (list[list[YoloAnnotations]]): Batch of ground truth annotations.
        priors (np.ndarray): Model priors.
        input_size (tuple[int, int]): Size of the input image (hw format).
        grid_size (tuple[int, int]): Size of the output grid (hw format).
        num_classes (int): Number of class.

    Returns:
        Tensor: Ground truth tensor.
    """
    batch_ground_truth = []

    for image_annotations in annotations:
        batch_ground_truth.append(_generate_single_ground_truth_tensor(
            annotations=image_annotations,
            priors=priors,
            input_size=input_size,
            grid_size=grid_size,
            num_classes=num_classes
        ))

    return torch.stack(batch_ground_truth, dim=0)
