"""Module containing Yolo post-processing functions."""

from typing import TypedDict

import numpy as np
from torch import Tensor


class DetectionResult(TypedDict):
    """Simple dict wrapping detection result."""

    boxes: np.ndarray
    """Object bounding boxes using the xywh format."""

    labels: np.ndarray
    """Object labels."""

    scores: np.ndarray
    """Object scores."""


def decode_boxes(
    output_tensor: Tensor,
    input_size: tuple[int, int],
    min_score: float = 0.05
) -> DetectionResult:
    """Decode Yolo outputs into boxes.

    Args:
        output_tensor (Tensor): Model output tensor (unbatched).
        input_size (int): Input image size.
        min_score (float): Box minimum score.

    Returns:
        DetectionResult: A dict containing the bounding boxes with their scores and
            labels.
    """
    output_tensor = output_tensor.detach().cpu().numpy()

    grid_size = output_tensor.shape[:2]
    scale_h, scale_w = input_size[0] / grid_size[0], input_size[1] / grid_size[1]

    # First, we split the predicted tensors to retrieve the coordinates and the
    # conditional probabilities.
    predicted_xy = output_tensor[..., :2]
    predicted_wh = output_tensor[..., 2:4]
    predicted_objectness = output_tensor[..., 4]
    predicted_probabilities = output_tensor[..., 5:]

    # We can retrieve the predicted label with its probability by taking the index of
    # the maximum probability.
    predicted_labels = np.argmax(predicted_probabilities, axis=-1)
    predicted_labels_probability = np.max(predicted_probabilities, axis=-1)

    # Then, we convert the coordinates to the standard format.
    predicted_x0_y0 = predicted_xy - predicted_wh / 2
    predicted_x1_y1 = predicted_xy + predicted_wh / 2

    # As the coordinates are expressed relative to the output grid, we have to scale
    # them back to the input image.
    boxes = np.concatenate([predicted_x0_y0, predicted_x1_y1], axis=-1)
    boxes = boxes * np.tile([scale_w, scale_h], 2)

    # Clip boxes to the input size, and remove invalid boxes.
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], a_min=0, a_max=input_size[1])
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], a_min=0, a_max=input_size[0])

    # After clipping some boxes may be empty, so we remove them.
    boxes_wh = boxes[..., 2:] - boxes[..., :2]
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    valid_boxes = boxes_area > 1

    # We also discard the bounding boxes that are below the objectness threshold.
    confidence = predicted_objectness * predicted_labels_probability
    valid_boxes *= confidence > min_score

    boxes = boxes[valid_boxes]
    labels = predicted_labels[valid_boxes]
    scores = confidence[valid_boxes]

    # Convert the boxes back to xywh format.
    boxes = np.concatenate([boxes[..., :2], boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return {
        "boxes": boxes.astype(int),
        "labels": labels.astype(int),
        "scores": scores.astype(np.float32)
    }


def non_maximum_suppression(
    detections: DetectionResult,
    nms_threshold: float = 0.7
) -> DetectionResult:
    """Filter out overlapping boxes using NMS.

    Args:
        detections (DetectionResult): Model output boxes.
        nms_threshold (float): NMS threshold.

    Returns:
        DetectionResult: Filtered boxes, scores and labels.
    """
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    kept_indices = []

    predicted_wh = boxes[..., 2:]
    predicted_x0_y0 = boxes[..., :2] - predicted_wh / 2
    predicted_x1_y1 = boxes[..., :2] + predicted_wh / 2

    predicted_area = predicted_wh[..., 0] * predicted_wh[..., 1] + 1e-6

    remaining = np.argsort(scores)

    while len(remaining) > 0:
        index = remaining[-1]
        kept_indices.append(index)

        remaining = remaining[:-1]
        indices_class = remaining[labels[index] == labels[remaining]]

        intersection_x0_y0 = np.maximum(
            predicted_x0_y0[index],
            predicted_x0_y0[indices_class]
        )

        intersection_x1_y1 = np.minimum(
            predicted_x1_y1[index],
            predicted_x1_y1[indices_class]
        )

        intersection_wh = intersection_x1_y1 - intersection_x0_y0
        intersection_wh = np.maximum(intersection_wh, 0.)

        intersection_areas = intersection_wh[:, 0] * intersection_wh[:, 1]
        union_areas = predicted_area[index] \
            + predicted_area[indices_class] - intersection_areas

        iou_scores = intersection_areas / union_areas

        indices_to_remove = indices_class[iou_scores > nms_threshold]
        remaining = remaining[~np.isin(remaining, indices_to_remove)]

    return {
        "boxes": boxes[kept_indices],
        "scores": scores[kept_indices],
        "labels": labels[kept_indices]
    }
