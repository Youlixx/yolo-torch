"""Utilities to display boxes."""

import json
import os
import random

import cv2
import numpy as np


def display_image_with_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    mode: str = "xyxy",
    display_labels: bool = True,
    scaling_factor: int | None = 2,
    box_color: tuple[int, int, int] = (0, 200, 200),
    text_color: tuple[int, int, int] = (0, 0, 0),
    thickness: float = 2
) -> np.ndarray:
    """Display bounding boxes on a image.

    Args:
        image (np.ndarray): Original image.
        boxes (np.ndarray): Bounding boxes to display.
        labels (np.ndarray): Boxes labels.
        scores (np.ndarray, optional): If specified, the boxes scores.
        mode (str): Format of the bounding boxes, either 'xyxy' or 'xywh'.
        display_labels (bool): If enabled, display the boxes labels and scores.
        scaling_factor (int, optional): Image scaling factor, can improve text size.
        box_color (tuple[int, int, int]): Bounding boxes color.
        text_color (tuple[int, int, int]): Labels color.
        thickness (float): Box and text thickness.

    Returns:
        np.ndarray: Image with the bounding boxes drawn on it.
    """
    image_with_boxes = image.copy()

    if scaling_factor is not None:
        image_width = image_with_boxes.shape[1] * scaling_factor
        image_height = image_with_boxes.shape[0] * scaling_factor

        image_with_boxes = cv2.resize(
            src=image_with_boxes,
            dsize=(image_width, image_height)
        )

    if scores is None:
        scores = [None for _ in range(len(boxes))]

    for box, label, score in zip(boxes, labels, scores):
        if mode == "xyxy":
            x0, y0, x1, y1 = box.tolist()
        elif mode == "xywh":
            x0, y0, w, h = box.tolist()
            x1 = x0 + w
            y1 = y0 + h
        else:
            raise ValueError(f"Unsupported box mode: '{mode}'.")

        if scaling_factor is not None:
            x0 *= scaling_factor
            y0 *= scaling_factor
            x1 *= scaling_factor
            y1 *= scaling_factor

        print(x0, y0, x1, y1)

        image_with_boxes = cv2.rectangle(
            img=image_with_boxes,
            pt1=(x0, y0),
            pt2=(x1, y1),
            color=box_color,
            thickness=thickness
        )

        if display_labels:
            if score is not None:
                text = f"{label} ({score.tolist():.2f})"
            else:
                text = str(label)

            (text_width, text_height), baseline = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                thickness=thickness
            )

            image_with_boxes = cv2.rectangle(
                img=image_with_boxes,
                pt1=(x0 - thickness // 2, y0),
                pt2=(x0 + text_width, y0 - baseline - text_height),
                color=box_color,
                thickness=-1
            )

            image_with_boxes = cv2.putText(
                img=image_with_boxes,
                text=text,
                org=(x0, y0 - baseline // 2),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=text_color,
                thickness=thickness
            )

    return image_with_boxes


def display_random_dataset_samples(
    path_dataset: str,
    sample_count: int = 10,
    display_labels: bool = True,
    scaling_factor: int | None = 2,
    box_color: tuple[int, int, int] = (0, 200, 200),
    text_color: tuple[int, int, int] = (0, 0, 0),
    thickness: float = 2
) -> list[np.ndarray]:
    """Sample a dataset and display random images with their annotations.

    Args:
        path_dataset (str): Path to the COCO dataset.
        sample_count (int): Number of sample.
        display_labels (bool): If enabled, display the boxes labels and scores.
        scaling_factor (int, optional): Image scaling factor, can improve text size.
        box_color (tuple[int, int, int]): Bounding boxes color.
        text_color (tuple[int, int, int]): Labels color.
        thickness (float): Box and text thickness.

    Returns:
        list[np.ndarray]: List of images with their annotations.
    """
    path_annotations = os.path.join(path_dataset, "annotations.json")
    path_images = os.path.join(path_dataset, "images")

    with open(path_annotations, "r") as file:
        annotations = json.load(file)

    random.shuffle(annotations["images"])

    images_path_mapping: dict[int, str] = {
        image["id"]: image["file_name"]
        for image in annotations["images"][:sample_count]
    }

    per_sample_annotations: dict[int, list] = {}

    for annotation in annotations["annotations"]:
        if annotation["image_id"] not in images_path_mapping.keys():
            continue

        if annotation["image_id"] not in per_sample_annotations:
            per_sample_annotations[annotation["image_id"]] = []
        per_sample_annotations[annotation["image_id"]].append(annotation)

    images: list[np.ndarray] = []

    for image_index, image_name in images_path_mapping.items():
        path_image = os.path.join(path_images, image_name)
        image = cv2.imread(path_image, cv2.IMREAD_COLOR)

        if image is None:
            continue

        boxes: list[tuple[int, int, int, int]] = []
        labels: list[int] = []

        for annotation in per_sample_annotations[image_index]:
            boxes.append(annotation["bbox"])
            labels.append(annotation["category_id"])

        images.append(display_image_with_boxes(
            image=image,
            boxes=np.array(boxes),
            labels=np.array(labels),
            scores=None,
            mode="xywh",
            display_labels=display_labels,
            scaling_factor=scaling_factor,
            box_color=box_color,
            text_color=text_color,
            thickness=thickness,
        ))

    return images
