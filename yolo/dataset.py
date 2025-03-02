"""Coco dataset typing utilities."""

import json
import os
from typing import Literal, TypedDict

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RLEMask(TypedDict):
    """Coco RLE encoded mask dict representation."""

    size: tuple[int, int]
    """Size of the binary mask."""

    counts: str
    """Encoded RLE counts."""


class CocoAnnotation(TypedDict):
    """Coco annotation dict representation."""

    id: int
    """Annotation index."""

    image_id: int
    """Index of the image on which the object is present, see `CocoImage`."""

    segmentation: RLEMask
    """RLE encoded segmentation mask."""

    category_id: int
    """Index of the category of the annotation, see `CocoCategory`."""

    bbox: tuple[int, int, int, int]
    """Object bounding box, in xywh format."""

    area: int
    """Object area in pixel squared."""

    iscrowd: Literal[0]
    """Internal flag used by Coco, should always be left to 0."""


class CocoImage(TypedDict):
    """Coco image dict representation."""

    id: int
    """Image index, referenced by the annotations."""

    file_name: str
    """Name of the image file."""

    width: int
    """Image width."""

    height: int
    """Image height."""


class CocoCategory(TypedDict):
    """Coco category dict representation."""

    id: int
    """Category index, referenced by the annotations."""

    name: str
    """Category name."""


class CocoDataset(TypedDict):
    """Coco dataset dict representation."""

    categories: list[CocoCategory]
    """Dataset categories."""

    images: list[CocoImage]
    """Dataset images."""

    annotations: list[CocoAnnotation]
    """Dataset annotations."""


class YoloAnnotation(TypedDict):
    """Yolo annotation internal dict representation."""

    label: int
    """Object class."""

    box: tuple[int, int, int, int]
    """Object bounding box, in the xywh format."""


class YoloDataset(Dataset):
    """Yolo dataset loader."""

    def __init__(self, path_annotations: str, path_images: str) -> None:
        """Initialize the dataset.

        Args:
            path_annotations (str): Path to the annotation file, using the Coco format.
            path_images (str): Path to the image folder.
        """
        with open(path_annotations, "r") as file:
            self.coco_dataset: CocoDataset = json.load(file)

        image_index_mapping = {
            image["id"]: os.path.join(path_images, image["file_name"])
            for image in self.coco_dataset["images"]
        }

        image_annotations: dict[int, list[CocoAnnotation]] = {}

        for annotation in self.coco_dataset["annotations"]:
            if annotation["image_id"] not in image_annotations:
                image_annotations[annotation["image_id"]] = []
            image_annotations[annotation["image_id"]].append(annotation)

        self.samples: list[tuple[str, list[YoloAnnotation]]] = []

        for image_index, path_image in image_index_mapping.items():
            if image_index in image_annotations:
                annotations: list[YoloAnnotation] = [
                    {
                        "box": annotation["bbox"],
                        "label": annotation["category_id"]
                    } for annotation in image_annotations[image_index]
                ]
            else:
                annotations = []

            self.samples.append((path_image, annotations))

    def get_priors(
        self,
        cluster_count: int,
        seed: int = 1337,
        max_iter: int = 300
    ) -> np.ndarray:
        """Compute the dataset priors.

        The priors are computed by clustering the dimensions of the bounding boxes using
        naive K-means with the IOU distance. The algorithm is initialized randomly,
        therefore the results may differ from run to run. You can set the seed of
        np.random to get reproducible results.

        Args:
            cluster_count (int): The number of clusters / priors.
            seed (int): Random seed.
            max_iter (int): The maximum number of iterations for the K-means algorithm.

        Returns:
            np.ndarray: The cluster representative.
        """
        # Note: the bounding boxes are stored in the xywh format.
        boxes = np.array([
            annotation["bbox"][2:] for annotation in self.coco_dataset["annotations"]
        ])

        np.random.seed(seed)

        # Randomly initialize the clusters. We must ensure that each centroid is unique.
        unique_boxes = np.unique(boxes, axis=0)
        centroids = unique_boxes[
            np.random.choice(len(unique_boxes),
            size=cluster_count,
            replace=False)
        ]

        boxes_wh = np.expand_dims(boxes, axis=1)

        for _ in range(max_iter):
            # As we use the IOU distance, we need to compute the intersection and union
            # areas between the boxes and the centroids. The IOU distance is computed as
            # if the two boxes are centered between each other.
            intersection_wh = np.minimum(boxes_wh, centroids)
            intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

            boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]
            center_area = centroids[..., 0] * centroids[..., 1]
            union_area = boxes_area + center_area - intersection_area

            # The IOU distance is computed as d(x, y) = 1 - IOU(x, y).
            # Note: union_area is always non zeros since the boxes are centered.
            iou_distances = 1 - intersection_area / union_area

            # We update the centers by taking the mean dimension of each cluster
            closest_center = np.argmin(iou_distances, axis=-1)
            new_centroids = np.array([
                np.mean(boxes[closest_center == k], axis=0)
                for k in range(cluster_count)
            ])

            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return centroids

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, list[YoloAnnotation]]:
        """Get a dataset sample.

        Samples are returned as a tuple (image, annotations), where the annotations is
        a list of dict {"box", "label"}. These samples cannot be directly used in a
        torch Dataloader and require a custom collate function.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, list[YoloAnnotation]]: A dataset sample.
        """
        path_image, annotations = self.samples[index]

        image = cv2.imread(path_image, cv2.IMREAD_COLOR_RGB)
        image = torch.from_numpy(image).to(torch.float32)
        image = torch.moveaxis(image, -1, 0) / 255

        return image, annotations


def collate_fn(
    samples: list[tuple[Tensor, list[YoloAnnotation]]]
) -> tuple[Tensor, list[list[YoloAnnotation]]]:
    """Yolo dataset collate function.

    Args:
        samples (list[tuple[Tensor, list[YoloAnnotation]]]): List of batch samples.

    Returns:
        tuple[Tensor, list[list[YoloAnnotation]]]: Collated samples.
    """
    batched_images = []
    batched_annotations = []

    for image, annotations in samples:
        batched_images.append(image)
        batched_annotations.append(annotations)

    return torch.stack(batched_images, axis=0), batched_annotations
