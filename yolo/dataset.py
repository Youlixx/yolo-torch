"""Coco dataset typing utilities."""

import json
import os
from typing import Literal, TypedDict

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
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

        k_means = KMeans(n_clusters=cluster_count, random_state=seed, max_iter=max_iter)
        k_means.fit(boxes)

        return k_means.cluster_centers_

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

        image = cv2.imread(path_image)
        image = torch.from_numpy(image).to(torch.float32)
        image = torch.swapaxes(image, -1, 0)
        # TODO pre-processing....

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
