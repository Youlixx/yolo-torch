"""Utilities to generate train / test datasets."""

import json
import os
import random

import cv2
import numpy as np
from pycocotools.mask import encode
from torch import Tensor
from torchvision.datasets.mnist import MNIST
from torchvision.transforms.functional import resize, rotate

random.seed(1337)

MAX_RETRIES = 10


def generate_sample(
    source_dataset: MNIST,
    object_indices: list[int],
    min_image_size: int = 100,
    max_image_size: int = 600,
    min_objects_per_image: int = 0,
    max_objects_per_image: int = 20,
    noise_strength: int = 20,
    min_object_scale_factor: float = 0.5,
    max_object_scale_factor: float = 2.0,
    max_object_angle: float = 30
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], dict, int, int]]]:
    """Generate a dataset sample.

    Args:
        source_dataset (MNIST): Object source dataset.
        object_indices (list[int]): Sampling indices.
        min_image_size (int): Image min size.
        max_image_size (int): Image max size.
        min_objects_per_image (int): Minimum number of object per image.
        max_objects_per_image (int): Maximum number of object per image.
        noise_strength (int): Background noise strength.
        min_object_scale_factor (float): Minimum object scaling factor.
        max_object_scale_factor (float): Maximum object scaling factor.
        max_object_angle (float): If different from 0, object will be randomly rotated.

    Returns:
        np.ndarray: Generated image.
        list[tuple[tuple[int, int, int, int], dict, int, int]]: Generated
            annotations data, with the bounding box first, the RLE encoded COCO mask,
            the object area and its label last.
    """
    assert min_image_size <= max_image_size
    assert min_objects_per_image <= max_objects_per_image
    assert min_object_scale_factor <= max_object_scale_factor

    image_width = random.randint(min_image_size, max_image_size)
    image_height = random.randint(min_image_size, max_image_size)
    object_count = random.randint(min_objects_per_image, max_objects_per_image)

    object_map = np.zeros((image_height, image_width), dtype=bool)
    background = np.abs(np.random.randn(image_height, image_width) * noise_strength)
    background = background.astype(np.uint8)

    annotations: list[tuple] = []

    for _ in range(object_count):
        if len(object_indices) == 0:
            raise IndexError(
                "Object indices exhausted, all the samples from the source dataset has "
                "been used."
            )

        object_index = object_indices.pop()
        object_image = source_dataset.data[object_index][None, ...]
        object_label = source_dataset.targets[object_index]

        if object_image is None or object_label is None:
            continue

        for _ in range(MAX_RETRIES):
            object_transformed: Tensor = object_image

            if min_object_scale_factor != 1 and max_object_scale_factor != 1:
                scale = min_object_scale_factor + random.random() * (
                    max_object_scale_factor - min_object_scale_factor
                )

                new_width = int(object_image.shape[2] * scale)
                new_height = int(object_image.shape[1] * scale)

                object_transformed = resize(object_transformed, (new_height, new_width))

            if max_object_angle > 0:
                angle = (random.random() * 2 - 1) * max_object_angle
                object_transformed = rotate(object_image, angle, expand=True)

            object_transformed = object_transformed.cpu().numpy()[0]
            object_mask = object_transformed != 0

            h, w = object_transformed.shape
            x = random.randint(0, image_width - w)
            y = random.randint(0, image_height - h)

            if np.sum(object_map[y:y+h, x:x+w] * object_mask) > 0:
                continue

            object_map[y:y+h,x:x+w] = object_mask
            background[y:y+h,x:x+w][object_mask] = 0
            background[y:y+h,x:x+w] += object_transformed

            coco_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            coco_mask[y:y+h,x:x+w] += object_mask

            position_y, position_x = np.where(coco_mask != 0)
            box_x0 = np.min(position_x).tolist()
            box_y0 = np.min(position_y).tolist()
            box_x1 = np.max(position_x).tolist() + 1
            box_y1 = np.max(position_y).tolist() + 1

            # Note: JSON cannot serialize bytes, so we have to convert counts to str.
            coco_encoded_mask = encode(np.asfortranarray(coco_mask, dtype=np.uint8))
            coco_encoded_mask["counts"] = coco_encoded_mask["counts"].decode("ascii")

            annotations.append((
                (box_x0, box_y0, box_x1 - box_x0, box_y1 - box_y0),
                coco_encoded_mask,
                np.sum(object_mask).tolist(),
                object_label.cpu().numpy().tolist()
            ))

            break

    return background, annotations


def generate_dataset(
    source_dataset: MNIST,
    path_dataset: str,
    sample_count: int,
    min_image_size: int = 100,
    max_image_size: int = 600,
    min_objects_per_image: int = 0,
    max_objects_per_image: int = 20,
    noise_strength: int = 32,
    min_object_scale_factor: float = 1,
    max_object_scale_factor: float = 1,
    max_object_angle: float = 0
) -> None:
    """Generate a dataset.

    Args:
        source_dataset (MNIST): Object source dataset.
        path_dataset (str): Path to the generated dataset.
        sample_count (int): Number of image in the dataset.
        min_image_size (int): Image min size.
        max_image_size (int): Image max size.
        min_objects_per_image (int): Minimum number of object per image.
        max_objects_per_image (int): Maximum number of object per image.
        noise_strength (int): Background noise strength.
        min_object_scale_factor (float): Minimum object scaling factor.
        max_object_scale_factor (float): Maximum object scaling factor.
        max_object_angle (float): If different from 0, object will be randomly rotated.
    """
    indices = list(range(len(source_dataset.data)))

    path_images = os.path.join(path_dataset, "images")
    path_annotations = os.path.join(path_dataset, "annotations.json")

    if os.path.exists(path_dataset):
        raise FileExistsError(f"{path_dataset} already exists.")

    os.mkdir(path_dataset)
    os.mkdir(path_images)

    coco_annotations = {
        "categories": [{"id": index, "name": str(index)} for index in range(10)],
        "images": [],
        "annotations": []
    }

    annotation_index = 0

    for image_index in range(sample_count):
        image, annotations = generate_sample(
            source_dataset=source_dataset,
            object_indices=indices,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
            min_objects_per_image=min_objects_per_image,
            max_objects_per_image=max_objects_per_image,
            noise_strength=noise_strength,
            min_object_scale_factor=min_object_scale_factor,
            max_object_scale_factor=max_object_scale_factor,
            max_object_angle=max_object_angle,
        )

        file_name = f"{str(image_index).zfill(5)}.png"
        cv2.imwrite(os.path.join(path_images, file_name), image)

        coco_annotations["images"].append({
            "id": image_index,
            "file_name": file_name,
            "width": image.shape[1],
            "height": image.shape[0],
        })

        for (box, mask, area, label) in annotations:
            coco_annotations["annotations"].append({
                "id": annotation_index,
                "image_id": image_index,
                "segmentation": mask,
                "category_id": label,
                "bbox": box,
                "area": area,
                "iscrowd": 0,
            })

            annotation_index += 1

    print(coco_annotations)
    with open(path_annotations, "w") as file:
        json.dump(coco_annotations, file)


