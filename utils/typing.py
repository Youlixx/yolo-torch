"""Coco dataset typing utilities."""

from typing import Literal, TypedDict


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
