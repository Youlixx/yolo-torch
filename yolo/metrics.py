"""Module containing a Coco evaluation helper."""

from typing import TypedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo.dataset import CocoAnnotation, CocoDataset, YoloAnnotation
from yolo.post_processor import DetectionResult


class CocoDetection(TypedDict):
    """Coco detection internal representation."""

    image_id: int
    """Index of the image on which the object is present, see `CocoImage`."""

    category_id: int
    """Index of the category of the annotation, see `CocoCategory`."""

    bbox: tuple[int, int, int, int]
    """Object bounding box, in xywh format."""

    score: float
    """Object detection score, between 0 and 1."""


class CocoEvaluator:
    """Simple coco wrapper."""

    def __init__(self) -> None:
        """Initialize the wrapper."""
        self._available_categories: set[int] = set()
        self._detections: list[CocoDetection] = []
        self._ground_truth: list[CocoAnnotation] = []

        self._index_image = 0
        self._index_annotation = 0

    def add_detections(
        self,
        detections: DetectionResult,
        ground_truth: list[YoloAnnotation]
    ) -> None:
        """Add detections to the running evaluation.

        Args:
            detections (DetectionResult): Decoded boxes. For more accurate results,
                these boxes should not be filtered by score yet.
            ground_truth (list[YoloAnnotation]): Ground truth bounding boxes.
        """
        for box, label, score in zip(
            detections["boxes"],
            detections["labels"],
            detections["scores"]
        ):
            self._detections.append({
                "image_id": self._index_image,
                "category_id": label.tolist(),
                "bbox": box.tolist(),
                "score": score.tolist()
            })

            self._available_categories.add(label.tolist())

        for annotation in ground_truth:
            self._ground_truth.append({
                "id": self._index_annotation,
                "image_id": self._index_image,
                "category_id": annotation["label"],
                "bbox": annotation["box"],
                "area": 0,
                "iscrowd": 0
            })

            self._index_annotation += 1
            self._available_categories.add(annotation["label"])

        self._index_image += 1


    def compute_metrics(self) -> float:
        """Compute the Coco metrics.

        Returns:
            float: Model mAP.
        """
        coco_dataset: CocoDataset = {
            "categories": [
                {"id": index, "name": ""} for index in self._available_categories
            ],
            "images": [
                {"id": index, "name": ""} for index in range(self._index_image)
            ],
            "annotations": self._ground_truth
        }

        coco_gt = COCO()
        coco_gt.dataset = coco_dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self._detections)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]
