"""Module containing wrapping all the Yolo functions into a single model."""

import numpy as np
from torch import Tensor, nn

from yolo.detection_head import YoloDetectionHead
from yolo.ground_truth import YoloAnnotation, generate_ground_truth_tensors
from yolo.loss_fn import loss_yolo
from yolo.post_processor import DetectionResult, decode_boxes


class YoloDetector(nn.Module):
    """Yolo detector model."""

    def __init__(
        self,
        backbone: nn.Module,
        priors: np.ndarray,
        num_classes: int,
        feature_map_depth: int = 1280
    ) -> None:
        """Initialize the model.

        Args:
            backbone (nn.Module): Feature extractor.
            priors (np.ndarray): Model priors.
            num_classes (int): Number of output class.
            feature_map_depth (int): Depth of the feature extractor output.
        """
        super().__init__()

        self.priors = priors

        self.backbone = backbone
        self.detection_head = YoloDetectionHead(feature_map_depth, priors, num_classes)

    def forward(
        self,
        images: Tensor,
        targets: list[list[YoloAnnotation]] | None = None
    ) -> Tensor | list[DetectionResult]:
        """Run the detector.

        This function behavior differ between training and evaluation.
        - when training, the targets (ground truth) must be supplied, and the loss is
          directly returned.
        - when not training, the decoded bounding boxes are returned (the boxes are not
          filtered yet, so more post-processing such as NMS may be required).
        This is done this way since it is how it is done in the torchvision library.

        Args:
            images (Tensor): Input images (must be batched).
            targets (list[list[YoloAnnotation]], optional): Used in training mode, the
                ground truth values.

        Returns:
            Tensor | list[DetectionResult]: Either the loss over the batch in training
                mode, or the detected boxes.
        """
        input_size = images.shape[2:]
        encoded_boxes_batched = self.detection_head(self.backbone(images), input_size)

        if self.training:
            assert targets is not None, "Targets are required when training."

            ground_truth = generate_ground_truth_tensors(
                annotations=targets,
                priors=self.priors,
                input_size=input_size,
                grid_size=encoded_boxes_batched.shape[1:3],
                num_classes=self.detection_head.num_classes
            ).to(images.device)

            return loss_yolo(encoded_boxes_batched, ground_truth)

        return [
            decode_boxes(encoded_boxes, input_size)
            for encoded_boxes in encoded_boxes_batched
        ]
