"""Module containing the Yolo detection head implementation."""

import numpy as np
import torch
from torch import Tensor, nn


class YoloDetectionHead(nn.Module):
    """Yolo detection head."""

    priors: Tensor
    """Model priors."""

    def __init__(
        self,
        in_channels: int,
        priors: Tensor | np.ndarray,
        num_classes: int
    ) -> None:
        """Initialize the detection head.

        Args:
            in_channels (int): Depth of the input feature map.
            priors (Tensor | np.ndarray): Pre-computed priors.
            num_classes (int): Number of output class.
        """
        super().__init__()

        self.num_classes = num_classes
        self.prior_count = priors.shape[0]

        if isinstance(priors, np.ndarray):
            priors = torch.from_numpy(priors)

        self.register_buffer("priors", priors)

        self.convolution_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.prior_count * (5 + num_classes),
            kernel_size=1
        )

    def decode_boxes(
        self,
        encoded_boxes: Tensor,
        input_size: tuple[int, int]
    ) -> Tensor:
        """Decode the output bounding boxes.

        The format of the tensor before the activation function is the following
        - x[..., 0] = x: the x position of the center of the box relative to its cell.
        - x[..., 1] = y: the y position of the center of the box relative to its cell.
        - x[..., 2] = w: the width of the box relative to its prior.
        - x[..., 3] = h: the height of the box relative to its prior.
        - x[..., 4] = L: the 'objectness' score of the box (pre-activation logit).
        - x[..., 5:] = Lc: the conditional probability of the box belonging to each
            class (pre-activation logit).

        The format of the tensor after the activation function is the following
        - y[..., 0] = x: the x position of the center of the box relative to the grid.
        - y[..., 1] = y: the y position of the center of the box relative to the grid.
        - y[..., 2] = w: the width of the box relative to the grid.
        - y[..., 3] = h: the height of the box relative to the grid.
        - y[..., 4] = P: the "objectness" score of the box.
        - y[..., 5:] = C: the conditional probability vector of the box belonging to
            each class.

        Args:
            encoded_boxes (Tensor): Encoded bounding boxes.
            input_size (tuple[int, int]): Shape of the input image.

        Returns:
            Tensor: Decoded bounding boxes.
        """
        grid_h, grid_w = encoded_boxes.shape[1:3]
        scale_h, scale_w = input_size[0] / grid_h, input_size[1] / grid_w

        # The predicted coordinates are local to the cell of the prediction, meaning it
        # is a value between 0 and 1, positioning the center of the box within that
        # cell. The goal of the activation function is to get the coordinates of the box
        # in relative to the grid (i.e. x in [0 WG[ and y in [0 HG[). To do this, first
        # we create a tensor in which each cell contains its own coordinates within the
        # grid (shape: (1 x HG x WG x 1 x 2)).
        cell_w = torch.arange(0, grid_w, dtype=torch.float32).to(encoded_boxes.device)
        cell_h = torch.arange(0, grid_h, dtype=torch.float32).to(encoded_boxes.device)
        cell_grid = torch.stack(torch.meshgrid(cell_w, cell_h, indexing="ij"))
        cell_grid = torch.swapaxes(cell_grid, 0, -1)
        cell_grid = torch.reshape(cell_grid, shape=(1, grid_h, grid_w, 1, 2))

        # Then, we add the coordinates of the cell to the coordinates of the box after
        # applying the sigmoid function (shape: (BS x HG x WG x B x 2)).
        predicted_xy = encoded_boxes[..., :2]
        predicted_xy = torch.sigmoid(predicted_xy)
        predicted_xy = cell_grid + predicted_xy

        # The predicted dimension of the box is relative to the associated prior. The
        # activation function used here is the exponential function, meaning that a
        # predicted size of 0 pre-activation will give the size of the prior (shape:
        # (BS x HG x WG x B x 2)).
        predicted_wh = encoded_boxes[..., 2:4]
        predicted_wh = torch.exp(predicted_wh)
        predicted_wh = self.priors * predicted_wh

        # Since the priors are relative to the input image size, we have to rescale the
        # boxes to the grid.
        predicted_wh[..., 0] /= scale_w
        predicted_wh[..., 1] /= scale_h

        # The predicted objectness is merely obtained by applying the sigmoid function
        # to the logit. This probability indicates whether the box actually contains an
        # object. The loss function used by YOLOv2 will make this value quantify the
        # quality of the box (shape: (BS x HG x WG x B x 1)).
        predicted_objectness = encoded_boxes[..., 4]
        predicted_objectness = torch.sigmoid(predicted_objectness)
        predicted_objectness = predicted_objectness.unsqueeze(dim=-1)

        # Finally, the conditional probability vector is obtained by applying the
        # softmax function (shape: (BS x HG x WG x B x C])).
        predicted_probabilities = encoded_boxes[..., 5:]
        predicted_probabilities = torch.softmax(predicted_probabilities, dim=-1)

        # The output tensor is then assembled by concatenating the previously computed
        # tensors (shape: (BS x HG x WG x B x (5+C))).
        decoded_boxes = torch.cat([
            predicted_xy,
            predicted_wh,
            predicted_objectness,
            predicted_probabilities
        ], axis=-1)

        return decoded_boxes

    def forward(
        self,
        feature_maps: Tensor,
        input_size: tuple[int, int]
    ) -> Tensor:
        """Compute bounding boxes based on the given feature map.

        Args:
            feature_maps (Tensor): Feature extractor output tensor.
            input_size (tuple[int, int]): Input sizes of each image of the batch.

        Returns:
            Tensor: Decoded bounding boxes.
        """
        encoded_boxes = self.convolution_head(feature_maps)

        # The decode function expects to receive a BS x HG x WG x B x (5+C) tensor.
        # Since torch uses a channels first format, we have a BS x [B*(5+C)] x HG x WG
        # tensor at this step, so we have to move the axes around. First, we switch from
        # channels first to channels last and then split the channels into priors x
        # classes.
        encoded_boxes_reshaped = torch.moveaxis(encoded_boxes, 1, -1)
        encoded_boxes_reshaped = torch.reshape(encoded_boxes_reshaped, shape=(
            *encoded_boxes_reshaped.shape[:-1], self.prior_count, 5 + self.num_classes
        ))

        decoded_boxes = self.decode_boxes(encoded_boxes_reshaped, input_size)

        return decoded_boxes
