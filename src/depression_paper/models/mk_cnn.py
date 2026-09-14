"""Multi-kernel CNN model for depression detection and severity estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MKCNNModel(nn.Module):
    """
    Multi-kernel CNN for temporal-context depression modelling.

    Architecture:

        Input: (batch, context, feature_dim)

        Three parallel first convolutional branches:
            Conv2D kernel (3, feature_dim), 50 channels
            Conv2D kernel (4, feature_dim), 50 channels
            Conv2D kernel (5, feature_dim), 50 channels

        Each branch:
            ReLU
            Dropout
            Conv2D kernel (4, 1), 50 channels
            ReLU

        Branch outputs are flattened and concatenated.

        FC: 100 units
        ReLU
        Dropout

        Detection: 2 outputs
        Severity: 1 output
    """

    def __init__(
        self,
        feature_dim: int,
        n_channels: int = 50,
        fc_units: int = 100,
        dropout_cnn: float = 0.3,
        dropout_fc: float = 0.3,
        task: str = "detection",
        context: int = 16,
    ):
        super().__init__()

        if context < 6:
            raise ValueError(
                f"Context length {context} is too small "
                "for the MK-CNN architecture."
            )

        if feature_dim <= 0:
            raise ValueError(
                "feature_dim must be positive."
            )

        self.task = task
        self.feature_dim = feature_dim
        self.context = context
        self.n_channels = n_channels

        # --------------------------------------------------
        # First multi-kernel convolutional layer
        # --------------------------------------------------

        self.conv1_3 = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(3, feature_dim),
        )

        self.conv1_4 = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(4, feature_dim),
        )

        self.conv1_5 = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=(5, feature_dim),
        )

        # --------------------------------------------------
        # Second convolutional layer
        #
        # Each first-layer branch has its own second
        # convolution. This preserves the temporal
        # dimensions of the three branches.
        # --------------------------------------------------

        self.conv2_3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4, 1),
        )

        self.conv2_4 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4, 1),
        )

        self.conv2_5 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4, 1),
        )

        self.dropout_cnn = nn.Dropout(
            dropout_cnn
        )

        self.dropout_fc = nn.Dropout(
            dropout_fc
        )

        # --------------------------------------------------
        # Calculate flattened dimensions
        # --------------------------------------------------
        #
        # After first convolution:
        #
        # kernel 3 -> context - 3 + 1
        # kernel 4 -> context - 4 + 1
        # kernel 5 -> context - 5 + 1
        #
        # After second kernel-4 convolution:
        #
        # branch3 = context - 3 + 1 - 4 + 1
        # branch4 = context - 4 + 1 - 4 + 1
        # branch5 = context - 5 + 1 - 4 + 1
        #
        # Width becomes 1 because the first convolution
        # spans the complete feature dimension.

        branch3 = (
            context - 3 + 1 - 4 + 1
        )

        branch4 = (
            context - 4 + 1 - 4 + 1
        )

        branch5 = (
            context - 5 + 1 - 4 + 1
        )

        if min(
            branch3,
            branch4,
            branch5,
        ) <= 0:
            raise ValueError(
                f"Context length {context} is too small "
                "for the MK-CNN architecture."
            )

        flattened_dim = n_channels * (
            branch3
            + branch4
            + branch5
        )

        # --------------------------------------------------
        # Fully connected layer
        # --------------------------------------------------

        self.fc = nn.Linear(
            flattened_dim,
            fc_units,
        )

        if task == "detection":
            out_units = 2
        elif task == "severity":
            out_units = 1
        else:
            raise ValueError(
                f"Unknown task: {task}"
            )

        self.out_layer = nn.Linear(
            fc_units,
            out_units,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Expected input:
        #
        # (batch, context, feature_dim)
        #
        # Conv2D expects:
        #
        # (batch, channels, context, feature_dim)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        elif x.dim() != 4:
            raise ValueError(
                "MKCNNModel expects input with shape "
                "(batch, context, feature_dim) or "
                "(batch, 1, context, feature_dim)."
            )

        # --------------------------------------------------
        # Branch 1: kernel size 3
        # --------------------------------------------------

        h3 = F.relu(
            self.conv1_3(x)
        )

        h3 = self.dropout_cnn(h3)

        h3 = F.relu(
            self.conv2_3(h3)
        )

        h3 = torch.flatten(
            h3,
            start_dim=1,
        )

        # --------------------------------------------------
        # Branch 2: kernel size 4
        # --------------------------------------------------

        h4 = F.relu(
            self.conv1_4(x)
        )

        h4 = self.dropout_cnn(h4)

        h4 = F.relu(
            self.conv2_4(h4)
        )

        h4 = torch.flatten(
            h4,
            start_dim=1,
        )

        # --------------------------------------------------
        # Branch 3: kernel size 5
        # --------------------------------------------------

        h5 = F.relu(
            self.conv1_5(x)
        )

        h5 = self.dropout_cnn(h5)

        h5 = F.relu(
            self.conv2_5(h5)
        )

        h5 = torch.flatten(
            h5,
            start_dim=1,
        )

        # --------------------------------------------------
        # Concatenate the three temporal branches
        # --------------------------------------------------

        fused = torch.cat(
            [
                h3,
                h4,
                h5,
            ],
            dim=1,
        )

        # --------------------------------------------------
        # FC-100 representation
        # --------------------------------------------------

        h = F.relu(
            self.fc(fused)
        )

        h = self.dropout_fc(h)

        return self.out_layer(h)