"""DNN baseline for depression detection and severity estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineDNN(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        context: int = 16,
        task: str = "detection",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.context = context
        self.task = task

        input_dim = feature_dim * context

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)

        self.dropout = nn.Dropout(dropout)

        if task == "detection":
            self.output = nn.Linear(128, 2)
        elif task == "severity":
            self.output = nn.Linear(128, 1)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        return self.output(x)