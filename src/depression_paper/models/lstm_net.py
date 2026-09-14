"""LSTM models for depression detection and severity estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    2-layer LSTM model with 128 hidden units per layer.

    The final timestep is passed through a 100-unit fully
    connected layer before the task-specific output layer.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        fc_units: int = 100,
        dropout_lstm: float = 0.4,
        dropout_fc: float = 0.3,
        task: str = "detection",
    ):
        super().__init__()

        self.task = task

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout_lstm = nn.Dropout(
            dropout_lstm
        )

        self.fc = nn.Linear(
            hidden_size,
            fc_units,
        )

        self.dropout_fc = nn.Dropout(
            dropout_fc
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

        output, _ = self.lstm(x)

        # Use the representation at the final
        # timestep of the temporal context.
        last_step = output[:, -1, :]

        last_step = self.dropout_lstm(
            last_step
        )

        h = F.relu(
            self.fc(last_step)
        )

        h = self.dropout_fc(h)

        return self.out_layer(h)