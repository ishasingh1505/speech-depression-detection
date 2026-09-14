"""Combined speaker-embedding and acoustic-feature models."""

from __future__ import annotations

import torch
import torch.nn as nn

from depression_paper.models.lstm_net import (
    LSTMModel,
)
from depression_paper.models.mk_cnn import (
    MKCNNModel,
)


class CombinedEmbeddingsModel(nn.Module):
    """
    Combined speaker-embedding + acoustic-feature model.

    Two independent branches process:
        1. speaker embeddings
        2. acoustic features

    Each branch produces a 100-dimensional representation.

    The two representations are combined by element-wise
    multiplication, implementing the paper's described
    dot-product-style fusion.

    The fused 100-dimensional representation is then passed
    through the final task-specific classifier.
    """

    def __init__(
        self,
        embedding_dim: int,
        acoustic_dim: int,
        backbone: str = "lstm",
        fc_units: int = 100,
        task: str = "detection",
        dropout_lstm: float = 0.4,
        dropout_cnn: float = 0.3,
        dropout_fc: float = 0.3,
        context: int = 16,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if acoustic_dim <= 0:
            raise ValueError(
                "acoustic_dim must be positive."
            )

        if backbone not in {
            "lstm",
            "mk_cnn",
        }:
            raise ValueError(
                f"Unknown backbone: {backbone}"
            )

        if task not in {
            "detection",
            "severity",
        }:
            raise ValueError(
                f"Unknown task: {task}"
            )

        self.task = task
        self.backbone = backbone
        self.fc_units = fc_units

        # --------------------------------------------------
        # Speaker embedding branch
        # --------------------------------------------------

        if backbone == "lstm":

            self.spk_branch = LSTMModel(
                feature_dim=embedding_dim,
                hidden_size=128,
                num_layers=2,
                fc_units=fc_units,
                dropout_lstm=dropout_lstm,
                dropout_fc=dropout_fc,
                task=task,
            )

            self.acoustic_branch = LSTMModel(
                feature_dim=acoustic_dim,
                hidden_size=128,
                num_layers=2,
                fc_units=fc_units,
                dropout_lstm=dropout_lstm,
                dropout_fc=dropout_fc,
                task=task,
            )

        else:

            self.spk_branch = MKCNNModel(
                feature_dim=embedding_dim,
                n_channels=50,
                fc_units=fc_units,
                dropout_cnn=dropout_cnn,
                dropout_fc=dropout_fc,
                task=task,
                context=context,
            )

            self.acoustic_branch = MKCNNModel(
                feature_dim=acoustic_dim,
                n_channels=50,
                fc_units=fc_units,
                dropout_cnn=dropout_cnn,
                dropout_fc=dropout_fc,
                task=task,
                context=context,
            )

        # --------------------------------------------------
        # Remove each branch's task-specific output layer.
        #
        # This leaves:
        #
        # branch → FC-100 → dropout → 100-D representation
        #
        # instead of:
        #
        # branch → FC-100 → output
        # --------------------------------------------------

        self.spk_branch.out_layer = (
            nn.Identity()
        )

        self.acoustic_branch.out_layer = (
            nn.Identity()
        )

        # --------------------------------------------------
        # Final classifier
        # --------------------------------------------------

        if task == "detection":
            out_dim = 2
        else:
            out_dim = 1

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(
                fc_units,
                out_dim,
            ),
        )

    def forward(
        self,
        spk_x: torch.Tensor,
        acoustic_x: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Obtain 100-D representations from both branches
        # --------------------------------------------------

        h_spk = self.spk_branch(
            spk_x
        )

        h_acoustic = self.acoustic_branch(
            acoustic_x
        )

        # Both must be 100-dimensional.
        if h_spk.shape[-1] != self.fc_units:
            raise RuntimeError(
                "Speaker branch produced "
                f"{h_spk.shape[-1]} dimensions; "
                f"expected {self.fc_units}."
            )

        if h_acoustic.shape[-1] != self.fc_units:
            raise RuntimeError(
                "Acoustic branch produced "
                f"{h_acoustic.shape[-1]} dimensions; "
                f"expected {self.fc_units}."
            )

        # --------------------------------------------------
        # Fusion
        #
        # Element-wise multiplication of the two 100-D
        # representations.
        # --------------------------------------------------

        fused = (
            h_spk * h_acoustic
        )

        # --------------------------------------------------
        # Final prediction
        # --------------------------------------------------

        return self.classifier(
            fused
        )