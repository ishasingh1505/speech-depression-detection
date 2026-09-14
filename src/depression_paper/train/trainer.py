"""Training loops for depression models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depression_paper.data.dataset import (
    ContextWindowDataset,
    collate_batch,
)
from depression_paper.eval.metrics import depression_metrics


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.0005
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _class_weights(manifest, participant_ids):
    participant_labels = (
        manifest[
            manifest["participant_id"].isin(participant_ids)
        ]
        .groupby("participant_id")["phq8_binary"]
        .first()
    )

    counts = participant_labels.value_counts()

    total = len(participant_labels)

    count_0 = int(counts.get(0, 1))
    count_1 = int(counts.get(1, 1))

    weight_0 = total / (2.0 * count_0)
    weight_1 = total / (2.0 * count_1)

    return torch.tensor(
        [weight_0, weight_1],
        dtype=torch.float32,
    )


def _aggregate_predictions(
    predictions,
    pids,
    manifest,
    task,
):
    participant_rows = (
        manifest
        .groupby("participant_id")
        .first()
        .reset_index()
    )

    label_column = (
        "phq8_binary"
        if task == "detection"
        else "phq8_score"
    )

    pid_to_label = dict(
        zip(
            participant_rows["participant_id"],
            participant_rows[label_column],
        )
    )

    unique_pids = sorted(set(pids))

    y_true = []
    y_pred = []

    for pid in unique_pids:

        indices = [
            i
            for i, current_pid in enumerate(pids)
            if current_pid == pid
        ]

        participant_preds = [
            predictions[i]
            for i in indices
        ]

        if task == "detection":

            labels = np.asarray(
                participant_preds,
                dtype=np.int64,
            )

            count_0 = np.sum(labels == 0)
            count_1 = np.sum(labels == 1)

            prediction = (
                1
                if count_1 > count_0
                else 0
            )

        else:

            prediction = float(
                np.mean(
                    participant_preds
                )
            )

        y_pred.append(prediction)
        y_true.append(pid_to_label[pid])

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
    )


def _validate_manifest_indices(
    manifest,
    features,
):
    indices = manifest.index.to_numpy()

    if len(indices) == 0:
        raise ValueError(
            "Manifest contains no rows."
        )

    if indices.min() < 0:
        raise ValueError(
            "Manifest contains negative indices."
        )

    if indices.max() >= len(features):
        raise IndexError(
            "Manifest indices exceed the feature array. "
            "The manifest and feature array are not aligned."
        )


def train_model(
    model: nn.Module,
    train_manifest,
    val_manifest,
    features: np.ndarray,
    train_pids,
    val_pids,
    task: str,
    cfg: TrainConfig,
    context: int = 16,
):
    set_seed(cfg.random_seed)

    device = torch.device(cfg.device)

    model = model.to(device)

    _validate_manifest_indices(
        train_manifest,
        features,
    )

    _validate_manifest_indices(
        val_manifest,
        features,
    )

    train_indices = train_manifest.index.to_numpy()

    mean = np.mean(
        features[train_indices],
        axis=0,
        keepdims=True,
    )

    std = np.std(
        features[train_indices],
        axis=0,
        keepdims=True,
    )

    std = std + 1e-8

    norm_features = (
        features - mean
    ) / std

    train_dataset = ContextWindowDataset(
        train_manifest,
        norm_features,
        train_pids,
        context=context,
        task=task,
    )

    val_dataset = ContextWindowDataset(
        val_manifest,
        norm_features,
        val_pids,
        context=context,
        task=task,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    if task == "detection":
        criterion = nn.CrossEntropyLoss(
            weight=_class_weights(
                train_manifest,
                train_pids,
            ).to(device)
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(
            cfg.adam_beta1,
            cfg.adam_beta2,
        ),
    )

    for _ in range(cfg.epochs):

        model.train()

        for xs, ys, _ in train_loader:

            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()

            outputs = model(xs)

            if task == "detection":
                loss = criterion(
                    outputs,
                    ys,
                )
            else:
                loss = criterion(
                    outputs.squeeze(-1),
                    ys,
                )

            loss.backward()
            optimizer.step()

    model.eval()

    all_predictions = []
    all_pids = []

    with torch.no_grad():

        for xs, _, pids in val_loader:

            xs = xs.to(device)

            outputs = model(xs)

            if task == "detection":

                predictions = (
                    outputs
                    .argmax(dim=-1)
                    .cpu()
                    .numpy()
                )

            else:

                predictions = (
                    outputs
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

            all_predictions.extend(
                predictions.tolist()
            )

            all_pids.extend(pids)

    y_true, y_pred = _aggregate_predictions(
        all_predictions,
        all_pids,
        val_manifest,
        task,
    )

    metrics = depression_metrics(
        y_true,
        y_pred,
        task,
    )

    return metrics, model


class CombinedDataset(ContextWindowDataset):

    def __init__(
        self,
        manifest,
        emb_features,
        ac_features,
        participant_ids,
        context,
        task,
        stride=1,
    ):
        super().__init__(
            manifest=manifest,
            features=emb_features,
            participant_ids=participant_ids,
            context=context,
            task=task,
            stride=stride,
        )

        self.ac_features = ac_features.astype(
            np.float32
        )

    def __getitem__(self, idx):

        window_indices, pid, target = (
            self.windows[idx]
        )

        emb = self.features[
            window_indices
        ]

        acoustic = self.ac_features[
            window_indices
        ]

        if self.task == "detection":
            y = torch.tensor(
                target,
                dtype=torch.long,
            )
        else:
            y = torch.tensor(
                target,
                dtype=torch.float32,
            )

        return (
            torch.from_numpy(emb),
            torch.from_numpy(acoustic),
            y,
            pid,
        )


def collate_combined(batch):

    embeddings, acoustic, ys, pids = zip(
        *batch
    )

    return (
        torch.stack(embeddings),
        torch.stack(acoustic),
        torch.stack(ys),
        list(pids),
    )


def train_combined_model(
    model: nn.Module,
    train_manifest,
    val_manifest,
    emb_features: np.ndarray,
    ac_features: np.ndarray,
    train_pids,
    val_pids,
    task: str,
    cfg: TrainConfig,
    context: int = 16,
):
    set_seed(cfg.random_seed)

    device = torch.device(cfg.device)

    model = model.to(device)

    _validate_manifest_indices(
        train_manifest,
        emb_features,
    )

    _validate_manifest_indices(
        val_manifest,
        emb_features,
    )

    _validate_manifest_indices(
        train_manifest,
        ac_features,
    )

    _validate_manifest_indices(
        val_manifest,
        ac_features,
    )

    train_indices = train_manifest.index.to_numpy()

    emb_mean = np.mean(
        emb_features[train_indices],
        axis=0,
        keepdims=True,
    )

    emb_std = np.std(
        emb_features[train_indices],
        axis=0,
        keepdims=True,
    ) + 1e-8

    ac_mean = np.mean(
        ac_features[train_indices],
        axis=0,
        keepdims=True,
    )

    ac_std = np.std(
        ac_features[train_indices],
        axis=0,
        keepdims=True,
    ) + 1e-8

    norm_emb = (
        emb_features - emb_mean
    ) / emb_std

    norm_ac = (
        ac_features - ac_mean
    ) / ac_std

    train_dataset = CombinedDataset(
        train_manifest,
        norm_emb,
        norm_ac,
        train_pids,
        context,
        task,
    )

    val_dataset = CombinedDataset(
        val_manifest,
        norm_emb,
        norm_ac,
        val_pids,
        context,
        task,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_combined,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_combined,
    )

    if task == "detection":

        criterion = nn.CrossEntropyLoss(
            weight=_class_weights(
                train_manifest,
                train_pids,
            ).to(device)
        )

    else:

        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(
            cfg.adam_beta1,
            cfg.adam_beta2,
        ),
    )

    for _ in range(cfg.epochs):

        model.train()

        for (
            emb_x,
            ac_x,
            ys,
            _,
        ) in train_loader:

            emb_x = emb_x.to(device)
            ac_x = ac_x.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()

            outputs = model(
                emb_x,
                ac_x,
            )

            if task == "detection":

                loss = criterion(
                    outputs,
                    ys,
                )

            else:

                loss = criterion(
                    outputs.squeeze(-1),
                    ys,
                )

            loss.backward()
            optimizer.step()

    model.eval()

    all_predictions = []
    all_pids = []

    with torch.no_grad():

        for (
            emb_x,
            ac_x,
            _,
            pids,
        ) in val_loader:

            emb_x = emb_x.to(device)
            ac_x = ac_x.to(device)

            outputs = model(
                emb_x,
                ac_x,
            )

            if task == "detection":

                predictions = (
                    outputs
                    .argmax(dim=-1)
                    .cpu()
                    .numpy()
                )

            else:

                predictions = (
                    outputs
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

            all_predictions.extend(
                predictions.tolist()
            )

            all_pids.extend(pids)

    y_true, y_pred = _aggregate_predictions(
        all_predictions,
        all_pids,
        val_manifest,
        task,
    )

    metrics = depression_metrics(
        y_true,
        y_pred,
        task,
    )

    return metrics, model