#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from depression_paper.config import load_config
from depression_paper.data.folds import create_speaker_folds
from depression_paper.data.dataset import ContextWindowDataset
from depression_paper.models.lstm_net import LSTMModel
from depression_paper.models.mk_cnn import MKCNNModel
from depression_paper.models.dnn import BaselineDNN


def check_manifest(cfg):
    path = (
        Path(cfg["preprocessing"]["output_dir"])
        / "segments_manifest.csv"
    )

    if not path.exists():
        print("Manifest not found.")
        print("Run 01_preprocess.py first.")
        return None

    manifest = pd.read_csv(path)

    print(
        f"Participants: "
        f"{manifest['participant_id'].nunique()}"
    )

    print(
        f"Segments: {len(manifest)}"
    )

    participant_labels = (
        manifest
        .groupby("participant_id")["phq8_binary"]
        .first()
    )

    depressed = int(participant_labels.sum())
    healthy = len(participant_labels) - depressed

    print(f"Depressed: {depressed}")
    print(f"Healthy: {healthy}")

    return manifest


def check_folds(manifest, cfg):
    folds = create_speaker_folds(
        manifest,
        cfg["training"]["n_folds"],
        cfg["training"]["random_seed"],
    )

    print(
        f"\nCreated {len(folds)} folds."
    )

    for i, fold in enumerate(folds):

        train_ids = set(
            fold["train_participants"]
        )

        val_ids = set(
            fold["val_participants"]
        )

        overlap = train_ids & val_ids

        if overlap:
            raise RuntimeError(
                f"Fold {i} has speaker leakage."
            )

        print(
            f"Fold {i}: "
            f"train={len(train_ids)}, "
            f"validation={len(val_ids)}"
        )


def check_feature(
    manifest,
    feature_path,
    name,
):
    if not feature_path.exists():
        print(
            f"\n{name}: not extracted yet."
        )
        return None

    features = np.load(
        feature_path
    )

    print(
        f"\n{name}: {features.shape}"
    )

    if len(manifest) != len(features):
        raise RuntimeError(
            f"{name} has {len(features)} rows "
            f"but manifest has {len(manifest)} rows."
        )

    return features


def check_models(feature_dim):
    context = 16

    x = torch.randn(
        4,
        context,
        feature_dim,
    )

    print("\nTesting LSTM...")

    lstm = LSTMModel(
        feature_dim=feature_dim,
        task="detection",
    )

    output = lstm(x)

    print(
        f"LSTM output: {tuple(output.shape)}"
    )

    print("\nTesting MK-CNN...")

    cnn = MKCNNModel(
        feature_dim=feature_dim,
        task="detection",
    )

    output = cnn(x)

    print(
        f"MK-CNN output: {tuple(output.shape)}"
    )

    print("\nTesting DNN...")

    dnn = BaselineDNN(
        feature_dim=feature_dim,
        context=context,
        task="detection",
    )

    output = dnn(x)

    print(
        f"DNN output: {tuple(output.shape)}"
    )


def check_dataset(
    manifest,
    features,
    cfg,
):
    if features is None:
        return

    participants = (
        manifest[
            "participant_id"
        ]
        .unique()
        .tolist()
    )

    context = cfg["training"]["temporal_context"]

    dataset = ContextWindowDataset(
        manifest=manifest,
        features=features,
        participant_ids=participants,
        context=context,
        task="detection",
    )

    print(
        f"\nContext windows: "
        f"{len(dataset)}"
    )

    if len(dataset) == 0:
        print(
            f"No {context}-segment windows "
            "were created."
        )
        return

    x, y, pid = dataset[0]

    print(
        f"Window shape: "
        f"{tuple(x.shape)}"
    )

    print(
        f"Label: {y.item()}"
    )

    print(
        f"Participant: {pid}"
    )


def main():
    config_path = (
        ROOT
        / "config"
        / "e_daic.yaml"
    )

    cfg = load_config(
        config_path
    )

    print(
        "=== E-DAIC PIPELINE CHECK ==="
    )

    manifest = check_manifest(
        cfg
    )

    if manifest is None:
        return

    check_folds(
        manifest,
        cfg,
    )

    feature_dir = Path(
        cfg["features"]["output_dir"]
    )

    ecapa = check_feature(
        manifest,
        feature_dir
        / "ecapa_embeddings.npy",
        "ECAPA",
    )

    opensmile = check_feature(
        manifest,
        feature_dir
        / "opensmile_is09.npy",
        "OpenSMILE",
    )

    if ecapa is not None:
        check_dataset(
            manifest,
            ecapa,
            cfg,
        )

        check_models(
            ecapa.shape[1]
        )

    if opensmile is not None:
        print(
            f"\nOpenSMILE dimension: "
            f"{opensmile.shape[1]}"
        )

    print(
        "\n=== CHECK COMPLETE ==="
    )


if __name__ == "__main__":
    main()