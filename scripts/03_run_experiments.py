#!/usr/bin/env python3

"""Run 5-Fold Cross Validation Experiments on E-DAIC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depression_paper.models.combined import CombinedEmbeddingsModel
from depression_paper.models.lstm_net import LSTMModel
from depression_paper.models.mk_cnn import MKCNNModel
from depression_paper.train.trainer import (
    TrainConfig,
    train_combined_model,
    train_model,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config


# ---------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------

def validate_feature_alignment(
    manifest: pd.DataFrame,
    *features: np.ndarray,
) -> None:
    """
    Verify that feature arrays are aligned with the
    complete global manifest.

    The manifest index is the row index into the feature
    arrays and must therefore NOT be reset after filtering.
    """

    if len(manifest) == 0:
        raise ValueError(
            "Manifest is empty."
        )

    manifest_indices = manifest.index.to_numpy()

    if not manifest.index.is_unique:
        raise ValueError(
            "Manifest index must be unique."
        )

    if manifest_indices.min() < 0:
        raise ValueError(
            "Manifest contains negative indices."
        )

    expected_length = len(manifest)

    for feature_array in features:

        if feature_array.ndim != 2:
            raise ValueError(
                "Feature arrays must be 2-dimensional."
            )

        if len(feature_array) != expected_length:
            raise ValueError(
                "Feature array length does not match "
                f"the complete manifest: "
                f"{len(feature_array)} vs "
                f"{expected_length}."
            )


# ---------------------------------------------------------
# Result saving
# ---------------------------------------------------------

def save_results(
    summary: dict,
    results_root: Path,
) -> Path:
    """
    Save experiment summary as JSON.

    Results are stored under:

        results/e_daic/detection/
        results/e_daic/severity/
    """

    task = summary["task"]
    feature = summary["feature"]
    model = summary["model"]

    output_dir = (
        results_root
        / task
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{feature}_{model}.json"
    )

    output_path = (
        output_dir / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    return output_path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run E-DAIC 5-Fold Experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        choices=[
            "opensmile",
            "ecapa",
            "ecapa_opensmile",
        ],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=[
            "lstm",
            "mk_cnn",
        ],
    )

    parser.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=[
            "detection",
            "severity",
        ],
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    cfg_dict = load_config(
        args.config
    )

    proc_dir = Path(
        cfg_dict.get(
            "preprocessing",
            {},
        ).get(
            "output_dir",
            "data/processed",
        )
    )

    feat_dir = Path(
        cfg_dict.get(
            "features",
            {},
        ).get(
            "output_dir",
            "data/features",
        )
    )

    # Results always go to project/results/e_daic
    results_root = (
        PROJECT_ROOT
        / "results"
        / "e_daic"
    )

    manifest_path = (
        proc_dir
        / "segments_manifest.csv"
    )

    folds_path = (
        proc_dir
        / "folds.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}\n"
            "Run 01_preprocess.py first."
        )

    if not folds_path.exists():
        raise FileNotFoundError(
            f"Missing folds file: {folds_path}\n"
            "Run 01_preprocess.py first."
        )

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    manifest = pd.read_csv(
        manifest_path
    )

    # Feature arrays were created in exactly this order.
    # Explicitly restore the global row index.
    manifest.index = np.arange(
        len(manifest)
    )

    # -----------------------------------------------------
    # Load folds
    # -----------------------------------------------------

    with open(
        folds_path,
        "r",
        encoding="utf-8",
    ) as f:
        folds_data = json.load(f)

    if len(folds_data) != 5:
        raise ValueError(
            f"Expected 5 folds, found "
            f"{len(folds_data)}."
        )

    # -----------------------------------------------------
    # Load features
    # -----------------------------------------------------

    if args.feature == "opensmile":

        feature_path = (
            feat_dir
            / "opensmile_is09.npy"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing OpenSMILE features: "
                f"{feature_path}"
            )

        features = np.load(
            feature_path
        )

        validate_feature_alignment(
            manifest,
            features,
        )

        print(
            f"Loaded OpenSMILE features: "
            f"{features.shape}"
        )

    elif args.feature == "ecapa":

        feature_path = (
            feat_dir
            / "ecapa_embeddings.npy"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing ECAPA features: "
                f"{feature_path}"
            )

        features = np.load(
            feature_path
        )

        validate_feature_alignment(
            manifest,
            features,
        )

        print(
            f"Loaded ECAPA features: "
            f"{features.shape}"
        )

    else:

        emb_path = (
            feat_dir
            / "ecapa_embeddings.npy"
        )

        acoustic_path = (
            feat_dir
            / "opensmile_is09.npy"
        )

        if not emb_path.exists():
            raise FileNotFoundError(
                f"Missing ECAPA features: "
                f"{emb_path}"
            )

        if not acoustic_path.exists():
            raise FileNotFoundError(
                f"Missing OpenSMILE features: "
                f"{acoustic_path}"
            )

        emb_features = np.load(
            emb_path
        )

        ac_features = np.load(
            acoustic_path
        )

        validate_feature_alignment(
            manifest,
            emb_features,
            ac_features,
        )

        print(
            f"Loaded ECAPA features: "
            f"{emb_features.shape}"
        )

        print(
            f"Loaded OpenSMILE features: "
            f"{ac_features.shape}"
        )

    # -----------------------------------------------------
    # Training configuration
    # -----------------------------------------------------

    training_cfg = cfg_dict.get(
        "training",
        {},
    )

    dropout_cfg = training_cfg.get(
        "dropout",
        {},
    )

    cfg = TrainConfig(
        epochs=training_cfg.get(
            "epochs",
            50,
        ),
        batch_size=training_cfg.get(
            "batch_size",
            128,
        ),
        learning_rate=training_cfg.get(
            "learning_rate",
            0.0005,
        ),
        adam_beta1=training_cfg.get(
            "adam_beta1",
            0.9,
        ),
        adam_beta2=training_cfg.get(
            "adam_beta2",
            0.99,
        ),
        device=training_cfg.get(
            "device",
            "cuda"
            if torch.cuda.is_available()
            else "cpu",
        ),
    )

    context = training_cfg.get(
        "temporal_context",
        16,
    )

    print(
        f"\nFeature: {args.feature}"
    )
    print(
        f"Model: {args.model}"
    )
    print(
        f"Task: {args.task}"
    )
    print(
        f"Temporal context: {context}"
    )
    print(
        f"Device: {cfg.device}"
    )
    print(
        f"Results directory: {results_root}"
    )

    fold_metrics = []

    # -----------------------------------------------------
    # Five-fold speaker-independent CV
    # -----------------------------------------------------

    for fold_idx, fold_info in enumerate(
        folds_data
    ):

        print(
            f"\n========== Fold {fold_idx} =========="
        )

        train_pids = [
            int(pid)
            for pid in fold_info["train"]
        ]

        val_pids = [
            int(pid)
            for pid in fold_info["val"]
        ]

        # -------------------------------------------------
        # IMPORTANT:
        # Do NOT reset index here.
        #
        # The index is the global row ID into the feature
        # arrays.
        # -------------------------------------------------

        train_manifest = manifest[
            manifest["participant_id"].isin(
                train_pids
            )
        ].copy()

        val_manifest = manifest[
            manifest["participant_id"].isin(
                val_pids
            )
        ].copy()

        if train_manifest.empty:
            raise ValueError(
                f"Fold {fold_idx}: "
                "training manifest is empty."
            )

        if val_manifest.empty:
            raise ValueError(
                f"Fold {fold_idx}: "
                "validation manifest is empty."
            )

        # -------------------------------------------------
        # Participant independence check
        # -------------------------------------------------

        overlap = (
            set(train_pids)
            & set(val_pids)
        )

        if overlap:
            raise ValueError(
                f"Fold {fold_idx} has participant "
                f"overlap: {sorted(overlap)}"
            )

        print(
            f"Training participants: "
            f"{len(train_pids)}"
        )

        print(
            f"Validation participants: "
            f"{len(val_pids)}"
        )

        print(
            f"Training segments: "
            f"{len(train_manifest)}"
        )

        print(
            f"Validation segments: "
            f"{len(val_manifest)}"
        )

        # -------------------------------------------------
        # Combined ECAPA + OpenSMILE
        # -------------------------------------------------

        if args.feature == "ecapa_opensmile":

            model = CombinedEmbeddingsModel(
                embedding_dim=(
                    emb_features.shape[1]
                ),
                acoustic_dim=(
                    ac_features.shape[1]
                ),
                backbone=args.model,
                fc_units=training_cfg.get(
                    "fc_units",
                    100,
                ),
                task=args.task,
                dropout_lstm=dropout_cfg.get(
                    "lstm",
                    0.4,
                ),
                dropout_cnn=dropout_cfg.get(
                    "cnn",
                    0.3,
                ),
                dropout_fc=dropout_cfg.get(
                    "fc",
                    0.3,
                ),
            )

            metrics, _ = train_combined_model(
                model=model,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                emb_features=emb_features,
                ac_features=ac_features,
                train_pids=train_pids,
                val_pids=val_pids,
                task=args.task,
                cfg=cfg,
                context=context,
            )

        # -------------------------------------------------
        # Single feature
        # -------------------------------------------------

        else:

            if args.model == "lstm":

                model = LSTMModel(
                    feature_dim=features.shape[1],
                    fc_units=training_cfg.get(
                        "fc_units",
                        100,
                    ),
                    dropout_lstm=dropout_cfg.get(
                        "lstm",
                        0.4,
                    ),
                    dropout_fc=dropout_cfg.get(
                        "fc",
                        0.3,
                    ),
                    task=args.task,
                )

            else:

                model = MKCNNModel(
                    feature_dim=features.shape[1],
                    fc_units=training_cfg.get(
                        "fc_units",
                        100,
                    ),
                    dropout_cnn=dropout_cfg.get(
                        "cnn",
                        0.3,
                    ),
                    dropout_fc=dropout_cfg.get(
                        "fc",
                        0.3,
                    ),
                    task=args.task,
                )

            metrics, _ = train_model(
                model=model,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                features=features,
                train_pids=train_pids,
                val_pids=val_pids,
                task=args.task,
                cfg=cfg,
                context=context,
            )

        metrics["fold"] = fold_idx

        print(
            f"Fold {fold_idx}: {metrics}"
        )

        fold_metrics.append(
            metrics
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    if args.task == "detection":

        bacs = [
            m["BAc"]
            for m in fold_metrics
        ]

        f1_d = [
            m["F1(D)"]
            for m in fold_metrics
        ]

        f1_h = [
            m["F1(H)"]
            for m in fold_metrics
        ]

        summary = {
            "feature": args.feature,
            "model": args.model,
            "task": args.task,
            "F1(D)": float(
                np.mean(f1_d)
            ),
            "F1(H)": float(
                np.mean(f1_h)
            ),
            "BAc": float(
                np.mean(bacs)
            ),
            "F1(D)_std": float(
                np.std(f1_d)
            ),
            "F1(H)_std": float(
                np.std(f1_h)
            ),
            "BAc_std": float(
                np.std(bacs)
            ),
            "folds": fold_metrics,
        }

    else:

        rmses = [
            m["RMSE"]
            for m in fold_metrics
        ]

        summary = {
            "feature": args.feature,
            "model": args.model,
            "task": args.task,
            "RMSE": float(
                np.mean(rmses)
            ),
            "RMSE_std": float(
                np.std(rmses)
            ),
            "folds": fold_metrics,
        }

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print(
        "\n================ SUMMARY ================"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    output_path = save_results(
        summary,
        results_root,
    )

    print(
        f"\nResult saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()