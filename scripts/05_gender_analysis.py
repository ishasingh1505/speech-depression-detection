#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from depression_paper.config import load_config
from depression_paper.eval.metrics import depression_metrics


def participant_features(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:

    if len(manifest) != len(embeddings):
        raise ValueError(
            "Manifest and embedding arrays "
            "must have the same number of rows."
        )

    rows = []

    for pid, group in manifest.groupby(
        "participant_id",
        sort=True,
    ):

        indices = group.index.to_numpy()

        participant_embedding = (
            embeddings[indices].mean(axis=0)
        )

        label = int(
            group["phq8_binary"].iloc[0]
        )

        gender = int(
            group["gender"].iloc[0]
        )

        rows.append(
            {
                "participant_id": int(pid),
                "gender": gender,
                "label": label,
                "embedding": participant_embedding,
            }
        )

    return pd.DataFrame(rows)


def gender_specific_detection(
    participant_df: pd.DataFrame,
    gender: int,
) -> dict:

    data = participant_df[
        participant_df["gender"] == gender
    ].copy()

    if len(data) < 4:
        return {
            "n_participants": int(len(data)),
            "error": "Not enough participants.",
        }

    X = np.stack(
        data["embedding"].to_numpy()
    )

    y = data["label"].to_numpy()

    if len(np.unique(y)) < 2:
        return {
            "n_participants": int(len(data)),
            "error": "Only one depression class is present.",
        }

    clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(
        X,
        y,
    )

    predictions = clf.predict(
        X
    )

    metrics = depression_metrics(
        y,
        predictions,
        "detection",
    )

    metrics["n_participants"] = int(
        len(data)
    )

    return metrics


def demographics_baseline(
    participant_df: pd.DataFrame,
) -> dict:

    data = participant_df.copy()

    X = data[
        ["gender"]
    ].to_numpy()

    y = data[
        "label"
    ].to_numpy()

    if len(np.unique(y)) < 2:
        return {
            "error": "Only one depression class is present."
        }

    clf = SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(
        X,
        y,
    )

    predictions = clf.predict(
        X
    )

    metrics = depression_metrics(
        y,
        predictions,
        "detection",
    )

    metrics["n_participants"] = int(
        len(data)
    )

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/daic_woz.yaml"
    )

    parser.add_argument(
        "--embedding",
        default="ecapa",
        choices=[
            "ecapa",
            "xvector",
            "dvector",
        ],
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT / args.config
    )

    manifest_path = (
        Path(
            cfg["preprocessing"]["output_dir"]
        )
        / "segments_manifest.csv"
    )

    feature_path = (
        Path(
            cfg["features"]["output_dir"]
        )
        / f"{args.embedding}_embeddings.npy"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {feature_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    embeddings = np.load(
        feature_path
    )

    participant_df = participant_features(
        manifest,
        embeddings,
    )

    results = {
        "embedding": args.embedding,
        "demographics_baseline":
            demographics_baseline(
                participant_df
            ),
        "female_ecapa":
            gender_specific_detection(
                participant_df,
                gender=1,
            ),
        "male_ecapa":
            gender_specific_detection(
                participant_df,
                gender=0,
            ),
    }

    results_dir = Path(
        cfg["output"]["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "gender_analysis.json"
    )

    with output_path.open(
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()