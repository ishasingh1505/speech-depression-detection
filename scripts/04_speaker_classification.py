#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from depression_paper.config import load_config


def equal_error_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    fnr = 1.0 - tpr

    differences = np.abs(
        fpr - fnr
    )

    index = np.argmin(
        differences
    )

    return float(
        (fpr[index] + fnr[index]) / 2.0
    )


def create_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_positive: int = 5000,
    max_negative: int = 5000,
):
    rng = np.random.RandomState(42)

    positive_scores = []
    negative_scores = []

    unique_labels = np.unique(
        labels
    )

    for label in unique_labels:

        indices = np.where(
            labels == label
        )[0]

        if len(indices) < 2:
            continue

        shuffled = indices.copy()

        rng.shuffle(
            shuffled
        )

        n_pairs = min(
            len(shuffled) // 2,
            max_positive,
        )

        for i in range(n_pairs):

            a = shuffled[
                2 * i
            ]

            b = shuffled[
                2 * i + 1
            ]

            similarity = np.dot(
                embeddings[a],
                embeddings[b],
            )

            norm_a = np.linalg.norm(
                embeddings[a]
            )

            norm_b = np.linalg.norm(
                embeddings[b]
            )

            if norm_a == 0 or norm_b == 0:
                continue

            similarity /= (
                norm_a * norm_b
            )

            positive_scores.append(
                similarity
            )

    for _ in range(
        max_negative
    ):

        label_a, label_b = rng.choice(
            unique_labels,
            size=2,
            replace=False,
        )

        index_a = rng.choice(
            np.where(
                labels == label_a
            )[0]
        )

        index_b = rng.choice(
            np.where(
                labels == label_b
            )[0]
        )

        vector_a = embeddings[
            index_a
        ]

        vector_b = embeddings[
            index_b
        ]

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        if norm_a == 0 or norm_b == 0:
            continue

        similarity = np.dot(
            vector_a,
            vector_b,
        ) / (
            norm_a * norm_b
        )

        negative_scores.append(
            similarity
        )

    scores = np.concatenate(
        [
            np.asarray(
                positive_scores
            ),
            np.asarray(
                negative_scores
            ),
        ]
    )

    targets = np.concatenate(
        [
            np.ones(
                len(positive_scores),
                dtype=int,
            ),
            np.zeros(
                len(negative_scores),
                dtype=int,
            ),
        ]
    )

    return targets, scores


def speaker_verification_eer(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    embedding_name: str,
) -> dict:

    labels = manifest[
        "participant_id"
    ].to_numpy()

    if len(embeddings) != len(labels):
        raise ValueError(
            f"{embedding_name}: "
            f"{len(embeddings)} embeddings "
            f"for {len(labels)} manifest rows."
        )

    targets, scores = create_pairs(
        embeddings,
        labels,
    )

    eer = equal_error_rate(
        targets,
        scores,
    )

    return {
        "embedding": embedding_name,
        "EER": eer,
        "n_speakers": int(
            len(np.unique(labels))
        ),
        "n_positive_pairs": int(
            np.sum(targets == 1)
        ),
        "n_negative_pairs": int(
            np.sum(targets == 0)
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/e_daic.yaml",
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

    manifest = pd.read_csv(
        manifest_path
    )

    feature_dir = Path(
        cfg["features"]["output_dir"]
    )

    results_dir = Path(
        cfg["output"]["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for name in [
        "ecapa",
        "xvector",
        "dvector",
    ]:

        path = (
            feature_dir
            / f"{name}_embeddings.npy"
        )

        if not path.exists():
            print(
                f"Skipping {name}: "
                "embedding file not found."
            )
            continue

        embeddings = np.load(
            path
        )

        result = (
            speaker_verification_eer(
                embeddings,
                manifest,
                name,
            )
        )

        results.append(
            result
        )

        print(
            f"{name}: "
            f"EER={result['EER']:.4f}"
        )

    output_path = (
        results_dir
        / "speaker_classification_eer.json"
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
        f"\nSaved results to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()