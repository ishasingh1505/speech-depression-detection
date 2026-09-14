#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from depression_paper.config import load_config

from depression_paper.eval.metrics import (
    phq8_severity_bins,
    severity_confusion_matrix,
)


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
):

    labels = [
        "None",
        "Mild",
        "Moderate",
        "Severe",
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
    )

    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    for row in range(4):

        for column in range(4):

            ax.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def majority_baseline_cm(
    y_true: np.ndarray,
) -> np.ndarray:

    true_bins = phq8_severity_bins(
        y_true
    )

    predicted_bins = np.zeros(
        len(true_bins),
        dtype=int,
    )

    return severity_confusion_matrix(
        true_bins,
        predicted_bins,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/e_daic.yaml",
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "CSV containing "
            "participant_id, y_true, y_pred"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT / args.config
    )

    predictions_path = Path(
        args.predictions
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{predictions_path}"
        )

    df = pd.read_csv(
        predictions_path
    )

    required_columns = [
        "participant_id",
        "y_true",
        "y_pred",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    y_true = df[
        "y_true"
    ].to_numpy()

    y_pred = df[
        "y_pred"
    ].to_numpy()

    model_cm = severity_confusion_matrix(
        y_true,
        y_pred,
    )

    baseline_cm = majority_baseline_cm(
        y_true
    )

    results_dir = Path(
        cfg["output"]["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        results_dir
        / "confusion_matrix_model.png"
    )

    baseline_path = (
        results_dir
        / "confusion_matrix_baseline.png"
    )

    plot_confusion_matrix(
        model_cm,
        "Model Severity Confusion Matrix",
        model_path,
    )

    plot_confusion_matrix(
        baseline_cm,
        "Majority Baseline",
        baseline_path,
    )

    results = {
        "model_cm": model_cm.tolist(),
        "baseline_cm": baseline_cm.tolist(),
    }

    output_path = (
        results_dir
        / "confusion_matrix.json"
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
        "Saved confusion matrices."
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Baseline: {baseline_path}"
    )


if __name__ == "__main__":
    main()