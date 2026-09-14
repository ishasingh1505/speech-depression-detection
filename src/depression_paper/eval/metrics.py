"""Evaluation metrics for depression detection and severity estimation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
)


def depression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> dict[str, float]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "detection":

        f1_depressed = f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        f1_healthy = f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        return {
            "F1(D)": float(f1_depressed),
            "F1(H)": float(f1_healthy),
            "BAc": float(balanced_accuracy),
        }

    if task == "severity":

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        return {
            "RMSE": float(rmse),
        }

    raise ValueError(
        f"Unknown task: {task}"
    )


def phq8_severity_bins(
    scores: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(scores)

    bins = np.zeros(
        scores.shape,
        dtype=int,
    )

    bins[
        (scores >= 5) & (scores <= 9)
    ] = 1

    bins[
        (scores >= 10) & (scores <= 14)
    ] = 2

    bins[
        scores >= 15
    ] = 3

    return bins


def severity_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:

    true_bins = phq8_severity_bins(
        np.asarray(y_true)
    )

    pred_bins = phq8_severity_bins(
        np.asarray(y_pred)
    )

    return confusion_matrix(
        true_bins,
        pred_bins,
        labels=[0, 1, 2, 3],
    )


def aggregate_participant_predictions(
    predictions,
    participant_ids,
    task: str,
):
    predictions = np.asarray(predictions)
    participant_ids = np.asarray(participant_ids)

    unique_pids = np.unique(
        participant_ids
    )

    results = {}

    for pid in unique_pids:

        values = predictions[
            participant_ids == pid
        ]

        if task == "detection":

            count_healthy = np.sum(
                values == 0
            )

            count_depressed = np.sum(
                values == 1
            )

            results[int(pid)] = int(
                count_depressed > count_healthy
            )

        elif task == "severity":

            results[int(pid)] = float(
                np.mean(values)
            )

        else:
            raise ValueError(
                f"Unknown task: {task}"
            )

    return results