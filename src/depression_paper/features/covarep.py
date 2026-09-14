"""Extract COVAREP acoustic features and segment-level statistics."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from depression_paper.data.preprocess import _participant_intervals


COVAREP_FEATURES = 74
COVAREP_FRAME_RATE = 100.1


def compute_covarep_statistics(
    frame_feats: np.ndarray,
) -> np.ndarray:
    """
    Compute six statistics over the 74 COVAREP frame-level
    features, producing 74 * 6 = 444 dimensions.

    Statistics:
        mean
        max
        min
        standard deviation
        skewness
        kurtosis
    """

    frame_feats = np.asarray(
        frame_feats,
        dtype=np.float32,
    )

    if frame_feats.ndim != 2:
        raise ValueError(
            "COVAREP frame features must be a 2-D array."
        )

    if frame_feats.shape[1] != COVAREP_FEATURES:
        raise ValueError(
            f"Expected {COVAREP_FEATURES} COVAREP features, "
            f"got {frame_feats.shape[1]}."
        )

    frame_feats = np.nan_to_num(
        frame_feats,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if frame_feats.shape[0] == 0:
        return np.zeros(
            COVAREP_FEATURES * 6,
            dtype=np.float32,
        )

    mean_v = np.mean(
        frame_feats,
        axis=0,
    )

    max_v = np.max(
        frame_feats,
        axis=0,
    )

    min_v = np.min(
        frame_feats,
        axis=0,
    )

    std_v = np.std(
        frame_feats,
        axis=0,
    )

    skew_v = stats.skew(
        frame_feats,
        axis=0,
        nan_policy="omit",
    )

    kurt_v = stats.kurtosis(
        frame_feats,
        axis=0,
        nan_policy="omit",
    )

    stat_vector = np.concatenate(
        [
            mean_v,
            max_v,
            min_v,
            std_v,
            skew_v,
            kurt_v,
        ],
        axis=0,
    )

    return np.nan_to_num(
        stat_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _load_covarep(
    audio_path: Path,
) -> np.ndarray:
    """
    Run the external COVAREP executable on an audio file.

    The COVAREP MATLAB toolbox is not distributed with E-DAIC,
    so this function expects a locally installed COVAREP
    executable/script.

    Set the environment variable COVAREP_COMMAND to the command
    used by the local installation.

    The command must accept:
        input_wav output_csv

    and produce a CSV containing 74 frame-level features.
    """

    import os

    command = os.environ.get(
        "COVAREP_COMMAND"
    )

    if not command:
        raise RuntimeError(
            "COVAREP_COMMAND is not set. "
            "E-DAIC does not contain COVAREP features, so "
            "COVAREP must be extracted separately before the "
            "COVAREP experiment can be run."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = (
            Path(tmpdir) / "covarep.csv"
        )

        command_parts = command.split()

        command_parts.extend(
            [
                str(audio_path),
                str(output_csv),
            ]
        )

        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "COVAREP extraction failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        if not output_csv.exists():
            raise FileNotFoundError(
                "COVAREP command completed but did not "
                f"produce {output_csv}."
            )

        frames = pd.read_csv(
            output_csv,
            header=None,
        ).to_numpy(
            dtype=np.float32
        )

    if frames.ndim != 2:
        raise ValueError(
            "COVAREP output must be a 2-D array."
        )

    if frames.shape[1] != COVAREP_FEATURES:
        raise ValueError(
            f"Expected {COVAREP_FEATURES} COVAREP "
            f"features, got {frames.shape[1]}."
        )

    return frames


def _extract_segment_frames(
    frames: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Extract COVAREP frames corresponding to a segment."""

    start_frame = max(
        0,
        int(
            round(
                start_sec
                * COVAREP_FRAME_RATE
            )
        ),
    )

    end_frame = min(
        len(frames),
        int(
            round(
                end_sec
                * COVAREP_FRAME_RATE
            )
        ),
    )

    if end_frame <= start_frame:
        return np.empty(
            (
                0,
                COVAREP_FEATURES,
            ),
            dtype=np.float32,
        )

    return frames[
        start_frame:end_frame
    ]


def extract_segment_covarep(
    manifest: pd.DataFrame,
    dataset_root: Path,
    output_path: Path,
) -> np.ndarray:
    """
    Extract 444-dimensional COVAREP statistics for every
    segment in the manifest.

    The dataset root is retained in the API for consistency
    with the other feature extractors. Segment audio paths
    come directly from the manifest.
    """

    del dataset_root

    stats_list: list[np.ndarray] = []

    participant_cache: dict[
        int,
        np.ndarray,
    ] = {}

    print(
        "Computing per-segment "
        "COVAREP 444-dim statistics..."
    )

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="COVAREP",
    ):
        participant_id = int(
            row["participant_id"]
        )

        wav_path = Path(
            row["wav_path"]
        )

        # COVAREP is extracted from the already-generated
        # fixed-duration segment. Therefore no transcript
        # filtering is needed here.
        #
        # Cache is intentionally not used for segment-level
        # extraction because each segment has its own audio.
        del participant_id

        frames = _load_covarep(
            wav_path
        )

        feat_vec = compute_covarep_statistics(
            frames
        )

        stats_list.append(
            feat_vec
        )

    if stats_list:
        stats_arr = np.stack(
            stats_list,
            axis=0,
        ).astype(np.float32)
    else:
        stats_arr = np.empty(
            (
                0,
                COVAREP_FEATURES * 6,
            ),
            dtype=np.float32,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        stats_arr,
    )

    print(
        f"COVAREP shape: "
        f"{stats_arr.shape}"
    )

    return stats_arr