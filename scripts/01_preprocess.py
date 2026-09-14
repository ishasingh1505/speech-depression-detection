#!/usr/bin/env python3

"""Extract participant speech and create fixed-duration segments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "src"
    ),
)


from depression_paper.config import load_config

from depression_paper.data.metadata import (
    load_participants,
    participants_to_dataframe,
)

from depression_paper.data.preprocess import (
    run_preprocessing,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess E-DAIC for "
            "depression paper reproduction"
        )
    )

    parser.add_argument(
        "--config",
        default="config/e_daic.yaml",
    )

    parser.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Limit participants for testing",
    )

    args = parser.parse_args()

    root = Path(
        __file__
    ).resolve().parents[1]

    cfg = load_config(
        root / args.config
    )

    records = load_participants(
        cfg["dataset"]["root"],
        splits=cfg["dataset"]["use_splits"],
        phq8_threshold=cfg["dataset"][
            "phq8_depression_threshold"
        ],
    )

    if args.max_participants is not None:
        records = records[
            : args.max_participants
        ]

    output_dir = Path(
        cfg["preprocessing"]["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    participants_df = participants_to_dataframe(
        records
    )

    participants_df.to_csv(
        output_dir / "participants.csv",
        index=False,
    )

    print(
        f"Participants loaded: "
        f"{len(records)}"
    )

    manifest = run_preprocessing(
        records=records,
        output_dir=output_dir,
        segment_duration=cfg["dataset"][
            "segment_duration_sec"
        ],
        target_sr=cfg["dataset"][
            "sample_rate"
        ],
        n_folds=cfg["training"]["n_folds"],
        random_seed=cfg["training"]["random_seed"],
    )

    if len(manifest) == 0:
        print(
            "No segments were generated."
        )
        return

    print(
        f"Done: "
        f"{len(manifest)} segments from "
        f"{manifest['participant_id'].nunique()} "
        f"participants"
    )


if __name__ == "__main__":
    main()