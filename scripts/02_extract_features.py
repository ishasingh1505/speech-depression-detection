#!/usr/bin/env python3

"""Extract pretrained embeddings and acoustic features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "src"
    ),
)


from depression_paper.config import load_config

from depression_paper.data.metadata import (
    load_participants,
)

from depression_paper.features.covarep import (
    extract_segment_covarep,
)

from depression_paper.features.embeddings import (
    EMBEDDING_MODELS,
    extract_embeddings_for_manifest,
)

from depression_paper.features.opensmile import (
    extract_opensmile_for_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for E-DAIC"
    )

    parser.add_argument(
        "--config",
        default="config/e_daic.yaml",
    )

    parser.add_argument(
        "--embedding",
        choices=list(
            EMBEDDING_MODELS.keys()
        ) + ["all"],
        default="ecapa",
    )

    parser.add_argument(
        "--skip-opensmile",
        action="store_true",
    )

    parser.add_argument(
        "--skip-covarep",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    root = Path(
        __file__
    ).resolve().parents[1]

    cfg = load_config(
        root / args.config
    )

    manifest_path = (
        Path(
            cfg["preprocessing"]["output_dir"]
        )
        / "segments_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}\n"
            "Run 01_preprocess.py first."
        )

    manifest = pd.read_csv(
        manifest_path
    )

    if len(manifest) == 0:
        raise ValueError(
            "The segment manifest is empty."
        )

    print(
        f"Loaded {len(manifest)} segments."
    )

    feature_dir = Path(
        cfg["features"]["output_dir"]
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        args.device
        or cfg["training"]["device"]
    )

    if args.embedding == "all":
        models = list(
            EMBEDDING_MODELS.keys()
        )
    else:
        models = [
            args.embedding
        ]

    for model_name in models:

        output_path = (
            feature_dir
            / f"{model_name}_embeddings.npy"
        )

        if output_path.exists():
            print(
                f"Skipping {model_name}: "
                f"{output_path} already exists."
            )
            continue

        print(
            f"\nExtracting {model_name}..."
        )

        extract_embeddings_for_manifest(
            manifest=manifest,
            output_path=output_path,
            model_name=model_name,
            device=device,
        )

    if not args.skip_opensmile:

        output_path = (
            feature_dir
            / "opensmile_is09.npy"
        )

        if output_path.exists():
            print(
                "Skipping OpenSMILE: "
                f"{output_path} already exists."
            )
        else:
            print(
                "\nExtracting OpenSMILE IS09..."
            )

            extract_opensmile_for_manifest(
                manifest,
                output_path,
            )

    if not args.skip_covarep:

        output_path = (
            feature_dir
            / "covarep_segment_stats.npy"
        )

        if output_path.exists():
            print(
                "Skipping COVAREP: "
                f"{output_path} already exists."
            )
        else:
            print(
                "\nExtracting COVAREP..."
            )

            records = load_participants(
                cfg["dataset"]["root"],
                cfg["dataset"]["use_splits"],
                cfg["dataset"][
                    "phq8_depression_threshold"
                ],
            )

            extract_segment_covarep(
                manifest,
                records,
                output_path,
            )

    print(
        "\nFeature extraction complete."
    )


if __name__ == "__main__":
    main()