"""Extract OpenSMILE IS09 functionals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from tqdm import tqdm


def create_smile_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS09,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_opensmile_for_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
) -> np.ndarray:

    smile = create_smile_extractor()

    features_list = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="OpenSMILE IS09",
    ):

        features = smile.process_file(
            row["wav_path"]
        )

        values = (
            features.values
            .flatten()
            .astype(np.float32)
        )

        features_list.append(
            values
        )

    features = np.stack(
        features_list,
        axis=0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        features,
    )

    print(
        f"OpenSMILE shape: "
        f"{features.shape}"
    )

    return features