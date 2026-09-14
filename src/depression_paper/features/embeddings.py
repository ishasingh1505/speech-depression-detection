"""Extract pretrained speaker embeddings using SpeechBrain."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm


EMBEDDING_MODELS = {
    "ecapa": {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "dim": 192,
    },
    "xvector": {
        "source": "speechbrain/spkrec-xvect-voxceleb",
        "dim": 512,
    },
}


def _load_classifier(
    model_name: str,
    device: str,
):
    from speechbrain.inference.speaker import (
        EncoderClassifier,
    )

    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding model: "
            f"{model_name}. "
            f"Available models: "
            f"{list(EMBEDDING_MODELS)}"
        )

    info = EMBEDDING_MODELS[
        model_name
    ]

    checkpoint_dir = (
        Path("checkpoints")
        / "pretrained"
        / model_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier = EncoderClassifier.from_hparams(
        source=info["source"],
        savedir=str(checkpoint_dir),
        run_opts={
            "device": device
        },
    )

    return classifier


def _load_audio(
    wav_path: str,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Load mono audio and resample to 16 kHz."""

    audio, sr = sf.read(
        wav_path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = audio.mean(
            axis=1
        )

    if sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sr,
        )

    signal = torch.from_numpy(
        audio.astype(np.float32)
    ).unsqueeze(0)

    return signal


@torch.no_grad()
def extract_embedding(
    classifier,
    wav_path: str,
    device: str,
) -> np.ndarray:
    """Extract one speaker embedding."""

    signal = _load_audio(
        wav_path
    ).to(device)

    embedding = classifier.encode_batch(
        signal
    )

    embedding = (
        embedding
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return embedding.astype(
        np.float32
    )


def extract_embeddings_for_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
    model_name: str,
    device: str = "cpu",
    batch_save_every: int = 500,
) -> np.ndarray:
    """
    Extract one pretrained speaker embedding per
    segment in the manifest.
    """

    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model: "
            f"{model_name}"
        )

    if len(manifest) == 0:
        dim = EMBEDDING_MODELS[
            model_name
        ]["dim"]

        embeddings = np.empty(
            (0, dim),
            dtype=np.float32,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            embeddings,
        )

        return embeddings

    classifier = _load_classifier(
        model_name,
        device,
    )

    wav_col = (
        "segment_path"
        if "segment_path" in manifest.columns
        else "wav_path"
    )

    first_embedding = extract_embedding(
        classifier,
        str(manifest.iloc[0][wav_col]),
        device,
    )

    actual_dim = (
        first_embedding.shape[0]
    )

    expected_dim = EMBEDDING_MODELS[
        model_name
    ]["dim"]

    # We use the dimension actually produced by the
    # pretrained model. Do not pad/truncate embeddings
    # merely to force the paper's reported dimension.
    if actual_dim != expected_dim:
        print(
            f"Warning: {model_name} model produced "
            f"{actual_dim}-D embeddings; "
            f"configured standard dimension is "
            f"{expected_dim}. Using {actual_dim}-D."
        )

    n = len(manifest)

    embeddings = np.zeros(
        (
            n,
            actual_dim,
        ),
        dtype=np.float32,
    )

    embeddings[0] = first_embedding

    for i in tqdm(
        range(1, n),
        desc=f"Embeddings ({model_name})",
    ):
        wav_path = manifest.iloc[
            i
        ][wav_col]

        embedding = extract_embedding(
            classifier,
            str(wav_path),
            device,
        )

        if embedding.shape[0] != actual_dim:
            raise ValueError(
                f"Inconsistent embedding dimension "
                f"at manifest row {i}: "
                f"expected {actual_dim}, "
                f"got {embedding.shape[0]}."
            )

        embeddings[i] = embedding

        if (
            (i + 1) % batch_save_every
            == 0
        ):
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                output_path,
                embeddings,
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        embeddings,
    )

    print(
        f"{model_name} embedding shape: "
        f"{embeddings.shape}"
    )

    return embeddings