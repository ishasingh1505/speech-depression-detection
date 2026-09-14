# Reproducing: Manifestation of depression in speech overlaps with speaker identity

Implementation of [Dumpala et al. (2023)](https://doi.org/10.1038/s41598-023-35184-7) for **DAIC-WOZ only**.

> **Note:** VocalMind (private clinical dataset) and longitudinal analysis (Section: Effect of depression on speaker embeddings) are **skipped** as that data is not publicly available.

## Paper vs. this reproduction

| Paper component | Status |
|-----------------|--------|
| DAIC-WOZ preprocessing (participant speech, 5s segments) | Implemented |
| Speaker embeddings: x-vector, ECAPA-TDNN, d-vector | Implemented via [SpeechBrain](https://speechbrain.github.io/) pretrained models |
| OpenSMILE IS09 (384-dim) | Implemented |
| COVAREP segment statistics (mean/max/min/std/skew/kurtosis) | Implemented |
| MK-CNN, LSTM, DNN, Combined Embeddings (CE) | Implemented |
| 5-fold speaker-independent CV | Implemented |
| Metrics: F1(D), F1(H), BAc, RMSE | Implemented |
| Speaker classification EER | Implemented |
| Gender-specific analysis | Implemented |
| PHQ-8 severity confusion matrix (Fig. 5) | Implemented |
| VocalMind experiments | Skipped (private) |
| Longitudinal MADRS analysis (Fig. 4) | Skipped (VocalMind only) |
| Comparison with Mockingjay/wav2vec2/TRILL (Table 4) | Optional (heavy; not included by default) |
| d-vector GE2E pre-training from scratch | Uses SpeechBrain `spkrec-resnet-voxceleb` instead |

### Expected DAIC-WOZ results (Table 2, LSTM)

From the paper (219 participants; you have **142 labeled** train+dev participants):

| Model | F1(D) | F1(H) | BAc | RMSE |
|-------|-------|-------|-----|------|
| OpenSMILE alone | 0.39 | 0.73 | 0.56 | 6.82 |
| ECAPA alone | 0.46 | 0.79 | 0.63 | 6.31 |
| ECAPA + OpenSMILE | **0.50** | **0.83** | **0.66** | **6.01** |

Your numbers may differ slightly due to dataset size (189 vs 219), CPU training, and pretrained model versions.

## Requirements

- Python 3.10+
- ~90 GB for DAIC-WOZ zips
- GPU recommended (CPU works but is slow)
- Network for first-run SpeechBrain model download

## Setup

```bash
cd /home/hp/depression-paper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Edit `config/daic_woz.yaml` if your DAIC-WOZ path differs from `/home/hp/DAIC-WOZ`.

## Pipeline

### Step 1 — Preprocess audio

Extracts **participant-only** speech from transcripts, segments into **5-second** non-overlapping chunks.

```bash
python scripts/01_preprocess.py
# smoke test:
python scripts/01_preprocess.py --max-participants 2
```

Outputs: `data/processed/segments_manifest.csv`, per-participant WAV segments.

### Step 2 — Extract features

```bash
python scripts/02_extract_features.py
# or one embedding at a time:
python scripts/02_extract_features.py --embedding ecapa
```

Outputs in `data/features/`:
- `ecapa_embeddings.npy`, `xvector_embeddings.npy`, `dvector_embeddings.npy`
- `opensmile_is09.npy`
- `covarep_segment_stats.npy`

### Step 3 — Train & evaluate (5-fold CV)

```bash
# Single experiment:
python scripts/03_run_experiments.py --feature ecapa_opensmile --model lstm --task detection

# All Table 2 experiments (detection + severity):
python scripts/03_run_experiments.py --run-all

# Quick smoke test (3 epochs):
python scripts/03_run_experiments.py --feature ecapa --model lstm --task detection --quick
```

Results saved to `results/`.

### Step 4 — Speaker classification (EER)

```bash
python scripts/04_speaker_classification.py
```

Paper reports EER ~1.10 (ECAPA) and ~1.29 (d-vector) on DAIC-WOZ.

### Step 5 — Gender analysis

```bash
python scripts/05_gender_analysis.py
```

### Step 6 — Confusion matrix (Fig. 5)

After saving participant-level severity predictions to CSV (`participant_id,y_true,y_pred`):

```bash
python scripts/06_confusion_matrix.py --predictions results/severity_predictions.csv
```

### Run everything

```bash
python run_pipeline.py          # full pipeline
python run_pipeline.py --quick  # 2 participants, smoke test
```

## Project structure

```
depression-paper/
├── config/daic_woz.yaml       # paths, hyperparameters (match paper)
├── scripts/                   # runnable pipeline steps
├── src/depression_paper/      # library code
│   ├── data/                  # preprocessing, folds, datasets
│   ├── features/              # embeddings, OpenSMILE, COVAREP
│   ├── models/                # MK-CNN, LSTM, DNN, CE
│   ├── train/                 # training loops
│   └── eval/                  # metrics
├── data/processed/            # generated segments
├── data/features/             # generated features
├── results/                   # experiment outputs
└── checkpoints/               # pretrained SpeechBrain models
```

## Hyperparameters (from paper)

- Temporal context: **16** contiguous segments (DAIC-WOZ)
- Adam: lr=0.0005, β1=0.9, β2=0.99
- Dropout: CNN 0.3, LSTM 0.4, FC 0.3
- Batch size: 128, Epochs: 50
- PHQ-8 ≥ 10 → depressed
- 5-fold CV, no speaker overlap, stratified by depression label

## Citation

```bibtex
@article{dumpala2023manifestation,
  title={Manifestation of depression in speech overlaps with characteristics used to represent and recognize speaker identity},
  author={Dumpala, Sri Harsha and Dikaios, Katerina and Rodriguez, Sebastian and others},
  journal={Scientific Reports},
  volume={13},
  number={11155},
  year={2023}
}
```

