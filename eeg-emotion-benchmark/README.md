# EEG Emotion-Recognition Benchmark (DREAMER)

Comparative benchmarking pipeline for a Master's thesis: 8 deep-learning
architectures for EEG-based emotion recognition, each evaluated on the DREAMER
dataset under **two validation protocols** (subject-dependent and
leave-one-subject-out), with a consolidated results table and a
generalization-gap analysis.

## Layout

```
eeg-emotion-benchmark/
├── data/
│   ├── raw/DREAMER.mat        # <- put the dataset here
│   ├── processed/             # per-subject .npz (produced by preprocessing)
│   └── splits/                # subject_dependent_splits.json, loso_splits.json
├── src/
│   ├── preprocessing.py       # DREAMER.mat -> windows + DE features + splits
│   ├── data_loader.py         # PyTorch Datasets/DataLoaders (raw & DE views)
│   ├── evaluation.py          # shared metrics + protocol runners + results IO
│   └── models/                # one file per architecture
├── notebooks/                 # 01–08 model notebooks + 09 aggregation
├── results/                   # {model}_results.json + comparison table + figures
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. preprocess (defaults: 4 s windows, band-pass 0.3–50 Hz, valence @ threshold 3)
cp /path/to/DREAMER.mat data/raw/DREAMER.mat
python -m src.preprocessing --raw data/raw/DREAMER.mat --out data/processed --splits data/splits

# 2. run notebooks 01–08 (each writes results/{model}_results.json)
#    set QUICK = True in a notebook first for a 3-subject / 3-fold smoke test
jupyter lab

# 3. run notebook 09 to build results/benchmark_comparison.{csv,md} + figures/
```

## The 8 models

| # | Notebook | Model | Family | Input |
|---|----------|-------|--------|-------|
| 1 | `01_EEGNet` | EEGNet-8,2 | CNN | raw `[14×T]` |
| 2 | `02_DGCNN` | Dynamical Graph CNN | Graph CNN | DE `[14×5]` |
| 3 | `03_SiamGCAN` | Siamese Graph Conv-Attention | Graph CNN + Siamese | DE pairs |
| 4 | `04_CIT_EmotionNet` | CNN-Transformer Interactive | CNN-Transformer | DE topomap |
| 5 | `05_SHAP_Diagnostic` | KNN / RF / LDA / CNN + SHAP | Diagnostic | DE flat `[70]` |
| 6 | `06_CNN_Transformer_ReducedElectrode` | CNN-Transformer, 5 electrodes | CNN-Transformer | raw `[5×T]` |
| 7 | `07_DADPc` | Domain-Adaptive Deep Possibilistic Clustering | Domain adaptation | DE `[14×5]` |
| 8 | `08_DualAttention_TransformerCNNBiLSTM` | Dual-attention CNN-BiLSTM-Transformer | Hybrid | raw `[14×T]` |

Every model is trained fresh per fold and scored through
`src/evaluation.py`, so the rows of `results/benchmark_comparison.md` are directly
comparable.
