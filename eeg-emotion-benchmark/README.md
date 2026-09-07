# EEG Emotion-Recognition Benchmark (DREAMER)

Comparative benchmarking pipeline for a Master's thesis: 8 deep-learning
architectures for EEG-based emotion recognition, each evaluated on the DREAMER
dataset under the **subject-independent (leave-one-subject-out) protocol**, with a
consolidated results table and an accuracy-vs-efficiency analysis.

## Layout

```
eeg-emotion-benchmark/
├── data/
│   ├── raw/DREAMER.mat        # <- put the dataset here
│   ├── processed/             # per-subject .npz (produced by preprocessing)
│   └── splits/                # loso_splits.json  (23-fold leave-one-subject-out)
├── src/
│   ├── preprocessing.py       # DREAMER.mat -> windows + DE features + LOSO split
│   ├── data_loader.py         # PyTorch Datasets/DataLoaders (raw & DE views)
│   ├── evaluation.py          # shared metrics + LOSO runner + results IO
│   ├── benchmark_si.py        # REGISTRY + run_one/run_all: the one shared run path
│   └── models/                # one file per architecture (arch + build(cfg))
├── notebooks/                 # 01–08 model notebooks + 10 the LOSO driver
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

# 2a. run every model + comparison table from the shell
python -m src.benchmark_si --quick                  # 3-fold / 5-epoch smoke test
python -m src.benchmark_si                          # full 23-fold run, every model

# 2b. …or run one model notebook at a time (01–08). Each is a thin wrapper that
#     calls `benchmark_si.run_one(MODEL_NAME, ...)` and writes
#     results/{model}_results.json. Notebook 10 drives the whole zoo + analysis.
jupyter lab
```

Every model's LOSO recipe (feature view, batch size, epochs, lr, any custom
training loop) lives once in `src/benchmark_si.py::REGISTRY`; `run_one` is the
single entry point shared by the notebooks and the CLI.

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

Every model is trained fresh per LOSO fold and scored through `src/evaluation.py`,
so the rows of `results/subject_independent_comparison.md` are directly comparable.
