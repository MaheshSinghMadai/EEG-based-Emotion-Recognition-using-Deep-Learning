# EEG Emotion-Recognition Benchmark

Code for the thesis *A Unified Benchmark of EEG Emotion Recognition Algorithms Under Strict
Subject- and Trial-Disjoint Evaluation, with Cross-Dataset Generalization*. Five published
models (DADPc, CIT-EmotionNet, STGATE, DAH-Net, CLISA) are evaluated on DEAP and DREAMER under one
harmonized preprocessing pipeline, frozen hash-stamped splits and automated leakage assertions.

**Plan, design decisions, and audit of earlier work:** [docs/THESIS_PLAN.md](docs/THESIS_PLAN.md)

## Layout

```
eeg-emotion-benchmark/
├── data/
│   ├── raw/DEAP/sXX.dat, raw/DREAMER.mat
│   ├── processed/{DEAP,DREAMER}/      # per-subject .npz (harmonized-v2)
│   └── splits/{DEAP,DREAMER}/         # frozen manifests: loso, subject_kfold, trial_kfold,
│                                      #   random_window, transfer_source
├── docs/
│   ├── THESIS_PLAN.md
│   └── paper_reported.json            # headline numbers from the papers (for table T3)
├── src/
│   ├── preprocessing.py               # harmonized DEAP/DREAMER pipeline (+ writes manifests)
│   ├── splits.py                      # protocol manifests, SHA-256 verification
│   ├── leakage.py                     # per-fold leakage assertions
│   ├── data_loader.py                 # manifest-driven, leakage-checked loaders
│   ├── training.py                    # per-paper recipes + shared stopping rule
│   ├── benchmark_proposal.py          # runs protocols, writes results/proposal/**
│   ├── analysis.py                    # thesis tables T1–T5 + leakage figure
│   ├── visualize_proposal.py          # comparison plots
│   └── models/proposal/               # the five paper-faithful models
├── tests/test_proposal_pipeline.py
└── results/proposal/                  # result JSONs, predictions, comparisons, analysis
```

`src/benchmark_si.py` and the other modules in `src/models/` are the earlier DREAMER-only model
zoo from before the proposal. They are not part of the thesis benchmark.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.preprocessing --dataset DEAP        # also writes data/splits/DEAP/*.json
python -m src.preprocessing --dataset DREAMER
python -m unittest tests.test_proposal_pipeline

# smoke test: one fold, two epochs
python -m src.benchmark_proposal --protocol loso --dataset DEAP --max-folds 1 --max-epochs 2

# full runs
python -m src.benchmark_proposal --protocol loso          --dataset DREAMER --seeds 42 43 44
python -m src.benchmark_proposal --protocol leakage_suite --dataset DREAMER --seeds 42 43 44
python -m src.benchmark_proposal --protocol transfer --source DEAP --target DREAMER --seeds 42 43 44

python -m src.analysis --label valence             # results/proposal/analysis/valence/tables.md
```

## Protocol guarantees

| Protocol | Subject-disjoint | Trial-disjoint | Purpose |
|---|:---:|:---:|---|
| `loso` | ✓ | ✓ | headline strict result |
| `subject_kfold` | ✓ | ✓ | strict, same budget as the ablations |
| `trial_kfold` | ✗ | ✓ | measures subject leakage |
| `random_window` | ✗ | ✗ | measures trial + subject leakage (common in the literature) |
| `transfer` | ✓ | ✓ | cross-dataset generalization |

Every fold is checked before training: the guarantees the protocol promises, window-level
disjointness, and normalization fitted on training rows only. A violation stops the run.
Each result file records the manifest SHA-256 and the per-fold leakage report. DADPc receives
the test inputs **without labels** (transductive adaptation, as in the paper);
`--dadpc-adaptation none` gives the inductive variant.

## Harmonization (summary)

14 shared electrodes selected by name (the DEAP channel map was corrected; see the plan, §2),
128 Hz, baseline mean removal, common average reference over the 14 channels, 4–45 Hz filter,
4 bands (θ, α, β, γ ≤ 45 Hz), 4 s non-overlapping windows, and label = rating > scale midpoint
(5 on DEAP, 3 on DREAMER). DEAP and DREAMER differ in label prior (52.7% vs 41.2% high valence),
so macro-F1 and the majority-class baseline are reported alongside accuracy.
