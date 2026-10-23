# Thesis plan — Unified, leakage-free benchmark of EEG emotion recognition

**Title:** A Unified Benchmark of EEG Emotion Recognition Algorithms Under Strict Subject- and
Trial-Disjoint Evaluation, with Cross-Dataset Generalization

**Current scope:** DEAP and DREAMER. SEED is added later through the same preprocessing and
split interfaces (see §8).

---

## 1. Research question and how each part is answered

> When five recent, architecturally distinct EEG emotion recognition models are forced onto one
> identical, strict subject-disjoint and trial-disjoint evaluation protocol, does their relative
> ranking change from what their papers report, and how well does each generalize across datasets?

| Proposal objective | Answered by | Output |
|---|---|---|
| O1 Re-implement five architectures faithfully; document deviations | `src/models/proposal/*.py` (deviations in each module docstring) | Methods chapter, fidelity table (§4) |
| O2 One leakage-free pipeline: frozen splits + automated leakage assertions | `src/splits.py`, `src/leakage.py`, `src/data_loader.py` | Every result JSON carries the manifest SHA-256 and a per-fold leakage report |
| O3 Within-dataset and cross-dataset evaluation, one channel/label harmonization | `src/preprocessing.py`, `benchmark_proposal --protocol loso / transfer` | Tables T1, T5 |
| O4 How protocol strictness shifts rankings; which paradigm generalizes best | `--protocol leakage_suite`, `src/analysis.py` | Tables T2–T4, leakage-inflation figure |
| Contribution 2: how much leakage inflates accuracy | `random_window − trial_kfold − subject_kfold` decomposition | Table T2 |

## 2. Audit of the existing work (what was found)

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **DEAP channel map was wrong.** Indices 18–31 omitted Fz and Cz, so F4, F8, FC6, T8, P8 and O2 were read from neighbouring electrodes (Fz, F4, F8, Cz, CP2, P8). Verified empirically: with the official order each electrode correlates most with its scalp neighbours. | Critical — all earlier DEAP and cross-dataset numbers are invalid | Fixed; regression test (`test_deap_channel_indices`, plus a data-driven neighbour test) |
| 2 | DEAP's release is already band-passed at 4–45 Hz, but DREAMER is not; the delta band (1–4 Hz) was empty on DEAP and populated on DREAMER. | High for cross-dataset | Harmonized 4–45 Hz filter, four bands (θ, α, β, γ≤45 Hz) for both |
| 3 | Different references (DEAP: 32-channel CAR, DREAMER: mastoid). | Medium | Common average reference over the 14 shared channels for both |
| 4 | No trial-disjoint protocol, no leaky comparison protocol, no automated leakage assertions, splits not hash-frozen. | High — objectives O2/O4 not addressable | Frozen manifests for 5 protocols + per-fold assertions |
| 5 | DADPc trained with plain cross-entropy (no reconstruction, no clustering, no target data). | High — not DADPc | Re-implemented from the paper (AE + adaptive loss + possibilistic clustering + memory-bank centroids) |
| 6 | CLISA used generic noise-augmentation pairs, not stimulus-aligned inter-subject pairs. | High — not CLISA | Stimulus-aligned positives from two different training subjects; frozen-encoder DE features + MLP |
| 7 | CIT-EmotionNet used an 8×9 map and a single cross-attention fusion. | Medium | 32×32 map, ResNet + ViT branches, L2G/G2L CIT modules after stages 1–3 |
| 8 | STGATE / DAH-Net were "inspired-by" baselines. | Medium | Rebuilt to the published structure; STGATE re-checked against the full PDF (multi-kernel conv TLB, input-dependent spatial attention with top-10 edges, ASTGCN temporal attention, paper recipe) |
| 12 | Fidelity pass against the full PDFs: CLISA base encoder used a depthwise temporal conv (16 maps instead of 16×16), random stimulus keys (several windows of one trial as negatives), no stratified normalization, non-paper recipe; CIT-EmotionNet Stage 0 kept full resolution instead of Conv7×7/s2 + MaxPool (Eq. 1); DAH-Net used GELU instead of ReLU | High (CLISA), Medium (CIT) | All fixed; existing CLISA, STGATE and CIT-EmotionNet results predate the fixes and must be re-run |
| 9 | Only 12 of 32 DEAP subjects are in `data/raw/DEAP`. | High for final results | **Action for you:** download the remaining 20 `sXX.dat` files |
| 10 | No majority-class floor; DEAP and DREAMER have different label priors (≈52% vs ≈41% high valence), so cross-dataset accuracy alone is misleading. | Medium | Majority baseline in every table; macro-F1 reported alongside accuracy |
| 11 | Paper protocols are themselves leaky: CIT-EmotionNet 98.57% uses a random 6:4 window split; DAH-Net 99.19% a random 80/20 split on a non-DEAP/DREAMER feature set. | Finding | Becomes a core result of Table T3 |

## 3. Experimental design

### 3.1 Data harmonization (documented, reusable protocol — Contribution 3)

| Step | DEAP | DREAMER | Harmonized choice |
|---|---|---|---|
| Channels | 32 BioSemi | 14 Emotiv EPOC | 14 shared electrodes, selected by name |
| Sampling | 128 Hz | 128 Hz | 128 Hz |
| Baseline | first 3 s of each trial | separate 61 s baseline | subtract per-trial baseline mean |
| Reference | CAR (32 ch) | mastoid | CAR over the 14 shared channels |
| Filter | 4–45 Hz already | raw | 4–45 Hz zero-phase Butterworth |
| Bands | — | — | θ 4–8, α 8–14, β 14–31, γ 31–45 Hz |
| Windows | 60 s stimulus | 65–393 s clips | 4 s non-overlapping, never crossing trials |
| Labels | 1–9 | 1–5 | high ⇔ rating > midpoint (5 / 3) |

Feature views written once per window: filtered signal, DE (4 s), DE sequence (4 × 1 s),
and a 273-d engineered vector (DAH-Net). Label-scheme sensitivity (e.g. per-subject median)
can be analysed later from the stored raw ratings without re-preprocessing.

### 3.2 Protocols (frozen, SHA-256-stamped manifests in `data/splits/<DATASET>/`)

| Protocol | Subject-disjoint | Trial-disjoint | Role |
|---|:---:|:---:|---|
| `loso` | ✓ | ✓ | **Headline strict result** (DEAP 12→32 folds, DREAMER 23 folds) |
| `subject_kfold` (5-fold) | ✓ | ✓ | Strict reference with the same 80/20 budget as the ablations |
| `trial_kfold` (5-fold) | ✗ | ✓ | Isolates *subject* leakage |
| `random_window` (5-fold) | ✗ | ✗ | Typical literature split; adds *trial* leakage |
| `transfer` | ✓ | ✓ | Train on dataset A (source validation subjects), test on all of B |

Validation data (early stopping) always comes from the training side, using the same
grouping unit as the protocol. Normalization is fitted on training rows only (asserted).

Leakage inflation is decomposed as
`Δtrial = acc(random_window) − acc(trial_kfold)`, `Δsubject = acc(trial_kfold) − acc(subject_kfold)`.

### 3.3 Training fairness

- Each model keeps its paper's optimiser, learning rate, batch size, scheduler and augmentation.
- All models share one stopping rule: ≤100 epochs, best validation cross-entropy, patience 15.
- Seeds: 42, 43, 44 for every final number (report mean over subjects; seed spread reported separately).

### 3.4 Metrics and statistics

- Unit of analysis: test subject. Each window is predicted exactly once per seed (asserted);
  accuracy and macro-F1 are computed per subject, then mean ± std over subjects.
- Strict protocol: Friedman test across models over subjects, pairwise Wilcoxon signed-rank
  with Holm correction (T4).
- Leakage: paired Wilcoxon over subjects, random_window vs subject_kfold (T2).
- Ranking shift: Kendall τ between leaky and strict rankings; side-by-side with paper claims (T3).
- Generalization gap: target-dataset LOSO accuracy − transfer accuracy (T5).
- Always shown with the majority-class floor.

## 4. Models and fidelity

| Model (year) | Paradigm | Input | Reproduced | Main documented deviations | Paper protocol |
|---|---|---|---|---|---|
| DADPc (2025) | Domain adaptation | DE [14×4] | AE, adaptive σ-loss, sharpened CE, possibilistic clustering with fuzzy entropy, memory-bank centroids, unlabeled target | per-epoch (not per-batch) bank refresh; λ3 fixed; 4 s windows, binary labels | LOSO, transductive — SEED 87.83% |
| CIT-EmotionNet (2023/24) | CNN–Transformer fusion | DE → 32×32×4 map | ResNet stages with Stage 0 Conv7×7/s2 + MaxPool, ViT (1 token/band, 3 blocks/stage), L2G/G2L after stages 1–3, Adam 1e-5, batch 32 | DE instead of PSD (affine-equivalent after z-scoring); 14 channels interpolated | random 6:4 windows — SEED 98.57% |
| STGATE (2023) | Graph attention + Transformer | DE seq [14×4×4] | TLB (1/3/3×3 multi-kernel conv + Transformer), input-dependent spatial attention (Eq. 9–11, BN, top-10, kNN-initialised), temporal attention, multi-head GAT, Adam 1e-5, batch 16, dropout 0.3 | temporal attention in ASTGCN form (printed Eq. 12 is dimensionally N×N); A enters GAT logits as a masked bias so it is trainable; unreported widths/heads chosen by us | LOSO, held-out subject also used as validation — DREAMER valence 77.44% |
| DAH-Net (2026) | CNN–BiLSTM–dual attention | 273-d engineered | 4 conv + residual, 2×BiLSTM(256), 16+8-head attention, max+avg pool, AdamW recipe, augmentation | 273-d recomputed features instead of the 988-d Kaggle set; shared epoch cap | random 80/20 — 99.19% |
| CLISA (2022) | Contrastive self-supervision | filtered signal [14×512] | per-trial stimulus-aligned sampler over all subject pairs, 16 spatial × 16 temporal base encoder, pooled depthwise projector, stratified normalization, NT-Xent, Adam 7e-4 + cosine warm restarts; 256-d trained-DE + MLP (2×30), Adam 5e-4, batch 256 | kernels scaled to 128 Hz (P1=31, S=15, P2=3); 4 s samples; in-window DE averaging instead of LDS; train-set (not test-adaptive) feature normalization; τ=0.07 (not reported) | LOSO with test-adaptive normalization — SEED 86.4% |

DADPc uses unlabeled target-domain inputs. This is not label leakage (asserted: the loader
yields inputs only), but it is a different information regime, so DADPc is additionally
run with `--dadpc-adaptation none` (reported as **DADPc-inductive**) for a like-for-like comparison.

## 5. Work plan

Weeks are relative; align W-numbers with your submission deadline and supervisor meetings.

| Phase | Weeks | Tasks | Done when |
|---|---|---|---|
| 0. Data completion | W1 | Download remaining 20 DEAP subjects; re-run `python -m src.preprocessing --dataset DEAP` (PDFs and `docs/paper_reported.json`: done) | 32 DEAP files, manifests regenerated, tests green |
| 1. Fidelity check | W1–W2 | ~~Compare STGATE and CLISA modules against the PDFs and fix~~ (done, audit #12); sanity-check every model under `random_window` (should reach clearly-above-chance accuracy — a model that cannot even fit leaky splits has a bug) | All five beat the majority baseline on `random_window` |
| 2. Strict within-dataset | W2–W4 | `--protocol loso` on DEAP and DREAMER, seeds 42 43 44, valence then arousal | T1, T4 complete |
| 3. Leakage ablation | W4–W5 | `--protocol leakage_suite` on both datasets, 3 seeds | T2, leakage figure |
| 4. Cross-dataset | W5–W6 | `--protocol transfer` both directions, 3 seeds; DADPc transductive + inductive | T5 |
| 5. Analysis | W6–W7 | `python -m src.analysis`; ranking shift vs paper claims (T3); error analysis per subject | All tables final |
| 6. Writing | W7–W12 | Chapters (see §7); figures; reproducibility appendix (manifest hashes, configs) | Full draft to supervisor |
| 7. Buffer / SEED | W12–W14 | Revisions; optionally add SEED (§8) | Submission |

### Compute budget (measured on this Mac, MPS)

Smoke runs: CIT-EmotionNet ≈30 s/epoch per DEAP-LOSO fold (21 M parameters); DAH-Net ≈10 s;
DADPc, STGATE, CLISA a few seconds. With early stopping (≈30–50 epochs) a full 3-seed run of
CIT-EmotionNet is roughly a day on DEAP-32 and 2–3 days on DREAMER; everything else fits in
hours. **Run CIT-EmotionNet on a CUDA GPU (Colab / university cluster).** The code selects
CUDA automatically. Run the cheap models locally first.

## 6. Commands (experiment matrix)

```bash
cd eeg-emotion-benchmark
python -m src.preprocessing --dataset DEAP
python -m src.preprocessing --dataset DREAMER
python -m unittest tests.test_proposal_pipeline

# smoke test (minutes)
python -m src.benchmark_proposal --protocol loso --dataset DEAP --max-folds 1 --max-epochs 2

# final runs (repeat with --label arousal)
for D in DEAP DREAMER; do
  python -m src.benchmark_proposal --protocol loso          --dataset $D --seeds 42 43 44
  python -m src.benchmark_proposal --protocol leakage_suite --dataset $D --seeds 42 43 44
  python -m src.benchmark_proposal --protocol loso --dataset $D --models DADPc --dadpc-adaptation none --seeds 42 43 44
done
python -m src.benchmark_proposal --protocol transfer --source DEAP --target DREAMER --seeds 42 43 44
python -m src.benchmark_proposal --protocol transfer --source DREAMER --target DEAP --seeds 42 43 44

python -m src.analysis --label valence
python -m src.visualize_proposal --label valence
```

## 7. Thesis outline (results → chapters)

1. Introduction — motivation, RQ, contributions.
2. Background and related work — EEG emotion recognition; the five paradigms; evaluation
   protocols and leakage in the literature (use the paper-protocol column of §4).
3. Methodology — datasets; harmonization protocol (§3.1); protocols and leakage assertions
   (§3.2); model re-implementations and deviations (§4); training fairness (§3.3); statistics (§3.4).
4. Results — T1 strict results; T2 leakage inflation + figure; T3 ranking shift; T4 significance;
   T5 cross-dataset generalization.
5. Discussion — which paradigm survives strict evaluation and transfer; why leaky protocols
   reward subject/trial memorisation; label-prior shift between datasets; limitations.
6. Conclusion and future work — SEED, more labels, test-time adaptation.
Appendix — manifest SHA-256 table, full configs, per-subject results.

## 8. Adding SEED later

1. Add a `run_seed` driver in `src/preprocessing.py` that selects the 14 shared channels by name
   from SEED's 62-channel montage and calls `process_trials` (SEED is 200 Hz → resample to 128 Hz
   first). Map SEED's three classes to the shared binary valence scheme
   (positive = high, negative = low; neutral excluded or ablated).
2. Add `"SEED"` to `DATASETS` and `LABEL_MIDPOINTS`; manifests and all protocols then work unchanged,
   giving the six directional transfer pairs promised in the proposal.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Strict accuracies near chance make rankings unstable | 3 seeds, per-subject tests, majority floor, report macro-F1 and ties honestly — a null ranking difference is itself a finding |
| CIT-EmotionNet lr 1e-5 under-trains within 100 epochs | Report the paper recipe as primary; add a single sensitivity run (lr 1e-4) clearly labelled |
| Unreported architecture details (widths, heads, τ) | Chosen once, documented in each module docstring, never tuned on test subjects |
| DADPc transductive regime questioned | Report both transductive and inductive variants |
| Compute | GPU for CIT-EmotionNet; the other four run locally |
| Incomplete DEAP (12/32) | Phase 0; if impossible, state the cohort explicitly in every table |
