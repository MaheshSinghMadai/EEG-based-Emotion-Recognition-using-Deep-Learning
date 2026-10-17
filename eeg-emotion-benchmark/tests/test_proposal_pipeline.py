import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data_loader import (
    UnlabeledDataset,
    build_fold_loaders,
    build_loso_loaders,
    build_transfer_fold_loaders,
    build_transfer_loaders,
)
from src.leakage import LeakageError, assert_unlabeled, check_partitions
from src.models.proposal import cit_emotionnet, clisa, dadpc, dah_net, stgate
from src.preprocessing import (
    COMMON_CHANNELS,
    DEAP_CHANNEL_NAMES,
    DEAP_COMMON_CHANNEL_INDICES,
    PreprocessConfig,
    process_trials,
)
from src.splits import ALL_PROTOCOLS, load_manifest, write_all_manifests
from src.training import TrainRecipe, fit_supervised

ROOT = Path(__file__).resolve().parent.parent
N_TRIALS, N_WIN, N_BANDS, N_FEAT = 5, 4, 4, 273


def write_subject(directory: Path, subject_id: str, value: float, n_trials=N_TRIALS, n_win=N_WIN):
    """Synthetic subject in the v2 processed-file schema."""
    rng = np.random.default_rng(abs(hash(subject_id)) % 2**32)
    n = n_trials * n_win
    trial_index = np.repeat(np.arange(n_trials), n_win)
    labels = (trial_index % 2).astype(np.int64)
    np.savez_compressed(
        directory / f"{subject_id}.npz",
        subject_id=np.array(subject_id),
        raw_windows=(value + rng.standard_normal((n, 14, 512))).astype(np.float32),
        de_features=(value + rng.standard_normal((n, 14, N_BANDS))).astype(np.float32),
        de_sequence=(value + rng.standard_normal((n, 14, 4, N_BANDS))).astype(np.float32),
        handcrafted=(value + rng.standard_normal((n, N_FEAT))).astype(np.float32),
        trial_index=trial_index,
        window_index=np.tile(np.arange(n_win), n_trials),
        labels_valence=labels, labels_arousal=labels, labels_dominance=labels,
        channel_names=np.array(COMMON_CHANNELS),
        band_names=np.array(["theta", "alpha", "beta", "gamma"]),
        fs=np.array(128), window_samples=np.array(512),
    )


class ChannelMapTests(unittest.TestCase):
    def test_deap_channel_indices(self):
        # Official DEAP preprocessed order: Fz is index 18 and Cz index 23.
        self.assertEqual(DEAP_CHANNEL_NAMES[18], "Fz")
        self.assertEqual(DEAP_CHANNEL_NAMES[23], "Cz")
        self.assertEqual(DEAP_COMMON_CHANNEL_INDICES,
                         [1, 3, 2, 4, 7, 11, 13, 31, 29, 25, 21, 19, 20, 17])

    def test_deap_channels_correlate_with_scalp_neighbours(self):
        path = ROOT / "data" / "raw" / "DEAP" / "s01.dat"
        if not path.exists():
            self.skipTest("raw DEAP s01.dat not available")
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")["data"][:, :32, 384:]
        corr = np.corrcoef(np.concatenate(list(data), axis=1))
        neighbours = {"F4": {"Fz", "FC2", "F8", "AF4", "Fp2", "FC6"},
                      "O2": {"PO4", "Oz", "P8", "P4", "O1"},
                      "T8": {"FC6", "CP6", "C4", "P8", "F8", "P4"}}
        for ch, expected in neighbours.items():
            i = DEAP_CHANNEL_NAMES.index(ch)
            top = {DEAP_CHANNEL_NAMES[j] for j in np.argsort(-corr[i])[1:4]}
            self.assertTrue(top & expected, f"{ch}: top correlates {top}")


class PreprocessingTests(unittest.TestCase):
    def test_process_trials_shapes_and_labels(self):
        cfg = PreprocessConfig(dataset="DEAP")
        rng = np.random.default_rng(0)
        stimuli = [rng.standard_normal((128 * 13, 14)) for _ in range(3)]
        baselines = [rng.standard_normal((384, 14)) for _ in range(3)]
        ratings = {"valence": np.array([2.0, 5.0, 7.5]), "arousal": np.ones(3), "dominance": np.ones(3)}
        out = process_trials(stimuli, baselines, ratings, cfg)
        n = 9  # 13 s -> three 4 s windows per trial
        self.assertEqual(out["raw_windows"].shape, (n, 14, 512))
        self.assertEqual(out["de_features"].shape, (n, 14, 4))
        self.assertEqual(out["de_sequence"].shape, (n, 14, 4, 4))
        self.assertEqual(out["handcrafted"].shape, (n, 273))
        np.testing.assert_array_equal(out["window_index"], np.tile([0, 1, 2], 3))
        # strictly above the midpoint (5 on DEAP): 5.0 is low
        np.testing.assert_array_equal(out["labels_valence"], np.repeat([0, 0, 1], 3))
        self.assertTrue(np.isfinite(out["handcrafted"]).all())


class SplitAndLeakageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src, self.tgt, self.splits = root / "A", root / "B", root / "splits"
        self.src.mkdir()
        self.tgt.mkdir()
        for i in range(6):
            write_subject(self.src, f"s{i}", float(i))
        for i in range(2):
            write_subject(self.tgt, f"s{i}", 100.0 + i)   # same ids as source on purpose
        self.paths = write_all_manifests("A", str(self.src), str(self.splits), seed=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_protocol_keeps_its_guarantees_and_tests_each_window_once(self):
        for protocol in ALL_PROTOCOLS:
            manifest = load_manifest(self.paths[protocol], str(self.src))
            tested = []
            for fold in range(manifest["n_folds"]):
                train, val, test, meta = build_fold_loaders(
                    str(self.src), self.paths[protocol], fold, feature="de", batch_size=8)
                self.assertTrue(meta["leakage"]["passed"])
                self.assertTrue(meta["leakage"]["normalizer_fit_on_train_only"])
                ds = test.dataset
                tested += list(zip(ds.subject_ids, ds.trial_ids, ds.window_ids))
            self.assertEqual(len(tested), len(set(tested)), protocol)
            self.assertEqual(len(tested), 6 * N_TRIALS * N_WIN, protocol)

    def test_strict_protocols_are_subject_disjoint_and_ablations_are_not(self):
        _, _, _, loso = build_fold_loaders(str(self.src), self.paths["loso"], 0, feature="de")
        _, _, _, trial = build_fold_loaders(str(self.src), self.paths["trial_kfold"], 0, feature="de")
        _, _, _, rand = build_fold_loaders(str(self.src), self.paths["random_window"], 0, feature="de")
        self.assertEqual(loso["leakage"]["subject_overlap_test"], 0)
        self.assertGreater(trial["leakage"]["subject_overlap_test"], 0)
        self.assertEqual(trial["leakage"]["trial_overlap_test"], 0)
        self.assertGreater(rand["leakage"]["trial_overlap_test"], 0)

    def test_tampered_manifest_is_rejected(self):
        with open(self.paths["loso"]) as f:
            manifest = json.load(f)
        manifest["folds"][0]["train"].append(manifest["folds"][0]["test"][0])
        tampered = Path(self.tmp.name) / "tampered.json"
        tampered.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_manifest(str(tampered))

    def test_broken_guarantee_raises(self):
        train, val, test, _ = build_fold_loaders(str(self.src), self.paths["loso"], 0, feature="de")
        with self.assertRaises(LeakageError):
            check_partitions(train.dataset, val.dataset, train.dataset,
                             {"subject_disjoint": True, "trial_disjoint": True, "window_disjoint": True})

    def test_transfer_uses_source_only_statistics_and_namespaces_domains(self):
        train, val, test, meta = build_transfer_fold_loaders(
            str(self.src), str(self.tgt), self.paths["transfer_source"], "A", "B", feature="de")
        self.assertTrue(meta["leakage"]["passed"])            # equal ids, different datasets
        self.assertEqual(meta["target_subjects"], ["s0", "s1"])
        np.testing.assert_allclose(train.dataset.x.mean(dim=0).numpy(), 0.0, atol=1e-5)
        self.assertGreater(float(test.dataset.x.mean()), 10.0)

    def test_unlabeled_target_never_exposes_labels(self):
        _, _, test, _ = build_fold_loaders(str(self.src), self.paths["loso"], 0, feature="de")
        batch = next(iter(torch.utils.data.DataLoader(UnlabeledDataset(test.dataset), batch_size=4)))
        assert_unlabeled(batch)
        with self.assertRaises(LeakageError):
            assert_unlabeled((batch, torch.zeros(4)))

    def test_clisa_pairs_share_stimulus_and_differ_in_subject(self):
        train, _, _, _ = build_fold_loaders(str(self.src), self.paths["loso"], 0, feature="raw")
        ds = train.dataset
        sampler = clisa.StimulusPairSampler(ds.subject_ids, ds.trial_ids, ds.window_ids, seed=0)
        a, b = sampler.sample(8)
        for i, j in zip(a, b):
            self.assertNotEqual(ds.subject_ids[i], ds.subject_ids[j])
            self.assertEqual((ds.trial_ids[i], ds.window_ids[i]), (ds.trial_ids[j], ds.window_ids[j]))

    def test_all_proposal_models_train_one_epoch(self):
        recipe = TrainRecipe(max_epochs=1, patience=1, min_epochs=1, batch_size=16)
        cases = [
            (dadpc, "de", dadpc.fit_dadpc, {"pretrain_epochs": 1}),
            (cit_emotionnet, "de", fit_supervised,
             {"build": {"widths": (8, 8, 8, 8), "dim": 16, "heads": 2, "blocks_per_stage": 1}}),
            (stgate, "de_seq", fit_supervised, {}),
            (dah_net, "handcrafted", fit_supervised, {"build": {"lstm_hidden": 16}}),
            (clisa, "raw", clisa.fit_clisa, {"pretrain_epochs": 1}),
        ]
        for module, feature, fit, extra in cases:
            train, val, test, meta = build_fold_loaders(
                str(self.src), self.paths["loso"], 0, feature=feature, batch_size=16)
            model = module.build({"meta": meta, **extra.pop("build", {})})
            kwargs = dict(extra)
            if module is dadpc:
                kwargs["unlabeled_loader"] = torch.utils.data.DataLoader(
                    UnlabeledDataset(test.dataset), batch_size=16)
            info = fit(model, train, val, torch.device("cpu"), recipe, **kwargs)
            self.assertIn("train_time_sec", info)
            with torch.no_grad():
                out = model.eval()(next(iter(test))[0])
            self.assertEqual(out.shape[1], 2, module.__name__)


class LegacyLoaderTests(unittest.TestCase):
    """The subject-list helpers still work on processed files without window_index."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src, self.tgt = root / "source", root / "target"
        self.src.mkdir()
        self.tgt.mkdir()
        for sid, v in (("s1", 1.0), ("s2", 2.0), ("s3", 3.0)):
            write_subject(self.src, sid, v, n_trials=2, n_win=2)
        write_subject(self.tgt, "t1", 100.0, n_trials=2, n_win=2)
        self.splits = root / "loso.json"
        self.splits.write_text(json.dumps({"n_folds": 1, "folds": [
            {"fold": 0, "test_subject": "s3", "train_subjects": ["s1", "s2"]}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loso_validation_is_subject_disjoint(self):
        train, val, test, meta = build_loso_loaders(
            str(self.src), str(self.splits), fold=0, feature="de", batch_size=2, val_frac=0.5, seed=4)
        tr, va, te = set(train.dataset.subject_ids), set(val.dataset.subject_ids), set(test.dataset.subject_ids)
        self.assertFalse(tr & va or tr & te or va & te)

    def test_transfer_loaders(self):
        train, val, test, _ = build_transfer_loaders(
            str(self.src), str(self.tgt), ["s1", "s2", "s3"], ["t1"], feature="de", batch_size=2)
        self.assertEqual(set(test.dataset.subject_ids), {"t1"})


if __name__ == "__main__":
    unittest.main()
