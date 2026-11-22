"""Tests for the newer features: softmax, Adam & friends, real datasets,
train/validation/test splits, experiments, sweeps, gradient checking,
the notebook and the browser playground.

Run from the project root:
    python -m unittest discover -s tests

No test writes into the real outputs/ folder (they use temporary folders).
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as cli  # noqa: E402
from datasets import (Standardizer, load_dataset, standardize_split,  # noqa: E402
                      train_val_test_split)
from experiment import (epochs_to_threshold, resolve_settings,  # noqa: E402
                        run_experiment)
from gradcheck import format_gradient_check, gradient_check  # noqa: E402
from neural_network import NeuralNetwork  # noqa: E402
from sweep import parse_grid, reliability_report, run_sweep  # noqa: E402

try:
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


def run_cli(*argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = cli.main(list(argv))
    return code, buffer.getvalue()


def one_hot(labels, classes):
    return np.eye(classes)[labels]


def small_multiclass_data(seed=0, rows=30, features=3, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features))
    return X, one_hot(rng.integers(0, classes, rows), classes)


class TestSoftmax(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        X, _ = small_multiclass_data()
        net = NeuralNetwork((3, 5, 3), output_activation="softmax", seed=1)
        self.assertTrue(net.is_multiclass)
        p = net.predict_proba(X)
        self.assertEqual(p.shape, (30, 3))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertEqual(net.predict(X).shape, (30,))

    def test_learns_separable_blobs(self):
        data = load_dataset("blobs", seed=0)
        net = NeuralNetwork((2, 8, data.n_classes), output_activation="softmax",
                            learning_rate=0.5, seed=0)
        y = one_hot(data.y, data.n_classes)
        net.train(data.X, y, epochs=800, verbose=False)
        self.assertGreater(net.evaluate(data.X, y)[1], 0.9)

    def test_confusion_matrix(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", seed=2)
        matrix = net.confusion_matrix(X, y)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.sum(), 30)
        np.testing.assert_array_equal(matrix.sum(axis=1), y.sum(axis=0))

    def test_confusion_matrix_needs_softmax(self):
        net = NeuralNetwork((2, 3, 1), seed=0)
        with self.assertRaises(ValueError):
            net.confusion_matrix(np.zeros((2, 2)), np.zeros((2, 1)))


class TestOptimizersAndOptions(unittest.TestCase):
    def test_adam_first_step_moves_each_weight_by_learning_rate(self):
        # With bias correction, Adam's very first step is lr * sign(gradient).
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", optimizer="adam",
                            learning_rate=0.01, seed=3)
        before = [w.copy() for w in net.weights]
        grads = net.backward(net.forward(X), y)
        net.update_parameters(*grads)
        for old, new, g in zip(before, net.weights, grads[0]):
            mask = np.abs(g) > 1e-6
            np.testing.assert_allclose((old - new)[mask], 0.01 * np.sign(g[mask]), rtol=1e-3)

    def test_rmsprop_first_step_is_analytic(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", optimizer="rmsprop",
                            learning_rate=0.01, seed=3)
        before = [w.copy() for w in net.weights]
        grads = net.backward(net.forward(X), y)
        net.update_parameters(*grads)
        for old, new, g in zip(before, net.weights, grads[0]):
            expected = 0.01 * g / (np.sqrt(0.1 * g * g) + 1e-8)     # cache = (1-0.9) g^2
            np.testing.assert_allclose(old - new, expected, rtol=1e-3, atol=1e-9)

    def test_each_optimizer_reduces_loss(self):
        data = load_dataset("blobs", seed=1)
        y = one_hot(data.y, data.n_classes)
        for optimizer, lr in (("gd", 0.5), ("momentum", 0.1), ("rmsprop", 0.01), ("adam", 0.01)):
            with self.subTest(optimizer=optimizer):
                net = NeuralNetwork((2, 6, data.n_classes), output_activation="softmax",
                                    optimizer=optimizer, learning_rate=lr, seed=0)
                net.train(data.X, y, epochs=200, verbose=False)
                self.assertLess(net.loss_history[-1], net.loss_history[0])

    def test_unknown_options_rejected(self):
        for kwargs in ({"optimizer": "lion"}, {"weight_init": "magic"},
                       {"lr_schedule": "zigzag"}, {"output_activation": "relu"},
                       {"dropout": 1.0}, {"l2": -1.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NeuralNetwork((2, 3, 1), **kwargs)

    def test_leaky_relu_and_weight_inits_run(self):
        X = np.random.default_rng(0).normal(size=(10, 2))
        for init in ("auto", "xavier", "he", "normal", "small"):
            net = NeuralNetwork((2, 5, 1), hidden_activation="leaky_relu", weight_init=init, seed=0)
            self.assertTrue(np.all(np.isfinite(net.predict_proba(X))))

    def test_he_init_scale(self):
        net = NeuralNetwork((400, 400, 1), hidden_activation="relu", weight_init="he", seed=0)
        self.assertAlmostEqual(net.weights[0].std(), np.sqrt(2 / 400), delta=0.003)

    def test_lr_schedules(self):
        X, y = small_multiclass_data()
        for schedule in ("constant", "step", "exponential", "cosine"):
            net = NeuralNetwork((3, 4, 3), output_activation="softmax", lr_schedule=schedule,
                                lr_decay=0.5, lr_step=10, learning_rate=0.1, seed=0)
            net.train(X, y, epochs=40, verbose=False)
            lr = net.history["lr"]
            self.assertEqual(len(lr), 40)
            self.assertAlmostEqual(lr[0], 0.1)
            if schedule == "constant":
                self.assertEqual(set(lr), {0.1})
            else:
                self.assertLess(lr[-1], lr[0])
                self.assertTrue(all(b <= a + 1e-12 for a, b in zip(lr, lr[1:])))

    def test_l2_adds_to_total_loss_only(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", l2=0.1, seed=0)
        p = net.forward(X)[-1]
        self.assertGreater(net.l2_penalty(), 0)
        self.assertAlmostEqual(net.total_loss(y, p), net.compute_loss(y, p) + net.l2_penalty())
        self.assertAlmostEqual(net.evaluate(X, y)[0], net.compute_loss(y, p))

    def test_dropout_only_active_in_training(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 20, 3), output_activation="softmax", dropout=0.5, seed=0)
        np.testing.assert_array_equal(net.predict_proba(X), net.predict_proba(X))
        used, raw, masks = net._forward_train(X)
        self.assertTrue(any((m == 0).any() for m in masks if m is not None))
        net.train(X, y, epochs=5, verbose=False)            # runs without error

    def test_dropout_changes_nothing_when_zero(self):
        a = NeuralNetwork((2, 4, 1), seed=5)
        b = NeuralNetwork((2, 4, 1), dropout=0.0, seed=5)
        X = np.array([[0, 1], [1, 0.0]])
        y = np.array([[1], [1]])
        a.train(X, y, epochs=20, verbose=False)
        b.train(X, y, epochs=20, verbose=False)
        np.testing.assert_array_equal(a.weights[0], b.weights[0])


class TestTrainingControls(unittest.TestCase):
    def test_history_keys_and_lengths(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", seed=0)
        net.train(X[:20], y[:20], epochs=15, verbose=False, validation_data=(X[20:], y[20:]))
        for key in ("loss", "accuracy", "val_loss", "val_accuracy", "lr", "weight_norm", "grad_norm"):
            self.assertEqual(len(net.history[key]), 15, key)
        self.assertIs(net.loss_history, net.history["loss"])

    def test_restore_best_returns_lowest_validation_loss(self):
        data = load_dataset("moons", seed=0)
        split = train_val_test_split(data.X, data.y, val_size=0.3, seed=0)
        net = NeuralNetwork((2, 30, 1), hidden_activation="tanh", learning_rate=0.5, seed=0)
        net.train(split.X_train, split.y_train.reshape(-1, 1), epochs=600, verbose=False,
                  validation_data=(split.X_val, split.y_val.reshape(-1, 1)), restore_best=True)
        best = min(net.history["val_loss"])
        val_loss = net.evaluate(split.X_val, split.y_val.reshape(-1, 1))[0]
        self.assertAlmostEqual(val_loss, best, places=9)
        self.assertEqual(net.history["val_loss"][net.best_epoch - 1], best)

    def test_early_stopping_uses_validation_loss(self):
        data = load_dataset("moons", seed=0)
        split = train_val_test_split(data.X, data.y, val_size=0.3, seed=0)
        net = NeuralNetwork((2, 30, 1), hidden_activation="tanh", learning_rate=0.5, seed=0)
        net.train(split.X_train, split.y_train.reshape(-1, 1), epochs=5000, verbose=False,
                  validation_data=(split.X_val, split.y_val.reshape(-1, 1)),
                  early_stopping_patience=20, min_delta=1e-4)
        self.assertTrue(net.stopped_early)
        self.assertLess(net.epochs_run, 5000)

    def test_bad_training_arguments(self):
        net = NeuralNetwork((2, 3, 1), seed=0)
        X, y = np.zeros((4, 2)), np.zeros((4, 1))
        for kwargs in ({"epochs": 0}, {"batch_size": 0}, {"early_stopping_patience": 0},
                       {"min_delta": -1}, {"callback_every": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    net.train(X, y, verbose=False, **kwargs)

    def test_divergence_raises_clear_error(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 4, 3), hidden_activation="relu", output_activation="softmax",
                            learning_rate=1e6, seed=0)
        with np.errstate(all="ignore"), self.assertRaises(RuntimeError):
            net.train(X * 5, y, epochs=200, verbose=False)


class TestSaveLoad(unittest.TestCase):
    def test_round_trip_with_extra_and_new_settings(self):
        X, y = small_multiclass_data()
        net = NeuralNetwork((3, 5, 3), hidden_activation="tanh", output_activation="softmax",
                            optimizer="adam", learning_rate=0.01, l2=0.01, weight_init="xavier",
                            lr_schedule="cosine", seed=4)
        net.train(X, y, epochs=10, verbose=False)
        with tempfile.TemporaryDirectory() as folder:
            path = net.save(os.path.join(folder, "m"), extra={"mean": np.arange(3.0)})
            self.assertTrue(path.endswith(".npz"))
            loaded = NeuralNetwork.load(path)
        np.testing.assert_array_equal(loaded.predict_proba(X), net.predict_proba(X))
        self.assertEqual(loaded.output_activation, "softmax")
        self.assertEqual(loaded.optimizer, "adam")
        self.assertEqual(loaded.lr_schedule, "cosine")
        np.testing.assert_array_equal(loaded.extra["mean"], np.arange(3.0))

    def test_old_format_files_still_load(self):
        net = NeuralNetwork((2, 3, 1), seed=1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.npz")
            np.savez(path, layer_sizes=np.array(net.layer_sizes),
                     hidden_activation=np.array("sigmoid"), learning_rate=np.array(0.5),
                     optimizer=np.array("gd"), momentum=np.array(0.9), seed=np.array(1),
                     loss_history=np.array([1.0, 0.5]),
                     W0=net.weights[0], b0=net.biases[0], W1=net.weights[1], b1=net.biases[1])
            loaded = NeuralNetwork.load(path)
        self.assertEqual(loaded.output_activation, "sigmoid")
        self.assertEqual(loaded.l2, 0.0)

    def test_corrupt_file_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.npz")
            np.savez(path, something=np.zeros(2))
            with self.assertRaises(ValueError):
                NeuralNetwork.load(path)


class TestSplitAndScaling(unittest.TestCase):
    def setUp(self):
        self.data = load_dataset("iris")

    def test_sizes_no_overlap_and_stratified(self):
        X, y = self.data.X, self.data.y
        # Tag each row with its index so overlap can be detected.
        tagged = np.column_stack([X, np.arange(len(X))])
        s = train_val_test_split(tagged, y, val_size=0.2, test_size=0.2, seed=1)
        ids = [set(part[:, -1].astype(int)) for part in (s.X_train, s.X_val, s.X_test)]
        self.assertEqual(len(ids[0] | ids[1] | ids[2]), 150)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for labels in (s.y_val, s.y_test):
            self.assertEqual(np.bincount(labels).tolist(), [10, 10, 10])

    def test_reproducible_and_seed_dependent(self):
        a = train_val_test_split(self.data.X, self.data.y, 0.2, 0.2, seed=3)
        b = train_val_test_split(self.data.X, self.data.y, 0.2, 0.2, seed=3)
        c = train_val_test_split(self.data.X, self.data.y, 0.2, 0.2, seed=4)
        np.testing.assert_array_equal(a.X_test, b.X_test)
        self.assertFalse(np.array_equal(a.X_test, c.X_test))

    def test_no_split_keeps_everything_for_training(self):
        s = train_val_test_split(self.data.X, self.data.y)
        self.assertEqual(len(s.X_train), 150)
        self.assertIsNone(s.X_test)
        self.assertIsNone(s.validation_data)

    def test_invalid_sizes(self):
        for kwargs in ({"val_size": -0.1}, {"test_size": 1.0}, {"val_size": 0.6, "test_size": 0.5}):
            with self.assertRaises(ValueError):
                train_val_test_split(self.data.X, self.data.y, **kwargs)

    def test_standardizer_uses_training_statistics_only(self):
        train = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
        scaler = Standardizer().fit(train)
        np.testing.assert_allclose(scaler.transform(train)[:, 0].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaler.transform(train)[:, 0].std(), 1.0)
        np.testing.assert_array_equal(scaler.transform(train)[:, 1], 0.0)   # constant column ok
        np.testing.assert_allclose(scaler.transform(np.array([[10.0, 5.0]]))[0, 0], (10 - 2) / scaler.std[0])
        with self.assertRaises(ValueError):
            Standardizer().transform(train)

    def test_standardize_split_fits_on_train(self):
        s = train_val_test_split(self.data.X, self.data.y, 0.2, 0.2, seed=0)
        scaled, scaler = standardize_split(s)
        np.testing.assert_allclose(scaled.X_train.mean(axis=0), 0.0, atol=1e-9)
        # validation/test are NOT forced to mean 0 (they use the training numbers)
        self.assertGreater(np.abs(scaled.X_test.mean(axis=0)).max(), 1e-6)


class TestRealDatasets(unittest.TestCase):
    def test_iris(self):
        d = load_dataset("iris")
        self.assertEqual(d.X.shape, (150, 4))
        self.assertEqual(np.bincount(d.y).tolist(), [50, 50, 50])
        self.assertEqual(d.class_names, ("setosa", "versicolor", "virginica"))
        self.assertTrue(d.is_real)
        self.assertEqual(d.task, "multiclass")

    def test_digits(self):
        d = load_dataset("digits8x8")
        self.assertEqual(d.X.shape, (1797, 64))
        self.assertGreaterEqual(d.X.min(), 0.0)
        self.assertLessEqual(d.X.max(), 1.0)
        self.assertEqual(d.image_shape, (8, 8))
        self.assertEqual(d.n_classes, 10)
        self.assertEqual(set(np.unique(d.y)), set(range(10)))

    def test_samples_option_rejected_for_real_data(self):
        with self.assertRaises(ValueError):
            load_dataset("iris", n_samples=50)

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            load_dataset("nonsense")

    def test_requirements_do_not_include_ml_frameworks(self):
        with open(os.path.join(ROOT, "requirements.txt")) as handle:
            text = handle.read().lower()
        for banned in ("scikit", "sklearn", "tensorflow", "torch", "keras"):
            self.assertNotIn(banned, text)

    def test_no_source_file_imports_ml_frameworks(self):
        for name in os.listdir(ROOT):
            if name.endswith(".py"):
                with open(os.path.join(ROOT, name)) as handle:
                    for line in handle:
                        if line.lstrip().startswith(("import ", "from ")):
                            for banned in ("sklearn", "tensorflow", "torch", "keras"):
                                self.assertNotIn(banned, line, f"{name}: {line}")


class TestExperimentsSweepsGradcheck(unittest.TestCase):
    def test_resolve_settings_precedence_and_unknown(self):
        d = load_dataset("iris")
        s = resolve_settings(d, {"epochs": 7, "lr": None})
        self.assertEqual(s["epochs"], 7)                        # override wins
        self.assertEqual(s["optimizer"], "adam")                # dataset default
        s = resolve_settings(d, {"optimizer": "gd"})
        self.assertEqual(s["lr"], 0.5)                          # lr follows optimizer
        with self.assertRaises(ValueError):
            resolve_settings(d, {"bogus": 1})

    def test_run_experiment_iris_is_deterministic_and_good(self):
        d = load_dataset("iris")
        a = run_experiment(d, {"epochs": 150, "seed": 7})
        b = run_experiment(d, {"epochs": 150, "seed": 7})
        self.assertEqual(a.metrics, b.metrics)
        for key in ("train_accuracy", "val_accuracy", "test_accuracy", "test_loss"):
            self.assertIn(key, a.metrics)
        self.assertGreater(a.metrics["test_accuracy"], 0.8)

    def test_epochs_to_threshold(self):
        self.assertEqual(epochs_to_threshold([1.0, 0.5, 0.05, 0.01]), 3)
        self.assertIsNone(epochs_to_threshold([1.0, 0.9]))

    def test_parse_grid(self):
        self.assertEqual(parse_grid(["lr=0.1,0.5", "hidden=4,8-4"]),
                         {"lr": [0.1, 0.5], "hidden": [[4], [8, 4]]})
        for bad in (["lr"], ["nonsense=1"], ["lr=0.1", "lr=0.2"], [], ["lr="], ["lr=abc"]):
            with self.assertRaises(ValueError):
                parse_grid(bad)

    def test_run_sweep_is_ranked(self):
        d = load_dataset("iris")
        rows = run_sweep(d, {"epochs": 60}, {"lr": [0.0001, 0.02]}, seeds=1, first_seed=1)
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(rows[0]["score"], rows[1]["score"])
        self.assertEqual(rows[0]["settings"], {"lr": 0.02})

    def test_reliability_report_on_xor(self):
        report = reliability_report(load_dataset("xor"), {"epochs": 3000}, 4, first_seed=0)
        self.assertEqual(report["seeds"], 4)
        self.assertEqual(len(report["train_accuracy"]), 4)
        self.assertTrue(0 <= report["solved"] <= 4)

    def test_gradient_check_passes_for_every_activation_and_output(self):
        X, y = small_multiclass_data(rows=8)
        for activation in ("sigmoid", "tanh", "relu", "leaky_relu"):
            with self.subTest(activation=activation):
                net = NeuralNetwork((3, 4, 3), hidden_activation=activation,
                                    output_activation="softmax", l2=0.05, seed=1)
                results = gradient_check(net, X, y)
                self.assertTrue(all(r["passed"] for r in results))
        text, ok = format_gradient_check(results)
        self.assertTrue(ok)
        self.assertIn("PASS", text)

    def test_gradient_check_detects_a_broken_backward(self):
        X, y = small_multiclass_data(rows=8)
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", seed=1)
        original = NeuralNetwork.backward

        def broken(self, *args, **kwargs):
            weights, biases = original(self, *args, **kwargs)
            weights[0] = weights[0] * 1.5
            return weights, biases

        NeuralNetwork.backward = broken
        try:
            _, ok = format_gradient_check(gradient_check(net, X, y))
        finally:
            NeuralNetwork.backward = original
        self.assertFalse(ok)

    def test_gradient_check_leaves_network_untouched(self):
        X, y = small_multiclass_data(rows=8)
        net = NeuralNetwork((3, 4, 3), output_activation="softmax", dropout=0.3, seed=1)
        before = [w.copy() for w in net.weights] + [b.copy() for b in net.biases]
        gradient_check(net, X, y)
        after = net.weights + net.biases
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(net.dropout, 0.3)


class TestCommandLineFeatures(unittest.TestCase):
    def test_iris_run(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = run_cli("--dataset", "iris", "--epochs", "80", "--print-every", "0",
                                 "--output-dir", folder, "--seed", "1")
            self.assertEqual(code, 0, text)
            self.assertIn("setosa", text)
            if HAS_MATPLOTLIB:
                self.assertTrue(any(f.startswith("iris") for f in os.listdir(folder)))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_digits_run_writes_prediction_picture(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = run_cli("--dataset", "digits8x8", "--epochs", "5", "--print-every", "0",
                                 "--output-dir", folder)
            self.assertEqual(code, 0, text)
            self.assertIn("digits8x8_predictions.png", os.listdir(folder))

    def test_config_file_and_command_line_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "c.json")
            with open(path, "w") as handle:
                json.dump({"dataset": "iris", "epochs": 40, "hidden": [6]}, handle)
            code, text = run_cli("--config", path, "--epochs", "20", "--print-every", "0",
                                 "--no-plots")
            self.assertEqual(code, 0, text)
            self.assertIn("20", text)

    def test_bad_config_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for content in ("{not json", "[1, 2]", '{"nonsense": 1}'):
                path = os.path.join(folder, "c.json")
                with open(path, "w") as handle:
                    handle.write(content)
                code, text = run_cli("--config", path, "--no-plots")
                self.assertNotEqual(code, 0, content)
            code, _ = run_cli("--config", os.path.join(folder, "missing.json"), "--no-plots")
            self.assertNotEqual(code, 0)

    def test_log_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "log.csv")
            code, text = run_cli("--epochs", "50", "--print-every", "0", "--no-plots",
                                 "--log-csv", path)
            self.assertEqual(code, 0, text)
            with open(path) as handle:
                lines = handle.read().strip().splitlines()
            self.assertIn("loss", lines[0])
            self.assertEqual(len(lines), 51)

    def test_gradient_check_mode(self):
        code, text = run_cli("--gradient-check", "--dataset", "iris", "--hidden", "5")
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)
        self.assertNotIn("FAIL", text)

    def test_sweep_mode(self):
        code, text = run_cli("--dataset", "iris", "--epochs", "30", "--sweep", "lr=0.001,0.02")
        self.assertEqual(code, 0, text)
        self.assertIn("0.02", text)

    def test_seeds_report_mode(self):
        code, text = run_cli("--dataset", "iris", "--epochs", "30", "--seeds-report", "3")
        self.assertEqual(code, 0, text)
        self.assertIn("3", text)

    def test_more_invalid_options(self):
        bad_runs = (("--dropout", "1"), ("--lr-decay", "0"), ("--test-size", "0.9", "--val-size", "0.2"),
                    ("--dataset", "iris", "--samples", "50"), ("--explain-example", "1"),
                    ("--dataset", "nonsense"), ("--sweep", "lr"))
        for argv in bad_runs:
            with self.subTest(argv=argv):
                try:
                    code, _ = run_cli(*argv, "--no-plots")
                except SystemExit as exit_:
                    code = exit_.code
                self.assertIn(code, (1, 2))


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
class TestImagePlot(unittest.TestCase):
    def test_save_image_predictions(self):
        import plots
        d = load_dataset("digits8x8")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p.png")
            plots.save_image_predictions(d.X[:30], d.y[:30], d.y[:30], d.image_shape, path)
            self.assertGreater(os.path.getsize(path), 1000)


class TestPlaygroundAndNotebook(unittest.TestCase):
    def test_notebook_is_valid_and_has_no_error_outputs(self):
        path = os.path.join(ROOT, "notebooks", "neural_network_walkthrough.ipynb")
        with open(path, encoding="utf-8") as handle:
            nb = json.load(handle)
        self.assertGreaterEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreater(len(code_cells), 10)
        for cell in code_cells:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")

    def test_playground_is_self_contained(self):
        with open(os.path.join(ROOT, "web", "playground.html"), encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn("<canvas", html)
        self.assertNotIn("src=\"http", html)
        self.assertNotIn("href=\"http", html)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_playground_script_parses(self):
        with open(os.path.join(ROOT, "web", "playground.html"), encoding="utf-8") as handle:
            html = handle.read()
        start = html.index("<script>") + len("<script>")
        script = html[start:html.rindex("</script>")]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(script)
            result = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
