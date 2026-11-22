"""Tests for datasets, the step-by-step explainer, plots and the command line.

Run from the project root:
    python -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli  # noqa: E402
from datasets import DATASET_NAMES, get_dataset  # noqa: E402
from explain import explain_training_step  # noqa: E402
from neural_network import NeuralNetwork  # noqa: E402

try:
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

XOR_X, XOR_Y = get_dataset("xor")


def run_cli(*argv):
    """Run main.main() quietly; return (exit_code, printed_text)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = cli.main(list(argv))
    return code, buffer.getvalue()


class TestDatasets(unittest.TestCase):
    def test_expected_truth_tables(self):
        expected = {"xor": [0, 1, 1, 0], "and": [0, 0, 0, 1],
                    "or": [0, 1, 1, 1], "nand": [1, 1, 1, 0]}
        for name, outputs in expected.items():
            X, y = get_dataset(name)
            self.assertEqual(X.shape, (4, 2))
            self.assertEqual(y.shape, (4, 1))
            self.assertEqual(y[:, 0].tolist(), outputs, msg=name)
        self.assertEqual(set(DATASET_NAMES), set(expected))

    def test_names_are_case_insensitive_and_validated(self):
        np.testing.assert_array_equal(get_dataset("XOR")[1], get_dataset("xor")[1])
        with self.assertRaises(ValueError):
            get_dataset("nor-ish")

    def test_returned_arrays_are_independent_copies(self):
        X, _ = get_dataset("xor")
        X[0, 0] = 99
        self.assertEqual(get_dataset("xor")[0][0, 0], 0.0)


class TestExplain(unittest.TestCase):
    def test_walkthrough_contents_and_self_check(self):
        net = NeuralNetwork((2, 4, 1), seed=42)
        text = explain_training_step(net, XOR_X, XOR_Y, example_index=3)
        for expected in ("FORWARD PROPAGATION", "LOSS", "BACKPROPAGATION",
                         "GRADIENT DESCENT", "match NeuralNetwork.backward(): yes"):
            self.assertIn(expected, text)

    def test_does_not_change_the_network(self):
        net = NeuralNetwork((2, 4, 1), seed=42)
        before = [w.copy() for w in net.weights] + [b.copy() for b in net.biases]
        explain_training_step(net, XOR_X, XOR_Y, example_index=1)
        after = net.weights + net.biases
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_preview_loss_decreases(self):
        net = NeuralNetwork((2, 4, 1), seed=42)
        text = explain_training_step(net, XOR_X, XOR_Y, example_index=3)
        line = next(l for l in text.splitlines() if l.startswith("Result: loss"))
        numbers = [float(t.rstrip(".")) for t in line.replace("to", " ").split() if t[0].isdigit()]
        self.assertLess(numbers[1], numbers[0])

    def test_works_for_deeper_networks_and_other_activations(self):
        net = NeuralNetwork((2, 5, 3, 1), hidden_activation="tanh", seed=1)
        self.assertIn("match NeuralNetwork.backward(): yes",
                      explain_training_step(net, XOR_X, XOR_Y, example_index=2))

    def test_zero_input_gets_an_explanatory_note(self):
        net = NeuralNetwork((2, 4, 1), seed=42)
        self.assertIn("every input is 0", explain_training_step(net, XOR_X, XOR_Y, 0))

    def test_bad_example_index(self):
        net = NeuralNetwork((2, 4, 1), seed=42)
        with self.assertRaises(ValueError):
            explain_training_step(net, XOR_X, XOR_Y, example_index=4)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plots
        cls.plots = plots
        cls.net = NeuralNetwork((2, 4, 1), seed=42)
        cls.net.train(XOR_X, XOR_Y, epochs=300, verbose=False)

    def assert_file(self, path):
        self.assertTrue(os.path.exists(path), path)
        self.assertGreater(os.path.getsize(path), 1000)

    def test_static_plots_are_created(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = {name: os.path.join(folder, "sub", name + ".png")
                     for name in ("loss", "boundary", "hidden", "compare")}
            self.plots.save_loss_plot(self.net.loss_history, paths["loss"])
            self.plots.save_decision_boundary(self.net, XOR_X, XOR_Y, paths["boundary"])
            self.plots.save_hidden_activations(self.net, XOR_X, paths["hidden"])
            self.plots.save_comparison_plot({"a": self.net.loss_history,
                                             "b": self.net.loss_history[::-1]}, paths["compare"])
            for path in paths.values():
                self.assert_file(path)

    def test_hidden_activation_layer_is_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.plots.save_hidden_activations(self.net, XOR_X, os.path.join(folder, "x.png"),
                                                   layer=2)  # layer 2 is the output layer

    def test_boundary_needs_two_inputs(self):
        net = NeuralNetwork((3, 2, 1), seed=0)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.plots.save_decision_boundary(net, XOR_X, XOR_Y, os.path.join(folder, "x.png"))

    def test_empty_comparison_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plots.save_comparison_plot({}, "unused.png")

    def test_recorder_limits_frames_and_makes_a_gif(self):
        recorder = self.plots.TrainingRecorder(total_epochs=60, max_frames=10, resolution=20)
        net = NeuralNetwork((2, 4, 1), seed=42)
        net.train(XOR_X, XOR_Y, epochs=60, verbose=False, callback=recorder)
        self.assertLessEqual(len(recorder.grids), 10)
        self.assertEqual(recorder.epochs[0], 1)
        self.assertEqual(recorder.epochs, sorted(set(recorder.epochs)))
        self.assertEqual(recorder.grids[0].shape, (20, 20))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "anim.gif")
            recorder.save_gif(XOR_X, XOR_Y, path)
            self.assert_file(path)

    def test_recorder_needs_two_frames(self):
        recorder = self.plots.TrainingRecorder(total_epochs=10)
        with self.assertRaises(ValueError):
            recorder.save_gif(XOR_X, XOR_Y, "unused.gif")


class TestCommandLine(unittest.TestCase):
    FAST = ("--epochs", "300", "--print-every", "0")

    def test_default_run_reports_predictions_and_accuracy(self):
        code, text = run_cli("--epochs", "10000", "--print-every", "5000", "--no-plots")
        self.assertEqual(code, 0)
        for expected in ("NEURAL NETWORK FROM SCRATCH", "Architecture: 2 -> 4 -> 1",
                         "Input: [0, 1] | Expected: 1", "Final Accuracy: 100%"):
            self.assertIn(expected, text)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_default_run_saves_graphs(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _ = run_cli(*self.FAST, "--output-dir", folder)
            self.assertEqual(code, 0)
            self.assertEqual(sorted(os.listdir(folder)),
                             ["xor_decision_boundary.png", "xor_hidden_activations.png",
                              "xor_loss_curve.png"])

    def test_no_plots_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            run_cli(*self.FAST, "--output-dir", folder, "--no-plots")
            self.assertEqual(os.listdir(folder), [])

    def test_architecture_and_activation_options(self):
        code, text = run_cli(*self.FAST, "--hidden", "6", "3", "--activation", "tanh",
                             "--optimizer", "momentum", "--batch-size", "2", "--no-plots")
        self.assertEqual(code, 0)
        self.assertIn("2 -> 6 -> 3 -> 1", text)
        self.assertIn("Tanh (hidden)", text)
        self.assertIn("Momentum", text)
        self.assertIn("mini-batch of 2", text)

    def test_all_datasets(self):
        code, text = run_cli("--dataset", "all", "--epochs", "3000", "--no-plots")
        self.assertEqual(code, 0)
        for name in ("XOR", "AND", "OR", "NAND"):
            self.assertIn(name, text)
        self.assertEqual(text.count("100%"), 4)

    def test_early_stopping_is_reported(self):
        code, text = run_cli("--epochs", "30000", "--early-stopping", "200", "--min-delta", "1e-4",
                             "--print-every", "0", "--no-plots")
        self.assertEqual(code, 0)
        self.assertIn("Early stopping at epoch", text)
        self.assertIn("Stopped early after", text)

    def test_save_then_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model")
            code, trained = run_cli("--epochs", "5000", "--print-every", "0", "--no-plots",
                                    "--save", path)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path + ".npz"))
            code, loaded = run_cli("--load", path, "--no-plots")
            self.assertEqual(code, 0)
            self.assertIn("Loaded model", loaded)
            # The header must describe the loaded model, not the command-line defaults.
            self.assertNotIn("Architecture:", loaded)
            self.assertNotIn("Optimizer:", loaded)
            probabilities = lambda t: [l for l in t.splitlines() if l.startswith("Input:")]
            self.assertEqual(probabilities(trained), probabilities(loaded))

    def test_explain_mode_does_not_train(self):
        code, text = run_cli("--explain")
        self.assertEqual(code, 0)
        self.assertIn("STEP-BY-STEP", text)
        self.assertNotIn("Training started", text)

    def test_comparisons(self):
        code, text = run_cli("--epochs", "1500", "--print-every", "0", "--no-plots",
                             "--compare-optimizers", "--compare-activations")
        self.assertEqual(code, 0)
        for expected in ("Optimizer comparison", "Momentum", "Mini-batch of 2",
                         "Activation comparison", "relu", "sigmoid", "tanh"):
            self.assertIn(expected, text)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_animation_option_writes_a_gif(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _ = run_cli("--epochs", "120", "--print-every", "0", "--animate",
                              "--output-dir", folder)
            self.assertEqual(code, 0)
            self.assertIn("xor_training.gif", os.listdir(folder))

    def test_invalid_options_exit_with_code_2(self):
        bad_runs = (("--lr", "-1"), ("--hidden", "0"), ("--momentum", "1.5"),
                    ("--dataset", "all", "--save", "x"), ("--activation", "swish"),
                    ("--load", "x.npz", "--animate"), ("--explain-example", "9"))
        for argv in bad_runs:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as caught:
                    run_cli(*argv)
                self.assertEqual(caught.exception.code, 2)

    def test_missing_model_file_exits_with_code_1(self):
        code, text = run_cli("--load", "no_such_model.npz", "--no-plots")
        self.assertEqual(code, 1)
        self.assertIn("No saved model found", text)


if __name__ == "__main__":
    unittest.main()
