"""Unit tests for the from-scratch neural network (core class).

Run from the project root:
    python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Make the project root importable when tests are run from the root folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (NeuralNetwork, binary_cross_entropy, relu,  # noqa: E402
                            relu_derivative, sigmoid, sigmoid_derivative, tanh,
                            tanh_derivative)

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=float)

# The default configuration used by main.py.
CONFIG = dict(layer_sizes=(2, 4, 1), learning_rate=0.5, seed=42)
EPOCHS = 10000


def first_epoch_below(history, threshold=0.1):
    return next((i + 1 for i, v in enumerate(history) if v < threshold), None)


class TestActivations(unittest.TestCase):
    def test_sigmoid_known_values(self):
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        self.assertGreater(float(sigmoid(10.0)), 0.9999)
        self.assertLess(float(sigmoid(-10.0)), 0.0001)

    def test_sigmoid_range_and_no_overflow(self):
        z = np.array([-1e6, -100.0, 0.0, 100.0, 1e6])
        with np.errstate(all="raise"):  # any overflow warning becomes an error
            out = sigmoid(z)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_sigmoid_derivative_uses_output(self):
        a = sigmoid(0.0)  # a = 0.5, so a * (1 - a) = 0.25
        self.assertAlmostEqual(float(sigmoid_derivative(a)), 0.25)

    def test_tanh_known_values(self):
        self.assertAlmostEqual(float(tanh(0.0)), 0.0)
        self.assertGreater(float(tanh(10.0)), 0.9999)
        self.assertLess(float(tanh(-10.0)), -0.9999)

    def test_relu_known_values(self):
        np.testing.assert_array_equal(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(relu_derivative(np.array([0.0, 0.5, 3.0])), [0.0, 1.0, 1.0])

    def test_derivatives_match_numerical_slope(self):
        z = np.array([-2.0, -0.5, 0.3, 1.7])
        h = 1e-6
        for function, derivative in ((sigmoid, sigmoid_derivative),
                                     (tanh, tanh_derivative),
                                     (relu, relu_derivative)):
            numeric = (function(z + h) - function(z - h)) / (2 * h)
            np.testing.assert_allclose(derivative(function(z)), numeric, atol=1e-8,
                                       err_msg=function.__name__)


class TestLoss(unittest.TestCase):
    def test_perfect_prediction_has_tiny_loss(self):
        self.assertLess(binary_cross_entropy(XOR_Y, XOR_Y), 1e-9)

    def test_confidently_wrong_is_finite_and_large(self):
        loss = binary_cross_entropy(XOR_Y, 1.0 - XOR_Y)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 10.0)


class TestShapesAndRanges(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork(**CONFIG)

    def test_parameter_shapes(self):
        self.assertEqual([w.shape for w in self.net.weights], [(2, 4), (4, 1)])
        self.assertEqual([b.shape for b in self.net.biases], [(1, 4), (1, 1)])
        self.assertEqual(self.net.count_parameters(), 17)

    def test_forward_shapes(self):
        activations = self.net.forward(XOR_X)
        self.assertEqual([a.shape for a in activations], [(4, 2), (4, 4), (4, 1)])

    def test_deeper_network_shapes(self):
        net = NeuralNetwork((2, 8, 4, 1), seed=1)
        self.assertEqual([a.shape for a in net.forward(XOR_X)],
                         [(4, 2), (4, 8), (4, 4), (4, 1)])
        self.assertEqual(net.count_parameters(), 2 * 8 + 8 + 8 * 4 + 4 + 4 * 1 + 1)

    def test_probabilities_in_valid_range(self):
        probs = self.net.predict_proba(XOR_X)
        self.assertEqual(probs.shape, (4, 1))
        self.assertTrue(np.all((probs > 0.0) & (probs < 1.0)))

    def test_predictions_are_binary(self):
        preds = self.net.predict(XOR_X)
        self.assertEqual(preds.shape, (4, 1))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_same_seed_gives_same_weights(self):
        other = NeuralNetwork(**CONFIG)
        for a, b in zip(self.net.weights, other.weights):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_weights(self):
        other = NeuralNetwork(**{**CONFIG, "seed": 7})
        self.assertFalse(np.array_equal(self.net.weights[0], other.weights[0]))

    def test_relu_uses_he_initialisation_scale(self):
        net = NeuralNetwork((2, 200, 1), hidden_activation="relu", seed=0)
        self.assertAlmostEqual(float(np.std(net.weights[0])), np.sqrt(2.0 / 2), delta=0.15)


class TestValidation(unittest.TestCase):
    def test_bad_layer_sizes(self):
        for bad in ((2,), (2, 0, 1), (2, -1, 1), (2, 1.5, 1), 5, (2, True, 1)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                NeuralNetwork(bad)

    def test_bad_settings(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(learning_rate=-0.1)
        with self.assertRaises(ValueError):
            NeuralNetwork(hidden_activation="swish")
        with self.assertRaises(ValueError):
            NeuralNetwork(optimizer="lion")
        with self.assertRaises(ValueError):
            NeuralNetwork(momentum=1.0)

    def test_bad_input_shape(self):
        net = NeuralNetwork(**CONFIG)
        with self.assertRaises(ValueError):
            net.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_nan_input_rejected(self):
        net = NeuralNetwork(**CONFIG)
        with self.assertRaises(ValueError):
            net.predict(np.array([[np.nan, 0.0]]))

    def test_bad_targets_rejected(self):
        net = NeuralNetwork(**CONFIG)
        with self.assertRaises(ValueError):
            net.train(XOR_X, np.array([[0], [1], [1]]), epochs=1, verbose=False)
        with self.assertRaises(ValueError):
            net.train(XOR_X, np.array([[0], [1], [1], [2]]), epochs=1, verbose=False)

    def test_bad_training_arguments(self):
        net = NeuralNetwork(**CONFIG)
        for kwargs in (dict(epochs=0), dict(batch_size=0), dict(early_stopping_patience=0),
                       dict(min_delta=-1.0), dict(callback_every=0)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                net.train(XOR_X, XOR_Y, **{"epochs": 1, "verbose": False, **kwargs})

    def test_diverging_training_raises_clear_error(self):
        net = NeuralNetwork((2, 4, 1), learning_rate=1e300, seed=0)
        with self.assertRaises(RuntimeError):
            with np.errstate(all="ignore"):
                net.train(XOR_X, XOR_Y, epochs=50, verbose=False)


class TestGradients(unittest.TestCase):
    """Compare backpropagation to numerical gradients (a classic sanity check)."""

    def check(self, net):
        # Networks start with zero biases, so for the input [0, 0] every hidden
        # pre-activation is exactly 0. That is ReLU's kink, where the slope is
        # undefined and a numerical gradient is meaningless. Non-zero biases
        # avoid that artefact (and test the bias gradients with real values).
        rng = np.random.default_rng(0)
        for bias in net.biases:
            bias[...] = rng.normal(0.0, 0.3, size=bias.shape)

        y = XOR_Y
        activations = net.forward(XOR_X)
        grad_w, grad_b = net.backward(activations, y)

        def loss():
            return net.compute_loss(y, net.forward(XOR_X)[-1])

        h = 1e-6
        for name, params, grads in (("weights", net.weights, grad_w),
                                    ("biases", net.biases, grad_b)):
            for layer, (param, grad) in enumerate(zip(params, grads)):
                self.assertEqual(grad.shape, param.shape)
                numeric = np.zeros_like(param)
                for idx in np.ndindex(param.shape):
                    original = param[idx]
                    param[idx] = original + h
                    plus = loss()
                    param[idx] = original - h
                    minus = loss()
                    param[idx] = original
                    numeric[idx] = (plus - minus) / (2 * h)
                np.testing.assert_allclose(grad, numeric, atol=1e-7,
                                           err_msg=f"{name} layer {layer}")

    def test_two_layer_sigmoid(self):
        self.check(NeuralNetwork(**CONFIG))

    def test_each_activation_in_a_deeper_network(self):
        for activation in ("sigmoid", "tanh", "relu"):
            with self.subTest(activation=activation):
                self.check(NeuralNetwork((2, 5, 3, 1), hidden_activation=activation, seed=3))


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = NeuralNetwork(**CONFIG)
        cls.initial_loss, _ = cls.net.evaluate(XOR_X, XOR_Y)
        cls.net.train(XOR_X, XOR_Y, epochs=EPOCHS, verbose=False)

    def test_loss_history_length(self):
        self.assertEqual(len(self.net.loss_history), EPOCHS)
        self.assertEqual(self.net.epochs_run, EPOCHS)
        self.assertFalse(self.net.stopped_early)

    def test_training_reduces_loss(self):
        final_loss, _ = self.net.evaluate(XOR_X, XOR_Y)
        self.assertLess(final_loss, self.initial_loss)
        self.assertLess(final_loss, 0.05)

    def test_loss_is_finite_and_generally_decreasing(self):
        history = np.array(self.net.loss_history)
        self.assertTrue(np.all(np.isfinite(history)))
        # Compare averages of early vs late training to ignore small wiggles.
        self.assertLess(history[-1000:].mean(), history[:1000].mean())
        self.assertLess(history[-1], history[0])

    def test_predicts_all_xor_combinations(self):
        np.testing.assert_array_equal(self.net.predict(XOR_X), XOR_Y.astype(int))

    def test_accuracy_is_100_percent(self):
        _, accuracy = self.net.evaluate(XOR_X, XOR_Y)
        self.assertEqual(accuracy, 1.0)

    def test_training_changed_all_parameters(self):
        fresh = NeuralNetwork(**CONFIG)
        for layer in range(self.net.n_layers):
            self.assertFalse(np.allclose(self.net.weights[layer], fresh.weights[layer]),
                             f"weights of layer {layer} were not updated")
            self.assertFalse(np.allclose(self.net.biases[layer], fresh.biases[layer]),
                             f"biases of layer {layer} were not updated")

    def test_default_run_reproduces_known_loss(self):
        # Guards against accidental changes to the maths or the random draws.
        self.assertAlmostEqual(self.net.loss_history[-1], 0.0011071721, places=8)


class TestVariants(unittest.TestCase):
    """Other activations, depths, optimizers and batch sizes all learn XOR."""

    def test_tanh(self):
        net = NeuralNetwork((2, 4, 1), hidden_activation="tanh", learning_rate=0.5, seed=42)
        net.train(XOR_X, XOR_Y, epochs=5000, verbose=False)
        self.assertEqual(net.evaluate(XOR_X, XOR_Y)[1], 1.0)

    def test_relu_with_wider_layer(self):
        net = NeuralNetwork((2, 8, 1), hidden_activation="relu", learning_rate=0.1, seed=42)
        net.train(XOR_X, XOR_Y, epochs=5000, verbose=False)
        self.assertEqual(net.evaluate(XOR_X, XOR_Y)[1], 1.0)

    def test_deeper_network(self):
        net = NeuralNetwork((2, 8, 4, 1), learning_rate=0.5, seed=42)
        net.train(XOR_X, XOR_Y, epochs=5000, verbose=False)
        self.assertEqual(net.evaluate(XOR_X, XOR_Y)[1], 1.0)

    def test_momentum_learns_faster_than_plain_gradient_descent(self):
        plain = NeuralNetwork(**CONFIG)
        plain.train(XOR_X, XOR_Y, epochs=3000, verbose=False)
        fast = NeuralNetwork(optimizer="momentum", momentum=0.9, **CONFIG)
        fast.train(XOR_X, XOR_Y, epochs=3000, verbose=False)
        self.assertLess(first_epoch_below(fast.loss_history),
                        first_epoch_below(plain.loss_history))
        self.assertEqual(fast.evaluate(XOR_X, XOR_Y)[1], 1.0)

    def test_momentum_zero_equals_plain_gradient_descent(self):
        plain = NeuralNetwork(**CONFIG)
        plain.train(XOR_X, XOR_Y, epochs=200, verbose=False)
        zero = NeuralNetwork(optimizer="momentum", momentum=0.0, **CONFIG)
        zero.train(XOR_X, XOR_Y, epochs=200, verbose=False)
        np.testing.assert_allclose(plain.loss_history, zero.loss_history)

    def test_mini_batch_learns_and_is_reproducible(self):
        results = []
        for _ in range(2):
            net = NeuralNetwork(**CONFIG)
            net.train(XOR_X, XOR_Y, epochs=3000, verbose=False, batch_size=2)
            results.append(net)
        self.assertEqual(results[0].evaluate(XOR_X, XOR_Y)[1], 1.0)
        np.testing.assert_array_equal(results[0].weights[0], results[1].weights[0])
        self.assertEqual(len(results[0].loss_history), 3000)

    def test_batch_size_larger_than_data_acts_like_full_batch(self):
        full = NeuralNetwork(**CONFIG)
        full.train(XOR_X, XOR_Y, epochs=100, verbose=False)
        big = NeuralNetwork(**CONFIG)
        big.train(XOR_X, XOR_Y, epochs=100, verbose=False, batch_size=1000)
        np.testing.assert_allclose(full.loss_history, big.loss_history)

    def test_all_logic_gates_are_learned(self):
        from datasets import DATASET_NAMES, get_dataset
        for name in DATASET_NAMES:
            X, y = get_dataset(name)
            net = NeuralNetwork(**CONFIG)
            net.train(X, y, epochs=3000, verbose=False)
            self.assertEqual(net.evaluate(X, y)[1], 1.0, msg=name)


class TestEarlyStoppingAndCallback(unittest.TestCase):
    def test_early_stopping_stops_before_the_limit(self):
        net = NeuralNetwork(**CONFIG)
        net.train(XOR_X, XOR_Y, epochs=30000, verbose=False,
                  early_stopping_patience=200, min_delta=1e-4)
        self.assertTrue(net.stopped_early)
        self.assertLess(net.epochs_run, 30000)
        self.assertEqual(len(net.loss_history), net.epochs_run)
        self.assertEqual(net.evaluate(XOR_X, XOR_Y)[1], 1.0)

    def test_final_parameters_match_last_recorded_loss(self):
        net = NeuralNetwork(**CONFIG)
        net.train(XOR_X, XOR_Y, epochs=30000, verbose=False,
                  early_stopping_patience=200, min_delta=1e-4)
        self.assertAlmostEqual(net.evaluate(XOR_X, XOR_Y)[0], net.loss_history[-1], places=12)

    def test_no_early_stopping_by_default(self):
        net = NeuralNetwork(**CONFIG)
        net.train(XOR_X, XOR_Y, epochs=500, verbose=False)
        self.assertFalse(net.stopped_early)
        self.assertEqual(net.epochs_run, 500)

    def test_callback_schedule(self):
        calls = []
        net = NeuralNetwork(**CONFIG)
        net.train(XOR_X, XOR_Y, epochs=10, verbose=False, callback_every=4,
                  callback=lambda epoch, network, loss: calls.append((epoch, network is net, loss)))
        self.assertEqual([c[0] for c in calls], [1, 5, 9])
        self.assertTrue(all(c[1] for c in calls))
        self.assertAlmostEqual(calls[0][2], net.loss_history[0])


class TestSaveAndLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = NeuralNetwork((2, 6, 3, 1), hidden_activation="tanh", learning_rate=0.2,
                                optimizer="momentum", momentum=0.8, seed=5)
        cls.net.train(XOR_X, XOR_Y, epochs=300, verbose=False)

    def test_round_trip_gives_identical_predictions_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.net.save(os.path.join(folder, "model"))
            self.assertTrue(path.endswith(".npz") and os.path.exists(path))
            loaded = NeuralNetwork.load(path)
        np.testing.assert_array_equal(loaded.predict_proba(XOR_X), self.net.predict_proba(XOR_X))
        self.assertEqual(loaded.layer_sizes, (2, 6, 3, 1))
        self.assertEqual(loaded.hidden_activation, "tanh")
        self.assertEqual(loaded.optimizer, "momentum")
        self.assertAlmostEqual(loaded.momentum, 0.8)
        self.assertAlmostEqual(loaded.learning_rate, 0.2)
        self.assertEqual(loaded.seed, 5)
        self.assertEqual(loaded.loss_history, self.net.loss_history)

    def test_load_without_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            self.net.save(os.path.join(folder, "model"))
            loaded = NeuralNetwork.load(os.path.join(folder, "model"))
        self.assertEqual(loaded.layer_sizes, self.net.layer_sizes)

    def test_unseeded_network_round_trips(self):
        net = NeuralNetwork((2, 3, 1))
        with tempfile.TemporaryDirectory() as folder:
            loaded = NeuralNetwork.load(net.save(os.path.join(folder, "m.npz")))
        self.assertIsNone(loaded.seed)

    def test_loaded_network_can_keep_training(self):
        with tempfile.TemporaryDirectory() as folder:
            loaded = NeuralNetwork.load(self.net.save(os.path.join(folder, "m")))
        before = loaded.evaluate(XOR_X, XOR_Y)[0]
        loaded.train(XOR_X, XOR_Y, epochs=200, verbose=False)
        self.assertLess(loaded.evaluate(XOR_X, XOR_Y)[0], before)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            NeuralNetwork.load("definitely_not_here.npz")

    def test_invalid_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.npz")
            np.savez(path, something=np.zeros(3))
            with self.assertRaises(ValueError):
                NeuralNetwork.load(path)

    def test_wrong_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            good = self.net.save(os.path.join(folder, "good"))
            with np.load(good) as data:
                contents = {key: data[key] for key in data.files}
            contents["W0"] = np.zeros((9, 9))
            bad = os.path.join(folder, "bad.npz")
            np.savez(bad, **contents)
            with self.assertRaises(ValueError):
                NeuralNetwork.load(bad)


if __name__ == "__main__":
    unittest.main()
