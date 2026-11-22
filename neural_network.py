"""A feedforward neural network implemented from scratch with NumPy.

Features
    * Any number of layers, e.g. layer_sizes=(2, 8, 4, 1).
    * Hidden activations: sigmoid, tanh, relu or leaky_relu.
    * Output layer: sigmoid (yes/no answers, binary cross-entropy) or softmax
      (one of several classes, categorical cross-entropy).
    * Optimizers: gradient descent ("gd"), "momentum", "rmsprop" and "adam",
      with full-batch or mini-batch training.
    * Learning-rate schedules, L2 regularisation, dropout, weight-initialisation
      choices, validation data, early stopping and best-weights restore.
    * Save/load of trained networks.

Shape conventions (m = number of examples, layer l maps size n_l -> n_{l+1}):
    activations[0] = X          : (m, n_0)
    weights[l]                  : (n_l, n_{l+1})
    biases[l]                   : (1, n_{l+1})
    Z_l = activations[l] @ weights[l] + biases[l]
    activations[l+1] = f(Z_l)   : (m, n_{l+1})

With layer_sizes=(2, 4, 1): weights[0] is "W1", weights[1] is "W2", and
activations[1], activations[2] are "A1", "A2" in the usual two-layer notation.
"""

import os

import numpy as np

# Probabilities are clipped to [EPSILON, 1 - EPSILON] before taking logs so
# that log(0) can never happen.
EPSILON = 1e-12
LEAKY_SLOPE = 0.01          # slope of leaky ReLU for negative inputs
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
RMSPROP_DECAY, RMSPROP_EPS = 0.9, 1e-8


# ----------------------------------------------------------------------
# Activation functions.
# Every derivative below is written in terms of the activation OUTPUT ``a``
# (not the raw input ``z``), because that is what forward propagation stores.
# ----------------------------------------------------------------------
def sigmoid(z):
    """Sigmoid activation: 1 / (1 + exp(-z)), always strictly inside (0, 1).

    The input is clipped to [-500, 500] so np.exp never overflows.
    """
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """d sigmoid / dz = a * (1 - a), where a = sigmoid(z)."""
    a = np.asarray(a, dtype=float)
    return a * (1.0 - a)


def tanh(z):
    """Hyperbolic tangent: squashes values into (-1, 1)."""
    return np.tanh(np.asarray(z, dtype=float))


def tanh_derivative(a):
    """d tanh / dz = 1 - a^2, where a = tanh(z)."""
    a = np.asarray(a, dtype=float)
    return 1.0 - a ** 2


def relu(z):
    """Rectified linear unit: max(0, z)."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_derivative(a):
    """d relu / dz = 1 where the output is positive, otherwise 0."""
    a = np.asarray(a, dtype=float)
    return (a > 0.0).astype(float)


def leaky_relu(z):
    """Leaky ReLU: z if z > 0 else 0.01 * z (never completely 'dead')."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0.0, z, LEAKY_SLOPE * z)


def leaky_relu_derivative(a):
    """Slope is 1 where the output is positive, 0.01 otherwise.

    Leaky ReLU keeps the sign of its input, so the output ``a`` is enough to
    tell which side of zero the input was on.
    """
    a = np.asarray(a, dtype=float)
    return np.where(a > 0.0, 1.0, LEAKY_SLOPE)


def softmax(z):
    """Softmax: turns each row of scores into probabilities that sum to 1.

    The row maximum is subtracted first, which does not change the result but
    keeps exp() from overflowing.
    """
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
}
OUTPUT_ACTIVATIONS = ("sigmoid", "softmax")
OPTIMIZERS = ("gd", "momentum", "rmsprop", "adam")
WEIGHT_INITS = ("auto", "xavier", "he", "normal", "small")
LR_SCHEDULES = ("constant", "step", "exponential", "cosine")


# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy: summed over outputs, averaged over examples.

    L = -mean( y*log(p) + (1-y)*log(1-p) )     (for a single output)

    It is small when the predicted probability p is close to the true label y
    and grows very large when the model is confidently wrong.
    """
    p = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    per_example = y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
    return float(-np.mean(np.sum(per_example, axis=-1)))


def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy for one-hot targets: -mean(sum(y * log(p)))."""
    p = np.clip(y_pred, EPSILON, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(p), axis=-1)))


class NeuralNetwork:
    """A fully connected feedforward network."""

    def __init__(self, layer_sizes=(2, 4, 1), hidden_activation="sigmoid",
                 learning_rate=0.5, optimizer="gd", momentum=0.9, seed=None,
                 output_activation="sigmoid", l2=0.0, dropout=0.0,
                 weight_init="auto", lr_schedule="constant", lr_decay=0.5,
                 lr_step=1000):
        # --- validation -------------------------------------------------
        try:
            sizes = list(layer_sizes)
        except TypeError:
            raise ValueError("layer_sizes must be a sequence of positive integers")
        if len(sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and an output size")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError(f"layer sizes must be positive integers, got {size!r}")
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"hidden_activation must be one of {sorted(ACTIVATIONS)}, "
                             f"got {hidden_activation!r}")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {OUTPUT_ACTIVATIONS}, "
                             f"got {output_activation!r}")
        if output_activation == "softmax" and sizes[-1] < 2:
            raise ValueError("softmax output needs at least 2 output neurons (one per class)")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum!r}")
        if l2 < 0:
            raise ValueError(f"l2 must be >= 0, got {l2!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if weight_init not in WEIGHT_INITS:
            raise ValueError(f"weight_init must be one of {WEIGHT_INITS}, got {weight_init!r}")
        if lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule!r}")
        if not 0.0 < lr_decay <= 1.0:
            raise ValueError(f"lr_decay must be in (0, 1], got {lr_decay!r}")
        if not isinstance(lr_step, (int, np.integer)) or lr_step < 1:
            raise ValueError(f"lr_step must be a positive integer, got {lr_step!r}")

        self.layer_sizes = tuple(int(s) for s in sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.weight_init = weight_init
        self.lr_schedule = lr_schedule
        self.lr_decay = float(lr_decay)
        self.lr_step = int(lr_step)
        self.seed = seed

        # A seeded generator makes every run reproducible. It is used for the
        # starting weights and for shuffling mini-batches. Dropout masks use a
        # separate generator so that turning dropout on does not change the
        # starting weights.
        self._rng = np.random.default_rng(seed)
        self._dropout_rng = np.random.default_rng(None if seed is None else seed + 10_000)

        # Weights: random values. They MUST differ from each other ("break
        # symmetry"), otherwise all neurons in a layer would learn the same
        # thing. Biases can safely start at zero.
        self.weights = []
        self.biases = []
        output_layer = len(self.layer_sizes) - 2    # index of the last weight layer
        for layer, (fan_in, fan_out) in enumerate(zip(self.layer_sizes[:-1],
                                                      self.layer_sizes[1:])):
            std = self._initial_std(fan_in, fan_out, is_hidden=layer != output_layer)
            self.weights.append(self._rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        self._current_lr = self.learning_rate
        self.history = {}
        self.loss_history = []
        self.epochs_run = 0
        self.stopped_early = False
        self.best_epoch = None
        self.extra = {}                 # filled by load() with anything save() was given
        self._reset_optimizer_state()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _initial_std(self, fan_in, fan_out, is_hidden):
        """Standard deviation of the random starting weights of one layer.

        auto    : the original behaviour. Standard normal (std 1), except that
                  hidden ReLU / leaky-ReLU layers use He initialisation.
        xavier  : sqrt(2 / (fan_in + fan_out)); good for sigmoid and tanh.
        he      : sqrt(2 / fan_in); good for ReLU.
        normal  : std 1.
        small   : std 0.1.
        """
        if self.weight_init == "xavier":
            return np.sqrt(2.0 / (fan_in + fan_out))
        if self.weight_init == "he":
            return np.sqrt(2.0 / fan_in)
        if self.weight_init == "normal":
            return 1.0
        if self.weight_init == "small":
            return 0.1
        if is_hidden and self.hidden_activation in ("relu", "leaky_relu"):
            return np.sqrt(2.0 / fan_in)
        return 1.0

    @property
    def input_size(self):
        return self.layer_sizes[0]

    @property
    def output_size(self):
        return self.layer_sizes[-1]

    @property
    def n_layers(self):
        """Number of weight layers (hidden layers + output layer)."""
        return len(self.weights)

    @property
    def is_multiclass(self):
        return self.output_activation == "softmax"

    def count_parameters(self):
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def summary(self):
        arrow = " -> ".join(str(s) for s in self.layer_sizes)
        text = (f"{arrow} | hidden activation: {self.hidden_activation}, "
                f"output: {self.output_activation} | {self.count_parameters()} parameters")
        extras = []
        if self.l2 > 0:
            extras.append(f"L2 {self.l2:g}")
        if self.dropout > 0:
            extras.append(f"dropout {self.dropout:g}")
        return text + (" | " + ", ".join(extras) if extras else "")

    def _reset_optimizer_state(self):
        self._step_count = 0
        self._m = {"w": [np.zeros_like(w) for w in self.weights],
                   "b": [np.zeros_like(b) for b in self.biases]}
        self._v = {"w": [np.zeros_like(w) for w in self.weights],
                   "b": [np.zeros_like(b) for b in self.biases]}

    def _layer_activation(self, layer):
        """(function, derivative) used by weight layer ``layer``.

        For a softmax output layer the derivative is None: backpropagation
        starts from the simplified error "prediction minus truth", so it is
        never needed.
        """
        if layer == self.n_layers - 1:
            if self.output_activation == "softmax":
                return softmax, None
            return ACTIVATIONS["sigmoid"]
        return ACTIVATIONS[self.hidden_activation]

    def _lr_at(self, epoch, total_epochs):
        """Learning rate for a 1-based epoch number, following the schedule."""
        passed = epoch - 1
        base = self.learning_rate
        if self.lr_schedule == "step":
            return base * self.lr_decay ** (passed // self.lr_step)
        if self.lr_schedule == "exponential":
            return base * self.lr_decay ** (passed / self.lr_step)
        if self.lr_schedule == "cosine":
            progress = passed / max(1, total_epochs - 1)
            return base * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress)))
        return base

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _check_inputs(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                f"X must have shape (m, {self.input_size}), got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X

    def _check_targets(self, y, m):
        """Return targets as an (m, output_size) matrix.

        Softmax networks also accept a 1-D array of integer class labels,
        which is converted to one-hot rows.
        """
        y = np.asarray(y, dtype=float)
        if self.is_multiclass:
            if y.ndim == 1:
                if y.shape[0] != m:
                    raise ValueError(f"y must have {m} labels, got {y.shape[0]}")
                labels = y.astype(int)
                if np.any(labels != y) or np.any((labels < 0) | (labels >= self.output_size)):
                    raise ValueError(
                        f"class labels must be integers from 0 to {self.output_size - 1}")
                return np.eye(self.output_size)[labels]
            if y.shape != (m, self.output_size):
                raise ValueError(
                    f"y must have shape ({m}, {self.output_size}) or be {m} integer labels, "
                    f"got {y.shape}")
            if np.any((y < 0) | (y > 1)) or not np.allclose(y.sum(axis=1), 1.0):
                raise ValueError("one-hot targets must be between 0 and 1 and sum to 1 per row")
            return y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape != (m, self.output_size):
            raise ValueError(
                f"y must have shape ({m}, {self.output_size}), got {y.shape}")
        if np.any((y < 0) | (y > 1)):
            raise ValueError("y values must be between 0 and 1")
        return y

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, X):
        """Run inputs through the network (no dropout).

        Returns the list [X, A1, A2, ..., A_last]. The last item holds the
        output probabilities; the others are kept for backpropagation.
        """
        activations = [X]
        for layer in range(self.n_layers):
            activate, _ = self._layer_activation(layer)
            Z = activations[-1] @ self.weights[layer] + self.biases[layer]
            activations.append(activate(Z))
        return activations

    def _forward_train(self, X):
        """Forward pass with dropout (used only while training).

        Returns (used, raw, masks):
          used  - the values actually fed to the next layer (after dropout),
          raw   - the plain activations before dropout (their slopes are needed),
          masks - the dropout masks (already divided by the keep probability).
        """
        keep = 1.0 - self.dropout
        used, raw, masks = [X], [X], [None]
        for layer in range(self.n_layers):
            activate, _ = self._layer_activation(layer)
            Z = used[-1] @ self.weights[layer] + self.biases[layer]
            A = activate(Z)
            raw.append(A)
            if layer < self.n_layers - 1 and self.dropout > 0:
                mask = (self._dropout_rng.random(A.shape) < keep) / keep
                used.append(A * mask)
                masks.append(mask)
            else:
                used.append(A)
                masks.append(None)
        return used, raw, masks

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, y_true, y_pred):
        """Cross-entropy between expected and predicted values (no penalty)."""
        if self.is_multiclass:
            return categorical_cross_entropy(y_true, y_pred)
        return binary_cross_entropy(y_true, y_pred)

    def l2_penalty(self):
        """0.5 * l2 * (sum of all squared weights). Biases are not penalised."""
        if self.l2 == 0:
            return 0.0
        return float(0.5 * self.l2 * sum(np.sum(w ** 2) for w in self.weights))

    def total_loss(self, y_true, y_pred):
        """The quantity training minimises: cross-entropy plus the L2 penalty."""
        return self.compute_loss(y_true, y_pred) + self.l2_penalty()

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def backward(self, activations, y, raw=None, masks=None):
        """Gradients of the total loss for every layer.

        ``raw`` and ``masks`` are only needed when dropout was used in the
        forward pass. Returns (grad_weights, grad_biases) in layer order.
        """
        m = y.shape[0]
        raw = activations if raw is None else raw
        grad_weights = [None] * self.n_layers
        grad_biases = [None] * self.n_layers

        # Output layer: sigmoid + binary cross-entropy, and softmax +
        # categorical cross-entropy, both simplify to "prediction minus truth".
        dZ = activations[-1] - y                                  # (m, n_out)

        for layer in reversed(range(self.n_layers)):
            A_prev = activations[layer]
            grad_weights[layer] = A_prev.T @ dZ / m               # same shape as W
            if self.l2 > 0:
                grad_weights[layer] = grad_weights[layer] + self.l2 * self.weights[layer]
            grad_biases[layer] = np.sum(dZ, axis=0, keepdims=True) / m
            if layer > 0:
                # Send the error back through the weights, undo dropout, then
                # multiply by the slope of the previous layer's activation.
                dA = dZ @ self.weights[layer].T
                if masks is not None and masks[layer] is not None:
                    dA = dA * masks[layer]
                _, derivative = self._layer_activation(layer - 1)
                dZ = dA * derivative(raw[layer])

        return grad_weights, grad_biases

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------
    def _optimizer_step(self, kind, layer, gradient):
        """Direction to move one parameter array, depending on the optimizer."""
        if self.optimizer == "momentum":
            velocity = self._m[kind][layer] = self.momentum * self._m[kind][layer] + gradient
            return velocity
        if self.optimizer == "rmsprop":
            average = self._v[kind][layer] = (RMSPROP_DECAY * self._v[kind][layer]
                                              + (1.0 - RMSPROP_DECAY) * gradient ** 2)
            return gradient / (np.sqrt(average) + RMSPROP_EPS)
        if self.optimizer == "adam":
            t = self._step_count
            first = self._m[kind][layer] = (ADAM_BETA1 * self._m[kind][layer]
                                            + (1.0 - ADAM_BETA1) * gradient)
            second = self._v[kind][layer] = (ADAM_BETA2 * self._v[kind][layer]
                                             + (1.0 - ADAM_BETA2) * gradient ** 2)
            first_hat = first / (1.0 - ADAM_BETA1 ** t)      # bias correction
            second_hat = second / (1.0 - ADAM_BETA2 ** t)
            return first_hat / (np.sqrt(second_hat) + ADAM_EPS)
        return gradient

    def update_parameters(self, grad_weights, grad_biases):
        """Move every parameter a small step against its gradient.

        gd:        param -= lr * gradient
        momentum:  velocity = momentum * velocity + gradient;  param -= lr * velocity
        rmsprop:   divide each gradient by a running average of its recent size
        adam:      momentum + rmsprop together, with bias correction
        ``lr`` is the learning rate for the current epoch (see lr_schedule).
        """
        self._step_count += 1
        for kind, params, grads in (("w", self.weights, grad_weights),
                                    ("b", self.biases, grad_biases)):
            for layer in range(self.n_layers):
                params[layer] -= self._current_lr * self._optimizer_step(kind, layer, grads[layer])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, X, y, epochs=10000, print_every=1000, verbose=True,
              batch_size=None, early_stopping_patience=None, min_delta=1e-6,
              callback=None, callback_every=1, validation_data=None,
              restore_best=False):
        """Train the network.

        batch_size              None = full batch (all rows every step);
                                otherwise rows are shuffled each epoch and
                                processed in chunks of this size.
        early_stopping_patience Stop when the monitored loss has not improved
                                by at least ``min_delta`` for this many epochs.
                                None turns early stopping off. The monitored
                                loss is the validation loss if validation_data
                                is given, otherwise the training loss.
        validation_data         Optional (X_val, y_val), used only for
                                monitoring, never for learning.
        restore_best            At the end, put back the weights from the epoch
                                with the lowest monitored loss.
        callback                Optional function(epoch, network, loss) called
                                every ``callback_every`` epochs (starting with
                                epoch 1), before that epoch's update.

        ``history`` records, per epoch and measured at the START of the epoch
        (before that epoch's updates): loss, accuracy, val_loss, val_accuracy,
        lr, weight_norm (per layer) and grad_norm (per layer; for mini-batches
        the last batch). ``loss_history`` is ``history["loss"]``: the whole-
        dataset training loss including any L2 penalty. The history is also
        returned as ``loss_history``.
        """
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        if batch_size is not None and (not isinstance(batch_size, (int, np.integer))
                                       or batch_size < 1):
            raise ValueError(f"batch_size must be a positive integer or None, got {batch_size!r}")
        if early_stopping_patience is not None and (
                not isinstance(early_stopping_patience, (int, np.integer))
                or early_stopping_patience < 1):
            raise ValueError("early_stopping_patience must be a positive integer or None")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta!r}")
        if callback_every < 1:
            raise ValueError(f"callback_every must be >= 1, got {callback_every!r}")

        X = self._check_inputs(X)
        y = self._check_targets(y, X.shape[0])
        m = X.shape[0]
        step_size = m if batch_size is None else min(int(batch_size), m)

        X_val = y_val = None
        if validation_data is not None:
            X_val = self._check_inputs(validation_data[0])
            y_val = self._check_targets(validation_data[1], X_val.shape[0])

        self.history = {key: [] for key in ("loss", "accuracy", "val_loss", "val_accuracy",
                                            "lr", "weight_norm", "grad_norm")}
        self.loss_history = self.history["loss"]
        self.epochs_run = 0
        self.stopped_early = False
        self.best_epoch = None
        self._reset_optimizer_state()
        best_loss = np.inf                  # for early stopping (needs min_delta)
        epochs_without_improvement = 0
        best_monitored = np.inf             # for restore_best (any improvement)
        snapshot = None
        last_grad_norm = [0.0] * self.n_layers

        for epoch in range(1, epochs + 1):
            self._current_lr = self._lr_at(epoch, epochs)

            # 1. Forward pass on all data: measure the loss for this epoch.
            full_activations = self.forward(X)
            predictions = full_activations[-1]
            loss = self.total_loss(y, predictions)
            if not np.isfinite(loss):
                raise RuntimeError(
                    f"Loss became {loss} at epoch {epoch}. The learning rate "
                    f"({self._current_lr}) is probably too high; try a smaller one.")
            self.history["loss"].append(loss)
            self.history["accuracy"].append(self._accuracy(predictions, y))
            self.history["lr"].append(float(self._current_lr))
            self.history["weight_norm"].append([float(np.linalg.norm(w)) for w in self.weights])
            monitored = loss
            if X_val is not None:
                val_predictions = self.forward(X_val)[-1]
                val_loss = self.compute_loss(y_val, val_predictions)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(self._accuracy(val_predictions, y_val))
                monitored = val_loss
            self.epochs_run = epoch

            if callback is not None and (epoch - 1) % callback_every == 0:
                callback(epoch, self, loss)
            if verbose and print_every and epoch % print_every == 0:
                line = f"Epoch {epoch:>6} | Loss: {loss:.6f}"
                if X_val is not None:
                    line += (f" | Val loss: {self.history['val_loss'][-1]:.6f}"
                             f" | Val acc: {self.history['val_accuracy'][-1] * 100:.1f}%")
                print(line)

            # 2. Remember the best weights so far, and check early stopping.
            #    Both happen BEFORE updating, so the weights that produced the
            #    last recorded loss are exactly the ones in the network.
            if restore_best and monitored < best_monitored:
                best_monitored = monitored
                self.best_epoch = epoch
                snapshot = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
            if early_stopping_patience is not None:
                if monitored < best_loss - min_delta:
                    best_loss = monitored
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        self.stopped_early = True
                        if verbose:
                            what = "validation loss" if X_val is not None else "loss"
                            print(f"Early stopping at epoch {epoch}: {what} did not improve "
                                  f"by {min_delta:g} for {early_stopping_patience} epochs.")
                        self.history["grad_norm"].append(list(last_grad_norm))
                        break

            # 3. Learn: gradients and parameter updates.
            if step_size == m:
                # Full batch. Without dropout, reuse the forward pass above.
                if self.dropout > 0:
                    used, raw, masks = self._forward_train(X)
                    grads = self.backward(used, y, raw, masks)
                else:
                    grads = self.backward(full_activations, y)
                last_grad_norm = [float(np.linalg.norm(g)) for g in grads[0]]
                self.update_parameters(*grads)
            else:
                order = self._rng.permutation(m)
                for start in range(0, m, step_size):
                    rows = order[start:start + step_size]
                    if self.dropout > 0:
                        used, raw, masks = self._forward_train(X[rows])
                        grads = self.backward(used, y[rows], raw, masks)
                    else:
                        grads = self.backward(self.forward(X[rows]), y[rows])
                    last_grad_norm = [float(np.linalg.norm(g)) for g in grads[0]]
                    self.update_parameters(*grads)
            self.history["grad_norm"].append(list(last_grad_norm))

        if restore_best and snapshot is not None:
            self.weights = [w.copy() for w in snapshot[0]]
            self.biases = [b.copy() for b in snapshot[1]]

        return self.loss_history

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------
    def _accuracy(self, probabilities, y):
        if self.is_multiclass:
            return float(np.mean(np.argmax(probabilities, axis=1) == np.argmax(y, axis=1)))
        return float(np.mean((probabilities >= 0.5).astype(int) == y))

    def predict_proba(self, X):
        """Return probabilities (values between 0 and 1).

        Shape (m, 1) for a sigmoid output; (m, classes) with rows summing to 1
        for a softmax output.
        """
        X = self._check_inputs(X)
        return self.forward(X)[-1]

    def predict(self, X, threshold=0.5):
        """Return final predictions.

        Sigmoid output: 0 or 1 with shape (m, 1). Softmax output: the most
        likely class number with shape (m,).
        """
        probabilities = self.predict_proba(X)
        if self.is_multiclass:
            return np.argmax(probabilities, axis=1)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y):
        """Return (loss, accuracy) on the given data.

        The loss is the plain cross-entropy (without any L2 penalty).
        """
        X = self._check_inputs(X)
        y = self._check_targets(y, X.shape[0])
        probabilities = self.forward(X)[-1]
        return self.compute_loss(y, probabilities), self._accuracy(probabilities, y)

    def confusion_matrix(self, X, y):
        """Rows = true class, columns = predicted class (softmax networks)."""
        if not self.is_multiclass:
            raise ValueError("confusion_matrix is only available for softmax networks")
        X = self._check_inputs(X)
        y = self._check_targets(y, X.shape[0])
        truth = np.argmax(y, axis=1)
        predicted = self.predict(X)
        matrix = np.zeros((self.output_size, self.output_size), dtype=int)
        np.add.at(matrix, (truth, predicted), 1)
        return matrix

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save(self, path, extra=None):
        """Save the learned parameters and settings to a .npz file.

        ``extra`` is an optional dictionary of additional arrays or numbers to
        store alongside (for example the input scaler). It comes back as
        ``network.extra`` after :meth:`load`. Returns the path actually written
        (".npz" is added if missing).
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        arrays = {}
        for layer in range(self.n_layers):
            arrays[f"W{layer}"] = self.weights[layer]
            arrays[f"b{layer}"] = self.biases[layer]
        for name, value in (extra or {}).items():
            if not str(name).isidentifier():
                raise ValueError(f"extra names must be simple identifiers, got {name!r}")
            arrays[f"extra_{name}"] = np.asarray(value)
        np.savez(
            path,
            layer_sizes=np.array(self.layer_sizes),
            hidden_activation=np.array(self.hidden_activation),
            output_activation=np.array(self.output_activation),
            learning_rate=np.array(self.learning_rate),
            optimizer=np.array(self.optimizer),
            momentum=np.array(self.momentum),
            l2=np.array(self.l2),
            dropout=np.array(self.dropout),
            weight_init=np.array(self.weight_init),
            lr_schedule=np.array(self.lr_schedule),
            lr_decay=np.array(self.lr_decay),
            lr_step=np.array(self.lr_step),
            seed=np.array(-1 if self.seed is None else self.seed),
            loss_history=np.array(self.loss_history, dtype=float),
            **arrays,
        )
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a network saved with :meth:`save`.

        Files written by the first version of this project (without the newer
        settings) still load; the missing settings take their defaults.
        """
        path = str(path)
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path += ".npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path!r}")

        # allow_pickle=False: the file can only contain plain arrays.
        with np.load(path, allow_pickle=False) as data:
            try:
                def optional(name, convert, default):
                    return convert(data[name]) if name in data.files else default

                seed = int(data["seed"])
                network = cls(
                    layer_sizes=tuple(int(s) for s in data["layer_sizes"]),
                    hidden_activation=str(data["hidden_activation"]),
                    output_activation=optional("output_activation", str, "sigmoid"),
                    learning_rate=float(data["learning_rate"]),
                    optimizer=str(data["optimizer"]),
                    momentum=float(data["momentum"]),
                    l2=optional("l2", float, 0.0),
                    dropout=optional("dropout", float, 0.0),
                    weight_init=optional("weight_init", str, "auto"),
                    lr_schedule=optional("lr_schedule", str, "constant"),
                    lr_decay=optional("lr_decay", float, 0.5),
                    lr_step=optional("lr_step", int, 1000),
                    seed=None if seed < 0 else seed,
                )
                for layer in range(network.n_layers):
                    weights, biases = data[f"W{layer}"], data[f"b{layer}"]
                    if (weights.shape != network.weights[layer].shape
                            or biases.shape != network.biases[layer].shape):
                        raise ValueError(f"saved parameters for layer {layer} have the wrong shape")
                    network.weights[layer] = weights.astype(float)
                    network.biases[layer] = biases.astype(float)
                network.loss_history = [float(v) for v in data["loss_history"]]
                network.history = {"loss": network.loss_history}
                network.extra = {name[len("extra_"):]: data[name].copy()
                                 for name in data.files if name.startswith("extra_")}
            except KeyError as error:
                raise ValueError(f"{path!r} is not a valid saved network (missing {error})")
        network.epochs_run = len(network.loss_history)
        network._reset_optimizer_state()
        return network
