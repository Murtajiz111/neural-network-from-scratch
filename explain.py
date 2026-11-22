"""Step-by-step walk-through of ONE forward and backward pass.

Everything is printed with real numbers from the network, so you can follow
the maths with a pencil. Nothing here changes the network: the "after update"
values are only a preview of what gradient descent would do.
"""

import numpy as np


def _fmt(array):
    """Compact, readable array formatting."""
    return np.array2string(np.asarray(array), precision=4, suppress_small=True,
                           floatmode="fixed", max_line_width=100)


def _layer_name(network, layer):
    if layer == network.n_layers - 1:
        return f"Layer {layer + 1} (output, {network.output_activation})"
    return f"Layer {layer + 1} (hidden, {network.hidden_activation})"


def explain_training_step(network, X, y, example_index=0):
    """Return a text walk-through of one training step on a single example."""
    X = network._check_inputs(X)
    y = network._check_targets(y, X.shape[0])
    if not 0 <= example_index < X.shape[0]:
        raise ValueError(f"example_index must be between 0 and {X.shape[0] - 1}")

    x = X[example_index:example_index + 1]      # keep 2-D: shape (1, n_inputs)
    target = y[example_index:example_index + 1]
    lines = []
    add = lines.append

    add("STEP-BY-STEP: ONE TRAINING STEP")
    add("=" * 32)
    add(f"Network : {network.summary()}")
    add(f"Example : input {_fmt(x[0])} | expected output {_fmt(target[0])}")
    add(f"(Using the network's CURRENT weights. Only one example is used, so m = 1.)")
    add("")

    # ---- 1. Forward propagation -----------------------------------------
    add("1) FORWARD PROPAGATION   Z = A_previous @ W + b,   A = activation(Z)")
    add("-" * 70)
    activations = [x]
    for layer in range(network.n_layers):
        activate, _ = network._layer_activation(layer)
        Z = activations[-1] @ network.weights[layer] + network.biases[layer]
        A = activate(Z)
        activations.append(A)
        add(f"{_layer_name(network, layer)}")
        add(f"  W {network.weights[layer].shape}:\n{_indent(_fmt(network.weights[layer]))}")
        add(f"  b {network.biases[layer].shape}: {_fmt(network.biases[layer])}")
        add(f"  Z = {_fmt(Z)}")
        add(f"  A = {_fmt(A)}")
        add("")

    prediction = activations[-1]
    loss = network.compute_loss(target, prediction)
    if network.is_multiclass:
        add("2) LOSS   L = -sum( y * log(p) )   (categorical cross-entropy; y is one-hot)")
    else:
        add("2) LOSS   L = -( y*log(p) + (1-y)*log(1-p) )")
    add("-" * 70)
    add(f"  prediction p = {_fmt(prediction)}   expected y = {_fmt(target)}")
    add(f"  loss = {loss:.6f}   (0 would be perfect; bigger means more wrong)")
    add("")

    # ---- 3. Backpropagation ---------------------------------------------
    add("3) BACKPROPAGATION   work backwards, from the output to the input")
    add("-" * 70)
    grad_weights = [None] * network.n_layers
    grad_biases = [None] * network.n_layers
    dZ = prediction - target
    add(f"  Output error  dZ = A_output - y = {_fmt(dZ)}")
    add("  (Positive: prediction too high. Negative: too low.)")
    add("")
    for layer in reversed(range(network.n_layers)):
        A_prev = activations[layer]
        grad_weights[layer] = A_prev.T @ dZ
        grad_biases[layer] = dZ.copy()
        add(f"{_layer_name(network, layer)}")
        add(f"  dW = A_previous.T @ dZ {grad_weights[layer].shape}:\n{_indent(_fmt(grad_weights[layer]))}")
        add(f"  db = dZ {grad_biases[layer].shape}: {_fmt(grad_biases[layer])}")
        if layer == 0 and not np.any(x):
            add("  Note: dW is all zeros because every input is 0 (dW = input * dZ), so only the")
            add("  biases move for this example. Try --explain-example 4 to see every weight change.")
        if layer > 0:
            _, derivative = network._layer_activation(layer - 1)
            slope = derivative(A_prev)
            dZ = (dZ @ network.weights[layer].T) * slope
            add(f"  Error sent back: dZ_previous = (dZ @ W.T) * slope = {_fmt(dZ)}")
            add(f"    where slope = derivative(A_previous) = {_fmt(slope)}")
        add("")

    # Self-check: the explanation must agree with the real implementation.
    real_w, real_b = network.backward(activations, target)
    agrees = (all(np.allclose(a, b) for a, b in zip(grad_weights, real_w))
              and all(np.allclose(a, b) for a, b in zip(grad_biases, real_b)))
    add(f"  Check: these gradients match NeuralNetwork.backward(): {'yes' if agrees else 'NO'}")
    add("")

    # ---- 4. Gradient descent preview ------------------------------------
    add(f"4) GRADIENT DESCENT   new = old - learning_rate * gradient   "
        f"(learning rate = {network.learning_rate})")
    add("-" * 70)
    for layer in range(network.n_layers):
        new_w = network.weights[layer] - network.learning_rate * grad_weights[layer]
        new_b = network.biases[layer] - network.learning_rate * grad_biases[layer]
        add(f"{_layer_name(network, layer)}")
        add(f"  W before:\n{_indent(_fmt(network.weights[layer]))}")
        add(f"  W after :\n{_indent(_fmt(new_w))}")
        add(f"  b before: {_fmt(network.biases[layer])}   b after: {_fmt(new_b)}")
        add("")

    # What the preview would do to this example's loss.
    preview_activations = x
    for layer in range(network.n_layers):
        activate, _ = network._layer_activation(layer)
        new_w = network.weights[layer] - network.learning_rate * grad_weights[layer]
        new_b = network.biases[layer] - network.learning_rate * grad_biases[layer]
        preview_activations = activate(preview_activations @ new_w + new_b)
    new_loss = network.compute_loss(target, preview_activations)
    add(f"Result: loss for this example would go from {loss:.6f} to {new_loss:.6f}.")
    add("Training repeats these four steps for every example, thousands of times.")
    add("(This preview did not change the network.)")
    return "\n".join(lines)


def _indent(text, spaces=4):
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
