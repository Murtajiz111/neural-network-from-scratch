"""Gradient checking: prove that backpropagation is correct.

Backpropagation computes every gradient with calculus (the chain rule). A
completely independent way to get the same number is to *measure* it: nudge a
parameter up and down by a tiny amount and see how much the loss changes,

    gradient ~= (loss(w + h) - loss(w - h)) / (2 * h).

If the two answers agree to many decimal places, the backpropagation code is
almost certainly right. If they disagree, there is a bug.
"""

import copy

import numpy as np


def gradient_check(network, X, y, epsilon=1e-6, seed=0):
    """Compare backpropagation with numerically measured gradients.

    The check runs on a COPY of the network (yours is not touched), with
    dropout switched off (random masks would make the loss change between
    evaluations) and with small random non-zero biases (zero biases put ReLU
    exactly on its kink, where its slope is undefined).

    Returns a list of dicts, one per parameter array, with the largest
    absolute and relative difference and a pass/fail flag.
    """
    X = network._check_inputs(X)
    y = network._check_targets(y, X.shape[0])
    checked = copy.deepcopy(network)
    checked.dropout = 0.0
    rng = np.random.default_rng(seed)
    for bias in checked.biases:
        bias[...] = rng.normal(0.0, 0.3, size=bias.shape)

    def loss():
        return checked.total_loss(y, checked.forward(X)[-1])

    grad_weights, grad_biases = checked.backward(checked.forward(X), y)
    results = []
    for kind, params, grads in (("weights", checked.weights, grad_weights),
                                ("biases", checked.biases, grad_biases)):
        for layer, (param, analytic) in enumerate(zip(params, grads)):
            numeric = np.zeros_like(param)
            for index in np.ndindex(param.shape):
                original = param[index]
                param[index] = original + epsilon
                plus = loss()
                param[index] = original - epsilon
                minus = loss()
                param[index] = original
                numeric[index] = (plus - minus) / (2 * epsilon)
            difference = np.abs(analytic - numeric)
            scale = np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-8)
            max_abs = float(difference.max())
            max_rel = float((difference / scale).max())
            results.append(dict(
                parameter=f"{kind} layer {layer + 1}", shape=param.shape,
                max_abs_error=max_abs, max_rel_error=max_rel,
                passed=bool(max_abs < 1e-7 or max_rel < 1e-5)))
    return results


def format_gradient_check(results):
    """Readable text version of :func:`gradient_check` results."""
    lines = ["GRADIENT CHECK: backpropagation vs numerically measured gradients",
             "=" * 66,
             f"{'parameter':<18}{'shape':<12}{'max abs error':>15}{'max rel error':>16}  result",
             "-" * 66]
    for row in results:
        lines.append(f"{row['parameter']:<18}{str(row['shape']):<12}"
                     f"{row['max_abs_error']:>15.2e}{row['max_rel_error']:>16.2e}  "
                     f"{'PASS' if row['passed'] else 'FAIL'}")
    ok = all(row["passed"] for row in results)
    lines.append("-" * 66)
    lines.append("Result: " + ("every gradient matches, so backpropagation is correct."
                               if ok else "MISMATCH: backpropagation disagrees with the numbers."))
    return "\n".join(lines), ok
