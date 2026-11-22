"""Shared helpers that turn a dataset plus a dictionary of settings into a
trained network. Used by main.py, sweep.py and the notebook.

Where a setting comes from (highest priority first):
    1. what you asked for (command-line option, config file, sweep value)
    2. the dataset's own recommended defaults (see datasets.py)
    3. the global defaults below
"""

from dataclasses import dataclass

import numpy as np

from datasets import standardize_split, train_val_test_split
from neural_network import NeuralNetwork

GLOBAL_DEFAULTS = dict(
    hidden=[4], activation="sigmoid", optimizer="gd", lr=None, momentum=0.9,
    epochs=10000, batch_size=None, early_stopping=None, min_delta=1e-6, restore_best=False,
    l2=0.0, dropout=0.0, weight_init="auto", lr_schedule="constant", lr_decay=0.5,
    lr_step=1000, test_size=0.0, val_size=0.0, standardize=False, samples=None,
    seed=42, data_seed=0,
)

# A sensible learning rate for each optimizer when none is given. Adam and
# RMSprop take much smaller steps than plain gradient descent.
OPTIMIZER_LR = {"gd": 0.5, "momentum": 0.5, "rmsprop": 0.01, "adam": 0.01}

SETTING_NAMES = tuple(GLOBAL_DEFAULTS)


def resolve_settings(dataset, overrides=None):
    """Combine overrides, dataset defaults and global defaults into one dict."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(SETTING_NAMES))
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(unknown)}. "
                         f"Valid settings: {', '.join(SETTING_NAMES)}")
    settings = dict(GLOBAL_DEFAULTS)
    settings.update(dataset.defaults)
    default_optimizer = settings["optimizer"]
    explicit = {key: value for key, value in overrides.items() if value is not None}
    settings.update(explicit)
    settings["hidden"] = list(settings["hidden"])

    # If you switch optimizer but do not give a learning rate, use the new
    # optimizer's usual learning rate instead of the dataset's.
    if settings["optimizer"] != default_optimizer and "lr" not in explicit:
        settings["lr"] = OPTIMIZER_LR[settings["optimizer"]]
    if settings["lr"] is None:
        settings["lr"] = OPTIMIZER_LR[settings["optimizer"]]
    return settings


def prepare_data(dataset, settings):
    """Split (and optionally standardize) the data. Returns (split, scaler)."""
    split = train_val_test_split(dataset.X, dataset.y, settings["val_size"],
                                 settings["test_size"], seed=settings["seed"])
    scaler = None
    if settings["standardize"]:
        split, scaler = standardize_split(split)
    return split, scaler


def build_network(dataset, settings):
    return NeuralNetwork(
        layer_sizes=(dataset.input_size, *settings["hidden"], dataset.output_size),
        hidden_activation=settings["activation"],
        output_activation="softmax" if dataset.task == "multiclass" else "sigmoid",
        learning_rate=settings["lr"], optimizer=settings["optimizer"],
        momentum=settings["momentum"], seed=settings["seed"], l2=settings["l2"],
        dropout=settings["dropout"], weight_init=settings["weight_init"],
        lr_schedule=settings["lr_schedule"], lr_decay=settings["lr_decay"],
        lr_step=settings["lr_step"])


def fit(network, split, settings, *, verbose=True, print_every=1000, callback=None,
        early_stopping=True, batch_size="settings"):
    """Train ``network`` on ``split`` using ``settings``."""
    return network.train(
        split.X_train, split.y_train, epochs=settings["epochs"], print_every=print_every,
        verbose=verbose,
        batch_size=settings["batch_size"] if batch_size == "settings" else batch_size,
        early_stopping_patience=settings["early_stopping"] if early_stopping else None,
        min_delta=settings["min_delta"], callback=callback,
        validation_data=split.validation_data, restore_best=settings["restore_best"])


def compute_metrics(network, split):
    """Loss and accuracy on every part of the split that exists."""
    metrics = {}
    for name, X, y in (("train", split.X_train, split.y_train),
                       ("val", split.X_val, split.y_val),
                       ("test", split.X_test, split.y_test)):
        if X is not None:
            loss, accuracy = network.evaluate(X, y)
            metrics[f"{name}_loss"] = loss
            metrics[f"{name}_accuracy"] = accuracy
    return metrics


@dataclass
class RunResult:
    network: NeuralNetwork
    split: object
    scaler: object
    settings: dict
    metrics: dict


def run_experiment(dataset, overrides=None, *, verbose=False, print_every=1000, callback=None):
    """Resolve settings, split, build, train and measure. One call, one result."""
    settings = resolve_settings(dataset, overrides)
    split, scaler = prepare_data(dataset, settings)
    network = build_network(dataset, settings)
    fit(network, split, settings, verbose=verbose, print_every=print_every, callback=callback)
    return RunResult(network, split, scaler, settings, compute_metrics(network, split))


def epochs_to_threshold(history, threshold=0.1):
    """First epoch whose loss is below ``threshold`` (None if never)."""
    for index, value in enumerate(history):
        if value < threshold:
            return index + 1
    return None


def format_table(headers, rows):
    """Return a plain-text table as a string."""
    rows = [[str(item) for item in row] for row in rows]
    widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    out = [line, "-" * len(line)]
    out += [" | ".join(f"{item:<{w}}" for item, w in zip(row, widths)) for row in rows]
    return "\n".join(out)
