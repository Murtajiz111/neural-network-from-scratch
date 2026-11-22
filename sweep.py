"""Hyperparameter sweeps and multi-seed reliability reports.

A *hyperparameter* is a setting you choose before training (learning rate,
layer sizes, activation, ...). A sweep trains one network for every combination
you list and ranks them. A reliability report trains the SAME configuration
with many different random seeds, because one lucky (or unlucky) run can be
misleading.
"""

import itertools

import numpy as np

from experiment import epochs_to_threshold, resolve_settings, run_experiment

# Settings that can appear in a sweep, with how to read their values from text.
def _hidden(text):
    return [int(part) for part in str(text).split("-")]


def _optional_int(text):
    return None if str(text).lower() == "none" else int(text)


SWEEP_PARSERS = {
    "lr": float, "hidden": _hidden, "activation": str, "optimizer": str,
    "batch_size": _optional_int, "l2": float, "dropout": float, "weight_init": str,
    "epochs": int, "momentum": float, "lr_schedule": str,
}


def parse_grid(tokens):
    """Turn ['lr=0.1,0.5', 'hidden=4,8-4'] into {'lr': [0.1, 0.5], 'hidden': [[4], [8, 4]]}.

    A hidden value like ``8-4`` means two hidden layers of 8 and 4 neurons.
    """
    grid = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"sweep entries look like name=value1,value2 (got {token!r})")
        name, _, values = token.partition("=")
        name = name.strip().replace("-", "_")
        if name not in SWEEP_PARSERS:
            raise ValueError(f"cannot sweep {name!r}. Choose from: {', '.join(SWEEP_PARSERS)}")
        if name in grid:
            raise ValueError(f"{name!r} appears twice in the sweep")
        try:
            parsed = [SWEEP_PARSERS[name](value) for value in values.split(",") if value != ""]
        except ValueError:
            raise ValueError(f"could not read the values in {token!r}")
        if not parsed:
            raise ValueError(f"no values given for {name!r}")
        grid[name] = parsed
    if not grid:
        raise ValueError("the sweep needs at least one name=values entry")
    return grid


def _score(metrics):
    """The number runs are ranked by: validation accuracy if there is a
    validation set, otherwise training accuracy."""
    return metrics.get("val_accuracy", metrics["train_accuracy"])


def run_sweep(dataset, base_overrides, grid, seeds=1, first_seed=None):
    """Train every combination in ``grid`` for each seed and return ranked rows."""
    base = dict(base_overrides or {})
    start = base.get("seed") if first_seed is None else first_seed
    start = 42 if start is None else start
    names = list(grid)
    rows = []
    for combination in itertools.product(*(grid[name] for name in names)):
        chosen = dict(zip(names, combination))
        scores, trains, tests, losses = [], [], [], []
        for offset in range(seeds):
            result = run_experiment(dataset, {**base, **chosen, "seed": start + offset})
            scores.append(_score(result.metrics))
            trains.append(result.metrics["train_accuracy"])
            losses.append(result.metrics["train_loss"])
            if "test_accuracy" in result.metrics:
                tests.append(result.metrics["test_accuracy"])
        rows.append(dict(
            settings=chosen, score=float(np.mean(scores)), train_accuracy=float(np.mean(trains)),
            test_accuracy=float(np.mean(tests)) if tests else None,
            train_loss=float(np.mean(losses)), runs=seeds))
    rows.sort(key=lambda row: (-row["score"], row["train_loss"]))
    return rows


def reliability_report(dataset, overrides, n_seeds, first_seed=None):
    """Train the same configuration with ``n_seeds`` different seeds."""
    first = overrides.get("seed") if first_seed is None else first_seed
    first = 42 if first is None else first
    settings = resolve_settings(dataset, overrides)
    train_acc, test_acc, val_acc, losses, speed = [], [], [], [], []
    for offset in range(n_seeds):
        result = run_experiment(dataset, {**overrides, "seed": first + offset})
        train_acc.append(result.metrics["train_accuracy"])
        losses.append(result.metrics["train_loss"])
        if "val_accuracy" in result.metrics:
            val_acc.append(result.metrics["val_accuracy"])
        if "test_accuracy" in result.metrics:
            test_acc.append(result.metrics["test_accuracy"])
        speed.append(epochs_to_threshold(result.network.loss_history))
    reached = [s for s in speed if s is not None]
    return dict(
        seeds=n_seeds, first_seed=first, settings=settings,
        solved=int(sum(a == 1.0 for a in train_acc)),
        train_accuracy=np.array(train_acc), val_accuracy=np.array(val_acc),
        test_accuracy=np.array(test_acc), final_loss=np.array(losses),
        median_epochs_to_threshold=float(np.median(reached)) if reached else None,
        reached_threshold=len(reached))
