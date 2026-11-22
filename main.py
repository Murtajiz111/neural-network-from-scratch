"""Train a neural network from scratch (NumPy only).

Run from the project root. With no options it trains the classic XOR network
(2 -> 4 -> 1, sigmoid, gradient descent):

    python main.py

Examples (see `python main.py --help` for every option):

    python main.py --dataset all                      # the four logic gates
    python main.py --dataset digits                   # 10-class problem, train/val/test split
    python main.py --dataset spirals --animate        # GIF of the boundary forming
    python main.py --optimizer adam --lr 0.01 --l2 0.001 --dropout 0.1
    python main.py --gradient-check                   # prove backpropagation is correct
    python main.py --sweep lr=0.1,0.5 hidden=4,8      # try every combination
    python main.py --seeds-report 20                  # reliability over 20 random seeds
    python main.py --explain                          # one training step with real numbers
    python main.py --config settings.json --log-csv outputs/run.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from datasets import (ALL_DATASET_NAMES, DATASET_NAMES, Standardizer, load_dataset,
                      train_val_test_split)
from experiment import (OPTIMIZER_LR, SETTING_NAMES, build_network, compute_metrics,
                        epochs_to_threshold, fit, format_table, prepare_data,
                        resolve_settings, run_experiment)
from explain import explain_training_step
from gradcheck import format_gradient_check, gradient_check
from neural_network import (ACTIVATIONS, LR_SCHEDULES, OPTIMIZERS, WEIGHT_INITS,
                            NeuralNetwork)
from sweep import SWEEP_PARSERS, parse_grid, reliability_report, run_sweep

LOSS_THRESHOLD = 0.1      # "learned it" level used in the comparison tables
MINI_BATCH_DEMO = 2       # batch size used by --compare-optimizers
CONFIG_ONLY_KEYS = ("dataset", "print_every", "output_dir", "no_plots", "animate",
                    "plot_stats", "log_csv")

OPTIMIZER_NAMES = {"gd": "Gradient Descent", "momentum": "Gradient Descent + Momentum",
                   "rmsprop": "RMSprop", "adam": "Adam"}


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _non_negative_float(text):
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def build_parser():
    """All options default to None so we can tell "not given" from "given".

    Real defaults come from the config file, the dataset's recommended
    settings, and finally the global defaults (see experiment.py).
    """
    parser = argparse.ArgumentParser(
        description="Neural network from scratch (NumPy). Options you leave out use "
                    "sensible defaults for the chosen dataset.")

    data = parser.add_argument_group("data and architecture")
    data.add_argument("--dataset", choices=ALL_DATASET_NAMES + ("all",),
                      help="what to learn: a logic gate (xor, and, or, nand), or moons, circles, "
                           "blobs, spirals, digits. 'all' trains one network per logic gate. "
                           "[default: xor]")
    data.add_argument("--samples", type=_positive_int, metavar="N",
                      help="number of rows to generate (synthetic datasets only)")
    data.add_argument("--data-seed", type=int, metavar="SEED",
                      help="seed for generating the synthetic data [default: 0]")
    data.add_argument("--hidden", type=_positive_int, nargs="+", metavar="N",
                      help="hidden layer sizes, e.g. '--hidden 8 4' for two hidden layers "
                           "[default: 4 for gates]")
    data.add_argument("--activation", choices=sorted(ACTIVATIONS),
                      help="hidden-layer activation (the output layer is sigmoid for 2-class "
                           "problems and softmax for more classes)")
    data.add_argument("--test-size", type=_non_negative_float, metavar="FRACTION",
                      help="share of rows held out for the final test (e.g. 0.2)")
    data.add_argument("--val-size", type=_non_negative_float, metavar="FRACTION",
                      help="share of rows held out to monitor training (e.g. 0.2)")
    data.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=None,
                      help="rescale inputs to mean 0, std 1 (fitted on the training rows only)")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=_positive_int)
    train.add_argument("--lr", type=_positive_float, help="learning rate [default depends on "
                       f"the optimizer: {OPTIMIZER_LR}]")
    train.add_argument("--optimizer", choices=OPTIMIZERS,
                       help="gd = gradient descent, momentum, rmsprop, adam")
    train.add_argument("--momentum", type=float,
                       help="momentum coefficient in [0, 1) [default: 0.9]")
    train.add_argument("--batch-size", type=_positive_int,
                       help="mini-batch size [default: full batch]")
    train.add_argument("--lr-schedule", choices=LR_SCHEDULES,
                       help="how the learning rate changes over time [default: constant]")
    train.add_argument("--lr-decay", type=float,
                       help="factor used by step/exponential schedules, in (0, 1] [default: 0.5]")
    train.add_argument("--lr-step", type=_positive_int,
                       help="epochs per decay step [default: 1000]")
    train.add_argument("--l2", type=_non_negative_float,
                       help="L2 regularisation strength (penalises large weights) [default: 0]")
    train.add_argument("--dropout", type=_non_negative_float,
                       help="share of hidden neurons randomly switched off while training, "
                            "in [0, 1) [default: 0]")
    train.add_argument("--weight-init", choices=WEIGHT_INITS,
                       help="how starting weights are drawn [default: auto]")
    train.add_argument("--early-stopping", type=_positive_int, metavar="PATIENCE",
                       help="stop if the monitored loss has not improved for this many epochs "
                            "(validation loss when --val-size is set, else training loss)")
    train.add_argument("--min-delta", type=_non_negative_float,
                       help="smallest loss decrease that counts as an improvement [default: 1e-6]")
    train.add_argument("--restore-best", action="store_true", default=None,
                       help="finish with the weights from the epoch with the lowest monitored loss")
    train.add_argument("--seed", type=int, help="random seed [default: 42]")
    train.add_argument("--print-every", type=_non_negative_int,
                       help="print the loss every N epochs, 0 = never [default: epochs / 10]")

    tools = parser.add_argument_group("saving, loading and learning aids")
    tools.add_argument("--save", metavar="PATH", help="save the trained model (.npz) to PATH")
    tools.add_argument("--load", metavar="PATH",
                       help="load a saved model instead of training, then evaluate it")
    tools.add_argument("--config", metavar="FILE",
                       help="read settings from a JSON file (options on the command line win)")
    tools.add_argument("--log-csv", metavar="FILE",
                       help="write loss, accuracy, learning rate, weight and gradient sizes "
                            "for every epoch to a CSV file")
    tools.add_argument("--explain", action="store_true", default=None,
                       help="print one forward + backward pass with real numbers from the "
                            "untrained (or --load) weights, then exit without training")
    tools.add_argument("--explain-example", type=_positive_int, metavar="ROW",
                       help="which training row --explain uses [default: row 4 for gates, else 1]")
    tools.add_argument("--gradient-check", action="store_true", default=None,
                       help="compare backpropagation with numerically measured gradients")
    tools.add_argument("--sweep", nargs="+", metavar="NAME=V1,V2",
                       help="train every combination, e.g. --sweep lr=0.01,0.1 hidden=4,8-4 "
                            f"(names: {', '.join(SWEEP_PARSERS)}); 8-4 means two hidden layers")
    tools.add_argument("--sweep-seeds", type=_positive_int, metavar="N", default=None,
                       help="seeds per sweep combination [default: 1]")
    tools.add_argument("--seeds-report", type=_positive_int, metavar="N",
                       help="train the same configuration with N different seeds and summarise")
    tools.add_argument("--compare-optimizers", action="store_true", default=None,
                       help="also compare gd, momentum, rmsprop, adam and mini-batches")
    tools.add_argument("--compare-activations", action="store_true", default=None,
                       help="also compare sigmoid, tanh, relu and leaky_relu")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", help="folder for graphs and animations [default: outputs]")
    out.add_argument("--no-plots", action="store_true", default=None,
                     help="do not create any graphs")
    out.add_argument("--animate", action="store_true", default=None,
                     help="save a GIF of the decision boundary forming (2-input datasets)")
    out.add_argument("--plot-stats", action="store_true", default=None,
                     help="also plot weight sizes, gradient sizes and learning rate over time")
    return parser


def load_config(path):
    """Read a JSON settings file into a dictionary with normalised key names."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"config file not found: {path!r}")
    except json.JSONDecodeError as error:
        raise ValueError(f"config file {path!r} is not valid JSON: {error}")
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path!r} must contain a JSON object like {{\"lr\": 0.1}}")
    allowed = set(SETTING_NAMES) | set(CONFIG_ONLY_KEYS)
    config = {}
    for key, value in raw.items():
        name = str(key).lower().replace("-", "_")
        if name not in allowed:
            raise ValueError(f"unknown key {key!r} in {path!r}. Allowed keys: "
                             f"{', '.join(sorted(allowed))}")
        config[name] = value
    if isinstance(config.get("hidden"), int):
        config["hidden"] = [config["hidden"]]
    return config


def validate_args(parser, args):
    """Reject option combinations that make no sense."""
    modes = [flag for flag in ("explain", "gradient_check", "sweep", "seeds_report")
             if getattr(args, flag)]
    if len(modes) > 1:
        parser.error("choose only one of --explain, --gradient-check, --sweep, --seeds-report")
    if args.dataset == "all":
        for flag in ("save", "load", "explain", "gradient_check", "sweep", "seeds_report",
                     "compare_optimizers", "compare_activations", "log_csv", "animate"):
            if getattr(args, flag):
                parser.error(f"--{flag.replace('_', '-')} needs one dataset, not '--dataset all'")
    if args.load:
        for flag in ("compare_optimizers", "compare_activations", "animate", "sweep",
                     "seeds_report", "gradient_check", "log_csv", "plot_stats"):
            if getattr(args, flag):
                parser.error(f"--{flag.replace('_', '-')} needs training, so it cannot be "
                             f"combined with --load")
    if args.sweep_seeds is not None and not args.sweep:
        parser.error("--sweep-seeds only makes sense together with --sweep")

    # Range checks, so a bad value is reported as a usage error (exit code 2)
    # before any training starts.
    if args.momentum is not None and not 0.0 <= args.momentum < 1.0:
        parser.error("--momentum must be in [0, 1)")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.lr_decay is not None and not 0.0 < args.lr_decay <= 1.0:
        parser.error("--lr-decay must be in (0, 1]")
    for name in ("test_size", "val_size"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.explain_example is not None and not args.explain:
        parser.error("--explain-example only makes sense together with --explain")
    is_gate_run = args.dataset in DATASET_NAMES or (args.dataset is None and not args.config)
    if args.explain_example is not None and is_gate_run and args.explain_example > 4:
        parser.error("--explain-example must be between 1 and 4 for the logic gates")


# ----------------------------------------------------------------------
# Resolving settings
# ----------------------------------------------------------------------
class Options:
    """Command line + config file, looked up by name (command line wins)."""

    def __init__(self, args, config):
        self.args, self.config = args, config

    def get(self, name, default=None):
        value = getattr(self.args, name, None)
        if value is None:
            value = self.config.get(name)
        return default if value is None else value

    def explicit(self, name):
        return getattr(self.args, name, None) is not None or name in self.config

    def overrides(self):
        """Training/data settings that were explicitly given."""
        return {name: self.get(name) for name in SETTING_NAMES if self.explicit(name)}


# ----------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------
def describe_optimizer(settings):
    text = OPTIMIZER_NAMES[settings["optimizer"]]
    if settings["optimizer"] == "momentum":
        text += f" ({settings['momentum']})"
    batch = "full batch" if settings["batch_size"] is None else f"mini-batch of {settings['batch_size']}"
    text += f", {batch}, learning rate {settings['lr']:g}"
    if settings["lr_schedule"] != "constant":
        text += f", {settings['lr_schedule']} schedule"
    return text


def print_header(dataset, settings, split):
    sizes = (dataset.input_size, *settings["hidden"], dataset.output_size)
    output = "Softmax" if dataset.task == "multiclass" else "Sigmoid"
    print("NEURAL NETWORK FROM SCRATCH")
    print("===========================\n")
    print(f"Dataset: {dataset.name.upper()}" + ("" if dataset.is_gate
                                                else f" - {dataset.description}"))
    if not dataset.is_gate:
        parts = [f"{len(split.X_train)} train"]
        if split.X_val is not None:
            parts.append(f"{len(split.X_val)} validation")
        if split.X_test is not None:
            parts.append(f"{len(split.X_test)} test")
        print(f"Rows: {dataset.n_samples} ({', '.join(parts)})"
              + (" | inputs standardized" if settings["standardize"] else ""))
    print(f"Architecture: {' -> '.join(str(n) for n in sizes)}")
    print(f"Activation: {settings['activation'].capitalize()} (hidden), {output} (output)")
    print("Loss: " + ("Categorical" if dataset.task == "multiclass" else "Binary")
          + " Cross-Entropy")
    print(f"Optimizer: {describe_optimizer(settings)}")
    extras = []
    if settings["l2"] > 0:
        extras.append(f"L2 {settings['l2']:g}")
    if settings["dropout"] > 0:
        extras.append(f"dropout {settings['dropout']:g}")
    if settings["weight_init"] != "auto":
        extras.append(f"{settings['weight_init']} init")
    if extras:
        print(f"Regularisation / init: {', '.join(extras)}")
    stopping = ("off" if settings["early_stopping"] is None
                else f"patience {settings['early_stopping']}, min delta {settings['min_delta']:g}")
    print(f"Epochs: {settings['epochs']} | Seed: {settings['seed']} | Early stopping: {stopping}\n")


def print_gate_predictions(network, X, y):
    probabilities = network.predict_proba(X)
    predictions = network.predict(X)
    print("Final Predictions:")
    for inputs, expected, prob, pred in zip(X, y, probabilities, predictions):
        print(f"Input: [{int(inputs[0])}, {int(inputs[1])}] | "
              f"Expected: {int(expected[0])} | "
              f"Probability: {prob[0]:.4f} | "
              f"Predicted: {int(pred[0])}")
    loss, accuracy = network.evaluate(X, y)
    print(f"\nFinal Loss: {loss:.6f}")
    print(f"Final Accuracy: {accuracy * 100:.0f}%")


def print_results(network, dataset, split):
    """Results for datasets with more rows than a logic gate."""
    metrics = compute_metrics(network, split)
    rows = []
    for label, key in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
        if f"{key}_loss" in metrics:
            rows.append((label, f"{metrics[f'{key}_loss']:.4f}",
                         f"{metrics[f'{key}_accuracy'] * 100:.1f}%"))
    print("Results:")
    print(format_table(("Data", "Loss", "Accuracy"), rows))

    X, y = (split.X_test, split.y_test) if split.X_test is not None else (
        split.X_train, split.y_train)
    which = "test" if split.X_test is not None else "training"
    if network.is_multiclass:
        matrix = network.confusion_matrix(X, y)
        print(f"\nConfusion matrix on the {which} rows (rows = true class, columns = predicted):")
        width = max(3, len(str(matrix.max())) + 1)
        print("      " + "".join(f"{c:>{width}}" for c in range(matrix.shape[0])))
        for cls, row in enumerate(matrix):
            print(f"  {cls:>3} " + "".join(f"{v:>{width}}" for v in row))
        if any(str(i) != name for i, name in enumerate(dataset.class_names)):
            print("  Classes: " + ", ".join(f"{i} = {name}"
                                            for i, name in enumerate(dataset.class_names)))
    probabilities = network.predict_proba(X[:5])
    predictions = network.predict(X[:5])
    print(f"\nFirst {len(predictions)} {which} rows:")
    for index, (prob, pred) in enumerate(zip(probabilities, predictions)):
        if network.is_multiclass:
            truth = int(np.asarray(y)[index])
            print(f"  Row {index + 1} | true: {truth} | predicted: {int(pred)} | "
                  f"confidence: {prob.max() * 100:.1f}%")
        else:
            print(f"  Row {index + 1} | true: {int(np.asarray(y)[index].ravel()[0])} | "
                  f"probability: {prob[0]:.4f} | predicted: {int(pred[0])}")
    return metrics


# ----------------------------------------------------------------------
# Files: plots, CSV log, saved model
# ----------------------------------------------------------------------
def save_standard_plots(network, dataset, split, options, recorder=None):
    """Loss curve, decision boundary / confusion matrix, hidden activations, extras."""
    if options.get("no_plots", False):
        return
    try:
        import plots
        folder, name = options.get("output_dir", "outputs"), dataset.name
        history, saved = network.history, []

        if network.loss_history:
            path = os.path.join(folder, f"{name}_loss_curve.png")
            has_val = bool(history.get("val_loss"))
            plots.save_loss_plot(
                network.loss_history, path, title=f"Training Loss vs Epochs ({name.upper()})",
                val_history=history.get("val_loss") if has_val else None,
                accuracy=history.get("accuracy") if has_val else None,
                val_accuracy=history.get("val_accuracy") if has_val else None)
            saved.append(path)
        if network.input_size == 2:
            path = os.path.join(folder, f"{name}_decision_boundary.png")
            test = (split.X_test, split.y_test) if split.X_test is not None else None
            plots.save_decision_boundary(network, split.X_train, split.y_train, path,
                                         title=f"What the network learned ({name.upper()})",
                                         test=test)
            saved.append(path)
        if network.is_multiclass and not dataset.is_gate:
            X, y = (split.X_test, split.y_test) if split.X_test is not None else (
                split.X_train, split.y_train)
            path = os.path.join(folder, f"{name}_confusion_matrix.png")
            plots.save_confusion_matrix(
                network.confusion_matrix(X, y), path, class_names=dataset.class_names,
                title=f"Confusion matrix ({'test' if split.X_test is not None else 'training'} rows)")
            saved.append(path)
        if dataset.image_shape is not None and network.is_multiclass:
            X, y = (split.X_test, split.y_test) if split.X_test is not None else (
                split.X_train, split.y_train)
            path = os.path.join(folder, f"{name}_predictions.png")
            plots.save_image_predictions(
                X, y, network.predict(X), dataset.image_shape, path,
                class_names=dataset.class_names,
                title=f"Predictions on {'unseen test' if split.X_test is not None else 'training'} images")
            saved.append(path)
        if len(split.X_train) <= 8 and network.n_layers >= 2:
            path = os.path.join(folder, f"{name}_hidden_activations.png")
            plots.save_hidden_activations(network, split.X_train, path)
            saved.append(path)
        if options.get("plot_stats", False) and history.get("weight_norm"):
            path = os.path.join(folder, f"{name}_training_stats.png")
            plots.save_training_stats_plot(history, path,
                                           title=f"Weights, gradients and learning rate ({name.upper()})")
            saved.append(path)
        if recorder is not None:
            if network.stopped_early:   # the schedule may have missed the final state
                recorder.record(network.epochs_run, network, network.loss_history[-1])
            path = os.path.join(folder, f"{name}_training.gif")
            recorder.save_gif(split.X_train, split.y_train, path, title=f"Learning {name.upper()}")
            saved.append(path)
        print("\nSaved graphs:")
        for path in saved:
            print(f"  {path}")
    except ImportError:
        print("\nMatplotlib is not installed, so graphs were skipped "
              "(install it with: pip install -r requirements.txt).")


def write_csv_log(network, path):
    """One row per epoch with everything that was recorded during training."""
    history = network.history
    if not history.get("loss"):
        raise ValueError("there is no training history to log")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    layers = network.n_layers
    header = (["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"]
              + [f"weight_norm_layer{i + 1}" for i in range(layers)]
              + [f"grad_norm_layer{i + 1}" for i in range(layers)])
    count = len(history["loss"])

    def value(key, index):
        series = history.get(key)
        return series[index] if series and index < len(series) else ""

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(count):
            writer.writerow([index + 1, value("loss", index), value("accuracy", index),
                             value("val_loss", index), value("val_accuracy", index),
                             value("lr", index)]
                            + list(history["weight_norm"][index])
                            + (list(history["grad_norm"][index])
                               if index < len(history["grad_norm"]) else [""] * layers))
    return count


def model_extras(dataset, settings, scaler):
    """Information stored inside the model file so --load can rebuild the same data."""
    extra = dict(dataset=dataset.name, data_seed=settings["data_seed"],
                 samples=-1 if settings["samples"] is None else settings["samples"],
                 seed=settings["seed"], val_size=settings["val_size"],
                 test_size=settings["test_size"], standardize=int(bool(settings["standardize"])))
    if scaler is not None:
        extra["scaler_mean"], extra["scaler_std"] = scaler.mean, scaler.std
    return extra


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_single(options, dataset_name):
    """Train one network on one dataset and report the results."""
    overrides = options.overrides()
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    settings = resolve_settings(dataset, overrides)
    split, scaler = prepare_data(dataset, settings)
    print_header(dataset, settings, split)

    network = build_network(dataset, settings)
    recorder = None
    if options.get("animate", False) and not options.get("no_plots", False):
        if network.input_size != 2:
            raise ValueError("--animate needs a dataset with 2 inputs "
                             "(moons, circles, blobs, spirals or a logic gate)")
        from plots import TrainingRecorder
        recorder = TrainingRecorder(settings["epochs"], X=split.X_train)

    print_every = options.get("print_every", max(1, settings["epochs"] // 10))
    print("Training started...")
    fit(network, split, settings, verbose=True, print_every=print_every, callback=recorder)
    print("Training completed.\n")

    report(network, dataset, split)
    if network.stopped_early:
        print(f"(Stopped early after {network.epochs_run} of {settings['epochs']} epochs.)")
    if settings["restore_best"] and network.best_epoch is not None:
        print(f"(Restored the weights from epoch {network.best_epoch}, "
              f"the lowest {'validation' if split.X_val is not None else 'training'} loss.)")
    save_standard_plots(network, dataset, split, options, recorder)

    if options.get("log_csv"):
        count = write_csv_log(network, options.get("log_csv"))
        print(f"\nTraining log ({count} epochs) saved to: {options.get('log_csv')}")
    if options.get("save"):
        path = network.save(options.get("save"), extra=model_extras(dataset, settings, scaler))
        print(f"\nModel saved to: {path}")
    return dataset, settings, split


def report(network, dataset, split):
    if dataset.is_gate and split.X_val is None and split.X_test is None:
        print_gate_predictions(network, split.X_train, split.y_train)
    else:
        print_results(network, dataset, split)


def run_load(options):
    """--load: evaluate a saved model without training."""
    network = NeuralNetwork.load(options.get("load"))
    extra = network.extra
    saved_name = str(extra["dataset"]) if "dataset" in extra else None
    if options.explicit("dataset") and saved_name and options.get("dataset") != saved_name:
        raise ValueError(f"the model was trained on {saved_name!r} but --dataset is "
                         f"{options.get('dataset')!r}")
    name = options.get("dataset") if options.explicit("dataset") else (saved_name or "xor")
    if name == "all":
        raise ValueError("--load needs one dataset, not 'all'")

    def stored(key, convert, fallback):
        return convert(extra[key]) if key in extra else fallback

    overrides = options.overrides()
    if saved_name:      # the model remembers how its data was made and split
        samples = stored("samples", int, -1)
        overrides.update(data_seed=stored("data_seed", int, 0),
                         samples=None if samples < 0 else samples,
                         seed=stored("seed", int, 42), val_size=stored("val_size", float, 0.0),
                         test_size=stored("test_size", float, 0.0), standardize=False)
    dataset = load_dataset(name, seed=overrides.get("data_seed", 0),
                           n_samples=overrides.get("samples"))
    if network.input_size != dataset.input_size or network.output_size != dataset.output_size:
        raise ValueError(f"the loaded model ({network.input_size} inputs, {network.output_size} "
                         f"outputs) does not fit dataset {name!r} ({dataset.input_size} inputs, "
                         f"{dataset.output_size} outputs)")
    settings = resolve_settings(dataset, overrides)
    split, _ = prepare_data(dataset, settings)
    if "scaler_mean" in extra:      # apply the scaling the model was trained with
        scaler = Standardizer()
        scaler.mean, scaler.std = extra["scaler_mean"], extra["scaler_std"]
        for key in ("X_train", "X_val", "X_test"):
            if getattr(split, key) is not None:
                setattr(split, key, scaler.transform(getattr(split, key)))

    print("NEURAL NETWORK FROM SCRATCH")
    print("===========================\n")
    print(f"Dataset: {dataset.name.upper()}")
    print(f"Loaded model from {options.get('load')}:\n  {network.summary()}")
    print("(No training; evaluating the saved weights.)\n")
    report(network, dataset, split)
    save_standard_plots(network, dataset, split, options)
    return network


def run_explain(options, dataset_name):
    """--explain: walk through one training step, without training first."""
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    if options.get("load"):
        network = NeuralNetwork.load(options.get("load"))
        settings = resolve_settings(dataset, options.overrides())
        split, _ = prepare_data(dataset, settings)
    else:
        settings = resolve_settings(dataset, options.overrides())
        split, _ = prepare_data(dataset, settings)
        network = build_network(dataset, settings)
    row = options.get("explain_example", 4 if dataset.is_gate else 1) - 1
    print(explain_training_step(network, split.X_train, split.y_train, example_index=row))


def run_gradient_check(options, dataset_name):
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    settings = resolve_settings(dataset, options.overrides())
    split, _ = prepare_data(dataset, settings)
    network = build_network(dataset, settings)
    rows = min(len(split.X_train), 16)
    print(f"Network: {network.summary()}")
    print(f"Checking on {rows} {dataset.name.upper()} rows (dropout off, small random biases).\n")
    text, ok = format_gradient_check(
        gradient_check(network, split.X_train[:rows], split.y_train[:rows]))
    print(text)
    return 0 if ok else 1


def run_sweep_mode(options, dataset_name):
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    grid = parse_grid(options.get("sweep"))
    seeds = options.get("sweep_seeds", 1)
    overrides = options.overrides()
    base = resolve_settings(dataset, overrides)
    combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"SWEEP on {dataset.name.upper()}: {combos} combination(s) x {seeds} seed(s) = "
          f"{combos * seeds} runs, {base['epochs']} epochs each\n")
    rows = run_sweep(dataset, overrides, grid, seeds=seeds)
    has_val = base["val_size"] > 0
    has_test = rows[0]["test_accuracy"] is not None
    headers = ["Rank", "Settings", "Val acc" if has_val else "Train acc"]
    if has_test:
        headers.append("Test acc")
    headers.append("Train loss")
    table = []
    for rank, row in enumerate(rows, start=1):
        settings_text = ", ".join(f"{k}={'-'.join(map(str, v)) if isinstance(v, list) else v}"
                                  for k, v in row["settings"].items())
        line = [rank, settings_text, f"{row['score'] * 100:.1f}%"]
        if has_test:
            line.append(f"{row['test_accuracy'] * 100:.1f}%")
        line.append(f"{row['train_loss']:.4f}")
        table.append(line)
    print(format_table(headers, table))
    print(f"\nRanked by {'validation' if has_val else 'training'} accuracy (mean over "
          f"{seeds} seed(s)). Best: " + table[0][1])
    return 0


def run_reliability(options, dataset_name):
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    n = options.get("seeds_report")
    overrides = options.overrides()
    result = reliability_report(dataset, overrides, n)
    s = result["settings"]
    print(f"RELIABILITY REPORT: {dataset.name.upper()} | hidden {s['hidden']} | {s['activation']} | "
          f"{s['optimizer']} lr {s['lr']:g} | {s['epochs']} epochs")
    print(f"Seeds {result['first_seed']} to {result['first_seed'] + n - 1} ({n} runs)\n")
    loss = result["final_loss"]
    rows = [("Runs reaching 100% training accuracy", f"{result['solved']} / {n}"),
            ("Final training loss (mean / median / worst)",
             f"{loss.mean():.4f} / {np.median(loss):.4f} / {loss.max():.4f}")]
    for label, key in (("Validation accuracy", "val_accuracy"), ("Test accuracy", "test_accuracy")):
        values = result[key]
        if len(values):
            rows.append((f"{label} (mean +/- std; min; max)",
                         f"{values.mean() * 100:.1f}% +/- {values.std() * 100:.1f}; "
                         f"{values.min() * 100:.1f}%; {values.max() * 100:.1f}%"))
    if result["median_epochs_to_threshold"] is not None:
        rows.append((f"Median epochs until loss < {LOSS_THRESHOLD}",
                     f"{result['median_epochs_to_threshold']:.0f} "
                     f"({result['reached_threshold']} of {n} runs got there)"))
    print(format_table(("Measure", "Value"), rows))
    return 0


def run_all_gates(options):
    """Train one network per logic gate and compare how quickly each is learned."""
    overrides = options.overrides()
    first = load_dataset(DATASET_NAMES[0])
    settings = resolve_settings(first, overrides)
    print("NEURAL NETWORK FROM SCRATCH")
    print("===========================\n")
    print(f"Datasets: {', '.join(n.upper() for n in DATASET_NAMES)}")
    print(f"Architecture: {' -> '.join(str(n) for n in (2, *settings['hidden'], 1))} "
          f"| Activation: {settings['activation']} | {describe_optimizer(settings)}")
    print(f"Epochs: {settings['epochs']} | Seed: {settings['seed']}\n")

    rows, histories = [], {}
    for name in DATASET_NAMES:
        dataset = load_dataset(name)
        result = run_experiment(dataset, overrides)
        network, split = result.network, result.split
        first_epoch = epochs_to_threshold(network.loss_history)
        rows.append((name.upper(), network.epochs_run, f"{result.metrics['train_loss']:.6f}",
                     f"{result.metrics['train_accuracy'] * 100:.0f}%",
                     first_epoch if first_epoch is not None else "never"))
        histories[name.upper()] = network.loss_history
        save_standard_plots(network, dataset, split, options)

    print(f"\nHow quickly each gate is learned (threshold: loss < {LOSS_THRESHOLD}):\n")
    print(format_table(("Dataset", "Epochs run", "Final loss", "Accuracy",
                        f"Epochs to loss<{LOSS_THRESHOLD}"), rows))
    if not options.get("no_plots", False):
        try:
            import plots
            path = os.path.join(options.get("output_dir", "outputs"), "datasets_loss_comparison.png")
            plots.save_comparison_plot(histories, path, title="Loss by logic gate")
            print(f"\nSaved comparison graph: {path}")
        except ImportError:
            pass
    print("\nNote: AND, OR and NAND are linearly separable, so a single straight line can solve\n"
          "them. XOR is not, which is why it needs the hidden layer.")


def _compare(options, dataset_name, variants, title, filename):
    """Train several variants on one dataset and print + plot the comparison."""
    dataset = load_dataset(dataset_name, seed=options.get("data_seed", 0),
                           n_samples=options.get("samples"))
    base = options.overrides()
    rows, histories = [], {}
    shown = None
    for label, overrides in variants:
        merged = {**base, **overrides}
        result = run_experiment(dataset, merged)
        network, metrics = result.network, result.metrics
        shown = "test_accuracy" if "test_accuracy" in metrics else "train_accuracy"
        first_epoch = epochs_to_threshold(network.loss_history)
        rows.append((label, f"{metrics['train_loss']:.6f}", f"{metrics[shown] * 100:.1f}%",
                     first_epoch if first_epoch is not None else "never"))
        histories[label] = network.loss_history
    epochs = result.settings["epochs"]
    seed = result.settings["seed"]
    print(f"\n{title} ({dataset.name.upper()}, {epochs} epochs, seed {seed})\n")
    print(format_table(("Variant", "Final loss",
                        "Test accuracy" if shown == "test_accuracy" else "Accuracy",
                        f"Epochs to loss<{LOSS_THRESHOLD}"), rows))
    if not options.get("no_plots", False):
        try:
            import plots
            path = os.path.join(options.get("output_dir", "outputs"),
                                f"{dataset.name}_{filename}.png")
            plots.save_comparison_plot(histories, path, title=f"{title} ({dataset.name.upper()})")
            print(f"\nSaved comparison graph: {path}")
        except ImportError:
            pass


def compare_optimizers(options, dataset_name):
    # Each optimizer gets its usual learning rate (unless you gave --lr), because
    # Adam and RMSprop need much smaller steps than plain gradient descent.
    lr = lambda name: {} if options.explicit("lr") else {"lr": OPTIMIZER_LR[name]}
    momentum = options.get("momentum", 0.9)
    variants = [
        ("Gradient descent (full batch)", dict(optimizer="gd", batch_size=None, **lr("gd"))),
        (f"Momentum {momentum} (full batch)",
         dict(optimizer="momentum", batch_size=None, **lr("momentum"))),
        (f"Mini-batch of {MINI_BATCH_DEMO}",
         dict(optimizer="gd", batch_size=MINI_BATCH_DEMO, **lr("gd"))),
        (f"Mini-batch of {MINI_BATCH_DEMO} + momentum",
         dict(optimizer="momentum", batch_size=MINI_BATCH_DEMO, **lr("momentum"))),
        ("RMSprop (full batch)", dict(optimizer="rmsprop", batch_size=None, **lr("rmsprop"))),
        ("Adam (full batch)", dict(optimizer="adam", batch_size=None, **lr("adam"))),
    ]
    # Early stopping would cut curves short; the comparison needs equal lengths.
    variants = [(label, {**overrides, "early_stopping": None, "restore_best": False})
                for label, overrides in variants]
    _compare(options, dataset_name, variants, "Optimizer comparison", "optimizer_comparison")
    print("Learning rates: " + ", ".join(f"{k} {v:g}" for k, v in OPTIMIZER_LR.items())
          + (" (overridden by --lr)." if options.explicit("lr") else ".")
          + " Early stopping is off so every curve covers the same epochs.")


def compare_activations(options, dataset_name):
    variants = [(act, {"activation": act, "early_stopping": None, "restore_best": False})
                for act in sorted(ACTIVATIONS)]
    _compare(options, dataset_name, variants, "Activation comparison", "activation_comparison")
    print("ReLU can 'die' (a neuron stuck at 0 never learns), so small ReLU networks can fail\n"
          "on some seeds. Leaky ReLU keeps a small slope for negative inputs to avoid that.\n"
          "Try a wider hidden layer, e.g. --hidden 8, or another --seed.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        config = load_config(args.config) if args.config else {}
        options = Options(args, config)
        name = options.get("dataset", "xor")
        if name not in ALL_DATASET_NAMES + ("all",):
            raise ValueError(f"unknown dataset {name!r} in the config file")
        if name == "all" and not args.dataset:
            for flag in ("save", "load", "explain", "gradient_check", "sweep", "seeds_report"):
                if getattr(args, flag):
                    parser.error(f"--{flag.replace('_', '-')} needs one dataset, not 'all'")

        if args.load and not args.explain:
            run_load(options)
            return 0
        if name == "all":
            run_all_gates(options)
            return 0
        if args.explain:
            run_explain(options, name)
            return 0
        if args.gradient_check:
            return run_gradient_check(options, name)
        if args.sweep:
            return run_sweep_mode(options, name)
        if args.seeds_report:
            return run_reliability(options, name)

        run_single(options, name)
        if options.get("compare_optimizers", False):
            compare_optimizers(options, name)
        if options.get("compare_activations", False):
            compare_activations(options, name)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
