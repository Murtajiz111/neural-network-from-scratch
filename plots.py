"""Matplotlib plots for the neural network project.

Matplotlib is imported lazily, so the rest of the project still works if it is
not installed. Plots are drawn to files (no window needed).
"""

import os

import numpy as np

DEFAULT_BOUNDS = (-0.5, 1.5, -0.5, 1.5)     # x_min, x_max, y_min, y_max (logic gates)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
BINARY_LABELS = {0: "expected 0", 1: "expected 1"}


def _pyplot():
    """Import pyplot using the file-only backend. Raises ImportError if missing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _prepare(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _labels(y):
    """Class number of every row, for (m, 1) binary or 1-D multiclass targets."""
    y = np.asarray(y)
    return y.reshape(len(y), -1)[:, 0].astype(int)


def _class_colors(n_classes):
    plt = _pyplot()
    palette = plt.get_cmap("tab10")
    return [palette(i % 10) for i in range(n_classes)]


def data_bounds(X, margin=0.4):
    """Plot limits (x_min, x_max, y_min, y_max) that frame the data.

    For the 0/1 logic-gate inputs this gives the familiar [-0.5, 1.5] square.
    """
    X = np.asarray(X, dtype=float)
    low, high = X.min(axis=0), X.max(axis=0)
    span = np.where(high - low == 0, 1.0, high - low)
    if np.allclose(low, 0) and np.allclose(high, 1):
        return DEFAULT_BOUNDS
    return (low[0] - margin * span[0], high[0] + margin * span[0],
            low[1] - margin * span[1], high[1] + margin * span[1])


def probability_grid(network, resolution=120, bounds=DEFAULT_BOUNDS):
    """Network output on a regular grid covering the input square.

    Returns (xx, yy, grid). For a sigmoid network ``grid`` has shape
    (resolution, resolution) and holds P(class 1). For a softmax network it has
    shape (resolution, resolution, classes).
    """
    x_min, x_max, y_min, y_max = bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities = network.predict_proba(points)
    if network.is_multiclass:
        return xx, yy, probabilities.reshape(xx.shape + (network.output_size,))
    return xx, yy, probabilities[:, 0].reshape(xx.shape)


def _scatter_examples(ax, X, y, n_classes, test=None):
    """Training points (large) and optional test points (small, grey-filled)."""
    labels = _labels(y)
    colors = _class_colors(n_classes)
    binary = n_classes == 2
    for cls in range(n_classes):
        rows = labels == cls
        if not rows.any():
            continue
        marker = MARKERS[cls % len(MARKERS)]
        label = BINARY_LABELS[cls] if binary else f"class {cls}"
        size = 140 if len(labels) <= 8 else 28
        ax.scatter(X[rows, 0], X[rows, 1], s=size, marker=marker, label=label,
                   facecolor="white" if binary else colors[cls], edgecolor="black",
                   linewidth=2 if len(labels) <= 8 else 0.6, zorder=3)
    if test is not None and test[0] is not None:
        test_labels = _labels(test[1])
        for cls in range(n_classes):
            rows = test_labels == cls
            if rows.any():
                ax.scatter(test[0][rows, 0], test[0][rows, 1], s=34, marker=MARKERS[cls % len(MARKERS)],
                           facecolor="lightgrey", edgecolor="black", linewidth=1.0, zorder=4,
                           label="test points" if cls == 0 else None)


def _style_boundary_axes(ax, bounds):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_aspect("equal")


def save_loss_plot(loss_history, path, title="Training Loss vs Epochs", log_scale=False,
                   val_history=None, accuracy=None, val_accuracy=None):
    """Loss versus epochs. Adds a validation curve and an accuracy panel if given."""
    plt = _pyplot()
    _prepare(path)
    epochs = np.arange(1, len(loss_history) + 1)
    show_accuracy = accuracy is not None and len(accuracy) > 0 and val_accuracy

    fig, axes = plt.subplots(1, 2 if show_accuracy else 1,
                             figsize=(12.5 if show_accuracy else 8, 5), squeeze=False)
    ax = axes[0, 0]
    ax.plot(epochs, loss_history, color="tab:blue", linewidth=2, label="training")
    if val_history:
        ax.plot(np.arange(1, len(val_history) + 1), val_history, color="tab:orange",
                linewidth=2, linestyle="--", label="validation")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if show_accuracy:
        ax2 = axes[0, 1]
        ax2.plot(epochs, np.asarray(accuracy) * 100, color="tab:blue", linewidth=2, label="training")
        ax2.plot(np.arange(1, len(val_accuracy) + 1), np.asarray(val_accuracy) * 100,
                 color="tab:orange", linewidth=2, linestyle="--", label="validation")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_decision_boundary(network, X, y, path, title="Decision boundary", test=None, bounds=None):
    """What the network has learned: its output over the whole input square.

    Binary networks: a probability map with a black line where the probability
    equals 0.5. Multi-class networks: each region is coloured with the class
    the network would predict there. ``test`` is an optional (X_test, y_test).
    """
    if network.input_size != 2:
        raise ValueError("Decision boundary plots need a network with 2 inputs")
    plt = _pyplot()
    _prepare(path)
    X = np.asarray(X, dtype=float)
    if bounds is None:
        bounds = data_bounds(X if test is None or test[0] is None else np.vstack([X, test[0]]))
    xx, yy, grid = probability_grid(network, bounds=bounds)
    n_classes = network.output_size if network.is_multiclass else 2

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    if network.is_multiclass:
        colors = _class_colors(n_classes)
        ax.contourf(xx, yy, np.argmax(grid, axis=2), levels=np.arange(n_classes + 1) - 0.5,
                    colors=colors, alpha=0.3)
        ax.contour(xx, yy, np.argmax(grid, axis=2), levels=np.arange(n_classes) + 0.5,
                   colors="black", linewidths=1.2)
    else:
        filled = ax.contourf(xx, yy, grid, levels=np.linspace(0, 1, 21), cmap="coolwarm",
                             vmin=0, vmax=1)
        ax.contour(xx, yy, grid, levels=[0.5], colors="black", linewidths=2)
        fig.colorbar(filled, ax=ax, label="Predicted probability of 1")
    _scatter_examples(ax, X, y, n_classes, test)
    _style_boundary_axes(ax, bounds)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(5, n_classes + 1),
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hidden_activations(network, X, path, layer=1, title=None):
    """Heatmap of one hidden layer's outputs for every input row.

    Shows what the hidden neurons 'see': each column is a neuron, each row an
    input. ``layer`` is 1 for the first hidden layer. Meant for tiny datasets
    such as the logic gates.
    """
    if not 1 <= layer <= network.n_layers - 1:
        raise ValueError(f"layer must be between 1 and {network.n_layers - 1} (hidden layers only)")
    plt = _pyplot()
    _prepare(path)
    values = network.forward(np.asarray(X, dtype=float))[layer]
    row_labels = [f"[{int(a)}, {int(b)}]" for a, b in X]

    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * values.shape[1], 1.2 + 0.7 * values.shape[0]))
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(values.shape[1]), [f"h{i + 1}" for i in range(values.shape[1])])
    ax.set_yticks(range(values.shape[0]), row_labels)
    ax.set_xlabel("Hidden neuron")
    ax.set_ylabel("Input")
    ax.set_title(title or f"Hidden layer {layer} activations ({network.hidden_activation})")
    midpoint = (values.max() + values.min()) / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            colour = "black" if values[row, col] > midpoint else "white"
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center",
                    color=colour, fontsize=9)
    fig.colorbar(image, ax=ax, label="Activation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(matrix, path, title="Confusion matrix", class_names=None):
    """Counts of (true class, predicted class). A perfect model fills the diagonal."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("the confusion matrix must be square")
    plt = _pyplot()
    _prepare(path)
    n = matrix.shape[0]
    names = list(class_names) if class_names else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(0.6 * n + 3.2, 0.6 * n + 2.8))
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(matrix / row_totals, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for row in range(n):
        for col in range(n):
            share = matrix[row, col] / row_totals[row, 0]
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if share > 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_image_predictions(images, true_labels, predicted_labels, image_shape, path,
                           class_names=None, max_images=24, title="Predictions on unseen images"):
    """A grid of pictures with the network's answer under each one.

    Wrong answers are shown first (in red) so mistakes are easy to inspect,
    followed by correct ones (in green) to fill the grid.
    """
    images = np.asarray(images, dtype=float)
    true_labels = np.asarray(true_labels).astype(int)
    predicted_labels = np.asarray(predicted_labels).astype(int)
    if images.ndim != 2 or images.shape[1] != int(np.prod(image_shape)):
        raise ValueError(f"each row must have {int(np.prod(image_shape))} values for a "
                         f"{image_shape} picture, got shape {images.shape}")
    if not (len(images) == len(true_labels) == len(predicted_labels)):
        raise ValueError("images, true_labels and predicted_labels must have the same length")
    if max_images < 1:
        raise ValueError("max_images must be at least 1")
    plt = _pyplot()
    _prepare(path)

    wrong = np.flatnonzero(true_labels != predicted_labels)
    right = np.flatnonzero(true_labels == predicted_labels)
    chosen = np.concatenate([wrong, right])[:max_images]
    columns = min(6, len(chosen))
    rows = int(np.ceil(len(chosen) / columns))
    names = list(class_names) if class_names else None
    label = (lambda c: names[c]) if names else (lambda c: str(c))

    fig, axes = plt.subplots(rows, columns, figsize=(1.7 * columns, 1.9 * rows + 0.6), squeeze=False)
    for slot, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if slot >= len(chosen):
            continue
        index = chosen[slot]
        ax.imshow(images[index].reshape(image_shape), cmap="gray_r", interpolation="nearest")
        ok = true_labels[index] == predicted_labels[index]
        ax.set_title(f"{label(predicted_labels[index])}" if ok
                     else f"{label(predicted_labels[index])} (is {label(true_labels[index])})",
                     fontsize=9, color="#1b7f4a" if ok else "#b3261e")
    fig.suptitle(f"{title}  -  {len(wrong)} wrong of {len(images)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_stats_plot(history, path, title="Training statistics"):
    """How the weights, gradients and learning rate behave during training.

    * Weight size per layer: steady growth is normal; explosion is a warning.
    * Gradient size per layer: a layer whose gradient is far smaller than the
      others is learning slowly (the "vanishing gradient" problem).
    * Learning rate: shows the schedule.
    """
    for key in ("weight_norm", "grad_norm", "lr"):
        if not history.get(key):
            raise ValueError(f"history has no {key!r} data (train a network first)")
    plt = _pyplot()
    _prepare(path)
    weight_norm = np.array(history["weight_norm"])
    grad_norm = np.array(history["grad_norm"])
    epochs_w = np.arange(1, len(weight_norm) + 1)
    epochs_g = np.arange(1, len(grad_norm) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for layer in range(weight_norm.shape[1]):
        axes[0].plot(epochs_w, weight_norm[:, layer], linewidth=2, label=f"layer {layer + 1}")
        axes[1].plot(epochs_g, grad_norm[:, layer], linewidth=2, label=f"layer {layer + 1}")
    axes[0].set_title("Weight size (L2 norm)")
    axes[1].set_title("Gradient size (L2 norm)")
    axes[1].set_yscale("log")
    axes[2].plot(np.arange(1, len(history["lr"]) + 1), history["lr"], color="tab:green", linewidth=2)
    axes[2].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_plot(histories, path, title="Loss comparison", log_scale=True):
    """Several loss curves (name -> list of losses) on one chart."""
    if not histories:
        raise ValueError("histories is empty")
    plt = _pyplot()
    _prepare(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = ["-", "--", "-.", ":"]
    for index, (name, history) in enumerate(histories.items()):
        ax.plot(np.arange(1, len(history) + 1), history, linewidth=2,
                linestyle=styles[index % len(styles)], label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class TrainingRecorder:
    """Collects decision-boundary snapshots during training for a GIF.

    Pass an instance as ``callback`` to ``NeuralNetwork.train``. Frames are
    spaced geometrically (dense early, when the network changes fastest), so
    at most ``max_frames`` snapshots are stored whatever the epoch count.
    Give ``X`` (the training inputs) to frame the plot around the data.
    Binary networks record a probability map; multi-class networks record a
    colour image in which each class has its own colour and uncertain regions
    blend.
    """

    def __init__(self, total_epochs, max_frames=60, resolution=60, X=None):
        if total_epochs < 1 or max_frames < 2:
            raise ValueError("total_epochs must be >= 1 and max_frames >= 2")
        schedule = np.geomspace(1, total_epochs, num=min(max_frames, total_epochs))
        self._wanted = set(int(round(v)) for v in schedule) | {1}
        self.resolution = resolution
        self.bounds = DEFAULT_BOUNDS if X is None else data_bounds(X)
        self.epochs = []
        self.losses = []
        self.grids = []

    def __call__(self, epoch, network, loss):
        if epoch in self._wanted:
            self.record(epoch, network, loss)

    def record(self, epoch, network, loss):
        """Store a snapshot (also usable manually, e.g. for the final state)."""
        if self.epochs and self.epochs[-1] == epoch:
            return
        _, _, grid = probability_grid(network, self.resolution, self.bounds)
        if network.is_multiclass:
            colors = np.array([c[:3] for c in _class_colors(network.output_size)])
            grid = grid @ colors                           # blend class colours by probability
        self.epochs.append(epoch)
        self.losses.append(float(loss))
        self.grids.append(grid.astype(np.float32))

    def save_gif(self, X, y, path, title="Learning to separate the classes", fps=6):
        """Write the recorded frames as an animated GIF."""
        if len(self.grids) < 2:
            raise ValueError("Need at least two recorded frames to make an animation")
        plt = _pyplot()
        from matplotlib.animation import FuncAnimation, PillowWriter
        _prepare(path)
        X = np.asarray(X, dtype=float)
        multiclass = self.grids[0].ndim == 3
        n_classes = max(2, len(np.unique(_labels(y))))

        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        image = ax.imshow(self.grids[0], origin="lower", cmap=None if multiclass else "coolwarm",
                          vmin=None if multiclass else 0, vmax=None if multiclass else 1,
                          extent=self.bounds, interpolation="bilinear")
        _scatter_examples(ax, X, y, n_classes)
        _style_boundary_axes(ax, self.bounds)
        if not multiclass:
            fig.colorbar(image, ax=ax, label="Predicted probability of 1")
        # Give the layout a two-line title up front so it leaves room for it.
        text = ax.set_title(f"{title}\nepoch {self.epochs[0]} | loss {self.losses[0]:.4f}")

        def draw(frame):
            image.set_data(self.grids[frame])
            text.set_text(f"{title}\nepoch {self.epochs[frame]} | loss {self.losses[frame]:.4f}")
            return image, text

        fig.tight_layout()
        animation = FuncAnimation(fig, draw, frames=len(self.grids), blit=False)
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
