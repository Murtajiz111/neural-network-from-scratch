"""Datasets for the neural network project.

Logic gates (xor, and, or, nand)
    Four rows each. XOR is the interesting one: it is NOT linearly separable,
    so a network needs a hidden layer to learn it. AND, OR and NAND are
    linearly separable, so they are learned faster.

Synthetic classification problems (generated with NumPy, nothing is downloaded)
    moons    two interleaving half circles          (2 inputs, 2 classes)
    circles  a ring inside a ring                   (2 inputs, 2 classes)
    blobs    three overlapping clusters             (2 inputs, 3 classes)
    spirals  three intertwined spiral arms          (2 inputs, 3 classes)
    digits   noisy 5x7 pixel pictures of 0-9        (35 inputs, 10 classes)

The synthetic sets are generated from a seed, so they are reproducible. They
are stand-ins for real data: bigger and messier than a logic gate, but not
photographs or measurements. "digits" is NOT the famous MNIST / scikit-learn
digits data; it is made from a tiny hand-written pixel font plus random noise.

REAL datasets (bundled as CSV files in the data/ folder, see data/README.md)
    iris       150 flowers, 4 measurements each, 3 species    (4 inputs, 3 classes)
    digits8x8  1,797 real handwritten digits, 8x8 pixels      (64 inputs, 10 classes)

These are measurements and scans of the real world, so they are noisy in the
way real data is. They are read from disk, not generated: ``seed`` and
``n_samples`` do not apply to them.
"""

import os
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ----------------------------------------------------------------------
# Logic gates
# ----------------------------------------------------------------------
_INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))

_OUTPUTS = {
    "xor": (0, 1, 1, 0),    # 1 when the inputs are different
    "and": (0, 0, 0, 1),    # 1 only when both inputs are 1
    "or": (0, 1, 1, 1),     # 1 when at least one input is 1
    "nand": (1, 1, 1, 0),   # NOT(AND)
}

DATASET_NAMES = tuple(_OUTPUTS)                       # the logic gates
EXTRA_DATASET_NAMES = ("moons", "circles", "blobs", "spirals", "digits")   # synthetic
REAL_DATASET_NAMES = ("iris", "digits8x8")                                  # real, from data/
ALL_DATASET_NAMES = DATASET_NAMES + EXTRA_DATASET_NAMES + REAL_DATASET_NAMES


def get_dataset(name):
    """Return (X, y) for a logic gate: X has shape (4, 2) and y has shape (4, 1)."""
    key = str(name).lower()
    if key not in _OUTPUTS:
        raise ValueError(f"Unknown logic gate {name!r}. Choose from: {', '.join(DATASET_NAMES)}")
    X = np.array(_INPUTS, dtype=float)
    y = np.array(_OUTPUTS[key], dtype=float).reshape(-1, 1)
    return X, y


# ----------------------------------------------------------------------
# A small description of any dataset
# ----------------------------------------------------------------------
@dataclass
class Dataset:
    """Inputs ``X`` plus targets ``y`` and a description.

    task "binary":     y has shape (m, 1) with values 0.0 / 1.0.
    task "multiclass": y is a 1-D array of integer class numbers.
    ``defaults`` are settings that work well for this dataset (hidden layers,
    optimizer, learning rate, ...). They can be overridden from the command line.
    """
    name: str
    X: np.ndarray
    y: np.ndarray
    task: str
    n_classes: int
    description: str
    class_names: tuple = ()
    is_gate: bool = False
    defaults: dict = field(default_factory=dict)
    is_real: bool = False               # True for measurements loaded from data/
    image_shape: tuple = None           # e.g. (8, 8) when every row is a flattened picture

    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        return 1 if self.task == "binary" else self.n_classes

    @property
    def n_samples(self):
        return self.X.shape[0]


# ----------------------------------------------------------------------
# Synthetic generators
# ----------------------------------------------------------------------
def _moons(n, rng, noise=0.15):
    first = n // 2
    second = n - first
    t1 = np.pi * rng.random(first)
    t2 = np.pi * rng.random(second)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])
    inner = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([outer, inner]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(first), np.ones(second)]).reshape(-1, 1)
    return X, y


def _circles(n, rng, noise=0.08, factor=0.5):
    first = n // 2
    second = n - first
    t1 = 2 * np.pi * rng.random(first)
    t2 = 2 * np.pi * rng.random(second)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])
    inner = factor * np.column_stack([np.cos(t2), np.sin(t2)])
    X = np.vstack([outer, inner]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(first), np.ones(second)]).reshape(-1, 1)
    return X, y


def _blobs(n, rng, spread=0.9):
    centers = np.array([[-1.6, -1.0], [1.6, -1.0], [0.0, 1.9]])
    labels = np.arange(n) % len(centers)
    X = centers[labels] + rng.normal(0.0, spread, size=(n, 2))
    return X, labels.astype(int)


def _spirals(n, rng, arms=3, noise=0.2):
    X_parts, y_parts = [], []
    for arm in range(arms):
        count = n // arms + (1 if arm < n % arms else 0)
        radius = np.linspace(0.05, 1.0, count)
        angle = np.linspace(arm * 4.0, (arm + 1) * 4.0, count) + rng.normal(0.0, noise, count)
        X_parts.append(np.column_stack([radius * np.sin(angle), radius * np.cos(angle)]))
        y_parts.append(np.full(count, arm))
    return np.vstack(X_parts), np.concatenate(y_parts).astype(int)


# A tiny 5x7 pixel font for the digits 0-9 ('#' = ink).
_DIGIT_FONT = {
    0: (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    1: ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    2: (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    3: ("#### ", "    #", "    #", " ### ", "    #", "    #", "#### "),
    4: ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    5: ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    6: ("  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "),
    7: ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    8: (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    9: (" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "),
}


def digit_template(digit):
    """The clean 5x7 picture of a digit as a flat array of 35 zeros and ones."""
    rows = _DIGIT_FONT[int(digit)]
    return np.array([[1.0 if ch == "#" else 0.0 for ch in row] for row in rows]).ravel()


def _digits(n, rng, flip_probability=0.12):
    labels = np.arange(n) % 10
    templates = np.array([digit_template(d) for d in range(10)])
    X = templates[labels]
    flips = rng.random(X.shape) < flip_probability       # random pixel flips = noise
    X = np.abs(X - flips)
    return X, labels.astype(int)


# ----------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------
_GATE_DEFAULTS = dict(hidden=[4], activation="sigmoid", optimizer="gd", lr=0.5, epochs=10000,
                      weight_init="auto", test_size=0.0, val_size=0.0, standardize=False)

_SYNTHETIC = {
    # name: (generator, default sample count, task, classes, description, defaults)
    "moons": (_moons, 400, "binary", 2, "two interleaving half circles (2 classes)",
              dict(hidden=[8], activation="tanh", optimizer="adam", lr=0.01, epochs=1000,
                   weight_init="xavier", test_size=0.2, val_size=0.2, standardize=True)),
    "circles": (_circles, 400, "binary", 2, "a ring inside a ring (2 classes)",
                dict(hidden=[8], activation="tanh", optimizer="adam", lr=0.01, epochs=1000,
                     weight_init="xavier", test_size=0.2, val_size=0.2, standardize=True)),
    "blobs": (_blobs, 300, "multiclass", 3, "three overlapping clusters (3 classes)",
              dict(hidden=[8], activation="tanh", optimizer="adam", lr=0.01, epochs=500,
                   weight_init="xavier", test_size=0.2, val_size=0.2, standardize=True)),
    "spirals": (_spirals, 300, "multiclass", 3, "three intertwined spiral arms (3 classes)",
                dict(hidden=[32, 16], activation="tanh", optimizer="adam", lr=0.01, epochs=2000,
                     weight_init="xavier", test_size=0.2, val_size=0.2, standardize=True)),
    "digits": (_digits, 800, "multiclass", 10, "noisy 5x7 pixel pictures of the digits 0-9 "
               "(10 classes)",
               dict(hidden=[24], activation="tanh", optimizer="adam", lr=0.01, epochs=300,
                    weight_init="xavier", test_size=0.2, val_size=0.2, standardize=False)),
}

# Real datasets: name -> file, description, class names, divisor applied to the
# raw numbers, picture shape, recommended settings.
_REAL = {
    "iris": dict(
        file="iris.csv", scale=1.0, image_shape=None,
        description="150 real iris flowers: 4 measurements -> species (3 classes)",
        class_names=("setosa", "versicolor", "virginica"),
        defaults=dict(hidden=[8], activation="tanh", optimizer="adam", lr=0.01, epochs=600,
                      weight_init="xavier", test_size=0.2, val_size=0.2, standardize=True,
                      early_stopping=40, restore_best=True)),
    "digits8x8": dict(
        file="digits8x8.csv", scale=16.0, image_shape=(8, 8),
        description="1,797 real handwritten digits as 8x8 pixel pictures (10 classes)",
        class_names=tuple(str(d) for d in range(10)),
        defaults=dict(hidden=[64], activation="relu", optimizer="adam", lr=0.01, epochs=100,
                      weight_init="he", test_size=0.2, val_size=0.2, standardize=False,
                      batch_size=32, early_stopping=10, restore_best=True)),
}


def _load_real(key):
    """Read a bundled CSV file into (X, labels). The last column is the class."""
    info = _REAL[key]
    path = os.path.join(DATA_DIR, info["file"])
    if not os.path.exists(path):
        raise ValueError(f"data file not found: {path!r}. The data/ folder must sit next to "
                         f"datasets.py (see data/README.md).")
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    if table.ndim != 2 or table.shape[1] < 2 or not np.all(np.isfinite(table)):
        raise ValueError(f"data file {path!r} is malformed (expected numeric rows, no gaps)")
    X, labels = table[:, :-1] / info["scale"], table[:, -1].astype(int)
    if labels.min() < 0 or labels.max() >= len(info["class_names"]):
        raise ValueError(f"data file {path!r} has class numbers outside 0..{len(info['class_names']) - 1}")
    return X, labels


_GATE_DESCRIPTIONS = {
    "xor": "1 when the two inputs are different (not linearly separable)",
    "and": "1 only when both inputs are 1",
    "or": "1 when at least one input is 1",
    "nand": "NOT(AND): 0 only when both inputs are 1",
}


def load_dataset(name, seed=0, n_samples=None):
    """Return a :class:`Dataset`.

    ``seed`` controls the random generation of the synthetic datasets and
    ``n_samples`` their size. Both are ignored for the four-row logic gates.
    """
    key = str(name).lower()
    if key in _OUTPUTS:
        X, y = get_dataset(key)
        return Dataset(name=key, X=X, y=y, task="binary", n_classes=2,
                       description=_GATE_DESCRIPTIONS[key], is_gate=True,
                       class_names=("0", "1"), defaults=dict(_GATE_DEFAULTS))
    if key in _REAL:
        if n_samples is not None:
            raise ValueError(f"--samples does not apply to the real dataset {key!r}: "
                             f"it is read from a file and always uses every row")
        info = _REAL[key]
        X, labels = _load_real(key)
        return Dataset(name=key, X=X, y=labels, task="multiclass",
                       n_classes=len(info["class_names"]), description=info["description"],
                       class_names=info["class_names"], is_gate=False, defaults=dict(info["defaults"]),
                       is_real=True, image_shape=info["image_shape"])
    if key not in _SYNTHETIC:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(ALL_DATASET_NAMES)}")
    generator, default_n, task, classes, description, defaults = _SYNTHETIC[key]
    count = default_n if n_samples is None else int(n_samples)
    if count < 10 * classes:
        raise ValueError(f"n_samples must be at least {10 * classes} for {key}")
    X, y = generator(count, np.random.default_rng(seed))
    return Dataset(name=key, X=X, y=y, task=task, n_classes=classes, description=description,
                   class_names=tuple(str(c) for c in range(classes)), is_gate=False,
                   defaults=dict(defaults))


# ----------------------------------------------------------------------
# Splitting and scaling
# ----------------------------------------------------------------------
@dataclass
class Split:
    """Training data plus optional validation and test data."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray = None
    y_val: np.ndarray = None
    X_test: np.ndarray = None
    y_test: np.ndarray = None

    @property
    def validation_data(self):
        return None if self.X_val is None else (self.X_val, self.y_val)


def train_val_test_split(X, y, val_size=0.0, test_size=0.0, seed=0):
    """Randomly split into training, validation and test sets.

    The split is *stratified*: every class keeps (roughly) its share in each
    part, so a small test set is not accidentally missing a class. With
    val_size = test_size = 0 everything is used for training.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    for name, value in (("val_size", val_size), ("test_size", test_size)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1")
    if val_size == 0 and test_size == 0:
        return Split(X_train=X, y_train=y)

    rng = np.random.default_rng(seed)
    class_of_row = y.reshape(len(y), -1)[:, 0].astype(int)
    train_rows, val_rows, test_rows = [], [], []
    for cls in np.unique(class_of_row):
        rows = rng.permutation(np.flatnonzero(class_of_row == cls))
        n_test = int(round(len(rows) * test_size))
        n_val = int(round(len(rows) * val_size))
        if len(rows) - n_test - n_val < 1:
            raise ValueError("too few examples of one class for this split; use smaller "
                             "val_size / test_size or more samples")
        test_rows.extend(rows[:n_test])
        val_rows.extend(rows[n_test:n_test + n_val])
        train_rows.extend(rows[n_test + n_val:])

    def part(indices):
        indices = rng.permutation(np.array(indices, dtype=int))
        return (X[indices], y[indices]) if len(indices) else (None, None)

    X_train, y_train = part(train_rows)
    X_val, y_val = part(val_rows)
    X_test, y_test = part(test_rows)
    return Split(X_train, y_train, X_val, y_val, X_test, y_test)


class Standardizer:
    """Rescale every input column to mean 0 and standard deviation 1.

    It is fitted on the TRAINING data only and then applied unchanged to the
    validation and test data, so nothing about the test set leaks into training.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < 1e-12, 1.0, std)      # constant columns stay as they are
        return self

    def transform(self, X):
        if self.mean is None:
            raise ValueError("call fit() before transform()")
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def standardize_split(split):
    """Standardize a :class:`Split` using statistics of its training part."""
    scaler = Standardizer().fit(split.X_train)
    return Split(
        scaler.transform(split.X_train), split.y_train,
        None if split.X_val is None else scaler.transform(split.X_val), split.y_val,
        None if split.X_test is None else scaler.transform(split.X_test), split.y_test,
    ), scaler
