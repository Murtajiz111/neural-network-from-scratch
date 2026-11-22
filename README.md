# Neural Network from Scratch (NumPy only)

A small, readable neural network library built with **only Python and NumPy** (Matplotlib draws the graphs). No TensorFlow, PyTorch, Keras or scikit-learn is used to build or train anything. It starts with the classic **XOR** problem and grows into a toolkit you can experiment with: other logic gates, harder 2D shapes, real datasets (Iris flowers, handwritten digits), several optimizers, validation and test splits, a notebook and a browser playground.

## What is in this project

| Path | What it does |
|---|---|
| `neural_network.py` | The network: forward pass, backpropagation, optimizers, regularization, save/load. Start reading here. |
| `main.py` | Command-line program that trains, prints results and saves graphs. |
| `datasets.py` | Logic gates, synthetic shapes, the two real datasets, train/validation/test split, standardization. |
| `experiment.py`, `sweep.py`, `gradcheck.py`, `explain.py` | Shared run logic, parameter sweeps and multi-seed reports, numerical gradient check, step-by-step walkthrough. |
| `plots.py` | Loss curves, decision boundaries, confusion matrices, digit pictures, training GIF. |
| `data/` | `iris.csv`, `digits8x8.csv` and `data/README.md` (where they came from). |
| `notebooks/neural_network_walkthrough.ipynb` | A guided notebook, already run, with outputs. |
| `web/playground.html` | A single HTML file: train an XOR network in your browser, no install. |
| `tests/` | 134 automated tests. |
| `outputs/` | Where graphs and models are written. Examples from a fresh run are included. |

Languages: Python (all of the learning code and tests), one HTML file with JavaScript (the playground), and a Jupyter notebook (JSON containing Python cells).

## Setup on Windows (VS Code + PowerShell)

1. Install Python 3.9 or newer from python.org (tick **Add Python to PATH**).
2. Open the project folder in VS Code, then open a terminal (**Terminal > New Terminal**).
3. Create and activate a virtual environment, then install the two libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again. (Linux/macOS: `source .venv/bin/activate`.)

## Quick start

```powershell
python main.py                                   # train on XOR, save 4 graphs in outputs/
python -m unittest discover -s tests             # run all tests
python main.py --dataset iris                    # real flowers
python main.py --dataset digits8x8               # real handwritten digits
python main.py --explain                         # one training step, number by number
python main.py --help                            # every option
```

Default XOR run (2 inputs, 4 hidden neurons, 1 output, sigmoid, gradient descent, learning rate 0.5, 10,000 epochs, seed 42) ends with loss 0.001107 and 100% accuracy. On Linux this was checked; the numbers on Windows may differ in the last digits.

## The ideas, in plain words

* **Forward pass**: each layer computes `Z = inputs @ weights + biases`, then an activation squashes it (`sigmoid`, `tanh`, `relu`, `leaky_relu`).
* **Loss**: how wrong the answer is. Yes/no problems use *binary cross-entropy* with a sigmoid output; many-class problems use *categorical cross-entropy* with a **softmax** output (probabilities that add to 1).
* **Backpropagation**: the chain rule, run backwards, gives the gradient of the loss for every weight. For both output types the first step is simply `prediction - truth`.
* **Optimizers**: *gradient descent* steps against the gradient; *momentum* keeps a running velocity; *RMSprop* scales each weight's step by its recent gradient size; **Adam** combines both (with bias correction). Mini-batches (`--batch-size`) use a few rows per step.
* **Train / validation / test**: train rows teach the network; validation rows are only used to decide when to stop and what to tune; test rows are touched once at the end to give an honest score. Splits are stratified (every class keeps its share). Inputs can be standardized using **training** statistics only.
* **Regularization**: `--l2` (penalize big weights), `--dropout` (randomly silence neurons while training), `--early-stopping N` and `--restore-best` (stop when validation loss stops improving and go back to the best weights).
* **Gradient check**: `--gradient-check` compares backpropagation with slowly measured numerical gradients. If they agree, the maths is right.

## Command-line cheat sheet

```powershell
python main.py --dataset moons --hidden 8 --activation tanh
python main.py --dataset spirals --hidden 32 16 --optimizer adam
python main.py --dataset all                      # XOR, AND, OR, NAND together
python main.py --dataset iris --sweep lr=0.001,0.01,0.05 hidden=4,8-4 --sweep-seeds 3
python main.py --dataset digits8x8 --seeds-report 10
python main.py --compare-optimizers --compare-activations
python main.py --config settings.json --log-csv outputs/run.csv
python main.py --save outputs/model          # later: python main.py --load outputs/model
python main.py --animate                      # training GIF (XOR)
```

Datasets: `xor and or nand` (4 rows), `moons circles blobs spirals digits` (generated; `--samples N`), `iris digits8x8` (real, from `data/`). Options you leave out take sensible defaults for the dataset. Command-line values beat `--config` values (a JSON file such as `{"dataset": "iris", "epochs": 300, "hidden": [8]}`). Bad options exit with code 2; runtime problems (for example a missing model file) exit with code 1.

## Real results (Linux, seeds shown)

Measured with `--seeds-report`, using the test rows only for reporting:

| Dataset | Setup | Test accuracy |
|---|---|---|
| Iris | 4-8-3, tanh, Adam, up to 600 epochs with early stopping | mean 96.3%, range 90 to 100% over 20 seeds (seeds 42 to 61) |
| Digits 8x8 | 64-64-10, ReLU, Adam, mini-batch 32 | mean 97.4%, range 96.4 to 98.6% over 10 seeds (seeds 42 to 51) |

With only 30 Iris test rows, one mistake is 3.3 points, so differences of a few percent are noise. Settings were chosen by validation accuracy, not test accuracy.

## Notebook and playground

* **Notebook**: open `notebooks/neural_network_walkthrough.ipynb` in VS Code (install the *Jupyter* extension and `pip install ipykernel`). It is shipped already executed; run it from the project root's environment so `import neural_network` works.
* **Playground**: double-click `web/playground.html`. It re-implements the same maths in JavaScript and lets you pick the logic gate, activation, hidden size, learning rate and seed. It was checked against the Python code for matching losses. Try ReLU with 4 hidden neurons: some seeds get stuck ("dead ReLU"; in Python, 9 of 30 seeds failed XOR, while 8 hidden neurons solved all 30).

## Tests

```powershell
python -m unittest discover -s tests
```

134 tests cover the maths (gradient checks for every activation, analytic first steps of Adam and RMSprop), training controls, saving and loading (including older files), data splits, the real datasets, the CLI, plots, the notebook file and the playground script. Tests write only to temporary folders. At the time of writing all pass on Linux; they have not been run on Windows.

## Honest limitations

* Small networks and small data: fast to read, not meant for large problems. No GPU, no convolutions.
* Digits use 8x8 pictures; this is not the 28x28 MNIST set.
* Hidden layers beyond a few can be hard to train with sigmoid; prefer `relu`/`tanh` with Adam.
* The playground is a separate JavaScript copy of the maths and covers the four logic gates only.
* Tested on Linux with Python 3 and NumPy/Matplotlib; Windows steps above follow standard practice but were not run here.
