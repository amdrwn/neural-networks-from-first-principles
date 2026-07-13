import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

from activations import Linear, ReLU
from datasets import make_moons, train_test_split
from losses import SoftmaxCrossEntropy
from network import MLP
from optimizers import Adam
from train_utils import train

rng_data = np.random.default_rng(42)

X, y = make_moons(
    n_samples=500,
    noise=0.15,
    rng=rng_data,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    rng=rng_data,
)

rng_model = np.random.default_rng(0)

model = MLP(
    layer_sizes=[2, 16, 16, 2],
    activations=[ReLU(), ReLU(), Linear()],
    initializer="he",
    rng=rng_model,
)

loss_fn = SoftmaxCrossEntropy()
optimizer = Adam(lr=0.01)

train(
    model,
    loss_fn,
    optimizer,
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs=200,
    log_every=50,
)

def activation_pattern(model, points):
    out = points
    patterns = []

    for layer in model.layers:
        out = layer.forward(out)

        if isinstance(layer, ReLU):
            patterns.append((out > 0).astype(int))

    return np.concatenate(patterns, axis=1)

def pattern_to_region_id(patterns):
    unique_patterns = {}
    region_ids = np.empty(len(patterns), dtype=int)

    for i, row in enumerate(map(tuple, patterns)):
        if row not in unique_patterns:
            unique_patterns[row] = len(unique_patterns)

        region_ids[i] = unique_patterns[row]

    return region_ids, len(unique_patterns)

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400),
)

grid_points = np.stack(
    [xx.ravel(), yy.ravel()],
    axis=1,
)

patterns = activation_pattern(model, grid_points)
region_ids, n_regions = pattern_to_region_id(patterns)
region_grid = region_ids.reshape(xx.shape)

logits = model.forward(grid_points)
decision_scores = (logits[:, 1] - logits[:, 0]).reshape(xx.shape)
predictions = (decision_scores > 0).astype(int)

print(
    f"Number of distinct activation patterns found on this grid: "
    f"{n_regions}"
)

rng_colour = np.random.default_rng(7)
colour_permutation = rng_colour.permutation(n_regions)
shuffled_grid = colour_permutation[region_grid]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(
    shuffled_grid,
    extent=(x_min, x_max, y_min, y_max),
    origin="lower",
    cmap="nipy_spectral",
    aspect="auto",
    interpolation="nearest",
)

axes[0].contour(
    xx,
    yy,
    decision_scores,
    levels=[0.0],
    colors="white",
    linewidths=2.5,
)

axes[0].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="k",
    s=15,
    linewidths=0.3,
)

axes[0].set_title(
    f"Activation regions of the trained network\n"
    f"({n_regions} patterns observed on the grid; "
    f"white line = decision boundary)"
)
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

axes[1].contourf(
    xx,
    yy,
    predictions,
    levels=[-0.5, 0.5, 1.5],
    cmap="coolwarm",
    alpha=0.6,
)

axes[1].contour(
    xx,
    yy,
    decision_scores,
    levels=[0.0],
    colors="black",
    linewidths=2.0,
)

axes[1].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="k",
    s=15,
    linewidths=0.3,
)

axes[1].set_title(
    "Piecewise-linear decision boundary\n"
    "(affine within each activation region)"
)
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "figures",
    "piecewise_linear_regions.png",
)

plt.savefig(out_path, dpi=130)
print(f"saved figure to {out_path}")