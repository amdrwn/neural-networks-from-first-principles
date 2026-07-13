import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import MLP
from activations import ReLU, Linear
from losses import SoftmaxCrossEntropy
from optimizers import Adam
from datasets import make_moons, train_test_split
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

layer_sizes = [2, 16, 16, 16, 16, 2]
initializers = ["zeros", "random_normal", "xavier", "he"]

results = {}

for initializer in initializers:
    rng_model = np.random.default_rng(0)

    activations = [
        ReLU() for _ in range(len(layer_sizes) - 2)
    ] + [Linear()]

    model = MLP(
        layer_sizes,
        activations,
        initializer=initializer,
        rng=rng_model,
    )

    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(lr=0.01)

    history = train(
        model,
        loss_fn,
        optimizer,
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs=200,
        log_every=5,
    )

    results[initializer] = history

    print(
        f"{initializer}: final loss={history['loss'][-1]:.4f}, "
        f"test_acc={history['test_acc'][-1]:.3f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = list(range(0, 200, 5)) + [199]

for initializer, history in results.items():
    axes[0].plot(
        epochs,
        history["loss"],
        label=initializer,
    )
    axes[1].plot(
        epochs,
        history["test_acc"],
        label=initializer,
    )

axes[0].set_title("Training loss by initialisation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8)

axes[1].set_title("Test accuracy by initialisation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "figures",
    "init_comparison.png",
)

plt.savefig(out_path, dpi=120)
print(f"saved figure to {out_path}")