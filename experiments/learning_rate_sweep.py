import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import MLP
from activations import ReLU, Linear
from losses import SoftmaxCrossEntropy
from optimizers import SGD
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

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.5, 10.0]

results = {}

for lr in learning_rates:
    rng_model = np.random.default_rng(0)

    model = MLP(
        layer_sizes=[2, 16, 16, 2],
        activations=[ReLU(), ReLU(), Linear()],
        initializer="he",
        rng=rng_model,
    )

    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=lr)

    history = train(
        model,
        loss_fn,
        optimizer,
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs=150,
        log_every=5,
    )

    results[lr] = history

    print(
        f"lr={lr}: final loss={history['loss'][-1]:.4f}, "
        f"test_acc={history['test_acc'][-1]:.3f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = list(range(0, 150, 5)) + [149]

for lr, history in results.items():
    axes[0].plot(epochs, history["loss"], label=f"lr={lr}")
    axes[1].plot(epochs, history["test_acc"], label=f"lr={lr}")

axes[0].set_title("Training loss vs learning rate")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_ylim(0, 2)
axes[0].legend(fontsize=8)

axes[1].set_title("Test accuracy vs learning rate")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "figures",
    "learning_rate_sweep.png",
)

plt.savefig(out_path, dpi=120)

print(f"saved figure to {out_path}")