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


rng_data = np.random.default_rng(1)

X, y = make_moons(
    n_samples=100,
    noise=0.35,
    rng=rng_data,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    rng=rng_data,
)


def build_plain():
    rng_model = np.random.default_rng(0)

    return MLP(
        [2, 64, 64, 2],
        [ReLU(), ReLU(), Linear()],
        initializer="he",
        rng=rng_model,
    )


def build_l2():
    rng_model = np.random.default_rng(0)

    return MLP(
        [2, 64, 64, 2],
        [ReLU(), ReLU(), Linear()],
        initializer="he",
        l2_lambda=0.05,
        rng=rng_model,
    )


configs = {
    "No regularisation": build_plain(),
    "L2 (lambda=0.05)": build_l2(),
}

results = {}

for name, model in configs.items():
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
        n_epochs=300,
        log_every=5,
    )

    results[name] = history

    gap = history["train_acc"][-1] - history["test_acc"][-1]

    print(
        f"{name}: train_acc={history['train_acc'][-1]:.3f}, "
        f"test_acc={history['test_acc'][-1]:.3f}, "
        f"gap={gap:.3f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = list(range(0, 300, 5)) + [299]

for name, history in results.items():
    axes[0].plot(
        epochs,
        history["train_acc"],
        label=f"{name} (train)",
        linestyle="--",
    )
    axes[0].plot(
        epochs,
        history["test_acc"],
        label=f"{name} (test)",
    )

axes[0].set_title("Train vs test accuracy: overfitting gap")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=7)

for name, history in results.items():
    axes[1].plot(
        epochs,
        history["loss"],
        label=name,
    )

axes[1].set_title("Training loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=8)

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "figures",
    "regularisation_comparison.png",
)

plt.savefig(out_path, dpi=120)
print(f"saved figure to {out_path}")