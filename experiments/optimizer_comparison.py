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
from optimizers import SGD, Momentum, Adam
from datasets import make_moons, train_test_split
from train_utils import train

rng_data = np.random.default_rng(42)
X, y = make_moons(n_samples=500, noise=0.15, rng=rng_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, rng=rng_data)

optimizers = {
    "SGD (lr=0.1)": SGD(lr=0.1),
    "Momentum (lr=0.1, mu=0.9)": Momentum(lr=0.1, mu=0.9),
    "Adam (lr=0.01)": Adam(lr=0.01),
}

results = {}
for name, opt in optimizers.items():
    rng_model = np.random.default_rng(0)  # same init every run
    model = MLP(
        layer_sizes=[2, 16, 16, 2],
        activations=[ReLU(), ReLU(), Linear()],
        initializer="he",
        rng=rng_model,
    )
    loss_fn = SoftmaxCrossEntropy()
    history = train(model, loss_fn, opt, X_train, y_train, X_test, y_test,
                     n_epochs=150, log_every=5)
    results[name] = history
    print(f"{name}: final test_acc={history['test_acc'][-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = list(range(0, 150, 5)) + [149]

for name, history in results.items():
    axes[0].plot(epochs, history["loss"], label=name)
    axes[1].plot(epochs, history["test_acc"], label=name)

axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("loss")
axes[0].legend(fontsize=8)

axes[1].set_title("Test accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "figures",
                         "optimizer_comparison.png")
plt.savefig(out_path, dpi=120)
print(f"saved figure to {out_path}")