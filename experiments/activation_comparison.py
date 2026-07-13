import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import MLP
from activations import Sigmoid, Tanh, ReLU, LeakyReLU, Linear
from losses import SoftmaxCrossEntropy
from optimizers import Adam
from datasets import make_moons, train_test_split
from train_utils import train

rng_data = np.random.default_rng(42)
X, y = make_moons(n_samples=500, noise=0.15, rng=rng_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, rng=rng_data)

architectures = {
    "Sigmoid": [Sigmoid() for _ in range(5)] + [Linear()],
    "Tanh": [Tanh() for _ in range(5)] + [Linear()],
    "ReLU": [ReLU() for _ in range(5)] + [Linear()],
    "LeakyReLU": [LeakyReLU(alpha=0.01) for _ in range(5)] + [Linear()],
}

layer_sizes = [2, 16, 16, 16, 16, 16, 2]

results = {}
for name, activations in architectures.items():
    rng_model = np.random.default_rng(0)
    init = "xavier" if name in ("Sigmoid", "Tanh") else "he"
    model = MLP(layer_sizes, activations, initializer=init, rng=rng_model)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(lr=0.01)
    history = train(model, loss_fn, opt, X_train, y_train, X_test, y_test,
                     n_epochs=200, log_every=5)
    results[name] = history
    print(f"{name}: final loss={history['loss'][-1]:.4f}, "
          f"test_acc={history['test_acc'][-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = list(range(0, 200, 5)) + [199]

for name, history in results.items():
    axes[0].plot(epochs, history["loss"], label=name)
    axes[1].plot(epochs, history["test_acc"], label=name)

axes[0].set_title("Training loss by activation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8)

axes[1].set_title("Test accuracy by activation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "figures",
                         "activation_comparison.png")
plt.savefig(out_path, dpi=120)
print(f"saved figure to {out_path}")