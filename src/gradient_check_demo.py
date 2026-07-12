import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from network import MLP
from activations import ReLU, Sigmoid, Linear
from losses import SoftmaxCrossEntropy
from gradient_check import (
    check_dense_layer_gradients,
    print_gradient_check_report,
)

rng = np.random.default_rng(0)

X = rng.normal(size=(4, 5))
y = rng.integers(0, 3, size=4)

model = MLP(
    layer_sizes=[5, 6, 4, 3],
    activations=[ReLU(), Sigmoid(), Linear()],
    initializer="xavier",
    l2_lambda=1e-3,
    rng=rng,
)

loss_fn = SoftmaxCrossEntropy()

def full_forward_loss():
    logits = model.forward(X)
    return loss_fn.forward(logits, y) + model.l2_penalty()


loss = full_forward_loss()
grad = loss_fn.backward()
model.backward(grad)

print(f"Initial loss: {loss:.6f}\n")
print("Checking analytical gradients against finite-difference estimates")
print("(centred difference, eps=1e-5, float64)\n")

results = check_dense_layer_gradients(
    model,
    full_forward_loss,
    X,
    y,
    eps=1e-5,
    sample_fraction=1.0,
    rng=rng,
)

all_passed = print_gradient_check_report(results, threshold=1e-5)

sys.exit(0 if all_passed else 1)