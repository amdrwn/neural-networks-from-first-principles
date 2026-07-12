import numpy as np
from layers import Dense

class MLP:
    def __init__(self, layer_sizes, activations, initializer="he",
                 l2_lambda=0.0, rng=None):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                "activations must contain one activation per Dense layer"
            )

        self.dense_layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1],
                  initializer=initializer, rng=rng)
            for i in range(len(layer_sizes) - 1)
        ]
        self.activations = activations
        self.l2_lambda = l2_lambda

        self.layers = []
        for dense, act in zip(self.dense_layers, self.activations):
            self.layers.append(dense)
            self.layers.append(act)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        if self.l2_lambda > 0:
            for dense in self.dense_layers:
                dense.dW += self.l2_lambda * dense.W

    def l2_penalty(self):
        if self.l2_lambda == 0:
            return 0.0
        return 0.5 * self.l2_lambda * sum(
            np.sum(d.W ** 2) for d in self.dense_layers
        )

    def set_training(self, mode):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = mode