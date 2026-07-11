import numpy as np
from initializers import INITIALIZERS

class Dense:
    def __init__(self, fan_in, fan_out, initializer="he", rng=None):
        init_fn = INITIALIZERS[initializer]
        self.W = init_fn(fan_in, fan_out, rng=rng)
        self.b = np.zeros((1, fan_out))
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.X.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW, "W"), (self.b, self.db, "b")]

class Dropout:
    def __init__(self, p=0.5, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.training = True
        self.rng = np.random.default_rng() if rng is None else rng

    def forward(self, X):
        if not self.training or self.p == 0:
            self.mask = np.ones_like(X)
            return X
        self.mask = (self.rng.random(X.shape) > self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask