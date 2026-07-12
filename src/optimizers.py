import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "params_and_grads"):
                continue
            for param, grad, name in layer.params_and_grads():
                param -= self.lr * grad

class Momentum:
    def __init__(self, lr=0.01, mu=0.9):
        self.lr = lr
        self.mu = mu
        self.velocity = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "params_and_grads"):
                continue
            for param, grad, name in layer.params_and_grads():
                key = (id(layer), name)
                v = self.velocity.get(key, np.zeros_like(param))
                v = self.mu * v - self.lr * grad
                self.velocity[key] = v
                param += v

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "params_and_grads"):
                continue
            for param, grad, name in layer.params_and_grads():
                key = (id(layer), name)
                m = self.m.get(key, np.zeros_like(param))
                v = self.v.get(key, np.zeros_like(param))

                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                self.m[key] = m
                self.v[key] = v

                # bias correction - m/v start at 0 so early steps are skewed
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
