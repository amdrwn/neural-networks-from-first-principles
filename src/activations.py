import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        z = np.asarray(z)
        self.out = np.empty_like(z, dtype=np.result_type(z, np.float64))

        positive = z >= 0
        negative = ~positive

        self.out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[negative])
        self.out[negative] = exp_z / (1.0 + exp_z)

        return self.out

    def backward(self, grad_output):
        local_grad = self.out * (1.0 - self.out)
        return grad_output * local_grad

class Tanh:
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)

class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0.0, z)

    def backward(self, grad_output):
        return grad_output * (self.z > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = alpha

    def forward(self, z):
        self.z = z
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, grad_output):
        local_grad = np.where(self.z > 0, 1.0, self.alpha)
        return grad_output * local_grad

class Linear:
    def forward(self, z):
        return z

    def backward(self, grad_output):
        return grad_output

class Softmax:
    def forward(self, z):
        if z.ndim != 2:
            raise ValueError("expected shape (batch_size, n_classes)")

        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        # d(softmax)_i/d(z)_j = out_i * (delta_ij - out_j)
        # -> Jacobian-vector product:
        #    out * (grad_output - sum(grad_output * out))
        if grad_output.shape != self.out.shape:
            raise ValueError("grad_output must match the softmax output shape")

        weighted_sum = np.sum(grad_output * self.out, axis=1, keepdims=True)
        return self.out * (grad_output - weighted_sum)