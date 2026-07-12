import numpy as np

class MSELoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)

    def backward(self):
        return 2 * (self.predictions - self.targets) / self.predictions.size

class SoftmaxCrossEntropy:
    def forward(self, logits, targets):
        self.targets = targets
        N = logits.shape[0]

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        log_likelihood = -np.log(self.probs[np.arange(N), targets] + 1e-12)
        return np.mean(log_likelihood)

    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.targets] -= 1
        return grad / N