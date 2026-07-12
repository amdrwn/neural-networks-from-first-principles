import numpy as np

from network import MLP
from activations import ReLU, Linear
from losses import SoftmaxCrossEntropy
from optimizers import Adam
from datasets import make_moons, train_test_split

rng = np.random.default_rng(42)

X, y = make_moons(n_samples=500, noise=0.15, rng=rng)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    rng=rng,
)

model = MLP(
    layer_sizes=[2, 16, 16, 2],
    activations=[ReLU(), ReLU(), Linear()],
    initializer="he",
    l2_lambda=1e-4,
    rng=rng,
)

loss_fn = SoftmaxCrossEntropy()
optimizer = Adam(lr=0.01)

n_epochs = 200

for epoch in range(n_epochs):
    logits = model.forward(X_train)
    loss = loss_fn.forward(logits, y_train) + model.l2_penalty()

    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step(model.dense_layers)

    if epoch % 20 == 0 or epoch == n_epochs - 1:
        train_logits = model.forward(X_train)
        test_logits = model.forward(X_test)

        train_acc = np.mean(
            np.argmax(train_logits, axis=1) == y_train
        )
        test_acc = np.mean(
            np.argmax(test_logits, axis=1) == y_test
        )

        print(
            f"epoch {epoch:3d} | loss {loss:.4f} | "
            f"train_acc {train_acc:.3f} | test_acc {test_acc:.3f}"
        )