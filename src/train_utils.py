import numpy as np

def train(model, loss_fn, optimizer, X_train, y_train, X_test, y_test,
          n_epochs=200, log_every=10):
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        model.set_training(True)

        logits = model.forward(X_train)
        loss = loss_fn.forward(logits, y_train) + model.l2_penalty()

        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step(model.dense_layers)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            model.set_training(False)

            train_logits = model.forward(X_train)
            test_logits = model.forward(X_test)

            train_acc = np.mean(
                np.argmax(train_logits, axis=1) == y_train
            )
            test_acc = np.mean(
                np.argmax(test_logits, axis=1) == y_test
            )

            history["loss"].append(loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

    return history