import numpy as np

def make_moons(n_samples=500, noise=0.15, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng

    n_class_a = n_samples // 2
    n_class_b = n_samples - n_class_a

    theta_a = np.linspace(0, np.pi, n_class_a)
    x_a = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)

    theta_b = np.linspace(0, np.pi, n_class_b)
    x_b = np.stack(
        [1 - np.cos(theta_b), 1 - np.sin(theta_b) - 0.5],
        axis=1,
    )

    X = np.vstack([x_a, x_b])
    y = np.concatenate([
        np.zeros(n_class_a, dtype=int),
        np.ones(n_class_b, dtype=int),
    ])

    X += rng.normal(scale=noise, size=X.shape)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.2, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng

    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]