import numpy as np


def zeros(fan_in, fan_out, rng=None):
    return np.zeros((fan_in, fan_out))


def random_normal(fan_in, fan_out, scale=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0, scale, size=(fan_in, fan_out))


def xavier(fan_in, fan_out, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, size=(fan_in, fan_out))


def he(fan_in, fan_out, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=(fan_in, fan_out))


INITIALIZERS = {
    "zeros": zeros,
    "random_normal": random_normal,
    "xavier": xavier,
    "he": he,
}