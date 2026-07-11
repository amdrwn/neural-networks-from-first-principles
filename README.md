# Neural Networks from First Principles: Matrix Calculus, Optimisation and Piecewise-Linear Geometry

A feedforward neural network implemented entirely in NumPy. No PyTorch,
TensorFlow or automatic differentiation is used in the core implementation.
Every gradient is derived by hand and verified numerically using finite
differences (see `src/gradient_check.py`).

Most "neural network from scratch" projects stop once the model trains.
This project also looks at the geometry of ReLU networks. A trained ReLU
network is a continuous piecewise-linear function, so every input belongs
to a particular linear region determined by its ReLU activation pattern.
The goal is to recover and visualise these regions directly, connecting
the implementation to the result of Zhang et al. (2018) that ReLU
networks compute tropical rational maps. See
`docs/backprop_derivation.md` (Section 7) and
`experiments/piecewise_linear_regions.py`.

This project is still in progress. The core neural network library and
gradient checking are complete, while the experiments, visualisations
and accompanying write-up will be added over the next few updates.

## What's implemented so far

- **Layers:** Dense (fully connected)
- **Activations:** Sigmoid, Tanh, ReLU, LeakyReLU, Linear and Softmax
  (including the full Jacobian-vector product, not just the combined
  softmax cross-entropy shortcut)
- **Initialisers:** Zeros, random normal, Xavier/Glorot and He
- **Gradient checking:** Analytical gradients verified against centred
  finite-difference estimates, typically agreeing to relative errors of
  around `1e-7`

More to come.