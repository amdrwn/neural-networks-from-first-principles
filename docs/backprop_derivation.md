# Deriving Backpropagation: A Mathematical Walkthrough

This document derives, from first principles, every gradient computed in
this codebase. The guiding idea throughout is the multivariate chain
rule applied to matrix-valued functions, tracked carefully by shape.

## 1. Setup and notation

A feedforward network is a composition of layers:

    X -> Dense_1 -> Activation_1 -> Dense_2 -> Activation_2 -> ... -> Loss

Each Dense layer computes `Z = XW + b`, where for a batch of N examples,
X has shape (N, fan_in), W has shape (fan_in, fan_out), b has shape
(1, fan_out) and broadcasts across the batch, and Z has shape
(N, fan_out).

Backpropagation is repeated application of the chain rule, propagating
the gradient of the loss with respect to a layer's *output* backward
into gradients with respect to that layer's *inputs and parameters*.
We denote the incoming gradient dL/d(layer output) as `grad_output`.

## 2. The Dense layer backward pass

Given `grad_output = dL/dZ`, we need `dL/dW`, `dL/db`, and `dL/dX`.

**dL/dW.** For a single example (row vector x, shape (1, fan_in)):

    z_j = sum_k x_k * W_kj + b_j
    dL/dW_kj = dL/dz_j * dz_j/dW_kj = dL/dz_j * x_k

Written as an outer product: dL/dW = x^T * grad_output. Summing over a
batch of N examples (since the loss is additive across examples):

    dL/dW = X^T @ grad_output          shape: (fan_in, fan_out)

**dL/db.** Since b_j shifts every example's z_j by the same amount,
its gradient sums contributions across the batch:

    dL/db_j = sum_i grad_output[i, j]
    dL/db = sum(grad_output, axis=0)   shape: (1, fan_out)

**dL/dX** (passed to the previous layer):

    dL/dx_k = sum_j dL/dz_j * dz_j/dx_k = sum_j grad_output_j * W_kj
    dL/dX = grad_output @ W^T          shape: (N, fan_in)

**Shape rule of thumb.** In every case, transpose whichever matrix is
*not* being differentiated with respect to, and place it on whichever
side makes the matrix product's shape match the target. This single
heuristic reproduces all three identities above without needing to
re-derive them from index notation each time.

## 3. Activation function backward passes

Activations are elementwise, so their backward pass is simply the
chain rule applied elementwise: `dL/dz = dL/da * da/dz`, where a is
the activation's output.

- **Sigmoid:** a = 1/(1+e^-z), da/dz = a(1-a)
- **Tanh:** a = tanh(z), da/dz = 1 - a^2
- **ReLU:** a = max(0, z), da/dz = 1 if z > 0 else 0
- **LeakyReLU:** a = z if z>0 else alpha*z, da/dz = 1 if z>0 else alpha

Each of these is a one-line elementwise multiplication in code, but the
consequences differ enormously: sigmoid's derivative is bounded above
by 0.25, so in a network with many layers the gradient shrinks
multiplicatively at every layer, causing the vanishing gradient
problem. This is exactly why the activation_comparison experiment
shows sigmoid struggling in a 5-hidden-layer network while ReLU does
not.

## 4. Softmax + cross-entropy: the combined gradient

Naively, one would compute the softmax Jacobian (dense, O(C^2) per
example) and then apply the chain rule through cross-entropy
separately. Instead, we compute the *combined* gradient directly,
which turns out to have a remarkably simple form.

Softmax: p_i = exp(z_i) / sum_k exp(z_k)
Cross-entropy loss (y one-hot): L = -sum_i y_i log(p_i)

Differentiating L with respect to the pre-softmax logit z_i (not p_i)
and using the fact that d(p_i)/d(z_j) = p_i(delta_ij - p_j):

    dL/dz_i = p_i - y_i

This is why `SoftmaxCrossEntropy.backward()` in this codebase simply
returns `(probs - one_hot_targets) / N`, since the entire softmax Jacobian
cancels out algebraically, leaving an expression as simple as linear
regression's residual.

## 5. Weight initialisation: why variance matters

For a Dense layer with fan_in inputs, and assuming inputs and weights
are independent with mean zero:

    Var(z_j) = fan_in * Var(W) * Var(x)

If Var(W) is constant regardless of fan_in, then Var(z) grows or
shrinks with layer width, compounding across depth. This is exactly
what the init_comparison experiment shows for the naive unit-variance
Gaussian initialiser (`random_normal`) versus `xavier`/`he`, which both choose Var(W) inversely proportional to
fan_in (and fan_out, for Xavier) specifically to keep Var(z) roughly
constant across layers regardless of width:

    Xavier: Var(W) = 2 / (fan_in + fan_out)     (derived assuming
             linear/symmetric activations (tanh, sigmoid), balancing
             forward-pass and backward-pass variance)
    He:     Var(W) = 2 / fan_in                 (derived for ReLU,
             which zeroes roughly half its inputs, halving the
             effective variance relative to a linear unit)

The `zeros` initialiser is included specifically to demonstrate a
different failure mode entirely: not vanishing/exploding variance, but
a *symmetry* problem. If every weight starts at zero, every unit in a
layer computes an identical output on every forward pass, and by the
dL/dW derivation above, every unit also receives an identical gradient
on every backward pass, so units never differentiate from each other,
no matter how long training runs. The init_comparison experiment
confirms this: zeros initialisation gets stuck at 43% accuracy
(worse than chance-adjacent for this near-balanced binary problem),
while every other initialiser reaches 98%+.

## 6. L2 regularisation

Adding an L2 penalty `(lambda/2) * ||W||^2` to the loss contributes an
extra `lambda * W` term to dL/dW at every Dense layer, since:

    d/dW [ (lambda/2) * sum(W^2) ] = lambda * W

This is implemented directly in `MLP.backward()` as
`dense.dW += self.l2_lambda * dense.W`, applied after the ordinary
backprop gradient is computed. Intuitively, this term continuously
pulls weights toward zero, competing against the data-fitting
gradient. The regularisation_comparison experiment shows this
closing an overfitting gap from 25 percentage points (train-test
accuracy) down to essentially zero on a small, noisy dataset.

## 7. Connection to piecewise-linear geometry

A network built entirely from Dense layers and ReLU activations
computes a **continuous piecewise-linear function** of its input. 
This is not merely an intuition. It is an exact mathematical statement. 
Composing affine maps with ReLU (itself piecewise linear) yields a function 
that is affine on each cell of a polyhedral partition of the input space, 
determined by which ReLU units are "active" (z > 0) versus "inactive" (z <= 0) 
for a given input.

The `piecewise_linear_regions.py` experiment makes this concrete: for
every point on a fine grid, we record the binary activation pattern of
every ReLU unit in the network. Points sharing an activation pattern
lie in the same linear region, because the network reduces to the
exact same affine composition on all of them. On the trained
two-moons classifier, this reveals 428 distinct activation patterns
(linear regions) on the sampled grid.

The decision boundary itself is piecewise linear too, but its pieces
do not run along the region edges. Within a single region the network
is affine, so the decision boundary there is wherever z1(x) - z0(x) =
0, a straight line that generally cuts through the region's
interior. The boundary only bends where it crosses from one region
into another, since that is where the underlying affine map changes.
The classification boundary is therefore a consequence of the region
structure without literally tracing its edges.

This connects to a broader research direction: Zhang et al. (2018)
show that ReLU networks can be understood as **tropical rational
maps**, ratios of tropical polynomials, where tropical addition is
min (or max) and tropical multiplication is ordinary addition. Under
this correspondence, a network's linear regions correspond to cells
of a polyhedral complex dual to a Newton polytope subdivision, in
exactly the sense that a tropical polynomial's domains of linearity
are dual to a regular subdivision of its Newton polytope. What this
experiment visualises empirically, the tiling of input space into
linear regions, is the network-side shadow of that dual polytope
structure. Making that correspondence computationally explicit (e.g.
extracting the actual tropical polynomial a trained network computes)
is a natural extension of this project beyond its current scope.

## 8. Gradient checking: verifying the derivation empirically

Every trainable parameter gradient derived above can be independently verified using
finite differences, without relying on the chain rule at all. This
matters because a derivation can be correct on paper while its
implementation still contains a bug (a transposed matrix, a sign
error, a missing sum). Deriving and implementing are different
sources of error, and only an independent check catches both.

**The idea.** By the definition of a derivative:

    dL/dw = lim(eps -> 0) [L(w + eps) - L(w - eps)] / (2*eps)

We can't take eps to zero on a computer, but for small eps (typically
1e-5 for float64), this centred-difference estimate is accurate to
several significant figures. "Centred" (using both +eps and -eps)
rather than a one-sided difference matters here: a one-sided estimate
has error proportional to eps, while the centred estimate's error is
proportional to eps^2, so for eps=1e-5, that's the difference between
~1e-5 error and ~1e-10 error.

**Why relative error, not absolute error.** Gradients across a
network can differ by several orders of magnitude (early layers often
have much smaller gradients than later ones, especially with sigmoid
activations. This is the vanishing gradient problem in Section 3
again). An absolute error of 1e-6 is negligible for a gradient of
size 1.0 but enormous for a gradient of size 1e-8. Relative error,

    |analytical - numerical| / max(|analytical|, |numerical|)

normalises for this, and is the standard metric used for gradient
checking. As a rule of thumb, relative error below 1e-5 indicates a
correct implementation (float64 precision limits mean you should not
expect exact agreement); the results in the README show errors in the
1e-7 to 1e-11 range across every parameter in a 3-layer network,
comfortably within that bound.

**What this caught (or would have caught).** Gradient checking is
most useful precisely when a derivation looks right but an off-by-one
transpose or a forgotten batch-sum makes the implementation wrong, a
class of bug that is easy to introduce and easy to miss by inspection,
since the resulting numbers still "look plausible." Running this check
against every Dense layer's W and b, including with L2 regularisation
active (which adds an extra term to dL/dW, per Section 6), confirms
the full backward pass, not just the core Z = XW + b identity, but
its interaction with regularisation, is implemented correctly.

## 9. Interpreting the experiments carefully

Each experiment isolates one variable, but it's worth being precise
about what each result does and doesn't support, since it's easy to
overstate a finding from a single dataset and a handful of runs.

**Optimisers.** SGD, Momentum, and Adam are each run at a different,
reasonable learning rate for that optimiser (0.1 for SGD and Momentum,
0.01 for Adam), not a single shared rate. The result supports
*"Momentum and Adam converge faster and reach higher test accuracy
than plain SGD under these practical configurations."* It does not
isolate optimiser choice as the sole variable, since the learning
rates differ too.

**Learning rate.** This sweep is a cleaner single-variable comparison
(everything else held fixed). It shows 0.001 and 0.01 converging
slowly, 0.5 performing especially well on this problem, 1.5 converging
fast but with a visible transient instability, and 10.0 oscillating
and performing poorly. The correct generalisation is *"an excessively
large learning rate produces unstable oscillation on this problem,"*
not "large learning rates always diverge." That's a property of this
loss landscape and architecture, not a universal law.

**Activations.** Sigmoid and Tanh are paired with Xavier
initialisation, while ReLU and LeakyReLU are paired with He,
matching each activation to its standard compatible initialisation
scheme, rather than holding initialisation fixed across all four. The
result is best described as *"activation functions compared using
their standard compatible initialisation schemes,"* showing sigmoid
training much more slowly in a deeper network while the others
converge effectively.

**Initialisation.** Zero-initialisation fails outright and
unambiguously, for the symmetry reason derived in Section 5.
`random_normal`, however, eventually reaches around 98% test accuracy.
It does not fail, it just starts from a poorly-scaled regime (a
much larger initial loss) and converges less smoothly than Xavier or
He. The accurate claim is *"random_normal begins with a very large
loss and converges less smoothly, while Xavier and He begin
better-scaled; zero initialisation fails outright because symmetry is
never broken,"* not "naive random initialisation fails."

**Piecewise-linear regions.** The count of distinct activation patterns 
(428 in the reported run) is the number of regions detected within the chosen 
plotting domain and sampling resolution. It is a lower bound observed empirically, 
not a proven total count of every linear region the network defines globally. The
qualitative claims, that the decision boundary is piecewise linear,
affine within each activation region and changes slope only when
crossing into a new region, are exact, not approximate; only the
specific count of 428 is grid-dependent.

## References

Zhang, L., Naitzat, G., & Lim, L.-H. (2018). Tropical Geometry of Deep Neural Networks. 
Proceedings of the 35th International Conference on Machine Learning (ICML 2018).
