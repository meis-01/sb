# Method

## Problem

We want a stochastic process `(X_t)_{t in [0,1]}` whose initial marginal is the MNIST digit-2 distribution and whose terminal marginal is the MNIST digit-8 distribution. A Schrodinger bridge solves this by finding the path measure closest in KL divergence to a reference diffusion while matching both endpoint marginals:

```text
min_pi KL(pi || P_ref)
subject to pi_0 = p_digit_2 and pi_1 = p_digit_8.
```

With a Brownian reference, the static endpoint version is an entropic optimal transport problem. On a finite mini-batch, this endpoint coupling is solved with Sinkhorn.

## Approximation Used Here

The exact high-dimensional bridge is expensive. This project uses an efficient image-scale approximation inspired by Diffusion Schrodinger Bridge (DSB) and I2SB:

1. Draw a batch of source images `x0` from digit 2 and target candidates `y` from digit 8.
2. Compute the mean squared image cost `C_ij = ||x0_i - y_j||^2 / d`.
3. Solve a mini-batch entropic OT problem:

   ```text
   gamma = argmin_gamma <gamma, C> + epsilon * sum gamma_ij log(gamma_ij)
   subject to gamma 1 = uniform and gamma^T 1 = uniform.
   ```

4. Sample target endpoints `x1` from each row of `gamma`.
5. For a random `t`, sample the conditional Brownian bridge marginal:

   ```text
   X_t | x0, x1 ~ Normal((1 - t) x0 + t x1, sigma^2 t(1 - t) I).
   ```

6. Train a network `m_theta(x_t, x0, t)` with MSE loss:

   ```text
   L(theta) = E ||m_theta(X_t, x0, t) - x1||^2.
   ```

The MSE minimizer estimates `E[X_1 | X_t, X_0, t]`. For a Brownian bridge to a known terminal point, the forward drift is `(x1 - x_t) / (1 - t)`. Replacing `x1` by the learned conditional expectation gives the learned bridge drift used at sampling time:

```text
dX_t = (m_theta(X_t, X_0, t) - X_t) / (1 - t) dt + eta * sigma dB_t.
```

`eta` controls sampling noise. Setting `eta = 0` gives a deterministic transport-like flow; positive `eta` keeps the stochastic character of the bridge.

## Why Sinkhorn Matters

Randomly pairing 2s and 8s trains a noisy bridge, but it ignores the static Schrodinger bridge endpoint structure. Sinkhorn gives an entropy-regularized coupling that prefers close image pairs while preserving both empirical marginals. In the Brownian reference case, the endpoint kernel has the form `exp(-cost / epsilon)`, so the Sinkhorn plan is the finite-sample static bridge counterpart.

## Relationship To The Papers

- DSB formulates generative modeling as a Schrodinger bridge and approximates IPF with score/drift learning. This project uses the same bridge objective viewpoint but avoids a full alternating IPF loop for MNIST simplicity.
- I2SB shows that, given boundary pairs, bridge marginals can be sampled analytically and trained with diffusion-model techniques. This project uses that paired-boundary bridge marginal with endpoint pairs produced by entropic OT rather than a supervised restoration dataset.

## Practical Hyperparameters

- `bridge_sigma`: Larger values create blurrier, more stochastic intermediate states. Good MNIST range: `0.3` to `0.6`.
- `sinkhorn_epsilon`: Lower values make pairings closer to hard OT but less smooth. Good MNIST range: `0.03` to `0.1`.
- `sample_eta`: Sampling noise. Use `0.0` for crisp deterministic examples, `0.2` to `0.5` for diversity.
- `base_channels`: Main model width. `24` is fine for CPU smoke tests; `48` or `64` is better for real training.

## Validation Ideas

The repository includes lightweight tests. For model quality, train a small MNIST classifier and measure whether transported samples are classified as 8 while source samples are classified as 2. Also inspect preview grids: the bridge should close loops and thicken strokes gradually instead of jumping directly from 2 to 8.

