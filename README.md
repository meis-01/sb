# MNIST 2 -> 8 Schrodinger Bridge

This project trains a compact PyTorch Schrodinger bridge that transports the MNIST digit-2 distribution into the digit-8 distribution. It is modular on purpose: data loading, mini-batch Sinkhorn coupling, Brownian bridge simulation, the neural drift model, training, and sampling live in separate files.

The implementation follows a practical image-to-image bridge recipe:

1. Filter MNIST into source samples `x0 ~ digit 2` and target samples `x1 ~ digit 8`.
2. Build a mini-batch entropic optimal transport coupling with log-domain Sinkhorn.
3. Sample intermediate states from a Brownian bridge,
   `x_t = (1 - t) x0 + t x1 + sigma sqrt(t(1 - t)) noise`.
4. Train a conditional U-Net to predict the target endpoint `x1` from `(x_t, x0, t)`.
5. Transport new 2s by integrating the learned bridge drift `(E[x1 | x_t, x0, t] - x_t) / (1 - t)`.

This is not a heavyweight full IPF implementation. It is an efficient, inspectable approximation: static Schrodinger bridge coupling via entropic OT, dynamic Brownian bridge interpolation, and a learned endpoint-conditioned drift. That makes it suitable for MNIST-scale experimentation while staying faithful to the core SB idea.

## References Checked

- De Bortoli, Thornton, Heng, and Doucet, "Diffusion Schrodinger Bridge with Applications to Score-Based Generative Modeling" ([arXiv](https://arxiv.org/abs/2106.01357), [official code](https://github.com/JTT94/diffusion_schrodinger_bridge)).
- Liu et al., "I2SB: Image-to-Image Schrodinger Bridge" ([PMLR](https://proceedings.mlr.press/v202/liu23ai.html), [official code](https://github.com/NVlabs/I2SB)).
- The project uses the same conceptual ingredients emphasized by those works: path-space KL to a reference diffusion, IPF/Sinkhorn-style entropic coupling, bridge marginals between boundary pairs, and diffusion-style neural training.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have PyTorch installed, install the matching torchvision build for your PyTorch version. The current machine has PyTorch available, but torchvision was not installed at project creation time.

## Train

```powershell
python scripts/train.py --steps 5000 --batch-size 128 --output-dir runs/mnist_2_to_8
```

Useful smaller CPU smoke run:

```powershell
python scripts/train.py --steps 100 --batch-size 32 --base-channels 24 --num-workers 0 --sample-every 100 --checkpoint-every 100
```

Training writes:

- `runs/mnist_2_to_8/config.json`
- `runs/mnist_2_to_8/metrics.csv`
- `runs/mnist_2_to_8/latest.pt`
- preview grids such as `preview_step_000500.png`

Each preview grid is arranged by bridge time: source 2s at the top, transported samples at later bridge times below.

## Sample

```powershell
python scripts/sample.py --checkpoint runs/mnist_2_to_8/latest.pt --output runs/mnist_2_to_8/samples.png --num-samples 16
```

Sampling options:

- `--steps`: Euler steps for transport. More steps are smoother but slower.
- `--eta`: stochasticity in the bridge sampler. Use `0` for deterministic transport, `0.2` to `0.5` for more target-marginal diversity.
- `--split`: `test` by default, so samples come from held-out MNIST 2s.

## Project Layout

```text
sb_mnist/
  bridge.py      Brownian bridge sampling and Euler transport
  config.py      Train/sample dataclasses
  data.py        MNIST digit filtering
  models.py      Conditional U-Net
  sampler.py     Sampling CLI implementation
  sinkhorn.py    Log-domain mini-batch Sinkhorn coupling
  trainer.py     Training loop
  utils.py       EMA, metrics, seeding, device helpers
scripts/
  train.py
  sample.py
docs/
  method.md
tests/
  test_bridge_model.py
  test_sinkhorn.py
```

## Tests

```powershell
python -m pytest
```

The tests cover Sinkhorn marginals, model shape behavior, and bridge sampler shape behavior. They do not download MNIST.

## Notes

- The model predicts the terminal digit-8 image, not class labels.
- The bridge is distributional: a source 2 is transported into the digit-8 distribution, not into a unique supervised target 8.
- `--coupling random` is available as an ablation; `--coupling sinkhorn` is the intended Schrodinger bridge setting.
- For better images, prefer a GPU, more steps, and a larger `base_channels` value.

