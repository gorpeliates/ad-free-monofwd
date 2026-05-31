# MF+DD: Mono-Forward with Directional Derivatives

Code for my bachelor thesis *"Adapting Mono-Forward with Zeroth-Order Gradient Estimation"*.

This repository implements **MF+DD**, which replaces automatic differentiation (AD) in the [Mono-Forward](https://arxiv.org/abs/2501.09238) local learning algorithm with zeroth-order gradient estimation via directional derivatives. The result is a training method free of both global backpropagation and automatic differentiation: each layer is trained independently using only two forward evaluations of its local cross-entropy loss per perturbation direction.

---

## Setup

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

GPU builds are selected automatically: CUDA 12.9 on Linux, CUDA 13.0 on Windows. To use a CPU-only environment, remove the `tool.uv.sources` overrides from `pyproject.toml`.

---

## Usage

### Local run

```bash
uv run python src/run.py \
  --dataset mnist \
  --model mlp \
  --train_method all \
  --epochs 200
```

`--train_method all` runs BP, MF+AD, and MF+DD sequentially. Pass `autodiff`, `dd`, or `backprop` to run a single method.

---
### SLURM 

A slurm script is provided for running experiments on HPC clusters. Example:

```bash
sbatch scripts/train.slurm --dataset fashionmnist --model cnn \
  --train_method dd --dd_num_perturbations 4 --epochs 200 --optimizer sgd
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | `mnist`, `fashionmnist`, `cifar10`, `cifar100`, or `all` |
| `--model` | `mlp` | `mlp` or `cnn` |
| `--train_method` | `all` | `autodiff`, `dd`, `backprop`, `bp_autodiff`, or `all` |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--optimizer` | `sgd` | `sgd` or `adam` (`adam` left as an option for AD and BP, `sgd` used in experiments) |
| `--dd_eps` | `1e-3` | Perturbation magnitude ε |
| `--dd_num_perturbations` | `1` | Perturbation directions P per block per step |
| `--dd_max_params_per_chunk` | `50000` | Max parameters per DD chunk (controls gradient stability) |
| `--cnn_proj_dim` | `16` | FFZero channel-wise random projection dimension |
| `--seed` | `42` | Random seed |
| `--no-early-stopping` | — | Disable early stopping. All experiments are run with this option, on as default for small tests. |
| `--logdir` | `runs` | TensorBoard log directory |

Results are saved to `results/<run_name>.json`. TensorBoard logs are written to `runs/`.


---

## Citation

If you use this code for your research, please cite the paper as follows:
```
@misc{gorpeliates2026mfdd,
  author = {Gorpeliates, Ates},
  title  = {Backpropagation- and Automatic-Differentiation-Free Image Classification
             via Mono-Forward and Zeroth-Order Optimization},
  year   = {2026},
  school = {Delft University of Technology}
}
```