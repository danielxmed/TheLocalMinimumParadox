# The Local Minimum Paradox

**Conservation Law Breaking at the Edge of Stability: A Spectral Theory of Non-Convex Neural Network Optimization**

[arXiv preprint (coming soon)]()

---

## The Research Question

Why does gradient descent reliably find good solutions in deep neural networks when the optimization landscape is provably NP-hard in the worst case? The loss surface has exponentially many critical points, and training is a non-convex optimization problem -- yet simple SGD works brilliantly in practice. This is the **Local Minimum Paradox**.

## Our Resolution

We show that **conservation laws** from the network's symmetry group serve as "guide rails" constraining optimization trajectories to structured submanifolds. Their **structured breaking** at the Edge of Stability is the mechanism that enables escape from bad regions of the landscape.

### Three Theories

| Theory | Status | Quality Gate |
|--------|--------|-------------|
| **A: Structured Conservation Law Breaking** | Complete first-principles theory | 8/8 |
| B: Percolation Phase Transition | Validated | 7/8 |
| C: Tropical Morse Theory | Validated | 7/8 |

### Key Theorems (Theory A)

| # | Theorem | Status |
|---|---------|--------|
| 1 | Conservation Laws: C_l = \|\|W_{l+1}\|\|^2 - \|\|W_l\|\|^2 conserved under gradient flow | **Proved** |
| 2' | Mean-Field Quasi-Convexity on M_C (2-layer, infinite width) | **Proved** |
| 3 | Drift Decomposition: drift = eta^2 * S(eta) | **Proved** |
| 4 | Linear networks give alpha = 1.10 (spectral, not nonlinearity) | **Proved** |
| 5 | Spectral Crossover Formula for S(eta) | **Proved** |
| 5b | Time-Dependent Spectral Crossover (CE Hessian compression) | **Proved** |
| 5b-i | Spectral Compression Bound | **Proved** |
| c_k | Mode coefficients: c_k proportional to e_k^2 * lambda_{x,k}^2 | **Proved + Validated** |
| 6' | EoS/Sub-EoS Dichotomy | **Empirical** |
| tau | tau = Theta(1/eta), n-independent | **Derived + Validated** |

## Experiments

23 experiments validating every prediction, all reproducible with fixed seeds.

| # | Name | Key Result |
|---|------|-----------|
| E1 | Conservation verification | Drift < 0.003% |
| E2 | Conservation with bias | Bias breaks conservation |
| E3 | Drift vs learning rate | Drift ~ eta scaling |
| E4 | EoS conservation breaking | 5500x drift increase at EoS |
| E5 | Drift scaling law | alpha = 1.16, R^2 > 0.99 |
| E6 | Depth dependence | alpha: 1.07 (2L) to 1.72 (8L) |
| E7 | Optimizer dependence | Adam: alpha = 0.56 |
| E8 | Spectral universality | 14-27% prediction error |
| E9-E11 | Linear-ReLU gap | 2.2% switch rate, smooth alpha transition |
| E12-E14 | Loss function interaction | Non-additive three-factor decomposition |
| E15 | Width switch rate | Per-neuron rate width-independent at EoS |
| E16-E17 | Time-dependent Hessian | CE R=0.988 at t=250; CE clamps alpha near 1.0 |
| E18 | CE Hessian evolution | 24x compression, n-independent decay rate |
| E19 | MSE fine width sweep | alpha-1 ~ width^1.18 |
| E20 | Linear c_k validation | R = 0.847 |
| E21 | ReLU c_k validation | R > 0.80 at all learning rates |
| E22 | Width-dimension transition | Transition depends on overparameterization, not m/d |
| E23 | tau vs learning rate | tau = 1.33/eta + 29, R^2 = 0.988 |

## Repository Structure

```
arxiv_submission/          # Paper (LaTeX source + figures)
output/
  theories/                # Formal theory documents
  code/                    # Experiment scripts + shared utilities
  experiments/             # Results (JSON configs + results)
  figures/                 # Publication-quality figures (PDF + PNG)
  wiki/                    # Compiled knowledge base
  literature/              # Verified papers + bibliography
context/                   # Research foundations
methodology/               # Creative thinking + proof standards
rules/                     # Research integrity rules
```

## Reproducing Results

```bash
# All experiments use pyenv Python 3.12.7
~/.pyenv/versions/3.12.7/bin/python output/code/exp_conservation_laws_v1.py

# Seeds: [42, 137, 256, 512, 1024]
# Hardware: Intel i5-1038NG7, 16GB RAM, CPU only
# Software: PyTorch 2.2.2, Python 3.12.7
```

Each experiment script saves `config.json` (full configuration) and `results.json` (processed results) to `output/experiments/<name>/`.

## Citation

```bibtex
@article{nobregamedeiros2026conservation,
  title={Conservation Law Breaking at the Edge of Stability: A Spectral Theory of Non-Convex Neural Network Optimization},
  author={Nobrega Medeiros, Daniel},
  journal={arXiv preprint},
  year={2026}
}
```

## Author

**Daniel Nobrega Medeiros**
Physician (M.D.) | Neurologist | AI Researcher | MSc Student, University of Colorado Boulder

- GitHub: [danielxmed](https://github.com/danielxmed)
- HuggingFace: [tylerxdurden](https://huggingface.co/tylerxdurden)
- LinkedIn: [daniel-nobrega-dnm](https://linkedin.com/in/daniel-nobrega-dnm)

## License

This research is shared for academic purposes. See the repository for details.
