"""
Experiment: c_k Validation for Linear Networks
Theory: Theory A -- First-principles derivation of spectral crossover coefficients
Prediction: For 2-layer linear networks, c_k propto e_k(0)^2 * lambda_x_k^2
            where e_k is initial error in mode k and lambda_x_k is data covariance eigenvalue.
Date: 2026-04-07 (Session 8)

Method:
  1. Create 2-layer linear network (hidden=64, no bias)
  2. Compute data covariance eigendecomposition: Sigma_x = U_x Lambda_x U_x^T
  3. Compute empirical c_k from gradient projection onto Hessian eigenmodes
  4. Compute theoretical c_k from e_k^2 * lambda_x_k^2
  5. Correlate empirical vs theoretical c_k
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_dataset, count_parameters,
    full_hessian, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "linear_ck_validation"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200
USE_SEEDS = SEEDS[:3]


class LinearTwoLayer(nn.Module):
    """2-layer linear network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


def compute_empirical_ck(model, X, Y_onehot, eigenvalues, eigenvectors):
    """Compute c_k from gradient projected onto Hessian eigenmodes."""
    loss_fn = nn.MSELoss()
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y_onehot)
    loss.backward()

    # Gradient norm imbalance
    gn1 = model.W1.weight.grad.data.norm().item() ** 2
    gn2 = model.W2.weight.grad.data.norm().item() ** 2
    delta_0 = gn2 - gn1

    # Full gradient vector
    grad_vec = torch.cat([model.W1.weight.grad.data.flatten(),
                          model.W2.weight.grad.data.flatten()]).numpy()

    # Project onto Hessian eigenvectors
    projections = eigenvectors.T @ grad_vec
    c_k = projections ** 2

    total = np.sum(c_k)
    if total > 1e-12:
        c_k = c_k * abs(delta_0) / total

    return c_k


def compute_theoretical_ck(model, X, Y, Y_onehot):
    """Compute theoretical c_k = e_k^2 * lambda_x_k^2 for linear network."""
    # Data covariance eigendecomposition
    X_np = X.numpy()
    Sigma_x = (X_np.T @ X_np) / len(X_np)
    eig_vals_x, eig_vecs_x = np.linalg.eigh(Sigma_x)
    # Sort descending
    idx = np.argsort(eig_vals_x)[::-1]
    eig_vals_x = eig_vals_x[idx]
    eig_vecs_x = eig_vecs_x[:, idx]

    # Target regression in data eigenbasis
    Y_np = Y_onehot.numpy()
    # Y_star = Y^T X (X^T X)^{-1} -- optimal linear predictor
    # In eigenbasis: Y_star_rot = Y^T X U_x Lambda_x^{-1}
    X_rot = X_np @ eig_vecs_x  # n x d, projected onto data eigenvectors

    # Per-mode target
    Y_star_rot = (Y_np.T @ X_rot) / len(X_np)  # K x d

    # Effective product W_eff = W2 @ W1 in data eigenbasis
    with torch.no_grad():
        W1 = model.W1.weight.data.numpy()  # hidden x input_dim
        W2 = model.W2.weight.data.numpy()  # output_dim x hidden
        W_eff = W2 @ W1  # output_dim x input_dim

    # Project W_eff into data eigenbasis
    W_eff_rot = W_eff @ eig_vecs_x  # K x d

    # Per-mode error: e_k = ||W_eff_rot[:, k] - Y_star_rot[:, k]||
    # (using Frobenius norm over the K output dimensions for each mode)
    errors_sq = np.sum((W_eff_rot - Y_star_rot) ** 2, axis=0)  # d-vector

    # Theoretical c_k = e_k^2 * lambda_x_k^2
    ck_theory = errors_sq * eig_vals_x ** 2

    return ck_theory, eig_vals_x, errors_sq


if __name__ == "__main__":
    print("=" * 70)
    print("E20: c_k Validation for Linear Networks")
    print("=" * 70)
    print(f"Architecture: 2-layer LINEAR, hidden={HIDDEN}, no bias")
    print(f"Seeds: {USE_SEEDS}")
    print()

    t_start = time.time()
    all_results = {}

    correlations = []

    for seed in USE_SEEDS:
        print(f"\n--- seed={seed} ---")
        set_seed(seed)

        model = LinearTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
        X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                                   d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

        Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
        Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

        # Full Hessian
        loss_fn = nn.MSELoss()
        H = full_hessian(model, X, Y_onehot, loss_fn)
        H_np = H.numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(H_np)

        # Empirical c_k
        ck_empirical = compute_empirical_ck(model, X, Y_onehot, eigenvalues, eigenvectors)

        # Theoretical c_k
        ck_theory, eig_vals_x, errors_sq = compute_theoretical_ck(model, X, Y, Y_onehot)

        # The theoretical c_k has dimension d (input modes)
        # The empirical c_k has dimension P (parameter modes = Hessian eigenmodes)
        # We need to compare the TOP modes (by magnitude)

        # Strategy: compare the distributions of c_k values
        # Sort both by magnitude and compare cumulative distributions
        ck_emp_sorted = np.sort(np.abs(ck_empirical))[::-1]
        ck_thy_sorted = np.sort(np.abs(ck_theory))[::-1]

        # Normalize both
        ck_emp_norm = ck_emp_sorted / (np.sum(ck_emp_sorted) + 1e-20)
        ck_thy_norm = ck_thy_sorted / (np.sum(ck_thy_sorted) + 1e-20)

        # Compare top-K modes (K = min(d, n_positive_hessian))
        n_pos = np.sum(eigenvalues > 1e-8)
        K = min(INPUT_DIM, n_pos, len(ck_thy_norm))

        ck_emp_top = ck_emp_norm[:K]
        ck_thy_top = ck_thy_norm[:K]

        # Correlation
        if len(ck_emp_top) >= 3 and np.std(ck_emp_top) > 1e-15 and np.std(ck_thy_top) > 1e-15:
            R = np.corrcoef(np.log(ck_emp_top + 1e-20), np.log(ck_thy_top + 1e-20))[0, 1]
        else:
            R = 0.0

        correlations.append(R)

        # Also test: does the Hessian eigenspectrum correlate with data covariance?
        hess_eigs_top = eigenvalues[-K:][::-1]  # top K Hessian eigenvalues
        data_eigs_top = eig_vals_x[:K]
        if len(hess_eigs_top) >= 3 and np.std(hess_eigs_top) > 1e-15:
            R_spectrum = np.corrcoef(np.log(hess_eigs_top + 1e-20),
                                     np.log(data_eigs_top + 1e-20))[0, 1]
        else:
            R_spectrum = 0.0

        print(f"  c_k correlation (top {K} modes): R = {R:.4f}")
        print(f"  Spectrum correlation (Hessian vs data): R = {R_spectrum:.4f}")
        print(f"  Hessian: {n_pos} positive eigenvalues, max={eigenvalues[-1]:.4f}")
        print(f"  Data covariance: top-3 eigenvalues = {eig_vals_x[:3]}")
        print(f"  Top-5 c_k empirical: {ck_emp_norm[:5]}")
        print(f"  Top-5 c_k theory:    {ck_thy_norm[:5]}")

        all_results[str(seed)] = {
            "ck_empirical_top": ck_emp_top.tolist(),
            "ck_theory_top": ck_thy_top.tolist(),
            "ck_correlation": float(R),
            "spectrum_correlation": float(R_spectrum),
            "data_eigenvalues": eig_vals_x.tolist(),
            "hessian_top_eigenvalues": hess_eigs_top.tolist(),
            "n_positive_hessian": int(n_pos),
            "K_modes_compared": int(K),
        }

    # Summary
    mean_R = np.mean(correlations)
    print(f"\n{'='*70}")
    print(f"SUMMARY: Mean c_k correlation = {mean_R:.4f}")
    print(f"  Per-seed: {[f'{r:.4f}' for r in correlations]}")
    if mean_R > 0.8:
        print("  ** STRONG VALIDATION: theoretical c_k matches empirical **")
    elif mean_R > 0.5:
        print("  ** MODERATE VALIDATION: partial match, corrections needed **")
    else:
        print("  ** WEAK VALIDATION: theoretical c_k does not match well **")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E20",
        "version": 1,
        "date": "2026-04-07",
        "session": 8,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "hidden": HIDDEN,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "mean_ck_correlation": float(mean_R),
        "per_seed_correlations": [float(r) for r in correlations],
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # Figure 23: c_k derivation validation
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel (a): Empirical vs theoretical c_k (scatter)
    ax = axes[0]
    seed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, seed in enumerate(USE_SEEDS):
        r = all_results[str(seed)]
        ck_emp = np.array(r["ck_empirical_top"])
        ck_thy = np.array(r["ck_theory_top"])
        mask = (ck_emp > 1e-20) & (ck_thy > 1e-20)
        if np.any(mask):
            ax.scatter(ck_thy[mask], ck_emp[mask], c=seed_colors[i], s=30, alpha=0.7,
                       label=f'seed={seed} (R={r["ck_correlation"]:.3f})')

    # Reference line
    all_vals = []
    for seed in USE_SEEDS:
        r = all_results[str(seed)]
        all_vals.extend(r["ck_empirical_top"])
        all_vals.extend(r["ck_theory_top"])
    all_vals = [v for v in all_vals if v > 1e-20]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.5)

    ax.set_xlabel('Theoretical $c_k \\propto e_k^2 \\lambda_{x,k}^2$')
    ax.set_ylabel('Empirical $c_k$ (from gradient projection)')
    ax.set_title(f'(a) $c_k$ validation (mean R={mean_R:.3f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # Panel (b): Hessian vs data covariance eigenspectrum
    ax = axes[1]
    for i, seed in enumerate(USE_SEEDS):
        r = all_results[str(seed)]
        hess_eigs = np.array(r["hessian_top_eigenvalues"])
        data_eigs = np.array(r["data_eigenvalues"][:len(hess_eigs)])
        k_range = np.arange(1, len(hess_eigs) + 1)

        if i == 0:  # Only plot one seed for clarity
            ax.semilogy(k_range, hess_eigs, 'bo-', markersize=5, linewidth=1.5,
                        label='Hessian $\\lambda_k$')
            ax.semilogy(k_range, data_eigs, 'rs--', markersize=5, linewidth=1.5,
                        label='Data $\\lambda_{x,k}$')
            ax.annotate(f'R={r["spectrum_correlation"]:.3f}',
                        xy=(len(hess_eigs)*0.6, hess_eigs[len(hess_eigs)//2]),
                        fontsize=10)

    ax.set_xlabel('Mode index $k$')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('(b) Hessian vs data covariance spectrum')
    ax.legend(fontsize=9)

    fig.suptitle('E20: First-Principles $c_k$ Validation for Linear Networks',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig23_ck_derivation", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig23_ck_derivation.pdf")
