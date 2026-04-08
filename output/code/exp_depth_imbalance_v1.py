"""
Experiment: Depth Dependence of Gradient Imbalance Decomposition
Theory: Theory A -- Why does alpha increase from 1.1 (2L) to 1.7 (8L)?
Prediction: Deeper networks have wider Hessian spectra with more unconverged modes,
            pulling the spectral crossover toward alpha = 2.
Date: 2026-04-07 (Session 4)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "depth_imbalance"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEPTHS = [2, 3, 4, 6, 8]  # Number of layers (including output)
LRS = [0.0001, 0.001, 0.01, 0.1]
N_STEPS = 2000
HIDDEN = 64


def run_depth_imbalance(depth, lr, seed):
    """Track gradient imbalance for a given depth."""
    set_seed(seed)
    if depth == 2:
        model = make_model("mlp_2layer_nobias", 20, 5, hidden=HIDDEN)
    else:
        model = make_model("mlp_deep_nobias", 20, 5, hidden=HIDDEN, depth=depth-1)

    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    n_layers = len(weight_params)

    # Initial conservation quantities
    norms_init = [p.data.norm().item()**2 for _, p in weight_params]
    C_init = [norms_init[l+1] - norms_init[l] for l in range(n_layers-1)]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    grad_imbalances = []
    total_grad_norms = []
    losses = []

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        gn = [p.grad.data.norm().item()**2 for _, p in weight_params]
        # Average gradient imbalance across all consecutive pairs
        delta = np.mean([gn[l+1] - gn[l] for l in range(n_layers-1)])
        grad_imbalances.append(delta)
        total_grad_norms.append(sum(gn))
        losses.append(loss.item())

        optimizer.step()

    # Final conservation
    norms_final = [p.data.norm().item()**2 for _, p in weight_params]
    C_final = [norms_final[l+1] - norms_final[l] for l in range(n_layers-1)]
    actual_drift = np.mean([abs(C_final[l] - C_init[l]) for l in range(n_layers-1)])

    S_eta = abs(sum(grad_imbalances))
    grad_integral = np.trapz(total_grad_norms)

    return {
        "depth": depth, "lr": lr, "seed": seed,
        "actual_drift": float(actual_drift),
        "S_eta": float(S_eta),
        "grad_integral": float(grad_integral),
        "S_over_grad_int": float(S_eta / grad_integral) if grad_integral > 1e-15 else 0,
        "final_loss": float(losses[-1]),
        "n_layers": n_layers,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Depth Dependence of Gradient Imbalance Decomposition")
    print("=" * 70)

    all_results = {}
    t_start = time.time()

    for depth in DEPTHS:
        print(f"\n=== Depth {depth}L ===")
        depth_results = {}

        for lr in LRS:
            drifts, Ss, ratios = [], [], []
            for seed in SEEDS[:3]:
                r = run_depth_imbalance(depth, lr, seed)
                drifts.append(r['actual_drift'])
                Ss.append(r['S_eta'])
                ratios.append(r['S_over_grad_int'])

            mean_drift = np.mean(drifts)
            mean_S = np.mean(Ss)
            mean_ratio = np.mean(ratios)
            print(f"  lr={lr}: drift={mean_drift:.6f}, S={mean_S:.4f}, "
                  f"drift/eta^2={mean_drift/lr**2:.2f}, S/grad_int={mean_ratio:.3f}")

            depth_results[str(lr)] = {
                "lr": lr, "mean_drift": float(mean_drift),
                "mean_S": float(mean_S), "drift_over_eta_sq": float(mean_drift/lr**2),
                "mean_S_over_grad_int": float(mean_ratio),
            }

        # Fit power law for this depth
        lrs_arr = np.array(LRS)
        drifts_arr = np.array([depth_results[str(lr)]["mean_drift"] for lr in LRS])
        mask = drifts_arr > 1e-15
        if mask.sum() >= 2:
            log_lr = np.log(lrs_arr[mask])
            log_d = np.log(drifts_arr[mask])
            alpha, intercept = np.polyfit(log_lr, log_d, 1)
            ss_res = np.sum((log_d - (alpha*log_lr + intercept))**2)
            ss_tot = np.sum((log_d - log_d.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            # Beta from S(eta)
            S_arr = drifts_arr[mask] / lrs_arr[mask]**2
            neg_beta, _ = np.polyfit(log_lr, np.log(S_arr), 1)
        else:
            alpha, r2, neg_beta = 0, 0, 0

        print(f"  --> alpha = {alpha:.3f} (R^2={r2:.3f}), beta = {-neg_beta:.3f}")

        all_results[str(depth)] = {
            "depth": depth, "alpha": float(alpha), "r2": float(r2),
            "beta": float(-neg_beta), "lr_results": depth_results,
        }

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Depth':>6} | {'alpha':>7} | {'R^2':>6} | {'beta':>6} | {'alpha=2-beta':>12}")
    print(f"{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}")
    for depth in DEPTHS:
        r = all_results[str(depth)]
        print(f"{depth:>5}L | {r['alpha']:>7.3f} | {r['r2']:>6.3f} | {r['beta']:>6.3f} | {2-r['beta']:>12.3f}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "session": 4, "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "depths": DEPTHS, "lrs": LRS, "hidden": HIDDEN, "n_steps": N_STEPS,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): alpha vs depth
    alphas = [all_results[str(d)]["alpha"] for d in DEPTHS]
    betas = [all_results[str(d)]["beta"] for d in DEPTHS]
    ax1.plot(DEPTHS, alphas, 'o-', color='darkblue', markersize=8, lw=2)
    ax1.axhline(y=1.0, color='green', ls='--', lw=1, alpha=0.5, label='$\\alpha=1$ (all converged)')
    ax1.axhline(y=2.0, color='red', ls='--', lw=1, alpha=0.5, label='$\\alpha=2$ (none converged)')
    ax1.set_xlabel('Network depth (layers)')
    ax1.set_ylabel('Drift exponent $\\alpha$')
    ax1.set_title('(a) Drift exponent vs depth')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.8, 2.2)

    # Panel (b): S/grad_integral ratio vs depth (Mechanism A stability)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(LRS)))
    for i, lr in enumerate(LRS):
        ratios = [all_results[str(d)]["lr_results"][str(lr)]["mean_S_over_grad_int"] for d in DEPTHS]
        ax2.plot(DEPTHS, ratios, 'o-', color=colors[i], markersize=6, label=f'$\\eta$={lr}')
    ax2.set_xlabel('Network depth (layers)')
    ax2.set_ylabel('$S(\\eta) \\, / \\, \\int\\|\\nabla L\\|^2 dt$')
    ax2.set_title('(b) Gradient imbalance fraction vs depth')
    ax2.legend(fontsize=9)

    fig.suptitle('Depth Dependence of the Drift Exponent', fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig10_depth_imbalance", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
