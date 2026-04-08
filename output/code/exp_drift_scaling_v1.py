"""
Experiment: Quantitative Scaling of Conservation Drift with Learning Rate
Theory: Theory A -- Theorem 2 predicts drift ~ eta^2
Prediction: The conservation drift Delta_C should scale as O(eta^2),
            confirming the discretization error as the breaking mechanism.
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "drift_scaling"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Many learning rates for precise scaling measurement
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
N_STEPS = 2000
HIDDEN = 64

def measure_drift(lr, seed, arch="mlp_2layer_nobias"):
    set_seed(seed)
    model = make_model(arch, 20, 5, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50, d=20, K=5, separation=2.0)

    # Record initial conservation quantity
    norms = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms.append(p.data.norm().item() ** 2)
    C_init = norms[1] - norms[0]  # ||W2||^2 - ||W1||^2

    # Train
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    # Final conservation quantity
    norms_final = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms_final.append(p.data.norm().item() ** 2)
    C_final = norms_final[1] - norms_final[0]

    drift = abs(C_final - C_init)
    rel_drift = drift / abs(C_init) if abs(C_init) > 1e-10 else drift
    final_loss = loss.item()

    return drift, rel_drift, final_loss

if __name__ == "__main__":
    print("Conservation Drift Scaling with Learning Rate")
    print(f"Learning rates: {LRS}")
    print(f"Architecture: mlp_2layer_nobias, hidden={HIDDEN}")
    print()

    results = {}
    t_start = time.time()

    for lr in LRS:
        drifts = []
        rel_drifts = []
        losses = []
        for seed in SEEDS[:5]:
            d, rd, l = measure_drift(lr, seed)
            drifts.append(d)
            rel_drifts.append(rd)
            losses.append(l)

        mean_drift = np.mean(drifts)
        std_drift = np.std(drifts)
        mean_rel = np.mean(rel_drifts)
        mean_loss = np.mean(losses)

        print(f"lr={lr:>8.4f}: drift={mean_drift:.6f} +/- {std_drift:.6f}, rel_drift={mean_rel:.6f}, loss={mean_loss:.4f}")

        results[str(lr)] = {
            "lr": lr, "mean_drift": float(mean_drift), "std_drift": float(std_drift),
            "mean_rel_drift": float(mean_rel), "mean_loss": float(mean_loss),
            "drifts": [float(d) for d in drifts],
        }

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Fit power law: drift = a * lr^b
    lrs_arr = np.array(LRS)
    drifts_arr = np.array([results[str(lr)]["mean_drift"] for lr in LRS])

    # Log-log fit (only for non-zero drifts)
    mask = drifts_arr > 1e-10
    if mask.sum() >= 2:
        log_lr = np.log(lrs_arr[mask])
        log_drift = np.log(drifts_arr[mask])
        slope, intercept = np.polyfit(log_lr, log_drift, 1)
        print(f"\n--- POWER LAW FIT ---")
        print(f"drift ~ lr^{slope:.2f}")
        print(f"Predicted: drift ~ lr^2 (discretization error)")
        print(f"Observed exponent: {slope:.2f}")
    else:
        slope = 0
        print("\nInsufficient data for power law fit")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:5], "lrs": LRS,
        "hidden": HIDDEN, "n_steps": N_STEPS, "power_law_exponent": float(slope),
    }
    save_results(results, OUTPUT_DIR, config)

    # Figure: log-log plot of drift vs lr
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): log-log drift vs lr
    ax1.errorbar(lrs_arr, drifts_arr,
                 yerr=[results[str(lr)]["std_drift"] for lr in LRS],
                 fmt='o', capsize=4, color='darkblue', markersize=6)
    if mask.sum() >= 2:
        lr_fit = np.logspace(np.log10(min(LRS)), np.log10(max(LRS)), 100)
        drift_fit = np.exp(intercept) * lr_fit ** slope
        ax1.plot(lr_fit, drift_fit, 'r--', linewidth=1.5,
                label=f'Fit: $\\Delta C \\propto \\eta^{{{slope:.2f}}}$')
        # Also show eta^2 reference
        drift_ref = np.exp(intercept) * lr_fit ** 2 * (lrs_arr[mask][0]**slope / lrs_arr[mask][0]**2)
        ax1.plot(lr_fit, drift_ref, 'g:', linewidth=1, alpha=0.7, label='$\\eta^2$ reference')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Learning Rate $\\eta$')
    ax1.set_ylabel('Conservation Drift $|\\Delta C|$')
    ax1.set_title(f'(a) Drift Scaling: $\\Delta C \\propto \\eta^{{{slope:.2f}}}$')
    ax1.legend()

    # Panel (b): final loss vs lr (to show training quality)
    losses_arr = [results[str(lr)]["mean_loss"] for lr in LRS]
    ax2.plot(lrs_arr, losses_arr, 'o-', color='darkred', markersize=5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Learning Rate $\\eta$')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('(b) Training Performance vs Learning Rate')

    fig.suptitle('Conservation Drift Scaling Law', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig5e_drift_scaling", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
