"""
Experiment: Adam Optimizer Gradient Imbalance Analysis
Theory: Theory A -- Why does Adam give alpha ~ 0.6 instead of ~1.1?
Prediction: Adam's adaptive per-parameter learning rates change the gradient imbalance
            structure. The effective eta_k = eta / sqrt(v_k) rescales the Hessian spectrum,
            potentially inverting the convergence ordering and producing alpha < 1.
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

EXPERIMENT_NAME = "adam_imbalance"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
N_STEPS = 2000
HIDDEN = 64


def run_optimizer_imbalance(lr, seed, optimizer_type="sgd"):
    """Track gradient imbalance for different optimizers."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", 20, 5, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]

    norms_init = [p.data.norm().item()**2 for _, p in weight_params]
    C_init = norms_init[1] - norms_init[0]

    loss_fn = nn.CrossEntropyLoss()
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    grad_imbalances = []
    # For Adam: track effective update magnitudes per layer
    update_norms_per_layer = [[], []]
    losses = []

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        gn = [p.grad.data.norm().item()**2 for _, p in weight_params]
        delta = gn[1] - gn[0]
        grad_imbalances.append(delta)
        losses.append(loss.item())

        # Snapshot weights before step
        w_before = [p.data.clone() for _, p in weight_params]
        optimizer.step()
        # Actual update magnitudes
        for l, (_, p) in enumerate(weight_params):
            update_norm = (p.data - w_before[l]).norm().item()**2
            update_norms_per_layer[l].append(update_norm)

    norms_final = [p.data.norm().item()**2 for _, p in weight_params]
    C_final = norms_final[1] - norms_final[0]
    actual_drift = abs(C_final - C_init)

    S_eta = abs(sum(grad_imbalances))

    # Update imbalance (for Adam, this differs from gradient imbalance)
    update_imbalance = [update_norms_per_layer[1][t] - update_norms_per_layer[0][t]
                        for t in range(N_STEPS)]
    S_update = abs(sum(update_imbalance))

    return {
        "lr": lr, "optimizer": optimizer_type,
        "actual_drift": float(actual_drift),
        "S_eta_gradient": float(S_eta),
        "S_eta_update": float(S_update),
        "final_loss": float(losses[-1]),
        "mean_update_ratio": float(
            np.mean([update_norms_per_layer[1][t] / max(update_norms_per_layer[0][t], 1e-20)
                     for t in range(N_STEPS)])),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Adam vs SGD Gradient Imbalance Analysis")
    print("=" * 70)

    all_results = {}
    t_start = time.time()

    for opt_type in ["sgd", "adam", "sgd_momentum"]:
        print(f"\n=== {opt_type.upper()} ===")
        opt_results = {}

        for lr in LRS:
            drifts, Ss = [], []
            for seed in SEEDS[:3]:
                r = run_optimizer_imbalance(lr, seed, opt_type)
                drifts.append(r['actual_drift'])
                Ss.append(r['S_eta_gradient'])

            mean_drift = np.mean(drifts)
            mean_S = np.mean(Ss)
            print(f"  lr={lr:>8.4f}: drift={mean_drift:.6f}, S={mean_S:.4f}, "
                  f"drift/eta^2={mean_drift/lr**2:.2f}")

            opt_results[str(lr)] = {
                "lr": lr, "mean_drift": float(mean_drift),
                "std_drift": float(np.std(drifts)),
                "mean_S": float(mean_S),
                "drift_over_eta_sq": float(mean_drift/lr**2),
            }

        # Power law fit
        lrs_arr = np.array(LRS)
        drifts_arr = np.array([opt_results[str(lr)]["mean_drift"] for lr in LRS])
        mask = (drifts_arr > 1e-15) & np.isfinite(drifts_arr)
        if mask.sum() >= 2:
            log_lr = np.log(lrs_arr[mask])
            log_d = np.log(drifts_arr[mask])
            alpha, intercept = np.polyfit(log_lr, log_d, 1)
            ss_res = np.sum((log_d - (alpha*log_lr + intercept))**2)
            ss_tot = np.sum((log_d - log_d.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        else:
            alpha, r2, intercept = 0, 0, 0

        print(f"  --> alpha = {alpha:.3f} (R^2={r2:.3f})")
        all_results[opt_type] = {
            "optimizer": opt_type, "alpha": float(alpha), "r2": float(r2),
            "intercept": float(intercept), "lr_results": opt_results,
        }

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Optimizer':>15} | {'alpha':>7} | {'R^2':>6}")
    print(f"{'-'*15}-+-{'-'*7}-+-{'-'*6}")
    for opt in ["sgd", "sgd_momentum", "adam"]:
        r = all_results[opt]
        print(f"{opt:>15} | {r['alpha']:>7.3f} | {r['r2']:>6.3f}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "session": 4, "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "lrs": LRS, "hidden": HIDDEN, "n_steps": N_STEPS,
        "optimizers": ["sgd", "adam", "sgd_momentum"],
    }
    save_results(all_results, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    opt_colors = {"sgd": "darkblue", "sgd_momentum": "steelblue", "adam": "darkred"}
    opt_markers = {"sgd": "o", "sgd_momentum": "s", "adam": "D"}

    for opt in ["sgd", "sgd_momentum", "adam"]:
        r = all_results[opt]
        d_arr = np.array([r["lr_results"][str(lr)]["mean_drift"] for lr in LRS])
        mask = (d_arr > 1e-15) & np.isfinite(d_arr)
        ax1.loglog(lrs_arr[mask], d_arr[mask], f'{opt_markers[opt]}-',
                   color=opt_colors[opt], markersize=7,
                   label=f'{opt} ($\\alpha$={r["alpha"]:.2f})')
        if mask.sum() >= 2:
            lr_fit = np.logspace(np.log10(lrs_arr[mask].min()), np.log10(lrs_arr[mask].max()), 50)
            ax1.loglog(lr_fit, np.exp(r["intercept"]) * lr_fit**r["alpha"],
                       '--', color=opt_colors[opt], lw=1, alpha=0.5)

    ax1.set_xlabel('Learning rate $\\eta$')
    ax1.set_ylabel('Conservation drift $|\\Delta C|$')
    ax1.set_title('(a) Drift scaling by optimizer')
    ax1.legend(fontsize=10)

    # Panel (b): S(eta) = drift/eta^2
    for opt in ["sgd", "sgd_momentum", "adam"]:
        r = all_results[opt]
        d_arr = np.array([r["lr_results"][str(lr)]["mean_drift"] for lr in LRS])
        S_arr = d_arr / lrs_arr**2
        mask = (S_arr > 1e-15) & np.isfinite(S_arr)
        ax2.loglog(lrs_arr[mask], S_arr[mask], f'{opt_markers[opt]}-',
                   color=opt_colors[opt], markersize=7, label=f'{opt}')

    ax2.set_xlabel('Learning rate $\\eta$')
    ax2.set_ylabel('$S(\\eta) = \\Delta C / \\eta^2$')
    ax2.set_title('(b) Gradient imbalance sum by optimizer')
    ax2.legend(fontsize=10)

    fig.suptitle('Optimizer Dependence of the Drift Exponent', fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig11_adam_imbalance", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
