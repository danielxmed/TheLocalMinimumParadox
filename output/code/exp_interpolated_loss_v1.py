"""
Experiment: Interpolated Loss Function — Smooth MSE-to-CrossEntropy Transition
Theory: Theory A -- Loss-Function Spectral Mechanism
Prediction: Alpha should transition smoothly from ~1.29 (MSE+ReLU) to ~1.07 (CE+ReLU)
            as the loss interpolation parameter increases from 0 to 1. Combined with
            E11 (activation interpolation), this creates a 2D map alpha(eps_act, eps_loss).
Date: 2026-04-07 (Session 6)

Method:
  Define L_eps = (1-eps)*L_MSE + eps*L_CE for eps in [0, 0.2, 0.5, 0.8, 1.0]
  - eps=0: pure MSE loss
  - eps=1: pure CrossEntropy loss
  Measure the drift exponent alpha for each loss interpolation value.
  Extended: run at multiple activation strengths for 2D map.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "interpolated_loss"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Loss interpolation parameter
LOSS_EPSILONS = [0.0, 0.2, 0.5, 0.8, 1.0]
# Activation interpolation for 2D map
ACT_EPSILONS = [0.0, 0.5, 1.0]
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
N_STEPS = 2000
HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200


# ============================================================================
# Network Definitions
# ============================================================================

class InterpolatedActivation(nn.Module):
    """sigma_eps(z) = (1-eps)*z + eps*max(0,z)"""
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        if self.epsilon == 0.0:
            return x
        elif self.epsilon == 1.0:
            return torch.relu(x)
        else:
            return (1.0 - self.epsilon) * x + self.epsilon * torch.relu(x)


class InterpolatedTwoLayer(nn.Module):
    """2-layer network with interpolated activation, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim, act_epsilon):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = InterpolatedActivation(act_epsilon)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.act(self.W1(x)))


# ============================================================================
# Training with Interpolated Loss
# ============================================================================

def measure_drift_interpolated_loss(act_eps, loss_eps, lr, seed, n_steps=N_STEPS):
    """
    Train with interpolated loss and measure conservation drift.
    L = (1-loss_eps)*MSE + loss_eps*CrossEntropy
    """
    set_seed(seed)
    model = InterpolatedTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM, act_eps).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    # Prepare both target formats
    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    grad_imbalances = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)

        # Interpolated loss
        if loss_eps == 0.0:
            loss = F.mse_loss(out, Y_onehot)
        elif loss_eps == 1.0:
            loss = F.cross_entropy(out, Y)
        else:
            loss_mse = F.mse_loss(out, Y_onehot)
            loss_ce = F.cross_entropy(out, Y)
            loss = (1.0 - loss_eps) * loss_mse + loss_eps * loss_ce

        loss.backward()

        gn1 = model.W1.weight.grad.data.norm().item() ** 2
        gn2 = model.W2.weight.grad.data.norm().item() ** 2
        grad_imbalances.append(gn2 - gn1)
        losses.append(loss.item())

        optimizer.step()

        if np.isnan(loss.item()):
            break

    C_final = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    actual_drift = abs(C_final - C_init)
    S_eta = abs(sum(grad_imbalances))

    return {
        "act_epsilon": float(act_eps),
        "loss_epsilon": float(loss_eps),
        "lr": float(lr),
        "seed": int(seed),
        "actual_drift": float(actual_drift),
        "S_eta": float(S_eta),
        "final_loss": float(losses[-1]) if losses and not np.isnan(losses[-1]) else float('inf'),
        "converged": len(losses) == n_steps and not np.isnan(losses[-1]),
    }


def fit_power_law(lrs, drifts):
    """Fit drift ~ eta^alpha and return alpha, R^2, beta."""
    lrs_arr = np.array(lrs)
    drifts_arr = np.array(drifts)
    mask = (drifts_arr > 1e-15) & np.isfinite(drifts_arr)

    if np.sum(mask) >= 3:
        log_lr = np.log(lrs_arr[mask])
        log_drift = np.log(drifts_arr[mask])
        alpha, intercept = np.polyfit(log_lr, log_drift, 1)
        ss_res = np.sum((log_drift - (alpha * log_lr + intercept)) ** 2)
        ss_tot = np.sum((log_drift - log_drift.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Beta: S(eta) ~ eta^beta where S = drift/eta^2
        S_arr = drifts_arr[mask] / lrs_arr[mask] ** 2
        beta, _ = np.polyfit(log_lr, np.log(S_arr), 1)
        return float(alpha), float(r2), float(beta)
    return float('nan'), 0.0, float('nan')


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E13: Interpolated Loss Function — MSE-to-CrossEntropy Transition")
    print("=" * 70)
    print(f"Loss epsilons: {LOSS_EPSILONS}")
    print(f"Activation epsilons: {ACT_EPSILONS}")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    # ---- Phase 1: Primary sweep (ReLU fixed, vary loss) ----
    print("\n=== PHASE 1: Loss interpolation with ReLU (act_eps=1.0) ===")
    primary_results = {}
    for loss_eps in LOSS_EPSILONS:
        print(f"\n--- loss_epsilon = {loss_eps} ---")
        eps_results = {}
        for lr in LRS:
            drifts = []
            S_vals = []
            for seed in USE_SEEDS:
                r = measure_drift_interpolated_loss(1.0, loss_eps, lr, seed)
                drifts.append(r["actual_drift"])
                S_vals.append(r["S_eta"])

            mean_drift = np.mean(drifts)
            mean_S = np.mean(S_vals)
            eps_results[str(lr)] = {
                "mean_drift": float(mean_drift),
                "std_drift": float(np.std(drifts)),
                "mean_S": float(mean_S),
                "drifts": [float(d) for d in drifts],
            }

        drifts_for_fit = [eps_results[str(lr)]["mean_drift"] for lr in LRS]
        alpha, r2, beta = fit_power_law(LRS, drifts_for_fit)

        primary_results[str(loss_eps)] = {
            "loss_epsilon": float(loss_eps),
            "act_epsilon": 1.0,
            "alpha": alpha,
            "r2": r2,
            "beta": beta,
            "per_lr": eps_results,
        }
        print(f"  alpha = {alpha:.4f} (R^2 = {r2:.4f})")

    # ---- Phase 2: 2D grid (vary both activation and loss) ----
    print("\n\n=== PHASE 2: 2D Grid — alpha(act_eps, loss_eps) ===")
    grid_results = {}
    for act_eps in ACT_EPSILONS:
        for loss_eps in LOSS_EPSILONS:
            combo_key = f"act{act_eps}_loss{loss_eps}"

            # Skip if already computed in Phase 1 (act_eps=1.0)
            if act_eps == 1.0:
                grid_results[combo_key] = primary_results[str(loss_eps)]
                continue

            print(f"\n--- act_eps={act_eps}, loss_eps={loss_eps} ---")
            eps_results = {}
            for lr in LRS:
                drifts = []
                S_vals = []
                for seed in USE_SEEDS:
                    r = measure_drift_interpolated_loss(act_eps, loss_eps, lr, seed)
                    drifts.append(r["actual_drift"])
                    S_vals.append(r["S_eta"])

                mean_drift = np.mean(drifts)
                eps_results[str(lr)] = {
                    "mean_drift": float(mean_drift),
                    "std_drift": float(np.std(drifts)),
                    "mean_S": float(np.mean(S_vals)),
                    "drifts": [float(d) for d in drifts],
                }

            drifts_for_fit = [eps_results[str(lr)]["mean_drift"] for lr in LRS]
            alpha, r2, beta = fit_power_law(LRS, drifts_for_fit)

            grid_results[combo_key] = {
                "act_epsilon": float(act_eps),
                "loss_epsilon": float(loss_eps),
                "alpha": alpha,
                "r2": r2,
                "beta": beta,
                "per_lr": eps_results,
            }
            print(f"  alpha = {alpha:.4f} (R^2 = {r2:.4f})")

    t_total = time.time() - t_start

    # ============================================================================
    # Summary
    # ============================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Loss Interpolation (ReLU fixed)")
    print(f"{'='*70}")
    print(f"{'loss_eps':>10} {'alpha':>10} {'R^2':>10}")
    print("-" * 35)
    for loss_eps in LOSS_EPSILONS:
        r = primary_results[str(loss_eps)]
        print(f"{loss_eps:>10.1f} {r['alpha']:>10.3f} {r['r2']:>10.4f}")

    alphas_loss = [primary_results[str(le)]["alpha"] for le in LOSS_EPSILONS]
    alpha_range_loss = max(alphas_loss) - min(alphas_loss)
    print(f"\nalpha range (loss interpolation): {alpha_range_loss:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY: 2D Grid — alpha(act_eps, loss_eps)")
    print(f"{'='*70}")
    print(f"{'act_eps':>10} {'loss_eps':>10} {'alpha':>10} {'R^2':>10}")
    print("-" * 45)
    for act_eps in ACT_EPSILONS:
        for loss_eps in LOSS_EPSILONS:
            combo_key = f"act{act_eps}_loss{loss_eps}"
            r = grid_results.get(combo_key, {})
            a = r.get("alpha", float('nan'))
            r2 = r.get("r2", 0)
            print(f"{act_eps:>10.1f} {loss_eps:>10.1f} {a:>10.3f} {r2:>10.4f}")

    print(f"\nTotal time: {t_total:.1f}s")

    # ============================================================================
    # Save
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 6,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "loss_epsilons": LOSS_EPSILONS,
        "act_epsilons": ACT_EPSILONS,
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "primary_summary": {str(le): primary_results[str(le)]["alpha"] for le in LOSS_EPSILONS},
        "alpha_range_loss": float(alpha_range_loss),
        "total_time_seconds": t_total,
    }
    save_results({"primary": primary_results, "grid": grid_results}, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 16: Interpolated Loss
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    loss_colors = plt.cm.viridis(np.linspace(0, 1, len(LOSS_EPSILONS)))

    # Panel (a): alpha vs loss_eps for ReLU
    ax = axes[0]
    loss_eps_arr = np.array(LOSS_EPSILONS)
    alphas_arr = np.array([primary_results[str(le)]["alpha"] for le in LOSS_EPSILONS])
    ax.plot(loss_eps_arr, alphas_arr, 'ko-', markersize=8, linewidth=2)
    ax.axhline(y=1.11, color='blue', ls='--', lw=1, alpha=0.5, label='Linear+MSE ($\\alpha \\approx 1.11$)')
    ax.axhline(y=1.29, color='red', ls='--', lw=1, alpha=0.5, label='ReLU+MSE ($\\alpha \\approx 1.29$)')
    ax.set_xlabel('Loss interpolation $\\epsilon_{loss}$')
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(a) $\\alpha$ vs loss interpolation (ReLU)')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)

    # Panel (b): S(eta) curves for all loss_eps
    ax = axes[1]
    for i, loss_eps in enumerate(LOSS_EPSILONS):
        r = primary_results[str(loss_eps)]
        valid_lrs = [float(lr) for lr in LRS if r["per_lr"][str(lr)]["mean_S"] > 1e-15]
        valid_S = [r["per_lr"][str(lr)]["mean_S"] for lr in valid_lrs]
        if valid_lrs and valid_S:
            ax.loglog(valid_lrs, valid_S, 'o-', color=loss_colors[i], markersize=5,
                      linewidth=1.5, label=f'$\\epsilon_L$={loss_eps}')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('$S(\\eta)$')
    ax.set_title('(b) $S(\\eta)$ for different loss mixtures')
    ax.legend(fontsize=8)

    # Panel (c): 2D heatmap
    ax = axes[2]
    grid_alpha = np.zeros((len(ACT_EPSILONS), len(LOSS_EPSILONS)))
    for i, act_eps in enumerate(ACT_EPSILONS):
        for j, loss_eps in enumerate(LOSS_EPSILONS):
            combo_key = f"act{act_eps}_loss{loss_eps}"
            r = grid_results.get(combo_key, {})
            grid_alpha[i, j] = r.get("alpha", float('nan'))

    im = ax.imshow(grid_alpha, origin='lower', aspect='auto', cmap='RdYlBu_r',
                    extent=[-0.1, 1.1, -0.1, 1.1])
    ax.set_xticks(LOSS_EPSILONS)
    ax.set_yticks(ACT_EPSILONS)
    ax.set_xlabel('Loss interpolation $\\epsilon_{loss}$')
    ax.set_ylabel('Activation interpolation $\\epsilon_{act}$')
    ax.set_title('(c) $\\alpha(\\epsilon_{act}, \\epsilon_{loss})$')
    plt.colorbar(im, ax=ax, label='$\\alpha$')

    # Annotate the 2D map
    for i, act_eps in enumerate(ACT_EPSILONS):
        for j, loss_eps in enumerate(LOSS_EPSILONS):
            val = grid_alpha[i, j]
            if not np.isnan(val):
                ax.text(loss_eps, act_eps, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if val > 1.15 else 'black')

    fig.suptitle('E13: Interpolated Loss Function — MSE-to-CrossEntropy Transition',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig16_interpolated_loss", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig16_interpolated_loss.pdf")
