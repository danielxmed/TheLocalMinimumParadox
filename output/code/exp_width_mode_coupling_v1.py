"""
Experiment: Width Dependence of Mode Coupling
Theory: Theory A -- Testing if delta_alpha = alpha_ReLU - alpha_linear vanishes with width
Prediction: Theorem 6 (perturbative stability) suggests the correction is O(1/sqrt(m)),
            so delta_alpha should decrease with width. If it persists, mode coupling is
            fundamental; if it vanishes, it's a finite-width artifact.
Date: 2026-04-07 (Session 6)

Method:
  For each width in [16, 32, 64, 128, 256]:
    For each activation interpolation eps in [0.0, 0.5, 1.0]:
      Measure alpha from drift ~ eta^alpha
    Compute delta_alpha(width) = alpha(eps=1.0) - alpha(eps=0.0)
  Fit delta_alpha ~ width^(-gamma)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "width_mode_coupling"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WIDTHS = [16, 32, 64, 128, 256]
ACT_EPSILONS = [0.0, 0.5, 1.0]
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
N_STEPS = 2000
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
# Training and Measurement
# ============================================================================

def measure_drift(width, act_eps, lr, seed):
    """Train and measure conservation drift for given width and activation."""
    set_seed(seed)
    model = InterpolatedTwoLayer(INPUT_DIM, width, OUTPUT_DIM, act_eps).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)
    loss_fn = nn.MSELoss()

    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    grad_imbalances = []

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_onehot)
        loss.backward()

        gn1 = model.W1.weight.grad.data.norm().item() ** 2
        gn2 = model.W2.weight.grad.data.norm().item() ** 2
        grad_imbalances.append(gn2 - gn1)

        optimizer.step()

        if np.isnan(loss.item()):
            break

    C_final = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    actual_drift = abs(C_final - C_init)

    return {
        "width": int(width),
        "act_epsilon": float(act_eps),
        "lr": float(lr),
        "seed": int(seed),
        "actual_drift": float(actual_drift),
        "S_eta": float(abs(sum(grad_imbalances))),
        "converged": len(grad_imbalances) == N_STEPS,
        "n_params": INPUT_DIM * width + width * OUTPUT_DIM,
    }


def fit_power_law(lrs, drifts):
    """Fit drift ~ eta^alpha."""
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
        return float(alpha), float(r2)
    return float('nan'), 0.0


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E14: Width Dependence of Mode Coupling")
    print("=" * 70)
    print(f"Widths: {WIDTHS}")
    print(f"Activation epsilons: {ACT_EPSILONS}")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}

    for width in WIDTHS:
        n_params = INPUT_DIM * width + width * OUTPUT_DIM
        print(f"\n{'='*50}")
        print(f"  Width = {width} (params = {n_params})")
        print(f"{'='*50}")

        width_results = {}
        for act_eps in ACT_EPSILONS:
            print(f"\n  --- act_eps = {act_eps} ---")
            eps_data = {}
            for lr in LRS:
                drifts = []
                for seed in USE_SEEDS:
                    r = measure_drift(width, act_eps, lr, seed)
                    drifts.append(r["actual_drift"])
                eps_data[str(lr)] = {
                    "mean_drift": float(np.mean(drifts)),
                    "std_drift": float(np.std(drifts)),
                    "drifts": [float(d) for d in drifts],
                }

            drifts_for_fit = [eps_data[str(lr)]["mean_drift"] for lr in LRS]
            alpha, r2 = fit_power_law(LRS, drifts_for_fit)

            width_results[str(act_eps)] = {
                "act_epsilon": float(act_eps),
                "alpha": alpha,
                "r2": r2,
                "per_lr": eps_data,
            }
            print(f"    alpha = {alpha:.4f} (R^2 = {r2:.4f})")

        all_results[str(width)] = {
            "width": int(width),
            "n_params": n_params,
            "per_epsilon": width_results,
        }

    t_total = time.time() - t_start

    # ============================================================================
    # Analysis
    # ============================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: alpha vs (width, activation)")
    print(f"{'='*70}")
    print(f"{'Width':>8} {'n_params':>10} {'eps=0.0':>10} {'eps=0.5':>10} {'eps=1.0':>10} {'delta_alpha':>12}")
    print("-" * 65)

    delta_alphas = []
    widths_arr = []

    for width in WIDTHS:
        r = all_results[str(width)]
        alphas = {str(e): r["per_epsilon"][str(e)]["alpha"] for e in ACT_EPSILONS}
        delta = alphas["1.0"] - alphas["0.0"]
        delta_alphas.append(delta)
        widths_arr.append(width)
        print(f"{width:>8} {r['n_params']:>10} {alphas['0.0']:>10.3f} {alphas['0.5']:>10.3f} "
              f"{alphas['1.0']:>10.3f} {delta:>12.4f}")

    # Fit delta_alpha ~ width^(-gamma)
    widths_arr = np.array(widths_arr, dtype=float)
    delta_arr = np.array(delta_alphas)

    # Only fit if all values are positive (delta_alpha > 0)
    mask_pos = delta_arr > 0
    if np.sum(mask_pos) >= 3:
        log_w = np.log(widths_arr[mask_pos])
        log_d = np.log(delta_arr[mask_pos])
        neg_gamma, intercept = np.polyfit(log_w, log_d, 1)
        gamma = -neg_gamma
        ss_res = np.sum((log_d - (neg_gamma * log_w + intercept)) ** 2)
        ss_tot = np.sum((log_d - log_d.mean()) ** 2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"\ndelta_alpha ~ width^(-{gamma:.3f}) (R^2 = {r2_fit:.4f})")
        if gamma > 0.3:
            print("  => Mode coupling VANISHES with width (finite-width correction)")
            print(f"     Predicted: O(1/sqrt(m)) gives gamma = 0.5")
        elif gamma > 0.1:
            print("  => Mode coupling DECREASES weakly with width")
        else:
            print("  => Mode coupling PERSISTS at large width (fundamental effect)")
    else:
        gamma, r2_fit = float('nan'), 0.0
        print("\nInsufficient positive delta_alpha values for power-law fit")

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
        "widths": WIDTHS,
        "act_epsilons": ACT_EPSILONS,
        "lrs": LRS,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "delta_alphas": {str(w): float(d) for w, d in zip(WIDTHS, delta_alphas)},
        "gamma": float(gamma) if not np.isnan(gamma) else None,
        "gamma_r2": float(r2_fit),
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 17: Width Dependence
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    width_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(WIDTHS)))

    # Panel (a): alpha vs eps for each width
    ax = axes[0]
    for i, width in enumerate(WIDTHS):
        r = all_results[str(width)]
        eps_arr = np.array(ACT_EPSILONS)
        alpha_arr = np.array([r["per_epsilon"][str(e)]["alpha"] for e in ACT_EPSILONS])
        ax.plot(eps_arr, alpha_arr, 'o-', color=width_colors[i], markersize=7,
                linewidth=2, label=f'm={width}')
    ax.set_xlabel('Activation interpolation $\\epsilon_{act}$')
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(a) $\\alpha$ vs activation strength per width')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)

    # Panel (b): delta_alpha vs width (log-log)
    ax = axes[1]
    mask_plot = np.array(delta_alphas) > 0
    if np.any(mask_plot):
        ax.loglog(np.array(WIDTHS)[mask_plot], np.array(delta_alphas)[mask_plot],
                  'ko-', markersize=8, linewidth=2, label='Measured')

        # Fit line
        if not np.isnan(gamma):
            w_fit = np.logspace(np.log10(min(WIDTHS)*0.8), np.log10(max(WIDTHS)*1.2), 50)
            d_fit = np.exp(intercept) * w_fit ** neg_gamma
            ax.loglog(w_fit, d_fit, 'r--', linewidth=1.5,
                      label=f'Fit: $\\sim m^{{-{gamma:.2f}}}$ (R$^2$={r2_fit:.3f})')

        # Reference lines
        w_ref = np.array([16, 256], dtype=float)
        d_ref_half = delta_alphas[0] * (w_ref / WIDTHS[0]) ** (-0.5)
        ax.loglog(w_ref, d_ref_half, 'b:', linewidth=1, alpha=0.5,
                  label='$O(1/\\sqrt{m})$ reference')

    ax.set_xlabel('Width $m$')
    ax.set_ylabel('$\\Delta\\alpha = \\alpha_{ReLU} - \\alpha_{linear}$')
    ax.set_title('(b) Mode coupling vs width')
    ax.legend(fontsize=9)

    fig.suptitle('E14: Width Dependence of Mode Coupling',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig17_width_mode_coupling", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig17_width_mode_coupling.pdf")
