"""
Experiment: CE-ReLU Interaction at Multiple Widths
Theory: Theory A -- Is the interaction term delta_interaction width-dependent?
Prediction: If delta_interaction ~ -0.21 at width 64 is fundamental, it should persist
            at other widths. If it vanishes, the interaction is a finite-width effect.
Date: 2026-04-07 (Session 7)

Method:
  For each width in [32, 64, 128]:
    For each combination in {MSE, CE} x {Linear, ReLU}:
      Measure alpha from drift ~ eta^alpha
    Compute delta_interaction(width) = alpha(relu_ce) - alpha(relu_mse)
                                      - alpha(linear_ce) + alpha(linear_mse)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "interaction_width"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WIDTHS = [32, 64, 128]
LRS = [0.001, 0.003, 0.01, 0.03, 0.1]
N_STEPS = 2000
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200

COMBOS = [
    ("linear", "mse"),
    ("linear", "ce"),
    ("relu", "mse"),
    ("relu", "ce"),
]


# ============================================================================
# Network Definitions
# ============================================================================

class LinearTwoLayer(nn.Module):
    """2-layer linear network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


class ReLUTwoLayer(nn.Module):
    """2-layer ReLU network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(torch.relu(self.W1(x)))


# ============================================================================
# Training and Measurement
# ============================================================================

def measure_drift(width, arch, loss_type, lr, seed):
    """Train and measure conservation drift."""
    set_seed(seed)

    if arch == "linear":
        model = LinearTwoLayer(INPUT_DIM, width, OUTPUT_DIM).to(DEVICE)
    else:
        model = ReLUTwoLayer(INPUT_DIM, width, OUTPUT_DIM).to(DEVICE)

    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
        Y_train = Y_onehot
    else:
        loss_fn = nn.CrossEntropyLoss()
        Y_train = Y

    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    grad_imbalances = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_train)
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
        "actual_drift": float(actual_drift),
        "S_eta": float(abs(sum(grad_imbalances))),
        "converged": len(grad_imbalances) == N_STEPS,
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
    print("E17: CE-ReLU Interaction at Multiple Widths")
    print("=" * 70)
    print(f"Widths: {WIDTHS}")
    print(f"Combinations: {COMBOS}")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}

    for width in WIDTHS:
        n_params = INPUT_DIM * width + width * OUTPUT_DIM
        print(f"\n{'='*60}")
        print(f"  Width = {width} (params = {n_params})")
        print(f"{'='*60}")

        width_results = {}
        for arch, loss_type in COMBOS:
            combo = f"{arch}_{loss_type}"
            print(f"\n  --- {combo} ---")

            drifts_per_lr = {}
            for lr in LRS:
                drifts = []
                for seed in USE_SEEDS:
                    r = measure_drift(width, arch, loss_type, lr, seed)
                    drifts.append(r["actual_drift"])
                drifts_per_lr[str(lr)] = {
                    "mean_drift": float(np.mean(drifts)),
                    "std_drift": float(np.std(drifts)),
                    "drifts": [float(d) for d in drifts],
                }

            mean_drifts = [drifts_per_lr[str(lr)]["mean_drift"] for lr in LRS]
            alpha, r2 = fit_power_law(LRS, mean_drifts)

            width_results[combo] = {
                "alpha": alpha,
                "r2": r2,
                "per_lr": drifts_per_lr,
            }
            print(f"    alpha = {alpha:.4f} (R^2 = {r2:.4f})")

        all_results[str(width)] = width_results

    t_total = time.time() - t_start

    # ============================================================================
    # Analysis: Interaction Term at Each Width
    # ============================================================================
    print(f"\n{'='*70}")
    print("THREE-FACTOR DECOMPOSITION AT EACH WIDTH")
    print(f"{'='*70}")

    print(f"\n{'Width':>8} {'lin_mse':>10} {'lin_ce':>10} {'relu_mse':>10} {'relu_ce':>10} {'d_act':>10} {'d_loss':>10} {'d_inter':>10}")
    print("-" * 90)

    interaction_data = {}
    for width in WIDTHS:
        r = all_results[str(width)]
        alphas = {combo: r[combo]["alpha"] for combo in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]}

        # Three-factor decomposition
        delta_activation = alphas["relu_mse"] - alphas["linear_mse"]
        delta_loss = alphas["linear_ce"] - alphas["linear_mse"]
        delta_interaction = (alphas["relu_ce"] - alphas["relu_mse"]) - (alphas["linear_ce"] - alphas["linear_mse"])

        interaction_data[str(width)] = {
            "width": int(width),
            "alphas": {k: float(v) for k, v in alphas.items()},
            "delta_activation": float(delta_activation),
            "delta_loss": float(delta_loss),
            "delta_interaction": float(delta_interaction),
            "total_change": float(alphas["relu_ce"] - alphas["linear_mse"]),
        }

        print(f"{width:>8} {alphas['linear_mse']:>10.3f} {alphas['linear_ce']:>10.3f} "
              f"{alphas['relu_mse']:>10.3f} {alphas['relu_ce']:>10.3f} "
              f"{delta_activation:>10.3f} {delta_loss:>10.3f} {delta_interaction:>10.3f}")

    # Check if interaction is width-dependent
    interactions = [interaction_data[str(w)]["delta_interaction"] for w in WIDTHS]
    mean_interaction = np.mean(interactions)
    std_interaction = np.std(interactions)

    print(f"\n--- INTERACTION STABILITY ---")
    print(f"  delta_interaction values: {[f'{x:.3f}' for x in interactions]}")
    print(f"  Mean: {mean_interaction:.3f} +/- {std_interaction:.3f}")
    print(f"  Coefficient of variation: {abs(std_interaction/mean_interaction):.2f}" if abs(mean_interaction) > 1e-6 else "")

    if abs(std_interaction / mean_interaction) < 0.3 if abs(mean_interaction) > 1e-6 else False:
        print(f"\n** INTERACTION IS WIDTH-INDEPENDENT (CV < 0.3) **")
        print(f"   delta_interaction ~ {mean_interaction:.3f} is a FUNDAMENTAL constant")
    else:
        print(f"\n** INTERACTION SHOWS WIDTH DEPENDENCE **")
        print(f"   Further investigation needed")

    print(f"\nTotal time: {t_total:.1f}s")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 7,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "widths": WIDTHS,
        "lrs": LRS,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "combinations": [f"{a}_{l}" for a, l in COMBOS],
        "interaction_data": interaction_data,
        "mean_interaction": float(mean_interaction),
        "std_interaction": float(std_interaction),
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 20: Interaction at Multiple Widths
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {
        "linear_mse": "darkblue",
        "linear_ce": "royalblue",
        "relu_mse": "darkred",
        "relu_ce": "orangered",
    }
    markers = {
        "linear_mse": "o",
        "linear_ce": "D",
        "relu_mse": "s",
        "relu_ce": "^",
    }
    labels_map = {
        "linear_mse": "Linear + MSE",
        "linear_ce": "Linear + CE",
        "relu_mse": "ReLU + MSE",
        "relu_ce": "ReLU + CE",
    }

    # Panel (a): All 4 alphas vs width
    ax = axes[0]
    for combo in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        alphas_plot = [interaction_data[str(w)]["alphas"][combo] for w in WIDTHS]
        ax.plot(WIDTHS, alphas_plot, f'{markers[combo]}-', color=colors[combo],
                markersize=8, linewidth=2, label=labels_map[combo])

    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(a) $\\alpha$ vs width for all 4 combinations')
    ax.legend(fontsize=9)
    ax.axhline(y=1.0, color='gray', ls=':', lw=0.8, alpha=0.5)

    # Panel (b): Interaction term vs width
    ax = axes[1]
    inter_vals = [interaction_data[str(w)]["delta_interaction"] for w in WIDTHS]
    d_act_vals = [interaction_data[str(w)]["delta_activation"] for w in WIDTHS]
    d_loss_vals = [interaction_data[str(w)]["delta_loss"] for w in WIDTHS]

    ax.plot(WIDTHS, inter_vals, 'ko-', markersize=10, linewidth=2.5, label='$\\delta_{interaction}$', zorder=5)
    ax.plot(WIDTHS, d_act_vals, 's--', color='darkred', markersize=7, linewidth=1.5,
            label='$\\delta_{activation}$', alpha=0.7)
    ax.plot(WIDTHS, d_loss_vals, 'D--', color='royalblue', markersize=7, linewidth=1.5,
            label='$\\delta_{loss}$', alpha=0.7)

    # Mean interaction band
    ax.axhline(y=mean_interaction, color='black', ls=':', lw=1.5, alpha=0.5)
    ax.fill_between(WIDTHS, mean_interaction - std_interaction, mean_interaction + std_interaction,
                     color='gray', alpha=0.15, label=f'Mean $\\pm$ std: {mean_interaction:.3f}$\\pm${std_interaction:.3f}')

    ax.axhline(y=0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Decomposition terms')
    ax.set_title('(b) Three-factor decomposition vs width')
    ax.legend(fontsize=8)

    fig.suptitle('E17: CE-ReLU Interaction at Multiple Widths — Is the Interaction Fundamental?',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig20_interaction_width", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig20_interaction_width.pdf")
