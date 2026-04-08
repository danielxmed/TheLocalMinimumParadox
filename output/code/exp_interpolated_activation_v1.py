"""
Experiment: Interpolated Activation Function — Smooth Linear-to-ReLU Transition
Theory: Theory A -- Testing whether alpha changes continuously from linear to ReLU
Prediction: If mode coupling is weak, alpha should be approximately constant across
            all epsilon values (since linear and ReLU both give alpha ~ 1.1).
            If mode coupling matters, there should be a visible transition.
Date: 2026-04-07 (Session 5)

Method:
  Define sigma_eps(z) = (1-eps)*z + eps*max(0,z) for eps in [0, 0.1, 0.2, 0.5, 0.8, 1.0]
  - eps=0: fully linear (identity activation)
  - eps=1: fully ReLU
  Measure the drift exponent alpha for each epsilon value.
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

EXPERIMENT_NAME = "interpolated_activation"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Epsilon values: 0 = linear, 1 = ReLU
EPSILONS = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
N_STEPS = 2000
HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200


class InterpolatedActivation(nn.Module):
    """sigma_eps(z) = (1-eps)*z + eps*max(0,z)
    At eps=0: identity (linear)
    At eps=1: ReLU
    """
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
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = InterpolatedActivation(epsilon)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.act(self.W1(x)))


def measure_drift_interpolated(epsilon, lr, seed, n_steps=N_STEPS):
    """Train interpolated network and measure conservation drift + S(eta)."""
    set_seed(seed)
    model = InterpolatedTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM, epsilon).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    # Use MSE loss for all (to keep consistent loss landscape)
    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)
    loss_fn = nn.MSELoss()

    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    grad_imbalances = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_onehot)
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
        "epsilon": float(epsilon),
        "lr": float(lr),
        "seed": int(seed),
        "actual_drift": float(actual_drift),
        "S_eta": float(S_eta),
        "final_loss": float(losses[-1]) if losses and not np.isnan(losses[-1]) else float('inf'),
        "converged": len(losses) == n_steps and not np.isnan(losses[-1]),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("E11: Interpolated Activation — Linear-to-ReLU Transition")
    print("=" * 70)
    print(f"Epsilons: {EPSILONS}")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}
    for eps in EPSILONS:
        print(f"\n--- epsilon = {eps} ---")
        eps_results = {}
        for lr in LRS:
            drifts = []
            S_vals = []
            for seed in USE_SEEDS:
                r = measure_drift_interpolated(eps, lr, seed)
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

        # Power-law fit: drift ~ eta^alpha
        lrs_arr = np.array(LRS)
        drifts_arr = np.array([eps_results[str(lr)]["mean_drift"] for lr in LRS])
        mask = (drifts_arr > 1e-15) & np.isfinite(drifts_arr)

        if np.sum(mask) >= 3:
            log_lr = np.log(lrs_arr[mask])
            log_drift = np.log(drifts_arr[mask])
            alpha, intercept = np.polyfit(log_lr, log_drift, 1)
            ss_res = np.sum((log_drift - (alpha * log_lr + intercept)) ** 2)
            ss_tot = np.sum((log_drift - log_drift.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # S(eta) scaling
            S_arr = drifts_arr[mask] / lrs_arr[mask] ** 2
            beta, _ = np.polyfit(log_lr, np.log(S_arr), 1)
        else:
            alpha, r2, beta = float('nan'), 0, float('nan')

        all_results[str(eps)] = {
            "epsilon": float(eps),
            "alpha": float(alpha),
            "r2": float(r2),
            "beta": float(beta),
            "per_lr": eps_results,
        }

        print(f"  alpha = {alpha:.4f} (R^2 = {r2:.4f}), S ~ eta^{beta:.3f}")

    t_total = time.time() - t_start

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY: alpha vs epsilon")
    print(f"{'='*70}")
    print(f"{'Epsilon':>10} {'alpha':>10} {'R^2':>10} {'beta':>10}")
    print("-" * 45)
    for eps in EPSILONS:
        r = all_results[str(eps)]
        print(f"{eps:>10.1f} {r['alpha']:>10.3f} {r['r2']:>10.4f} {r['beta']:>10.3f}")

    # Check if alpha is approximately constant
    alphas = [all_results[str(eps)]["alpha"] for eps in EPSILONS]
    alpha_range = max(alphas) - min(alphas)
    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)

    print(f"\nalpha range: {alpha_range:.4f}")
    print(f"alpha mean: {alpha_mean:.4f} +/- {alpha_std:.4f}")

    if alpha_range < 0.15:
        print("\n** ALPHA IS APPROXIMATELY CONSTANT ACROSS EPSILON **")
        print("   The transition from linear to ReLU does NOT change the drift exponent.")
        print("   Mode coupling from ReLU is negligible.")
    elif alpha_range < 0.3:
        print("\n** ALPHA VARIES MILDLY ACROSS EPSILON **")
        print("   Mode coupling has a weak but measurable effect.")
    else:
        print("\n** ALPHA VARIES SIGNIFICANTLY ACROSS EPSILON **")
        print("   Mode coupling from ReLU is important and changes the drift scaling.")

    print(f"\nTotal time: {t_total:.1f}s")

    # ---- Save ----
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 5,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "epsilons": EPSILONS,
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "alpha_summary": {str(eps): all_results[str(eps)]["alpha"] for eps in EPSILONS},
        "alpha_range": float(alpha_range),
        "alpha_mean": float(alpha_mean),
        "alpha_std": float(alpha_std),
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ---- Figure 14: Interpolated Activation ----
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): alpha vs epsilon
    alphas_plot = [all_results[str(eps)]["alpha"] for eps in EPSILONS]
    r2s_plot = [all_results[str(eps)]["r2"] for eps in EPSILONS]

    ax1.plot(EPSILONS, alphas_plot, 'o-', color='darkblue', markersize=10, linewidth=2)
    ax1.fill_between(EPSILONS,
                     [a - 0.05 for a in alphas_plot],
                     [a + 0.05 for a in alphas_plot],
                     alpha=0.2, color='blue')
    ax1.axhline(y=alpha_mean, color='gray', ls='--', lw=1, label=f'Mean: {alpha_mean:.3f}')
    ax1.axhline(y=1.0, color='red', ls=':', lw=1, alpha=0.5, label='$\\alpha = 1$ (converged)')
    ax1.axhline(y=2.0, color='green', ls=':', lw=1, alpha=0.5, label='$\\alpha = 2$ (unconverged)')
    ax1.set_xlabel('$\\varepsilon$ (0 = linear, 1 = ReLU)')
    ax1.set_ylabel('Drift exponent $\\alpha$')
    ax1.set_title(f'(a) $\\alpha$ vs activation nonlinearity (range = {alpha_range:.3f})')
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.05, 1.05)

    # Panel (b): S(eta) curves for different epsilon
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(EPSILONS)))
    for i, eps in enumerate(EPSILONS):
        per_lr = all_results[str(eps)]["per_lr"]
        lrs_valid = [lr for lr in LRS if per_lr[str(lr)]["mean_drift"] > 1e-15]
        S_valid = [per_lr[str(lr)]["mean_drift"] / lr**2 for lr in lrs_valid]
        if lrs_valid:
            ax2.loglog(lrs_valid, S_valid, 'o-', color=cmap[i], markersize=6,
                       label=f'$\\varepsilon$={eps}', linewidth=1.5)

    ax2.set_xlabel('Learning rate $\\eta$')
    ax2.set_ylabel('$S(\\eta) = \\Delta C / \\eta^2$')
    ax2.set_title('(b) Gradient imbalance sum across nonlinearity levels')
    ax2.legend(fontsize=8)

    fig.suptitle('E11: Does the Drift Exponent Change from Linear to ReLU?',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig14_interpolated_activation", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}")
