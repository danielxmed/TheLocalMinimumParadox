"""
Experiment: Drift Exponent in 2-Layer LINEAR Networks (Quadratic Loss Landscape)
Theory: Theory A -- Testing whether the sub-quadratic exponent requires nonlinearity
Prediction: If alpha = 2 for linear networks, the ~1.1 exponent is purely from ReLU.
            If alpha < 2, spectral effects alone create sub-quadratic drift.
Date: 2026-04-07 (Session 4)

For f(x) = W_2 W_1 x (no activation, no bias):
- The loss L(W_1, W_2) = (1/2n)||W_2 W_1 X - Y||^2 is quartic in weights
- Conservation law C = ||W_2||^2 - ||W_1||^2 still holds under gradient flow
- Per-step drift: Delta_C = eta^2 * [||dL/dW_2||^2 - ||dL/dW_1||^2]
- The dynamics are analytically tractable (Saxe et al. 2014)

This experiment measures the drift exponent for the linear case and compares
with the ReLU case (alpha ~ 1.16) to isolate the effect of nonlinearity.
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

EXPERIMENT_NAME = "quadratic_model"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
N_STEPS = 2000
HIDDEN = 64


class LinearTwoLayer(nn.Module):
    """2-layer linear network: f(x) = W_2 W_1 x (no activation, no bias)."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


def measure_drift_linear(lr, seed, n_steps=N_STEPS):
    """Train 2-layer linear network and measure conservation drift."""
    set_seed(seed)
    model = LinearTwoLayer(20, HIDDEN, 5).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    # Convert to regression targets (one-hot for MSE)
    Y_onehot = torch.zeros(Y.shape[0], 5, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    # Initial conservation quantity
    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Track per-step gradient imbalance
    grad_imbalances = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_onehot)
        loss.backward()

        # Per-step gradient imbalance
        gn_W1 = model.W1.weight.grad.data.norm().item()**2
        gn_W2 = model.W2.weight.grad.data.norm().item()**2
        grad_imbalances.append(gn_W2 - gn_W1)
        losses.append(loss.item())

        optimizer.step()

    # Final conservation
    C_final = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    actual_drift = abs(C_final - C_init)

    # S(eta) from decomposition
    S_eta = abs(sum(grad_imbalances))
    predicted_drift = lr**2 * S_eta

    return {
        "lr": lr,
        "actual_drift": float(actual_drift),
        "S_eta": float(S_eta),
        "predicted_drift": float(predicted_drift),
        "decomposition_error": float(abs(actual_drift - predicted_drift) / max(actual_drift, 1e-15)),
        "final_loss": float(losses[-1]),
        "grad_imbalances_downsampled": [float(x) for x in np.array(grad_imbalances)[::10]],
    }


def measure_drift_relu(lr, seed, n_steps=N_STEPS):
    """Train 2-layer ReLU network for comparison."""
    set_seed(seed)
    from utils_v1 import make_model
    model = make_model("mlp_2layer_nobias", 20, 5, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    C_init = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    C_final = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2
    return float(abs(C_final - C_init))


if __name__ == "__main__":
    print("=" * 70)
    print("Drift Exponent: 2-Layer LINEAR vs ReLU Networks")
    print("=" * 70)
    print(f"Architecture: 2-layer, hidden={HIDDEN}, no bias")
    print(f"Data: Gaussian mixture (n=200, d=20, K=5)")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()

    # === LINEAR NETWORK ===
    print("--- LINEAR NETWORK (f = W_2 W_1 x) ---")
    linear_results = {}
    for lr in LRS:
        drifts = []
        S_vals = []
        for seed in SEEDS[:5]:
            r = measure_drift_linear(lr, seed)
            drifts.append(r['actual_drift'])
            S_vals.append(r['S_eta'])

        mean_drift = np.mean(drifts)
        mean_S = np.mean(S_vals)
        print(f"lr={lr:>8.4f}: drift={mean_drift:.6f}, S={mean_S:.4f}, drift/eta^2={mean_drift/lr**2:.4f}")

        linear_results[str(lr)] = {
            "lr": lr,
            "mean_drift": float(mean_drift),
            "std_drift": float(np.std(drifts)),
            "mean_S": float(mean_S),
            "drift_over_eta_sq": float(mean_drift / lr**2),
            "drifts": [float(d) for d in drifts],
        }

    # === ReLU NETWORK (comparison) ===
    print("\n--- ReLU NETWORK (f = W_2 ReLU(W_1 x)) ---")
    relu_results = {}
    # Use fewer LRs for ReLU (we already have full data in drift_scaling experiment)
    relu_lrs = [0.0001, 0.001, 0.01, 0.1]
    for lr in relu_lrs:
        drifts = []
        for seed in SEEDS[:5]:
            d = measure_drift_relu(lr, seed)
            drifts.append(d)

        mean_drift = np.mean(drifts)
        print(f"lr={lr:>8.4f}: drift={mean_drift:.6f}, drift/eta^2={mean_drift/lr**2:.4f}")

        relu_results[str(lr)] = {
            "lr": lr,
            "mean_drift": float(mean_drift),
            "drift_over_eta_sq": float(mean_drift / lr**2),
        }

    # === POWER LAW FITS ===
    lrs_lin = np.array(LRS)
    drifts_lin = np.array([linear_results[str(lr)]["mean_drift"] for lr in LRS])

    mask = drifts_lin > 1e-15
    log_lr = np.log(lrs_lin[mask])
    log_drift = np.log(drifts_lin[mask])
    alpha_linear, int_lin = np.polyfit(log_lr, log_drift, 1)
    ss_res = np.sum((log_drift - (alpha_linear * log_lr + int_lin))**2)
    ss_tot = np.sum((log_drift - log_drift.mean())**2)
    r2_linear = 1 - ss_res / ss_tot

    # S(eta) scaling for linear
    S_lin = drifts_lin[mask] / lrs_lin[mask]**2
    log_S = np.log(S_lin)
    beta_lin, int_beta = np.polyfit(log_lr, log_S, 1)

    # ReLU fit
    lrs_relu = np.array(relu_lrs)
    drifts_relu = np.array([relu_results[str(lr)]["mean_drift"] for lr in relu_lrs])
    mask_r = drifts_relu > 1e-15
    alpha_relu, int_relu = np.polyfit(np.log(lrs_relu[mask_r]), np.log(drifts_relu[mask_r]), 1)

    t_total = time.time() - t_start

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"LINEAR network:  alpha = {alpha_linear:.4f} (R^2 = {r2_linear:.4f})")
    print(f"  S(eta) scaling: S ~ eta^{beta_lin:.4f}, so alpha = 2 + ({beta_lin:.4f}) = {2+beta_lin:.4f}")
    print(f"ReLU network:    alpha = {alpha_relu:.4f}")
    print(f"\nDifference: alpha_ReLU - alpha_LINEAR = {alpha_relu - alpha_linear:.4f}")
    print(f"Total time: {t_total:.1f}s")

    if abs(alpha_linear - 2.0) < 0.15:
        print(f"\n*** LINEAR alpha ≈ 2.0: Sub-quadratic exponent is PURELY from nonlinearity ***")
    else:
        print(f"\n*** LINEAR alpha = {alpha_linear:.2f} ≠ 2.0: Spectral effects contribute ***")

    # === Save ===
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 4,
        "hardware": get_hardware_info(),
        "seeds": SEEDS[:5],
        "lrs_linear": LRS,
        "lrs_relu": relu_lrs,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "alpha_linear": float(alpha_linear),
        "r2_linear": float(r2_linear),
        "alpha_relu": float(alpha_relu),
        "beta_linear": float(beta_lin),
    }
    combined = {"linear": linear_results, "relu": relu_results}
    save_results(combined, OUTPUT_DIR, config)

    # === Figure ===
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): drift vs eta for both
    lr_fit = np.logspace(np.log10(min(LRS)), np.log10(max(LRS)), 100)
    ax1.loglog(lrs_lin, drifts_lin, 'o-', color='darkblue', markersize=7,
               label=f'Linear ($\\alpha$={alpha_linear:.2f})')
    ax1.loglog(lr_fit, np.exp(int_lin) * lr_fit**alpha_linear, '--', color='blue', lw=1.5, alpha=0.6)

    ax1.loglog(lrs_relu, drifts_relu, 's-', color='darkred', markersize=7,
               label=f'ReLU ($\\alpha$={alpha_relu:.2f})')

    # eta^2 reference
    ax1.loglog(lr_fit, lr_fit**2 * drifts_lin[0] / lrs_lin[0]**2, ':', color='gray', lw=1,
               alpha=0.5, label='$\\eta^2$ reference')

    ax1.set_xlabel('Learning rate $\\eta$')
    ax1.set_ylabel('Conservation drift $|\\Delta C|$')
    ax1.set_title('(a) Linear vs ReLU drift scaling')
    ax1.legend(fontsize=10)

    # Panel (b): S(eta) = drift/eta^2 for both
    S_lin_all = drifts_lin / lrs_lin**2
    S_relu_all = drifts_relu / lrs_relu**2

    ax2.loglog(lrs_lin, S_lin_all, 'o-', color='darkblue', markersize=7,
               label=f'Linear: $S \\propto \\eta^{{{beta_lin:.2f}}}$')
    ax2.loglog(lrs_relu, S_relu_all, 's-', color='darkred', markersize=7,
               label='ReLU')
    ax2.axhline(y=S_lin_all[0], color='blue', ls=':', lw=0.8, alpha=0.4)
    ax2.set_xlabel('Learning rate $\\eta$')
    ax2.set_ylabel('$S(\\eta) = \\Delta C / \\eta^2$')
    ax2.set_title('(b) Gradient imbalance sum')
    ax2.legend(fontsize=10)

    fig.suptitle('Linear vs Nonlinear: Isolating the Source of $\\alpha \\approx 1.1$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig8_quadratic_model_comparison", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}")
