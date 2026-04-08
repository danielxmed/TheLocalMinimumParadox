"""
Experiment: Width-Dependent Activation Switch Rate
Theory: Theory A -- Testing Hypothesis A for why mode coupling increases with width
Prediction: Per-neuron switch rate decreases as width^(-beta) where beta ~ 0.5,
            but total switches per step grow as width^(1-beta) ~ sqrt(m).
            If confirmed, this explains why delta_alpha does NOT vanish with width
            (E14 finding) -- the total mode coupling grows even as per-neuron rates drop.
Date: 2026-04-07 (Session 7)

Method:
  For each width in [16, 32, 64, 128, 256]:
    For each learning rate in [0.001, 0.01, 0.1]:
      Train 2-layer ReLU network (no bias) with MSE loss
      At every step, track:
        - Per-neuron switch rate (fraction of (neuron, sample) pairs that change)
        - Total switches per step (absolute count)
        - Number of neurons that changed for at least one sample
      Fit per_neuron_rate ~ width^(-beta)
      Compute total_switches ~ width^(1-beta)
      Correlate with delta_alpha from E14
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

EXPERIMENT_NAME = "width_switch_rate"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WIDTHS = [16, 32, 64, 128, 256]
LRS = [0.001, 0.01, 0.1]
N_STEPS = 2000
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200


# ============================================================================
# Network Definition (ReLU only, no bias)
# ============================================================================

class ReLUTwoLayer(nn.Module):
    """2-layer ReLU network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(torch.relu(self.W1(x)))


# ============================================================================
# Activation Tracking
# ============================================================================

def get_activation_pattern(model, X):
    """Binary activation pattern: 1 if neuron fires, 0 otherwise."""
    with torch.no_grad():
        pre_act = X @ model.W1.weight.data.t()  # (n_samples, hidden)
        return (pre_act > 0).int()


def run_switch_tracking(width, lr, seed):
    """Train ReLU network and track activation switch dynamics at every step."""
    set_seed(seed)
    model = ReLUTwoLayer(INPUT_DIM, width, OUTPUT_DIM).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    prev_pattern = get_activation_pattern(model, X)

    # Tracking arrays
    per_neuron_rates = []       # fraction of (neuron, sample) pairs changed
    total_switches_list = []    # absolute count of changed pairs
    neurons_changed_list = []   # number of neurons with at least one change

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_onehot)
        loss.backward()
        optimizer.step()

        if np.isnan(loss.item()):
            break

        curr_pattern = get_activation_pattern(model, X)
        diff = (prev_pattern != curr_pattern).float()

        # Per-neuron switch rate: fraction of all (neuron, sample) pairs
        total_changes = diff.sum().item()
        n_elements = diff.numel()  # n_samples * width
        per_neuron_rate = total_changes / n_elements

        # Absolute total switches
        total_switches = total_changes

        # Number of neurons where at least 1 sample changed
        neurons_changed = (diff.sum(dim=0) > 0).sum().item()

        per_neuron_rates.append(per_neuron_rate)
        total_switches_list.append(total_switches)
        neurons_changed_list.append(neurons_changed)

        prev_pattern = curr_pattern

    per_neuron_arr = np.array(per_neuron_rates)
    total_arr = np.array(total_switches_list)
    neurons_arr = np.array(neurons_changed_list)

    return {
        "width": int(width),
        "lr": float(lr),
        "seed": int(seed),
        "n_steps_completed": len(per_neuron_rates),
        "mean_per_neuron_rate": float(np.mean(per_neuron_arr)),
        "std_per_neuron_rate": float(np.std(per_neuron_arr)),
        "mean_total_switches": float(np.mean(total_arr)),
        "std_total_switches": float(np.std(total_arr)),
        "mean_neurons_changed": float(np.mean(neurons_arr)),
        "std_neurons_changed": float(np.std(neurons_arr)),
        "frac_neurons_changed": float(np.mean(neurons_arr) / width),
        "zero_change_fraction": float(np.mean(per_neuron_arr == 0)),
        # Downsampled trajectories for plotting
        "per_neuron_trajectory": [float(x) for x in per_neuron_arr[::20]],
        "total_switches_trajectory": [float(x) for x in total_arr[::20]],
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E15: Width-Dependent Activation Switch Rate")
    print("=" * 70)
    print(f"Widths: {WIDTHS}")
    print(f"Learning rates: {LRS}")
    print(f"Architecture: 2-layer ReLU, no bias")
    print(f"Loss: MSE, Data: Gaussian mixture (n={N_TRAIN}, d={INPUT_DIM}, K={OUTPUT_DIM})")
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
        for lr in LRS:
            print(f"\n  --- lr = {lr} ---")
            lr_results = []
            for seed in USE_SEEDS:
                r = run_switch_tracking(width, lr, seed)
                lr_results.append(r)
                print(f"    seed={seed}: per_neuron={r['mean_per_neuron_rate']:.6f}, "
                      f"total={r['mean_total_switches']:.1f}, "
                      f"neurons_changed={r['mean_neurons_changed']:.1f}/{width}")

            width_results[str(lr)] = lr_results

        all_results[str(width)] = width_results

    t_total = time.time() - t_start

    # ============================================================================
    # Analysis: Width Scaling of Switch Rates
    # ============================================================================
    print(f"\n{'='*70}")
    print("WIDTH SCALING ANALYSIS")
    print(f"{'='*70}")

    # Aggregate across seeds and LRs
    width_summary = {}
    for width in WIDTHS:
        all_per_neuron = []
        all_total = []
        all_neurons_changed = []
        all_frac_neurons = []
        for lr in LRS:
            for r in all_results[str(width)][str(lr)]:
                all_per_neuron.append(r["mean_per_neuron_rate"])
                all_total.append(r["mean_total_switches"])
                all_neurons_changed.append(r["mean_neurons_changed"])
                all_frac_neurons.append(r["frac_neurons_changed"])

        width_summary[str(width)] = {
            "width": int(width),
            "mean_per_neuron_rate": float(np.mean(all_per_neuron)),
            "std_per_neuron_rate": float(np.std(all_per_neuron)),
            "mean_total_switches": float(np.mean(all_total)),
            "std_total_switches": float(np.std(all_total)),
            "mean_neurons_changed": float(np.mean(all_neurons_changed)),
            "mean_frac_neurons_changed": float(np.mean(all_frac_neurons)),
        }

    print(f"\n{'Width':>8} {'PerNeuron':>12} {'Total':>12} {'Neurons':>12} {'FracNeurons':>12}")
    print("-" * 60)
    for width in WIDTHS:
        s = width_summary[str(width)]
        print(f"{width:>8} {s['mean_per_neuron_rate']:>12.6f} {s['mean_total_switches']:>12.1f} "
              f"{s['mean_neurons_changed']:>12.1f} {s['mean_frac_neurons_changed']:>12.3f}")

    # Fit per_neuron_rate ~ width^(-beta)
    widths_arr = np.array(WIDTHS, dtype=float)
    per_neuron_arr = np.array([width_summary[str(w)]["mean_per_neuron_rate"] for w in WIDTHS])
    total_arr = np.array([width_summary[str(w)]["mean_total_switches"] for w in WIDTHS])

    # Log-log fit for per-neuron rate
    mask = per_neuron_arr > 1e-15
    if np.sum(mask) >= 3:
        log_w = np.log(widths_arr[mask])
        log_r = np.log(per_neuron_arr[mask])
        neg_beta, intercept_r = np.polyfit(log_w, log_r, 1)
        beta = -neg_beta
        ss_res = np.sum((log_r - (neg_beta * log_w + intercept_r)) ** 2)
        ss_tot = np.sum((log_r - log_r.mean()) ** 2)
        r2_beta = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        beta, r2_beta, intercept_r = float('nan'), 0.0, 0.0

    # Log-log fit for total switches
    mask_t = total_arr > 1e-15
    if np.sum(mask_t) >= 3:
        log_t = np.log(total_arr[mask_t])
        gamma_total, intercept_t = np.polyfit(log_w[mask_t[mask]], log_t, 1)
        ss_res_t = np.sum((log_t - (gamma_total * log_w[mask_t[mask]] + intercept_t)) ** 2)
        ss_tot_t = np.sum((log_t - log_t.mean()) ** 2)
        r2_total = 1 - ss_res_t / ss_tot_t if ss_tot_t > 0 else 0
    else:
        gamma_total, r2_total, intercept_t = float('nan'), 0.0, 0.0

    print(f"\n--- SCALING FITS ---")
    print(f"  per_neuron_rate ~ width^(-{beta:.3f})  (R^2 = {r2_beta:.4f})")
    print(f"  total_switches ~ width^({gamma_total:.3f})   (R^2 = {r2_total:.4f})")
    print(f"  Predicted: if beta=0.5, then total ~ width^0.5 (sqrt(m))")
    print(f"  Measured: total ~ width^{gamma_total:.3f}")

    if beta > 0.3 and beta < 0.7:
        print(f"\n** HYPOTHESIS A CONFIRMED: beta = {beta:.3f} ~ 0.5 **")
        print(f"   Per-neuron rate decreases as width^(-{beta:.2f})")
        print(f"   Total switches grow as width^({gamma_total:.2f})")
        print(f"   Mode coupling is extensive: total effect grows with width")
    elif beta > 0:
        print(f"\n** PARTIAL SUPPORT: beta = {beta:.3f} (not exactly 0.5) **")
        print(f"   Per-neuron rate does decrease, but scaling differs from 1/sqrt(m)")
    else:
        print(f"\n** HYPOTHESIS A REFUTED: beta = {beta:.3f} <= 0 **")
        print(f"   Per-neuron rate does NOT decrease with width")
        print(f"   Mode coupling is intensive: per-neuron effect is width-independent")

    # Per-LR analysis
    print(f"\n--- PER-LR SCALING ---")
    per_lr_betas = {}
    for lr in LRS:
        rates = []
        totals = []
        for width in WIDTHS:
            seed_rates = [r["mean_per_neuron_rate"] for r in all_results[str(width)][str(lr)]]
            seed_totals = [r["mean_total_switches"] for r in all_results[str(width)][str(lr)]]
            rates.append(np.mean(seed_rates))
            totals.append(np.mean(seed_totals))

        rates_arr = np.array(rates)
        mask_lr = rates_arr > 1e-15
        if np.sum(mask_lr) >= 3:
            log_r_lr = np.log(rates_arr[mask_lr])
            log_w_lr = np.log(widths_arr[mask_lr])
            neg_b, _ = np.polyfit(log_w_lr, log_r_lr, 1)
            b = -neg_b
        else:
            b = float('nan')
        per_lr_betas[str(lr)] = float(b)
        print(f"  lr={lr}: beta = {b:.3f}")

    # Delta_alpha from E14 for correlation
    delta_alphas_e14 = {
        "16": -0.112,
        "32": -0.026,
        "64": 0.185,
        "128": -0.141,
        "256": 0.568,
    }

    print(f"\n--- CORRELATION WITH E14 delta_alpha ---")
    da_vals = [delta_alphas_e14.get(str(w), 0) for w in WIDTHS]
    total_vals = [width_summary[str(w)]["mean_total_switches"] for w in WIDTHS]
    if len(da_vals) >= 3:
        corr_da_total = np.corrcoef(total_vals, da_vals)[0, 1]
        print(f"  Correlation(total_switches, delta_alpha): {corr_da_total:.3f}")
    else:
        corr_da_total = float('nan')

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
        "width_summary": width_summary,
        "scaling_fits": {
            "per_neuron_beta": float(beta),
            "per_neuron_r2": float(r2_beta),
            "total_gamma": float(gamma_total),
            "total_r2": float(r2_total),
        },
        "per_lr_betas": per_lr_betas,
        "correlation_with_e14": float(corr_da_total) if not np.isnan(corr_da_total) else None,
        "delta_alphas_e14": delta_alphas_e14,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 18: Width-Dependent Switch Rate
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    lr_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(LRS)))

    # Panel (a): Per-neuron switch rate vs width (log-log)
    ax = axes[0]
    for i, lr in enumerate(LRS):
        rates = []
        rates_std = []
        for width in WIDTHS:
            seed_rates = [r["mean_per_neuron_rate"] for r in all_results[str(width)][str(lr)]]
            rates.append(np.mean(seed_rates))
            rates_std.append(np.std(seed_rates))
        ax.errorbar(WIDTHS, rates, yerr=rates_std, fmt='o-', color=lr_colors[i],
                     markersize=7, linewidth=1.5, capsize=3, label=f'$\\eta$={lr}')

    # Aggregate fit line
    if not np.isnan(beta):
        w_fit = np.logspace(np.log10(12), np.log10(300), 50)
        r_fit = np.exp(intercept_r) * w_fit ** neg_beta
        ax.plot(w_fit, r_fit, 'k--', linewidth=1.5, alpha=0.7,
                label=f'Fit: $\\sim m^{{-{beta:.2f}}}$ (R$^2$={r2_beta:.3f})')

    # Reference: 1/sqrt(m)
    w_ref = np.array([16, 256], dtype=float)
    r_ref = per_neuron_arr[0] * (w_ref / WIDTHS[0]) ** (-0.5)
    ax.plot(w_ref, r_ref, 'r:', linewidth=1, alpha=0.5, label='$O(1/\\sqrt{m})$ ref')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Per-neuron switch rate')
    ax.set_title('(a) Per-neuron switch rate vs width')
    ax.legend(fontsize=8)

    # Panel (b): Total switches per step vs width (log-log)
    ax = axes[1]
    for i, lr in enumerate(LRS):
        totals = []
        totals_std = []
        for width in WIDTHS:
            seed_totals = [r["mean_total_switches"] for r in all_results[str(width)][str(lr)]]
            totals.append(np.mean(seed_totals))
            totals_std.append(np.std(seed_totals))
        ax.errorbar(WIDTHS, totals, yerr=totals_std, fmt='s-', color=lr_colors[i],
                     markersize=7, linewidth=1.5, capsize=3, label=f'$\\eta$={lr}')

    # Reference: sqrt(m)
    t_ref = total_arr[0] * (w_ref / WIDTHS[0]) ** 0.5
    ax.plot(w_ref, t_ref, 'r:', linewidth=1, alpha=0.5, label='$O(\\sqrt{m})$ ref')

    # Reference: linear
    t_ref_lin = total_arr[0] * (w_ref / WIDTHS[0]) ** 1.0
    ax.plot(w_ref, t_ref_lin, 'b:', linewidth=1, alpha=0.5, label='$O(m)$ ref')

    if not np.isnan(gamma_total):
        w_fit = np.logspace(np.log10(12), np.log10(300), 50)
        t_fit = np.exp(intercept_t) * w_fit ** gamma_total
        ax.plot(w_fit, t_fit, 'k--', linewidth=1.5, alpha=0.7,
                label=f'Fit: $\\sim m^{{{gamma_total:.2f}}}$ (R$^2$={r2_total:.3f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Total switches per step')
    ax.set_title('(b) Total switches vs width')
    ax.legend(fontsize=8)

    # Panel (c): Total switches vs delta_alpha from E14
    ax = axes[2]
    da_plot = [delta_alphas_e14.get(str(w), 0) for w in WIDTHS]
    total_plot = [width_summary[str(w)]["mean_total_switches"] for w in WIDTHS]

    ax.scatter(total_plot, da_plot, c=[np.log2(w) for w in WIDTHS],
               cmap='plasma', s=100, zorder=5, edgecolors='black')

    for w, t, d in zip(WIDTHS, total_plot, da_plot):
        ax.annotate(f'm={w}', (t, d), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.axhline(y=0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Mean total switches per step')
    ax.set_ylabel('$\\Delta\\alpha$ (from E14)')
    ax.set_title(f'(c) Mode coupling vs $\\Delta\\alpha$ (R={corr_da_total:.2f})')

    fig.suptitle('E15: Width-Dependent Activation Switch Rate — Testing Hypothesis A',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig18_width_switch_rate", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig18_width_switch_rate.pdf")
