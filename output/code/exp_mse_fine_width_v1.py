"""
Experiment: MSE Alpha Divergence — Fine Width Sweep
Theory: Theory A -- Characterizing how MSE drift exponent grows with width
Prediction: E17 showed lin_mse alpha reaches 1.63 at width 128. This experiment
            measures alpha at 8 widths with 5 seeds to characterize the scaling law
            and detect potential breakdown of the power-law fit.
Date: 2026-04-07 (Session 8)

Method:
  For each width in [16, 24, 32, 48, 64, 96, 128, 192]:
    For each LR in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
      Train 2-layer LINEAR network (no bias) with MSE, 5 seeds
      Measure conservation drift
    Fit alpha from drift ~ eta^alpha
    Check R^2 of power-law fit (does it degrade?)
    Test for curvature in log-log space
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

EXPERIMENT_NAME = "mse_fine_width"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WIDTHS = [16, 24, 32, 48, 64, 96, 128, 192]
LRS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
N_STEPS = 2000
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200


class LinearTwoLayer(nn.Module):
    """2-layer linear network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


def measure_drift(width, lr, seed):
    """Train and measure conservation drift for linear MSE."""
    set_seed(seed)
    model = LinearTwoLayer(INPUT_DIM, width, OUTPUT_DIM).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    loss_fn = nn.MSELoss()
    C_init = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_onehot)
        loss.backward()
        optimizer.step()
        if np.isnan(loss.item()):
            return {"actual_drift": float('nan'), "converged": False}

    C_final = model.W2.weight.data.norm().item()**2 - model.W1.weight.data.norm().item()**2
    return {
        "actual_drift": float(abs(C_final - C_init)),
        "converged": True,
    }


def fit_power_law(lrs, drifts):
    """Fit drift ~ eta^alpha in log-log space."""
    lrs_arr = np.array(lrs)
    drifts_arr = np.array(drifts)
    mask = (drifts_arr > 1e-15) & np.isfinite(drifts_arr)

    if np.sum(mask) >= 3:
        log_lr = np.log(lrs_arr[mask])
        log_drift = np.log(drifts_arr[mask])
        alpha, intercept = np.polyfit(log_lr, log_drift, 1)
        y_pred = alpha * log_lr + intercept
        ss_res = np.sum((log_drift - y_pred) ** 2)
        ss_tot = np.sum((log_drift - log_drift.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Quadratic fit to detect curvature
        if np.sum(mask) >= 4:
            coeffs_quad = np.polyfit(log_lr, log_drift, 2)
            y_pred_quad = np.polyval(coeffs_quad, log_lr)
            ss_res_quad = np.sum((log_drift - y_pred_quad) ** 2)
            r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
            curvature = float(coeffs_quad[0])
        else:
            r2_quad = r2
            curvature = 0.0

        # Residuals per LR
        residuals = {}
        for i, lr in enumerate(lrs_arr[mask]):
            residuals[f"{lr:.4f}"] = float(log_drift[i] - (alpha * log_lr[i] + intercept))

        return {
            "alpha": float(alpha),
            "r2": float(r2),
            "r2_quad": float(r2_quad),
            "curvature": curvature,
            "intercept": float(intercept),
            "residuals": residuals,
        }
    return {"alpha": float('nan'), "r2": 0, "r2_quad": 0, "curvature": 0,
            "intercept": 0, "residuals": {}}


if __name__ == "__main__":
    print("=" * 70)
    print("E19: MSE Alpha Divergence — Fine Width Sweep")
    print("=" * 70)
    print(f"Architecture: 2-layer LINEAR, no bias")
    print(f"Widths: {WIDTHS}")
    print(f"LRs: {LRS}")
    print(f"Seeds: {SEEDS}")
    print()

    t_start = time.time()
    all_results = {}

    for width in WIDTHS:
        print(f"\n--- Width = {width} ---")
        width_results = {}
        mean_drifts = {}

        for lr in LRS:
            seed_drifts = []
            for seed in SEEDS:
                r = measure_drift(width, lr, seed)
                seed_drifts.append(r["actual_drift"])

            valid = [d for d in seed_drifts if np.isfinite(d)]
            mean_d = float(np.mean(valid)) if valid else float('nan')
            std_d = float(np.std(valid)) if len(valid) > 1 else 0.0
            mean_drifts[str(lr)] = mean_d

            width_results[str(lr)] = {
                "mean_drift": mean_d,
                "std_drift": std_d,
                "per_seed": seed_drifts,
                "n_converged": sum(1 for d in seed_drifts if np.isfinite(d)),
            }

        # Power-law fit
        lrs_list = [lr for lr in LRS if np.isfinite(mean_drifts.get(str(lr), float('nan')))]
        drifts_list = [mean_drifts[str(lr)] for lr in lrs_list]
        fit = fit_power_law(lrs_list, drifts_list)

        width_results["fit"] = fit
        all_results[str(width)] = width_results

        print(f"  alpha = {fit['alpha']:.4f}, R^2 = {fit['r2']:.4f}, "
              f"curvature = {fit['curvature']:.4f}, R^2_quad = {fit['r2_quad']:.4f}")

    # ============================================================================
    # Analysis: Alpha vs Width Scaling
    # ============================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: ALPHA VS WIDTH")
    print(f"{'='*70}")

    alphas = []
    r2s = []
    curvatures = []
    for width in WIDTHS:
        fit = all_results[str(width)]["fit"]
        alphas.append(fit["alpha"])
        r2s.append(fit["r2"])
        curvatures.append(fit["curvature"])
        print(f"  width={width:>4}: alpha={fit['alpha']:.4f}, R^2={fit['r2']:.4f}, "
              f"curv={fit['curvature']:.4f}")

    # Fit alpha(width) = a * width^gamma + b
    valid_mask = np.isfinite(alphas)
    if sum(valid_mask) >= 3:
        w_arr = np.array(WIDTHS)[valid_mask]
        a_arr = np.array(alphas)[valid_mask]

        # Log-log fit: alpha - 1 vs width (since alpha -> 1 as width -> 0)
        a_shifted = a_arr - 1.0
        pos_mask = a_shifted > 0
        if sum(pos_mask) >= 3:
            log_w = np.log(w_arr[pos_mask])
            log_a = np.log(a_shifted[pos_mask])
            gamma, log_c = np.polyfit(log_w, log_a, 1)
            c = np.exp(log_c)

            y_pred = gamma * log_w + log_c
            ss_res = np.sum((log_a - y_pred) ** 2)
            ss_tot = np.sum((log_a - log_a.mean()) ** 2)
            r2_scaling = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"\n  Scaling: alpha - 1 ~ {c:.4f} * width^{gamma:.3f}")
            print(f"  R^2 = {r2_scaling:.4f}")
        else:
            gamma, c, r2_scaling = float('nan'), 0, 0
            print(f"\n  Not enough positive (alpha-1) values for scaling fit")

        # Also test log scaling: alpha = a + b*log(width)
        log_fit = np.polyfit(np.log(w_arr), a_arr, 1)
        a_pred_log = np.polyval(log_fit, np.log(w_arr))
        ss_res_log = np.sum((a_arr - a_pred_log) ** 2)
        ss_tot_log = np.sum((a_arr - a_arr.mean()) ** 2)
        r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 0
        print(f"  Log scaling: alpha = {log_fit[1]:.4f} + {log_fit[0]:.4f} * log(width), R^2 = {r2_log:.4f}")
    else:
        gamma, c, r2_scaling, r2_log = float('nan'), 0, 0, 0
        log_fit = [0, 0]

    # Does R^2 degrade with width?
    print(f"\n--- R^2 degradation check ---")
    r2_corr = np.corrcoef(np.log(WIDTHS), r2s)[0, 1] if len(r2s) >= 3 else 0
    print(f"  Correlation between log(width) and R^2: {r2_corr:.4f}")
    if r2_corr < -0.5:
        print("  ** POWER LAW FIT DEGRADES WITH WIDTH **")
    else:
        print("  ** Power law fit quality stable across widths **")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E19",
        "version": 1,
        "date": "2026-04-07",
        "session": 8,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in SEEDS],
        "widths": WIDTHS,
        "lrs": LRS,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "analysis": {
            "alphas": alphas,
            "r2s": r2s,
            "curvatures": curvatures,
            "power_law_gamma": float(gamma) if np.isfinite(gamma) else None,
            "power_law_c": float(c),
            "power_law_r2": float(r2_scaling),
            "log_fit_slope": float(log_fit[0]),
            "log_fit_intercept": float(log_fit[1]),
            "log_fit_r2": float(r2_log),
            "r2_degradation_corr": float(r2_corr),
        },
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 22: MSE Width Divergence
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): Alpha vs width with error bars
    ax = axes[0]
    alpha_means = []
    alpha_stds = []
    for width in WIDTHS:
        # Get per-seed alpha fits
        w_data = all_results[str(width)]
        seed_alphas = []
        for seed in SEEDS:
            seed_drifts = []
            seed_lrs = []
            for lr in LRS:
                d = w_data[str(lr)]["per_seed"][SEEDS.index(seed)]
                if np.isfinite(d) and d > 1e-15:
                    seed_drifts.append(d)
                    seed_lrs.append(lr)
            if len(seed_lrs) >= 3:
                s_fit = fit_power_law(seed_lrs, seed_drifts)
                if np.isfinite(s_fit["alpha"]):
                    seed_alphas.append(s_fit["alpha"])
        alpha_means.append(np.mean(seed_alphas) if seed_alphas else float('nan'))
        alpha_stds.append(np.std(seed_alphas) if len(seed_alphas) > 1 else 0)

    ax.errorbar(WIDTHS, alpha_means, yerr=alpha_stds, fmt='ko-', markersize=8,
                linewidth=2, capsize=4, label='Measured $\\alpha$')

    # Power-law fit overlay
    valid = [(w, a) for w, a in zip(WIDTHS, alpha_means) if np.isfinite(a) and a > 1]
    if len(valid) >= 3 and np.isfinite(gamma):
        w_fine = np.linspace(min(WIDTHS), max(WIDTHS)*1.1, 100)
        ax.plot(w_fine, 1 + c * w_fine**gamma, 'r--', linewidth=1.5,
                label=f'$\\alpha = 1 + {c:.3f} \\cdot m^{{{gamma:.2f}}}$')

    # Log fit overlay
    if r2_log > 0.8:
        w_fine = np.linspace(min(WIDTHS), max(WIDTHS)*1.1, 100)
        ax.plot(w_fine, log_fit[0] * np.log(w_fine) + log_fit[1], 'b:', linewidth=1.5,
                label=f'$\\alpha = {log_fit[1]:.2f} + {log_fit[0]:.2f}\\ln m$ ($R^2={r2_log:.3f}$)')

    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(a) MSE $\\alpha$ diverges with width')
    ax.legend(fontsize=8)
    ax.axhline(y=1.0, color='gray', ls=':', lw=0.5)

    # Panel (b): R^2 of power-law fit vs width
    ax = axes[1]
    ax.plot(WIDTHS, r2s, 'ko-', markersize=8, linewidth=2)
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('$R^2$ of power-law fit')
    ax.set_title('(b) Power-law fit quality vs width')
    ax.axhline(y=0.95, color='red', ls='--', lw=1, alpha=0.5, label='$R^2 = 0.95$')
    ax.legend(fontsize=9)
    ax.set_ylim(0.8, 1.02)

    # Panel (c): Drift vs eta at selected widths (log-log)
    ax = axes[2]
    width_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(WIDTHS)))
    for i, width in enumerate(WIDTHS):
        w_data = all_results[str(width)]
        lrs_plot = []
        drifts_plot = []
        for lr in LRS:
            d = w_data[str(lr)]["mean_drift"]
            if np.isfinite(d) and d > 1e-15:
                lrs_plot.append(lr)
                drifts_plot.append(d)
        if lrs_plot:
            ax.loglog(lrs_plot, drifts_plot, 'o-', color=width_colors[i],
                      markersize=5, linewidth=1.5, label=f'm={width}')

    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Conservation law drift $|\\Delta C|$')
    ax.set_title('(c) Drift vs $\\eta$ (log-log) — curvature detection')
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle('E19: MSE Alpha Divergence with Width — Fine Sweep',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig22_mse_width_divergence", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig22_mse_width_divergence.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"  Alpha range: {min(a for a in alphas if np.isfinite(a)):.3f} "
          f"to {max(a for a in alphas if np.isfinite(a)):.3f}")
    if np.isfinite(gamma):
        print(f"  Scaling: alpha - 1 ~ width^{gamma:.3f} (R^2={r2_scaling:.4f})")
    print(f"  Log fit: alpha = {log_fit[1]:.3f} + {log_fit[0]:.3f}*ln(width) (R^2={r2_log:.4f})")
    print(f"  R^2 degradation: {'YES' if r2_corr < -0.5 else 'NO'} (corr={r2_corr:.3f})")

    if r2_corr < -0.5:
        print(f"\n  ** The power-law model drift ~ eta^alpha BREAKS DOWN at large widths. **")
        print(f"  ** Alternative scaling forms should be explored. **")
    elif max(a for a in alphas if np.isfinite(a)) > 1.5:
        print(f"\n  ** Alpha continues to grow with width. MSE training becomes increasingly **")
        print(f"  ** non-perturbative at EoS, consistent with extensive mode coupling (E15). **")
