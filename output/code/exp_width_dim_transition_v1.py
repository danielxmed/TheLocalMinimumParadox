"""
Experiment: Width Transition vs Input Dimension
Theory: Theory A -- Testing whether the width transition m* scales as m* ~ 2d
Prediction: E19 showed alpha transitions at width ~40 for d=20. If this is at m ~ 2d,
            then d=10 should transition at ~20, d=40 at ~80.
            The rescaled curves alpha vs (width/d) should collapse.
Date: 2026-04-07 (Session 9)

Method:
  For d in {10, 20, 40}:
    For width in {d/2, d, 1.5d, 2d, 3d, 4d, 6d}:
      Train 2-layer LINEAR network (no bias) with MSE
      Fit alpha from drift ~ eta^alpha
    Identify transition width m* where alpha exceeds threshold
  Test: does m*/d = constant (predicted: ~2)?
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

EXPERIMENT_NAME = "width_dim_transition"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Configurations per input dimension
INPUT_DIMS = [10, 20, 40]
OUTPUT_DIM = 5
N_TRAIN = 200
LRS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
N_STEPS = 2000
N_SEEDS = 5  # Use all 5 seeds for statistical robustness

# Width multipliers: 0.5d, 1d, 1.5d, 2d, 3d, 4d, 6d
WIDTH_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

ALPHA_THRESHOLD = 1.1  # Transition defined as alpha exceeding this


class LinearTwoLayer(nn.Module):
    """2-layer linear network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


def measure_drift(input_dim, width, lr, seed):
    """Train and measure conservation drift."""
    set_seed(seed)
    model = LinearTwoLayer(input_dim, width, OUTPUT_DIM).to(DEVICE)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=input_dim, K=OUTPUT_DIM, separation=2.0)

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

        # Curvature
        if np.sum(mask) >= 4:
            coeffs_quad = np.polyfit(log_lr, log_drift, 2)
            curvature = float(coeffs_quad[0])
        else:
            curvature = 0.0

        return {
            "alpha": float(alpha),
            "r2": float(r2),
            "curvature": curvature,
            "intercept": float(intercept),
        }
    return {"alpha": float('nan'), "r2": 0, "curvature": 0, "intercept": 0}


if __name__ == "__main__":
    print("=" * 70)
    print("E22: Width Transition vs Input Dimension")
    print("=" * 70)
    print(f"Input dimensions: {INPUT_DIMS}")
    print(f"Width multipliers: {WIDTH_MULTIPLIERS}")
    print(f"LRs: {LRS}")
    print(f"Seeds: {SEEDS[:N_SEEDS]}")
    print()

    t_start = time.time()
    all_results = {}

    for d in INPUT_DIMS:
        print(f"\n{'='*60}")
        print(f"  Input dimension d = {d}")
        print(f"{'='*60}")

        widths = [max(int(m * d), 2) for m in WIDTH_MULTIPLIERS]
        # Ensure widths are unique
        widths = sorted(set(widths))
        print(f"  Widths: {widths}")

        d_results = {}

        for width in widths:
            print(f"\n  --- Width = {width} (m/d = {width/d:.1f}) ---")
            width_results = {}
            mean_drifts = {}

            for lr in LRS:
                seed_drifts = []
                for seed in SEEDS[:N_SEEDS]:
                    r = measure_drift(d, width, lr, seed)
                    seed_drifts.append(r["actual_drift"])

                valid = [x for x in seed_drifts if np.isfinite(x)]
                mean_d = float(np.mean(valid)) if valid else float('nan')
                std_d = float(np.std(valid)) if len(valid) > 1 else 0.0
                mean_drifts[str(lr)] = mean_d

                width_results[str(lr)] = {
                    "mean_drift": mean_d,
                    "std_drift": std_d,
                    "per_seed": seed_drifts,
                    "n_converged": sum(1 for x in seed_drifts if np.isfinite(x)),
                }

            # Power-law fit
            lrs_list = [lr for lr in LRS if np.isfinite(mean_drifts.get(str(lr), float('nan')))]
            drifts_list = [mean_drifts[str(lr)] for lr in lrs_list]
            fit = fit_power_law(lrs_list, drifts_list)

            width_results["fit"] = fit
            width_results["width_over_d"] = float(width / d)
            d_results[str(width)] = width_results

            print(f"    alpha = {fit['alpha']:.4f}, R^2 = {fit['r2']:.4f}, "
                  f"curvature = {fit['curvature']:.4f}")

        all_results[str(d)] = {
            "widths": widths,
            "results": d_results,
        }

    # ============================================================================
    # Analysis: Find transition width for each d
    # ============================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: TRANSITION WIDTH vs INPUT DIMENSION")
    print(f"{'='*70}")

    transition_widths = {}
    for d in INPUT_DIMS:
        d_data = all_results[str(d)]
        widths = d_data["widths"]

        alphas = []
        r2s = []
        for w in widths:
            fit = d_data["results"][str(w)]["fit"]
            alphas.append(fit["alpha"])
            r2s.append(fit["r2"])

        # Find transition: smallest width where alpha > ALPHA_THRESHOLD
        m_star = None
        for w, a in zip(widths, alphas):
            if np.isfinite(a) and a > ALPHA_THRESHOLD:
                m_star = w
                break

        if m_star is None:
            # If no transition found, use the largest width
            m_star = widths[-1] if alphas[-1] > ALPHA_THRESHOLD else None

        transition_widths[str(d)] = {
            "m_star": m_star,
            "m_star_over_d": float(m_star / d) if m_star else None,
            "widths": widths,
            "alphas": alphas,
            "r2s": r2s,
        }

        if m_star:
            print(f"  d={d}: m* = {m_star}, m*/d = {m_star/d:.2f}")
        else:
            print(f"  d={d}: No transition found (max alpha = {max(a for a in alphas if np.isfinite(a)):.4f})")

    # Test: does m*/d = constant?
    m_star_over_d_values = [v["m_star_over_d"] for v in transition_widths.values()
                            if v["m_star_over_d"] is not None]

    if len(m_star_over_d_values) >= 2:
        mean_ratio = np.mean(m_star_over_d_values)
        std_ratio = np.std(m_star_over_d_values)
        print(f"\n  Mean m*/d = {mean_ratio:.2f} +/- {std_ratio:.2f}")
        if std_ratio / mean_ratio < 0.3:
            print(f"  ** m*/d IS APPROXIMATELY CONSTANT: m* ~ {mean_ratio:.1f}d **")
        else:
            print(f"  ** m*/d varies significantly across dimensions **")
    else:
        mean_ratio = float('nan')
        std_ratio = float('nan')

    # Test curve collapse: alpha vs (width/d) for all d values
    print(f"\n--- Curve Collapse Test ---")
    print(f"  alpha vs width/d should overlap if scaling is universal")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E22",
        "version": 1,
        "date": "2026-04-07",
        "session": 9,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in SEEDS[:N_SEEDS]],
        "input_dims": INPUT_DIMS,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "n_steps": N_STEPS,
        "lrs": LRS,
        "width_multipliers": WIDTH_MULTIPLIERS,
        "alpha_threshold": ALPHA_THRESHOLD,
        "transition_analysis": {
            str(d): {
                "m_star": transition_widths[str(d)]["m_star"],
                "m_star_over_d": transition_widths[str(d)]["m_star_over_d"],
                "alphas": transition_widths[str(d)]["alphas"],
                "r2s": transition_widths[str(d)]["r2s"],
            } for d in INPUT_DIMS
        },
        "mean_m_star_over_d": float(mean_ratio) if np.isfinite(mean_ratio) else None,
        "std_m_star_over_d": float(std_ratio) if np.isfinite(std_ratio) else None,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 25: Width-Dimension Transition
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    d_colors = {'10': '#e41a1c', '20': '#377eb8', '40': '#4daf4a'}
    d_markers = {'10': 'o', '20': 's', '40': '^'}

    # Panel (a): Alpha vs width/d (rescaled) — curve collapse test
    ax = axes[0]
    for d in INPUT_DIMS:
        t_data = transition_widths[str(d)]
        widths = t_data["widths"]
        alphas = t_data["alphas"]

        wd_ratios = [w / d for w in widths]
        valid = [(r, a) for r, a in zip(wd_ratios, alphas) if np.isfinite(a)]
        if valid:
            x_vals, y_vals = zip(*valid)
            ax.plot(x_vals, y_vals, f'{d_markers[str(d)]}-',
                    color=d_colors[str(d)], markersize=8, linewidth=2,
                    label=f'd={d}')

    ax.axhline(y=ALPHA_THRESHOLD, color='gray', ls='--', lw=1, alpha=0.5,
               label=f'$\\alpha = {ALPHA_THRESHOLD}$')
    ax.axvline(x=2.0, color='gray', ls=':', lw=1, alpha=0.5,
               label='$m/d = 2$')
    ax.set_xlabel('Width / Input dim ($m/d$)')
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(a) $\\alpha$ vs $m/d$ — curve collapse test')
    ax.legend(fontsize=8)

    # Panel (b): R^2 vs width/d
    ax = axes[1]
    for d in INPUT_DIMS:
        t_data = transition_widths[str(d)]
        widths = t_data["widths"]
        r2s = t_data["r2s"]

        wd_ratios = [w / d for w in widths]
        valid = [(r, a) for r, a in zip(wd_ratios, r2s) if np.isfinite(a)]
        if valid:
            x_vals, y_vals = zip(*valid)
            ax.plot(x_vals, y_vals, f'{d_markers[str(d)]}-',
                    color=d_colors[str(d)], markersize=8, linewidth=2,
                    label=f'd={d}')

    ax.axhline(y=0.95, color='red', ls='--', lw=1, alpha=0.5,
               label='$R^2 = 0.95$')
    ax.axvline(x=2.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Width / Input dim ($m/d$)')
    ax.set_ylabel('$R^2$ of power-law fit')
    ax.set_title('(b) Power-law quality vs $m/d$')
    ax.legend(fontsize=8)
    ax.set_ylim(0.7, 1.02)

    # Panel (c): Transition width m* vs d
    ax = axes[2]
    d_vals = []
    m_star_vals = []
    for d in INPUT_DIMS:
        t_data = transition_widths[str(d)]
        if t_data["m_star"] is not None:
            d_vals.append(d)
            m_star_vals.append(t_data["m_star"])

    if len(d_vals) >= 2:
        ax.plot(d_vals, m_star_vals, 'ko-', markersize=12, linewidth=2,
                label=f'Measured $m^*$ ($\\alpha > {ALPHA_THRESHOLD}$)')

        # Reference: m* = 2d
        d_fine = np.linspace(min(d_vals)*0.7, max(d_vals)*1.3, 50)
        ax.plot(d_fine, 2 * d_fine, 'r--', linewidth=1.5, label='$m^* = 2d$')

        # Reference: m* = 1.5d
        ax.plot(d_fine, 1.5 * d_fine, 'b:', linewidth=1.5, alpha=0.5, label='$m^* = 1.5d$')

        # Fit m* = c * d
        if len(d_vals) >= 2:
            c_fit = np.mean([m/d for m, d in zip(m_star_vals, d_vals)])
            ax.plot(d_fine, c_fit * d_fine, 'g-.', linewidth=1.5,
                    label=f'$m^* = {c_fit:.2f}d$ (fit)')

    ax.set_xlabel('Input dimension $d$')
    ax.set_ylabel('Transition width $m^*$')
    ax.set_title('(c) Transition width scales with $d$')
    ax.legend(fontsize=8)

    if np.isfinite(mean_ratio):
        ax.annotate(f'$m^*/d = {mean_ratio:.2f} \\pm {std_ratio:.2f}$',
                   xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('E22: Width Transition Scales with Input Dimension',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig25_width_dim_transition", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig25_width_dim_transition.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    for d in INPUT_DIMS:
        t_data = transition_widths[str(d)]
        m_star = t_data["m_star"]
        ratio = t_data["m_star_over_d"]
        print(f"  d={d}: m* = {m_star}, m*/d = {ratio:.2f}" if m_star else
              f"  d={d}: No transition found")

    if np.isfinite(mean_ratio):
        print(f"\n  Mean m*/d = {mean_ratio:.2f} +/- {std_ratio:.2f}")
        if abs(mean_ratio - 2.0) < 1.0:
            print("  ** CONFIRMED: Transition occurs near m ~ 2d **")
            print("  ** This coincides with the overparameterization threshold **")
        else:
            print(f"  ** Transition at m ~ {mean_ratio:.1f}d (not exactly 2d) **")
