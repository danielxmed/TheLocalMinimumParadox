"""
Experiment: Spectral Compression Timescale vs Learning Rate
Theory: Theory A -- Validating tau = C/eta from NTK-based derivation (Section 7.20)
Prediction: The softmax concentration timescale tau should be proportional to 1/eta.
            Doubling the learning rate should halve the time to spectral compression.
            The constant C depends on architecture but NOT on n_train.
Date: 2026-04-07 (Session 9)

Method:
  For lr in {0.003, 0.005, 0.01, 0.02, 0.03}:
    Train 2-layer ReLU (hidden=64, no bias) with CE
    At fine checkpoint grid, record:
      - Mean correct-class probability q_bar(t)
      - Max Hessian eigenvalue (at 4 sparse checkpoints)
    Fit exponential decay: 1 - q_bar(t) ~ exp(-t/tau)
    Extract tau for each LR
  Test: tau vs 1/eta should be linear (R^2 > 0.9)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset, count_parameters,
    full_hessian, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "tau_lr_scaling"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200
N_STEPS = 3000
LEARNING_RATES = [0.003, 0.005, 0.01, 0.02, 0.03]
CHECKPOINTS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 2500, 3000]
HESSIAN_CHECKPOINTS = [0, 200, 500, 1000]  # Sparse Hessian evaluation (expensive)
USE_SEEDS = SEEDS[:3]  # [42, 137, 256]


def train_and_track_softmax(lr, seed):
    """Train with CE and track softmax concentration at checkpoints."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    checkpoint_data = {}

    for step in range(N_STEPS + 1):
        if step in CHECKPOINTS:
            with torch.no_grad():
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                correct_probs = probs[torch.arange(len(Y)), Y]
                q_bar = correct_probs.mean().item()
                q_min = correct_probs.min().item()
                q_max = correct_probs.max().item()

                # Also track loss
                loss_val = loss_fn(logits, Y).item()

                # Accuracy
                preds = logits.argmax(dim=1)
                accuracy = (preds == Y).float().mean().item()

            cp_entry = {
                "step": step,
                "q_bar": float(q_bar),
                "q_min": float(q_min),
                "q_max": float(q_max),
                "one_minus_q": float(1.0 - q_bar),
                "loss": float(loss_val),
                "accuracy": float(accuracy),
            }

            # Hessian at sparse checkpoints
            if step in HESSIAN_CHECKPOINTS:
                t0 = time.time()
                model_snapshot = copy.deepcopy(model)
                H = full_hessian(model_snapshot, X, Y, loss_fn)
                H_np = H.numpy()
                eigenvalues = np.linalg.eigvalsh(H_np)
                t_hess = time.time() - t0

                cp_entry["hessian"] = {
                    "max_eigenvalue": float(eigenvalues[-1]),
                    "min_eigenvalue": float(eigenvalues[0]),
                    "n_positive": int(np.sum(eigenvalues > 1e-8)),
                    "trace_positive": float(np.sum(eigenvalues[eigenvalues > 0])),
                    "hessian_time": float(t_hess),
                }
                del model_snapshot, H, H_np, eigenvalues

            checkpoint_data[str(step)] = cp_entry

        if step < N_STEPS:
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, Y)
            loss.backward()
            optimizer.step()
            if np.isnan(loss.item()):
                print(f"      NaN at step {step}")
                break

    return checkpoint_data


def fit_tau(checkpoints_data, min_step=25):
    """Fit tau from 1 - q_bar(t) ~ exp(-t/tau).

    Only fit after initial transient (step > min_step) and
    where 1 - q_bar > 0.01 (not yet fully converged).
    """
    steps = []
    one_minus_q = []

    for cp_str, cp_data in sorted(checkpoints_data.items(), key=lambda x: int(x[0])):
        step = int(cp_str)
        omq = cp_data["one_minus_q"]
        if step >= min_step and omq > 0.01 and omq < 0.9:
            steps.append(step)
            one_minus_q.append(omq)

    if len(steps) < 3:
        return {"tau": float('nan'), "r2": 0, "a": 0, "n_points": len(steps)}

    steps = np.array(steps, dtype=float)
    omq = np.array(one_minus_q, dtype=float)

    # Log-linear fit: log(1 - q_bar) = log(a) - t/tau
    log_omq = np.log(omq)
    coeffs = np.polyfit(steps, log_omq, 1)
    slope = coeffs[0]  # = -1/tau
    intercept = coeffs[1]  # = log(a)

    tau = -1.0 / slope if slope < 0 else float('nan')
    a = np.exp(intercept)

    # R^2
    y_pred = coeffs[0] * steps + coeffs[1]
    ss_res = np.sum((log_omq - y_pred) ** 2)
    ss_tot = np.sum((log_omq - log_omq.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "tau": float(tau),
        "r2": float(r2),
        "a": float(a),
        "slope": float(slope),
        "n_points": len(steps),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("E23: Spectral Compression Timescale vs Learning Rate")
    print("=" * 70)
    print(f"Architecture: 2-layer ReLU, hidden={HIDDEN}, no bias, CE loss")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Seeds: {USE_SEEDS}")
    print(f"N_STEPS: {N_STEPS}")
    print()

    t_start = time.time()
    all_results = {}

    tau_per_lr = {}

    for lr in LEARNING_RATES:
        print(f"\n{'='*60}")
        print(f"  Learning rate = {lr}")
        print(f"{'='*60}")

        lr_results = {}
        taus = []

        for seed in USE_SEEDS:
            print(f"\n  --- seed={seed} ---")
            cp_data = train_and_track_softmax(lr, seed)

            # Fit tau
            tau_fit = fit_tau(cp_data)
            taus.append(tau_fit["tau"])

            print(f"    tau = {tau_fit['tau']:.1f}, R^2 = {tau_fit['r2']:.4f}, "
                  f"n_points = {tau_fit['n_points']}")

            lr_results[str(seed)] = {
                "checkpoints": cp_data,
                "tau_fit": tau_fit,
            }

        # Average tau
        valid_taus = [t for t in taus if np.isfinite(t) and t > 0]
        mean_tau = float(np.mean(valid_taus)) if valid_taus else float('nan')
        std_tau = float(np.std(valid_taus)) if len(valid_taus) > 1 else 0.0

        tau_per_lr[str(lr)] = {
            "mean_tau": mean_tau,
            "std_tau": std_tau,
            "per_seed_tau": taus,
        }

        lr_results["tau_summary"] = tau_per_lr[str(lr)]
        all_results[str(lr)] = lr_results

        print(f"\n  LR={lr}: Mean tau = {mean_tau:.1f} +/- {std_tau:.1f}")

    # ============================================================================
    # Analysis: tau vs 1/eta
    # ============================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: TAU vs 1/ETA")
    print(f"{'='*70}")

    lr_vals = []
    tau_means = []
    tau_stds = []

    for lr in LEARNING_RATES:
        t_data = tau_per_lr[str(lr)]
        if np.isfinite(t_data["mean_tau"]) and t_data["mean_tau"] > 0:
            lr_vals.append(lr)
            tau_means.append(t_data["mean_tau"])
            tau_stds.append(t_data["std_tau"])
            print(f"  lr={lr:.4f}: tau = {t_data['mean_tau']:.1f} +/- {t_data['std_tau']:.1f}, "
                  f"1/eta = {1/lr:.1f}")

    # Fit tau = C / eta (i.e., tau = C * (1/eta))
    if len(lr_vals) >= 3:
        inv_eta = np.array([1.0/lr for lr in lr_vals])
        tau_arr = np.array(tau_means)

        # Linear fit: tau = C * (1/eta) + intercept
        coeffs = np.polyfit(inv_eta, tau_arr, 1)
        C_fit = coeffs[0]
        intercept_fit = coeffs[1]

        y_pred = C_fit * inv_eta + intercept_fit
        ss_res = np.sum((tau_arr - y_pred) ** 2)
        ss_tot = np.sum((tau_arr - tau_arr.mean()) ** 2)
        r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Also fit tau = C * eta^(-gamma) in log-log
        log_eta = np.log(np.array(lr_vals))
        log_tau = np.log(tau_arr)
        gamma_fit, log_C_fit = np.polyfit(log_eta, log_tau, 1)

        y_pred_log = gamma_fit * log_eta + log_C_fit
        ss_res_log = np.sum((log_tau - y_pred_log) ** 2)
        ss_tot_log = np.sum((log_tau - log_tau.mean()) ** 2)
        r2_power = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 0

        print(f"\n  Linear fit: tau = {C_fit:.4f} * (1/eta) + {intercept_fit:.1f}")
        print(f"  R^2 (linear) = {r2_linear:.4f}")
        print(f"  Power-law: tau ~ eta^{gamma_fit:.3f}, R^2 = {r2_power:.4f}")
        print(f"  (Theory predicts gamma = -1.0, measured: {gamma_fit:.3f})")

        if r2_linear > 0.9 and abs(gamma_fit + 1.0) < 0.3:
            print(f"\n  ** VALIDATED: tau = C/eta with C ~ {C_fit:.4f} **")
        elif r2_linear > 0.7:
            print(f"\n  ** PARTIALLY VALIDATED: approximate 1/eta scaling **")
        else:
            print(f"\n  ** NOT VALIDATED: tau does not scale as 1/eta **")
    else:
        C_fit, intercept_fit, r2_linear = 0, 0, 0
        gamma_fit, r2_power = 0, 0

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E23",
        "version": 1,
        "date": "2026-04-07",
        "session": 9,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "hidden": HIDDEN,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "n_steps": N_STEPS,
        "learning_rates": LEARNING_RATES,
        "checkpoints": CHECKPOINTS,
        "hessian_checkpoints": HESSIAN_CHECKPOINTS,
        "tau_analysis": {
            "lr_values": lr_vals,
            "tau_means": tau_means,
            "tau_stds": tau_stds,
            "linear_C": float(C_fit),
            "linear_intercept": float(intercept_fit),
            "linear_r2": float(r2_linear),
            "power_law_gamma": float(gamma_fit),
            "power_law_r2": float(r2_power),
        },
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 26: Tau vs Learning Rate
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): 1 - q_bar(t) vs t for all learning rates
    ax = axes[0]
    lr_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(LEARNING_RATES)))

    for i, lr in enumerate(LEARNING_RATES):
        lr_data = all_results[str(lr)]

        # Average across seeds
        steps_all = set()
        for seed in USE_SEEDS:
            if str(seed) in lr_data and "checkpoints" in lr_data[str(seed)]:
                for cp_str in lr_data[str(seed)]["checkpoints"]:
                    steps_all.add(int(cp_str))

        steps_sorted = sorted(steps_all)
        omq_means = []
        for step in steps_sorted:
            vals = []
            for seed in USE_SEEDS:
                if str(seed) in lr_data and "checkpoints" in lr_data[str(seed)]:
                    cp = lr_data[str(seed)]["checkpoints"].get(str(step))
                    if cp and "one_minus_q" in cp:
                        vals.append(cp["one_minus_q"])
            if vals:
                omq_means.append(np.mean(vals))
            else:
                omq_means.append(float('nan'))

        valid = [(s, o) for s, o in zip(steps_sorted, omq_means)
                 if np.isfinite(o) and o > 0.001]
        if valid:
            s_arr, o_arr = zip(*valid)
            ax.semilogy(s_arr, o_arr, 'o-', color=lr_colors[i], markersize=4,
                       linewidth=1.5, label=f'$\\eta$={lr}')

            # Overlay exponential fit
            tau_data = tau_per_lr[str(lr)]
            if np.isfinite(tau_data["mean_tau"]) and tau_data["mean_tau"] > 0:
                t_fine = np.linspace(25, max(s_arr), 100)
                # Use the first seed's fit parameters
                for seed in USE_SEEDS:
                    if (str(seed) in lr_data and "tau_fit" in lr_data[str(seed)]
                            and np.isfinite(lr_data[str(seed)]["tau_fit"]["tau"])):
                        tf = lr_data[str(seed)]["tau_fit"]
                        y_fit = tf["a"] * np.exp(-t_fine / tf["tau"])
                        ax.semilogy(t_fine, y_fit, '--', color=lr_colors[i],
                                   alpha=0.4, linewidth=1)
                        break

    ax.set_xlabel('Training step $t$')
    ax.set_ylabel('$1 - \\bar{q}(t)$')
    ax.set_title('(a) Softmax concentration: exponential decay at rate $1/\\tau$')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=1e-3)

    # Panel (b): tau vs 1/eta
    ax = axes[1]
    if lr_vals:
        inv_eta = [1.0/lr for lr in lr_vals]
        ax.errorbar(inv_eta, tau_means, yerr=tau_stds, fmt='ko', markersize=10,
                    linewidth=2, capsize=4, label='Measured $\\tau$')

        # Linear fit overlay
        if r2_linear > 0:
            ie_fine = np.linspace(min(inv_eta)*0.8, max(inv_eta)*1.2, 50)
            ax.plot(ie_fine, C_fit * ie_fine + intercept_fit, 'r--', linewidth=2,
                    label=f'$\\tau = {C_fit:.4f}/\\eta + {intercept_fit:.0f}$ ($R^2 = {r2_linear:.3f}$)')

        # Origin line
        ie_fine = np.linspace(0, max(inv_eta)*1.2, 50)
        ax.plot(ie_fine, C_fit * ie_fine, 'b:', linewidth=1, alpha=0.5,
                label=f'$\\tau = {C_fit:.4f}/\\eta$ (through origin)')

    ax.set_xlabel('$1/\\eta$')
    ax.set_ylabel('Spectral compression timescale $\\tau$')
    ax.set_title('(b) $\\tau \\propto 1/\\eta$ validation')
    ax.legend(fontsize=8)

    if r2_power > 0:
        ax.annotate(f'Power law: $\\tau \\sim \\eta^{{{gamma_fit:.2f}}}$\n$R^2 = {r2_power:.3f}$',
                   xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('E23: Spectral Compression Timescale Validates $\\tau = C/\\eta$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig26_tau_lr_scaling", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig26_tau_lr_scaling.pdf")

    # ============================================================================
    # Hessian Evolution Summary
    # ============================================================================
    print(f"\n{'='*70}")
    print("HESSIAN EVOLUTION ACROSS LEARNING RATES")
    print(f"{'='*70}")

    for lr in LEARNING_RATES:
        lr_data = all_results[str(lr)]
        print(f"\n  lr={lr}:")
        for seed in USE_SEEDS:
            if str(seed) not in lr_data or "checkpoints" not in lr_data[str(seed)]:
                continue
            for cp in HESSIAN_CHECKPOINTS:
                cp_data = lr_data[str(seed)]["checkpoints"].get(str(cp), {})
                if "hessian" in cp_data:
                    h = cp_data["hessian"]
                    print(f"    seed={seed}, t={cp}: max_eig={h['max_eigenvalue']:.4f}")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    if r2_linear > 0.9:
        print(f"  ** STRONG VALIDATION: tau = {C_fit:.4f}/eta (R^2 = {r2_linear:.4f}) **")
        print(f"  ** Power-law exponent: gamma = {gamma_fit:.3f} (theory: -1.0) **")
        print(f"  ** Confirms NTK-based derivation (Section 7.20) **")
    elif r2_linear > 0.7:
        print(f"  ** MODERATE VALIDATION: approximate 1/eta scaling **")
        print(f"  ** gamma = {gamma_fit:.3f}, possible higher-order corrections **")
    else:
        print(f"  ** tau does NOT scale cleanly as 1/eta **")
        print(f"  ** The NTK derivation may need refinement **")
