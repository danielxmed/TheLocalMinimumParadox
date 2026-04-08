"""
Experiment: Per-Step Gradient Imbalance Decomposition of Conservation Drift
Theory: Theory A -- Explaining WHY drift ~ eta^1.16 (not eta^2)
Prediction: Total drift = eta^2 * S(eta) where S(eta) = sum of per-step gradient
            imbalances. We predict S(eta) ~ eta^{-0.84}, explaining the sub-quadratic
            exponent as a convergence-speedup effect.
Date: 2026-04-07 (Session 4)

Key insight: The exact per-step drift is
  Delta_C_l = eta^2 * [||dL/dW_{l+1}||^2 - ||dL/dW_l||^2]
so total drift = eta^2 * S(eta) where S = sum of gradient imbalances.
For drift ~ eta^alpha with alpha < 2, we need S(eta) ~ eta^{-(2-alpha)}.

This experiment tracks per-layer gradient norms at EVERY step to:
1. Verify the decomposition (E1)
2. Relate S(eta) to the loss integral (E2)
3. Analyze sign alternation patterns at Edge of Stability (E4)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "gradient_imbalance"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LRS = [0.0001, 0.001, 0.01, 0.1, 1.0]
N_STEPS = 2000
HIDDEN = 64


def run_gradient_imbalance_tracking(lr, seed, arch="mlp_2layer_nobias"):
    """Train and record per-layer gradient norms at every step."""
    set_seed(seed)
    model = make_model(arch, 20, 5, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    # Identify weight layers (skip ReLU activations)
    weight_params = [(name, p) for name, p in model.named_parameters() if 'weight' in name]
    n_layers = len(weight_params)

    # Record initial conservation quantities
    norms_init = [p.data.norm().item() ** 2 for _, p in weight_params]
    C_init = [norms_init[l+1] - norms_init[l] for l in range(n_layers - 1)]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Per-step tracking arrays
    grad_norms_per_layer = [[] for _ in range(n_layers)]  # ||dL/dW_l||^2
    grad_imbalance = []       # delta(t) = ||dL/dW_2||^2 - ||dL/dW_1||^2
    cumulative_S = []         # cumulative sum of delta(t)
    losses = []               # loss at each step
    total_grad_norms = []     # ||grad L||^2 at each step

    running_S = 0.0

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        # Record per-layer gradient norms BEFORE the step
        step_grad_norms = []
        total_gn = 0.0
        for l, (name, p) in enumerate(weight_params):
            gn = p.grad.data.norm().item() ** 2
            grad_norms_per_layer[l].append(gn)
            step_grad_norms.append(gn)
            total_gn += gn

        # Gradient imbalance: delta = ||dL/dW_2||^2 - ||dL/dW_1||^2
        # For multi-layer, average over consecutive pairs
        delta = 0.0
        for l in range(n_layers - 1):
            delta += step_grad_norms[l+1] - step_grad_norms[l]
        delta /= (n_layers - 1)

        grad_imbalance.append(delta)
        running_S += delta
        cumulative_S.append(running_S)
        losses.append(loss.item())
        total_grad_norms.append(total_gn)

        optimizer.step()

    # Final conservation quantities
    norms_final = [p.data.norm().item() ** 2 for _, p in weight_params]
    C_final = [norms_final[l+1] - norms_final[l] for l in range(n_layers - 1)]

    # Actual drift (should match eta^2 * S)
    actual_drift = np.mean([abs(C_final[l] - C_init[l]) for l in range(n_layers - 1)])

    # S(eta) = sum of gradient imbalances (the quantity whose eta-dependence we study)
    S_eta = abs(sum(grad_imbalance))

    # Predicted drift from decomposition
    predicted_drift = lr ** 2 * S_eta

    # Loss integral (trapezoidal rule)
    loss_integral = np.trapz(losses)

    # Gradient norm integral
    grad_integral = np.trapz(total_grad_norms)

    # Sign alternation analysis
    signs = np.sign(grad_imbalance)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    sign_change_rate = sign_changes / (N_STEPS - 1)

    # Hurst exponent via rescaled range (R/S) analysis
    gi_array = np.array(grad_imbalance)
    cumsum = np.cumsum(gi_array - gi_array.mean())
    R = cumsum.max() - cumsum.min()
    S_std = gi_array.std()
    hurst_rs = np.log(R / S_std) / np.log(N_STEPS) if S_std > 1e-15 else 0.5

    return {
        "lr": lr,
        "seed": seed,
        "actual_drift": float(actual_drift),
        "S_eta": float(S_eta),
        "predicted_drift": float(predicted_drift),
        "decomposition_error": float(abs(actual_drift - predicted_drift) / max(actual_drift, 1e-15)),
        "loss_integral": float(loss_integral),
        "grad_integral": float(grad_integral),
        "final_loss": float(losses[-1]),
        "sign_change_rate": float(sign_change_rate),
        "hurst_exponent": float(hurst_rs),
        "n_positive_delta": int(np.sum(gi_array > 0)),
        "n_negative_delta": int(np.sum(gi_array < 0)),
        # Time series (downsample for storage: every 10 steps)
        "grad_imbalance_trajectory": [float(x) for x in gi_array[::10]],
        "cumulative_S_trajectory": [float(x) for x in np.array(cumulative_S)[::10]],
        "loss_trajectory": [float(x) for x in np.array(losses)[::10]],
        "grad_norm_trajectory": [float(x) for x in np.array(total_grad_norms)[::10]],
        # Per-layer gradient norms (downsampled)
        "layer_grad_norms": [[float(x) for x in np.array(gn)[::10]]
                             for gn in grad_norms_per_layer],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Per-Step Gradient Imbalance Decomposition of Conservation Drift")
    print("=" * 70)
    print(f"Architecture: mlp_2layer_nobias, hidden={HIDDEN}")
    print(f"Data: Gaussian mixture (n=200, d=20, K=5)")
    print(f"Learning rates: {LRS}")
    print(f"Steps: {N_STEPS}, Seeds: {SEEDS[:5]}")
    print()

    all_results = {}
    t_start = time.time()

    for lr in LRS:
        print(f"\n--- Learning rate: {lr} ---")
        seed_results = []
        for seed in SEEDS[:5]:
            r = run_gradient_imbalance_tracking(lr, seed)
            seed_results.append(r)
            print(f"  seed={seed}: actual_drift={r['actual_drift']:.6f}, "
                  f"S(eta)={r['S_eta']:.4f}, "
                  f"predicted={r['predicted_drift']:.6f}, "
                  f"error={r['decomposition_error']:.4f}, "
                  f"sign_changes={r['sign_change_rate']:.3f}, "
                  f"hurst={r['hurst_exponent']:.3f}")

        # Aggregate across seeds
        mean_drift = np.mean([r['actual_drift'] for r in seed_results])
        mean_S = np.mean([r['S_eta'] for r in seed_results])
        std_S = np.std([r['S_eta'] for r in seed_results])
        mean_error = np.mean([r['decomposition_error'] for r in seed_results])
        mean_loss_int = np.mean([r['loss_integral'] for r in seed_results])
        mean_grad_int = np.mean([r['grad_integral'] for r in seed_results])
        mean_sign_rate = np.mean([r['sign_change_rate'] for r in seed_results])
        mean_hurst = np.mean([r['hurst_exponent'] for r in seed_results])

        print(f"\n  AGGREGATE: drift={mean_drift:.6f}, S(eta)={mean_S:.4f} +/- {std_S:.4f}")
        print(f"  decomposition_error={mean_error:.6f}")
        print(f"  drift/eta^2 = {mean_drift/lr**2:.4f}")
        print(f"  loss_integral={mean_loss_int:.4f}, grad_integral={mean_grad_int:.4f}")
        print(f"  S/grad_integral={mean_S/mean_grad_int:.6f}" if mean_grad_int > 1e-10 else "")
        print(f"  sign_change_rate={mean_sign_rate:.3f}, hurst={mean_hurst:.3f}")

        all_results[str(lr)] = {
            "lr": lr,
            "mean_actual_drift": float(mean_drift),
            "mean_S_eta": float(mean_S),
            "std_S_eta": float(std_S),
            "drift_over_eta_sq": float(mean_drift / lr**2),
            "mean_decomposition_error": float(mean_error),
            "mean_loss_integral": float(mean_loss_int),
            "mean_grad_integral": float(mean_grad_int),
            "S_over_grad_integral": float(mean_S / mean_grad_int) if mean_grad_int > 1e-10 else None,
            "mean_sign_change_rate": float(mean_sign_rate),
            "mean_hurst_exponent": float(mean_hurst),
            "seed_results": seed_results,
        }

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total time: {t_total:.1f}s")

    # === ANALYSIS 1: Power law fit of S(eta) vs eta ===
    lrs_arr = np.array(LRS)
    S_arr = np.array([all_results[str(lr)]["mean_S_eta"] for lr in LRS])
    drift_arr = np.array([all_results[str(lr)]["mean_actual_drift"] for lr in LRS])

    # Fit drift ~ eta^alpha (direct)
    mask = drift_arr > 1e-15
    log_lr = np.log(lrs_arr[mask])
    log_drift = np.log(drift_arr[mask])
    alpha_direct, intercept_drift = np.polyfit(log_lr, log_drift, 1)
    ss_res = np.sum((log_drift - (alpha_direct * log_lr + intercept_drift))**2)
    ss_tot = np.sum((log_drift - log_drift.mean())**2)
    r2_alpha = 1 - ss_res / ss_tot

    # Fit S(eta) ~ eta^(-beta)
    S_over_eta2 = drift_arr[mask] / lrs_arr[mask]**2
    log_S = np.log(S_over_eta2)
    neg_beta, intercept_S = np.polyfit(log_lr, log_S, 1)
    ss_res_S = np.sum((log_S - (neg_beta * log_lr + intercept_S))**2)
    ss_tot_S = np.sum((log_S - log_S.mean())**2)
    r2_beta = 1 - ss_res_S / ss_tot_S

    print(f"\n--- POWER LAW ANALYSIS ---")
    print(f"Direct fit: drift ~ eta^{alpha_direct:.4f} (R^2 = {r2_alpha:.4f})")
    print(f"Decomposition: S(eta) ~ eta^{neg_beta:.4f} (R^2 = {r2_beta:.4f})")
    print(f"Consistency check: alpha = 2 + ({neg_beta:.4f}) = {2 + neg_beta:.4f}")
    print(f"  (should match direct alpha = {alpha_direct:.4f})")

    # === ANALYSIS 2: Mechanism A test (S vs grad_integral) ===
    grad_ints = np.array([all_results[str(lr)]["mean_grad_integral"] for lr in LRS])
    S_ratio = S_arr / grad_ints
    print(f"\n--- MECHANISM A TEST: S(eta) / integral(||grad||^2) ---")
    for i, lr in enumerate(LRS):
        print(f"  eta={lr}: S={S_arr[i]:.4f}, grad_int={grad_ints[i]:.4f}, ratio={S_ratio[i]:.6f}")
    print(f"  Ratio CV = {np.std(S_ratio)/np.mean(S_ratio)*100:.1f}%")
    print(f"  (Low CV => Mechanism A dominant; High CV => other mechanisms contribute)")

    # === ANALYSIS 3: Sign alternation (Mechanism B test) ===
    print(f"\n--- MECHANISM B TEST: Sign alternation at EoS ---")
    for lr in LRS:
        r = all_results[str(lr)]
        print(f"  eta={lr}: sign_change_rate={r['mean_sign_change_rate']:.3f}, "
              f"hurst={r['mean_hurst_exponent']:.3f}")
    print(f"  (High sign_change_rate + low Hurst => EoS oscillation cancellation)")

    # === Save results ===
    analysis = {
        "alpha_direct": float(alpha_direct),
        "r2_alpha": float(r2_alpha),
        "beta_decomposition": float(-neg_beta),
        "r2_beta": float(r2_beta),
        "alpha_from_decomposition": float(2 + neg_beta),
        "mechanism_A_ratio_cv": float(np.std(S_ratio)/np.mean(S_ratio)*100),
    }

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 4,
        "hardware": get_hardware_info(),
        "seeds": SEEDS[:5],
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "arch": "mlp_2layer_nobias",
        "dataset": "gaussian_mixture",
        "analysis": analysis,
    }

    # Save main results (without trajectories for compactness)
    summary = {}
    for lr_key, r in all_results.items():
        summary[lr_key] = {k: v for k, v in r.items() if k != "seed_results"}
    save_results(summary, OUTPUT_DIR, config)

    # Save trajectories separately (larger file)
    trajectories = {}
    for lr_key, r in all_results.items():
        trajectories[lr_key] = {
            "lr": r["lr"],
            "seed_trajectories": [
                {
                    "seed": sr["seed"],
                    "grad_imbalance": sr["grad_imbalance_trajectory"],
                    "cumulative_S": sr["cumulative_S_trajectory"],
                    "loss": sr["loss_trajectory"],
                    "grad_norm": sr["grad_norm_trajectory"],
                    "layer_grad_norms": sr["layer_grad_norms"],
                }
                for sr in r["seed_results"]
            ]
        }
    traj_path = os.path.join(OUTPUT_DIR, "trajectories.json")
    with open(traj_path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nTrajectories saved to {traj_path}")

    # === FIGURES ===
    setup_publication_style()

    # Figure 1: 2x2 panel - the key decomposition
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel (a): drift vs eta with decomposition
    ax = axes[0, 0]
    ax.loglog(lrs_arr, drift_arr, 'o-', color='darkblue', markersize=7, label='Measured drift')
    lr_fit = np.logspace(np.log10(min(LRS)), np.log10(max(LRS)), 100)
    ax.loglog(lr_fit, np.exp(intercept_drift) * lr_fit**alpha_direct, 'r--', lw=1.5,
              label=f'Fit: $\\eta^{{{alpha_direct:.2f}}}$ ($R^2$={r2_alpha:.3f})')
    ax.loglog(lr_fit, np.exp(intercept_drift) * lr_fit**2 * lrs_arr[0]**(alpha_direct-2),
              'g:', lw=1, alpha=0.6, label='$\\eta^2$ reference')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Conservation drift $|\\Delta C|$')
    ax.set_title('(a) Drift scaling: $\\Delta C \\propto \\eta^{\\alpha}$')
    ax.legend(fontsize=10)

    # Panel (b): S(eta) = drift/eta^2 vs eta
    ax = axes[0, 1]
    ax.loglog(lrs_arr, S_over_eta2, 's-', color='darkred', markersize=7,
              label='$S(\\eta) = \\Delta C / \\eta^2$')
    ax.loglog(lr_fit, np.exp(intercept_S) * lr_fit**neg_beta, 'r--', lw=1.5,
              label=f'Fit: $\\eta^{{{neg_beta:.2f}}}$ ($R^2$={r2_beta:.3f})')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Gradient imbalance sum $S(\\eta)$')
    ax.set_title(f'(b) $S(\\eta) \\propto \\eta^{{{neg_beta:.2f}}}$, so $\\alpha = 2{neg_beta:+.2f} = {2+neg_beta:.2f}$')
    ax.legend(fontsize=10)

    # Panel (c): Gradient imbalance trajectory for different eta
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(LRS)))
    for i, lr in enumerate(LRS):
        traj = all_results[str(lr)]["seed_results"][0]["grad_imbalance_trajectory"]
        steps = np.arange(0, N_STEPS, 10)[:len(traj)]
        ax.plot(steps, traj, color=colors[i], alpha=0.8, lw=0.8,
                label=f'$\\eta$={lr}')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Gradient imbalance $\\delta(t)$')
    ax.set_title('(c) Per-step gradient imbalance trajectory')
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=0, color='gray', ls=':', lw=0.5)

    # Panel (d): Sign change rate and Hurst exponent vs eta
    ax = axes[1, 1]
    sign_rates = [all_results[str(lr)]["mean_sign_change_rate"] for lr in LRS]
    hursts = [all_results[str(lr)]["mean_hurst_exponent"] for lr in LRS]
    ax2 = ax.twinx()
    l1, = ax.semilogx(lrs_arr, sign_rates, 'o-', color='darkblue', markersize=7,
                       label='Sign change rate')
    l2, = ax2.semilogx(lrs_arr, hursts, 's-', color='darkred', markersize=7,
                        label='Hurst exponent')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Sign change rate', color='darkblue')
    ax2.set_ylabel('Hurst exponent $H$', color='darkred')
    ax.set_title('(d) EoS oscillation signatures')
    ax2.axhline(y=0.5, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.legend(handles=[l1, l2], fontsize=10, loc='center left')

    fig.suptitle('Drift Exponent Decomposition: $\\Delta C = \\eta^2 \\cdot S(\\eta)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig7_gradient_imbalance_decomposition", FIGURE_DIR)
    plt.close()

    # Figure 2: Mechanism A test (S vs grad integral)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(grad_ints, S_arr, 'o', color='darkblue', markersize=8)
    for i, lr in enumerate(LRS):
        ax1.annotate(f'$\\eta$={lr}', (grad_ints[i], S_arr[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    # Fit
    log_gi = np.log(grad_ints)
    log_S_direct = np.log(S_arr)
    slope_gi, int_gi = np.polyfit(log_gi, log_S_direct, 1)
    gi_fit = np.logspace(np.log10(grad_ints.min()), np.log10(grad_ints.max()), 50)
    ax1.loglog(gi_fit, np.exp(int_gi) * gi_fit**slope_gi, 'r--', lw=1.5,
               label=f'slope={slope_gi:.2f}')
    ax1.set_xlabel('$\\int_0^T \\|\\nabla L\\|^2 \\, dt$')
    ax1.set_ylabel('$S(\\eta)$')
    ax1.set_title('(a) Mechanism A: S vs gradient integral')
    ax1.legend()

    ax2.semilogx(lrs_arr, S_ratio, 'o-', color='darkgreen', markersize=8)
    ax2.set_xlabel('Learning rate $\\eta$')
    ax2.set_ylabel('$S(\\eta) \\, / \\, \\int \\|\\nabla L\\|^2 dt$')
    ax2.set_title(f'(b) Ratio (CV = {np.std(S_ratio)/np.mean(S_ratio)*100:.1f}%)')
    ax2.axhline(y=np.mean(S_ratio), color='gray', ls='--', lw=1, alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "fig7b_mechanism_A_test", FIGURE_DIR)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}")
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  Direct alpha = {alpha_direct:.4f}")
    print(f"  Decomposition: alpha = 2 - beta = 2 - {-neg_beta:.4f} = {2+neg_beta:.4f}")
    print(f"  Mechanism A ratio CV = {np.std(S_ratio)/np.mean(S_ratio)*100:.1f}%")
    print(f"{'=' * 70}")
