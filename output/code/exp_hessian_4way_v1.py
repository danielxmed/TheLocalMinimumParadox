"""
Experiment: 4-Way Hessian Comparison — {MSE, CrossEntropy} x {Linear, ReLU}
Theory: Theory A -- Loss-Function Spectral Mechanism
Prediction: The Hessian spectrum alone (via Theorem 5) should predict the drift
            exponent alpha for all 4 combinations. If CrossEntropy compresses the
            spectrum relative to MSE, this explains why CE decreases alpha.
Date: 2026-04-07 (Session 6)

Method:
  1. For each of {MSE, CE} x {Linear, ReLU}:
     - Compute full Hessian at initialization
     - Extract eigenvalues and eigenvectors
     - Compute c_k from initial gradient projection
     - Predict S(eta) via Theorem 5
     - Train and measure actual S(eta)
     - Fit alpha from log-log regression
  2. Compare 4 predicted alphas vs 4 measured alphas
  3. Compare eigenvalue distributions across loss functions
  4. Compute spectral ratio lambda_CE / lambda_MSE

Key question: Does the loss function modify alpha purely through the Hessian spectrum?
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset, count_parameters,
    full_hessian, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "hessian_4way"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
N_STEPS = 2000
HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200

# The 4 combinations
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
    """2-layer linear network: f(x) = W_2 W_1 x (no activation, no bias)."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.W2(self.W1(x))


# ============================================================================
# Theorem 5: Spectral Crossover Formula
# ============================================================================

def theorem5_predict_S(eigenvalues, c_k, eta, T):
    """
    Predict S(eta) from Hessian eigenvalues using Theorem 5.
    S(eta) = sum_k c_k * (1 - rho_k^{2T}) / (eta * lam_k * (2 - eta * lam_k))
    where rho_k = 1 - eta * lam_k
    """
    S = 0.0
    for k, lam_k in enumerate(eigenvalues):
        if abs(lam_k) < 1e-12 or lam_k < 0:
            continue

        rho_k = 1.0 - eta * lam_k
        denom = eta * lam_k * (2.0 - eta * lam_k)

        if abs(rho_k) >= 1.0:
            if abs(denom) > 1e-12:
                S += c_k[k] * T / max(abs(denom), 1e-12)
            continue

        numerator = 1.0 - rho_k ** (2 * T)
        if abs(denom) < 1e-12:
            continue
        S += c_k[k] * numerator / denom

    return S


def compute_initial_c_k(model, X, Y, loss_fn, eigenvectors, eigenvalues):
    """
    Estimate c_k from the initial gradient projected onto Hessian eigenmodes.
    """
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y)
    loss.backward()

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    if len(weight_params) < 2:
        return np.ones(len(eigenvalues))

    gn1 = weight_params[0][1].grad.data.norm().item() ** 2
    gn2 = weight_params[1][1].grad.data.norm().item() ** 2
    delta_0 = gn2 - gn1

    grad_vec = torch.cat([p.grad.data.flatten() for _, p in weight_params])
    grad_np = grad_vec.numpy()
    projections = eigenvectors.T @ grad_np

    c_k = projections ** 2
    total = np.sum(c_k)
    if total > 1e-12:
        c_k = c_k * abs(delta_0) / total

    return c_k


# ============================================================================
# Training and Measurement
# ============================================================================

def create_model_and_data(arch, loss_type, seed):
    """Create model, data, and loss function for a given combination."""
    set_seed(seed)

    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    # One-hot for MSE
    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    if arch == "linear":
        model = LinearTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
    else:
        model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
        Y_train = Y_onehot
    else:
        loss_fn = nn.CrossEntropyLoss()
        Y_train = Y

    return model, X, Y_train, loss_fn, Y_onehot, Y


def train_and_measure(model, X, Y, loss_fn, lr, n_steps):
    """Train model and track per-step gradient imbalance for S(eta) measurement."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]

    C_init = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2

    grad_imbalances = []
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        gn1 = weight_params[0][1].grad.data.norm().item() ** 2
        gn2 = weight_params[1][1].grad.data.norm().item() ** 2
        grad_imbalances.append(gn2 - gn1)
        losses.append(loss.item())

        optimizer.step()

        if np.isnan(loss.item()):
            break

    C_final = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2
    actual_drift = abs(C_final - C_init)
    S_measured = abs(sum(grad_imbalances))

    return {
        "actual_drift": float(actual_drift),
        "S_measured": float(S_measured),
        "final_loss": float(losses[-1]) if losses and not np.isnan(losses[-1]) else float('inf'),
        "converged": len(losses) == n_steps and not np.isnan(losses[-1]),
    }


def run_4way(arch, loss_type, seed):
    """Full pipeline for one (arch, loss, seed) combination."""
    model, X, Y_train, loss_fn, Y_onehot, Y_int = create_model_and_data(arch, loss_type, seed)
    n_params = count_parameters(model)
    combo_name = f"{arch}_{loss_type}"
    print(f"  [{combo_name}] seed={seed}, params={n_params}")

    # ---- Step 1: Compute Hessian at initialization ----
    print(f"    Computing full Hessian ({n_params}x{n_params})...")
    t0 = time.time()
    H_init = full_hessian(model, X, Y_train, loss_fn)
    H_np = H_init.numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    t_hessian = time.time() - t0
    print(f"    Hessian computed in {t_hessian:.1f}s. Eigenvalues: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")

    # ---- Step 2: Compute c_k ----
    c_k = compute_initial_c_k(model, X, Y_train, loss_fn, eigenvectors, eigenvalues)

    # ---- Step 3: Predict S(eta) from Theorem 5 ----
    predictions = {}
    for lr in LRS:
        S_pred = theorem5_predict_S(eigenvalues, c_k, lr, N_STEPS)
        predictions[str(lr)] = float(S_pred)

    # ---- Step 4: Train and measure actual S(eta) ----
    measurements = {}
    for lr in LRS:
        set_seed(seed)
        model_train, X_t, Y_t, loss_fn_t, _, _ = create_model_and_data(arch, loss_type, seed)
        result = train_and_measure(model_train, X_t, Y_t, loss_fn_t, lr, N_STEPS)
        measurements[str(lr)] = result
        S_meas = result["S_measured"]
        S_pred = predictions[str(lr)]
        ratio = S_pred / S_meas if S_meas > 1e-12 else float('inf')
        print(f"    lr={lr:.4f}: S_meas={S_meas:.4f}, S_pred={S_pred:.4f}, ratio={ratio:.3f}")

    # Eigenvalue statistics
    evals = eigenvalues
    n_pos = int(np.sum(evals > 1e-8))
    n_neg = int(np.sum(evals < -1e-8))
    n_zero = int(np.sum(np.abs(evals) <= 1e-8))
    evals_pos = evals[evals > 1e-8]

    return {
        "arch": arch,
        "loss_type": loss_type,
        "seed": int(seed),
        "n_params": n_params,
        "hessian_time": t_hessian,
        "eigenvalues": eigenvalues.tolist(),
        "eigenvalue_stats": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_zero": n_zero,
            "max": float(evals[-1]),
            "min": float(evals[0]),
            "median_positive": float(np.median(evals_pos)) if n_pos > 0 else 0.0,
            "mean_positive": float(np.mean(evals_pos)) if n_pos > 0 else 0.0,
            "effective_rank": float(np.sum(evals_pos) / evals_pos[-1]) if n_pos > 0 and evals_pos[-1] > 0 else 0.0,
            "spectral_width": float(evals_pos[-1] / np.median(evals_pos)) if n_pos > 0 and np.median(evals_pos) > 1e-12 else float('inf'),
        },
        "predictions": predictions,
        "measurements": measurements,
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E12: 4-Way Hessian Comparison — {MSE, CE} x {Linear, ReLU}")
    print("=" * 70)
    print(f"Architecture: 2-layer, hidden={HIDDEN}, no bias")
    print(f"Data: Gaussian mixture (n={N_TRAIN}, d={INPUT_DIM}, K={OUTPUT_DIM})")
    print(f"Learning rates: {LRS}")
    print(f"Training steps: {N_STEPS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}
    for arch, loss_type in COMBOS:
        combo = f"{arch}_{loss_type}"
        print(f"\n{'='*50}")
        print(f"  Combination: {combo.upper()}")
        print(f"{'='*50}")
        all_results[combo] = []
        for seed in USE_SEEDS:
            result = run_4way(arch, loss_type, seed)
            all_results[combo].append(result)

    # ============================================================================
    # Aggregate Results
    # ============================================================================
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")

    summary = {}
    for arch, loss_type in COMBOS:
        combo = f"{arch}_{loss_type}"
        print(f"\n--- {combo.upper()} ---")

        S_measured_avg = {}
        S_predicted_avg = {}

        for lr in LRS:
            S_m_list = []
            S_p_list = []
            for r in all_results[combo]:
                m = r["measurements"].get(str(lr))
                p = r["predictions"].get(str(lr))
                if m and m["converged"]:
                    S_m_list.append(m["S_measured"])
                    S_p_list.append(p)

            if S_m_list:
                S_m_mean = np.mean(S_m_list)
                S_p_mean = np.mean(S_p_list)
                ratio = S_p_mean / S_m_mean if S_m_mean > 1e-12 else float('inf')
                S_measured_avg[str(lr)] = float(S_m_mean)
                S_predicted_avg[str(lr)] = float(S_p_mean)
                print(f"  lr={lr:.4f}: S_meas={S_m_mean:>10.4f}, S_pred={S_p_mean:>10.4f}, ratio={ratio:.3f}")

        # Power-law fit on measured S(eta) -> drift ~ eta^alpha
        valid_lrs = [float(lr) for lr in LRS if str(lr) in S_measured_avg and S_measured_avg[str(lr)] > 1e-12]
        valid_S = [S_measured_avg[str(lr)] for lr in valid_lrs]

        if len(valid_lrs) >= 3:
            log_lr = np.log(valid_lrs)

            # Fit drift = actual_drift ~ eta^alpha
            # Collect actual drifts for the fit
            drift_avg = {}
            for lr in LRS:
                d_list = [r["measurements"][str(lr)]["actual_drift"]
                          for r in all_results[combo]
                          if r["measurements"][str(lr)]["converged"]]
                if d_list:
                    drift_avg[str(lr)] = float(np.mean(d_list))

            valid_drifts = [drift_avg[str(lr)] for lr in valid_lrs if str(lr) in drift_avg and drift_avg[str(lr)] > 1e-15]
            valid_lrs_d = [lr for lr in valid_lrs if str(lr) in drift_avg and drift_avg[str(lr)] > 1e-15]

            if len(valid_lrs_d) >= 3:
                log_lr_d = np.log(valid_lrs_d)
                log_drift_d = np.log(valid_drifts)
                alpha_slope, intercept = np.polyfit(log_lr_d, log_drift_d, 1)
                ss_res = np.sum((log_drift_d - (alpha_slope * log_lr_d + intercept)) ** 2)
                ss_tot = np.sum((log_drift_d - np.mean(log_drift_d)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                alpha = alpha_slope
            else:
                alpha, r2 = None, None

            # S(eta) scaling: S = drift/eta^2, so S ~ eta^(alpha-2) = eta^beta
            log_S = np.log(valid_S)
            beta, _ = np.polyfit(log_lr[:len(valid_S)], log_S, 1)

            # Prediction correlation
            valid_S_pred = [S_predicted_avg[str(lr)] for lr in valid_lrs if S_predicted_avg.get(str(lr), 0) > 0]
            valid_S_meas_corr = [S_measured_avg[str(lr)] for lr in valid_lrs if S_predicted_avg.get(str(lr), 0) > 0]
            if len(valid_S_pred) >= 3:
                corr = np.corrcoef(np.log(np.array(valid_S_pred)+1e-20),
                                   np.log(np.array(valid_S_meas_corr)+1e-20))[0, 1]
            else:
                corr = None
        else:
            alpha, r2, beta, corr = None, None, None, None

        if alpha is not None:
            print(f"  drift ~ eta^{alpha:.3f} (R^2 = {r2:.4f})")
        if corr is not None:
            print(f"  Prediction correlation R = {corr:.4f}")

        # Eigenvalue stats (first seed)
        ev_stats = all_results[combo][0]["eigenvalue_stats"]
        print(f"  Eigenvalues: {ev_stats['n_positive']} pos, {ev_stats['n_negative']} neg, "
              f"{ev_stats['n_zero']} zero, max={ev_stats['max']:.4f}")

        summary[combo] = {
            "alpha": float(alpha) if alpha is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "beta": float(beta) if beta is not None else None,
            "prediction_correlation": float(corr) if corr is not None else None,
            "S_measured_avg": S_measured_avg,
            "S_predicted_avg": S_predicted_avg,
            "eigenvalue_stats": ev_stats,
        }

    # ============================================================================
    # Cross-Comparison: The Key Analysis
    # ============================================================================
    print(f"\n{'='*70}")
    print("CROSS-COMPARISON: THE THREE-FACTOR DECOMPOSITION")
    print(f"{'='*70}")
    print(f"\n{'Combination':<20} {'alpha':>10} {'R^2':>8} {'Pred R':>8} {'max_eig':>10} {'eff_rank':>10}")
    print("-" * 70)
    for combo_key in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        s = summary.get(combo_key, {})
        alpha_val = s.get("alpha")
        r2_val = s.get("r2")
        corr_val = s.get("prediction_correlation")
        ev = s.get("eigenvalue_stats", {})
        print(f"{combo_key:<20} {alpha_val if alpha_val else 'N/A':>10.3f} "
              f"{r2_val if r2_val else 0:>8.4f} "
              f"{corr_val if corr_val else 0:>8.4f} "
              f"{ev.get('max', 0):>10.4f} "
              f"{ev.get('effective_rank', 0):>10.1f}")

    # Decomposition analysis
    print(f"\n--- DECOMPOSITION ---")
    a = {k: summary[k]["alpha"] for k in summary if summary[k]["alpha"] is not None}
    if "linear_mse" in a and "relu_mse" in a:
        print(f"  ReLU mode coupling (MSE fixed):   delta_alpha = {a['relu_mse'] - a['linear_mse']:.3f}")
    if "relu_mse" in a and "relu_ce" in a:
        print(f"  CE spectral effect (ReLU fixed):  delta_alpha = {a['relu_ce'] - a['relu_mse']:.3f}")
    if "linear_mse" in a and "linear_ce" in a:
        print(f"  CE spectral effect (Linear fixed): delta_alpha = {a['linear_ce'] - a['linear_mse']:.3f}")
    if "linear_mse" in a and "relu_ce" in a:
        print(f"  Total (linear_mse -> relu_ce):    delta_alpha = {a['relu_ce'] - a['linear_mse']:.3f}")

    # Spectral ratio analysis
    print(f"\n--- SPECTRAL RATIO: CE vs MSE EIGENVALUES ---")
    for arch in ["linear", "relu"]:
        combo_mse = f"{arch}_mse"
        combo_ce = f"{arch}_ce"
        if combo_mse in all_results and combo_ce in all_results:
            evals_mse = np.array(all_results[combo_mse][0]["eigenvalues"])
            evals_ce = np.array(all_results[combo_ce][0]["eigenvalues"])

            # Compare positive eigenvalues
            pos_mse = evals_mse[evals_mse > 1e-8]
            pos_ce = evals_ce[evals_ce > 1e-8]

            print(f"\n  {arch.upper()}:")
            print(f"    MSE: {len(pos_mse)} positive eigenvalues, max={pos_mse[-1]:.4f}, median={np.median(pos_mse):.6f}")
            print(f"    CE:  {len(pos_ce)} positive eigenvalues, max={pos_ce[-1]:.4f}, median={np.median(pos_ce):.6f}")
            print(f"    Max ratio (CE/MSE): {pos_ce[-1]/pos_mse[-1]:.4f}")
            print(f"    Median ratio: {np.median(pos_ce)/np.median(pos_mse):.4f}" if np.median(pos_mse) > 1e-12 else "")

            # For comparable eigenvalues (sorted), compute ratios
            n_compare = min(len(pos_mse), len(pos_ce))
            if n_compare > 10:
                ratios = pos_ce[-n_compare:] / (pos_mse[-n_compare:] + 1e-15)
                print(f"    Spectral ratio (top {n_compare}): mean={np.mean(ratios):.4f}, "
                      f"std={np.std(ratios):.4f}, range=[{ratios.min():.4f}, {ratios.max():.4f}]")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 6,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "combinations": [f"{a}_{l}" for a, l in COMBOS],
        "summary": summary,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 15: 4-Way Hessian Comparison
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

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
    labels = {
        "linear_mse": "Linear + MSE",
        "linear_ce": "Linear + CE",
        "relu_mse": "ReLU + MSE",
        "relu_ce": "ReLU + CE",
    }

    # Panel (a): Eigenvalue CDFs
    ax = axes[0, 0]
    for combo_key in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        evals = np.array(all_results[combo_key][0]["eigenvalues"])
        evals_pos = evals[evals > 1e-8]
        if len(evals_pos) > 0:
            sorted_evals = np.sort(evals_pos)
            cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals)
            ax.semilogx(sorted_evals, cdf, color=colors[combo_key], linewidth=1.5,
                        label=f'{labels[combo_key]} (n={len(evals_pos)})')
    ax.set_xlabel('Eigenvalue $\\lambda$')
    ax.set_ylabel('CDF')
    ax.set_title('(a) Hessian eigenvalue CDFs at initialization')
    ax.legend(fontsize=8)

    # Panel (b): Predicted vs Measured alpha
    ax = axes[0, 1]
    alphas_measured = []
    alphas_labels = []
    for combo_key in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        s = summary.get(combo_key, {})
        alpha_val = s.get("alpha")
        if alpha_val is not None:
            ax.scatter(combo_key, alpha_val, color=colors[combo_key], marker=markers[combo_key],
                       s=150, zorder=5, label=labels[combo_key])
            alphas_measured.append(alpha_val)
            alphas_labels.append(combo_key)
    ax.set_ylabel('Drift exponent $\\alpha$')
    ax.set_title('(b) Measured $\\alpha$ for all 4 combinations')
    ax.legend(fontsize=8)
    ax.set_xticklabels([])
    if alphas_measured:
        ax.set_ylim(min(alphas_measured) - 0.1, max(alphas_measured) + 0.1)
    # Add horizontal lines for reference
    ax.axhline(y=1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(y=2.0, color='gray', ls=':', lw=0.8, alpha=0.5)

    # Panel (c): Spectral ratio CE/MSE
    ax = axes[1, 0]
    for arch in ["linear", "relu"]:
        combo_mse = f"{arch}_mse"
        combo_ce = f"{arch}_ce"
        if combo_mse in all_results and combo_ce in all_results:
            evals_mse = np.sort(np.array(all_results[combo_mse][0]["eigenvalues"]))
            evals_ce = np.sort(np.array(all_results[combo_ce][0]["eigenvalues"]))
            # Compare only positive eigenvalues
            pos_mse = evals_mse[evals_mse > 1e-8]
            pos_ce = evals_ce[evals_ce > 1e-8]
            n_compare = min(len(pos_mse), len(pos_ce))
            if n_compare > 10:
                ratios = pos_ce[-n_compare:] / (pos_mse[-n_compare:] + 1e-15)
                ax.plot(range(n_compare), ratios, color=colors[f"{arch}_ce"],
                        linewidth=1.5, label=f'{arch.upper()}: CE/MSE ratio')
    ax.axhline(y=0.25, color='green', ls='--', lw=1, alpha=0.7, label='Softmax bound (0.25)')
    ax.axhline(y=0.16, color='orange', ls='--', lw=1, alpha=0.7, label='Uniform K=5 (0.16)')
    ax.set_xlabel('Eigenvalue index (sorted)')
    ax.set_ylabel('$\\lambda_k^{CE} / \\lambda_k^{MSE}$')
    ax.set_title('(c) Spectral compression ratio')
    ax.legend(fontsize=8)

    # Panel (d): S(eta) curves for all 4 combinations
    ax = axes[1, 1]
    for combo_key in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        s = summary.get(combo_key, {})
        sm = s.get("S_measured_avg", {})
        valid_lrs = sorted([float(k) for k in sm.keys()])
        valid_S = [sm[str(lr)] for lr in valid_lrs]
        if valid_lrs and valid_S:
            ax.loglog(valid_lrs, valid_S, f'{markers[combo_key]}-', color=colors[combo_key],
                      markersize=6, linewidth=1.5, label=labels[combo_key])
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('$S(\\eta)$')
    ax.set_title('(d) Measured $S(\\eta)$ for all 4 combinations')
    ax.legend(fontsize=8)

    fig.suptitle('E12: 4-Way Hessian Comparison — Loss-Function Spectral Mechanism',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig15_hessian_4way", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig15_hessian_4way.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    for combo_key in ["linear_mse", "linear_ce", "relu_mse", "relu_ce"]:
        s = summary.get(combo_key, {})
        alpha_val = s.get("alpha")
        corr_val = s.get("prediction_correlation")
        if alpha_val is not None:
            print(f"  {labels[combo_key]}: alpha = {alpha_val:.3f}" +
                  (f", prediction R = {corr_val:.3f}" if corr_val else ""))
