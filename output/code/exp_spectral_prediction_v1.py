"""
Experiment: Spectral Prediction of S(eta) — Testing Theorem 5 on ReLU Hessian
Theory: Theory A -- Closing the Linear-to-ReLU Gap
Prediction: If Theorem 5 (derived for linear networks) correctly predicts S(eta) for
            ReLU networks when evaluated on the actual ReLU Hessian eigenvalues, then
            mode coupling from ReLU activations is negligible and the spectral crossover
            formula is UNIVERSAL.
Date: 2026-04-07 (Session 5)

Method:
  1. Initialize 2-layer networks (linear and ReLU, hidden=64, no bias)
  2. Compute full Hessian at initialization → extract eigenvalues {lambda_k}
  3. Apply Theorem 5: S_pred(eta) = sum_k c_k * (1-(1-eta*lam_k)^{2T}) / (eta*lam_k*(2-eta*lam_k))
  4. Train and measure actual S(eta) = drift / eta^2
  5. Compare predicted vs measured for both architectures

Key question: Does the spectral crossover formula close the Linear-to-ReLU gap?
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset, count_parameters,
    full_hessian, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "spectral_prediction"
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

    For eigenvalues <= 0 or where eta*lam_k >= 2 (unstable), we handle separately.
    """
    S = 0.0
    n_stable = 0
    n_unstable = 0
    n_negative = 0
    n_zero = 0

    for k, lam_k in enumerate(eigenvalues):
        if abs(lam_k) < 1e-12:
            # Zero eigenvalue: no contribution (flat direction)
            n_zero += 1
            continue
        if lam_k < 0:
            # Negative eigenvalue (saddle direction): skip for now
            n_negative += 1
            continue

        rho_k = 1.0 - eta * lam_k
        denom = eta * lam_k * (2.0 - eta * lam_k)

        if abs(rho_k) >= 1.0:
            # Unstable mode: |rho_k| >= 1 means divergence
            # At EoS boundary (eta*lam_k = 2), rho_k = -1, contribution is T*c_k/denom
            # Beyond EoS, this mode diverges -- cap contribution
            n_unstable += 1
            if abs(denom) > 1e-12:
                # Use T * c_k as upper bound (all steps contribute)
                S += c_k[k] * T / max(abs(denom), 1e-12)
            continue

        n_stable += 1
        numerator = 1.0 - rho_k ** (2 * T)
        if abs(denom) < 1e-12:
            continue
        S += c_k[k] * numerator / denom

    return S, {"n_stable": n_stable, "n_unstable": n_unstable,
               "n_negative": n_negative, "n_zero": n_zero}


def compute_initial_c_k(model, X, Y, loss_fn, eigenvectors, eigenvalues):
    """
    Estimate c_k from the initial gradient projected onto Hessian eigenmodes.

    For the gradient imbalance decomposition, c_k relates to how much the gradient
    imbalance projects onto eigenmode k. We approximate c_k from the initial
    gradient structure.
    """
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y)
    loss.backward()

    # Get weight parameters
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    if len(weight_params) < 2:
        return np.ones(len(eigenvalues))

    # Compute per-layer gradient norms at initialization
    gn1 = weight_params[0][1].grad.data.norm().item() ** 2
    gn2 = weight_params[1][1].grad.data.norm().item() ** 2
    delta_0 = gn2 - gn1  # Initial gradient imbalance

    # Full gradient vector
    grad_vec = torch.cat([p.grad.data.flatten() for _, p in weight_params])

    # Project gradient onto eigenmodes
    grad_np = grad_vec.numpy()
    projections = eigenvectors.T @ grad_np  # projections[k] = <grad, v_k>

    # c_k proportional to |projection_k|^2 weighted by eigenvalue structure
    # The imbalance delta relates to how the gradient distributes across layers.
    # For mode k: the contribution to delta scales with |proj_k|^2 * (layer_weight_k)
    # As a practical approximation: c_k ~ |proj_k|^2 * sign(delta_0)
    c_k = projections ** 2

    # Normalize so that sum(c_k * f(lam_k)) ~ |delta_0| at initialization
    # f(lam_k, t=0) is just the first term contribution
    total = np.sum(c_k)
    if total > 1e-12:
        c_k = c_k * abs(delta_0) / total

    return c_k


# ============================================================================
# Training and Measurement
# ============================================================================

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

        # Check for NaN
        if np.isnan(loss.item()):
            break

    C_final = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2
    actual_drift = abs(C_final - C_init)
    S_measured = abs(sum(grad_imbalances))
    grad_integral = sum(gn1 + gn2 for gn1, gn2 in
                        zip([0]*len(grad_imbalances), [0]*len(grad_imbalances)))

    # Recompute grad_integral properly
    grad_integral = 0.0
    # We need total ||grad||^2 -- reuse the stored norms
    # Actually we need to recompute. Use loss history as proxy.

    return {
        "actual_drift": float(actual_drift),
        "S_measured": float(S_measured),
        "final_loss": float(losses[-1]) if losses and not np.isnan(losses[-1]) else float('inf'),
        "converged": len(losses) == n_steps and not np.isnan(losses[-1]),
        "grad_imbalances": grad_imbalances,
    }


def run_spectral_prediction(arch_type, seed, compute_hessian_at_mid=True):
    """
    Full spectral prediction pipeline for one seed.

    arch_type: "linear" or "relu"
    """
    set_seed(seed)

    # Create model
    if arch_type == "linear":
        model = LinearTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
        X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                                   d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)
        Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
        Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)
        loss_fn = nn.MSELoss()
        Y_train = Y_onehot
    else:
        model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
        X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                                   d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)
        loss_fn = nn.CrossEntropyLoss()
        Y_train = Y

    n_params = count_parameters(model)
    print(f"  [{arch_type}] seed={seed}, params={n_params}")

    # ---- Step 1: Compute Hessian at initialization ----
    print(f"    Computing full Hessian at initialization ({n_params}x{n_params})...")
    t0 = time.time()
    H_init = full_hessian(model, X, Y_train, loss_fn)
    H_np = H_init.numpy()
    eigenvalues_init, eigenvectors_init = np.linalg.eigh(H_np)
    t_hessian = time.time() - t0
    print(f"    Hessian computed in {t_hessian:.1f}s. Eigenvalue range: [{eigenvalues_init[0]:.6f}, {eigenvalues_init[-1]:.6f}]")

    # ---- Step 2: Compute c_k from initial gradient structure ----
    c_k_init = compute_initial_c_k(model, X, Y_train, loss_fn, eigenvectors_init, eigenvalues_init)

    # Also try uniform c_k for comparison
    c_k_uniform = np.ones(len(eigenvalues_init))
    c_k_uniform = c_k_uniform / np.sum(c_k_uniform)  # normalize

    # ---- Step 3: Predict S(eta) from Theorem 5 for each learning rate ----
    predictions_fitted = {}
    predictions_uniform = {}

    for lr in LRS:
        S_pred_fitted, info_f = theorem5_predict_S(eigenvalues_init, c_k_init, lr, N_STEPS)
        S_pred_uniform, info_u = theorem5_predict_S(eigenvalues_init, c_k_uniform, lr, N_STEPS)
        predictions_fitted[str(lr)] = {"S_predicted": float(S_pred_fitted), "mode_info": info_f}
        predictions_uniform[str(lr)] = {"S_predicted": float(S_pred_uniform), "mode_info": info_u}

    # ---- Step 4: Train and measure actual S(eta) for each learning rate ----
    measurements = {}
    for lr in LRS:
        # Re-initialize model with same seed for each LR
        set_seed(seed)
        if arch_type == "linear":
            model_train = LinearTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
            result = train_and_measure(model_train, X, Y_onehot, loss_fn, lr, N_STEPS)
        else:
            model_train = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
            result = train_and_measure(model_train, X, Y_train, loss_fn, lr, N_STEPS)
        measurements[str(lr)] = result
        S_meas = result["S_measured"]
        S_pred_f = predictions_fitted[str(lr)]["S_predicted"]
        ratio = S_pred_f / S_meas if S_meas > 1e-12 else float('inf')
        print(f"    lr={lr:.4f}: S_measured={S_meas:.4f}, S_predicted(fitted)={S_pred_f:.4f}, ratio={ratio:.3f}")

    # ---- Step 5: Optionally compute Hessian at mid-training ----
    eigenvalues_mid = None
    if compute_hessian_at_mid:
        set_seed(seed)
        if arch_type == "linear":
            model_mid = LinearTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
            Y_mid = Y_onehot
        else:
            model_mid = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
            Y_mid = Y_train

        opt_mid = torch.optim.SGD(model_mid.parameters(), lr=0.01)
        for step in range(N_STEPS // 2):
            opt_mid.zero_grad()
            out = model_mid(X)
            loss = loss_fn(out, Y_mid)
            loss.backward()
            opt_mid.step()

        print(f"    Computing Hessian at step {N_STEPS//2}...")
        H_mid = full_hessian(model_mid, X, Y_mid, loss_fn)
        eigenvalues_mid = np.linalg.eigh(H_mid.numpy())[0]
        print(f"    Mid-training eigenvalue range: [{eigenvalues_mid[0]:.6f}, {eigenvalues_mid[-1]:.6f}]")

    return {
        "arch_type": arch_type,
        "seed": seed,
        "n_params": n_params,
        "hessian_time": t_hessian,
        "eigenvalues_init": eigenvalues_init.tolist(),
        "eigenvalues_mid": eigenvalues_mid.tolist() if eigenvalues_mid is not None else None,
        "c_k_init_stats": {
            "mean": float(np.mean(c_k_init)),
            "std": float(np.std(c_k_init)),
            "max": float(np.max(c_k_init)),
            "n_nonzero": int(np.sum(c_k_init > 1e-12)),
        },
        "predictions_fitted": predictions_fitted,
        "predictions_uniform": predictions_uniform,
        "measurements": measurements,
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E8: Spectral Prediction of S(eta) — Testing Theorem 5 Universality")
    print("=" * 70)
    print(f"Architecture: 2-layer, hidden={HIDDEN}, no bias")
    print(f"Data: Gaussian mixture (n={N_TRAIN}, d={INPUT_DIM}, K={OUTPUT_DIM})")
    print(f"Learning rates: {LRS}")
    print(f"Training steps: {N_STEPS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]  # 3 seeds for speed (Hessian is expensive)

    # ---- Run for both architectures ----
    all_results = {"linear": [], "relu": []}

    for arch in ["linear", "relu"]:
        print(f"\n{'='*50}")
        print(f"  Architecture: {arch.upper()}")
        print(f"{'='*50}")
        for seed in USE_SEEDS:
            result = run_spectral_prediction(arch, seed, compute_hessian_at_mid=True)
            all_results[arch].append(result)

    # ---- Aggregate Results ----
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")

    summary = {}
    for arch in ["linear", "relu"]:
        print(f"\n--- {arch.upper()} ---")
        S_measured_avg = {}
        S_predicted_avg = {}

        for lr in LRS:
            S_m_list = []
            S_p_list = []
            for r in all_results[arch]:
                m = r["measurements"].get(str(lr))
                p = r["predictions_fitted"].get(str(lr))
                if m and m["converged"]:
                    S_m_list.append(m["S_measured"])
                    S_p_list.append(p["S_predicted"])

            if S_m_list:
                S_m_mean = np.mean(S_m_list)
                S_p_mean = np.mean(S_p_list)
                ratio = S_p_mean / S_m_mean if S_m_mean > 1e-12 else float('inf')
                rel_err = abs(S_p_mean - S_m_mean) / S_m_mean if S_m_mean > 1e-12 else float('inf')
                S_measured_avg[str(lr)] = float(S_m_mean)
                S_predicted_avg[str(lr)] = float(S_p_mean)
                print(f"  lr={lr:.4f}: S_meas={S_m_mean:>10.4f}, S_pred={S_p_mean:>10.4f}, "
                      f"ratio={ratio:.3f}, rel_error={rel_err:.1%}")

        # Power-law fit on measured S(eta)
        valid_lrs = [float(lr) for lr in LRS if str(lr) in S_measured_avg and S_measured_avg[str(lr)] > 1e-12]
        valid_S = [S_measured_avg[str(lr)] for lr in valid_lrs]
        if len(valid_lrs) >= 3:
            log_lr = np.log(valid_lrs)
            log_S = np.log(valid_S)
            beta, intercept = np.polyfit(log_lr, log_S, 1)
            alpha = 2 + beta
            ss_res = np.sum((log_S - (beta * log_lr + intercept)) ** 2)
            ss_tot = np.sum((log_S - np.mean(log_S)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  S(eta) ~ eta^{beta:.3f} => alpha = {alpha:.3f} (R^2 = {r2:.4f})")
        else:
            alpha, r2, beta = None, None, None

        # Prediction quality: R^2 between log(S_pred) and log(S_meas)
        if len(valid_lrs) >= 3:
            valid_S_pred = [S_predicted_avg[str(lr)] for lr in valid_lrs if S_predicted_avg.get(str(lr), 0) > 0]
            valid_S_meas_for_corr = [S_measured_avg[str(lr)] for lr in valid_lrs if S_predicted_avg.get(str(lr), 0) > 0]
            if len(valid_S_pred) >= 3:
                log_pred = np.log(np.array(valid_S_pred) + 1e-20)
                log_meas = np.log(np.array(valid_S_meas_for_corr) + 1e-20)
                corr = np.corrcoef(log_pred, log_meas)[0, 1]
                print(f"  Prediction R (log-log correlation): {corr:.4f}")
            else:
                corr = None
        else:
            corr = None

        summary[arch] = {
            "S_measured_avg": S_measured_avg,
            "S_predicted_avg": S_predicted_avg,
            "alpha": float(alpha) if alpha is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "beta": float(beta) if beta is not None else None,
            "prediction_correlation": float(corr) if corr is not None else None,
        }

    # ---- Eigenvalue comparison ----
    print(f"\n--- EIGENVALUE COMPARISON ---")
    for arch in ["linear", "relu"]:
        evals = all_results[arch][0]["eigenvalues_init"]
        evals = np.array(evals)
        n_pos = np.sum(evals > 1e-8)
        n_neg = np.sum(evals < -1e-8)
        n_zero = np.sum(np.abs(evals) <= 1e-8)
        print(f"  {arch}: positive={n_pos}, negative={n_neg}, zero={n_zero}, "
              f"max={evals[-1]:.6f}, median_pos={np.median(evals[evals>1e-8]):.6f}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # ---- Save ----
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 5,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "summary": summary,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ---- Figure 12: Spectral Prediction ----
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {"linear": "darkblue", "relu": "darkred"}
    markers = {"linear": "o", "relu": "s"}

    # Panel (a): Predicted vs Measured S(eta)
    ax = axes[0]
    for arch in ["linear", "relu"]:
        s = summary[arch]
        valid_lrs = sorted([float(k) for k in s["S_measured_avg"].keys()])
        S_m = [s["S_measured_avg"][str(lr)] for lr in valid_lrs]
        S_p = [s["S_predicted_avg"][str(lr)] for lr in valid_lrs]

        ax.loglog(valid_lrs, S_m, f'{markers[arch]}-', color=colors[arch],
                  markersize=7, label=f'{arch} measured', linewidth=1.5)
        ax.loglog(valid_lrs, S_p, f'{markers[arch]}--', color=colors[arch],
                  markersize=5, alpha=0.6, label=f'{arch} predicted (Thm 5)')

    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('$S(\\eta)$')
    ax.set_title('(a) Theorem 5 prediction vs measurement')
    ax.legend(fontsize=8)

    # Panel (b): Hessian eigenvalue density comparison
    ax = axes[1]
    for arch in ["linear", "relu"]:
        evals = np.array(all_results[arch][0]["eigenvalues_init"])
        evals_pos = evals[evals > 1e-8]
        if len(evals_pos) > 0:
            ax.hist(np.log10(evals_pos), bins=40, alpha=0.5, color=colors[arch],
                    label=f'{arch} (n={len(evals_pos)} positive)', density=True)
    ax.set_xlabel('$\\log_{10}(\\lambda_k)$')
    ax.set_ylabel('Density')
    ax.set_title('(b) Hessian eigenvalue spectrum at initialization')
    ax.legend(fontsize=9)

    # Panel (c): Prediction error vs eta
    ax = axes[2]
    for arch in ["linear", "relu"]:
        s = summary[arch]
        valid_lrs = sorted([float(k) for k in s["S_measured_avg"].keys()])
        errors = []
        lrs_for_err = []
        for lr in valid_lrs:
            S_m = s["S_measured_avg"][str(lr)]
            S_p = s["S_predicted_avg"][str(lr)]
            if S_m > 1e-12:
                errors.append(abs(S_p - S_m) / S_m)
                lrs_for_err.append(lr)
        if lrs_for_err:
            ax.semilogx(lrs_for_err, errors, f'{markers[arch]}-', color=colors[arch],
                        markersize=7, label=f'{arch}', linewidth=1.5)

    ax.axhline(y=0.2, color='green', ls='--', lw=1, alpha=0.6, label='20% threshold (Gold)')
    ax.axhline(y=0.5, color='orange', ls='--', lw=1, alpha=0.6, label='50% threshold (Silver+)')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Relative prediction error')
    ax.set_title('(c) Spectral prediction accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.0)

    fig.suptitle('E8: Testing Theorem 5 Universality — Does the Spectral Formula Work for ReLU?',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig12_spectral_prediction", FIGURE_DIR)
    plt.close()

    # ---- Eigenvalue evolution figure ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for idx, arch in enumerate(["linear", "relu"]):
        ax = axes2[idx]
        r = all_results[arch][0]
        evals_init = np.array(r["eigenvalues_init"])
        ax.plot(range(len(evals_init)), evals_init, 'b-', alpha=0.7, label='Initialization')
        if r["eigenvalues_mid"] is not None:
            evals_mid = np.array(r["eigenvalues_mid"])
            ax.plot(range(len(evals_mid)), evals_mid, 'r-', alpha=0.7, label=f'Step {N_STEPS//2}')
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('$\\lambda_k$')
        ax.set_title(f'{arch.upper()} Hessian spectrum evolution')
        ax.legend()
    plt.tight_layout()
    save_figure(fig2, "fig12b_hessian_evolution", FIGURE_DIR)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}")
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    for arch in ["linear", "relu"]:
        s = summary[arch]
        if s["prediction_correlation"] is not None:
            print(f"  {arch}: prediction correlation R = {s['prediction_correlation']:.3f}, alpha = {s['alpha']:.3f}")
        else:
            print(f"  {arch}: insufficient data for correlation")
