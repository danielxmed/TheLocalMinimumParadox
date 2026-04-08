"""
Experiment: CE Hessian Evolution Validation (Theorem 5b)
Theory: Theory A -- Proving time-dependent spectral crossover
Prediction: (1) lambda_max(H_CE(t)) decays exponentially as softmax concentrates
            (2) Optimal checkpoint t* scales linearly with n_train
            (3) CE prediction R > 0.95 at optimal t* for all n_train values
Date: 2026-04-07 (Session 8)

Method:
  For n_train in {100, 200, 400}:
    Train 2-layer ReLU (hidden=64, no bias) with CE at lr=0.01 for 2000 steps
    At fine checkpoint grid t = {0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000}:
      1. Compute full Hessian H(t)
      2. Extract eigenvalues
      3. Compute c_k from current gradient
      4. Predict S(eta) via Theorem 5 using H(t)
      5. Record mean correct-class probability q(t)
      6. Record softmax matrix eigenvalue stats
    Measure actual S(eta) at evaluation LRs
    Fit exponential decay to lambda_max(t)
    Find optimal t* for each n_train
    Test t* ~ n_train scaling
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

EXPERIMENT_NAME = "ce_hessian_evolution"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN_VALUES = [100, 200, 400]
TRAIN_LR = 0.01
N_STEPS = 2000
CHECKPOINTS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
EVAL_LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
USE_SEEDS = SEEDS[:3]  # [42, 137, 256]


# ============================================================================
# Theorem 5: Spectral Crossover Formula
# ============================================================================

def theorem5_predict_S(eigenvalues, c_k, eta, T):
    """Predict S(eta) from Hessian eigenvalues using Theorem 5."""
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


def compute_c_k(model, X, Y, loss_fn, eigenvectors, eigenvalues):
    """Compute c_k from current gradient projected onto Hessian eigenmodes."""
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


def compute_softmax_stats(model, X, Y):
    """Compute softmax probability statistics for the current model."""
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)

        # Correct-class probabilities
        correct_probs = probs[torch.arange(len(Y)), Y]
        mean_correct = correct_probs.mean().item()
        min_correct = correct_probs.min().item()
        max_correct = correct_probs.max().item()

        # Softmax matrix S_i = diag(p_i) - p_i p_i^T eigenvalue stats
        # For each sample, max eigenvalue of S_i is bounded by q_i(1-q_i)
        q_vals = correct_probs.numpy()
        max_eig_bound = np.mean(q_vals * (1 - q_vals))

        # Accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == Y).float().mean().item()

    return {
        "mean_correct_prob": float(mean_correct),
        "min_correct_prob": float(min_correct),
        "max_correct_prob": float(max_correct),
        "softmax_eig_bound": float(max_eig_bound),
        "accuracy": float(accuracy),
    }


# ============================================================================
# Training and Measurement
# ============================================================================

def train_and_measure_S(n_train, lr_eval, seed):
    """Train a fresh model at lr_eval and measure actual S(eta)."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=n_train, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_eval)
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    C_init = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2

    grad_imbalances = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        gn1 = weight_params[0][1].grad.data.norm().item() ** 2
        gn2 = weight_params[1][1].grad.data.norm().item() ** 2
        grad_imbalances.append(gn2 - gn1)
        optimizer.step()

        if np.isnan(loss.item()):
            break

    C_final = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2
    actual_drift = abs(C_final - C_init)
    S_measured = abs(sum(grad_imbalances))

    return {
        "actual_drift": float(actual_drift),
        "S_measured": float(S_measured),
        "converged": len(grad_imbalances) == N_STEPS,
    }


def run_hessian_evolution(n_train, seed):
    """Train at TRAIN_LR with CE, compute Hessian at checkpoints."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=n_train, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    loss_fn = nn.CrossEntropyLoss()
    n_params = count_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=TRAIN_LR)

    checkpoint_data = {}
    losses = []

    for step in range(N_STEPS + 1):
        if step in CHECKPOINTS:
            print(f"      [n={n_train}] Checkpoint t={step}: computing Hessian...")
            t0 = time.time()

            model_snapshot = copy.deepcopy(model)

            # Softmax probability stats
            sm_stats = compute_softmax_stats(model_snapshot, X, Y)

            # Compute Hessian
            H = full_hessian(model_snapshot, X, Y, loss_fn)
            H_np = H.numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            t_hess = time.time() - t0

            # Compute c_k
            c_k = compute_c_k(model_snapshot, X, Y, loss_fn, eigenvectors, eigenvalues)

            # Predict S(eta) for each eval LR
            predictions = {}
            for lr_eval in EVAL_LRS:
                S_pred = theorem5_predict_S(eigenvalues, c_k, lr_eval, N_STEPS)
                predictions[str(lr_eval)] = float(S_pred)

            # Eigenvalue statistics
            evals_pos = eigenvalues[eigenvalues > 1e-8]
            n_pos = len(evals_pos)

            checkpoint_data[str(step)] = {
                "step": step,
                "hessian_time": t_hess,
                "eigenvalue_stats": {
                    "n_positive": int(n_pos),
                    "max": float(eigenvalues[-1]),
                    "min": float(eigenvalues[0]),
                    "effective_rank": float(np.sum(evals_pos) / evals_pos[-1]) if n_pos > 0 and evals_pos[-1] > 0 else 0.0,
                    "trace_positive": float(np.sum(evals_pos)),
                    "spectral_norm": float(np.max(np.abs(eigenvalues))),
                },
                "softmax_stats": sm_stats,
                "predictions": predictions,
                "loss_at_checkpoint": float(losses[-1]) if losses else float('nan'),
            }
            print(f"        Hessian in {t_hess:.1f}s, max_eig={eigenvalues[-1]:.4f}, "
                  f"q_mean={sm_stats['mean_correct_prob']:.3f}, "
                  f"eff_rank={checkpoint_data[str(step)]['eigenvalue_stats']['effective_rank']:.1f}")

            del model_snapshot, H, H_np

        if step < N_STEPS:
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, Y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if np.isnan(loss.item()):
                print(f"      NaN at step {step}, stopping")
                break

    return checkpoint_data


def fit_exponential_decay(x, y):
    """Fit y = a * exp(-b * x) + c using log-linear fit on y - min(y)."""
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)

    # Offset to handle the asymptotic value
    y_min = np.min(y) * 0.9  # small margin
    y_shifted = y - y_min
    y_shifted = np.maximum(y_shifted, 1e-15)

    # Log-linear fit
    log_y = np.log(y_shifted)
    if len(x) >= 2:
        coeffs = np.polyfit(x, log_y, 1)
        b = -coeffs[0]  # decay rate
        a = np.exp(coeffs[1])
        c = y_min

        # Predicted values
        y_pred = a * np.exp(-b * x) + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {"a": float(a), "b": float(b), "c": float(c), "r2": float(r2)}
    return {"a": 0, "b": 0, "c": 0, "r2": 0}


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E18: CE Hessian Evolution — Theorem 5b Validation")
    print("=" * 70)
    print(f"Architecture: 2-layer ReLU, hidden={HIDDEN}, no bias")
    print(f"n_train values: {N_TRAIN_VALUES}")
    print(f"Training LR: {TRAIN_LR}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Seeds: {USE_SEEDS}")
    print()

    t_start = time.time()
    all_results = {}

    for n_train in N_TRAIN_VALUES:
        print(f"\n{'='*60}")
        print(f"  n_train = {n_train}")
        print(f"{'='*60}")

        n_results = {}

        # Step 1: Hessian at checkpoints
        for seed in USE_SEEDS:
            print(f"\n  --- seed={seed}: Hessian evolution ---")
            cp_data = run_hessian_evolution(n_train, seed)
            n_results[f"hessian_seed_{seed}"] = cp_data

        # Step 2: Measure actual S(eta)
        print(f"\n  --- Measuring actual S(eta) ---")
        actual_measurements = {}
        for lr_eval in EVAL_LRS:
            lr_data = []
            for seed in USE_SEEDS:
                r = train_and_measure_S(n_train, lr_eval, seed)
                lr_data.append(r)
            actual_measurements[str(lr_eval)] = {
                "mean_S": float(np.mean([r["S_measured"] for r in lr_data])),
                "mean_drift": float(np.mean([r["actual_drift"] for r in lr_data])),
                "per_seed": lr_data,
            }
            print(f"    lr={lr_eval:.4f}: S_measured={actual_measurements[str(lr_eval)]['mean_S']:.6f}")

        n_results["actual_measurements"] = actual_measurements
        all_results[str(n_train)] = n_results

    # ============================================================================
    # Analysis: Prediction Quality and Decay Fits
    # ============================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: PREDICTION QUALITY VS CHECKPOINT")
    print(f"{'='*70}")

    analysis = {}

    for n_train in N_TRAIN_VALUES:
        print(f"\n--- n_train = {n_train} ---")
        n_data = all_results[str(n_train)]
        actual_S = n_data["actual_measurements"]

        quality_per_cp = {}
        max_eigs = []
        mean_probs = []
        cp_list = []

        for cp in CHECKPOINTS:
            # Average predictions across seeds
            avg_predictions = {}
            cp_max_eigs = []
            cp_mean_probs = []

            for lr_eval in EVAL_LRS:
                preds = []
                for seed in USE_SEEDS:
                    cp_data = n_data[f"hessian_seed_{seed}"]
                    if str(cp) in cp_data:
                        preds.append(cp_data[str(cp)]["predictions"][str(lr_eval)])
                if preds:
                    avg_predictions[str(lr_eval)] = float(np.mean(preds))

            for seed in USE_SEEDS:
                cp_data = n_data[f"hessian_seed_{seed}"]
                if str(cp) in cp_data:
                    cp_max_eigs.append(cp_data[str(cp)]["eigenvalue_stats"]["max"])
                    cp_mean_probs.append(cp_data[str(cp)]["softmax_stats"]["mean_correct_prob"])

            # Correlation R
            valid_lrs = [lr for lr in EVAL_LRS
                         if str(lr) in avg_predictions
                         and str(lr) in actual_S
                         and avg_predictions[str(lr)] > 0
                         and actual_S[str(lr)]["mean_S"] > 1e-12]

            if len(valid_lrs) >= 3:
                pred_vals = [avg_predictions[str(lr)] for lr in valid_lrs]
                meas_vals = [actual_S[str(lr)]["mean_S"] for lr in valid_lrs]
                log_pred = np.log(np.array(pred_vals) + 1e-20)
                log_meas = np.log(np.array(meas_vals) + 1e-20)
                R = float(np.corrcoef(log_pred, log_meas)[0, 1])
            else:
                R = float('nan')

            quality_per_cp[str(cp)] = {
                "checkpoint": cp,
                "prediction_R": R,
                "mean_max_eig": float(np.mean(cp_max_eigs)) if cp_max_eigs else 0,
                "mean_correct_prob": float(np.mean(cp_mean_probs)) if cp_mean_probs else 0,
            }

            cp_list.append(cp)
            max_eigs.append(float(np.mean(cp_max_eigs)) if cp_max_eigs else 0)
            mean_probs.append(float(np.mean(cp_mean_probs)) if cp_mean_probs else 0)

            print(f"  t={cp:>5}: R = {R:.4f}, max_eig = {quality_per_cp[str(cp)]['mean_max_eig']:.4f}, "
                  f"q_mean = {quality_per_cp[str(cp)]['mean_correct_prob']:.3f}")

        # Find optimal checkpoint
        Rs = [quality_per_cp[str(cp)]["prediction_R"] for cp in CHECKPOINTS]
        valid_Rs = [(cp, R) for cp, R in zip(CHECKPOINTS, Rs) if not np.isnan(R)]
        if valid_Rs:
            best_cp, best_R = max(valid_Rs, key=lambda x: x[1])
            print(f"  ** OPTIMAL t* = {best_cp}, R = {best_R:.4f} **")
        else:
            best_cp, best_R = 0, 0

        # Fit exponential decay to max eigenvalue
        if len(max_eigs) >= 3:
            decay_fit = fit_exponential_decay(np.array(cp_list), np.array(max_eigs))
            print(f"  Exponential decay fit: a={decay_fit['a']:.4f}, b={decay_fit['b']:.6f}, "
                  f"c={decay_fit['c']:.4f}, R^2={decay_fit['r2']:.4f}")
        else:
            decay_fit = {"a": 0, "b": 0, "c": 0, "r2": 0}

        analysis[str(n_train)] = {
            "quality_per_checkpoint": quality_per_cp,
            "optimal_checkpoint": best_cp,
            "optimal_R": best_R,
            "decay_fit": decay_fit,
            "max_eigenvalues": max_eigs,
            "mean_correct_probs": mean_probs,
            "checkpoints": cp_list,
        }

    # ============================================================================
    # Key Test: Does t* scale with n_train?
    # ============================================================================
    print(f"\n{'='*70}")
    print("KEY TEST: Does optimal t* scale linearly with n_train?")
    print(f"{'='*70}")

    t_star_values = []
    for n_train in N_TRAIN_VALUES:
        t_star = analysis[str(n_train)]["optimal_checkpoint"]
        R_star = analysis[str(n_train)]["optimal_R"]
        t_star_values.append(t_star)
        print(f"  n_train={n_train}: t* = {t_star}, R* = {R_star:.4f}")

    # Fit t* vs n_train
    if len(t_star_values) >= 2:
        n_arr = np.array(N_TRAIN_VALUES, dtype=float)
        t_arr = np.array(t_star_values, dtype=float)
        if np.std(t_arr) > 0:
            slope, intercept = np.polyfit(n_arr, t_arr, 1)
            R_linear = np.corrcoef(n_arr, t_arr)[0, 1]
            print(f"\n  Linear fit: t* = {slope:.3f} * n + {intercept:.1f}")
            print(f"  Correlation R = {R_linear:.4f}")
            print(f"  Prediction: t* ~ {slope:.3f} * n (Theory predicts t* ~ n/(K*||J||^2))")
        else:
            slope, intercept, R_linear = 0, t_arr[0], 0
            print(f"\n  All t* values are identical: {t_star_values[0]}")
    else:
        slope, intercept, R_linear = 0, 0, 0

    # ============================================================================
    # Decay rate comparison across n_train
    # ============================================================================
    print(f"\n{'='*70}")
    print("DECAY RATE COMPARISON")
    print(f"{'='*70}")
    for n_train in N_TRAIN_VALUES:
        df = analysis[str(n_train)]["decay_fit"]
        print(f"  n_train={n_train}: decay_rate b = {df['b']:.6f}, "
              f"half-life = {np.log(2)/df['b']:.1f} steps" if df['b'] > 0 else
              f"  n_train={n_train}: decay_rate b = {df['b']:.6f}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E18",
        "version": 1,
        "date": "2026-04-07",
        "session": 8,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "hidden": HIDDEN,
        "train_lr": TRAIN_LR,
        "n_steps": N_STEPS,
        "checkpoints": CHECKPOINTS,
        "eval_lrs": EVAL_LRS,
        "n_train_values": N_TRAIN_VALUES,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "analysis": analysis,
        "t_star_scaling": {
            "n_train_values": N_TRAIN_VALUES,
            "t_star_values": t_star_values,
            "linear_slope": float(slope),
            "linear_intercept": float(intercept),
            "linear_R": float(R_linear),
        },
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 21: Theorem 5b Validation
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    n_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(N_TRAIN_VALUES)))

    # Panel (a): lambda_max(H_CE(t)) vs t for each n_train
    ax = axes[0, 0]
    for i, n_train in enumerate(N_TRAIN_VALUES):
        a_data = analysis[str(n_train)]
        cps = a_data["checkpoints"]
        max_eigs = a_data["max_eigenvalues"]
        ax.plot(cps, max_eigs, 'o-', color=n_colors[i], markersize=6, linewidth=2,
                label=f'n={n_train}')

        # Overlay exponential fit
        df = a_data["decay_fit"]
        if df["r2"] > 0.5 and df["b"] > 0:
            t_fine = np.linspace(0, max(cps), 100)
            y_fit = df["a"] * np.exp(-df["b"] * t_fine) + df["c"]
            ax.plot(t_fine, y_fit, '--', color=n_colors[i], alpha=0.6, linewidth=1)

    ax.set_xlabel('Training step $t$')
    ax.set_ylabel('$\\lambda_{\\max}(H_{CE}(t))$')
    ax.set_title('(a) CE Hessian max eigenvalue decay')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Panel (b): Optimal t* vs n_train
    ax = axes[0, 1]
    ax.plot(N_TRAIN_VALUES, t_star_values, 'ko-', markersize=10, linewidth=2)
    if abs(slope) > 0:
        n_fine = np.linspace(min(N_TRAIN_VALUES)*0.8, max(N_TRAIN_VALUES)*1.2, 50)
        ax.plot(n_fine, slope * n_fine + intercept, 'r--', linewidth=1.5,
                label=f'$t^* = {slope:.2f}n + {intercept:.0f}$')
    ax.set_xlabel('$n_{train}$')
    ax.set_ylabel('Optimal checkpoint $t^*$')
    ax.set_title(f'(b) $t^*$ vs $n_{{train}}$ (theory: $t^* \\propto n$)')
    ax.legend(fontsize=9)

    # Panel (c): Prediction R vs checkpoint for each n_train
    ax = axes[1, 0]
    for i, n_train in enumerate(N_TRAIN_VALUES):
        a_data = analysis[str(n_train)]
        qpc = a_data["quality_per_checkpoint"]
        cps = [int(cp) for cp in sorted(qpc.keys())]
        Rs = [qpc[str(cp)]["prediction_R"] for cp in cps]
        ax.plot(cps, Rs, 'o-', color=n_colors[i], markersize=6, linewidth=2,
                label=f'n={n_train} ($t^*$={a_data["optimal_checkpoint"]})')

        # Mark optimal
        ax.plot(a_data["optimal_checkpoint"], a_data["optimal_R"], '*',
                color=n_colors[i], markersize=15)

    ax.set_xlabel('Training step (checkpoint)')
    ax.set_ylabel('Prediction correlation $R$')
    ax.set_title('(c) Theorem 5 prediction quality vs Hessian checkpoint')
    ax.legend(fontsize=8)
    ax.axhline(y=0.95, color='gray', ls=':', lw=1, alpha=0.5)
    ax.annotate('R=0.95', xy=(0, 0.95), fontsize=8, color='gray')

    # Panel (d): Mean correct-class probability and Hessian compression
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for i, n_train in enumerate(N_TRAIN_VALUES):
        a_data = analysis[str(n_train)]
        cps = a_data["checkpoints"]
        probs = a_data["mean_correct_probs"]
        max_eigs = a_data["max_eigenvalues"]

        # Compression ratio: max_eig(t) / max_eig(0)
        if max_eigs[0] > 0:
            compression = [me / max_eigs[0] for me in max_eigs]
        else:
            compression = max_eigs

        l1 = ax.plot(cps, probs, 'o-', color=n_colors[i], markersize=5, linewidth=1.5,
                     label=f'n={n_train} prob')
        l2 = ax2.plot(cps, compression, 's--', color=n_colors[i], markersize=5,
                      linewidth=1.5, alpha=0.6, label=f'n={n_train} compress.')

    ax.set_xlabel('Training step')
    ax.set_ylabel('Mean correct-class probability $\\bar{q}(t)$')
    ax2.set_ylabel('Hessian compression ratio $\\lambda_{max}(t)/\\lambda_{max}(0)$')
    ax.set_title('(d) Softmax concentration drives Hessian compression')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')

    fig.suptitle('E18: Theorem 5b Validation — Time-Dependent Spectral Crossover',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig21_theorem5b_validation", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig21_theorem5b_validation.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    for n_train in N_TRAIN_VALUES:
        a = analysis[str(n_train)]
        print(f"  n={n_train}: t*={a['optimal_checkpoint']}, R*={a['optimal_R']:.4f}, "
              f"decay_R2={a['decay_fit']['r2']:.4f}")

    print(f"\n  t* scaling: slope={slope:.4f}, R={R_linear:.4f}")
    if abs(R_linear) > 0.9:
        print("  ** t* scales linearly with n_train — CONSISTENT WITH THEORY **")
    elif abs(R_linear) > 0.7:
        print("  ** Moderate linear scaling — theory partially supported **")
    else:
        print("  ** Weak or no linear scaling — theory needs refinement **")

    print(f"\n  Total experiment time: {t_total:.1f}s ({t_total/60:.1f} min)")
