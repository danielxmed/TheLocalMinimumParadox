"""
Experiment: Time-Dependent Hessian for CrossEntropy
Theory: Theory A -- Understanding why initialization Hessian fails for CE prediction
Prediction: The softmax probabilities evolve rapidly during CE training, changing the
            effective Hessian. Using a later-time or time-averaged Hessian should improve
            Theorem 5 predictions for CE (currently R=0.37-0.42).
Date: 2026-04-07 (Session 7)

Method:
  For each loss in {MSE, CE} with ReLU activation:
    Train at lr=0.01 for 2000 steps
    At checkpoints t = {0, 250, 500, 1000, 2000}:
      1. Compute full Hessian H(t)
      2. Extract eigenvalues and eigenvectors
      3. Compute c_k from current gradient
      4. Predict S(eta) via Theorem 5 using H(t)
    Also measure actual S(eta) at evaluation LRs
    Compare: does R improve for CE when using H(t > 0)?
    Test time-averaged Hessian prediction
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

EXPERIMENT_NAME = "hessian_time_evolution"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200
TRAIN_LR = 0.01
N_STEPS = 2000
CHECKPOINTS = [0, 250, 500, 1000, 2000]
EVAL_LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


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


# ============================================================================
# Training and Measurement
# ============================================================================

def train_and_measure_S(loss_type, lr_eval, seed):
    """Train a fresh model at lr_eval and measure actual S(eta)."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
        Y_train = Y_onehot
    else:
        loss_fn = nn.CrossEntropyLoss()
        Y_train = Y

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_eval)
    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    C_init = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2

    grad_imbalances = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y_train)
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


def run_hessian_evolution(loss_type, seed):
    """Train at TRAIN_LR, compute Hessian at checkpoints, predict S(eta)."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
    Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
        Y_train = Y_onehot
    else:
        loss_fn = nn.CrossEntropyLoss()
        Y_train = Y

    n_params = count_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=TRAIN_LR)

    checkpoint_data = {}
    losses = []

    for step in range(N_STEPS + 1):
        if step in CHECKPOINTS:
            print(f"      Checkpoint t={step}: computing Hessian...")
            t0 = time.time()

            # Save model state for Hessian computation
            model_snapshot = copy.deepcopy(model)

            # Compute Hessian
            H = full_hessian(model_snapshot, X, Y_train, loss_fn)
            H_np = H.numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            t_hess = time.time() - t0

            # Compute c_k
            c_k = compute_c_k(model_snapshot, X, Y_train, loss_fn, eigenvectors, eigenvalues)

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
                "eigenvalues": eigenvalues.tolist(),
                "eigenvalue_stats": {
                    "n_positive": int(n_pos),
                    "max": float(eigenvalues[-1]),
                    "min": float(eigenvalues[0]),
                    "effective_rank": float(np.sum(evals_pos) / evals_pos[-1]) if n_pos > 0 and evals_pos[-1] > 0 else 0.0,
                    "median_positive": float(np.median(evals_pos)) if n_pos > 0 else 0.0,
                    "spectral_norm": float(np.max(np.abs(eigenvalues))),
                },
                "predictions": predictions,
                "loss_at_checkpoint": float(losses[-1]) if losses else float('nan'),
            }
            print(f"        Hessian in {t_hess:.1f}s, max_eig={eigenvalues[-1]:.4f}, "
                  f"eff_rank={checkpoint_data[str(step)]['eigenvalue_stats']['effective_rank']:.1f}")

            del model_snapshot, H, H_np

        if step < N_STEPS:
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, Y_train)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if np.isnan(loss.item()):
                print(f"      NaN at step {step}, stopping")
                break

    return checkpoint_data


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E16: Time-Dependent Hessian for CrossEntropy")
    print("=" * 70)
    print(f"Architecture: 2-layer ReLU, hidden={HIDDEN}, no bias")
    print(f"Training LR: {TRAIN_LR}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Eval LRs for S(eta): {EVAL_LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}

    for loss_type in ["mse", "ce"]:
        print(f"\n{'='*60}")
        print(f"  Loss: {loss_type.upper()}")
        print(f"{'='*60}")

        loss_results = {}

        # Step 1: Compute Hessian at checkpoints
        for seed in USE_SEEDS:
            print(f"\n  --- seed={seed}: Hessian evolution ---")
            cp_data = run_hessian_evolution(loss_type, seed)
            loss_results[f"hessian_seed_{seed}"] = cp_data

        # Step 2: Measure actual S(eta) for each eval LR
        print(f"\n  --- Measuring actual S(eta) ---")
        actual_measurements = {}
        for lr_eval in EVAL_LRS:
            lr_data = []
            for seed in USE_SEEDS:
                r = train_and_measure_S(loss_type, lr_eval, seed)
                lr_data.append(r)
            actual_measurements[str(lr_eval)] = {
                "mean_S": float(np.mean([r["S_measured"] for r in lr_data])),
                "mean_drift": float(np.mean([r["actual_drift"] for r in lr_data])),
                "per_seed": lr_data,
            }
            print(f"    lr={lr_eval:.4f}: S_measured={actual_measurements[str(lr_eval)]['mean_S']:.4f}")

        loss_results["actual_measurements"] = actual_measurements
        all_results[loss_type] = loss_results

    # ============================================================================
    # Analysis: Prediction Quality at Each Checkpoint
    # ============================================================================
    print(f"\n{'='*70}")
    print("PREDICTION QUALITY VS TRAINING CHECKPOINT")
    print(f"{'='*70}")

    prediction_quality = {}

    for loss_type in ["mse", "ce"]:
        print(f"\n--- {loss_type.upper()} ---")
        actual_S = all_results[loss_type]["actual_measurements"]

        quality_per_checkpoint = {}
        all_eigenvalues_by_checkpoint = {}

        for cp in CHECKPOINTS:
            # Average predictions across seeds
            avg_predictions = {}
            for lr_eval in EVAL_LRS:
                preds = []
                for seed in USE_SEEDS:
                    cp_data = all_results[loss_type][f"hessian_seed_{seed}"]
                    if str(cp) in cp_data:
                        preds.append(cp_data[str(cp)]["predictions"][str(lr_eval)])
                if preds:
                    avg_predictions[str(lr_eval)] = float(np.mean(preds))

            # Compute correlation R between predicted and measured S(eta)
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

            # Eigenvalue stats (first seed)
            cp_data_0 = all_results[loss_type][f"hessian_seed_{USE_SEEDS[0]}"]
            ev_stats = cp_data_0[str(cp)]["eigenvalue_stats"] if str(cp) in cp_data_0 else {}

            quality_per_checkpoint[str(cp)] = {
                "checkpoint": cp,
                "prediction_R": R,
                "avg_predictions": avg_predictions,
                "eigenvalue_stats": ev_stats,
            }

            # Collect eigenvalues for plotting
            if str(cp) in cp_data_0:
                all_eigenvalues_by_checkpoint[str(cp)] = cp_data_0[str(cp)]["eigenvalues"]

            print(f"  t={cp:>5}: R = {R:.4f}, max_eig = {ev_stats.get('max', 0):.4f}, "
                  f"eff_rank = {ev_stats.get('effective_rank', 0):.1f}")

        prediction_quality[loss_type] = quality_per_checkpoint

    # Time-averaged Hessian prediction
    print(f"\n--- TIME-AVERAGED HESSIAN PREDICTION ---")
    for loss_type in ["mse", "ce"]:
        actual_S = all_results[loss_type]["actual_measurements"]

        # Average eigenvalues across checkpoints (first seed)
        all_evals = []
        for cp in CHECKPOINTS:
            cp_data = all_results[loss_type][f"hessian_seed_{USE_SEEDS[0]}"]
            if str(cp) in cp_data:
                all_evals.append(np.array(cp_data[str(cp)]["eigenvalues"]))

        if len(all_evals) >= 2:
            avg_eigenvalues = np.mean(all_evals, axis=0)
            # Use c_k from the middle checkpoint
            mid_cp = str(CHECKPOINTS[len(CHECKPOINTS)//2])
            cp_data_mid = all_results[loss_type][f"hessian_seed_{USE_SEEDS[0]}"]

            # Simple c_k: uniform
            c_k_uniform = np.ones(len(avg_eigenvalues)) / len(avg_eigenvalues)

            avg_predictions = {}
            for lr_eval in EVAL_LRS:
                S_pred = theorem5_predict_S(avg_eigenvalues, c_k_uniform, lr_eval, N_STEPS)
                avg_predictions[str(lr_eval)] = float(S_pred)

            valid_lrs = [lr for lr in EVAL_LRS
                         if str(lr) in avg_predictions
                         and str(lr) in actual_S
                         and avg_predictions[str(lr)] > 0
                         and actual_S[str(lr)]["mean_S"] > 1e-12]

            if len(valid_lrs) >= 3:
                pred_vals = [avg_predictions[str(lr)] for lr in valid_lrs]
                meas_vals = [actual_S[str(lr)]["mean_S"] for lr in valid_lrs]
                R_avg = float(np.corrcoef(np.log(np.array(pred_vals)+1e-20),
                                           np.log(np.array(meas_vals)+1e-20))[0, 1])
            else:
                R_avg = float('nan')

            print(f"  {loss_type.upper()}: time-averaged Hessian R = {R_avg:.4f}")
            prediction_quality[loss_type]["time_averaged"] = {
                "prediction_R": R_avg,
            }

    # Drift exponent fit
    print(f"\n--- DRIFT EXPONENT (alpha) ---")
    for loss_type in ["mse", "ce"]:
        actual_S = all_results[loss_type]["actual_measurements"]
        valid_lrs = [lr for lr in EVAL_LRS
                     if str(lr) in actual_S and actual_S[str(lr)]["mean_drift"] > 1e-15]
        valid_drifts = [actual_S[str(lr)]["mean_drift"] for lr in valid_lrs]

        if len(valid_lrs) >= 3:
            log_lr = np.log(valid_lrs)
            log_drift = np.log(valid_drifts)
            alpha, intercept = np.polyfit(log_lr, log_drift, 1)
            ss_res = np.sum((log_drift - (alpha * log_lr + intercept)) ** 2)
            ss_tot = np.sum((log_drift - log_drift.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  {loss_type.upper()}: alpha = {alpha:.3f} (R^2 = {r2:.4f})")
        else:
            print(f"  {loss_type.upper()}: insufficient data for alpha fit")

    t_total = time.time() - t_start
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
        "hidden": HIDDEN,
        "train_lr": TRAIN_LR,
        "n_steps": N_STEPS,
        "checkpoints": CHECKPOINTS,
        "eval_lrs": EVAL_LRS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "prediction_quality": prediction_quality,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 19: Hessian Time Evolution
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    cp_colors = plt.cm.coolwarm(np.linspace(0, 1, len(CHECKPOINTS)))

    # Panel (a): Eigenvalue CDFs at each checkpoint — CE
    ax = axes[0, 0]
    cp_data_ce = all_results["ce"][f"hessian_seed_{USE_SEEDS[0]}"]
    for i, cp in enumerate(CHECKPOINTS):
        if str(cp) in cp_data_ce:
            evals = np.array(cp_data_ce[str(cp)]["eigenvalues"])
            evals_pos = evals[evals > 1e-8]
            if len(evals_pos) > 0:
                sorted_evals = np.sort(evals_pos)
                cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals)
                ax.semilogx(sorted_evals, cdf, color=cp_colors[i], linewidth=1.5,
                            label=f't={cp}')
    ax.set_xlabel('Eigenvalue $\\lambda$')
    ax.set_ylabel('CDF')
    ax.set_title('(a) CE: Hessian eigenvalue evolution')
    ax.legend(fontsize=8)

    # Panel (b): Eigenvalue CDFs at each checkpoint — MSE
    ax = axes[0, 1]
    cp_data_mse = all_results["mse"][f"hessian_seed_{USE_SEEDS[0]}"]
    for i, cp in enumerate(CHECKPOINTS):
        if str(cp) in cp_data_mse:
            evals = np.array(cp_data_mse[str(cp)]["eigenvalues"])
            evals_pos = evals[evals > 1e-8]
            if len(evals_pos) > 0:
                sorted_evals = np.sort(evals_pos)
                cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals)
                ax.semilogx(sorted_evals, cdf, color=cp_colors[i], linewidth=1.5,
                            label=f't={cp}')
    ax.set_xlabel('Eigenvalue $\\lambda$')
    ax.set_ylabel('CDF')
    ax.set_title('(b) MSE: Hessian eigenvalue evolution')
    ax.legend(fontsize=8)

    # Panel (c): Prediction R vs checkpoint
    ax = axes[1, 0]
    for loss_type, color, marker in [("mse", "darkblue", "o"), ("ce", "darkred", "s")]:
        pq = prediction_quality[loss_type]
        cps = [int(cp) for cp in sorted(pq.keys()) if cp != "time_averaged" and not np.isnan(pq[cp]["prediction_R"])]
        Rs = [pq[str(cp)]["prediction_R"] for cp in cps]
        ax.plot(cps, Rs, f'{marker}-', color=color, markersize=8, linewidth=2,
                label=f'{loss_type.upper()}')

        # Time-averaged
        if "time_averaged" in pq:
            R_avg = pq["time_averaged"]["prediction_R"]
            if not np.isnan(R_avg):
                ax.axhline(y=R_avg, color=color, ls='--', lw=1, alpha=0.6)
                ax.annotate(f'{loss_type.upper()} time-avg: {R_avg:.3f}',
                            xy=(CHECKPOINTS[-1]*0.6, R_avg),
                            fontsize=8, color=color)

    ax.set_xlabel('Training step (checkpoint)')
    ax.set_ylabel('Prediction correlation $R$')
    ax.set_title('(c) Theorem 5 prediction quality vs training time')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', ls=':', lw=0.5)

    # Panel (d): Effective rank and max eigenvalue evolution
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for loss_type, color in [("mse", "darkblue"), ("ce", "darkred")]:
        pq = prediction_quality[loss_type]
        cps = [int(cp) for cp in sorted(pq.keys()) if cp != "time_averaged"]
        eff_ranks = [pq[str(cp)]["eigenvalue_stats"].get("effective_rank", 0) for cp in cps]
        max_eigs = [pq[str(cp)]["eigenvalue_stats"].get("max", 0) for cp in cps]

        l1 = ax.plot(cps, eff_ranks, 'o-', color=color, markersize=6, linewidth=1.5,
                     label=f'{loss_type.upper()} eff. rank')
        l2 = ax2.plot(cps, max_eigs, 's--', color=color, markersize=6, linewidth=1.5,
                      alpha=0.6, label=f'{loss_type.upper()} max $\\lambda$')

    ax.set_xlabel('Training step')
    ax.set_ylabel('Effective rank', color='black')
    ax2.set_ylabel('Max eigenvalue', color='gray')
    ax.set_title('(d) Spectral evolution during training')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

    fig.suptitle('E16: Time-Dependent Hessian — Does Later Hessian Fix CE Prediction?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig19_hessian_time_evolution", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig19_hessian_time_evolution.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    for loss_type in ["mse", "ce"]:
        pq = prediction_quality[loss_type]
        R_init = pq.get("0", {}).get("prediction_R", float('nan'))
        R_final = pq.get(str(CHECKPOINTS[-1]), {}).get("prediction_R", float('nan'))
        R_avg = pq.get("time_averaged", {}).get("prediction_R", float('nan'))
        print(f"  {loss_type.upper()}: R(t=0) = {R_init:.4f}, R(t={CHECKPOINTS[-1]}) = {R_final:.4f}, "
              f"R(time-avg) = {R_avg:.4f}")
        if not np.isnan(R_init) and not np.isnan(R_final):
            if R_final > R_init + 0.1:
                print(f"    ** LATER HESSIAN IMPROVES PREDICTION by {R_final - R_init:.3f} **")
            elif R_final < R_init - 0.1:
                print(f"    ** LATER HESSIAN WORSENS PREDICTION by {R_init - R_final:.3f} **")
            else:
                print(f"    ** NO SIGNIFICANT CHANGE in prediction quality **")
