"""
Experiment: Edge of Stability in Deep Narrow Networks with MSE Loss
Theory: Edge of Stability (Cohen et al., 2021)
Prediction: With a 4-layer MLP (hidden=32) trained on MNIST using MSE loss and
            full-batch GD, lambda_max should exhibit progressive sharpening and
            approach the stability threshold 2/eta for large learning rates.
            Conservation quantities C_l = ||W_{l+1}||^2 - ||W_l||^2 should show
            increasing drift as lambda_max nears the threshold.

Key design choices (addressing why prior experiments failed):
  - Deep network (4 layers) -- deeper = more non-convexity = slower convergence
  - Narrow hidden layers (32) -- fewer parameters = less overparameterization
  - MSE loss (not cross-entropy) -- MSE produces sharper Hessians per Cohen et al.
  - Full-batch GD -- eliminates stochastic noise, isolates curvature dynamics
  - Multiple learning rates -- maps out the transition to EoS regime
  - 5000 steps -- long enough for progressive sharpening to develop

Date: 2026-04-07
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, DEVICE, make_model, make_dataset,
    count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info, top_hessian_eigenvalue,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "eos_deep"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", EXPERIMENT_NAME,
)
FIGURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

N_TRAIN = 2000
N_TEST = 200
DEPTH = 4
HIDDEN = 32
INPUT_DIM = 784
OUTPUT_DIM = 10

LEARNING_RATES = [0.1, 0.5, 1.0, 2.0]
N_STEPS = 5000
LOG_EVERY = 25          # compute lambda_max every 25 steps
HESSIAN_ITERS = 30      # power-iteration steps for top eigenvalue
SEEDS = [42, 137, 256]

# ---------------------------------------------------------------------------
# Helper: Conservation quantities
# ---------------------------------------------------------------------------

def get_weight_norms_sq(model):
    """Return list of squared Frobenius norms of each weight matrix."""
    norms = []
    for name, p in model.named_parameters():
        if "weight" in name:
            norms.append((p.data ** 2).sum().item())
    return norms


def compute_conservation_quantities(model):
    """Compute C_l = ||W_{l+1}||^2 - ||W_l||^2 for consecutive layer pairs."""
    norms = get_weight_norms_sq(model)
    return [norms[l + 1] - norms[l] for l in range(len(norms) - 1)]


# ---------------------------------------------------------------------------
# MSE loss function compatible with top_hessian_eigenvalue
# ---------------------------------------------------------------------------

class MSELossFn(nn.Module):
    """MSE loss that takes (predictions, one-hot targets)."""
    def forward(self, output, target):
        return F.mse_loss(output, target)

mse_loss_fn = MSELossFn()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(seed, lr):
    """Train one model with given seed and learning rate, recording diagnostics."""
    set_seed(seed)

    # Build model
    model = make_model(
        "mlp_deep_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN, depth=DEPTH,
    )

    # Load dataset
    X_train, Y_train_int, X_test, Y_test_int = make_dataset(
        "mnist", n_train=N_TRAIN, n_test=N_TEST,
    )

    # One-hot encode targets for MSE loss
    Y_train = F.one_hot(Y_train_int, OUTPUT_DIM).float()
    Y_test = F.one_hot(Y_test_int, OUTPUT_DIM).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Storage
    steps_log = []
    losses_log = []
    lmax_log = []
    train_acc_log = []
    conservation_log = []   # list of lists
    weight_norms_log = []   # list of lists

    stability_threshold = 2.0 / lr
    diverged = False

    for step in range(1, N_STEPS + 1):
        # ----- Training step (full-batch GD) -----
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = F.mse_loss(out, Y_train)

        # Check for divergence
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
            print(f"      [DIVERGED at step {step}, loss={loss.item():.2e}]")
            diverged = True
            break

        loss.backward()
        optimizer.step()

        # ----- Logging every LOG_EVERY steps -----
        if step % LOG_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                curr_loss = F.mse_loss(model(X_train), Y_train).item()
                pred = model(X_train).argmax(1)
                acc = (pred == Y_train_int).float().mean().item()

            # Top Hessian eigenvalue via power iteration
            lmax = top_hessian_eigenvalue(
                model, X_train, Y_train, mse_loss_fn, n_iter=HESSIAN_ITERS,
            )

            # Conservation quantities
            C_vals = compute_conservation_quantities(model)
            w_norms = get_weight_norms_sq(model)

            steps_log.append(step)
            losses_log.append(curr_loss)
            lmax_log.append(lmax)
            train_acc_log.append(acc)
            conservation_log.append(C_vals)
            weight_norms_log.append(w_norms)

            if step % 500 == 0 or step == 1:
                C_str = ", ".join(f"{c:.4f}" for c in C_vals)
                print(
                    f"      step {step:>5} | loss={curr_loss:.6f} | "
                    f"lmax={lmax:.2f} | acc={acc:.3f} | "
                    f"2/eta={stability_threshold:.1f} | C=[{C_str}]"
                )

    return {
        "seed": seed,
        "lr": lr,
        "steps": steps_log,
        "losses": losses_log,
        "lambda_max": lmax_log,
        "train_acc": train_acc_log,
        "conservation": conservation_log,
        "weight_norms": weight_norms_log,
        "diverged": diverged,
        "n_params": count_parameters(model),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eos_panels(all_results):
    """4-panel figure: lambda_max vs step for each learning rate."""
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, lr in enumerate(LEARNING_RATES):
        ax = axes[idx]
        threshold = 2.0 / lr
        lr_results = [r for r in all_results if r["lr"] == lr]

        for j, r in enumerate(lr_results):
            if r["diverged"] and len(r["steps"]) == 0:
                continue
            ax.plot(
                r["steps"], r["lambda_max"],
                color=colors[j], alpha=0.7, linewidth=1.0,
                label=f"Seed {r['seed']}",
            )

        ax.axhline(
            y=threshold, color="red", linestyle="--", linewidth=2.0,
            alpha=0.8, label=f"$2/\\eta = {threshold:.1f}$",
        )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("$\\lambda_{{\\max}}(\\nabla^2 L)$")
        ax.set_title(f"$\\eta = {lr}$")
        ax.legend(fontsize=9, loc="best")

        # Set y-axis upper limit to 2x threshold for visibility
        if lr_results and any(len(r["steps"]) > 0 for r in lr_results):
            all_lmax = []
            for r in lr_results:
                all_lmax.extend(r["lambda_max"])
            if all_lmax:
                y_max = min(max(all_lmax) * 1.3, threshold * 2.5)
                ax.set_ylim(bottom=0, top=max(y_max, threshold * 1.3))

    fig.suptitle(
        "Edge of Stability: Deep Narrow MLP + MSE Loss on MNIST\n"
        f"(depth={DEPTH}, hidden={HIDDEN}, n={N_TRAIN}, full-batch GD, "
        f"{N_STEPS} steps)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig4c_eos_deep", FIGURE_DIR)
    plt.close()
    print(f"  EoS figure saved to {FIGURE_DIR}/fig4c_eos_deep.png")


def plot_conservation_trajectories(all_results, target_lr=1.0):
    """Plot conservation quantity trajectories for a specific learning rate."""
    setup_publication_style()

    lr_results = [r for r in all_results if r["lr"] == target_lr and not r["diverged"]]
    if not lr_results:
        # Fall back to first non-diverged lr
        for lr in LEARNING_RATES:
            lr_results = [r for r in all_results if r["lr"] == lr and not r["diverged"]]
            if lr_results:
                target_lr = lr
                break
    if not lr_results:
        print("  No non-diverged runs found for conservation plot. Skipping.")
        return

    n_conservation = len(lr_results[0]["conservation"][0]) if lr_results[0]["conservation"] else 0
    if n_conservation == 0:
        print("  No conservation data. Skipping.")
        return

    fig, axes = plt.subplots(1, n_conservation + 1, figsize=(5 * (n_conservation + 1), 4.5))
    if n_conservation + 1 == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Plot each conservation quantity
    for c_idx in range(n_conservation):
        ax = axes[c_idx]
        for j, r in enumerate(lr_results):
            C_trajectory = [c[c_idx] for c in r["conservation"]]
            ax.plot(
                r["steps"], C_trajectory,
                color=colors[j % len(colors)], alpha=0.7, linewidth=1.0,
                label=f"Seed {r['seed']}",
            )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"$C_{{{c_idx+1}}}$")
        ax.set_title(f"$C_{{{c_idx+1}}} = ||W_{{{c_idx+2}}}||^2 - ||W_{{{c_idx+1}}}||^2$")
        ax.legend(fontsize=8)

    # Plot total conservation drift
    ax = axes[n_conservation]
    for j, r in enumerate(lr_results):
        total_drift = []
        C_init = r["conservation"][0] if r["conservation"] else [0] * n_conservation
        for c_vals in r["conservation"]:
            drift = sum(abs(c_vals[k] - C_init[k]) for k in range(n_conservation))
            total_drift.append(drift)
        ax.plot(
            r["steps"], total_drift,
            color=colors[j % len(colors)], alpha=0.7, linewidth=1.0,
            label=f"Seed {r['seed']}",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Conservation Drift")
    ax.set_title(f"$\\sum_l |C_l(t) - C_l(0)|$")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Conservation Quantity Trajectories ($\\eta = {target_lr}$)\n"
        f"(depth={DEPTH}, hidden={HIDDEN}, MSE loss, no bias)",
        fontsize=13, y=1.04,
    )
    plt.tight_layout()
    save_figure(fig, "fig4d_conservation_deep", FIGURE_DIR)
    plt.close()
    print(f"  Conservation figure saved to {FIGURE_DIR}/fig4d_conservation_deep.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("EXPERIMENT: Edge of Stability in Deep Narrow Networks + MSE Loss")
    print("=" * 72)
    print(f"  Architecture: mlp_deep_nobias (depth={DEPTH}, hidden={HIDDEN})")
    print(f"  Dataset: MNIST (n_train={N_TRAIN}, n_test={N_TEST})")
    print(f"  Loss: MSE (one-hot targets)")
    print(f"  Optimizer: SGD (full-batch GD)")
    print(f"  Learning rates: {LEARNING_RATES}")
    print(f"  Steps: {N_STEPS}, log every {LOG_EVERY}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")

    # Quick model info
    set_seed(42)
    tmp_model = make_model("mlp_deep_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN, depth=DEPTH)
    n_params = count_parameters(tmp_model)
    n_layers = sum(1 for name, _ in tmp_model.named_parameters() if "weight" in name)
    print(f"  Parameters: {n_params}")
    print(f"  Weight matrices: {n_layers}")
    print(f"  Conservation quantities: {n_layers - 1}")
    del tmp_model

    for lr in LEARNING_RATES:
        print(f"  2/eta (lr={lr}): {2.0/lr:.1f}")
    print("=" * 72)

    # ----- Run all experiments -----
    all_results = []
    t_start = time.time()

    for lr in LEARNING_RATES:
        print(f"\n{'='*50}")
        print(f"  Learning rate: {lr} (2/eta = {2.0/lr:.1f})")
        print(f"{'='*50}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            r = run_single(seed, lr)
            all_results.append(r)

            if r["diverged"]:
                n_steps_completed = len(r["steps"])
                print(f"    DIVERGED after {n_steps_completed} logged steps")
            else:
                final_loss = r["losses"][-1] if r["losses"] else float("nan")
                final_lmax = r["lambda_max"][-1] if r["lambda_max"] else float("nan")
                final_acc = r["train_acc"][-1] if r["train_acc"] else float("nan")
                print(
                    f"    Final: loss={final_loss:.6f}, "
                    f"lmax={final_lmax:.2f}, acc={final_acc:.3f}"
                )

    t_total = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"Total training time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # ----- Summary -----
    print(f"\n{'='*72}")
    print("EDGE OF STABILITY SUMMARY")
    print(f"{'='*72}")
    print(f"{'LR':>6} | {'2/eta':>8} | {'max lmax':>10} | {'final lmax':>11} | "
          f"{'reached 80%':>12} | {'conservation drift':>18}")
    print("-" * 80)

    for lr in LEARNING_RATES:
        threshold = 2.0 / lr
        lr_results = [r for r in all_results if r["lr"] == lr]

        for r in lr_results:
            if r["diverged"] and len(r["lambda_max"]) == 0:
                print(f"{lr:>6.1f} | {threshold:>8.1f} | {'DIVERGED':>10} | "
                      f"{'---':>11} | {'---':>12} | {'---':>18}")
                continue

            max_lmax = max(r["lambda_max"]) if r["lambda_max"] else 0
            final_lmax = r["lambda_max"][-1] if r["lambda_max"] else 0
            reached_80 = max_lmax >= 0.8 * threshold
            reached_str = "YES" if reached_80 else "NO"

            # Conservation drift: total absolute change from initial values
            if r["conservation"] and len(r["conservation"]) > 1:
                C_init = r["conservation"][0]
                C_final = r["conservation"][-1]
                drift = sum(abs(C_final[k] - C_init[k]) for k in range(len(C_init)))
            else:
                drift = 0.0

            print(
                f"{lr:>6.1f} | {threshold:>8.1f} | {max_lmax:>10.2f} | "
                f"{final_lmax:>11.2f} | {reached_str:>12} | {drift:>18.4f}"
            )

    # ----- EoS detection analysis -----
    print(f"\n{'='*72}")
    print("EOS DETECTION ANALYSIS")
    print(f"{'='*72}")
    for lr in LEARNING_RATES:
        threshold = 2.0 / lr
        lr_results = [r for r in all_results if r["lr"] == lr and not r["diverged"]]
        if not lr_results:
            print(f"\n  lr={lr}: ALL DIVERGED")
            continue

        max_lmax_vals = [max(r["lambda_max"]) for r in lr_results if r["lambda_max"]]
        final_lmax_vals = [r["lambda_max"][-1] for r in lr_results if r["lambda_max"]]
        mean_max = np.mean(max_lmax_vals) if max_lmax_vals else 0
        mean_final = np.mean(final_lmax_vals) if final_lmax_vals else 0

        # Check if lambda_max oscillates near threshold in late training
        eos_detected = False
        for r in lr_results:
            if len(r["lambda_max"]) > 20:
                late_lmax = r["lambda_max"][-20:]
                mean_late = np.mean(late_lmax)
                if mean_late >= 0.6 * threshold:
                    eos_detected = True

        status = "EoS DETECTED" if eos_detected else "No clear EoS"
        print(f"\n  lr={lr} (2/eta={threshold:.1f}):")
        print(f"    Mean max lambda_max: {mean_max:.2f}")
        print(f"    Mean final lambda_max: {mean_final:.2f}")
        print(f"    Ratio to threshold: {mean_max/threshold:.2f}")
        print(f"    Status: {status}")

    # ----- Save results -----
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "hardware": get_hardware_info(),
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "depth": DEPTH,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "log_every": LOG_EVERY,
        "hessian_iters": HESSIAN_ITERS,
        "loss": "MSE",
        "architecture": "mlp_deep_nobias",
        "optimizer": "SGD (full-batch)",
    }

    results_save = {
        "learning_rates": LEARNING_RATES,
        "thresholds": {str(lr): 2.0 / lr for lr in LEARNING_RATES},
        "runs": [],
    }
    for r in all_results:
        run_data = {
            "seed": r["seed"],
            "lr": r["lr"],
            "diverged": r["diverged"],
            "n_params": r["n_params"],
            "steps": r["steps"],
            "losses": r["losses"],
            "lambda_max": r["lambda_max"],
            "train_acc": r["train_acc"],
            "conservation": r["conservation"],
            "weight_norms": r["weight_norms"],
        }
        results_save["runs"].append(run_data)

    save_results(results_save, OUTPUT_DIR, config)
    print(f"\nResults saved to {OUTPUT_DIR}")

    # ----- Generate figures -----
    print("\nGenerating figures...")
    plot_eos_panels(all_results)
    plot_conservation_trajectories(all_results, target_lr=1.0)

    print(f"\nExperiment complete.")
