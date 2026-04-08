"""
Experiment: Mode Connectivity in Deep Networks with Permutation Alignment
Theory: Theory B -- Permutation Symmetry and Basin Structure
Prediction: Deep (4-layer) MLPs should exhibit loss barriers along linear
            interpolation paths between independently trained models. These
            barriers should diminish (or vanish) after permutation alignment,
            confirming that the apparent disconnection is an artifact of the
            enlarged permutation symmetry group in deep networks.
            Wider networks should have smaller post-alignment barriers.

Prior result: 2-layer networks showed zero barriers everywhere.
New test:     4-layer networks should reveal non-trivial barrier structure.

Date: 2026-04-07
Version: 1
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, linear_interpolation_loss, permutation_align,
    save_results, setup_publication_style, save_figure, get_hardware_info,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "deep_connectivity"
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

# Data: MNIST, n=1000, flattened to 784 dimensions, 10 classes
DATA_NAME = "mnist"
N_TRAIN = 1000
N_TEST = 200

# Architecture: 4-layer deep MLP
ARCH = "mlp_deep"
DEPTH = 4
WIDTHS = [8, 16, 32, 64, 128]

# Training
LR = 0.01
N_STEPS = 3000
LOSS_FN = nn.CrossEntropyLoss()

# Interpolation
N_INTERP = 21


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_model(width, seed, X_train, Y_train):
    """Train one model with given width and seed. Return trained model."""
    set_seed(seed)
    model = make_model(ARCH, input_dim=784, output_dim=10, hidden=width, depth=DEPTH)
    train(model, X_train, Y_train, lr=LR, n_steps=N_STEPS, log_every=N_STEPS)
    return model


def evaluate(model, X, Y):
    """Return (loss, accuracy) on the given data."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = LOSS_FN(logits, Y).item()
        acc = (logits.argmax(1) == Y).float().mean().item()
    return loss, acc


def compute_barrier(model1, model2, X, Y):
    """
    Compute the linear interpolation barrier between two models.
    Barrier = max(interpolated losses) - max(endpoint losses).
    Returns (barrier, list_of_losses).
    """
    interp = linear_interpolation_loss(model1, model2, X, Y, LOSS_FN, n_points=N_INTERP)
    losses = [l for _, l, _ in interp]
    endpoint_max = max(losses[0], losses[-1])
    barrier = max(losses) - endpoint_max
    return max(barrier, 0.0), losses


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Experiment: Deep Network Mode Connectivity with Permutation Alignment")
    print(f"Architecture: {ARCH}, depth={DEPTH}")
    print(f"Data: MNIST, n_train={N_TRAIN}, n_test={N_TEST}")
    print(f"Training: lr={LR}, steps={N_STEPS}, full-batch GD, cross-entropy")
    print(f"Widths: {WIDTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load data once (shared across all models)
    print("\nLoading MNIST data...")
    set_seed(0)
    X_train, Y_train, X_test, Y_test = make_dataset(DATA_NAME, N_TRAIN, N_TEST)
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

    all_results = {}
    t_start = time.time()

    for width in WIDTHS:
        n_params = count_parameters(
            make_model(ARCH, input_dim=784, output_dim=10, hidden=width, depth=DEPTH)
        )
        print(f"\n{'='*60}")
        print(f"Width = {width}  ({n_params:,} parameters)")
        print(f"{'='*60}")

        # Train all models for this width
        models = []
        train_losses = []
        train_accs = []
        test_accs = []

        for seed in SEEDS:
            t_model = time.time()
            print(f"  Training seed={seed}...", end=" ", flush=True)
            model = train_model(width, seed, X_train, Y_train)
            tr_loss, tr_acc = evaluate(model, X_train, Y_train)
            _, te_acc = evaluate(model, X_test, Y_test)
            models.append(model)
            train_losses.append(tr_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            dt = time.time() - t_model
            print(f"loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  te_acc={te_acc:.3f}  ({dt:.1f}s)")

        # Pairwise barriers BEFORE alignment
        print(f"\n  Computing pairwise barriers (before alignment)...")
        barriers_before = []
        pair_details_before = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                b, losses = compute_barrier(models[i], models[j], X_train, Y_train)
                barriers_before.append(b)
                pair_details_before.append({
                    "seeds": [SEEDS[i], SEEDS[j]],
                    "barrier": float(b),
                })
                print(f"    ({SEEDS[i]}, {SEEDS[j]}): barrier = {b:.6f}")

        mean_before = float(np.mean(barriers_before))
        std_before = float(np.std(barriers_before))
        print(f"  BEFORE alignment: mean barrier = {mean_before:.6f} +/- {std_before:.6f}")

        # Pairwise barriers AFTER permutation alignment
        print(f"\n  Computing pairwise barriers (after permutation alignment)...")
        barriers_after = []
        pair_details_after = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                # Align model j to model i
                aligned_j = permutation_align(models[i], models[j], X_train)
                b, losses = compute_barrier(models[i], aligned_j, X_train, Y_train)
                barriers_after.append(b)
                pair_details_after.append({
                    "seeds": [SEEDS[i], SEEDS[j]],
                    "barrier": float(b),
                })
                print(f"    ({SEEDS[i]}, {SEEDS[j]}): barrier = {b:.6f}")

        mean_after = float(np.mean(barriers_after))
        std_after = float(np.std(barriers_after))
        print(f"  AFTER alignment:  mean barrier = {mean_after:.6f} +/- {std_after:.6f}")

        reduction = (mean_before - mean_after) / max(mean_before, 1e-10) * 100
        print(f"  Reduction: {reduction:.1f}%")

        all_results[width] = {
            "width": width,
            "depth": DEPTH,
            "n_params": n_params,
            "mean_train_loss": float(np.mean(train_losses)),
            "mean_train_acc": float(np.mean(train_accs)),
            "mean_test_acc": float(np.mean(test_accs)),
            "mean_barrier_before": mean_before,
            "std_barrier_before": std_before,
            "barriers_before": [float(b) for b in barriers_before],
            "pair_details_before": pair_details_before,
            "mean_barrier_after": mean_after,
            "std_barrier_after": std_after,
            "barriers_after": [float(b) for b in barriers_after],
            "pair_details_after": pair_details_after,
            "barrier_reduction_pct": float(reduction),
        }

    t_total = time.time() - t_start

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: Deep Network Mode Connectivity (depth=4)")
    print("=" * 80)
    print(f"{'Width':>8} {'Params':>8} {'TrAcc':>8} {'TeAcc':>8} "
          f"{'Barr(raw)':>12} {'Barr(align)':>12} {'Reduction':>10}")
    print("-" * 80)
    for w in sorted(all_results.keys()):
        r = all_results[w]
        print(f"{w:>8} {r['n_params']:>8,} {r['mean_train_acc']:>8.3f} "
              f"{r['mean_test_acc']:>8.3f} {r['mean_barrier_before']:>12.6f} "
              f"{r['mean_barrier_after']:>12.6f} {r['barrier_reduction_pct']:>9.1f}%")

    print(f"\nTotal wall time: {t_total:.1f}s")

    # Check predictions
    print("\n--- THEORY B PREDICTIONS ---")
    widths_sorted = sorted(all_results.keys())

    # Prediction 1: Deep networks should show barriers before alignment
    has_barriers = any(all_results[w]["mean_barrier_before"] > 0.01 for w in widths_sorted)
    print(f"[{'CONFIRMED' if has_barriers else 'REJECTED'}] "
          f"Deep networks show non-trivial barriers without alignment")

    # Prediction 2: Barriers should decrease after alignment
    all_decrease = all(
        all_results[w]["mean_barrier_after"] <= all_results[w]["mean_barrier_before"] + 1e-8
        for w in widths_sorted
    )
    print(f"[{'CONFIRMED' if all_decrease else 'REJECTED'}] "
          f"Barriers decrease after permutation alignment")

    # Prediction 3: Wider networks should have smaller post-alignment barriers
    post_barriers = [all_results[w]["mean_barrier_after"] for w in widths_sorted]
    monotone_decrease = all(
        post_barriers[i] >= post_barriers[i + 1] - 1e-6
        for i in range(len(post_barriers) - 1)
    )
    print(f"[{'CONFIRMED' if monotone_decrease else 'PARTIAL'}] "
          f"Post-alignment barriers decrease with width")

    # Prediction 4: Width threshold where post-alignment barriers vanish
    vanish_threshold = None
    for w in widths_sorted:
        if all_results[w]["mean_barrier_after"] < 0.01:
            vanish_threshold = w
            break
    if vanish_threshold:
        print(f"[CONFIRMED] Post-alignment barriers vanish at width >= {vanish_threshold}")
    else:
        print(f"[NOT YET] No width found where post-alignment barriers vanish (<0.01)")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "theory": "B_permutation_symmetry_deep",
        "version": 1,
        "date": "2026-04-07",
        "hardware": get_hardware_info(),
        "architecture": ARCH,
        "depth": DEPTH,
        "widths": WIDTHS,
        "seeds": SEEDS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "lr": LR,
        "n_steps": N_STEPS,
        "n_interp_points": N_INTERP,
        "data": DATA_NAME,
        "total_time_seconds": t_total,
    }
    save_results({str(k): v for k, v in all_results.items()}, OUTPUT_DIR, config)
    print(f"\nResults saved to {OUTPUT_DIR}")

    # ---------------------------------------------------------------------------
    # Figure: Bar chart -- barrier before vs after alignment
    # ---------------------------------------------------------------------------
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(WIDTHS))
    bar_width = 0.35

    before_means = [all_results[w]["mean_barrier_before"] for w in WIDTHS]
    before_stds = [all_results[w]["std_barrier_before"] for w in WIDTHS]
    after_means = [all_results[w]["mean_barrier_after"] for w in WIDTHS]
    after_stds = [all_results[w]["std_barrier_after"] for w in WIDTHS]

    bars1 = ax.bar(
        x_positions - bar_width / 2, before_means, bar_width,
        yerr=before_stds, capsize=4,
        label="Before alignment", color="#3274A1", edgecolor="black", linewidth=0.5,
    )
    bars2 = ax.bar(
        x_positions + bar_width / 2, after_means, bar_width,
        yerr=after_stds, capsize=4,
        label="After alignment", color="#E1812C", edgecolor="black", linewidth=0.5,
    )

    ax.set_xlabel("Network Width", fontsize=12)
    ax.set_ylabel("Mean Loss Barrier", fontsize=12)
    ax.set_title(
        f"Mode Connectivity: 4-Layer MLP on MNIST (n={N_TRAIN})\n"
        f"Linear Interpolation Barriers Before vs After Permutation Alignment",
        fontsize=11,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.legend(fontsize=10, frameon=True)
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0.001:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    plt.tight_layout()
    save_figure(fig, "fig3d_deep_connectivity", FIGURE_DIR)
    plt.close()
    print(f"Figure saved to {FIGURE_DIR}/fig3d_deep_connectivity.png")
