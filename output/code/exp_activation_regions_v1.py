"""
Experiment: Activation Region Analysis on MNIST for ReLU Networks
Theory: Theory C -- Tropical Morse Theory
Prediction: After gradient-descent training, networks converge to the INTERIOR
            of activation regions (where the gradient is well-defined), NOT to
            PL critical cells on activation boundaries. This supports the claim
            that trained networks avoid the non-smooth locus of the PL loss surface.

Methodology:
  1. Train 2-layer ReLU networks (no bias) on MNIST (n=200, 784-dim) with
     varying widths [4, 8, 16, 32, 64].
  2. Before and after training, record the activation pattern of every training
     point (which neurons fire) and count distinct patterns.
  3. Track the number of "boundary neurons" (pre-activation near zero) over
     training to see whether training moves parameters away from boundaries.
  4. At convergence, check whether the model sits in the interior of its
     activation region (no neurons have pre-activation near zero) or on a
     boundary (some neurons do).

Date: 2026-04-07
"""

import os
import sys
import json
import time
import warnings
import numpy as np

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE,
    make_model, make_dataset, train,
    setup_publication_style, save_figure, save_results, get_hardware_info,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "activation_regions"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WIDTHS = [4, 8, 16, 32, 64]
N_TRAIN = 200
INPUT_DIM = 784          # flattened 28x28
OUTPUT_DIM = 10
LR = 0.01
N_STEPS = 2000
LOG_EVERY = 50
BOUNDARY_THRESHOLD = 1e-3  # pre-activation |z| < threshold => "near boundary"
BOUNDARY_THRESHOLDS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # for graduated analysis
SEEDS_TO_USE = SEEDS[:3]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_preactivations(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return pre-activation values of the hidden layer (before ReLU).

    For a 2-layer no-bias MLP: model = Linear -> ReLU -> Linear.
    The first Linear layer produces the pre-activations.

    Returns:
        Tensor of shape (n_samples, hidden_width).
    """
    with torch.no_grad():
        W1 = list(model.parameters())[0]  # shape (hidden, input_dim)
        pre_act = X @ W1.t()              # shape (n_samples, hidden)
    return pre_act


def get_activation_pattern(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return binary activation pattern: 1 if neuron fires, 0 otherwise.

    Returns:
        Tensor of shape (n_samples, hidden_width), dtype int.
    """
    pre_act = get_preactivations(model, X)
    return (pre_act > 0).int()


def count_distinct_patterns(model: nn.Module, X: torch.Tensor) -> int:
    """Count how many distinct activation patterns the data induces."""
    pattern = get_activation_pattern(model, X)
    # Convert each row to a hashable tuple
    pattern_set = set()
    for i in range(pattern.shape[0]):
        pattern_set.add(tuple(pattern[i].tolist()))
    return len(pattern_set)


def count_boundary_neurons(model: nn.Module, X: torch.Tensor,
                           threshold: float = BOUNDARY_THRESHOLD) -> dict:
    """Count neurons whose pre-activation is near zero (on an activation boundary).

    Returns dict with:
      - n_boundary_total: total (neuron, sample) pairs near boundary
      - n_boundary_neurons: number of distinct neurons that are near boundary
                            for at least one training sample
      - frac_boundary: fraction of all (neuron, sample) entries that are near zero
    """
    pre_act = get_preactivations(model, X)
    near_zero = (pre_act.abs() < threshold)
    n_total = near_zero.sum().item()
    n_neurons_with_boundary = (near_zero.any(dim=0)).sum().item()
    frac = n_total / max(pre_act.numel(), 1)
    return {
        "n_boundary_total": int(n_total),
        "n_boundary_neurons": int(n_neurons_with_boundary),
        "frac_boundary": float(frac),
    }


def compute_gradient_norm(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                          loss_fn: nn.Module) -> float:
    """Compute the full-batch gradient norm at current parameters."""
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y)
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    model.zero_grad()
    return float(np.sqrt(grad_norm))


def is_interior_point(model: nn.Module, X: torch.Tensor,
                      threshold: float = BOUNDARY_THRESHOLD) -> bool:
    """Check if current parameters are in the interior of their activation region.

    True if NO neuron has pre-activation near zero for ANY training sample.
    This means the gradient is well-defined (not on a PL critical cell).
    """
    pre_act = get_preactivations(model, X)
    return bool((pre_act.abs() >= threshold).all().item())


def compute_min_margin(model: nn.Module, X: torch.Tensor) -> dict:
    """Compute the minimum absolute pre-activation across all neurons and samples.

    This is the "margin" to the nearest activation boundary. Larger values
    mean the parameters are deeper in the interior of their activation region.

    Returns dict with:
      - min_margin: smallest |z_i(x_j)| across all neurons i and samples j
      - mean_margin: average |z_i(x_j)|
      - median_margin: median |z_i(x_j)|
      - percentile_5: 5th percentile of |z_i(x_j)| (near-boundary tail)
    """
    pre_act = get_preactivations(model, X)
    abs_pre = pre_act.abs()
    return {
        "min_margin": float(abs_pre.min().item()),
        "mean_margin": float(abs_pre.mean().item()),
        "median_margin": float(abs_pre.median().item()),
        "percentile_5": float(torch.quantile(abs_pre.float(), 0.05).item()),
    }


def count_boundary_at_thresholds(model: nn.Module, X: torch.Tensor,
                                 thresholds: list = None) -> dict:
    """Count boundary neurons at multiple thresholds for graduated analysis."""
    if thresholds is None:
        thresholds = BOUNDARY_THRESHOLDS
    pre_act = get_preactivations(model, X)
    abs_pre = pre_act.abs()
    result = {}
    for t in thresholds:
        near = (abs_pre < t).sum().item()
        frac = near / max(abs_pre.numel(), 1)
        result[f"boundary_frac_at_{t:.0e}"] = float(frac)
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_single(width: int, seed: int, X_train: torch.Tensor,
               Y_train: torch.Tensor) -> dict:
    """Run one experiment: train a 2-layer ReLU net, track activation regions."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", input_dim=INPUT_DIM,
                       output_dim=OUTPUT_DIM, hidden=width)
    loss_fn = nn.CrossEntropyLoss()

    # ----- Before training -----
    n_patterns_before = count_distinct_patterns(model, X_train)
    boundary_before = count_boundary_neurons(model, X_train)
    interior_before = is_interior_point(model, X_train)
    grad_norm_before = compute_gradient_norm(model, X_train, Y_train, loss_fn)
    margin_before = compute_min_margin(model, X_train)
    threshold_before = count_boundary_at_thresholds(model, X_train)

    # ----- Training with boundary tracking -----
    boundary_history = []
    step_history = []

    def callback(mdl, step, loss_val):
        """Record boundary neuron count at each logged step."""
        bd = count_boundary_neurons(mdl, X_train)
        boundary_history.append(bd["n_boundary_total"])
        step_history.append(step)
        return {"boundary_total": bd["n_boundary_total"]}

    print(f"\n  Width={width}, seed={seed}")
    log = train(model, X_train, Y_train, lr=LR, n_steps=N_STEPS,
                loss_fn=loss_fn, optimizer_type="sgd",
                callback=callback, log_every=LOG_EVERY)

    # ----- After training -----
    n_patterns_after = count_distinct_patterns(model, X_train)
    boundary_after = count_boundary_neurons(model, X_train)
    interior_after = is_interior_point(model, X_train)
    grad_norm_after = compute_gradient_norm(model, X_train, Y_train, loss_fn)
    margin_after = compute_min_margin(model, X_train)
    threshold_after = count_boundary_at_thresholds(model, X_train)
    final_loss = log[-1]["loss"]

    result = {
        "width": width,
        "seed": seed,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_patterns_before": n_patterns_before,
        "n_patterns_after": n_patterns_after,
        "boundary_before": boundary_before,
        "boundary_after": boundary_after,
        "interior_before": interior_before,
        "interior_after": interior_after,
        "grad_norm_before": grad_norm_before,
        "grad_norm_after": grad_norm_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "threshold_before": threshold_before,
        "threshold_after": threshold_after,
        "final_loss": final_loss,
        "boundary_history": boundary_history,
        "step_history": step_history,
    }

    print(f"    Patterns: {n_patterns_before} -> {n_patterns_after}")
    print(f"    Boundary neurons (total): "
          f"{boundary_before['n_boundary_total']} -> {boundary_after['n_boundary_total']}")
    print(f"    Interior point: {interior_before} -> {interior_after}")
    print(f"    Min margin: {margin_before['min_margin']:.6f} -> {margin_after['min_margin']:.6f}")
    print(f"    5th pctile margin: {margin_before['percentile_5']:.6f} -> {margin_after['percentile_5']:.6f}")
    print(f"    Grad norm: {grad_norm_before:.4f} -> {grad_norm_after:.4f}")
    print(f"    Final loss: {final_loss:.6f}")

    return result


def main():
    t_start = time.time()
    print("=" * 70)
    print("Activation Region Analysis on MNIST (Theory C: Tropical Morse Theory)")
    print("=" * 70)
    print(f"Widths: {WIDTHS}")
    print(f"Training: {N_STEPS} steps, lr={LR}, full-batch SGD")
    print(f"Data: MNIST, n={N_TRAIN}, flattened to {INPUT_DIM}-dim")
    print(f"Seeds: {SEEDS_TO_USE}")
    print(f"Boundary threshold: {BOUNDARY_THRESHOLD}")
    print()

    # Load MNIST data
    set_seed(42)
    X_train, Y_train, X_test, Y_test = make_dataset(
        "mnist", n_train=N_TRAIN, n_test=50
    )
    print(f"Data loaded: X_train={X_train.shape}, Y_train={Y_train.shape}")

    # Run experiments
    all_results = {}
    for width in WIDTHS:
        all_results[width] = []
        for seed in SEEDS_TO_USE:
            result = run_single(width, seed, X_train, Y_train)
            all_results[width].append(result)

    t_total = time.time() - t_start

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = {}
    for width in WIDTHS:
        runs = all_results[width]
        pats_before = [r["n_patterns_before"] for r in runs]
        pats_after = [r["n_patterns_after"] for r in runs]
        interior_after_list = [r["interior_after"] for r in runs]
        grad_norms = [r["grad_norm_after"] for r in runs]
        losses = [r["final_loss"] for r in runs]
        min_margins_before = [r["margin_before"]["min_margin"] for r in runs]
        min_margins_after = [r["margin_after"]["min_margin"] for r in runs]
        p5_margins_before = [r["margin_before"]["percentile_5"] for r in runs]
        p5_margins_after = [r["margin_after"]["percentile_5"] for r in runs]
        mean_margins_before = [r["margin_before"]["mean_margin"] for r in runs]
        mean_margins_after = [r["margin_after"]["mean_margin"] for r in runs]

        summary[width] = {
            "width": width,
            "n_params": runs[0]["n_params"],
            "patterns_before_mean": float(np.mean(pats_before)),
            "patterns_before_std": float(np.std(pats_before)),
            "patterns_after_mean": float(np.mean(pats_after)),
            "patterns_after_std": float(np.std(pats_after)),
            "frac_interior_after": float(np.mean(interior_after_list)),
            "grad_norm_after_mean": float(np.mean(grad_norms)),
            "grad_norm_after_std": float(np.std(grad_norms)),
            "final_loss_mean": float(np.mean(losses)),
            "final_loss_std": float(np.std(losses)),
            "min_margin_before_mean": float(np.mean(min_margins_before)),
            "min_margin_after_mean": float(np.mean(min_margins_after)),
            "p5_margin_before_mean": float(np.mean(p5_margins_before)),
            "p5_margin_after_mean": float(np.mean(p5_margins_after)),
            "mean_margin_before_mean": float(np.mean(mean_margins_before)),
            "mean_margin_after_mean": float(np.mean(mean_margins_after)),
        }

        print(f"Width={width:3d} | Patterns: "
              f"{summary[width]['patterns_before_mean']:.0f} -> "
              f"{summary[width]['patterns_after_mean']:.0f} | "
              f"Interior: {summary[width]['frac_interior_after']*100:.0f}% | "
              f"Grad norm: {summary[width]['grad_norm_after_mean']:.4f} | "
              f"Loss: {summary[width]['final_loss_mean']:.4f}")
        print(f"          | Min margin: "
              f"{summary[width]['min_margin_before_mean']:.6f} -> "
              f"{summary[width]['min_margin_after_mean']:.6f} | "
              f"5th pctile: "
              f"{summary[width]['p5_margin_before_mean']:.4f} -> "
              f"{summary[width]['p5_margin_after_mean']:.4f} | "
              f"Mean margin: "
              f"{summary[width]['mean_margin_before_mean']:.4f} -> "
              f"{summary[width]['mean_margin_after_mean']:.4f}")

    # Key prediction check
    print("\n--- PREDICTION CHECK ---")
    print("Prediction: After training, models should be in the INTERIOR of")
    print("activation regions (gradient well-defined), supporting the tropical")
    print("Morse theory claim that trained networks avoid PL critical cells.")
    print()
    print("Analysis at threshold 1e-3 (strict):")
    for width in WIDTHS:
        frac = summary[width]["frac_interior_after"]
        status = "CONFIRMED" if frac > 0.5 else "MIXED" if frac > 0 else "REFUTED"
        print(f"  Width={width:3d}: {frac*100:.0f}% runs in interior -> {status}")

    print()
    print("Nuanced analysis (margin-based):")
    print("  The 5th percentile margin and mean margin reveal whether pre-activations")
    print("  are MOVING TOWARD or AWAY from boundaries during training.")
    for width in WIDTHS:
        p5_before = summary[width]["p5_margin_before_mean"]
        p5_after = summary[width]["p5_margin_after_mean"]
        mean_before = summary[width]["mean_margin_before_mean"]
        mean_after = summary[width]["mean_margin_after_mean"]
        p5_change = "GREW" if p5_after > p5_before else "SHRANK"
        mean_change = "GREW" if mean_after > mean_before else "SHRANK"
        print(f"  Width={width:3d}: 5th pctile margin {p5_change} "
              f"({p5_before:.4f} -> {p5_after:.4f}), "
              f"mean margin {mean_change} "
              f"({mean_before:.4f} -> {mean_after:.4f})")

    print(f"\nTotal time: {t_total:.1f}s")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Convert all_results for JSON serialization
    json_results = {}
    for width in WIDTHS:
        json_results[str(width)] = []
        for r in all_results[width]:
            entry = {k: v for k, v in r.items()
                     if k not in ("boundary_history", "step_history")}
            entry["boundary_history"] = r["boundary_history"]
            entry["step_history"] = r["step_history"]
            json_results[str(width)].append(entry)

    json_results["summary"] = {str(k): v for k, v in summary.items()}

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "theory": "Theory C -- Tropical Morse Theory",
        "prediction": ("After training, models should be in the INTERIOR of "
                       "activation regions (gradient well-defined), not on "
                       "boundaries (PL critical cells)."),
        "hardware": get_hardware_info(),
        "widths": WIDTHS,
        "n_train": N_TRAIN,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "lr": LR,
        "n_steps": N_STEPS,
        "boundary_threshold": BOUNDARY_THRESHOLD,
        "seeds": SEEDS_TO_USE,
        "total_time_s": t_total,
    }

    save_results(json_results, OUTPUT_DIR, config)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): Distinct activation patterns vs width ---
    ax = axes[0]
    ws = np.array(WIDTHS)

    before_means = [summary[w]["patterns_before_mean"] for w in WIDTHS]
    before_stds = [summary[w]["patterns_before_std"] for w in WIDTHS]
    after_means = [summary[w]["patterns_after_mean"] for w in WIDTHS]
    after_stds = [summary[w]["patterns_after_std"] for w in WIDTHS]

    ax.errorbar(ws, before_means, yerr=before_stds, fmt="o--", color="steelblue",
                capsize=4, markersize=6, label="Before training")
    ax.errorbar(ws, after_means, yerr=after_stds, fmt="s-", color="darkred",
                capsize=4, markersize=6, label="After training")
    ax.set_xlabel("Hidden width $m$")
    ax.set_ylabel("Distinct activation patterns")
    ax.set_title("(a) Activation Regions vs Width")
    ax.legend(frameon=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])

    # --- Panel (b): Boundary neurons vs training step ---
    ax = axes[1]
    cmap = plt.cm.viridis
    for i, width in enumerate(WIDTHS):
        color = cmap(i / max(len(WIDTHS) - 1, 1))
        # Average boundary history across seeds
        runs = all_results[width]
        min_len = min(len(r["boundary_history"]) for r in runs)
        avg_boundary = np.mean(
            [r["boundary_history"][:min_len] for r in runs], axis=0
        )
        steps = runs[0]["step_history"][:min_len]
        ax.plot(steps, avg_boundary, color=color, linewidth=1.5,
                label=f"$m={width}$")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Boundary neuron count\n(pre-activation $|z| < 10^{-3}$)")
    ax.set_title("(b) Boundary Neurons During Training")
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle("Activation Region Structure on MNIST "
                 "(Theory C: Tropical Morse Theory)", y=1.03, fontsize=14)
    plt.tight_layout()
    save_figure(fig, "fig_activation_regions", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig_activation_regions.png")
    print("Done.")


if __name__ == "__main__":
    main()
