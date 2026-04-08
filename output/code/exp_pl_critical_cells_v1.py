"""
Experiment: PL Critical Cell Enumeration for Tiny ReLU Networks
Theory: Theory C -- Tropical Morse Theory
Prediction: The number of PL critical cells (boundary points where gradient
            direction changes) should decrease with width, and most critical
            cells at low loss should be saddle-like (high index), not minima.
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, save_results, setup_publication_style,
    save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "pl_critical_cells"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tiny setting for exhaustive enumeration
INPUT_DIM = 2
OUTPUT_DIM = 1
N_DATA = 20
WIDTHS = [2, 3, 4, 5, 6, 8, 10, 15, 20]
N_SAMPLES = 5000  # Random parameter samples to probe landscape
N_BOUNDARY_PROBES = 10000  # Random directions to find activation boundaries

def make_tiny_model(width):
    """Create a simple 2-layer ReLU network: Linear(2, m) -> ReLU -> Linear(m, 1)."""
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, OUTPUT_DIM, bias=False)
    )
    return model

def get_activation_pattern(model, X):
    """Get binary activation pattern for all neurons on all data points."""
    with torch.no_grad():
        W1 = list(model.parameters())[0]  # (m, d)
        pre_act = X @ W1.t()  # (n, m)
        pattern = (pre_act > 0).int()  # (n, m)
    return pattern

def count_activation_regions(model, X, n_samples=5000, seed=42):
    """Count distinct activation patterns by random sampling in parameter space."""
    set_seed(seed)
    W1_orig = list(model.parameters())[0].data.clone()
    W2_orig = list(model.parameters())[1].data.clone()

    patterns_seen = set()

    for i in range(n_samples):
        # Random parameters
        scale = 2.0
        W1_new = torch.randn_like(W1_orig) * scale
        W2_new = torch.randn_like(W2_orig) * scale
        list(model.parameters())[0].data.copy_(W1_new)
        list(model.parameters())[1].data.copy_(W2_new)

        pattern = get_activation_pattern(model, X)
        pattern_tuple = tuple(pattern.flatten().tolist())
        patterns_seen.add(pattern_tuple)

    # Restore original
    list(model.parameters())[0].data.copy_(W1_orig)
    list(model.parameters())[1].data.copy_(W2_orig)

    return len(patterns_seen)

def find_boundary_critical_cells(model, X, Y, n_probes=10000, seed=42):
    """Find approximate PL critical cells by probing activation boundaries.

    Strategy: Sample random parameter directions. Along each direction,
    find where activation patterns change. At those boundary points,
    check if the gradient direction reverses (indicating a PL critical cell).
    """
    set_seed(seed)
    loss_fn = nn.MSELoss()
    params = [p for p in model.parameters()]
    n_params = sum(p.numel() for p in params)

    critical_cells = []
    boundary_points = 0
    saddle_cells = 0  # Gradient reverses in some but not all directions
    minima_cells = 0  # Gradient reverses in all directions

    for probe in range(n_probes):
        # Random starting point and direction
        theta_start = torch.randn(n_params) * 2.0
        direction = torch.randn(n_params)
        direction = direction / direction.norm()

        # Set parameters to theta_start
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(theta_start[offset:offset+n].reshape(p.shape))
            offset += n

        pattern_start = get_activation_pattern(model, X)

        # Move along direction and find where pattern changes
        for alpha in np.linspace(0.01, 4.0, 100):
            theta_new = theta_start + alpha * direction
            offset = 0
            for p in params:
                n = p.numel()
                p.data.copy_(theta_new[offset:offset+n].reshape(p.shape))
                offset += n

            pattern_new = get_activation_pattern(model, X)
            if not torch.equal(pattern_start, pattern_new):
                # Found a boundary! Check gradient on both sides
                boundary_points += 1

                # Gradient just before boundary
                theta_before = theta_start + (alpha - 0.005) * direction
                offset = 0
                for p in params:
                    n = p.numel()
                    p.data.copy_(theta_before[offset:offset+n].reshape(p.shape))
                    offset += n
                model.zero_grad()
                out = model(X)
                loss_before = loss_fn(out, Y)
                loss_before.backward()
                grad_before = torch.cat([p.grad.flatten() for p in params]).clone()
                grad_proj_before = (grad_before * direction).sum().item()

                # Gradient just after boundary
                theta_after = theta_start + (alpha + 0.005) * direction
                offset = 0
                for p in params:
                    n = p.numel()
                    p.data.copy_(theta_after[offset:offset+n].reshape(p.shape))
                    offset += n
                model.zero_grad()
                out = model(X)
                loss_after = loss_fn(out, Y)
                loss_after.backward()
                grad_after = torch.cat([p.grad.flatten() for p in params]).clone()
                grad_proj_after = (grad_after * direction).sum().item()

                # Check if gradient projection reverses sign
                if grad_proj_before * grad_proj_after < 0:
                    # This is a PL critical cell along this direction!
                    loss_val = 0.5 * (loss_before.item() + loss_after.item())
                    critical_cells.append({
                        "loss": loss_val,
                        "alpha": alpha,
                        "grad_before": grad_proj_before,
                        "grad_after": grad_proj_after,
                    })

                break  # Move to next probe

    # Classify: what fraction of boundary points are critical?
    n_critical = len(critical_cells)
    frac_critical = n_critical / max(boundary_points, 1)

    return {
        "n_boundary_points": boundary_points,
        "n_critical_cells": n_critical,
        "frac_critical": frac_critical,
        "critical_losses": [c["loss"] for c in critical_cells],
    }

if __name__ == "__main__":
    print("PL Critical Cell Enumeration")
    print(f"Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")
    print(f"Data points: {N_DATA}")
    print(f"Widths: {WIDTHS}")
    print()

    # Generate fixed data
    set_seed(42)
    X = torch.randn(N_DATA, INPUT_DIM)
    Y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)  # XOR-like pattern

    all_results = {}
    t_start = time.time()

    for width in WIDTHS:
        model = make_tiny_model(width)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Width={width} ({n_params} params):")

        # Count activation regions
        n_regions = count_activation_regions(model, X, n_samples=N_SAMPLES)
        print(f"  Activation regions found: {n_regions} (from {N_SAMPLES} samples)")

        # Find PL critical cells
        result = find_boundary_critical_cells(model, X, Y, n_probes=N_BOUNDARY_PROBES)
        print(f"  Boundary points: {result['n_boundary_points']}")
        print(f"  PL critical cells: {result['n_critical_cells']} ({result['frac_critical']:.3f} of boundaries)")
        if result['critical_losses']:
            print(f"  Critical cell losses: mean={np.mean(result['critical_losses']):.4f}, min={np.min(result['critical_losses']):.4f}")

        all_results[width] = {
            "width": width, "n_params": n_params,
            "n_regions": n_regions,
            **result,
        }

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary
    print("\n--- PL CRITICAL CELL SUMMARY ---")
    print(f"{'Width':>6} {'Params':>7} {'Regions':>8} {'Boundaries':>11} {'Critical':>9} {'Frac':>8}")
    print("-" * 55)
    for w in WIDTHS:
        r = all_results[w]
        print(f"{w:>6} {r['n_params']:>7} {r['n_regions']:>8} {r['n_boundary_points']:>11} {r['n_critical_cells']:>9} {r['frac_critical']:>8.3f}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
        "n_data": N_DATA, "widths": WIDTHS, "n_samples": N_SAMPLES, "n_probes": N_BOUNDARY_PROBES,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ws = WIDTHS
    regions = [all_results[w]["n_regions"] for w in ws]
    critical = [all_results[w]["n_critical_cells"] for w in ws]
    fracs = [all_results[w]["frac_critical"] for w in ws]

    axes[0].plot(ws, regions, 'o-', color='darkblue', markersize=5)
    axes[0].set_xlabel('Width $m$')
    axes[0].set_ylabel('Number of Activation Regions')
    axes[0].set_title('(a) Activation Region Count')

    axes[1].plot(ws, critical, 'o-', color='darkred', markersize=5)
    axes[1].set_xlabel('Width $m$')
    axes[1].set_ylabel('PL Critical Cells Found')
    axes[1].set_title('(b) PL Critical Cell Count')

    axes[2].plot(ws, fracs, 'o-', color='darkgreen', markersize=5)
    axes[2].set_xlabel('Width $m$')
    axes[2].set_ylabel('Critical / Boundary Ratio')
    axes[2].set_title('(c) Fraction of Boundaries that are Critical')

    fig.suptitle('PL Critical Cell Structure vs Width (2-Layer ReLU, XOR data)', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig_tropical_critical_cells", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
