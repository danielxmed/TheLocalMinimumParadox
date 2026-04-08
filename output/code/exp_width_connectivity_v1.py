"""
Experiment: Width Sweep for Mode Connectivity Phase Transition
Theory: Theory B -- Percolation Phase Transition for Mode Connectivity
Prediction: There exists a critical width m* where the linear interpolation
            barrier between trained models drops sharply from high to near-zero,
            resembling a percolation phase transition.
Date: 2026-04-07
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info, linear_interpolation_loss
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# Configuration
# ============================================================

EXPERIMENT_NAME = "width_connectivity"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Dataset config (fixed across all widths)
DATA_CONFIG = {"name": "gaussian_mixture", "n_train": 200, "n_test": 50,
               "d": 20, "K": 5, "separation": 2.0}
LR = 0.01
N_STEPS = 2000
N_INTERP = 21  # Points along interpolation path

# Width sweep
WIDTHS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]

def train_model(width, seed):
    """Train a single 2-layer ReLU model."""
    set_seed(seed)
    model = make_model("mlp_2layer", DATA_CONFIG["d"], DATA_CONFIG["K"], hidden=width)
    X_train, Y_train, X_test, Y_test = make_dataset(
        DATA_CONFIG["name"], DATA_CONFIG["n_train"], DATA_CONFIG["n_test"],
        d=DATA_CONFIG["d"], K=DATA_CONFIG["K"], separation=DATA_CONFIG["separation"]
    )
    history = train(model, X_train, Y_train, lr=LR, n_steps=N_STEPS, log_every=N_STEPS)
    final_loss = history[-1]["loss"]

    # Compute test accuracy
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
        test_acc = (pred == Y_test).float().mean().item()
        pred_train = model(X_train).argmax(dim=1)
        train_acc = (pred_train == Y_train).float().mean().item()

    return model, X_train, Y_train, X_test, Y_test, final_loss, train_acc, test_acc

def compute_barrier(model1, model2, X, Y):
    """Compute the loss barrier along linear interpolation."""
    loss_fn = nn.CrossEntropyLoss()
    interp = linear_interpolation_loss(model1, model2, X, Y, loss_fn, n_points=N_INTERP)
    losses = [l for _, l, _ in interp]
    endpoint_loss = max(losses[0], losses[-1])
    barrier = max(losses) - endpoint_loss
    return barrier, losses

def sigmoid_transition(x, x0, k, L, b):
    """Sigmoid model for fitting the phase transition."""
    return L / (1 + np.exp(k * (x - x0))) + b

def run_experiment():
    print("="*60)
    print("Experiment: Width Sweep for Mode Connectivity")
    print("="*60)
    print(f"Widths: {WIDTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Data: n={DATA_CONFIG['n_train']}, d={DATA_CONFIG['d']}, K={DATA_CONFIG['K']}")
    print()

    all_results = {}

    for width in WIDTHS:
        print(f"\n--- Width = {width} ({count_parameters(make_model('mlp_2layer', 20, 5, hidden=width))} params) ---")

        # Train models with different seeds
        models = []
        train_data = None
        train_losses = []
        train_accs = []
        test_accs = []

        for seed in SEEDS:
            print(f"  Training seed {seed}...", end=" ", flush=True)
            t0 = time.time()
            model, X_tr, Y_tr, X_te, Y_te, loss, tr_acc, te_acc = train_model(width, seed)
            dt = time.time() - t0
            print(f"done ({dt:.1f}s) loss={loss:.4f} train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
            models.append(model)
            train_data = (X_tr, Y_tr)
            train_losses.append(loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

        # Compute pairwise barriers
        barriers = []
        all_interp_losses = []
        pair_count = 0
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                barrier, losses = compute_barrier(models[i], models[j], train_data[0], train_data[1])
                barriers.append(barrier)
                all_interp_losses.append(losses)
                pair_count += 1

        mean_barrier = np.mean(barriers)
        std_barrier = np.std(barriers)
        max_barrier = np.max(barriers)
        zero_barrier_frac = np.mean([1.0 if b < 0.01 else 0.0 for b in barriers])

        print(f"  Barriers: mean={mean_barrier:.4f} +/- {std_barrier:.4f}, max={max_barrier:.4f}")
        print(f"  Zero-barrier fraction: {zero_barrier_frac:.2f}")

        all_results[width] = {
            "width": width,
            "n_params": count_parameters(models[0]),
            "mean_barrier": float(mean_barrier),
            "std_barrier": float(std_barrier),
            "max_barrier": float(max_barrier),
            "barriers": [float(b) for b in barriers],
            "zero_barrier_frac": float(zero_barrier_frac),
            "mean_train_loss": float(np.mean(train_losses)),
            "mean_train_acc": float(np.mean(train_accs)),
            "mean_test_acc": float(np.mean(test_accs)),
            "interp_losses": [l for l in all_interp_losses],
        }

    return all_results

def generate_figures(results):
    setup_publication_style()

    widths = sorted(results.keys())
    mean_barriers = [results[w]["mean_barrier"] for w in widths]
    std_barriers = [results[w]["std_barrier"] for w in widths]
    train_accs = [results[w]["mean_train_acc"] for w in widths]
    test_accs = [results[w]["mean_test_acc"] for w in widths]
    zero_fracs = [results[w]["zero_barrier_frac"] for w in widths]

    # Figure 3: Mode connectivity barriers vs width (THE key figure)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel (a): Barrier height vs width
    axes[0].errorbar(widths, mean_barriers, yerr=std_barriers, fmt='o-', capsize=4,
                     color='darkblue', markersize=5)
    axes[0].set_xlabel('Network Width $m$')
    axes[0].set_ylabel('Mean Loss Barrier')
    axes[0].set_title('(a) Linear Interpolation Barrier')
    axes[0].set_xscale('log')

    # Try to fit sigmoid transition
    try:
        widths_arr = np.array(widths, dtype=float)
        barriers_arr = np.array(mean_barriers)
        if barriers_arr[0] > 0.01:  # Only fit if there's a visible transition
            popt, pcov = curve_fit(sigmoid_transition, np.log2(widths_arr), barriers_arr,
                                   p0=[np.log2(50), 2.0, barriers_arr[0], 0.0],
                                   maxfev=5000)
            x_fit = np.linspace(np.log2(min(widths)), np.log2(max(widths)), 100)
            axes[0].plot(2**x_fit, sigmoid_transition(x_fit, *popt), 'r--', alpha=0.7,
                        label=f'$m^* \\approx {2**popt[0]:.0f}$')
            axes[0].axvline(x=2**popt[0], color='red', linestyle=':', alpha=0.5)
            axes[0].legend()
    except Exception:
        pass

    # Panel (b): Fraction of zero-barrier pairs vs width
    axes[1].plot(widths, zero_fracs, 'o-', color='darkgreen', markersize=5)
    axes[1].set_xlabel('Network Width $m$')
    axes[1].set_ylabel('Fraction of Connected Pairs')
    axes[1].set_title('(b) Connectivity Probability')
    axes[1].set_xscale('log')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel (c): Training loss and accuracy vs width
    ax_loss = axes[2]
    ax_acc = ax_loss.twinx()
    l1 = ax_loss.plot(widths, [results[w]["mean_train_loss"] for w in widths],
                      'o-', color='darkblue', markersize=4, label='Train Loss')
    l2 = ax_acc.plot(widths, train_accs, 's-', color='darkred', markersize=4, label='Train Acc')
    l3 = ax_acc.plot(widths, test_accs, 'd-', color='orange', markersize=4, label='Test Acc')
    ax_loss.set_xlabel('Network Width $m$')
    ax_loss.set_ylabel('Training Loss', color='darkblue')
    ax_acc.set_ylabel('Accuracy', color='darkred')
    ax_loss.set_xscale('log')
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='center right', fontsize=9)
    axes[2].set_title('(c) Training Performance')

    plt.tight_layout()
    save_figure(fig, "fig3_width_phase_transition", FIGURE_DIR)
    plt.close()

    # Figure: Individual interpolation profiles for select widths
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    select_widths = [w for w in [8, 16, 32, 64, 128, 512] if w in results]
    alphas = np.linspace(0, 1, N_INTERP)

    for idx, w in enumerate(select_widths[:6]):
        ax = axes[idx // 3][idx % 3]
        for losses in results[w]["interp_losses"][:5]:  # Plot first 5 pairs
            ax.plot(alphas, losses, alpha=0.5, linewidth=0.8)
        ax.set_title(f'Width = {w}')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('Loss')

    fig.suptitle('Linear Interpolation Profiles Between Trained Models', y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig3b_interpolation_profiles", FIGURE_DIR)
    plt.close()

    print("Figures saved.")

if __name__ == "__main__":
    t_start = time.time()
    results = run_experiment()
    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Total time: {t_total:.1f}s")
    print(f"{'='*60}")

    # Summary
    print("\n--- PHASE TRANSITION SUMMARY ---")
    print(f"{'Width':>8} {'Params':>8} {'Barrier':>10} {'Connected':>12} {'TrainAcc':>10}")
    print("-" * 55)
    for w in sorted(results.keys()):
        r = results[w]
        print(f"{w:>8} {r['n_params']:>8} {r['mean_barrier']:>10.4f} {r['zero_barrier_frac']:>12.2f} {r['mean_train_acc']:>10.3f}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "theory": "percolation_phase_transition",
        "version": 1,
        "date": "2026-04-07",
        "hardware": get_hardware_info(),
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "lr": LR,
        "data_config": DATA_CONFIG,
        "widths": WIDTHS,
    }

    results_serializable = {str(k): v for k, v in results.items()}
    save_results(results_serializable, OUTPUT_DIR, config)

    generate_figures(results)
    print(f"\nResults saved to {OUTPUT_DIR}")
