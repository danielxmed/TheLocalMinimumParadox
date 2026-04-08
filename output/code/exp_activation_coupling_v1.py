"""
Experiment: Activation Pattern Coupling Dynamics
Theory: Theory A -- Understanding ReLU mode coupling strength
Prediction: If activation patterns are stable during training (low Hamming distance
            between consecutive steps), then mode coupling from ReLU is weak and
            Theorem 5 (derived for linear networks) applies to ReLU networks.
Date: 2026-04-07 (Session 5)

Method:
  1. Train 2-layer ReLU network (hidden=64, no bias) at various learning rates
  2. At every step, record:
     - Activation pattern: which neurons fire for each training point
     - Hamming distance between consecutive activation patterns
     - Per-step gradient imbalance delta(t)
     - Minimum activation margin (min |pre-activation|)
  3. Compute:
     - Activation stability: fraction of steps with zero pattern changes
     - Correlation between pattern changes and gradient imbalance magnitude
     - Margin evolution: does training increase or decrease the margin?
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "activation_coupling"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

LRS = [0.001, 0.003, 0.01, 0.03, 0.1]
N_STEPS = 2000
HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200


def get_preactivations(model, X):
    """Get pre-activation values of the hidden layer (before ReLU)."""
    with torch.no_grad():
        W1 = list(model.parameters())[0]  # shape (hidden, input_dim)
        return X @ W1.t()  # shape (n_samples, hidden)


def get_activation_pattern(model, X):
    """Binary activation pattern: 1 if neuron fires, 0 otherwise."""
    pre_act = get_preactivations(model, X)
    return (pre_act > 0).int()


def hamming_distance(pattern1, pattern2):
    """Average Hamming distance between two activation pattern matrices.
    Returns: (mean_distance, fraction_changed, n_neurons_changed)
    """
    diff = (pattern1 != pattern2).float()
    total_changes = diff.sum().item()
    n_elements = diff.numel()
    frac_changed = total_changes / n_elements

    # Number of neurons that changed for at least one data point
    neurons_changed = (diff.sum(dim=0) > 0).sum().item()

    return frac_changed, neurons_changed


def run_activation_tracking(lr, seed):
    """Train ReLU network and track activation pattern dynamics at every step."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", INPUT_DIM, OUTPUT_DIM, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                               d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]

    # Initial state
    prev_pattern = get_activation_pattern(model, X)
    C_init = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2

    # Tracking arrays
    hamming_dists = []         # Fraction of (neuron, sample) pairs that changed
    neurons_changed_list = []  # Number of neurons that changed
    grad_imbalances = []       # Per-step gradient imbalance
    min_margins = []           # Minimum |pre-activation| across all neurons and samples
    mean_margins = []          # Mean |pre-activation|
    losses = []
    n_distinct_patterns = []   # How many distinct activation patterns exist

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        # Gradient imbalance
        gn1 = weight_params[0][1].grad.data.norm().item() ** 2
        gn2 = weight_params[1][1].grad.data.norm().item() ** 2
        grad_imbalances.append(gn2 - gn1)
        losses.append(loss.item())

        optimizer.step()

        if np.isnan(loss.item()):
            break

        # Activation pattern after the step
        curr_pattern = get_activation_pattern(model, X)
        frac_changed, n_neurons = hamming_distance(prev_pattern, curr_pattern)
        hamming_dists.append(frac_changed)
        neurons_changed_list.append(n_neurons)

        # Margin analysis
        pre_act = get_preactivations(model, X)
        margins = pre_act.abs()
        min_margins.append(margins.min().item())
        mean_margins.append(margins.mean().item())

        # Count distinct patterns (every 100 steps to save time)
        if step % 100 == 0:
            pattern_set = set()
            for i in range(curr_pattern.shape[0]):
                pattern_set.add(tuple(curr_pattern[i].tolist()))
            n_distinct_patterns.append((step, len(pattern_set)))

        prev_pattern = curr_pattern

    C_final = weight_params[1][1].data.norm().item()**2 - weight_params[0][1].data.norm().item()**2
    actual_drift = abs(C_final - C_init)
    S_measured = abs(sum(grad_imbalances))

    # Compute summary statistics
    hamming_arr = np.array(hamming_dists)
    neurons_arr = np.array(neurons_changed_list)
    zero_change_frac = np.mean(hamming_arr == 0)  # Fraction of steps with NO activation changes

    # Correlation between hamming distance and |grad_imbalance|
    if len(hamming_dists) == len(grad_imbalances):
        abs_imbalances = np.abs(grad_imbalances)
        if np.std(hamming_arr) > 1e-12 and np.std(abs_imbalances) > 1e-12:
            corr = np.corrcoef(hamming_arr, abs_imbalances)[0, 1]
        else:
            corr = 0.0
    else:
        corr = 0.0

    return {
        "lr": lr,
        "seed": seed,
        "actual_drift": float(actual_drift),
        "S_measured": float(S_measured),
        "final_loss": float(losses[-1]) if losses else float('inf'),
        "n_steps_completed": len(hamming_dists),
        # Activation dynamics
        "mean_hamming_dist": float(np.mean(hamming_arr)),
        "max_hamming_dist": float(np.max(hamming_arr)) if len(hamming_arr) > 0 else 0,
        "zero_change_fraction": float(zero_change_frac),
        "mean_neurons_changed": float(np.mean(neurons_arr)),
        "total_pattern_changes": float(np.sum(hamming_arr > 0)),
        # Margin analysis
        "min_margin_init": float(min_margins[0]) if min_margins else 0,
        "min_margin_final": float(min_margins[-1]) if min_margins else 0,
        "mean_margin_init": float(mean_margins[0]) if mean_margins else 0,
        "mean_margin_final": float(mean_margins[-1]) if mean_margins else 0,
        # Correlation
        "hamming_imbalance_correlation": float(corr),
        # Distinct patterns
        "distinct_patterns": n_distinct_patterns,
        # Downsampled trajectories for plotting
        "hamming_trajectory": [float(x) for x in hamming_arr[::10]],
        "margin_trajectory": [float(x) for x in np.array(min_margins)[::10]] if min_margins else [],
        "neurons_changed_trajectory": [float(x) for x in neurons_arr[::10]],
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E9: Activation Pattern Coupling Dynamics")
    print("=" * 70)
    print(f"Architecture: 2-layer ReLU, hidden={HIDDEN}, no bias")
    print(f"Data: Gaussian mixture (n={N_TRAIN}, d={INPUT_DIM}, K={OUTPUT_DIM})")
    print(f"Learning rates: {LRS}")
    print()

    t_start = time.time()
    USE_SEEDS = SEEDS[:3]

    all_results = {}
    for lr in LRS:
        print(f"\n--- lr = {lr} ---")
        results_for_lr = []
        for seed in USE_SEEDS:
            r = run_activation_tracking(lr, seed)
            results_for_lr.append(r)
            print(f"  seed={seed}: hamming_mean={r['mean_hamming_dist']:.6f}, "
                  f"zero_change={r['zero_change_fraction']:.1%}, "
                  f"neurons_changed_mean={r['mean_neurons_changed']:.1f}/{HIDDEN}, "
                  f"corr={r['hamming_imbalance_correlation']:.3f}")

        all_results[str(lr)] = results_for_lr

    # ---- Aggregate ----
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")

    summary_table = []
    for lr in LRS:
        results = all_results[str(lr)]
        avg_hamming = np.mean([r['mean_hamming_dist'] for r in results])
        avg_zero_frac = np.mean([r['zero_change_fraction'] for r in results])
        avg_neurons = np.mean([r['mean_neurons_changed'] for r in results])
        avg_corr = np.mean([r['hamming_imbalance_correlation'] for r in results])
        avg_margin_init = np.mean([r['min_margin_init'] for r in results])
        avg_margin_final = np.mean([r['min_margin_final'] for r in results])
        avg_S = np.mean([r['S_measured'] for r in results])

        row = {
            "lr": lr,
            "mean_hamming": float(avg_hamming),
            "zero_change_frac": float(avg_zero_frac),
            "mean_neurons_changed": float(avg_neurons),
            "hamming_imbalance_corr": float(avg_corr),
            "min_margin_init": float(avg_margin_init),
            "min_margin_final": float(avg_margin_final),
            "S_measured": float(avg_S),
        }
        summary_table.append(row)

        print(f"  lr={lr:.3f}: hamming={avg_hamming:.6f}, zero_steps={avg_zero_frac:.1%}, "
              f"neurons={avg_neurons:.1f}/{HIDDEN}, corr={avg_corr:.3f}, "
              f"margin_init={avg_margin_init:.4f}->final={avg_margin_final:.4f}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # ---- Key Interpretation ----
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    avg_overall_zero = np.mean([s["zero_change_frac"] for s in summary_table])
    avg_overall_neurons = np.mean([s["mean_neurons_changed"] for s in summary_table])
    avg_overall_corr = np.mean([s["hamming_imbalance_corr"] for s in summary_table])

    print(f"Average zero-change fraction across LRs: {avg_overall_zero:.1%}")
    print(f"Average neurons changed per step: {avg_overall_neurons:.1f} / {HIDDEN}")
    print(f"Average hamming-imbalance correlation: {avg_overall_corr:.3f}")

    if avg_overall_zero > 0.5:
        print("\n** ACTIVATION PATTERNS ARE HIGHLY STABLE **")
        print("   > 50% of steps have ZERO activation changes.")
        print("   Mode coupling from ReLU is WEAK.")
        print("   Theorem 5 likely applies to ReLU networks.")
    elif avg_overall_neurons < HIDDEN * 0.1:
        print("\n** ACTIVATION CHANGES ARE SPARSE **")
        print(f"   Only {avg_overall_neurons:.1f}/{HIDDEN} neurons change per step.")
        print("   Mode coupling is limited to a few dimensions.")
    else:
        print("\n** ACTIVATION PATTERNS CHANGE SIGNIFICANTLY **")
        print("   Mode coupling may be important.")
        print("   Theorem 5 may need mode-coupling correction for ReLU.")

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
        "summary_table": summary_table,
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ---- Figure 13: Activation Coupling ----
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Hamming distance over training for different LRs
    ax = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(LRS)))
    for i, lr in enumerate(LRS):
        traj = all_results[str(lr)][0]["hamming_trajectory"]
        steps = np.arange(len(traj)) * 10
        ax.plot(steps, traj, color=cmap[i], label=f'$\\eta$={lr}', alpha=0.8)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Activation Hamming distance')
    ax.set_title('(a) Activation pattern changes over training')
    ax.legend(fontsize=8)

    # Panel (b): Activation change rate vs learning rate
    ax = axes[0, 1]
    lrs_plot = [s["lr"] for s in summary_table]
    hamming_plot = [s["mean_hamming"] for s in summary_table]
    zero_frac_plot = [s["zero_change_frac"] for s in summary_table]

    ax2 = ax.twinx()
    l1 = ax.semilogx(lrs_plot, hamming_plot, 'o-', color='darkblue', markersize=8,
                      label='Mean Hamming distance')
    l2 = ax2.semilogx(lrs_plot, zero_frac_plot, 's--', color='darkred', markersize=8,
                       label='Zero-change fraction')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Mean Hamming distance', color='darkblue')
    ax2.set_ylabel('Zero-change fraction', color='darkred')
    ax.set_title('(b) Activation stability vs learning rate')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='center right')

    # Panel (c): Correlation between Hamming distance and |gradient imbalance|
    ax = axes[1, 0]
    corrs = [s["hamming_imbalance_corr"] for s in summary_table]
    ax.bar(range(len(LRS)), corrs, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(LRS)))
    ax.set_xticklabels([f'{lr}' for lr in LRS], rotation=45)
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Correlation')
    ax.set_title('(c) Hamming-imbalance correlation')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)

    # Panel (d): Minimum margin evolution
    ax = axes[1, 1]
    for i, lr in enumerate(LRS):
        traj = all_results[str(lr)][0]["margin_trajectory"]
        steps = np.arange(len(traj)) * 10
        ax.plot(steps, traj, color=cmap[i], label=f'$\\eta$={lr}', alpha=0.8)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Minimum activation margin')
    ax.set_title('(d) Activation margin evolution')
    ax.legend(fontsize=8)

    fig.suptitle('E9: Activation Pattern Coupling Dynamics — Is Mode Coupling Weak?',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig13_activation_coupling", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}")
