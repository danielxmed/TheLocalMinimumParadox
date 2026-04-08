"""
Experiment: Width Sweep for Mode Connectivity Phase Transition (HARD SETTING)
Theory: Theory B -- Percolation Phase Transition
Prediction: With harder data (low separation, more points), there should be
            a visible phase transition in barrier height around the interpolation threshold.
Date: 2026-04-07
Version: 2 (harder data setting to find the transition)
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

EXPERIMENT_NAME = "width_connectivity_hard"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HARD setting: overlapping classes, more data, fewer training steps
DATA_CONFIG = {"name": "gaussian_mixture", "n_train": 500, "n_test": 100,
               "d": 10, "K": 3, "separation": 0.5}
# Interpolation threshold: n*K / (d+K) = 500*3 / 13 ≈ 115 neurons
# So the transition should be around width 50-200

LR = 0.01
N_STEPS = 3000
N_INTERP = 21
WIDTHS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]

def train_and_measure(width, seed, data_config):
    set_seed(seed)
    model = make_model("mlp_2layer", data_config["d"], data_config["K"], hidden=width)
    X_train, Y_train, X_test, Y_test = make_dataset(
        data_config["name"], data_config["n_train"], data_config["n_test"],
        d=data_config["d"], K=data_config["K"], separation=data_config["separation"]
    )
    history = train(model, X_train, Y_train, lr=LR, n_steps=N_STEPS, log_every=N_STEPS)

    with torch.no_grad():
        train_loss = nn.CrossEntropyLoss()(model(X_train), Y_train).item()
        pred_tr = model(X_train).argmax(1)
        train_acc = (pred_tr == Y_train).float().mean().item()
        pred_te = model(X_test).argmax(1)
        test_acc = (pred_te == Y_test).float().mean().item()

    return model, X_train, Y_train, train_loss, train_acc, test_acc

def compute_barrier(m1, m2, X, Y):
    loss_fn = nn.CrossEntropyLoss()
    interp = linear_interpolation_loss(m1, m2, X, Y, loss_fn, n_points=N_INTERP)
    losses = [l for _, l, _ in interp]
    endpoint_max = max(losses[0], losses[-1])
    barrier = max(losses) - endpoint_max
    return barrier, losses

if __name__ == "__main__":
    print("="*60)
    print("Experiment: Width Sweep (HARD SETTING)")
    print(f"Data: n={DATA_CONFIG['n_train']}, d={DATA_CONFIG['d']}, K={DATA_CONFIG['K']}, sep={DATA_CONFIG['separation']}")
    print(f"Interpolation threshold estimate: ~{DATA_CONFIG['n_train']*DATA_CONFIG['K'] // (DATA_CONFIG['d']+DATA_CONFIG['K'])} neurons")
    print("="*60)

    all_results = {}
    t_start = time.time()

    for width in WIDTHS:
        n_params = count_parameters(make_model("mlp_2layer", DATA_CONFIG["d"], DATA_CONFIG["K"], hidden=width))
        print(f"\n--- Width={width} ({n_params} params) ---")

        models = []
        X_data, Y_data = None, None
        losses_list, taccs, teaccs = [], [], []

        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)
            m, X, Y, loss, ta, tea = train_and_measure(width, seed, DATA_CONFIG)
            models.append(m)
            X_data, Y_data = X, Y
            losses_list.append(loss)
            taccs.append(ta)
            teaccs.append(tea)
            print(f"loss={loss:.4f} tr_acc={ta:.3f} te_acc={tea:.3f}")

        # Pairwise barriers
        barriers = []
        for i in range(len(SEEDS)):
            for j in range(i+1, len(SEEDS)):
                b, _ = compute_barrier(models[i], models[j], X_data, Y_data)
                barriers.append(b)

        mb = np.mean(barriers)
        sb = np.std(barriers)
        zf = np.mean([1.0 if b < 0.01 else 0.0 for b in barriers])
        print(f"  barrier={mb:.4f}+/-{sb:.4f} connected_frac={zf:.2f} tr_acc={np.mean(taccs):.3f}")

        all_results[width] = {
            "width": width, "n_params": n_params,
            "mean_barrier": float(mb), "std_barrier": float(sb),
            "zero_barrier_frac": float(zf),
            "barriers": [float(b) for b in barriers],
            "mean_train_loss": float(np.mean(losses_list)),
            "mean_train_acc": float(np.mean(taccs)),
            "mean_test_acc": float(np.mean(teaccs)),
        }

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary
    print("\n--- PHASE TRANSITION SUMMARY (HARD SETTING) ---")
    print(f"{'Width':>8} {'Params':>8} {'Barrier':>10} {'Connected':>12} {'TrainAcc':>10} {'TestAcc':>10}")
    print("-" * 65)
    for w in sorted(all_results.keys()):
        r = all_results[w]
        print(f"{w:>8} {r['n_params']:>8} {r['mean_barrier']:>10.4f} {r['zero_barrier_frac']:>12.2f} {r['mean_train_acc']:>10.3f} {r['mean_test_acc']:>10.3f}")

    # Identify transition
    widths_sorted = sorted(all_results.keys())
    for i in range(len(widths_sorted) - 1):
        w1, w2 = widths_sorted[i], widths_sorted[i+1]
        if all_results[w1]["zero_barrier_frac"] < 0.5 and all_results[w2]["zero_barrier_frac"] >= 0.5:
            print(f"\n** TRANSITION DETECTED between width {w1} and {w2} **")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "theory": "percolation_phase_transition",
        "version": 2, "date": "2026-04-07", "hardware": get_hardware_info(),
        "seeds": SEEDS, "n_steps": N_STEPS, "lr": LR, "data_config": DATA_CONFIG,
        "widths": WIDTHS,
    }
    save_results({str(k): v for k, v in all_results.items()}, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ws = sorted(all_results.keys())
    ax1.errorbar(ws, [all_results[w]["mean_barrier"] for w in ws],
                 yerr=[all_results[w]["std_barrier"] for w in ws],
                 fmt='o-', capsize=4, color='darkblue', markersize=5)
    ax1.set_xlabel('Network Width $m$')
    ax1.set_ylabel('Mean Loss Barrier')
    ax1.set_title(f'Linear Interpolation Barrier\n(n={DATA_CONFIG["n_train"]}, d={DATA_CONFIG["d"]}, K={DATA_CONFIG["K"]}, sep={DATA_CONFIG["separation"]})')
    ax1.set_xscale('log')

    ax2.plot(ws, [all_results[w]["zero_barrier_frac"] for w in ws], 'o-', color='darkgreen', markersize=5)
    ax2.plot(ws, [all_results[w]["mean_train_acc"] for w in ws], 's--', color='darkred', markersize=4, label='Train Acc')
    ax2.set_xlabel('Network Width $m$')
    ax2.set_ylabel('Fraction / Accuracy')
    ax2.set_title('Connectivity Probability & Training Accuracy')
    ax2.set_xscale('log')
    ax2.legend(['Connected Frac', 'Train Acc'])
    ax2.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    save_figure(fig, "fig3c_width_transition_hard", FIGURE_DIR)
    plt.close()
    print(f"\nResults saved to {OUTPUT_DIR}")
