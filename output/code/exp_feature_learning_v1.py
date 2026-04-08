"""
Experiment: Feature Learning vs Lazy Training (NTK alignment)
Theory: Context for all theories -- measures whether networks learn features or stay lazy
Prediction: Narrow networks change their NTK more (feature learning).
            Wide networks change NTK less (lazy regime).
            The NTK change should correlate with conservation law breaking.
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, compute_ntk, linear_cka,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "feature_learning"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Small data subset for NTK tractability
N_DATA = 100  # NTK is n x n, so keep small
N_STEPS = 2000
LR = 0.01

CONFIGS = [
    {"name": "narrow_16", "hidden": 16},
    {"name": "medium_64", "hidden": 64},
    {"name": "wide_256", "hidden": 256},
    {"name": "very_wide_512", "hidden": 512},
]

def run_ntk_experiment(hidden, seed):
    set_seed(seed)
    model = make_model("mlp_2layer", 20, 5, hidden=hidden)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_DATA, n_test=30, d=20, K=5, separation=1.5)

    # NTK at initialization
    K_init = compute_ntk(model, X[:N_DATA])

    # Record conservation
    norms_init = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms_init.append(p.data.norm().item() ** 2)

    # Train
    history = train(model, X, Y, lr=LR, n_steps=N_STEPS, log_every=N_STEPS)

    # NTK after training
    K_final = compute_ntk(model, X[:N_DATA])

    # Measure NTK change
    K_init_np = K_init.detach().numpy()
    K_final_np = K_final.detach().numpy()

    # CKA between initial and final NTK
    cka = linear_cka(torch.tensor(K_init_np), torch.tensor(K_final_np))

    # Frobenius distance
    frob_change = np.linalg.norm(K_final_np - K_init_np) / np.linalg.norm(K_init_np)

    # Conservation drift
    norms_final = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms_final.append(p.data.norm().item() ** 2)
    drift = abs((norms_final[1] - norms_final[0]) - (norms_init[1] - norms_init[0]))

    return {
        "cka": float(cka),
        "frob_change": float(frob_change),
        "conservation_drift": float(drift),
        "final_loss": history[-1]["loss"],
    }

if __name__ == "__main__":
    print("Feature Learning vs Lazy Training")
    print(f"NTK computed on {N_DATA} data points")
    t_start = time.time()

    all_results = {}
    for config in CONFIGS:
        print(f"\nHidden={config['hidden']}:")
        seed_results = []
        for seed in SEEDS[:3]:
            print(f"  Seed {seed}...", end=" ", flush=True)
            r = run_ntk_experiment(config["hidden"], seed)
            print(f"CKA={r['cka']:.4f}, frob_change={r['frob_change']:.4f}, C_drift={r['conservation_drift']:.6f}")
            seed_results.append(r)

        mean_cka = np.mean([r["cka"] for r in seed_results])
        mean_frob = np.mean([r["frob_change"] for r in seed_results])
        mean_drift = np.mean([r["conservation_drift"] for r in seed_results])

        all_results[config["name"]] = {
            "hidden": config["hidden"],
            "mean_cka": float(mean_cka),
            "mean_frob_change": float(mean_frob),
            "mean_conservation_drift": float(mean_drift),
            "seeds": seed_results,
        }
        print(f"  Mean: CKA={mean_cka:.4f}, frob={mean_frob:.4f}, drift={mean_drift:.6f}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary
    print("\n--- FEATURE LEARNING SUMMARY ---")
    print(f"{'Config':<15} {'CKA(K0,KT)':>12} {'Frob Change':>12} {'C Drift':>12} {'Regime':>12}")
    print("-" * 65)
    for name, data in all_results.items():
        cka = data["mean_cka"]
        regime = "LAZY" if cka > 0.95 else ("FEATURE" if cka < 0.80 else "TRANSITION")
        print(f"{name:<15} {cka:>12.4f} {data['mean_frob_change']:>12.4f} {data['mean_conservation_drift']:>12.6f} {regime:>12}")

    # Save
    config_save = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "n_data": N_DATA, "lr": LR, "n_steps": N_STEPS,
    }
    save_results(all_results, OUTPUT_DIR, config_save)

    # Figure
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    widths = [all_results[c["name"]]["hidden"] for c in CONFIGS]
    ckas = [all_results[c["name"]]["mean_cka"] for c in CONFIGS]
    frobs = [all_results[c["name"]]["mean_frob_change"] for c in CONFIGS]
    drifts = [all_results[c["name"]]["mean_conservation_drift"] for c in CONFIGS]

    ax1.plot(widths, ckas, 'o-', color='darkblue', markersize=6, label='CKA($K_0$, $K_T$)')
    ax1.set_xlabel('Hidden Width $m$')
    ax1.set_ylabel('CKA (Initial vs Final NTK)')
    ax1.set_title('(a) NTK Alignment: Lazy vs Feature Learning')
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Lazy threshold')
    ax1.set_xscale('log')
    ax1.legend()

    ax2.plot(widths, drifts, 'o-', color='darkred', markersize=6)
    ax2.set_xlabel('Hidden Width $m$')
    ax2.set_ylabel('Conservation Drift')
    ax2.set_title('(b) Conservation Drift vs Width')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    fig.suptitle('Feature Learning and Conservation Laws', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig7_feature_learning", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
