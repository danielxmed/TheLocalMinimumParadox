"""
Experiment: Falsification Tests for Theory A
Theory: Theory A -- Conservation Laws
Purpose: Actively try to break the theory by testing at assumption boundaries.
Predictions:
  1. Non-ReLU activations (tanh, GELU) should NOT conserve C_l (breaks A1)
  2. Networks with bias should NOT conserve C_l (breaks A1)
  3. Adam optimizer should NOT conserve C_l even for ReLU nobias (breaks gradient flow)
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_dataset,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "falsification"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_STEPS = 2000
LR = 0.001

def make_custom_model(input_dim, hidden, output_dim, activation="relu", bias=False):
    """Create 2-layer model with custom activation and bias setting."""
    layers = [nn.Linear(input_dim, hidden, bias=bias)]
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "tanh":
        layers.append(nn.Tanh())
    elif activation == "gelu":
        layers.append(nn.GELU())
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.1))
    layers.append(nn.Linear(hidden, output_dim, bias=bias))
    return nn.Sequential(*layers)

def measure_conservation(model, X, Y, lr, n_steps, optimizer_type="sgd"):
    """Measure conservation drift during training."""
    loss_fn = nn.CrossEntropyLoss()
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initial conservation quantities
    norms_init = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms_init.append(p.data.norm().item() ** 2)
    C_init = [norms_init[i+1] - norms_init[i] for i in range(len(norms_init)-1)]

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    # Final
    norms_final = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms_final.append(p.data.norm().item() ** 2)
    C_final = [norms_final[i+1] - norms_final[i] for i in range(len(norms_final)-1)]

    drifts = [abs(C_final[i] - C_init[i]) for i in range(len(C_init))]
    rel_drifts = [d / abs(c) if abs(c) > 1e-10 else d for d, c in zip(drifts, C_init)]
    final_loss = loss.item()

    return {
        "C_init": C_init, "C_final": C_final,
        "drifts": drifts, "rel_drifts": rel_drifts,
        "mean_drift": float(np.mean(drifts)),
        "mean_rel_drift": float(np.mean(rel_drifts)),
        "final_loss": final_loss,
    }

if __name__ == "__main__":
    print("="*60)
    print("Falsification Tests for Theory A")
    print("="*60)

    set_seed(42)
    X, Y, _, _ = make_dataset("gaussian_mixture", 200, 50, d=20, K=5, separation=2.0)

    TESTS = [
        # (name, activation, bias, optimizer, prediction)
        ("ReLU_nobias_sgd", "relu", False, "sgd", "CONSERVED (baseline)"),
        ("ReLU_bias_sgd", "relu", True, "sgd", "BROKEN (bias breaks symmetry)"),
        ("Tanh_nobias_sgd", "tanh", False, "sgd", "BROKEN (tanh not homogeneous)"),
        ("GELU_nobias_sgd", "gelu", False, "sgd", "BROKEN (GELU not homogeneous)"),
        ("Sigmoid_nobias_sgd", "sigmoid", False, "sgd", "BROKEN (sigmoid not homogeneous)"),
        ("LeakyReLU_nobias_sgd", "leaky_relu", False, "sgd", "CONSERVED? (leaky ReLU IS homogeneous)"),
        ("ReLU_nobias_adam", "relu", False, "adam", "BROKEN (Adam breaks gradient flow)"),
    ]

    results = {}
    print(f"\n{'Test':<25} {'Drift':>12} {'RelDrift':>12} {'Loss':>10} {'Prediction':>35} {'Match?':>8}")
    print("-" * 105)

    for name, act, bias, opt, prediction in TESTS:
        drifts_all = []
        for seed in SEEDS[:3]:
            set_seed(seed)
            model = make_custom_model(20, 64, 5, activation=act, bias=bias)
            r = measure_conservation(model, X, Y, LR, N_STEPS, optimizer_type=opt)
            drifts_all.append(r["mean_rel_drift"])

        mean_d = np.mean(drifts_all)
        std_d = np.std(drifts_all)
        conserved = mean_d < 0.01  # <1% drift = conserved
        expected_conserved = "CONSERVED" in prediction

        if conserved == expected_conserved:
            match = "YES"
        else:
            match = "**NO**"

        status = "CONSERVED" if conserved else "BROKEN"
        print(f"{name:<25} {mean_d:>12.6f} {std_d:>12.6f} {r['final_loss']:>10.4f} {prediction:>35} {match:>8}")

        results[name] = {
            "activation": act, "bias": bias, "optimizer": opt,
            "mean_rel_drift": float(mean_d), "std_drift": float(std_d),
            "conserved": conserved, "prediction_matched": conserved == expected_conserved,
            "prediction": prediction,
        }

    # Summary
    n_matched = sum(1 for r in results.values() if r["prediction_matched"])
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"Predictions matched: {n_matched}/{n_total}")
    print(f"{'='*60}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "lr": LR, "n_steps": N_STEPS,
    }
    save_results(results, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [t[0] for t in TESTS]
    drifts = [results[n]["mean_rel_drift"] for n in names]
    colors = ['green' if results[n]["conserved"] else 'red' for n in names]

    bars = ax.bar(range(len(names)), drifts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Relative Conservation Drift')
    ax.set_title('Falsification Tests: Conservation Law Breaking\n(Green = Conserved, Red = Broken)')
    ax.axhline(y=0.01, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1% threshold')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, "fig5f_falsification", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
