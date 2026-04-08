"""
Experiment: Conservation Law Breaking at Edge of Stability
Theory: Theory A (pivoted) -- Structured breaking as optimization mechanism
Prediction: (1) Conservation drift grows at EoS onset
            (2) Breaking drives layer norms toward equality (self-balancing)
            (3) Self-balancing correlates with decreased sharpness
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info, top_hessian_eigenvalue
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "conservation_breaking_eos"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key insight: use LARGE learning rate + bias-free network to see both EoS AND conservation breaking
# MNIST subset to create a harder problem that shows EoS dynamics
N_TRAIN = 500
LR = 0.5  # Large enough for EoS on small MNIST
HIDDEN = 64
N_STEPS = 3000
LOG_EVERY = 5

def get_layer_norms_sq(model):
    """Get squared Frobenius norms of weight matrices."""
    norms = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            norms.append(p.data.norm().item() ** 2)
    return norms

def compute_balancedness(model):
    """Compute C_l = ||W_{l+1}||^2 - ||W_l||^2 for all layer pairs."""
    norms = get_layer_norms_sq(model)
    return [norms[l+1] - norms[l] for l in range(len(norms)-1)]

def compute_imbalance(model):
    """Compute total imbalance: sum_l |C_l|."""
    C = compute_balancedness(model)
    return sum(abs(c) for c in C)

def run_experiment(seed, use_bias=False):
    set_seed(seed)

    arch = "mlp_2layer_nobias" if not use_bias else "mlp_2layer"
    model = make_model(arch, 784, 10, hidden=HIDDEN)

    X_train, Y_train, X_test, Y_test = make_dataset("mnist", n_train=N_TRAIN, n_test=200)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Record initial state
    C_init = compute_balancedness(model)
    norms_init = get_layer_norms_sq(model)

    records = []
    for step in range(1, N_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, Y_train)
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                curr_loss = loss_fn(model(X_train), Y_train).item()
                pred = model(X_train).argmax(1)
                acc = (pred == Y_train).float().mean().item()

            C = compute_balancedness(model)
            norms = get_layer_norms_sq(model)
            imbalance = compute_imbalance(model)

            # Conservation drift
            drifts = [C[l] - C_init[l] for l in range(len(C))]

            # Compute lambda_max every 50 steps (expensive)
            lmax = None
            if step % 50 == 0 or step == 1:
                lmax = top_hessian_eigenvalue(model, X_train, Y_train, loss_fn, n_iter=20)

            records.append({
                "step": step, "loss": curr_loss, "acc": acc,
                "C": C, "norms": norms, "drifts": drifts,
                "imbalance": imbalance, "lambda_max": lmax,
            })

            if step % 500 == 0:
                drift_str = ", ".join(f"{d:.4f}" for d in drifts)
                lmax_str = f"{lmax:.2f}" if lmax else "N/A"
                print(f"  step {step:>5} | loss={curr_loss:.4f} | acc={acc:.3f} | imbalance={imbalance:.4f} | lmax={lmax_str} | drifts=[{drift_str}]")

    return {
        "seed": seed, "use_bias": use_bias,
        "C_init": C_init, "norms_init": norms_init,
        "records": records,
        "eos_threshold": 2.0 / LR,
    }

if __name__ == "__main__":
    print("="*70)
    print("Conservation Law Breaking at Edge of Stability")
    print(f"lr={LR}, 2/eta={2/LR:.1f}, hidden={HIDDEN}, n_train={N_TRAIN}")
    print("="*70)

    all_results = {"nobias": [], "bias": []}
    t_start = time.time()

    # Run without bias (conservation laws hold initially)
    for seed in SEEDS[:3]:
        print(f"\n--- No Bias, Seed {seed} ---")
        r = run_experiment(seed, use_bias=False)
        all_results["nobias"].append(r)

    # Run with bias (control: conservation laws may not hold)
    for seed in SEEDS[:3]:
        print(f"\n--- With Bias, Seed {seed} ---")
        r = run_experiment(seed, use_bias=True)
        all_results["bias"].append(r)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Analysis: Does conservation breaking correlate with EoS?
    print("\n--- CONSERVATION BREAKING ANALYSIS ---")
    for config_name, results in all_results.items():
        print(f"\n{config_name.upper()}:")
        for r in results:
            records = r["records"]
            # Find when lambda_max is closest to 2/eta
            lmax_records = [(rec["step"], rec["lambda_max"]) for rec in records if rec["lambda_max"] is not None]
            if lmax_records:
                max_lmax = max(l for _, l in lmax_records)
                imbalance_init = records[0]["imbalance"]
                imbalance_final = records[-1]["imbalance"]
                final_drift = sum(abs(d) for d in records[-1]["drifts"])
                print(f"  Seed {r['seed']}: max_lmax={max_lmax:.2f}, imbalance {imbalance_init:.4f} -> {imbalance_final:.4f}, total_drift={final_drift:.4f}")

    # Save results
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "lr": LR, "hidden": HIDDEN, "n_train": N_TRAIN, "n_steps": N_STEPS,
    }

    # Compact for JSON
    results_save = {}
    for config_name, results in all_results.items():
        results_save[config_name] = [{
            "seed": r["seed"], "C_init": r["C_init"],
            "steps": [rec["step"] for rec in r["records"]],
            "losses": [rec["loss"] for rec in r["records"]],
            "imbalances": [rec["imbalance"] for rec in r["records"]],
            "drifts": [rec["drifts"] for rec in r["records"]],
            "lambda_max": [(rec["step"], rec["lambda_max"]) for rec in r["records"] if rec["lambda_max"] is not None],
        } for r in results]

    save_results(results_save, OUTPUT_DIR, config)

    # Generate THE key figure: conservation drift vs training dynamics
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Use first seed of nobias
    r = all_results["nobias"][0]
    records = r["records"]
    steps = [rec["step"] for rec in records]
    losses = [rec["loss"] for rec in records]
    imbalances = [rec["imbalance"] for rec in records]
    lmax_steps = [rec["step"] for rec in records if rec["lambda_max"] is not None]
    lmax_vals = [rec["lambda_max"] for rec in records if rec["lambda_max"] is not None]
    drifts_0 = [abs(rec["drifts"][0]) for rec in records]

    # Panel (a): Loss and lambda_max
    ax1 = axes[0][0]
    ax1.plot(steps, losses, 'b-', linewidth=0.8, label='Loss')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_title('(a) Loss and Sharpness')
    ax1b = ax1.twinx()
    ax1b.plot(lmax_steps, lmax_vals, 'r-', linewidth=1, label='$\\lambda_{max}$')
    ax1b.axhline(y=2/LR, color='red', linestyle='--', alpha=0.5, label=f'$2/\\eta={2/LR:.0f}$')
    ax1b.set_ylabel('$\\lambda_{max}$', color='red')
    ax1b.legend(loc='upper right', fontsize=9)

    # Panel (b): Conservation drift |Delta C|
    ax2 = axes[0][1]
    ax2.plot(steps, drifts_0, 'darkgreen', linewidth=1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('$|\\Delta C_1|$')
    ax2.set_title('(b) Conservation Law Drift (No Bias)')

    # Panel (c): Imbalance
    ax3 = axes[1][0]
    for i, r in enumerate(all_results["nobias"]):
        recs = r["records"]
        ax3.plot([rec["step"] for rec in recs], [rec["imbalance"] for rec in recs],
                 alpha=0.7, linewidth=1, label=f'Seed {SEEDS[i]}')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Total Imbalance $\\sum_l |C_l|$')
    ax3.set_title('(c) Self-Balancing Dynamics (No Bias)')
    ax3.legend(fontsize=9)

    # Panel (d): Comparison nobias vs bias
    ax4 = axes[1][1]
    for r in all_results["nobias"][:1]:
        recs = r["records"]
        ax4.plot([rec["step"] for rec in recs], [rec["imbalance"] for rec in recs],
                 'b-', linewidth=1, label='No Bias')
    for r in all_results["bias"][:1]:
        recs = r["records"]
        ax4.plot([rec["step"] for rec in recs], [rec["imbalance"] for rec in recs],
                 'r-', linewidth=1, label='With Bias')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Total Imbalance')
    ax4.set_title('(d) Bias vs No Bias Comparison')
    ax4.legend(fontsize=9)

    fig.suptitle('Conservation Law Breaking at Edge of Stability\n(MNIST, lr=0.5, hidden=64)', y=1.02, fontsize=14)
    plt.tight_layout()
    save_figure(fig, "fig5d_conservation_breaking_eos", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
