"""
Experiment: Conservation Law Breaking During Grokking
Theory: Theory A -- Does conservation drift trigger the grokking transition?
Prediction: If conservation law breaking is the mechanism that triggers delayed
            generalization (grokking), we expect a sharp increase in conservation
            drift at or just before the grokking transition point.
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, top_hessian_eigenvalue,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "grokking_conservation"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grokking parameters
P = 97           # Prime for modular arithmetic
FRAC_TRAIN = 0.3 # Fraction of data for training
N_STEPS = 50000  # Grokking typically happens at ~10K-30K steps
HIDDEN = 128     # Width
LR = 0.01        # Learning rate for SGD
WEIGHT_DECAY = 0.01  # Standard for grokking experiments
LOG_INTERVAL = 100   # Log every N steps
HESSIAN_INTERVAL = 2000  # Hessian eigenvalue every N steps (expensive)


def make_modular_addition_dataset(p, frac_train, seed):
    """Create modular addition dataset: (a + b) mod p.
    Input: one-hot encoding of (a, b), dimension 2p.
    Output: class label in {0, ..., p-1}.
    """
    set_seed(seed)

    # Generate all p^2 pairs
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    y = (a + b) % p

    # One-hot encode inputs: concatenate one-hot(a) and one-hot(b)
    x_a = torch.zeros(p * p, p)
    x_a.scatter_(1, a.unsqueeze(1), 1.0)
    x_b = torch.zeros(p * p, p)
    x_b.scatter_(1, b.unsqueeze(1), 1.0)
    x = torch.cat([x_a, x_b], dim=1)  # shape: (p^2, 2p)

    # Random train/test split
    n_total = p * p
    n_train = int(frac_train * n_total)
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = x[train_idx].to(DEVICE)
    Y_train = y[train_idx].to(DEVICE)
    X_test = x[test_idx].to(DEVICE)
    Y_test = y[test_idx].to(DEVICE)

    return X_train, Y_train, X_test, Y_test


def get_weight_norms(model):
    """Get squared Frobenius norms of weight layers."""
    norms = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            norms.append(module.weight.data.norm().item() ** 2)
    return norms


def compute_accuracy(model, X, Y):
    """Compute classification accuracy."""
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        return (preds == Y).float().mean().item()


def run_grokking_experiment(seed, use_weight_decay=True):
    """Run a single grokking experiment tracking conservation drift."""
    print(f"\n  Seed {seed}, weight_decay={'ON' if use_weight_decay else 'OFF'}")

    X_train, Y_train, X_test, Y_test = make_modular_addition_dataset(P, FRAC_TRAIN, seed)

    set_seed(seed + 1000)  # Different seed for model init
    model = make_model("mlp_2layer_nobias", input_dim=2*P, output_dim=P, hidden=HIDDEN)

    loss_fn = nn.CrossEntropyLoss()
    wd = WEIGHT_DECAY if use_weight_decay else 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=wd)

    # Record initial conservation quantity
    norms_init = get_weight_norms(model)
    C_init = norms_init[1] - norms_init[0]

    # Tracking arrays
    steps = []
    train_losses = []
    train_accs = []
    test_accs = []
    conservation_drifts = []
    weight_norms_history = []
    hessian_eigenvalues = []
    hessian_steps = []

    for step in range(N_STEPS + 1):
        if step % LOG_INTERVAL == 0:
            with torch.no_grad():
                logits = model(X_train)
                loss_val = loss_fn(logits, Y_train).item()
                train_acc = compute_accuracy(model, X_train, Y_train)
                test_acc = compute_accuracy(model, X_test, Y_test)

            # Conservation drift
            norms = get_weight_norms(model)
            C_current = norms[1] - norms[0]
            drift = abs(C_current - C_init)

            steps.append(step)
            train_losses.append(loss_val)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            conservation_drifts.append(drift)
            weight_norms_history.append(norms)

            if step % (LOG_INTERVAL * 10) == 0:
                print(f"    Step {step:>6d}: loss={loss_val:.4f}, "
                      f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                      f"drift={drift:.6f}")

        # Hessian eigenvalue (expensive, do less frequently)
        if step % HESSIAN_INTERVAL == 0 and step > 0:
            lam_max = top_hessian_eigenvalue(model, X_train, Y_train, loss_fn, n_iter=30)
            hessian_eigenvalues.append(lam_max)
            hessian_steps.append(step)

        # Training step
        if step < N_STEPS:
            optimizer.zero_grad()
            out = model(X_train)
            loss = loss_fn(out, Y_train)
            loss.backward()
            optimizer.step()

    # Detect grokking point: first step where test_acc > 0.9
    grokking_step = None
    for i, (s, ta) in enumerate(zip(steps, test_accs)):
        if ta > 0.9:
            grokking_step = s
            break

    # Detect memorization point: first step where train_acc > 0.99
    memorization_step = None
    for i, (s, ta) in enumerate(zip(steps, train_accs)):
        if ta > 0.99:
            memorization_step = s
            break

    print(f"    Memorization at step: {memorization_step}")
    print(f"    Grokking at step: {grokking_step}")

    return {
        "seed": seed,
        "weight_decay": wd,
        "steps": steps,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "conservation_drifts": conservation_drifts,
        "hessian_eigenvalues": hessian_eigenvalues,
        "hessian_steps": hessian_steps,
        "grokking_step": grokking_step,
        "memorization_step": memorization_step,
        "C_init": C_init,
        "weight_norms_history": [[float(n) for n in norms] for norms in weight_norms_history],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  CONSERVATION LAW BREAKING DURING GROKKING")
    print(f"  Modular addition mod {P}, train fraction={FRAC_TRAIN}")
    print(f"  Architecture: mlp_2layer_nobias, hidden={HIDDEN}")
    print(f"  SGD lr={LR}, weight_decay={WEIGHT_DECAY}, steps={N_STEPS}")
    print("=" * 70)

    t_start = time.time()

    # Run with weight decay (standard grokking)
    results_wd = {}
    for seed in SEEDS[:3]:  # 3 seeds to save time (50K steps each)
        res = run_grokking_experiment(seed, use_weight_decay=True)
        results_wd[f"seed_{seed}"] = res

    # Run one without weight decay (control)
    print("\n\n--- Control: NO weight decay ---")
    res_no_wd = run_grokking_experiment(SEEDS[0], use_weight_decay=False)

    t_total = time.time() - t_start
    print(f"\n\nTotal runtime: {t_total:.1f}s")

    # Compile results
    all_results = {
        "with_weight_decay": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "weight_norms_history"}
                              for k, v in results_wd.items()},
        "no_weight_decay": {kk: vv for kk, vv in res_no_wd.items()
                           if kk != "weight_norms_history"},
    }

    # Summary statistics
    grokking_steps = [v["grokking_step"] for v in results_wd.values() if v["grokking_step"]]
    memo_steps = [v["memorization_step"] for v in results_wd.values() if v["memorization_step"]]

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    if grokking_steps:
        print(f"  Grokking steps (with WD): {grokking_steps}")
        print(f"  Mean grokking step: {np.mean(grokking_steps):.0f}")
    else:
        print("  No grokking observed in {N_STEPS} steps")
    if memo_steps:
        print(f"  Memorization steps: {memo_steps}")
    print(f"  Control (no WD) grokking: {res_no_wd['grokking_step']}")

    # Analyze drift at grokking transition
    print(f"\n  Conservation drift analysis:")
    for seed_key, res in results_wd.items():
        steps = res["steps"]
        drifts = res["conservation_drifts"]
        grok = res["grokking_step"]
        if grok and len(steps) > 10:
            # Find drift just before and after grokking
            pre_grok_drifts = [d for s, d in zip(steps, drifts) if s < grok and s > grok - 5000]
            post_grok_drifts = [d for s, d in zip(steps, drifts) if s > grok and s < grok + 5000]
            if pre_grok_drifts and post_grok_drifts:
                print(f"    {seed_key}: pre-grok drift={np.mean(pre_grok_drifts):.6f}, "
                      f"post-grok drift={np.mean(post_grok_drifts):.6f}, "
                      f"ratio={np.mean(post_grok_drifts)/np.mean(pre_grok_drifts):.2f}x")

    # Save results
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "p": P, "frac_train": FRAC_TRAIN,
        "n_steps": N_STEPS, "hidden": HIDDEN, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "seeds": SEEDS[:3],
        "total_runtime_s": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    setup_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use first seed with weight decay for main plots
    first_key = list(results_wd.keys())[0]
    main_res = results_wd[first_key]
    steps = main_res["steps"]

    # Panel (a): Training dynamics (loss + accuracies)
    ax = axes[0, 0]
    ax.plot(steps, main_res["train_losses"], 'b-', linewidth=1, label='Train Loss')
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    ax2.plot(steps, main_res["train_accs"], 'g--', linewidth=1, alpha=0.7, label='Train Acc')
    ax2.plot(steps, main_res["test_accs"], 'r-', linewidth=1.5, label='Test Acc')
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.05, 1.05)
    if main_res["grokking_step"]:
        ax.axvline(x=main_res["grokking_step"], color='orange', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Grokking ({main_res["grokking_step"]})')
    if main_res["memorization_step"]:
        ax.axvline(x=main_res["memorization_step"], color='purple', linestyle=':',
                   linewidth=1, alpha=0.5, label=f'Memorization ({main_res["memorization_step"]})')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
    ax.set_title('(a) Grokking Dynamics')

    # Panel (b): Conservation drift over time
    ax = axes[0, 1]
    for seed_key, res in results_wd.items():
        ax.plot(res["steps"], res["conservation_drifts"], linewidth=1, alpha=0.7,
                label=f'{seed_key} (WD)')
    ax.plot(res_no_wd["steps"], res_no_wd["conservation_drifts"], 'k--',
            linewidth=1.5, alpha=0.8, label='No WD (control)')
    if main_res["grokking_step"]:
        ax.axvline(x=main_res["grokking_step"], color='orange', linestyle='--',
                   linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Conservation Drift $|\\Delta C|$')
    ax.set_title('(b) Conservation Drift During Grokking')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel (c): Drift rate (derivative of drift)
    ax = axes[1, 0]
    drift_arr = np.array(main_res["conservation_drifts"])
    steps_arr = np.array(steps)
    if len(drift_arr) > 5:
        # Smooth derivative via finite differences with window
        window = 10
        drift_rate = np.zeros_like(drift_arr)
        for i in range(window, len(drift_arr)):
            drift_rate[i] = (drift_arr[i] - drift_arr[i - window]) / (steps_arr[i] - steps_arr[i - window] + 1e-10)
        ax.plot(steps_arr[window:], drift_rate[window:], 'darkblue', linewidth=1)
        if main_res["grokking_step"]:
            ax.axvline(x=main_res["grokking_step"], color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.7, label='Grokking')
        if main_res["memorization_step"]:
            ax.axvline(x=main_res["memorization_step"], color='purple', linestyle=':',
                       linewidth=1, alpha=0.5, label='Memorization')
    ax.set_xlabel('Step')
    ax.set_ylabel('Drift Rate $d|\\Delta C|/dt$')
    ax.set_title('(c) Conservation Drift Rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel (d): Hessian top eigenvalue
    ax = axes[1, 1]
    if main_res["hessian_eigenvalues"]:
        ax.plot(main_res["hessian_steps"], main_res["hessian_eigenvalues"],
                'o-', color='darkred', markersize=4, linewidth=1)
        # EoS threshold
        ax.axhline(y=2.0/LR, color='gray', linestyle='--', linewidth=1,
                    alpha=0.5, label=f'$2/\\eta = {2.0/LR:.0f}$')
        if main_res["grokking_step"]:
            ax.axvline(x=main_res["grokking_step"], color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.7, label='Grokking')
    ax.set_xlabel('Step')
    ax.set_ylabel('$\\lambda_{\\max}(H)$')
    ax.set_title('(d) Top Hessian Eigenvalue')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Conservation Law Breaking During Grokking (mod {P})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig7_grokking_conservation", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}")
    print(f"Results saved to {OUTPUT_DIR}")
