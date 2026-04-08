"""
Experiment: Conservation Law Breaking During Grokking (v2 -- revised hyperparameters)
Theory: Theory A -- Does conservation drift correlate with the grokking transition?
Prediction: Conservation drift should show characteristic behavior at grokking onset.
Date: 2026-04-07
Note: v1 failed to produce grokking (lr=0.01 too low, no memorization).
      v2 uses larger lr, allows bias for comparison, smaller prime for faster grokking.
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

EXPERIMENT_NAME = "grokking_conservation_v2"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

P = 23             # Smaller prime for faster grokking
FRAC_TRAIN = 0.5   # 50% train (more data helps memorization)
N_STEPS = 40000
HIDDEN = 256        # Wider network for modular arithmetic
LOG_INTERVAL = 50
HESSIAN_INTERVAL = 2000


def make_modular_addition_dataset(p, frac_train, seed):
    """Create (a + b) mod p with one-hot encoding."""
    set_seed(seed)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    y = (a + b) % p

    x_a = torch.zeros(p * p, p)
    x_a.scatter_(1, a.unsqueeze(1), 1.0)
    x_b = torch.zeros(p * p, p)
    x_b.scatter_(1, b.unsqueeze(1), 1.0)
    x = torch.cat([x_a, x_b], dim=1)

    n_total = p * p
    n_train = int(frac_train * n_total)
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return (x[train_idx].to(DEVICE), y[train_idx].to(DEVICE),
            x[test_idx].to(DEVICE), y[test_idx].to(DEVICE))


def get_weight_norms(model):
    """Get squared Frobenius norms of weight layers."""
    norms = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            norms.append(module.weight.data.norm().item() ** 2)
    return norms


def compute_accuracy(model, X, Y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == Y).float().mean().item()


def run_grokking(seed, lr, weight_decay, use_bias, optimizer_type="sgd"):
    """Run a single grokking experiment."""
    label = f"lr{lr}_wd{weight_decay}_{'bias' if use_bias else 'nobias'}_{optimizer_type}"
    print(f"\n  Config: {label}, seed={seed}")

    X_tr, Y_tr, X_te, Y_te = make_modular_addition_dataset(P, FRAC_TRAIN, seed)

    set_seed(seed + 1000)
    arch = "mlp_2layer" if use_bias else "mlp_2layer_nobias"
    model = make_model(arch, input_dim=2*P, output_dim=P, hidden=HIDDEN)

    loss_fn = nn.CrossEntropyLoss()
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initial conservation quantity (mean over all L-1 pairs)
    norms_init = get_weight_norms(model)
    C_inits = [norms_init[l+1] - norms_init[l] for l in range(len(norms_init)-1)]

    steps, train_losses, train_accs, test_accs = [], [], [], []
    conservation_drifts, total_weight_norms = [], []
    hessian_eigenvalues, hessian_steps = [], []

    for step in range(N_STEPS + 1):
        if step % LOG_INTERVAL == 0:
            with torch.no_grad():
                loss_val = loss_fn(model(X_tr), Y_tr).item()
                tr_acc = compute_accuracy(model, X_tr, Y_tr)
                te_acc = compute_accuracy(model, X_te, Y_te)

            norms = get_weight_norms(model)
            C_currents = [norms[l+1] - norms[l] for l in range(len(norms)-1)]
            drift = float(np.mean([abs(c - c0) for c, c0 in zip(C_currents, C_inits)]))
            total_norm = sum(norms)

            steps.append(step)
            train_losses.append(loss_val)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            conservation_drifts.append(drift)
            total_weight_norms.append(total_norm)

            if step % (LOG_INTERVAL * 20) == 0:
                print(f"    Step {step:>6d}: loss={loss_val:.4f}, "
                      f"tr={tr_acc:.3f}, te={te_acc:.3f}, drift={drift:.4f}, norm={total_norm:.1f}")

        if step % HESSIAN_INTERVAL == 0 and step > 0:
            lam = top_hessian_eigenvalue(model, X_tr, Y_tr, loss_fn, n_iter=20)
            hessian_eigenvalues.append(lam)
            hessian_steps.append(step)

        if step < N_STEPS:
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr), Y_tr)
            loss.backward()
            optimizer.step()

    grokking_step = next((s for s, a in zip(steps, test_accs) if a > 0.9), None)
    memorization_step = next((s for s, a in zip(steps, train_accs) if a > 0.99), None)

    print(f"    Memorization: {memorization_step}, Grokking: {grokking_step}")

    return {
        "label": label, "seed": seed, "lr": lr, "weight_decay": weight_decay,
        "use_bias": use_bias, "optimizer_type": optimizer_type,
        "steps": steps, "train_losses": train_losses,
        "train_accs": train_accs, "test_accs": test_accs,
        "conservation_drifts": conservation_drifts,
        "total_weight_norms": total_weight_norms,
        "hessian_eigenvalues": hessian_eigenvalues,
        "hessian_steps": hessian_steps,
        "grokking_step": grokking_step,
        "memorization_step": memorization_step,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  CONSERVATION LAW BREAKING DURING GROKKING (v2)")
    print(f"  Modular addition mod {P}, train={FRAC_TRAIN}, hidden={HIDDEN}")
    print("=" * 70)

    t_start = time.time()
    all_results = {}

    # Configuration grid: test what produces grokking
    configs = [
        # SGD configs (conservation-relevant)
        {"lr": 1.0,  "weight_decay": 0.1, "use_bias": False, "optimizer_type": "sgd"},
        {"lr": 0.5,  "weight_decay": 0.1, "use_bias": False, "optimizer_type": "sgd"},
        {"lr": 1.0,  "weight_decay": 0.1, "use_bias": True,  "optimizer_type": "sgd"},
        # Adam config (standard grokking, control)
        {"lr": 0.001, "weight_decay": 1.0, "use_bias": True, "optimizer_type": "adam"},
    ]

    for cfg in configs:
        seed = SEEDS[0]
        res = run_grokking(seed, **cfg)
        all_results[res["label"]] = {k: v for k, v in res.items()}

        # If grokking found, run more seeds
        if res["grokking_step"] is not None:
            print(f"    >>> GROKKING FOUND! Running additional seeds...")
            for extra_seed in SEEDS[1:3]:
                res2 = run_grokking(extra_seed, **cfg)
                all_results[res2["label"] + f"_s{extra_seed}"] = {k: v for k, v in res2.items()}

    t_total = time.time() - t_start

    # Summary
    print(f"\n\n{'='*70}")
    print("  GROKKING RESULTS SUMMARY")
    print(f"{'='*70}")
    for label, res in all_results.items():
        grok = res.get("grokking_step", None)
        memo = res.get("memorization_step", None)
        final_te = res["test_accs"][-1] if res["test_accs"] else 0
        final_drift = res["conservation_drifts"][-1] if res["conservation_drifts"] else 0
        print(f"  {label:<50s} memo={str(memo):>6s} grok={str(grok):>6s} "
              f"te_acc={final_te:.3f} drift={final_drift:.4f}")

    # Analyze drift at grokking transition for configs that grokked
    print(f"\n  Drift analysis at grokking transition:")
    for label, res in all_results.items():
        grok = res.get("grokking_step")
        if grok:
            steps = res["steps"]
            drifts = res["conservation_drifts"]
            pre = [d for s, d in zip(steps, drifts) if grok - 3000 < s < grok]
            post = [d for s, d in zip(steps, drifts) if grok < s < grok + 3000]
            if pre and post:
                ratio = np.mean(post) / max(np.mean(pre), 1e-10)
                print(f"    {label}: pre={np.mean(pre):.6f}, post={np.mean(post):.6f}, ratio={ratio:.2f}x")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 2, "date": "2026-04-07",
        "hardware": get_hardware_info(), "p": P, "frac_train": FRAC_TRAIN,
        "n_steps": N_STEPS, "hidden": HIDDEN, "configs": configs,
        "total_runtime_s": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    setup_publication_style()

    # Find a config that grokked for the main plot
    grokked_configs = [(l, r) for l, r in all_results.items() if r.get("grokking_step")]
    non_grokked = [(l, r) for l, r in all_results.items() if not r.get("grokking_step")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Panel (a): Training dynamics for all configs
    ax = axes[0, 0]
    for i, (label, res) in enumerate(all_results.items()):
        if '_s' not in label:  # Only plot first seed per config
            ax.plot(res["steps"], res["test_accs"], color=colors[i], linewidth=1.2,
                    label=f'{label[:30]} (te)')
            ax.plot(res["steps"], res["train_accs"], color=colors[i], linewidth=0.8,
                    linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Training Dynamics: Train (dashed) / Test (solid)')
    ax.legend(fontsize=6, loc='center right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel (b): Conservation drift for all configs
    ax = axes[0, 1]
    for i, (label, res) in enumerate(all_results.items()):
        if '_s' not in label:
            ax.plot(res["steps"], res["conservation_drifts"], color=colors[i],
                    linewidth=1.2, label=label[:30])
            if res.get("grokking_step"):
                ax.axvline(x=res["grokking_step"], color=colors[i],
                          linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Conservation Drift $|\\Delta C|$')
    ax.set_title('(b) Conservation Drift')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel (c): If grokking found, detailed drift + accuracy overlay
    ax = axes[1, 0]
    if grokked_configs:
        label, res = grokked_configs[0]
        ax.plot(res["steps"], res["conservation_drifts"], 'b-', linewidth=1.2, label='Drift')
        ax.set_ylabel('Conservation Drift', color='blue')
        ax2 = ax.twinx()
        ax2.plot(res["steps"], res["test_accs"], 'r-', linewidth=1.5, label='Test Acc')
        ax2.set_ylabel('Test Accuracy', color='red')
        ax2.set_ylim(-0.05, 1.05)
        if res["grokking_step"]:
            ax.axvline(x=res["grokking_step"], color='orange', linestyle='--',
                       linewidth=2, alpha=0.8, label=f'Grokking @ {res["grokking_step"]}')
        if res["memorization_step"]:
            ax.axvline(x=res["memorization_step"], color='purple', linestyle=':',
                       linewidth=1.5, alpha=0.6, label=f'Memorize @ {res["memorization_step"]}')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center left')
        ax.set_title(f'(c) Drift vs Grokking ({label[:25]})')
    else:
        ax.text(0.5, 0.5, 'No grokking observed\nin any configuration',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('(c) Drift vs Grokking (N/A)')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    # Panel (d): Total weight norm over time
    ax = axes[1, 1]
    for i, (label, res) in enumerate(all_results.items()):
        if '_s' not in label:
            ax.plot(res["steps"], res["total_weight_norms"], color=colors[i],
                    linewidth=1.2, label=label[:30])
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Weight Norm $\\sum ||W_l||_F^2$')
    ax.set_title('(d) Weight Norm Evolution')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Conservation Laws and Grokking (mod {P})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig7_grokking_conservation", FIGURE_DIR)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}")
    print(f"Total runtime: {t_total:.1f}s")
