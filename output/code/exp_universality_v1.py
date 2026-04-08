"""
Experiment: Universality of Conservation Drift Scaling Exponent
Theory: Theory A -- Is the drift ~ lr^{1.16} exponent universal?
Prediction: If the exponent is universal across depth, width, dataset, and optimizer,
            it represents a fundamental constant of neural network training dynamics.
            If it varies systematically, the variation reveals the mechanism.
Date: 2026-04-07
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

# Shared parameters
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
N_STEPS = 2000
N_SEEDS = 5


def get_weight_layers(model):
    """Extract weight parameter tensors (not bias) from a Sequential model."""
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight)
    return weights


def compute_conservation_drifts(weights_init, weights_final):
    """Compute absolute drift for all L-1 conservation quantities.
    C_l = ||W_{l+1}||^2 - ||W_l||^2 for l=1,...,L-1
    Returns mean absolute drift across all conservation quantities.
    """
    L = len(weights_init)
    if L < 2:
        return 0.0, []

    drifts = []
    for l in range(L - 1):
        C_init = weights_init[l+1].norm().item()**2 - weights_init[l].norm().item()**2
        C_final = weights_final[l+1].norm().item()**2 - weights_final[l].norm().item()**2
        drifts.append(abs(C_final - C_init))

    return float(np.mean(drifts)), drifts


def measure_drift_general(lr, seed, arch, input_dim, output_dim, hidden,
                          dataset_name, dataset_kwargs, n_train=200,
                          optimizer_type="sgd", depth=None):
    """General drift measurement for any configuration."""
    set_seed(seed)

    model_kwargs = {"hidden": hidden}
    if depth is not None:
        model_kwargs["depth"] = depth
    model = make_model(arch, input_dim, output_dim, **model_kwargs)

    X, Y, _, _ = make_dataset(dataset_name, n_train=n_train, n_test=50, **dataset_kwargs)

    # Record initial weight norms
    weights = get_weight_layers(model)
    norms_init = [(w.data.norm().item()**2) for w in weights]

    # Set up optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    loss_fn = nn.CrossEntropyLoss()

    # Train
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    # Final weight norms
    weights_final = get_weight_layers(model)
    norms_final = [(w.data.norm().item()**2) for w in weights_final]

    # Compute conservation drifts
    n_layers = len(norms_init)
    per_layer_drifts = []
    for l in range(n_layers - 1):
        C_init = norms_init[l+1] - norms_init[l]
        C_final = norms_final[l+1] - norms_final[l]
        per_layer_drifts.append(abs(C_final - C_init))

    mean_drift = float(np.mean(per_layer_drifts)) if per_layer_drifts else 0.0
    final_loss = loss.item()

    return mean_drift, per_layer_drifts, final_loss


def fit_power_law(lrs, drifts):
    """Fit drift = a * lr^b in log-log space. Returns (exponent, R^2)."""
    lrs_arr = np.array(lrs)
    drifts_arr = np.array(drifts)
    mask = drifts_arr > 1e-10
    if mask.sum() < 3:
        return 0.0, 0.0, np.zeros_like(lrs_arr)

    log_lr = np.log(lrs_arr[mask])
    log_drift = np.log(drifts_arr[mask])
    slope, intercept = np.polyfit(log_lr, log_drift, 1)

    # R^2
    predicted = slope * log_lr + intercept
    ss_res = np.sum((log_drift - predicted) ** 2)
    ss_tot = np.sum((log_drift - np.mean(log_drift)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Fitted curve over full range
    fit_curve = np.exp(intercept) * lrs_arr ** slope

    return float(slope), float(r_squared), fit_curve


def run_sweep(sweep_name, configs, output_subdir):
    """Run a sweep over multiple configurations, fitting power law for each."""
    output_dir = os.path.join(BASE_DIR, "experiments", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    summary = []

    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {sweep_name}: {label}")
        print(f"{'='*60}")

        lr_results = {}
        for lr in LRS:
            drifts = []
            losses = []
            for seed in SEEDS[:N_SEEDS]:
                d, _, l = measure_drift_general(
                    lr=lr, seed=seed,
                    arch=cfg["arch"], input_dim=cfg["input_dim"],
                    output_dim=cfg["output_dim"], hidden=cfg["hidden"],
                    dataset_name=cfg["dataset"], dataset_kwargs=cfg["dataset_kwargs"],
                    n_train=cfg.get("n_train", 200),
                    optimizer_type=cfg.get("optimizer", "sgd"),
                    depth=cfg.get("depth", None),
                )
                drifts.append(d)
                losses.append(l)

            mean_drift = float(np.mean(drifts))
            std_drift = float(np.std(drifts))
            print(f"  lr={lr:>8.4f}: drift={mean_drift:.6f} +/- {std_drift:.6f}, loss={np.mean(losses):.4f}")

            lr_results[str(lr)] = {
                "lr": lr, "mean_drift": mean_drift, "std_drift": std_drift,
                "mean_loss": float(np.mean(losses)), "drifts": [float(d) for d in drifts],
            }

        # Fit power law
        mean_drifts = [lr_results[str(lr)]["mean_drift"] for lr in LRS]
        exponent, r_sq, fit_curve = fit_power_law(LRS, mean_drifts)

        print(f"\n  >>> Power law: drift ~ lr^{exponent:.3f} (R^2 = {r_sq:.4f})")

        all_results[label] = {
            "config": {k: v for k, v in cfg.items() if k != "dataset_kwargs"},
            "lr_results": lr_results,
            "power_law_exponent": exponent,
            "r_squared": r_sq,
        }
        summary.append({"label": label, "exponent": exponent, "r_squared": r_sq})

    # Save results
    config = {
        "experiment_name": output_subdir, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "sweep_name": sweep_name,
        "seeds": SEEDS[:N_SEEDS], "lrs": LRS, "n_steps": N_STEPS,
        "summary": summary,
    }
    save_results(all_results, output_dir, config)

    return all_results, summary


# ============================================================================
#  SWEEP 1A: DEPTH
# ============================================================================
def sweep_depth():
    configs = []
    for depth in [1, 2, 3, 5, 7]:  # depth=1 -> 2 weight layers, depth=7 -> 8 layers
        n_layers = depth + 1
        arch = "mlp_2layer_nobias" if depth == 1 else "mlp_deep_nobias"
        configs.append({
            "label": f"depth_{n_layers}L",
            "arch": arch, "input_dim": 20, "output_dim": 5, "hidden": 64,
            "dataset": "gaussian_mixture",
            "dataset_kwargs": {"d": 20, "K": 5, "separation": 2.0},
            "depth": depth if depth > 1 else None,
        })
    return run_sweep("Depth Sweep", configs, "universality_depth")


# ============================================================================
#  SWEEP 1B: WIDTH
# ============================================================================
def sweep_width():
    configs = []
    for width in [16, 32, 64, 128, 256]:
        configs.append({
            "label": f"width_{width}",
            "arch": "mlp_2layer_nobias", "input_dim": 20, "output_dim": 5,
            "hidden": width,
            "dataset": "gaussian_mixture",
            "dataset_kwargs": {"d": 20, "K": 5, "separation": 2.0},
        })
    return run_sweep("Width Sweep", configs, "universality_width")


# ============================================================================
#  SWEEP 1C: DATASET
# ============================================================================
def sweep_dataset():
    configs = [
        {
            "label": "gaussian_d20",
            "arch": "mlp_2layer_nobias", "input_dim": 20, "output_dim": 5,
            "hidden": 64, "n_train": 200,
            "dataset": "gaussian_mixture",
            "dataset_kwargs": {"d": 20, "K": 5, "separation": 2.0},
        },
        {
            "label": "xor_d2",
            "arch": "mlp_2layer_nobias", "input_dim": 2, "output_dim": 2,
            "hidden": 64, "n_train": 200,
            "dataset": "xor_2d",
            "dataset_kwargs": {},
        },
        {
            "label": "spheres_d10",
            "arch": "mlp_2layer_nobias", "input_dim": 10, "output_dim": 2,
            "hidden": 64, "n_train": 200,
            "dataset": "concentric_spheres",
            "dataset_kwargs": {"d": 10},
        },
        {
            "label": "mnist_d784",
            "arch": "mlp_2layer_nobias", "input_dim": 784, "output_dim": 10,
            "hidden": 64, "n_train": 500,
            "dataset": "mnist",
            "dataset_kwargs": {"flatten": True},
        },
    ]
    return run_sweep("Dataset Sweep", configs, "universality_dataset")


# ============================================================================
#  SWEEP 1D: OPTIMIZER
# ============================================================================
def sweep_optimizer():
    base = {
        "arch": "mlp_2layer_nobias", "input_dim": 20, "output_dim": 5,
        "hidden": 64, "dataset": "gaussian_mixture",
        "dataset_kwargs": {"d": 20, "K": 5, "separation": 2.0},
    }
    configs = [
        {**base, "label": "sgd", "optimizer": "sgd"},
        {**base, "label": "sgd_momentum", "optimizer": "sgd_momentum"},
        {**base, "label": "adam", "optimizer": "adam"},
    ]
    return run_sweep("Optimizer Sweep", configs, "universality_optimizer")


# ============================================================================
#  VISUALIZATION
# ============================================================================
def plot_universality(all_summaries):
    """Create publication-quality figures for all sweeps."""
    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sweep_names = ["Depth Sweep", "Width Sweep", "Dataset Sweep", "Optimizer Sweep"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (ax, sweep_name, panel) in enumerate(zip(axes.flat, sweep_names, panel_labels)):
        results, summary = all_summaries[idx]

        for i, (label, data) in enumerate(results.items()):
            lr_data = data["lr_results"]
            lrs = [lr_data[str(lr)]["lr"] for lr in LRS]
            drifts = [lr_data[str(lr)]["mean_drift"] for lr in LRS]
            stds = [lr_data[str(lr)]["std_drift"] for lr in LRS]

            exp = data["power_law_exponent"]
            r2 = data["r_squared"]

            ax.errorbar(lrs, drifts, yerr=stds, fmt='o-', capsize=3,
                       color=colors[i], markersize=4, linewidth=1.2,
                       label=f'{label} ($\\alpha$={exp:.2f}, R$^2$={r2:.3f})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Learning Rate $\\eta$')
        ax.set_ylabel('Mean Conservation Drift')
        ax.set_title(f'{panel} {sweep_name}')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Reference line at eta^1.16
        lr_ref = np.logspace(-4, 0, 50)
        ax.plot(lr_ref, 0.5 * lr_ref**1.16, 'k--', alpha=0.3, linewidth=1,
                label='$\\eta^{1.16}$ ref' if idx == 0 else None)

    fig.suptitle('Universality of Conservation Drift Scaling Exponent', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig6_universality", FIGURE_DIR)
    plt.close()

    # Summary bar chart of exponents
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    all_labels = []
    all_exponents = []
    all_r2 = []
    sweep_boundaries = []

    for results, summary in all_summaries:
        sweep_boundaries.append(len(all_labels))
        for s in summary:
            all_labels.append(s["label"])
            all_exponents.append(s["exponent"])
            all_r2.append(s["r_squared"])

    x = np.arange(len(all_labels))
    bars = ax2.bar(x, all_exponents, color=[colors[i % 10] for i in range(len(x))],
                   edgecolor='black', linewidth=0.5)

    # Reference line at 1.16
    ax2.axhline(y=1.16, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Original exponent (1.16)')
    # Reference at 2.0 (naive discretization)
    ax2.axhline(y=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                label='Naive $\\eta^2$ prediction')

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Power Law Exponent $\\alpha$')
    ax2.set_title('Conservation Drift Exponent: $\\Delta C \\propto \\eta^\\alpha$')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    # Add R^2 annotations
    for i, (exp, r2) in enumerate(zip(all_exponents, all_r2)):
        ax2.annotate(f'R$^2$={r2:.2f}', (i, exp), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=6)

    # Sweep separators
    for b in sweep_boundaries[1:]:
        ax2.axvline(x=b - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    save_figure(fig2, "fig6b_exponent_summary", FIGURE_DIR)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}")


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  UNIVERSALITY OF CONSERVATION DRIFT SCALING EXPONENT")
    print("  Testing drift ~ lr^alpha across depth, width, dataset, optimizer")
    print("=" * 70)

    t_start = time.time()

    # Run all four sweeps
    print("\n\n>>> SWEEP 1A: DEPTH")
    res_depth = sweep_depth()

    print("\n\n>>> SWEEP 1B: WIDTH")
    res_width = sweep_width()

    print("\n\n>>> SWEEP 1C: DATASET")
    res_dataset = sweep_dataset()

    print("\n\n>>> SWEEP 1D: OPTIMIZER")
    res_optimizer = sweep_optimizer()

    t_total = time.time() - t_start
    print(f"\n\nTotal runtime: {t_total:.1f}s")

    # Compile universality table
    print("\n" + "=" * 70)
    print("  UNIVERSALITY TABLE")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Exponent':>10} {'R^2':>8}")
    print("-" * 45)

    all_summaries = [res_depth, res_width, res_dataset, res_optimizer]
    sweep_labels = ["DEPTH", "WIDTH", "DATASET", "OPTIMIZER"]

    for sweep_label, (_, summary) in zip(sweep_labels, all_summaries):
        print(f"\n  {sweep_label}:")
        for s in summary:
            print(f"    {s['label']:<21} {s['exponent']:>10.3f} {s['r_squared']:>8.4f}")

    # Compute overall statistics
    all_exponents = []
    for _, summary in all_summaries:
        for s in summary:
            if s["r_squared"] > 0.8:  # Only include well-fit cases
                all_exponents.append(s["exponent"])

    if all_exponents:
        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents)
        print(f"\n{'='*45}")
        print(f"  Well-fit exponents (R^2 > 0.8): {len(all_exponents)}")
        print(f"  Mean exponent: {mean_exp:.3f} +/- {std_exp:.3f}")
        print(f"  Range: [{min(all_exponents):.3f}, {max(all_exponents):.3f}]")
        if std_exp / mean_exp < 0.15:
            print(f"  >>> UNIVERSAL: CV = {std_exp/mean_exp:.2%} < 15%")
        else:
            print(f"  >>> VARIES SYSTEMATICALLY: CV = {std_exp/mean_exp:.2%} >= 15%")

    # Plot
    plot_universality(all_summaries)

    print(f"\nExperiment complete. Total time: {t_total:.1f}s")
