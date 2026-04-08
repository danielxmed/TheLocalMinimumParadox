"""
Experiment: Conservation Laws Verification
Theory: Theory A -- Noether Conservation Laws for Gradient Flow
Prediction: For bias-free homogeneous networks under gradient flow (small lr),
            C_l = ||W_{l+1}||_F^2 - ||W_l||_F^2 is conserved.
            With bias, conservation breaks. With large lr, drift increases.
Date: 2026-04-07
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn

# Add parent dir for utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Experiment Configuration
# ============================================================

EXPERIMENT_NAME = "conservation_laws"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

N_STEPS = 2000
LOG_EVERY = 1  # Log every step for conservation tracking

# Configurations to test
CONFIGS = [
    {"name": "2layer_nobias_lr0.001", "arch": "mlp_2layer_nobias", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "lr": 0.001, "bias": False, "depth": None},
    {"name": "2layer_bias_lr0.001", "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "lr": 0.001, "bias": True, "depth": None},
    {"name": "deep4_nobias_lr0.001", "arch": "mlp_deep_nobias", "hidden": 32,
     "input_dim": 20, "output_dim": 5, "lr": 0.001, "bias": False, "depth": 4},
    {"name": "deep4_bias_lr0.001", "arch": "mlp_deep", "hidden": 32,
     "input_dim": 20, "output_dim": 5, "lr": 0.001, "bias": True, "depth": 4},
    {"name": "2layer_nobias_lr0.01", "arch": "mlp_2layer_nobias", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "lr": 0.01, "bias": False, "depth": None},
    {"name": "2layer_nobias_lr0.1", "arch": "mlp_2layer_nobias", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "lr": 0.1, "bias": False, "depth": None},
]

def get_layer_norms(model):
    """Extract Frobenius norms of each weight matrix."""
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            norms.append(param.data.norm().item() ** 2)  # ||W_l||_F^2
    return norms

def compute_conservation_quantities(model):
    """Compute C_l = ||W_{l+1}||_F^2 - ||W_l||_F^2 for all layer pairs."""
    norms_sq = get_layer_norms(model)
    C = []
    for l in range(len(norms_sq) - 1):
        C.append(norms_sq[l + 1] - norms_sq[l])
    return C, norms_sq

def conservation_callback(model, step, loss):
    """Callback to track conservation quantities during training."""
    C, norms_sq = compute_conservation_quantities(model)
    result = {"conservation_quantities": C, "layer_norms_sq": norms_sq}
    # Also track gradient norm
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm().item() ** 2
    result["grad_norm"] = total_grad_norm ** 0.5
    return result

def run_single_config(config, seed):
    """Run one configuration with one seed."""
    set_seed(seed)

    # Create model
    kwargs = {"hidden": config["hidden"]}
    if config["depth"] is not None:
        kwargs["depth"] = config["depth"]
    model = make_model(config["arch"], config["input_dim"], config["output_dim"], **kwargs)
    model.to(DEVICE)

    # Create data
    X_train, Y_train, _, _ = make_dataset(
        "gaussian_mixture", n_train=200, n_test=50,
        d=config["input_dim"], K=config["output_dim"], separation=2.0
    )

    # Record initial conservation quantities
    C_init, norms_init = compute_conservation_quantities(model)

    # Train
    history = train(
        model, X_train, Y_train,
        lr=config["lr"], n_steps=N_STEPS,
        optimizer_type="sgd", batch_size=None,  # Full-batch GD
        callback=conservation_callback,
        log_every=LOG_EVERY
    )

    # Extract conservation trajectories
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    grad_norms = [h.get("grad_norm", 0) for h in history]

    # Conservation quantities over time
    n_conservation = len(C_init)
    C_trajectories = np.zeros((len(history), n_conservation))
    norms_trajectories = np.zeros((len(history), n_conservation + 1))

    for i, h in enumerate(history):
        C_trajectories[i] = h.get("conservation_quantities", C_init)
        norms_trajectories[i] = h.get("layer_norms_sq", norms_init)

    # Compute relative drift for each C_l
    drifts = []
    for l in range(n_conservation):
        if abs(C_init[l]) > 1e-10:
            max_drift = np.max(np.abs(C_trajectories[:, l] - C_init[l])) / abs(C_init[l])
        else:
            max_drift = np.max(np.abs(C_trajectories[:, l] - C_init[l]))
        drifts.append(max_drift)

    return {
        "seed": seed,
        "C_init": C_init,
        "C_final": C_trajectories[-1].tolist(),
        "max_relative_drifts": drifts,
        "mean_relative_drift": float(np.mean(drifts)),
        "final_loss": losses[-1],
        "steps": steps,
        "losses": losses,
        "C_trajectories": C_trajectories.tolist(),
        "norms_trajectories": norms_trajectories.tolist(),
        "grad_norms": grad_norms,
    }

def run_experiment():
    """Run all configurations across all seeds."""
    all_results = {}

    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"{'='*60}")

        config_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            t0 = time.time()
            result = run_single_config(config, seed)
            dt = time.time() - t0
            print(f"done ({dt:.1f}s) | drift={result['mean_relative_drift']:.6f} | loss={result['final_loss']:.4f}")
            config_results.append(result)

        # Aggregate across seeds
        mean_drifts = [r["mean_relative_drift"] for r in config_results]
        all_results[config["name"]] = {
            "config": config,
            "seed_results": config_results,
            "mean_drift_across_seeds": float(np.mean(mean_drifts)),
            "std_drift_across_seeds": float(np.std(mean_drifts)),
            "max_drift_across_seeds": float(np.max(mean_drifts)),
        }

        print(f"  Mean drift: {np.mean(mean_drifts):.6f} +/- {np.std(mean_drifts):.6f}")

    return all_results

def generate_figures(all_results):
    """Generate publication-quality figures."""
    setup_publication_style()

    # Figure 1: Conservation quantities over training (nobias vs bias, 2-layer)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel (a): No bias, small lr -- should be conserved
    config_name = "2layer_nobias_lr0.001"
    if config_name in all_results:
        data = all_results[config_name]["seed_results"][0]  # First seed
        steps = data["steps"]
        C_traj = np.array(data["C_trajectories"])
        C_init = data["C_init"][0]
        axes[0].plot(steps, C_traj[:, 0], 'b-', linewidth=1)
        axes[0].axhline(y=C_init, color='r', linestyle='--', alpha=0.7, label=f'C(0) = {C_init:.4f}')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('$C_1 = \\|W_2\\|_F^2 - \\|W_1\\|_F^2$')
        axes[0].set_title('No Bias, lr=0.001')
        axes[0].legend()

    # Panel (b): With bias -- should drift
    config_name = "2layer_bias_lr0.001"
    if config_name in all_results:
        data = all_results[config_name]["seed_results"][0]
        steps = data["steps"]
        C_traj = np.array(data["C_trajectories"])
        C_init = data["C_init"][0]
        axes[1].plot(steps, C_traj[:, 0], 'b-', linewidth=1)
        axes[1].axhline(y=C_init, color='r', linestyle='--', alpha=0.7, label=f'C(0) = {C_init:.4f}')
        axes[1].set_xlabel('Training Step')
        axes[1].set_title('With Bias, lr=0.001')
        axes[1].legend()

    # Panel (c): No bias, large lr -- discretization drift
    config_name = "2layer_nobias_lr0.1"
    if config_name in all_results:
        data = all_results[config_name]["seed_results"][0]
        steps = data["steps"]
        C_traj = np.array(data["C_trajectories"])
        C_init = data["C_init"][0]
        axes[2].plot(steps, C_traj[:, 0], 'b-', linewidth=1)
        axes[2].axhline(y=C_init, color='r', linestyle='--', alpha=0.7, label=f'C(0) = {C_init:.4f}')
        axes[2].set_xlabel('Training Step')
        axes[2].set_title('No Bias, lr=0.1')
        axes[2].legend()

    fig.suptitle('Conservation Law: $C = \\|W_2\\|_F^2 - \\|W_1\\|_F^2$', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig5_conservation_laws", FIGURE_DIR)
    plt.close()

    # Figure 2: Drift vs learning rate
    fig, ax = plt.subplots(figsize=(6, 4))
    lr_configs = [
        ("2layer_nobias_lr0.001", 0.001),
        ("2layer_nobias_lr0.01", 0.01),
        ("2layer_nobias_lr0.1", 0.1),
    ]
    lrs = []
    mean_drifts = []
    std_drifts = []
    for name, lr in lr_configs:
        if name in all_results:
            drifts = [r["mean_relative_drift"] for r in all_results[name]["seed_results"]]
            lrs.append(lr)
            mean_drifts.append(np.mean(drifts))
            std_drifts.append(np.std(drifts))

    if lrs:
        ax.errorbar(lrs, mean_drifts, yerr=std_drifts, fmt='o-', capsize=5, color='darkblue')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Mean Relative Drift of $C$')
        ax.set_title('Conservation Drift vs Learning Rate (No Bias)')

    plt.tight_layout()
    save_figure(fig, "fig5b_drift_vs_lr", FIGURE_DIR)
    plt.close()

    # Figure 3: Layer norm evolution (deep network)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (config_name, title) in enumerate([
        ("deep4_nobias_lr0.001", "4-Layer, No Bias"),
        ("deep4_bias_lr0.001", "4-Layer, With Bias")
    ]):
        if config_name in all_results:
            data = all_results[config_name]["seed_results"][0]
            steps = data["steps"]
            norms = np.array(data["norms_trajectories"])
            for l in range(norms.shape[1]):
                axes[idx].plot(steps, norms[:, l], label=f'$\\|W_{l+1}\\|_F^2$')
            axes[idx].set_xlabel('Training Step')
            axes[idx].set_ylabel('$\\|W_l\\|_F^2$')
            axes[idx].set_title(title)
            axes[idx].legend()

    plt.tight_layout()
    save_figure(fig, "fig5c_layer_norms", FIGURE_DIR)
    plt.close()

    print("Figures saved to", FIGURE_DIR)

if __name__ == "__main__":
    print("="*60)
    print("Experiment: Conservation Laws Verification")
    print("="*60)
    print(f"Hardware: {get_hardware_info()['processor']}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Steps: {N_STEPS}")
    print()

    t_start = time.time()
    results = run_experiment()
    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Total time: {t_total:.1f}s")
    print(f"{'='*60}")

    # Summary table
    print("\n--- SUMMARY ---")
    print(f"{'Config':<30} {'Mean Drift':>12} {'Std':>10} {'Conservation?':>15}")
    print("-" * 70)
    for name, data in results.items():
        drift = data["mean_drift_across_seeds"]
        std = data["std_drift_across_seeds"]
        conserved = "YES" if drift < 0.01 else ("PARTIAL" if drift < 0.1 else "NO")
        print(f"{name:<30} {drift:>12.6f} {std:>10.6f} {conserved:>15}")

    # Save results
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "theory": "conservation_laws",
        "version": 1,
        "date": "2026-04-07",
        "hardware": get_hardware_info(),
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "configs": CONFIGS,
    }

    # Prepare serializable results (remove large trajectories for JSON)
    results_summary = {}
    for name, data in results.items():
        results_summary[name] = {
            "config": data["config"],
            "mean_drift": data["mean_drift_across_seeds"],
            "std_drift": data["std_drift_across_seeds"],
            "max_drift": data["max_drift_across_seeds"],
            "per_seed": [{
                "seed": r["seed"],
                "mean_drift": r["mean_relative_drift"],
                "max_drifts": r["max_relative_drifts"],
                "C_init": r["C_init"],
                "C_final": r["C_final"],
                "final_loss": r["final_loss"],
            } for r in data["seed_results"]]
        }

    save_results(results_summary, OUTPUT_DIR, config)

    # Save raw trajectories as numpy
    for name, data in results.items():
        for r in data["seed_results"]:
            np.save(
                os.path.join(OUTPUT_DIR, f"C_trajectory_{name}_seed{r['seed']}.npy"),
                np.array(r["C_trajectories"])
            )

    # Generate figures
    generate_figures(results)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("Done!")
