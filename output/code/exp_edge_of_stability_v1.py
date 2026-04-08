"""
Experiment: Edge of Stability Dynamics
Theory: Connects to Theory A (conservation laws may explain EoS)
Prediction: lambda_max(Hessian) rises during training and stabilizes at ~2/eta.
            The 2/eta threshold should be visible for sufficiently large learning rates.
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
    get_hardware_info, top_hessian_eigenvalue
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "edge_of_stability"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurations: vary learning rate to see EoS at different thresholds
CONFIGS = [
    {"name": "lr0.01", "lr": 0.01, "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "n_steps": 3000, "eos_every": 20},
    {"name": "lr0.05", "lr": 0.05, "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "n_steps": 3000, "eos_every": 20},
    {"name": "lr0.1", "lr": 0.1, "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "n_steps": 3000, "eos_every": 20},
    {"name": "lr0.5", "lr": 0.5, "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 20, "output_dim": 5, "n_steps": 2000, "eos_every": 20},
    {"name": "deep_lr0.05", "lr": 0.05, "arch": "mlp_deep", "hidden": 32,
     "input_dim": 20, "output_dim": 5, "n_steps": 3000, "eos_every": 20, "depth": 4},
]

DATA_CONFIG = {"name": "gaussian_mixture", "n_train": 300, "n_test": 50,
               "d": 20, "K": 5, "separation": 1.5}

def run_eos_experiment(config, seed):
    """Train model and track lambda_max at regular intervals."""
    set_seed(seed)

    kwargs = {"hidden": config["hidden"]}
    if "depth" in config:
        kwargs["depth"] = config["depth"]
    model = make_model(config["arch"], config["input_dim"], config["output_dim"], **kwargs)

    X_train, Y_train, _, _ = make_dataset(
        DATA_CONFIG["name"], DATA_CONFIG["n_train"], DATA_CONFIG["n_test"],
        d=DATA_CONFIG["d"], K=DATA_CONFIG["K"], separation=DATA_CONFIG["separation"]
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    steps_record = []
    losses_record = []
    lambda_max_record = []
    grad_norms_record = []

    eos_every = config["eos_every"]
    n_steps = config["n_steps"]

    for step in range(1, n_steps + 1):
        # Forward + backward
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, Y_train)
        loss.backward()

        # Record gradient norm
        grad_norm = sum(p.grad.data.norm()**2 for p in model.parameters() if p.grad is not None).sqrt().item()

        optimizer.step()

        # Record lambda_max periodically
        if step % eos_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                current_loss = loss_fn(model(X_train), Y_train).item()

            lmax = top_hessian_eigenvalue(model, X_train, Y_train, loss_fn, n_iter=30)

            steps_record.append(step)
            losses_record.append(current_loss)
            lambda_max_record.append(lmax)
            grad_norms_record.append(grad_norm)

            if step % (eos_every * 10) == 0:
                print(f"    step {step:>5}/{n_steps} | loss={current_loss:.4f} | lambda_max={lmax:.2f} | 2/eta={2/config['lr']:.2f}")

    return {
        "steps": steps_record,
        "losses": losses_record,
        "lambda_max": lambda_max_record,
        "grad_norms": grad_norms_record,
        "eos_threshold": 2.0 / config["lr"],
    }

def run_all():
    all_results = {}

    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']} (lr={config['lr']}, 2/eta={2/config['lr']:.1f})")
        print(f"{'='*60}")

        config_results = []
        for seed in SEEDS[:3]:  # Use 3 seeds for speed (EoS is computationally expensive)
            print(f"  Seed {seed}:")
            t0 = time.time()
            result = run_eos_experiment(config, seed)
            dt = time.time() - t0
            print(f"  Done ({dt:.1f}s)")
            config_results.append(result)

        all_results[config["name"]] = {
            "config": config,
            "seed_results": config_results,
            "eos_threshold": 2.0 / config["lr"],
        }

    return all_results

def generate_figures(all_results):
    setup_publication_style()

    # Main figure: lambda_max vs training step for different learning rates
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    lr_configs = ["lr0.01", "lr0.05", "lr0.1", "lr0.5", "deep_lr0.05"]
    titles = ["lr=0.01 (2/lr=200)", "lr=0.05 (2/lr=40)", "lr=0.1 (2/lr=20)",
              "lr=0.5 (2/lr=4)", "Deep 4L, lr=0.05 (2/lr=40)"]

    for idx, (name, title) in enumerate(zip(lr_configs, titles)):
        if name not in all_results:
            continue
        ax = axes[idx // 3][idx % 3]
        data = all_results[name]
        threshold = data["eos_threshold"]

        for i, result in enumerate(data["seed_results"]):
            steps = result["steps"]
            lmax = result["lambda_max"]
            ax.plot(steps, lmax, alpha=0.7, linewidth=0.8, label=f'seed {i+1}' if idx == 0 else None)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'$2/\\eta = {threshold:.0f}$')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('$\\lambda_{\\max}$')
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Loss subplot
    ax_loss = axes[1][2]
    for name in ["lr0.01", "lr0.05", "lr0.1"]:
        if name in all_results:
            result = all_results[name]["seed_results"][0]
            ax_loss.plot(result["steps"], result["losses"], linewidth=1, label=name)
    ax_loss.set_xlabel('Training Step')
    ax_loss.set_ylabel('Training Loss')
    ax_loss.set_title('Loss Dynamics')
    ax_loss.legend(fontsize=8)
    ax_loss.set_yscale('log')

    fig.suptitle('Edge of Stability: $\\lambda_{\\max}$ vs Training Step', y=1.01, fontsize=16)
    plt.tight_layout()
    save_figure(fig, "fig4_edge_of_stability", FIGURE_DIR)
    plt.close()

    print("Figures saved.")

if __name__ == "__main__":
    print("Edge of Stability Experiment")
    print(f"Hardware: {get_hardware_info()['processor']}")
    t_start = time.time()

    results = run_all()

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary: does lambda_max reach 2/eta?
    print("\n--- EDGE OF STABILITY SUMMARY ---")
    for name, data in results.items():
        threshold = data["eos_threshold"]
        for i, r in enumerate(data["seed_results"]):
            max_lmax = max(r["lambda_max"])
            final_lmax = np.mean(r["lambda_max"][-5:])
            reached = "YES" if max_lmax > 0.8 * threshold else "NO"
            print(f"  {name} seed{i}: max_lmax={max_lmax:.2f}, final_lmax={final_lmax:.2f}, 2/eta={threshold:.1f}, reached_EoS={reached}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "theory": "edge_of_stability",
        "version": 1, "date": "2026-04-07", "hardware": get_hardware_info(),
        "seeds": SEEDS[:3], "configs": CONFIGS, "data_config": DATA_CONFIG,
    }

    results_save = {}
    for name, data in results.items():
        results_save[name] = {
            "config": data["config"],
            "eos_threshold": data["eos_threshold"],
            "seeds": [{
                "steps": r["steps"],
                "lambda_max": r["lambda_max"],
                "losses": r["losses"],
            } for r in data["seed_results"]]
        }

    save_results(results_save, OUTPUT_DIR, config)
    generate_figures(results)
    print(f"\nResults saved to {OUTPUT_DIR}")
