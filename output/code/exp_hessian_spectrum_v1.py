"""
Experiment: Hessian Spectrum Evolution During Training
Theory: Foundational -- supports all three theories
Prediction: (1) Spectrum shifts from negative+positive eigenvalues to concentrated-near-zero
            (2) Bulk near zero (overparameterization) with C-1 outlier eigenvalues for C classes
            (3) Negative eigenvalues disappear at low loss (Bray-Dean prediction)
Date: 2026-04-07
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    train, count_parameters, save_results, setup_publication_style, save_figure,
    get_hardware_info, full_hessian, hessian_eigenvalues_lanczos
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "hessian_spectrum"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use tiny networks where full Hessian is feasible
CONFIGS = [
    {"name": "tiny_2layer", "arch": "tiny_mlp", "hidden": 20,
     "input_dim": 10, "output_dim": 3, "method": "full"},  # ~253 params
    {"name": "small_2layer", "arch": "tiny_mlp", "hidden": 50,
     "input_dim": 10, "output_dim": 3, "method": "full"},  # ~583 params
    {"name": "medium_2layer", "arch": "mlp_2layer", "hidden": 64,
     "input_dim": 10, "output_dim": 3, "method": "lanczos"},  # ~899 params
]

DATA_CONFIG = {"name": "gaussian_mixture", "n_train": 100, "n_test": 30,
               "d": 10, "K": 3, "separation": 1.5}
LR = 0.01
N_STEPS = 2000
CHECKPOINTS = [0, 100, 500, 1000, 2000]

def run_hessian_experiment(config, seed):
    set_seed(seed)

    model = make_model(config["arch"], config["input_dim"], config["output_dim"],
                       hidden=config["hidden"])
    n_params = count_parameters(model)
    X_train, Y_train, _, _ = make_dataset(
        DATA_CONFIG["name"], DATA_CONFIG["n_train"], DATA_CONFIG["n_test"],
        d=DATA_CONFIG["d"], K=DATA_CONFIG["K"], separation=DATA_CONFIG["separation"]
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    spectra_at_checkpoints = {}

    for step in range(N_STEPS + 1):
        # Compute spectrum at checkpoints
        if step in CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                curr_loss = loss_fn(model(X_train), Y_train).item()

            if config["method"] == "full":
                H = full_hessian(model, X_train, Y_train, loss_fn)
                eigenvalues = np.sort(np.linalg.eigvalsh(H.numpy()))
            else:
                eigenvalues = hessian_eigenvalues_lanczos(model, X_train, Y_train, loss_fn, n_iter=min(n_params, 200))

            n_negative = np.sum(eigenvalues < -1e-6)
            n_zero = np.sum(np.abs(eigenvalues) < 1e-6)
            n_positive = np.sum(eigenvalues > 1e-6)

            spectra_at_checkpoints[step] = {
                "eigenvalues": eigenvalues.tolist(),
                "loss": curr_loss,
                "n_negative": int(n_negative),
                "n_zero": int(n_zero),
                "n_positive": int(n_positive),
                "max_eigenvalue": float(np.max(eigenvalues)),
                "min_eigenvalue": float(np.min(eigenvalues)),
                "trace": float(np.sum(eigenvalues)),
            }
            print(f"    step {step}: loss={curr_loss:.4f}, neg={n_negative}, zero={n_zero}, pos={n_positive}, max={np.max(eigenvalues):.4f}, min={np.min(eigenvalues):.4f}")

        # Training step
        if step < N_STEPS:
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = loss_fn(out, Y_train)
            loss.backward()
            optimizer.step()

    return spectra_at_checkpoints

if __name__ == "__main__":
    print("Hessian Spectrum Evolution Experiment")
    print(f"Checkpoints: {CHECKPOINTS}")
    t_start = time.time()

    all_results = {}
    for config in CONFIGS:
        n_params = count_parameters(make_model(config["arch"], config["input_dim"],
                                               config["output_dim"], hidden=config["hidden"]))
        print(f"\n{'='*60}")
        print(f"Config: {config['name']} ({n_params} params, method={config['method']})")
        print(f"{'='*60}")

        config_results = []
        for seed in SEEDS[:3]:
            print(f"  Seed {seed}:")
            r = run_hessian_experiment(config, seed)
            config_results.append(r)

        all_results[config["name"]] = {"config": config, "n_params": n_params, "seeds": config_results}

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Summary
    print("\n--- HESSIAN SPECTRUM SUMMARY ---")
    for name, data in all_results.items():
        print(f"\n{name} ({data['n_params']} params):")
        for cp in CHECKPOINTS:
            neg_counts = [s[cp]["n_negative"] for s in data["seeds"] if cp in s]
            losses = [s[cp]["loss"] for s in data["seeds"] if cp in s]
            if neg_counts:
                print(f"  step {cp:>5}: loss={np.mean(losses):.4f}, neg_eigs={np.mean(neg_counts):.1f}+/-{np.std(neg_counts):.1f}")

    # Save
    config_save = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "configs": CONFIGS, "data_config": DATA_CONFIG,
        "checkpoints": CHECKPOINTS,
    }
    save_results(all_results, OUTPUT_DIR, config_save)

    # Figures
    setup_publication_style()
    fig, axes = plt.subplots(len(CONFIGS), len(CHECKPOINTS), figsize=(4*len(CHECKPOINTS), 3.5*len(CONFIGS)))

    for row, (name, data) in enumerate(all_results.items()):
        for col, cp in enumerate(CHECKPOINTS):
            ax = axes[row][col] if len(CONFIGS) > 1 else axes[col]
            # Plot histogram of eigenvalues (first seed)
            if cp in data["seeds"][0]:
                eigs = np.array(data["seeds"][0][cp]["eigenvalues"])
                ax.hist(eigs, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
                loss = data["seeds"][0][cp]["loss"]
                n_neg = data["seeds"][0][cp]["n_negative"]
                ax.set_title(f'Step {cp}\nloss={loss:.3f}, neg={n_neg}', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'{name}\n({data["n_params"]} params)', fontsize=9)
                if row == len(CONFIGS) - 1:
                    ax.set_xlabel('Eigenvalue')

    fig.suptitle('Hessian Spectrum Evolution During Training', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig1_hessian_spectrum", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
