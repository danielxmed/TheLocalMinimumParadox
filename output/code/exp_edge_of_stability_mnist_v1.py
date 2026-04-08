"""
Experiment: Edge of Stability on MNIST
Theory: Edge of Stability + Conservation Laws connection
Prediction: On MNIST (harder than Gaussian), lambda_max should progressively
            sharpen and stabilize at 2/eta, showing classical EoS behavior.
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
    get_hardware_info, top_hessian_eigenvalue
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "eos_mnist"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use subset of MNIST for tractability on CPU
N_TRAIN = 1000
LR = 0.1  # Large enough to see EoS
N_STEPS = 2000
EOS_EVERY = 10  # Track lambda_max every 10 steps
HIDDEN = 128

def run_eos_mnist(seed):
    set_seed(seed)

    model = make_model("mlp_2layer", 784, 10, hidden=HIDDEN)
    X_train, Y_train, X_test, Y_test = make_dataset("mnist", n_train=N_TRAIN, n_test=200)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    steps, losses, lmax_vals, train_accs = [], [], [], []

    for step in range(1, N_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, Y_train)
        loss.backward()
        optimizer.step()

        if step % EOS_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                curr_loss = loss_fn(model(X_train), Y_train).item()
                pred = model(X_train).argmax(1)
                acc = (pred == Y_train).float().mean().item()

            lmax = top_hessian_eigenvalue(model, X_train, Y_train, loss_fn, n_iter=30)

            steps.append(step)
            losses.append(curr_loss)
            lmax_vals.append(lmax)
            train_accs.append(acc)

            if step % 200 == 0:
                print(f"    step {step:>5} | loss={curr_loss:.4f} | lmax={lmax:.2f} | acc={acc:.3f} | 2/eta={2/LR:.1f}")

    return {"steps": steps, "losses": losses, "lambda_max": lmax_vals, "train_accs": train_accs}

if __name__ == "__main__":
    print(f"Edge of Stability on MNIST (n={N_TRAIN}, hidden={HIDDEN}, lr={LR})")
    print(f"2/eta = {2/LR:.1f}")
    print()

    all_results = []
    t_start = time.time()

    for seed in SEEDS[:3]:
        print(f"Seed {seed}:")
        r = run_eos_mnist(seed)
        all_results.append(r)
        print(f"  Final: loss={r['losses'][-1]:.4f}, lmax={r['lambda_max'][-1]:.2f}, acc={r['train_accs'][-1]:.3f}")
        print()

    t_total = time.time() - t_start
    print(f"Total time: {t_total:.1f}s")

    # Analysis: did lambda_max reach 2/eta?
    threshold = 2.0 / LR
    print(f"\n--- EoS Analysis (2/eta = {threshold:.1f}) ---")
    for i, r in enumerate(all_results):
        max_lmax = max(r["lambda_max"])
        # Find if lambda_max stabilized near threshold
        late_lmax = r["lambda_max"][-20:]  # Last 200 steps
        mean_late = np.mean(late_lmax)
        std_late = np.std(late_lmax)
        near_threshold = abs(mean_late - threshold) / threshold < 0.3
        print(f"  Seed {i}: max_lmax={max_lmax:.2f}, late_mean={mean_late:.2f}+/-{std_late:.2f}, near_2/eta={near_threshold}")

    # Save
    config = {
        "experiment_name": EXPERIMENT_NAME, "version": 1, "date": "2026-04-07",
        "hardware": get_hardware_info(), "seeds": SEEDS[:3],
        "n_train": N_TRAIN, "lr": LR, "hidden": HIDDEN, "n_steps": N_STEPS,
    }
    results_save = {"threshold": threshold, "seeds": [
        {"steps": r["steps"], "losses": r["losses"], "lambda_max": r["lambda_max"], "train_accs": r["train_accs"]}
        for r in all_results
    ]}
    save_results(results_save, OUTPUT_DIR, config)

    # Figure
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, r in enumerate(all_results):
        ax1.plot(r["steps"], r["lambda_max"], alpha=0.7, linewidth=1, label=f'Seed {SEEDS[i]}')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'$2/\\eta = {threshold:.0f}$')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('$\\lambda_{\\max}(\\nabla^2 L)$')
    ax1.set_title(f'Edge of Stability on MNIST (lr={LR})')
    ax1.legend()

    for i, r in enumerate(all_results):
        ax2.plot(r["steps"], r["losses"], alpha=0.7, linewidth=1)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Loss Dynamics')
    ax2.set_yscale('log')

    plt.tight_layout()
    save_figure(fig, "fig4b_eos_mnist", FIGURE_DIR)
    plt.close()
    print(f"\nFigure saved to {FIGURE_DIR}")
