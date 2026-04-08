"""
Experiment: EoS Phase Decomposition of Gradient Imbalance Sum
Theory: Theory A -- Does the drift accumulate mainly before or during EoS?
Prediction: Pre-EoS accumulates coherently (S grows steadily).
            At-EoS accumulation slows due to EoS self-tuning dynamics.
Date: 2026-04-07 (Session 4)

We decompose S(eta) into contributions from three training phases:
  Phase 1: Pre-EoS (lambda_max < 0.5 * 2/eta)
  Phase 2: EoS onset (0.5 * 2/eta < lambda_max < 2/eta)
  Phase 3: At EoS (lambda_max oscillating around 2/eta)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset,
    top_hessian_eigenvalue,
    save_results, setup_publication_style, save_figure, get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "eos_gradient_decomposition"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LRS = [0.001, 0.01, 0.1, 0.5]
N_STEPS = 2000
HIDDEN = 64
HESSIAN_EVERY = 50  # Compute lambda_max every N steps (expensive)


def run_eos_decomposition(lr, seed):
    """Track gradient imbalance AND lambda_max to decompose by EoS phase."""
    set_seed(seed)
    model = make_model("mlp_2layer_nobias", 20, 5, hidden=HIDDEN)
    X, Y, _, _ = make_dataset("gaussian_mixture", n_train=200, n_test=50,
                               d=20, K=5, separation=2.0)

    weight_params = [(n, p) for n, p in model.named_parameters() if 'weight' in n]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    eos_threshold = 2.0 / lr

    # Tracking
    grad_imbalances = []
    lambda_max_history = []
    loss_history = []
    phases = []  # 0=pre-EoS, 1=onset, 2=at-EoS

    current_lmax = 0.0

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()

        # Gradient imbalance
        gn = [p.grad.data.norm().item()**2 for _, p in weight_params]
        delta = gn[1] - gn[0]
        grad_imbalances.append(delta)
        loss_history.append(loss.item())

        # Periodic Hessian computation
        if step % HESSIAN_EVERY == 0:
            try:
                current_lmax = top_hessian_eigenvalue(model, X, Y, loss_fn, n_iter=20)
            except Exception:
                current_lmax = 0.0

        lambda_max_history.append(current_lmax)

        # Phase classification
        if current_lmax < 0.5 * eos_threshold:
            phases.append(0)  # Pre-EoS
        elif current_lmax < eos_threshold:
            phases.append(1)  # Onset
        else:
            phases.append(2)  # At EoS

        optimizer.step()

    # Decompose S by phase
    gi = np.array(grad_imbalances)
    ph = np.array(phases)

    S_total = abs(gi.sum())
    S_pre = abs(gi[ph == 0].sum()) if (ph == 0).any() else 0.0
    S_onset = abs(gi[ph == 1].sum()) if (ph == 1).any() else 0.0
    S_eos = abs(gi[ph == 2].sum()) if (ph == 2).any() else 0.0

    n_pre = (ph == 0).sum()
    n_onset = (ph == 1).sum()
    n_eos = (ph == 2).sum()

    # Per-step magnitude in each phase
    avg_pre = np.mean(np.abs(gi[ph == 0])) if n_pre > 0 else 0.0
    avg_onset = np.mean(np.abs(gi[ph == 1])) if n_onset > 0 else 0.0
    avg_eos = np.mean(np.abs(gi[ph == 2])) if n_eos > 0 else 0.0

    max_lmax = max(lambda_max_history)
    eos_ratio = max_lmax / eos_threshold

    return {
        "lr": lr, "seed": seed,
        "S_total": float(S_total),
        "S_pre_eos": float(S_pre),
        "S_onset": float(S_onset),
        "S_at_eos": float(S_eos),
        "n_steps_pre": int(n_pre),
        "n_steps_onset": int(n_onset),
        "n_steps_eos": int(n_eos),
        "avg_delta_pre": float(avg_pre),
        "avg_delta_onset": float(avg_onset),
        "avg_delta_eos": float(avg_eos),
        "max_lambda_max": float(max_lmax),
        "eos_threshold": float(eos_threshold),
        "eos_ratio": float(eos_ratio),
        "eos_reached": bool(max_lmax > eos_threshold),
        "final_loss": float(loss_history[-1]),
        # Downsampled trajectories
        "lambda_max_traj": [float(x) for x in np.array(lambda_max_history)[::10]],
        "cumulative_S_traj": [float(x) for x in np.cumsum(gi)[::10]],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("EoS Phase Decomposition of Gradient Imbalance")
    print("=" * 70)
    print(f"Learning rates: {LRS}")
    print(f"Hessian computed every {HESSIAN_EVERY} steps")
    print()

    all_results = {}
    t_start = time.time()

    for lr in LRS:
        print(f"\n--- Learning rate: {lr} (EoS threshold = {2/lr:.1f}) ---")
        seed_results = []
        for seed in SEEDS[:3]:  # 3 seeds (Hessian computation is expensive)
            r = run_eos_decomposition(lr, seed)
            seed_results.append(r)
            print(f"  seed={seed}: S_total={r['S_total']:.4f}, "
                  f"S_pre={r['S_pre_eos']:.4f} ({r['n_steps_pre']} steps), "
                  f"S_eos={r['S_at_eos']:.4f} ({r['n_steps_eos']} steps), "
                  f"lmax_max={r['max_lambda_max']:.2f} ({r['eos_ratio']:.1%} of 2/eta)")

        # Aggregate
        agg = {k: np.mean([r[k] for r in seed_results])
               for k in ['S_total', 'S_pre_eos', 'S_onset', 'S_at_eos',
                         'avg_delta_pre', 'avg_delta_onset', 'avg_delta_eos',
                         'max_lambda_max', 'eos_ratio']}
        agg['lr'] = lr
        agg['eos_reached'] = any(r['eos_reached'] for r in seed_results)
        agg['seed_results'] = seed_results

        # Phase fractions
        frac_pre = np.mean([r['S_pre_eos'] / max(r['S_total'], 1e-15) for r in seed_results])
        frac_eos = np.mean([r['S_at_eos'] / max(r['S_total'], 1e-15) for r in seed_results])
        agg['frac_pre_eos'] = float(frac_pre)
        agg['frac_at_eos'] = float(frac_eos)

        print(f"  AGGREGATE: S_total={agg['S_total']:.4f}")
        print(f"    Pre-EoS: {frac_pre:.1%} of S | At-EoS: {frac_eos:.1%} of S")
        print(f"    Per-step |delta|: pre={agg['avg_delta_pre']:.6f}, eos={agg['avg_delta_eos']:.6f}")

        all_results[str(lr)] = agg

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # === Save ===
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "version": 1,
        "date": "2026-04-07",
        "session": 4,
        "hardware": get_hardware_info(),
        "seeds": SEEDS[:3],
        "lrs": LRS,
        "hidden": HIDDEN,
        "n_steps": N_STEPS,
        "hessian_every": HESSIAN_EVERY,
    }
    summary = {k: {kk: v for kk, v in d.items() if kk != "seed_results"}
               for k, d in all_results.items()}
    save_results(summary, OUTPUT_DIR, config)

    # === Figure ===
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Stacked bar of S contributions by phase
    ax = axes[0]
    x_pos = np.arange(len(LRS))
    s_pre = [all_results[str(lr)]['S_pre_eos'] for lr in LRS]
    s_onset = [all_results[str(lr)].get('S_onset', 0) for lr in LRS]
    s_eos = [all_results[str(lr)]['S_at_eos'] for lr in LRS]
    s_onset_mean = [np.mean([r['S_onset'] for r in all_results[str(lr)]['seed_results']]) for lr in LRS]

    ax.bar(x_pos, s_pre, color='steelblue', label='Pre-EoS')
    ax.bar(x_pos, s_onset_mean, bottom=s_pre, color='orange', label='Onset')
    ax.bar(x_pos, s_eos, bottom=[a+b for a,b in zip(s_pre, s_onset_mean)],
           color='firebrick', label='At EoS')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{lr}' for lr in LRS])
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('$|S(\\eta)|$')
    ax.set_title('(a) S decomposed by EoS phase')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Panel (b): Per-step |delta| in each phase
    ax = axes[1]
    avg_pre_arr = [all_results[str(lr)]['avg_delta_pre'] for lr in LRS]
    avg_eos_arr = [all_results[str(lr)]['avg_delta_eos'] for lr in LRS]
    ax.semilogy(LRS, avg_pre_arr, 'o-', color='steelblue', label='Pre-EoS', markersize=7)
    ax.semilogy(LRS, [max(x, 1e-15) for x in avg_eos_arr], 's-', color='firebrick',
                label='At EoS', markersize=7)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('Mean per-step $|\\delta(t)|$')
    ax.set_title('(b) Per-step imbalance by phase')
    ax.legend(fontsize=10)

    # Panel (c): lambda_max and cumulative S trajectories for two LRs
    ax = axes[2]
    for lr, color, ls in [(LRS[0], 'steelblue', '-'), (LRS[-1], 'firebrick', '-')]:
        traj = all_results[str(lr)]['seed_results'][0]
        steps = np.arange(0, N_STEPS, 10)[:len(traj['cumulative_S_traj'])]
        cum_s = np.array(traj['cumulative_S_traj'])
        ax.plot(steps, cum_s / max(abs(cum_s[-1]), 1e-10), color=color, ls=ls,
                label=f'$\\eta$={lr}', lw=1.5)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Cumulative $S(t) / S(T)$ (normalized)')
    ax.set_title('(c) S accumulation over training')
    ax.legend(fontsize=10)

    fig.suptitle('EoS Phase Decomposition of Drift', fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig9_eos_phase_decomposition", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}")
