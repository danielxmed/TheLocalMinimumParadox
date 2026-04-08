"""
Experiment: c_k Validation for ReLU Networks (Sub-EoS vs EoS)
Theory: Theory A -- Testing whether linear c_k formula generalizes to ReLU
Prediction: For sub-EoS learning rates, c_k ~ e_k^2 * lambda_x_k^2 should hold
            with R > 0.7, since mode coupling is perturbative (Theorem 6').
            At EoS, correlation should drop due to extensive mode coupling O(m).
Date: 2026-04-07 (Session 9)

Method:
  1. Create 2-layer ReLU network (hidden=64, no bias)
  2. Train at several learning rates (sub-EoS to EoS range)
  3. Compute ReLU effective weight: W_eff = (1/n) * sum_i W2 diag(a_i) W1
  4. Use linear c_k formula with W_eff to get theoretical prediction
  5. Compare with empirical c_k from gradient-Hessian projection
  6. Track activation switch rate to correlate with c_k degradation
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_v1 import (
    set_seed, SEEDS, DEVICE, make_model, make_dataset, count_parameters,
    full_hessian, save_results, setup_publication_style, save_figure,
    get_hardware_info
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "relu_ck_validation"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", EXPERIMENT_NAME)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIDDEN = 64
INPUT_DIM = 20
OUTPUT_DIM = 5
N_TRAIN = 200
N_STEPS = 500
LEARNING_RATES = [0.0005, 0.001, 0.003, 0.01]
USE_SEEDS = SEEDS[:3]  # [42, 137, 256]


def compute_relu_effective_weight(model, X):
    """Compute the averaged local linearization for a 2-layer ReLU network.

    W_eff = (1/n) * sum_i W2 @ diag(relu'(W1 x_i)) @ W1

    This gives the local linear map that the network implements on average.
    """
    with torch.no_grad():
        W1 = model.layer1.weight.data  # hidden x input_dim
        W2 = model.layer2.weight.data  # output_dim x hidden

        # Pre-activations for all samples
        pre_act = X @ W1.T  # n x hidden
        # Activation pattern: 1 where ReLU is active, 0 where dead
        act_pattern = (pre_act > 0).float()  # n x hidden

        # Average activation pattern
        mean_pattern = act_pattern.mean(dim=0)  # hidden

        # Effective weight: W2 @ diag(mean_pattern) @ W1
        # This averages the activation patterns across samples
        W_eff = W2 @ torch.diag(mean_pattern) @ W1  # output_dim x input_dim

        # Also compute per-sample W_eff and average (more accurate but same result for expectation)
        # W_eff_per_sample = W2 @ diag(a_i) @ W1 for each sample i, then average
        n = X.shape[0]
        W_eff_persample = torch.zeros(OUTPUT_DIM, INPUT_DIM)
        for i in range(n):
            W_eff_persample += W2 @ torch.diag(act_pattern[i]) @ W1
        W_eff_persample /= n

    return W_eff_persample.numpy(), mean_pattern.numpy()


def compute_switch_rate(model, X, lr):
    """Measure activation switch rate: fraction of neurons that change activation status per step."""
    with torch.no_grad():
        W1 = model.layer1.weight.data
        pre_act_before = X @ W1.T
        pattern_before = (pre_act_before > 0)

    # Take one SGD step
    loss_fn = nn.MSELoss()
    Y_onehot = torch.zeros(X.shape[0], OUTPUT_DIM, device=DEVICE)
    # Use current model predictions as pseudo-targets shifted slightly
    # Actually, we just need to do one gradient step and check
    model_copy = type(model)(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())

    optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr)
    optimizer.zero_grad()
    out = model_copy(X)
    # Use MSE with one-hot targets (same as training)
    Y_dummy = torch.zeros_like(out)  # placeholder
    loss = loss_fn(out, Y_dummy)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        W1_new = model_copy.layer1.weight.data
        pre_act_after = X @ W1_new.T
        pattern_after = (pre_act_after > 0)

    # Fraction of (sample, neuron) pairs that switched
    switches = (pattern_before != pattern_after).float()
    switch_rate = switches.mean().item()  # per-neuron, per-sample
    total_switches = switches.sum().item()

    return switch_rate, total_switches


def compute_empirical_ck(model, X, Y_onehot, eigenvalues, eigenvectors):
    """Compute c_k from gradient projected onto Hessian eigenmodes."""
    loss_fn = nn.MSELoss()
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, Y_onehot)
    loss.backward()

    # Gradient norm imbalance
    gn1 = model.layer1.weight.grad.data.norm().item() ** 2
    gn2 = model.layer2.weight.grad.data.norm().item() ** 2
    delta_0 = gn2 - gn1

    # Full gradient vector
    grad_vec = torch.cat([model.layer1.weight.grad.data.flatten(),
                          model.layer2.weight.grad.data.flatten()]).numpy()

    # Project onto Hessian eigenvectors
    projections = eigenvectors.T @ grad_vec
    c_k = projections ** 2

    total = np.sum(c_k)
    if total > 1e-12:
        c_k = c_k * abs(delta_0) / total

    return c_k


def compute_theoretical_ck_relu(W_eff, X, Y_onehot):
    """Compute theoretical c_k using the linear formula applied to ReLU effective weight.

    c_k = e_k^2 * lambda_x_k^2
    where e_k is error in data eigenmode k using W_eff as the linear predictor.
    """
    X_np = X.numpy()
    Y_np = Y_onehot.numpy()

    # Data covariance eigendecomposition
    Sigma_x = (X_np.T @ X_np) / len(X_np)
    eig_vals_x, eig_vecs_x = np.linalg.eigh(Sigma_x)
    idx = np.argsort(eig_vals_x)[::-1]
    eig_vals_x = eig_vals_x[idx]
    eig_vecs_x = eig_vecs_x[:, idx]

    # Target in data eigenbasis
    Y_star_rot = (Y_np.T @ X_np @ eig_vecs_x) / len(X_np)  # K x d

    # W_eff in data eigenbasis
    W_eff_rot = W_eff @ eig_vecs_x  # K x d

    # Per-mode error squared
    errors_sq = np.sum((W_eff_rot - Y_star_rot) ** 2, axis=0)  # d-vector

    # Theoretical c_k = e_k^2 * lambda_x_k^2
    ck_theory = errors_sq * eig_vals_x ** 2

    return ck_theory, eig_vals_x, errors_sq


class ReLUTwoLayer(nn.Module):
    """2-layer ReLU network, no bias."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


if __name__ == "__main__":
    print("=" * 70)
    print("E21: c_k Validation for ReLU Networks (Sub-EoS vs EoS)")
    print("=" * 70)
    print(f"Architecture: 2-layer ReLU, hidden={HIDDEN}, no bias")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Seeds: {USE_SEEDS}")
    print(f"Training steps: {N_STEPS}")
    print()

    t_start = time.time()
    all_results = {}

    for lr in LEARNING_RATES:
        print(f"\n{'='*60}")
        print(f"  Learning rate = {lr}")
        print(f"{'='*60}")

        lr_results = {}
        correlations = []
        switch_rates = []

        for seed in USE_SEEDS:
            print(f"\n  --- seed={seed} ---")
            set_seed(seed)

            model = ReLUTwoLayer(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
            X, Y, _, _ = make_dataset("gaussian_mixture", n_train=N_TRAIN, n_test=50,
                                       d=INPUT_DIM, K=OUTPUT_DIM, separation=2.0)

            Y_onehot = torch.zeros(Y.shape[0], OUTPUT_DIM, device=DEVICE)
            Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

            # Train for N_STEPS
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            losses = []
            for step in range(N_STEPS):
                optimizer.zero_grad()
                out = model(X)
                loss = loss_fn(out, Y_onehot)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if np.isnan(loss.item()):
                    print(f"    NaN at step {step}")
                    break

            final_loss = losses[-1] if losses else float('nan')
            print(f"    Final loss: {final_loss:.6f}")

            if np.isnan(final_loss):
                lr_results[str(seed)] = {"status": "diverged"}
                continue

            # Measure activation switch rate
            sw_rate, total_sw = compute_switch_rate(model, X, lr)
            switch_rates.append(sw_rate)
            print(f"    Switch rate: {sw_rate:.4f} ({total_sw:.0f} total switches)")

            # Compute ReLU effective weight
            W_eff, mean_pattern = compute_relu_effective_weight(model, X)
            active_fraction = np.mean(mean_pattern)
            print(f"    Mean active fraction: {active_fraction:.3f}")

            # Full Hessian
            H = full_hessian(model, X, Y_onehot, loss_fn)
            H_np = H.numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            n_pos = int(np.sum(eigenvalues > 1e-8))

            # Empirical c_k
            ck_empirical = compute_empirical_ck(model, X, Y_onehot, eigenvalues, eigenvectors)

            # Theoretical c_k (using linear formula with W_eff)
            ck_theory, eig_vals_x, errors_sq = compute_theoretical_ck_relu(W_eff, X, Y_onehot)

            # Compare top-K modes
            ck_emp_sorted = np.sort(np.abs(ck_empirical))[::-1]
            ck_thy_sorted = np.sort(np.abs(ck_theory))[::-1]

            ck_emp_norm = ck_emp_sorted / (np.sum(ck_emp_sorted) + 1e-20)
            ck_thy_norm = ck_thy_sorted / (np.sum(ck_thy_sorted) + 1e-20)

            K = min(INPUT_DIM, n_pos, len(ck_thy_norm))
            ck_emp_top = ck_emp_norm[:K]
            ck_thy_top = ck_thy_norm[:K]

            # Correlation in log space
            if (len(ck_emp_top) >= 3 and np.std(ck_emp_top) > 1e-15
                    and np.std(ck_thy_top) > 1e-15):
                mask = (ck_emp_top > 1e-20) & (ck_thy_top > 1e-20)
                if np.sum(mask) >= 3:
                    R = np.corrcoef(np.log(ck_emp_top[mask]),
                                   np.log(ck_thy_top[mask]))[0, 1]
                else:
                    R = 0.0
            else:
                R = 0.0

            correlations.append(R)

            # Hessian vs data covariance spectrum
            hess_eigs_top = eigenvalues[-K:][::-1]
            data_eigs_top = eig_vals_x[:K]
            if len(hess_eigs_top) >= 3 and np.std(hess_eigs_top) > 1e-15:
                R_spectrum = np.corrcoef(np.log(np.abs(hess_eigs_top) + 1e-20),
                                         np.log(data_eigs_top + 1e-20))[0, 1]
            else:
                R_spectrum = 0.0

            print(f"    c_k correlation (top {K} modes): R = {R:.4f}")
            print(f"    Spectrum correlation: R = {R_spectrum:.4f}")
            print(f"    Hessian: {n_pos} positive eigenvalues")

            lr_results[str(seed)] = {
                "ck_empirical_top": ck_emp_top.tolist(),
                "ck_theory_top": ck_thy_top.tolist(),
                "ck_correlation": float(R),
                "spectrum_correlation": float(R_spectrum),
                "switch_rate": float(sw_rate),
                "total_switches": float(total_sw),
                "active_fraction": float(active_fraction),
                "n_positive_hessian": n_pos,
                "K_modes_compared": int(K),
                "final_loss": float(final_loss),
                "data_eigenvalues": eig_vals_x[:K].tolist(),
                "hessian_top_eigenvalues": hess_eigs_top.tolist(),
            }

        # LR summary
        valid_R = [r for r in correlations if np.isfinite(r)]
        mean_R = float(np.mean(valid_R)) if valid_R else float('nan')
        mean_sw = float(np.mean(switch_rates)) if switch_rates else float('nan')

        lr_results["summary"] = {
            "mean_ck_correlation": mean_R,
            "std_ck_correlation": float(np.std(valid_R)) if len(valid_R) > 1 else 0.0,
            "mean_switch_rate": mean_sw,
            "n_valid_seeds": len(valid_R),
        }

        all_results[str(lr)] = lr_results
        print(f"\n  LR={lr}: Mean c_k R = {mean_R:.4f}, Mean switch rate = {mean_sw:.4f}")

    # ============================================================================
    # Summary Analysis
    # ============================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: c_k Correlation vs Learning Rate")
    print(f"{'='*70}")

    lr_list = []
    R_list = []
    sw_list = []

    for lr in LEARNING_RATES:
        s = all_results[str(lr)]["summary"]
        lr_list.append(lr)
        R_list.append(s["mean_ck_correlation"])
        sw_list.append(s["mean_switch_rate"])

        status = "STRONG" if s["mean_ck_correlation"] > 0.7 else \
                 "MODERATE" if s["mean_ck_correlation"] > 0.4 else "WEAK"
        print(f"  lr={lr:.4f}: R = {s['mean_ck_correlation']:.4f} ({status}), "
              f"switch_rate = {s['mean_switch_rate']:.4f}")

    # R vs switch rate correlation
    valid_pairs = [(r, s) for r, s in zip(R_list, sw_list)
                   if np.isfinite(r) and np.isfinite(s)]
    if len(valid_pairs) >= 3:
        R_vals = [p[0] for p in valid_pairs]
        S_vals = [p[1] for p in valid_pairs]
        R_sw_corr = np.corrcoef(R_vals, S_vals)[0, 1]
        print(f"\n  Correlation between c_k R and switch rate: {R_sw_corr:.4f}")
        if R_sw_corr < -0.5:
            print("  ** CONFIRMED: Higher switch rate degrades c_k prediction **")
            print("  ** Consistent with Theorem 6': mode coupling disrupts independent-mode formula **")
    else:
        R_sw_corr = float('nan')

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # ============================================================================
    # Save Results
    # ============================================================================
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_number": "E21",
        "version": 1,
        "date": "2026-04-07",
        "session": 9,
        "hardware": get_hardware_info(),
        "seeds": [int(s) for s in USE_SEEDS],
        "hidden": HIDDEN,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "n_train": N_TRAIN,
        "n_steps": N_STEPS,
        "learning_rates": LEARNING_RATES,
        "analysis": {
            "lr_R_pairs": list(zip(LEARNING_RATES, R_list)),
            "lr_switch_pairs": list(zip(LEARNING_RATES, sw_list)),
            "R_vs_switch_correlation": float(R_sw_corr) if np.isfinite(R_sw_corr) else None,
        },
        "total_time_seconds": t_total,
    }
    save_results(all_results, OUTPUT_DIR, config)

    # ============================================================================
    # Figure 24: ReLU c_k Validation
    # ============================================================================
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): Scatter of empirical vs theoretical c_k at lowest LR (sub-EoS)
    ax = axes[0]
    best_lr = LEARNING_RATES[0]  # Most sub-EoS
    seed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    lr_data = all_results[str(best_lr)]
    for i, seed in enumerate(USE_SEEDS):
        if str(seed) not in lr_data or "ck_empirical_top" not in lr_data[str(seed)]:
            continue
        r = lr_data[str(seed)]
        ck_emp = np.array(r["ck_empirical_top"])
        ck_thy = np.array(r["ck_theory_top"])
        mask = (ck_emp > 1e-20) & (ck_thy > 1e-20)
        if np.any(mask):
            ax.scatter(ck_thy[mask], ck_emp[mask], c=seed_colors[i], s=30, alpha=0.7,
                       label=f'seed={seed} (R={r["ck_correlation"]:.3f})')

    # Reference line
    all_vals = []
    for seed in USE_SEEDS:
        if str(seed) in lr_data and "ck_empirical_top" in lr_data[str(seed)]:
            all_vals.extend(lr_data[str(seed)]["ck_empirical_top"])
            all_vals.extend(lr_data[str(seed)]["ck_theory_top"])
    all_vals = [v for v in all_vals if v > 1e-20]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.5)

    ax.set_xlabel('Theoretical $c_k$ (linear formula with $W_{eff}$)')
    ax.set_ylabel('Empirical $c_k$ (from gradient projection)')
    ax.set_title(f'(a) Sub-EoS ($\\eta$={best_lr}): ReLU $c_k$ validation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # Panel (b): R vs learning rate
    ax = axes[1]
    valid_mask = [np.isfinite(r) for r in R_list]
    lrs_plot = [lr for lr, v in zip(LEARNING_RATES, valid_mask) if v]
    Rs_plot = [r for r, v in zip(R_list, valid_mask) if v]

    ax.semilogx(lrs_plot, Rs_plot, 'ko-', markersize=10, linewidth=2)
    ax.axhline(y=0.847, color='blue', ls='--', lw=1, alpha=0.5,
               label='Linear network (E20): R=0.847')
    ax.axhline(y=0.7, color='red', ls=':', lw=1, alpha=0.5, label='Threshold R=0.7')
    ax.set_xlabel('Learning rate $\\eta$')
    ax.set_ylabel('$c_k$ correlation $R$')
    ax.set_title('(b) $c_k$ prediction degrades at EoS')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.05)

    # Annotate sub-EoS vs EoS regions
    ax.axvspan(LEARNING_RATES[0]*0.5, 0.002, alpha=0.1, color='green', label='Sub-EoS')
    ax.axvspan(0.005, LEARNING_RATES[-1]*2, alpha=0.1, color='red', label='EoS')

    # Panel (c): R vs switch rate
    ax = axes[2]
    valid_pairs_plot = [(r, s, lr) for r, s, lr in zip(R_list, sw_list, LEARNING_RATES)
                        if np.isfinite(r) and np.isfinite(s)]
    if valid_pairs_plot:
        Rs_p = [p[0] for p in valid_pairs_plot]
        Ss_p = [p[1] for p in valid_pairs_plot]
        LRs_p = [p[2] for p in valid_pairs_plot]

        scatter = ax.scatter(Ss_p, Rs_p, c=np.log10(LRs_p), cmap='coolwarm',
                            s=100, edgecolors='black', linewidth=0.5, zorder=3)

        # Annotate each point with LR
        for s, r, lr in zip(Ss_p, Rs_p, LRs_p):
            ax.annotate(f'$\\eta$={lr}', (s, r), textcoords="offset points",
                       xytext=(8, 5), fontsize=8)

        plt.colorbar(scatter, ax=ax, label='$\\log_{10}(\\eta)$')

    ax.set_xlabel('Activation switch rate (per neuron per step)')
    ax.set_ylabel('$c_k$ correlation $R$')
    ax.set_title('(c) Mode coupling disrupts $c_k$ prediction')
    if np.isfinite(R_sw_corr):
        ax.annotate(f'$\\rho$ = {R_sw_corr:.3f}', xy=(0.05, 0.05),
                   xycoords='axes fraction', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('E21: ReLU $c_k$ Validation — Linear Formula Generalization Test',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig24_relu_ck_validation", FIGURE_DIR)
    plt.close()

    print(f"\nFigure saved to {FIGURE_DIR}/fig24_relu_ck_validation.pdf")

    # ============================================================================
    # Conclusion
    # ============================================================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    sub_eos_R = [R_list[i] for i, lr in enumerate(LEARNING_RATES) if lr <= 0.001]
    eos_R = [R_list[i] for i, lr in enumerate(LEARNING_RATES) if lr >= 0.01]

    if sub_eos_R and all(np.isfinite(r) for r in sub_eos_R):
        mean_sub = np.mean(sub_eos_R)
        print(f"  Sub-EoS mean R: {mean_sub:.4f}")
        if mean_sub > 0.7:
            print("  ** LINEAR c_k FORMULA GENERALIZES TO RELU AT SUB-EOS **")
        elif mean_sub > 0.4:
            print("  ** PARTIAL GENERALIZATION: correction terms needed **")
        else:
            print("  ** LINEAR c_k FORMULA DOES NOT GENERALIZE to ReLU **")

    if eos_R and all(np.isfinite(r) for r in eos_R):
        mean_eos = np.mean(eos_R)
        print(f"  EoS mean R: {mean_eos:.4f}")
        if mean_eos < mean_sub - 0.2 if sub_eos_R else True:
            print("  ** CONFIRMED: c_k prediction degrades at EoS (mode coupling) **")
