#!/usr/bin/env python3
"""
appendixA_robust.py — Multi-seed robust wrapper for Appendix A validation.

Runs the appendixA_check.py first-order expansion validation across multiple
random seeds, aggregates results (mean ± std), and produces publication-quality
plots with shaded uncertainty bands.

Includes outlier detection: seeds whose truncation error at ε=1 falls outside
1.5×IQR of the distribution are flagged and excluded from aggregation/plotting.

Usage:
  python appendixA_robust.py --num-seeds 20 --device cpu
  python appendixA_robust.py --seeds 0 1 2 3 4
"""

import argparse, os, sys, csv
import numpy as np

# CRITICAL: set float64 BEFORE importing torch or appendixA_check
# to ensure all tensor creation uses float64 from the start.
import torch
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Import core functions from original appendixA_check.py ---- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appendixA_check import (
    BaseCell, GateScalar, GateMulti,
    make_signal, build_factors,
    compute_errors, norms_and_ratios,
    operator_norm,
)


def robust_fit_slope(eps_arr, errs_arr):
    """
    Fit log(error) = intercept + slope * log(eps) via least squares.
    Returns slope. Expects eps_arr and errs_arr as 1D numpy arrays.
    """
    valid = (errs_arr > 0) & (eps_arr > 0) & np.isfinite(errs_arr)
    if valid.sum() < 2:
        return np.nan
    log_eps = np.log(eps_arr[valid])
    log_err = np.log(errs_arr[valid])
    X = np.column_stack([np.ones_like(log_eps), log_eps])
    beta, *_ = np.linalg.lstsq(X, log_err, rcond=None)
    return float(beta[1])


def filter_outliers_iqr(values, factor=1.5):
    """
    Returns a boolean mask of non-outlier indices using the IQR method.
    values: 1D array, one per seed.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (values >= lower) & (values <= upper)


def run_single_seed(gate_type, seed, T, H, ni, device_str, eps_list, fit_first):
    """
    Run one validation sweep for a single seed.
    Returns truncation errors, per-step norms, and summary metrics.
    """
    # Ensure float64 is active
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    cell = BaseCell(ni, H).to(device)
    gate = (GateMulti(ni, H) if gate_type == "multi" else GateScalar(ni, H)).to(device)
    scalar_gate = (gate_type == "scalar")

    # Verify dtype
    assert cell.Wr.weight.dtype == torch.float64, \
        f"Expected float64, got {cell.Wr.weight.dtype}"

    u = make_signal(T=T, B=1, ni=ni, seed=seed, device=device)
    assert u.dtype == torch.float64, f"Signal dtype: {u.dtype}"

    A_list, B_list = build_factors(cell, gate, u, device=device, scalar_gate=scalar_gate)

    # Verify factor dtypes
    assert A_list[0].dtype == torch.float64, f"A dtype: {A_list[0].dtype}"
    assert B_list[0].dtype == torch.float64, f"B dtype: {B_list[0].dtype}"

    # Truncation errors
    errs, F0, L1 = compute_errors(A_list, B_list, eps_list)

    # Slope (robust internal computation)
    eps_arr = np.array(eps_list)
    errs_arr = np.array(errs)
    N = min(max(2, fit_first), len(eps_list))
    slope = robust_fit_slope(eps_arr[:N], errs_arr[:N])

    # Per-step norms
    As_norms = [operator_norm(A) for A in A_list]
    Bs_norms = [operator_norm(B) for B in B_list]
    ratios = [b / (a + 1e-12) for a, b in zip(As_norms, Bs_norms)]

    # C2 remainder
    C2 = errs_arr / (eps_arr ** 2)

    return {
        "errs": errs_arr,
        "C2": C2,
        "slope": slope,
        "As_norms": np.array(As_norms),
        "Bs_norms": np.array(Bs_norms),
        "ratios": np.array(ratios),
        "errs_first3": errs_arr[:3].tolist(),  # for diagnostics
    }


def aggregate_and_plot(all_seed_results, seeds, gate_type, eps_list, T, outdir):
    """
    Aggregate across seeds (with outlier filtering) and produce robust plots.
    """
    os.makedirs(outdir, exist_ok=True)
    n_seeds_total = len(seeds)
    prefix = gate_type
    eps_arr = np.array(eps_list)

    # ---- Outlier detection based on error norm at ε=1 ---- #
    err_at_eps1 = np.array([r["errs"][-1] for r in all_seed_results])
    keep_mask = filter_outliers_iqr(err_at_eps1, factor=1.5)

    # Also filter seeds with non-finite slopes
    slopes_all = np.array([r["slope"] for r in all_seed_results])
    keep_mask = keep_mask & np.isfinite(slopes_all)

    kept_indices = np.where(keep_mask)[0]
    excluded_indices = np.where(~keep_mask)[0]
    n_kept = len(kept_indices)

    kept_seeds = [seeds[i] for i in kept_indices]
    kept_results = [all_seed_results[i] for i in kept_indices]

    if len(excluded_indices) > 0:
        excluded_seeds = [seeds[i] for i in excluded_indices]
        print(f"    [FILTER] Excluded {len(excluded_seeds)} outlier seed(s): "
              f"{excluded_seeds} (err@ε=1: "
              f"{[f'{err_at_eps1[i]:.2e}' for i in excluded_indices]})")
    print(f"    [FILTER] Keeping {n_kept}/{n_seeds_total} seeds")

    if n_kept < 2:
        print(f"    [WARN] Too few seeds after filtering, skipping plots for {gate_type}")
        return

    # Stack filtered arrays
    errs_all = np.stack([r["errs"] for r in kept_results])
    C2_all = np.stack([r["C2"] for r in kept_results])
    As_all = np.stack([r["As_norms"] for r in kept_results])
    Bs_all = np.stack([r["Bs_norms"] for r in kept_results])
    Rs_all = np.stack([r["ratios"] for r in kept_results])
    slopes = np.array([r["slope"] for r in kept_results])

    # --- Truncation error vs eps (median + IQR) --- #
    med_err = np.median(errs_all, axis=0)
    q25_err = np.percentile(errs_all, 25, axis=0)
    q75_err = np.percentile(errs_all, 75, axis=0)

    plt.figure()
    plt.loglog(eps_arr, med_err, marker='o', lw=2, color='C0',
               label='median')
    plt.fill_between(eps_arr, q25_err, q75_err,
                     alpha=0.25, color='C0', label='IQR (25–75%)')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$E_{\mathrm{trunc}}(\varepsilon)$')
    plt.title(f'Truncation error vs $\\varepsilon$ ({gate_type} gate)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_robust_error_vs_eps.png"), dpi=600)
    plt.close()

    # --- C2 remainder vs eps (median + IQR) --- #
    med_C2 = np.median(C2_all, axis=0)
    q25_C2 = np.percentile(C2_all, 25, axis=0)
    q75_C2 = np.percentile(C2_all, 75, axis=0)

    plt.figure()
    plt.semilogx(eps_arr, med_C2, marker='o', lw=2, color='C0', label='median')
    plt.fill_between(eps_arr, q25_C2, q75_C2,
                     alpha=0.25, color='C0', label='IQR (25–75%)')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$C_2(\varepsilon)$')
    plt.title(f'Normalized remainder $C_2(\\varepsilon)$ ({gate_type} gate)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_robust_C2_vs_eps.png"), dpi=600)
    plt.close()

    # --- Per-step norms timeseries (mean ± std bands) --- #
    t_steps = np.arange(As_all.shape[1])
    mean_As = As_all.mean(axis=0)
    std_As = As_all.std(axis=0)
    mean_Bs = Bs_all.mean(axis=0)
    std_Bs = Bs_all.std(axis=0)
    mean_Rs = Rs_all.mean(axis=0)
    std_Rs = Rs_all.std(axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_steps, mean_As, lw=2, color='C0',
             label=r'$\|A_j\|_2$')
    ax1.fill_between(t_steps, mean_As - std_As, mean_As + std_As,
                     alpha=0.2, color='C0')
    ax1.plot(t_steps, mean_Bs, lw=2, color='C1',
             label=r'$\|B_j\|_2$')
    ax1.fill_between(t_steps, mean_Bs - std_Bs, mean_Bs + std_Bs,
                     alpha=0.2, color='C1')
    ax1.set_xlabel('time step $j$')
    ax1.set_ylabel(r'$\ell_2$ norm')
    ax2 = ax1.twinx()
    ax2.plot(t_steps, mean_Rs, lw=1.5, linestyle=':', color='C2',
             label=r'$r_j$', alpha=0.8)
    ax2.fill_between(t_steps, mean_Rs - std_Rs, mean_Rs + std_Rs,
                     alpha=0.15, color='C2')
    ax2.set_ylabel(r'$r_j$')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=8, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f'Per-step $\\ell_2$ norms ({gate_type} gate)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_robust_norms_timeseries.png"), dpi=600)
    plt.close(fig)

    # --- Ratio histogram (overlaid from kept seeds) --- #
    plt.figure()
    for i, (seed, r) in enumerate(zip(kept_seeds, kept_results)):
        plt.hist(r["ratios"], bins=30, alpha=max(0.15, 0.5 / n_kept * 5),
                 label=f'seed={seed}' if n_kept <= 10 else None)
    plt.xlabel(r'$r_j$')
    plt.ylabel('count')
    plt.title(f'Distribution of $r_j$ ({gate_type} gate)')
    if n_kept <= 10:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_robust_ratio_hist.png"), dpi=600)
    plt.close()

    # --- Summary CSV --- #
    with open(os.path.join(outdir, f"{prefix}_robust_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "median", "n_kept", "n_total",
                     "excluded_seeds"])
        excluded_str = ";".join(str(seeds[i]) for i in excluded_indices)
        w.writerow(["small_eps_slope",
                     f"{slopes.mean():.4f}", f"{slopes.std():.4f}",
                     f"{np.median(slopes):.4f}",
                     n_kept, n_seeds_total, excluded_str])
        per_seed_medians = np.array([np.median(r['ratios']) for r in kept_results])
        w.writerow(["ratio_median",
                     f"{per_seed_medians.mean():.6e}",
                     f"{per_seed_medians.std():.6e}",
                     f"{np.median(per_seed_medians):.6e}",
                     n_kept, n_seeds_total, excluded_str])

    print(f"  [{gate_type.upper()}] {n_kept}/{n_seeds_total} seeds: "
          f"slope={slopes.mean():.2f}±{slopes.std():.2f} "
          f"(median={np.median(slopes):.2f}, expect ≈2)")


def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed robust runner for Appendix A validation")
    ap.add_argument('--num-seeds', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--input-dim', type=int, default=3)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--eps', type=str,
                    default='1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
    ap.add_argument('--fit-first', type=int, default=5)
    ap.add_argument('--out', type=str, default='appendixA_robust')
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(range(0, args.num_seeds))
    eps_list = [float(x) for x in args.eps.split(",") if x.strip()]

    print(f"=== Appendix A Robust Runner ===")
    print(f"Seeds: {seeds}")
    print(f"Torch default dtype: {torch.get_default_dtype()}")
    print()

    os.makedirs(args.out, exist_ok=True)

    for gate_type in ["scalar", "multi"]:
        print(f"[AppA] Running {gate_type} gate across {len(seeds)} seeds...")
        all_seed_results = []
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            res = run_single_seed(
                gate_type=gate_type, seed=seed,
                T=args.T, H=args.hidden, ni=args.input_dim,
                device_str=args.device, eps_list=eps_list,
                fit_first=args.fit_first
            )
            all_seed_results.append(res)
            # Diagnostic: print first 3 error values + slope
            e3 = res["errs_first3"]
            print(f"done (slope={res['slope']:.3f}, "
                  f"err@ε=[1e-5,3e-5,1e-4]=[{e3[0]:.2e},{e3[1]:.2e},{e3[2]:.2e}])")

        # Cross-check: compute slope from median errors
        med_errs = np.median(np.stack([r["errs"] for r in all_seed_results]), axis=0)
        eps_arr = np.array(eps_list)
        N = min(max(2, args.fit_first), len(eps_list))
        global_slope = robust_fit_slope(eps_arr[:N], med_errs[:N])
        print(f"  [DIAGNOSTIC] Slope from median errors (first {N} eps): {global_slope:.3f}")

        aggregate_and_plot(
            all_seed_results, seeds, gate_type,
            eps_list, args.T, args.out
        )

    print("\n=== Appendix A Robust Runner Complete ===")


if __name__ == "__main__":
    main()
