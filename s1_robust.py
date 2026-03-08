#!/usr/bin/env python3
"""
s1_robust.py — Multi-seed robust wrapper for the Effective LR experiment (S1).

Runs the s1_effective_lr.py training+probing logic across multiple random seeds,
aggregates results (mean ± std), and produces publication-quality plots with
shaded uncertainty bands and error bars.

Usage:
  python s1_robust.py --num-seeds 5 --tasks adding --models leaky scalar multigate --device cpu
  python s1_robust.py --seeds 1 2 3 4 5 --tasks adding --models leaky scalar multigate
"""

import argparse, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Import core functions from original s1_effective_lr.py ---- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_effective_lr import (
    set_seed, build_model, make_batch,
    compute_J_blocks, product_sensitivities, gate_product_predictor,
    per_lag_median_profile, fit_loglog_slope, to_numpy
)


def run_single_seed(task, model_kind, seed, T, Btrain, Bprobe, ni, nh, no,
                    alpha, mu, steps, checkpoints, trim, power_iters, device):
    """
    Run one full training + probing cycle for a single seed.
    Returns a dict with per-checkpoint results.
    """
    set_seed(seed)
    dev = torch.device(device)
    gen_train = torch.Generator(device=dev).manual_seed(seed + 123)
    gen_probe = torch.Generator(device=dev).manual_seed(seed + 999)

    model = build_model(model_kind, ni=ni, nh=nh, no=no, alpha=alpha).to(dev)
    optim = torch.optim.SGD(model.parameters(), lr=mu)

    ckpts = sorted(set(checkpoints))
    ckpt_set = set(ckpts)

    U_probe, Y_probe = make_batch(task, T, Bprobe, ni, dev, gen_probe)

    results = {}

    def probe(ckpt_iter):
        X, As, Phis, G, AG, yhat = model.forward_full(U_probe)
        Js = compute_J_blocks(model, X, As, G, AG)
        S = product_sensitivities(Js, iters=power_iters)
        GP = gate_product_predictor(
            model, G if model_kind != 'leaky' else None,
            T, Bprobe, dev, alpha=alpha
        )
        s, r2, lo, hi = fit_loglog_slope(S, GP, trim=trim)
        S_med, GP_med = per_lag_median_profile(S, GP)
        eps = 1e-12
        S_norm = S_med / (S_med[0] + eps)
        P0_norm = GP_med / (GP_med[0] + eps)
        Pf_norm = torch.pow(P0_norm + eps, s if np.isfinite(s) else 1.0)

        results[ckpt_iter] = {
            "slope": float(s) if np.isfinite(s) else None,
            "R2": float(r2) if np.isfinite(r2) else None,
            "S_norm": to_numpy(S_norm).tolist(),
            "P0_norm": to_numpy(P0_norm).tolist(),
            "Pf_norm": to_numpy(Pf_norm).tolist(),
        }

    for it in range(steps + 1):
        if it in ckpt_set:
            probe(it)
        if it == steps:
            break
        U_tr, Y_tr = make_batch(task, T, Btrain, ni, dev, gen_train)
        X, As, Phis, G, AG, y_hat = model.forward_full(U_tr)
        loss = F.mse_loss(y_hat, Y_tr)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return results


def aggregate_and_plot(all_seed_results, seeds, task, model_kind, checkpoints, T, outdir):
    """
    Aggregate across seeds and produce robust plots.
    all_seed_results: list of dicts (one per seed), each mapping ckpt_iter -> result dict.
    """
    os.makedirs(outdir, exist_ok=True)
    ckpts = sorted(checkpoints)
    lags = np.arange(1, T)
    n_seeds = len(seeds)

    # Collect per-checkpoint arrays
    agg = {}
    for it in ckpts:
        profiles = []
        p0_profiles = []
        pf_profiles = []
        slopes = []
        r2s = []
        for sr in all_seed_results:
            if it in sr and sr[it]["slope"] is not None:
                profiles.append(np.array(sr[it]["S_norm"]))
                p0_profiles.append(np.array(sr[it]["P0_norm"]))
                pf_profiles.append(np.array(sr[it]["Pf_norm"]))
                slopes.append(sr[it]["slope"])
                r2s.append(sr[it]["R2"])
        if len(profiles) == 0:
            continue
        agg[it] = {
            "profiles": np.array(profiles),
            "p0_profiles": np.array(p0_profiles),
            "pf_profiles": np.array(pf_profiles),
            "slopes": np.array(slopes),
            "r2s": np.array(r2s),
        }

    # --- Per-checkpoint profile plots with shaded bands --- #
    for it, d in agg.items():
        mean_prof = d["profiles"].mean(axis=0)
        std_prof = d["profiles"].std(axis=0)
        mean_p0 = d["p0_profiles"].mean(axis=0)
        mean_pf = d["pf_profiles"].mean(axis=0)
        mean_s = d["slopes"].mean()
        mean_r2 = d["r2s"].mean()

        plt.figure(figsize=(7, 5))
        plt.plot(lags, mean_prof, lw=2, color='C0',
                 label=r'Empirical $\tilde{\mu}_{\mathrm{eff}}(h;\ell)$ (mean)')
        plt.fill_between(lags, mean_prof - std_prof, mean_prof + std_prof,
                         alpha=0.25, color='C0', label=r'$\pm 1$ std')
        plt.plot(lags, mean_p0, lw=2, linestyle='--', color='C1',
                 label=r'Zeroth-order $\tilde{\mu}^{(0)}_{\mathrm{pred}}$')
        plt.plot(lags, mean_pf, lw=2, linestyle='-.', color='C2',
                 label=r'Fitted-power $\tilde{\mu}^{(\mathrm{fit})}$'
                       f', $s={mean_s:.2f}$, $R^2={mean_r2:.2f}$')
        plt.xlabel('Lag $h$')
        plt.ylabel('Normalized effective LR (lag-wise)')
        plt.title(f'Effective LR profile (task={task}, model={model_kind}, iter={it})')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"s1_profile_iter{it}_{model_kind}.png"), dpi=600)
        plt.close()

    # --- Slope vs iteration with error bars --- #
    if len(agg) >= 2:
        its = sorted(agg.keys())
        mean_slopes = [agg[it]["slopes"].mean() for it in its]
        std_slopes = [agg[it]["slopes"].std() for it in its]

        plt.figure(figsize=(7, 5))
        plt.errorbar(its, mean_slopes, yerr=std_slopes, marker='o', lw=2,
                     capsize=5, capthick=1.5, label='slope $s(\\ell)$')
        plt.xlabel('Iteration')
        plt.ylabel('Log–log slope $s(\\ell)$')
        plt.title(f'Log–log slope over training (task={task}, model={model_kind})')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"s1_slope_vs_iter_{model_kind}.png"), dpi=600)
        plt.close()

    # --- Averaged heatmap --- #
    if len(agg) >= 2:
        its = sorted(agg.keys())
        H_all = np.stack([agg[it]["profiles"].mean(axis=0) for it in its], axis=0)

        plt.figure(figsize=(8, 3 + 0.12 * len(its)))
        im = plt.imshow(H_all, aspect='auto', origin='lower',
                        extent=[lags[0], lags[-1], 0, len(its) - 1])
        plt.colorbar(im, label='Normalized effective LR (averaged)')
        plt.yticks(np.arange(len(its)), [str(c) for c in its])
        plt.xlabel('Lag $h$')
        plt.ylabel('Checkpoint (iteration)')
        plt.title(f'Effective LR profile heatmap (task={task}, model={model_kind})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"s1_heatmap_{model_kind}.png"), dpi=600)
        plt.close()

    # --- Save JSON summary --- #
    summary = {
        "task": task,
        "model": model_kind,
        "seeds": seeds,
        "n_seeds": n_seeds,
        "checkpoints": ckpts,
        "aggregated": {},
        "per_seed": {}
    }
    for it, d in agg.items():
        summary["aggregated"][int(it)] = {
            "slope_mean": float(d["slopes"].mean()),
            "slope_std": float(d["slopes"].std()),
            "R2_mean": float(d["r2s"].mean()),
            "R2_std": float(d["r2s"].std()),
        }
    for i, seed in enumerate(seeds):
        summary["per_seed"][int(seed)] = {}
        sr = all_seed_results[i]
        for it in ckpts:
            if it in sr:
                summary["per_seed"][int(seed)][int(it)] = sr[it]

    with open(os.path.join(outdir, f"s1_robust_summary_{model_kind}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [DONE] {task}/{model_kind}: {n_seeds} seeds, "
          f"outputs in {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed robust runner for S1 (Effective LR profiles)")
    ap.add_argument('--num-seeds', type=int, default=5,
                    help='Number of seeds (generates 1..N)')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Explicit seed list (overrides --num-seeds)')
    ap.add_argument('--tasks', type=str, nargs='+', default=['adding'],
                    choices=['adding', 'recall', 'sinmix', 'short'])
    ap.add_argument('--models', type=str, nargs='+',
                    default=['leaky', 'scalar', 'multigate'],
                    choices=['leaky', 'scalar', 'multigate'])
    ap.add_argument('--alpha', type=float, default=0.8)
    ap.add_argument('--T', type=int, default=80)
    ap.add_argument('--Btrain', type=int, default=64)
    ap.add_argument('--Bprobe', type=int, default=12)
    ap.add_argument('--ni', type=int, default=4)
    ap.add_argument('--nh', type=int, default=64)
    ap.add_argument('--no', type=int, default=1)
    ap.add_argument('--mu', type=float, default=1e-2)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--checkpoints', type=str, default='0,50,200,400,600')
    ap.add_argument('--trim', type=float, default=0.01)
    ap.add_argument('--power-iters', type=int, default=5)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out', type=str, default='s1_robust')
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(range(1, args.num_seeds + 1))
    checkpoints = sorted(set(int(x) for x in args.checkpoints.split(',')))

    print(f"=== S1 Robust Runner ===")
    print(f"Seeds: {seeds}")
    print(f"Tasks: {args.tasks}, Models: {args.models}")
    print(f"Steps: {args.steps}, Checkpoints: {checkpoints}")
    print()

    for task in args.tasks:
        for model_kind in args.models:
            print(f"[S1] Running task={task}, model={model_kind} "
                  f"across {len(seeds)} seeds...")
            all_seed_results = []
            for seed in seeds:
                print(f"  seed={seed}...", end=" ", flush=True)
                res = run_single_seed(
                    task=task, model_kind=model_kind, seed=seed,
                    T=args.T, Btrain=args.Btrain, Bprobe=args.Bprobe,
                    ni=args.ni, nh=args.nh, no=args.no,
                    alpha=args.alpha, mu=args.mu, steps=args.steps,
                    checkpoints=checkpoints, trim=args.trim,
                    power_iters=args.power_iters, device=args.device
                )
                all_seed_results.append(res)
                print("done")

            outdir = os.path.join(args.out, task, model_kind)
            aggregate_and_plot(
                all_seed_results, seeds, task, model_kind,
                checkpoints, args.T, outdir
            )
    print("\n=== S1 Robust Runner Complete ===")


if __name__ == "__main__":
    main()
