#!/usr/bin/env python3
"""
s2_robust.py — Multi-seed robust wrapper for the Joint Anisotropy experiment (S2).

Runs the s2_joint_anisotropy.py training + analysis logic across multiple random seeds,
aggregates results (mean ± std), and produces publication-quality plots with
error bars.

Usage:
  python s2_robust.py --num-seeds 5 --tasks adding ar2 movingavg delaysum narma10 --device cpu
  python s2_robust.py --seeds 1 2 3 4 5 --tasks adding
"""

import argparse, os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim_module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Import core functions from original s2_joint_anisotropy.py ---- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_joint_anisotropy import (
    set_seed, ensure_dir, to_np, d_tanh, median_iqr, safe_unit,
    make_batch,
    PlainRNN, ScalarGateRNN, MultiGateRNN,
    jacobian_blocks_plain, jacobian_blocks_scalar, jacobian_blocks_multigate,
    sample_tk_pairs, product_matrix, topk_svd,
    anisotropy_index, energy_concentration,
    train_with_opt,
    collect_grad_matrix, gradcov_metrics_from_G,
)


def run_single_seed(task, seed, T, B, Bprobe, M, ni, nh, no, iters,
                    checkpoints, lags, pairs_per_lag, svd_k, ai_r, ce_r,
                    lr_adam, lr_sgd, device_str):
    """
    Run one full training + analysis cycle for a single seed.
    Returns a dict with per-checkpoint results for all 3 models.
    """
    set_seed(seed)
    device = torch.device(device_str)

    valid_lags = [h for h in lags if 0 < h < T]
    ckpt_list = sorted(set(checkpoints))
    ckpt_set = set(ckpt_list)

    # Fixed probe batch
    u_probe, _ = make_batch(task, T, Bprobe, ni, device, seed=seed)

    # Train all 3 models
    plain = PlainRNN(ni, nh, no).to(device)
    scalar = ScalarGateRNN(ni, nh, no).to(device)
    multi = MultiGateRNN(ni, nh, no).to(device)

    ck_plain = train_with_opt(plain, optim_module.Adam(plain.parameters(), lr=lr_adam),
                              task, T, ni, device, iters, B, ckpt_set)
    ck_scalar = train_with_opt(scalar, optim_module.SGD(scalar.parameters(), lr=lr_sgd),
                               task, T, ni, device, iters, B, ckpt_set)
    ck_multi = train_with_opt(multi, optim_module.SGD(multi.parameters(), lr=lr_sgd),
                              task, T, ni, device, iters, B, ckpt_set)

    rng = np.random.RandomState(seed)
    results = {}

    for it in ckpt_list:
        plain.load_state_dict(ck_plain[it])
        scalar.load_state_dict(ck_scalar[it])
        multi.load_state_dict(ck_multi[it])

        # Jacobians
        with torch.no_grad():
            _, (Xp, Ap, _) = plain.forward_buffers(u_probe)
            Jp = jacobian_blocks_plain(plain, Xp, Ap)
            _, (Xs, As, _, Gs, AGs) = scalar.forward_buffers(u_probe)
            Js = jacobian_blocks_scalar(scalar, Xs, As, Gs, AGs)
            _, (Xm, Am, _, Gm, AGm) = multi.forward_buffers(u_probe)
            Jm = jacobian_blocks_multigate(multi, Xm, Am, Gm, AGm)

        pairs = sample_tk_pairs(T, Bprobe, valid_lags,
                                per_lag=pairs_per_lag, rng=rng)

        def collect_stats(Js_list):
            AI = {h: [] for h in valid_lags}
            CE = {h: [] for h in valid_lags}
            for h in valid_lags:
                for (b, t, k) in pairs[h]:
                    Mprod = product_matrix(Js_list, b, t, k)
                    if not torch.isfinite(Mprod).all():
                        continue
                    try:
                        _, S, _ = topk_svd(Mprod, k=svd_k)
                    except torch._C._LinAlgError:
                        continue
                    AI[h].append(anisotropy_index(S, r=ai_r))
                    CE[h].append(energy_concentration(S, r=ce_r))
            xs, AI_med, CE_med = [], [], []
            for h in valid_lags:
                if len(AI[h]) == 0:
                    continue
                m_ai, _, _ = median_iqr(AI[h])
                m_ce, _, _ = median_iqr(CE[h])
                if np.isfinite(m_ai) and np.isfinite(m_ce):
                    xs.append(h)
                    AI_med.append(m_ai)
                    CE_med.append(m_ce)
            return np.array(xs), np.array(AI_med), np.array(CE_med)

        h_p, AI_p, CE_p = collect_stats(Jp)
        h_s, AI_s, CE_s = collect_stats(Js)
        h_m, AI_m, CE_m = collect_stats(Jm)

        # Gradient covariance
        def compute_gc(model, name):
            G = collect_grad_matrix(
                model, task, T, Bprobe, ni, M, device,
                base_seed=seed + it + (0 if name == 'plain' else
                                       (1000 if name == 'scalar' else 2000))
            )
            AI_gc, CE_gc = gradcov_metrics_from_G(G, r=ce_r)
            return AI_gc, CE_gc

        AIg_p, CEg_p = compute_gc(plain, 'plain')
        AIg_s, CEg_s = compute_gc(scalar, 'scalar')
        AIg_m, CEg_m = compute_gc(multi, 'multi')

        results[it] = {
            "jacobian": {
                "plain": {"h": h_p.tolist(), "AI": AI_p.tolist(), "CE": CE_p.tolist()},
                "scalar": {"h": h_s.tolist(), "AI": AI_s.tolist(), "CE": CE_s.tolist()},
                "multi": {"h": h_m.tolist(), "AI": AI_m.tolist(), "CE": CE_m.tolist()},
            },
            "gradcov": {
                "plain": {"AI": float(AIg_p), "CE": float(CEg_p)},
                "scalar": {"AI": float(AIg_s), "CE": float(CEg_s)},
                "multi": {"AI": float(AIg_m), "CE": float(CEg_m)},
            }
        }

    return results


def aggregate_and_plot(all_seed_results, seeds, task, checkpoints, lags,
                       ai_r, ce_r, outdir):
    """
    Aggregate across seeds and produce robust plots.
    """
    os.makedirs(outdir, exist_ok=True)
    ckpts = sorted(checkpoints)
    n_seeds = len(seeds)
    model_names = ['plain', 'scalar', 'multi']
    model_labels = ['Plain+Adam', 'ScalarGate+SGD', 'MultiGate+SGD']
    markers = ['o', '^', 's']
    colors = ['C0', 'C1', 'C2']

    for it in ckpts:
        # Collect jacobian data across seeds per model
        for plot_type in ['AI', 'CE']:
            plt.figure(figsize=(7.4, 5.6))

            for mi, (mname, mlabel) in enumerate(zip(model_names, model_labels)):
                # Gather per-seed curves
                all_h = []
                all_vals = []
                for sr in all_seed_results:
                    if it in sr:
                        jd = sr[it]["jacobian"][mname]
                        all_h.append(np.array(jd["h"]))
                        all_vals.append(np.array(jd[plot_type]))

                if len(all_vals) == 0:
                    continue

                # Find common lags across all seeds
                common_h = all_h[0]
                for h_arr in all_h[1:]:
                    common_h = np.intersect1d(common_h, h_arr)

                if len(common_h) == 0:
                    continue

                # Extract values at common lags
                vals_matrix = []
                for h_arr, v_arr in zip(all_h, all_vals):
                    idxs = [np.where(h_arr == h)[0][0] for h in common_h]
                    vals_matrix.append(v_arr[idxs])
                vals_matrix = np.array(vals_matrix)  # (n_seeds, n_lags)

                median_v = np.median(vals_matrix, axis=0)
                q25_v = np.percentile(vals_matrix, 25, axis=0)
                q75_v = np.percentile(vals_matrix, 75, axis=0)

                plt.plot(common_h, median_v,
                         marker=markers[mi], lw=2, color=colors[mi],
                         label=f'{mlabel} (r={ai_r if plot_type == "AI" else ce_r})')
                plt.fill_between(common_h, q25_v, q75_v,
                                 alpha=0.18, color=colors[mi])

            if plot_type == 'AI':
                plt.yscale('log')
                plt.ylabel(r'Jacobian AI $\sigma_1/\sigma_{r}$')
                plt.title(f'Jacobian anisotropy vs lag (task={task}, iter={it})')
            else:
                plt.ylim(0.0, 1.02)
                plt.ylabel(fr'Jacobian energy concentration CE$_{{{ce_r}}}$')
                plt.title(f'Jacobian energy concentration vs lag (task={task}, iter={it})')

            plt.xlabel('Lag $h$')
            plt.legend()
            plt.tight_layout()
            fname = f"s2_Jacobian_{plot_type}_iter{it}_{task}.png"
            plt.savefig(os.path.join(outdir, fname), dpi=300)
            plt.close()

        # --- GradCov bar chart with error bars --- #
        gc_data = {mname: {"AI": [], "CE": []} for mname in model_names}
        for sr in all_seed_results:
            if it in sr:
                for mname in model_names:
                    gc_data[mname]["AI"].append(sr[it]["gradcov"][mname]["AI"])
                    gc_data[mname]["CE"].append(sr[it]["gradcov"][mname]["CE"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x_pos = np.arange(len(model_names))
        width = 0.35

        for ax, metric in [(ax1, "AI"), (ax2, "CE")]:
            medians = [float(np.nanmedian(gc_data[m][metric])) for m in model_names]
            q25 = [float(np.nanpercentile(gc_data[m][metric], 25)) for m in model_names]
            q75 = [float(np.nanpercentile(gc_data[m][metric], 75)) for m in model_names]
            yerr_lo = [med - lo for med, lo in zip(medians, q25)]
            yerr_hi = [hi - med for med, hi in zip(medians, q75)]
            bars = ax.bar(x_pos, medians, width, yerr=[yerr_lo, yerr_hi],
                          capsize=5, color=colors, alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_labels, rotation=15, fontsize=9)
            ax.set_ylabel(f'GradCov {metric} (r={ce_r})')
            ax.set_title(f'Gradient-covariance {metric} (task={task}, iter={it})')

        fig.suptitle(f'Gradient covariance (task={task}, iter={it})', fontsize=10)
        fig.tight_layout()
        plt.savefig(os.path.join(outdir, f"s2_gradcov_iter{it}_{task}.png"), dpi=300)
        plt.close()

    # --- Save JSON summary --- #
    summary = {
        "task": task,
        "seeds": seeds,
        "n_seeds": n_seeds,
        "checkpoints": ckpts,
        "aggregated": {},
        "per_seed": {}
    }

    for it in ckpts:
        agg_it = {}
        for mname in model_names:
            jac_ai_all, jac_ce_all = [], []
            gc_ai_all, gc_ce_all = [], []
            for sr in all_seed_results:
                if it in sr:
                    jac_ai_all.append(sr[it]["jacobian"][mname]["AI"])
                    jac_ce_all.append(sr[it]["jacobian"][mname]["CE"])
                    gc_ai_all.append(sr[it]["gradcov"][mname]["AI"])
                    gc_ce_all.append(sr[it]["gradcov"][mname]["CE"])
            # Jacobian: each seed contributes a list of values (one per lag).
            # Flatten across seeds, then report median and IQR.
            jac_ai_flat = [v for lst in jac_ai_all for v in lst if np.isfinite(v)]
            jac_ce_flat = [v for lst in jac_ce_all for v in lst if np.isfinite(v)]
            agg_it[mname] = {
                "jacobian_AI_median": float(np.median(jac_ai_flat)) if jac_ai_flat else float('nan'),
                "jacobian_AI_q25": float(np.percentile(jac_ai_flat, 25)) if jac_ai_flat else float('nan'),
                "jacobian_AI_q75": float(np.percentile(jac_ai_flat, 75)) if jac_ai_flat else float('nan'),
                "jacobian_CE_median": float(np.median(jac_ce_flat)) if jac_ce_flat else float('nan'),
                "jacobian_CE_q25": float(np.percentile(jac_ce_flat, 25)) if jac_ce_flat else float('nan'),
                "jacobian_CE_q75": float(np.percentile(jac_ce_flat, 75)) if jac_ce_flat else float('nan'),
                "gradcov_AI_median": float(np.nanmedian(gc_ai_all)) if gc_ai_all else float('nan'),
                "gradcov_AI_q25": float(np.nanpercentile(gc_ai_all, 25)) if gc_ai_all else float('nan'),
                "gradcov_AI_q75": float(np.nanpercentile(gc_ai_all, 75)) if gc_ai_all else float('nan'),
                "gradcov_CE_median": float(np.nanmedian(gc_ce_all)) if gc_ce_all else float('nan'),
                "gradcov_CE_q25": float(np.nanpercentile(gc_ce_all, 25)) if gc_ce_all else float('nan'),
                "gradcov_CE_q75": float(np.nanpercentile(gc_ce_all, 75)) if gc_ce_all else float('nan'),
            }
        summary["aggregated"][int(it)] = agg_it

    for i, seed in enumerate(seeds):
        summary["per_seed"][int(seed)] = all_seed_results[i]

    with open(os.path.join(outdir, f"s2_robust_summary_{task}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  [DONE] task={task}: {n_seeds} seeds, outputs in {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed robust runner for S2 (Joint Anisotropy)")
    ap.add_argument('--num-seeds', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--tasks', type=str, nargs='+',
                    default=['adding', 'ar2', 'movingavg', 'delaysum', 'narma10'],
                    choices=['adding', 'narma10', 'narma20', 'movingavg', 'delaysum', 'ar2'])
    ap.add_argument('--device', default='cpu')

    ap.add_argument('--T', type=int, default=120)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--Bprobe', type=int, default=64)
    ap.add_argument('--M', type=int, default=256)

    ap.add_argument('--ni', type=int, default=6)
    ap.add_argument('--nh', type=int, default=64)
    ap.add_argument('--no', type=int, default=1)

    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--checkpoints', type=int, nargs='+',
                    default=[0, 400, 800, 1200])

    ap.add_argument('--lags', type=int, nargs='+',
                    default=[1, 2, 4, 8, 12, 16, 24, 32, 40])
    ap.add_argument('--pairs-per-lag', type=int, default=64)
    ap.add_argument('--svd-k', type=int, default=16)
    ap.add_argument('--ai-r', type=int, default=10)
    ap.add_argument('--ce-r', type=int, default=10)

    ap.add_argument('--lr_adam', type=float, default=1e-3)
    ap.add_argument('--lr_sgd', type=float, default=1e-2)
    ap.add_argument('--out', type=str, default='s2_robust')
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(range(1, args.num_seeds + 1))
    ckpts = sorted(set(args.checkpoints))

    print(f"=== S2 Robust Runner ===")
    print(f"Seeds: {seeds}")
    print(f"Tasks: {args.tasks}")
    print(f"Iters: {args.iters}, Checkpoints: {ckpts}")
    print()

    for task in args.tasks:
        print(f"[S2] Running task={task} across {len(seeds)} seeds...")
        all_seed_results = []
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            res = run_single_seed(
                task=task, seed=seed,
                T=args.T, B=args.B, Bprobe=args.Bprobe, M=args.M,
                ni=args.ni, nh=args.nh, no=args.no,
                iters=args.iters, checkpoints=ckpts,
                lags=args.lags, pairs_per_lag=args.pairs_per_lag,
                svd_k=args.svd_k, ai_r=args.ai_r, ce_r=args.ce_r,
                lr_adam=args.lr_adam, lr_sgd=args.lr_sgd,
                device_str=args.device
            )
            all_seed_results.append(res)
            print("done")

        outdir = os.path.join(args.out, task)
        aggregate_and_plot(
            all_seed_results, seeds, task, ckpts,
            args.lags, args.ai_r, args.ce_r, outdir
        )

    print("\n=== S2 Robust Runner Complete ===")


if __name__ == "__main__":
    main()
