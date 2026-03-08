#!/usr/bin/env python3
"""
run_all_robust.py — Master runner for all robust experiments.

Single entry-point that sequentially runs s1_robust, s2_robust, and
appendixA_robust with consistent seed lists and paper-matching defaults.

Usage:
  # Full paper replication with 5 seeds (default)
  python run_all_robust.py --device cpu

  # Quick verification with 2 seeds and reduced iterations
  python run_all_robust.py --num-seeds 2 --s1-steps 100 --s2-iters 200 --device cpu

  # Run only specific experiments
  python run_all_robust.py --skip-s1 --skip-appendix --device cpu
"""

import argparse, os, sys, time, subprocess


def run_cmd(cmd, label):
    """Run a command and print timing info."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.1f}s\n")
    return result.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Master runner for all robust multi-seed experiments")
    ap.add_argument('--num-seeds', type=int, default=5,
                    help='Number of seeds (generates 1..N for s1/s2, 0..N-1 for appendix)')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Explicit seed list (overrides --num-seeds)')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out-root', type=str, default='.',
                    help='Root directory for all outputs')

    # S1 overrides
    ap.add_argument('--s1-steps', type=int, default=800)
    ap.add_argument('--s1-tasks', type=str, nargs='+', default=['adding'])
    ap.add_argument('--s1-models', type=str, nargs='+',
                    default=['leaky', 'scalar', 'multigate'])
    ap.add_argument('--s1-checkpoints', type=str, default='0,50,200,400,600,800')

    # S2 overrides
    ap.add_argument('--s2-iters', type=int, default=1200)
    ap.add_argument('--s2-tasks', type=str, nargs='+',
                    default=['adding', 'ar2', 'movingavg', 'delaysum', 'narma10'])
    ap.add_argument('--s2-checkpoints', type=int, nargs='+',
                    default=[0, 400, 800, 1200])

    # Skip flags
    ap.add_argument('--skip-s1', action='store_true')
    ap.add_argument('--skip-s2', action='store_true')
    ap.add_argument('--skip-appendix', action='store_true')

    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds = args.seeds if args.seeds else list(range(1, args.num_seeds + 1))
    seeds_str = ' '.join(str(s) for s in seeds)

    # Seeds for appendix (traditionally start at 0)
    seeds_app = args.seeds if args.seeds else list(range(0, args.num_seeds))
    seeds_app_str = ' '.join(str(s) for s in seeds_app)

    print(f"╔{'═'*58}╗")
    print(f"║  Robust Multi-Seed Experiment Runner                     ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Seeds:   {seeds_str:<47}║")
    print(f"║  Device:  {args.device:<47}║")
    print(f"║  S1:      {'SKIP' if args.skip_s1 else 'RUN':<47}║")
    print(f"║  S2:      {'SKIP' if args.skip_s2 else 'RUN':<47}║")
    print(f"║  App A:   {'SKIP' if args.skip_appendix else 'RUN':<47}║")
    print(f"╚{'═'*58}╝")

    t_total = time.time()
    results = {}

    # ---- S1: Effective LR ---- #
    if not args.skip_s1:
        cmd = [
            sys.executable, os.path.join(script_dir, 's1_robust.py'),
            '--seeds'] + [str(s) for s in seeds] + [
            '--tasks'] + args.s1_tasks + [
            '--models'] + args.s1_models + [
            '--steps', str(args.s1_steps),
            '--checkpoints', args.s1_checkpoints,
            '--device', args.device,
            '--out', os.path.join(args.out_root, 's1_robust'),
        ]
        results['s1'] = run_cmd(cmd, "S1: Effective LR profiles (multi-seed)")

    # ---- S2: Joint Anisotropy ---- #
    if not args.skip_s2:
        cmd = [
            sys.executable, os.path.join(script_dir, 's2_robust.py'),
            '--seeds'] + [str(s) for s in seeds] + [
            '--tasks'] + args.s2_tasks + [
            '--iters', str(args.s2_iters),
            '--checkpoints'] + [str(c) for c in args.s2_checkpoints] + [
            '--device', args.device,
            '--out', os.path.join(args.out_root, 's2_robust'),
        ]
        results['s2'] = run_cmd(cmd, "S2: Joint Anisotropy (multi-seed)")

    # ---- Appendix A ---- #
    if not args.skip_appendix:
        cmd = [
            sys.executable, os.path.join(script_dir, 'appendixA_robust.py'),
            '--seeds'] + [str(s) for s in seeds_app] + [
            '--device', args.device,
            '--out', os.path.join(args.out_root, 'appendixA_robust'),
        ]
        results['appendix'] = run_cmd(cmd, "Appendix A: First-order expansion validation (multi-seed)")

    elapsed_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {elapsed_total:.1f}s")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "OK" if v == 0 else "FAILED"
        print(f"  {k}: {status}")
    print()

    # Clean up __pycache__
    import shutil
    pycache = os.path.join(script_dir, '__pycache__')
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)


if __name__ == "__main__":
    main()
