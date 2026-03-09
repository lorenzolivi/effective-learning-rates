# appendixA_validation.py
# Single-sweep validation of the first-order matrix-product expansion.
# - One epsilon grid (includes ε=1)
# - Truncation error vs ε and second-order remainder C2(ε) = ||F - T1|| / ε^2
# - Per-step ||A_j|| (dominant part), ||B_j|| (gate correction), ratio hist + timeseries
# - Small-ε slope fitted from the first N eps of the same grid
# - All outputs saved in one folder with per-gate filenames.

import argparse, csv, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Use float64 for stable small-ε behavior
torch.set_default_dtype(torch.float64)

# ------------------------ Utilities ------------------------

def parse_eps(s):
    return [float(x) for x in s.split(",") if x.strip()!=""]

def diag(v):
    return torch.diag(v)

def product(mats):
    P = torch.eye(mats[0].shape[0], device=mats[0].device)
    for M in mats[::-1]:
        P = M @ P
    return P

def first_derivative(A_list, B_list):
    """L_F(0,B) = sum_i (prod_{j<i} A_j) B_i (prod_{j>i} A_j).

    Since product() computes F = A_0 @ A_1 @ ... @ A_{n-1}, the
    Frechet derivative at eps=0 of prod(A_j + eps B_j) is:
        dF/deps = sum_i  L[i] @ B_i @ R[i+1]
    where L[i] = A_0 @ ... @ A_{i-1}  (left prefix, L[0] = I)
          R[i] = A_i @ ... @ A_{n-1}  (right suffix, R[n] = I).
    """
    n = len(A_list); H = A_list[0].shape[0]
    I = torch.eye(H, device=A_list[0].device)
    R = [None]*(n+1); L = [None]*(n+1)
    R[n] = I
    for i in range(n-1, -1, -1):
        R[i] = A_list[i] @ R[i+1]
    L[0] = I
    for i in range(1, n+1):
        L[i] = L[i-1] @ A_list[i-1]
    S = torch.zeros_like(I)
    for i in range(n):
        S = S + L[i] @ B_list[i] @ R[i+1]
    return S

def operator_norm(M):
    return torch.linalg.svdvals(M).max().item()

def fit_slope(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    X = torch.stack([x.log(), torch.ones_like(x)], dim=1)
    beta = torch.linalg.lstsq(X, y.log().unsqueeze(1)).solution.squeeze(1)
    return float(beta[0].item())

# ------------------------ Minimal gated RNN ------------------------

class BaseCell(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        self.Wi = nn.Linear(ni, nh, bias=False)
        self.Wr = nn.Linear(nh, nh, bias=False)
        nn.init.normal_(self.Wi.weight, std=0.2)
        nn.init.normal_(self.Wr.weight, std=0.2)
    def preact(self, x, h):
        return self.Wi(x) + self.Wr(h)

class GateScalar(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        self.gx = nn.Linear(nh, 1, bias=False)
        self.gu = nn.Linear(ni, 1, bias=False)
        nn.init.normal_(self.gx.weight, std=0.2)
        nn.init.normal_(self.gu.weight, std=0.2)
    def forward(self, x, h):
        return torch.sigmoid(self.gx(h) + self.gu(x))  # [B,1]

class GateMulti(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        self.gx = nn.Linear(nh, nh, bias=False)
        self.gu = nn.Linear(ni, nh, bias=False)
        nn.init.normal_(self.gx.weight, std=0.2)
        nn.init.normal_(self.gu.weight, std=0.2)
    def forward(self, x, h):
        return torch.sigmoid(self.gx(h) + self.gu(x))  # [B,H]

def make_signal(T, B, ni, seed=0, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    t = torch.linspace(0, 12, T, device=device)
    sigs = []
    base = torch.linspace(0.03, 0.09, ni, device=device)
    for k in range(ni):
        phase = torch.randn((), generator=gen, device=device)
        freq = base[k]
        sigs.append(torch.sin(freq * t + phase))
    u = torch.stack(sigs, dim=1).unsqueeze(1).repeat(1, B, 1)  # [T,B,ni]
    u = u + 0.05 * torch.randn(u.shape, generator=gen, device=device)
    return u

def build_factors(cell, gate, u, device="cpu", scalar_gate=False):
    """
    Constructs per-step Jacobian factors along one trajectory (batch=1):

      Dominant part (called A_j in the paper):
         A_j = diag(1 - g_j) + diag(g_j) @ (diag(phi'(a_j)) @ W_r)

      Gate-induced correction:
         B_j = diag(d_j) @ J^g_j,  with  d_j = phi(a_j) - h_j ,
         phi = tanh,  J^g_j = diag(g_j*(1-g_j)) @ W_{r,g}  (scalar variant uses mean slope).

    Returns lists [A_1,...,A_T], [B_1,...,B_T].
    """
    phi = torch.tanh
    T, B, ni = u.shape
    H = cell.Wr.out_features
    h = torch.zeros(B, H, device=device)
    A_list, B_list = [], []

    Wrg = gate.gx.weight  # [1,H] (scalar) or [H,H] (multi)

    for j in range(T):
        a = cell.preact(u[j], h)     # [B,H]
        x = phi(a)                   # [B,H]
        g = gate(u[j], h)            # [B,1] or [B,H]
        if scalar_gate and g.size(1) == 1:
            g = g.expand(B, H)
        h = g * x + (1.0 - g) * h

        a0, x0, g0, h0 = a[0], x[0], g[0], h[0]
        D = 1.0 - torch.tanh(a0)**2
        Acore = diag(D) @ cell.Wr.weight
        A_list.append(diag(1.0 - g0) + diag(g0) @ Acore)

        if scalar_gate:
            scalar = (g0 * (1.0 - g0)).mean()
            Dg = scalar * torch.ones(H, device=device)
            Jg = diag(Dg) @ Wrg.repeat(H, 1)
        else:
            Dg = g0 * (1.0 - g0)
            Jg = diag(Dg) @ Wrg

        d = x0 - h0
        B_list.append(diag(d) @ Jg)

    return A_list, B_list

# ------------------------ Plots ------------------------

def plot_error_vs_eps(prefix, eps_list, errs, outdir):
    plt.figure()
    plt.loglog(eps_list, errs, marker='o',
               label=r'$E_{\mathrm{trunc}}$')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$E_{\mathrm{trunc}}(\varepsilon)$')
    plt.title(r'Truncation error vs $\varepsilon$')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_error_vs_eps.png")
    plt.savefig(path, dpi=300); plt.close()

def plot_C2_vs_eps(prefix, eps_list, errs, outdir):
    C2 = np.array(errs) / (np.array(eps_list)**2)
    plt.figure()
    plt.semilogx(eps_list, C2, marker='o')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$C_2(\varepsilon)$')
    plt.title(r'Normalized remainder $C_2(\varepsilon)$')
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_C2_vs_eps.png")
    plt.savefig(path, dpi=300); plt.close()

def plot_ratio_hist(prefix, Rs, outdir):
    plt.figure()
    plt.hist(Rs, bins=30)
    plt.xlabel(r'$r_j$')
    plt.ylabel('count')
    plt.title(r'Distribution of $r_j$')
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_ratio_hist.png")
    plt.savefig(path, dpi=300); plt.close()

def plot_norms_timeseries(prefix, As, Bs, Rs, outdir):
    t = np.arange(len(As))
    fig, ax1 = plt.subplots()
    ax1.plot(t, As, marker='o', linestyle='-', label=r'$\|A_j\|_2$')
    ax1.plot(t, Bs, marker='o', linestyle='-', label=r'$\|B_j\|_2$')
    ax1.set_xlabel('time step $j$')
    ax1.set_ylabel(r'$\ell_2$ norm')
    ax2 = ax1.twinx()
    ax2.plot(t, Rs, marker='.', linestyle=':', label=r'$r_j$', alpha=0.7)
    ax2.set_ylabel(r'$r_j$')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines+lines2, labels+labels2, loc='center')
    fig.suptitle(r'Per-step $\ell_2$ norms')
    fig.tight_layout()
    path = os.path.join(outdir, f"{prefix}_norms_timeseries.png")
    fig.savefig(path, dpi=300); plt.close(fig)

# ------------------------ Core ------------------------

def norms_and_ratios(A_list, B_list, csv_path):
    As, Bs, Rs = [], [], []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","norm_A_dominant","norm_B_correction","ratio_B_over_A"])
        for t, (A, Bb) in enumerate(zip(A_list, B_list)):
            na = operator_norm(A)
            nb = operator_norm(Bb)
            r = (nb / na) if na > 0 else float("inf")
            w.writerow([t, na, nb, r])
            As.append(na); Bs.append(nb); Rs.append(r)
    return np.array(As), np.array(Bs), np.array(Rs)

def compute_errors(A_list, B_list, eps_list):
    F0 = product(A_list)
    L1 = first_derivative(A_list, B_list)
    errs = []
    for e in eps_list:
        M_eps = [A_list[i] + e*B_list[i] for i in range(len(A_list))]
        F_eps = product(M_eps)
        T1 = F0 + e*L1
        errs.append(torch.linalg.norm(F_eps - T1).item())
    return np.array(errs), F0, L1

def run_one(gate_type, T, H, ni, seed, device, eps_list, fit_first, outdir):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device)

    # Build frozen trajectory and factors
    cell = BaseCell(ni, H).to(device)
    gate = (GateMulti(ni, H) if gate_type == "multi" else GateScalar(ni, H)).to(device)
    scalar_gate = (gate_type == "scalar")
    u = make_signal(T=T, B=1, ni=ni, seed=seed, device=device)
    A_list, B_list = build_factors(cell, gate, u, device=device, scalar_gate=scalar_gate)

    prefix = f"{gate_type}"

    # One ε sweep (includes ε=1)
    errs, F0, L1 = compute_errors(A_list, B_list, eps_list)
    plot_error_vs_eps(prefix, eps_list, errs, outdir)
    plot_C2_vs_eps(prefix, eps_list, errs, outdir)

    # Small-ε slope estimated from first N points of this grid
    N = min(max(2, fit_first), len(eps_list))
    slope = fit_slope(np.array(eps_list[:N]), np.array(errs[:N]))

    # Per-step norms/ratios
    As, Bs, Rs = norms_and_ratios(A_list, B_list, os.path.join(outdir, f"{prefix}_perstep_norms.csv"))
    plot_ratio_hist(prefix, Rs, outdir)
    plot_norms_timeseries(prefix, As, Bs, Rs, outdir)

    # Second-order remainder at ε=1 (absolute and relative)
    if any(np.isclose(np.array(eps_list), 1.0)):
        eidx = int(np.where(np.isclose(np.array(eps_list), 1.0))[0][0])
        C2_eps1 = errs[eidx]  # divide by 1^2
        M_eps1 = [A_list[i] + B_list[i] for i in range(len(A_list))]
        F1 = product(M_eps1)
        T1_1 = F0 + L1
        relF = C2_eps1 / (torch.linalg.norm(F1).item() + 1e-12)
        relT1 = C2_eps1 / (torch.linalg.norm(T1_1).item() + 1e-12)
    else:
        C2_eps1 = float('nan'); relF = float('nan'); relT1 = float('nan')

    # Summary CSV
    with open(os.path.join(outdir, f"{prefix}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["small_eps_slope_from_first_N", f"{slope:.6f}"])
        w.writerow(["fit_first_N", N])
        w.writerow(["second_order_remainder_at_eps_1_norm", f"{C2_eps1:.6e}"])
        w.writerow(["second_order_remainder_at_eps_1_over_normF", f"{relF:.6e}"])
        w.writerow(["second_order_remainder_at_eps_1_over_normT1", f"{relT1:.6e}"])
        w.writerow(["ratio_median", f"{np.median(Rs):.6e}"])
        w.writerow(["ratio_p90", f"{np.percentile(Rs,90):.6e}"])
        w.writerow(["ratio_p99", f"{np.percentile(Rs,99):.6e}"])
        w.writerow(["ratio_max", f"{np.max(Rs):.6e}"])

    # Console summary
    print(f"\n[{gate_type.upper()}] Small-ε slope (first {N} eps): {slope:.2f} (expect ≈ 2)")
    print(f"[{gate_type.upper()}] Second-order remainder at ε=1: ||F - T1|| = {C2_eps1:.3e}")
    print(f"[{gate_type.upper()}] Relative: /||F|| = {relF:.3e}, /||T1|| = {relT1:.3e}")
    print(f"[{gate_type.upper()}] Ratios r_j = ||B||/||A|| -> median={np.median(Rs):.3e}, "
          f"p90={np.percentile(Rs,90):.3e}, p99={np.percentile(Rs,99):.3e}, max={np.max(Rs):.3e}")

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Single-sweep validation of first-order expansion (scalar & multi).")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--input-dim", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--eps", type=str,
        default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1"
    )
    ap.add_argument("--fit-first", type=int, default=5, help="use first N eps values to fit small-ε slope")
    ap.add_argument("--outdir", type=str, default="appendixA_figs_single_sweep")
    args = ap.parse_args()

    eps_list = parse_eps(args.eps)
    os.makedirs(args.outdir, exist_ok=True)

    for gate_type in ["scalar", "multi"]:
        run_one(
            gate_type=gate_type,
            T=args.T, H=args.hidden, ni=args.input_dim,
            seed=args.seed, device=args.device,
            eps_list=eps_list, fit_first=args.fit_first,
            outdir=args.outdir
        )

if __name__ == "__main__":
    main()