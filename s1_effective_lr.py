#!/usr/bin/env python3
"""
S1: Effective LR profiles during training (with theory overlays) + multiple tasks

Goal
----
With a fixed global SGD step size (no momentum/Adam), train a small RNN (leaky / scalar gate / multi-gate)
on a chosen task, and at selected checkpoints measure a lag-conditioned "effective LR profile"
via the sensitivity S_{t,k} = || ∏_{j=k+1}^t J_j ||_2, aggregated by lag h = t-k.

We plot (per checkpoint) the normalized profile:
  \tilde{mu}_eff(h; ℓ) = median_{t-k=h} S_{t,k}(θ_ℓ) / median_{t-k=1} S_{t,k}(θ_ℓ)

Overlays:
  - Zeroth-order:   \tilde{mu}_pred^(0)(h; ℓ) = \bar P(h; ℓ)/\bar P(1; ℓ),
      where P_{t,k} = α^{t-k} (leaky) or ∏ g_{j-1} (scalar) or mean_i ∏ g^{(i)}_{j-1} (multi).
  - Fitted-power:   \tilde{mu}_pred^(fit)(h; ℓ) = [ \bar P(h; ℓ)/\bar P(1; ℓ) ]^{ s(ℓ) },
      where s(ℓ) fits log S ≈ a + s log P on central predictor quantiles.

Tasks (select via --task):
  - adding  : classic two-marker adding task (long-range credit).
  - recall  : copy/recall-like: K marked values early, target is their sum at the end.
  - sinmix  : noisy sinusoid mixing; target combines early-vs-late averages (long-range).
  - short   : short-horizon regression; target depends only on last L steps (myopic).

Outputs (saved under --out, no timestamps):
  - Per-checkpoint profile plots with overlays
  - Heatmap of empirical normalized profiles vs lag and checkpoint
  - Slope-vs-iteration plot (s(ℓ))
  - JSON summaries per checkpoint (slope, R^2, trims, etc.)
"""

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ------------------------------- Utilities -------------------------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def d_tanh(a: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(a) ** 2


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# ------------------------------- Data tasks -------------------------------- #

@torch.no_grad()
def make_batch_adding(T, B, ni_base, device, gen: torch.Generator):
    """
    Adding: input has (ni_base + 1) channels: ni_base noise + 1 mask.
    Target is the sum of the two masked values from first noise channel.
    """
    x = torch.rand((B, T, ni_base), generator=gen, device=device)
    m = torch.zeros((B, T, 1), device=device)
    idx1 = torch.randint(0, T // 2, (B,), generator=gen, device=device)
    idx2 = torch.randint(T // 2, T, (B,), generator=gen, device=device)
    m[torch.arange(B), idx1, 0] = 1.0
    m[torch.arange(B), idx2, 0] = 1.0
    u = torch.cat([x, m], dim=-1)  # (B,T,ni_base+1)
    y = (x[..., 0:1] * m).sum(dim=1)  # (B,1)
    return u, y


@torch.no_grad()
def make_batch_recall(T, B, ni_base, device, gen: torch.Generator, K=5):
    """
    Copy/Recall-like: choose K positions in first third, mark them (mask channel),
    fill ni_base noise channels; target is the sum of the K values from the first noise channel.
    Forces longer memory than 'adding' by packing events early.
    """
    x = torch.rand((B, T, ni_base), generator=gen, device=device)
    m = torch.zeros((B, T, 1), device=device)
    third = max(3, T // 3)
    # choose K distinct indices in [0, third)
    idxs = torch.stack([
        torch.randperm(third, generator=gen, device=device)[:K] for _ in range(B)
    ], dim=0)  # (B,K)
    m[torch.arange(B).unsqueeze(1), idxs, 0] = 1.0
    u = torch.cat([x, m], dim=-1)
    y = (x[..., 0:1] * m).sum(dim=1)  # sum of K marked values
    return u, y


@torch.no_grad()
def make_batch_sinmix(T, B, ni_base, device, gen: torch.Generator):
    """
    Noisy sinusoid mixing: two sinusoids with random freq/phase in channels 0 and 1,
    plus (ni_base-2) noise channels; target contrasts early-vs-late averages (long memory).
    """
    t = torch.linspace(0, 1, T, device=device).view(1, T, 1).expand(B, -1, -1)
    f1 = torch.rand((B, 1, 1), generator=gen, device=device) * 6 + 1   # [1,7] Hz
    f2 = torch.rand((B, 1, 1), generator=gen, device=device) * 6 + 1
    p1 = torch.rand((B, 1, 1), generator=gen, device=device) * 2 * np.pi
    p2 = torch.rand((B, 1, 1), generator=gen, device=device) * 2 * np.pi
    s1 = torch.sin(2 * np.pi * f1 * t + p1)
    s2 = torch.sin(2 * np.pi * f2 * t + p2)
    noise = torch.randn((B, T, max(0, ni_base - 2)), generator=gen, device=device) * 0.1
    x = torch.cat([s1, s2, noise], dim=-1) if ni_base >= 2 else s1
    # mask channel always 0 here (not needed, but keep ni_total consistent)
    m = torch.zeros((B, T, 1), device=device)
    u = torch.cat([x, m], dim=-1)
    # target: (late avg of s1) - (early avg of s1) + (early avg of s2) - (late avg of s2)
    early = slice(0, T // 4)
    late = slice(3 * T // 4, T)
    s1e = s1[:, early, 0].mean(dim=1, keepdim=True)
    s1l = s1[:, late, 0].mean(dim=1, keepdim=True)
    s2e = s2[:, early, 0].mean(dim=1, keepdim=True)
    s2l = s2[:, late, 0].mean(dim=1, keepdim=True)
    y = (s1l - s1e) + (s2e - s2l)  # (B,1)
    return u, y


@torch.no_grad()
def make_batch_short(T, B, ni_base, device, gen: torch.Generator, L=6):
    """
    Short-horizon regression: target depends only on last L steps of channel 0.
    Encourages myopic credit assignment (short memory).
    """
    x = torch.rand((B, T, ni_base), generator=gen, device=device)
    m = torch.zeros((B, T, 1), device=device)  # not used, just to keep shape consistent
    u = torch.cat([x, m], dim=-1)
    last = slice(max(0, T - L), T)
    y = x[:, last, 0:1].sum(dim=1)  # (B,1)
    return u, y


def make_batch(task: str, T, B, ni_total, device, gen: torch.Generator):
    ni_base = ni_total - 1
    if task == "adding":
        return make_batch_adding(T, B, ni_base, device, gen)
    if task == "recall":
        return make_batch_recall(T, B, ni_base, device, gen, K=5)
    if task == "sinmix":
        return make_batch_sinmix(T, B, ni_base, device, gen)
    if task == "short":
        return make_batch_short(T, B, ni_base, device, gen, L=6)
    raise ValueError(task)


# -------------------------------- Models ---------------------------------- #

class LeakyRNN(nn.Module):
    def __init__(self, ni, nh, no, alpha: float):
        super().__init__()
        self.Wr = nn.Linear(nh, nh, bias=False)
        self.Wi = nn.Linear(ni, nh, bias=False)
        self.Wo = nn.Linear(nh, no, bias=False)
        self.alpha = float(alpha)

    def step(self, x, u):
        a = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        x_next = self.alpha * phi + (1 - self.alpha) * x
        return x_next, a, phi, None, None

    def forward_full(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device)
        X, As, Phis = [], [], []
        for t in range(T):
            x, a, phi, _, _ = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi)
        X = torch.stack(X, 1)
        y_hat = self.Wo(X[:, -1, :])
        return X, torch.stack(As, 1), torch.stack(Phis, 1), None, None, y_hat


class ScalarGateRNN(LeakyRNN):
    def __init__(self, ni, nh, no):
        super().__init__(ni, nh, no, alpha=1.0)
        self.Wrg = nn.Linear(nh, 1, bias=False)
        self.Wig = nn.Linear(ni, 1, bias=False)

    def step(self, x, u):
        a = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        ag = self.Wrg(x) + self.Wig(u)     # (B,1)
        g = torch.sigmoid(ag)              # (B,1)
        x_next = g * phi + (1 - g) * x
        return x_next, a, phi, g, ag

    def forward_full(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device)
        X, As, Phis, Gs, AGs = [], [], [], [], []
        for t in range(T):
            x, a, phi, g, ag = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi); Gs.append(g); AGs.append(ag)
        X = torch.stack(X, 1)
        y_hat = self.Wo(X[:, -1, :])
        return X, torch.stack(As, 1), torch.stack(Phis, 1), torch.stack(Gs, 1), torch.stack(AGs, 1), y_hat


class MultiGateRNN(LeakyRNN):
    def __init__(self, ni, nh, no):
        super().__init__(ni, nh, no, alpha=1.0)
        self.Wrg = nn.Linear(nh, nh, bias=False)
        self.Wig = nn.Linear(ni, nh, bias=False)

    def step(self, x, u):
        a = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        ag = self.Wrg(x) + self.Wig(u)     # (B,nh)
        g = torch.sigmoid(ag)              # (B,nh)
        x_next = g * phi + (1 - g) * x
        return x_next, a, phi, g, ag

    def forward_full(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device)
        X, As, Phis, Gs, AGs = [], [], [], [], []
        for t in range(T):
            x, a, phi, g, ag = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi); Gs.append(g); AGs.append(ag)
        X = torch.stack(X, 1)
        y_hat = self.Wo(X[:, -1, :])
        return X, torch.stack(As, 1), torch.stack(Phis, 1), torch.stack(Gs, 1), torch.stack(AGs, 1), y_hat


def build_model(kind, ni, nh, no, alpha):
    if kind == 'leaky':
        return LeakyRNN(ni, nh, no, alpha)
    elif kind == 'scalar':
        return ScalarGateRNN(ni, nh, no)
    elif kind == 'multigate':
        return MultiGateRNN(ni, nh, no)
    else:
        raise ValueError(kind)


# ----------------------------- Jacobian blocks ---------------------------- #

def compute_J_blocks(model, X, As, G=None, AG=None):
    """
    Build per-time Jacobian blocks J_j = ∂x_j / ∂x_{j-1}, list(T) of (B,nh,nh).
    """
    B, T, nh = X.shape
    I = torch.eye(nh, device=X.device).unsqueeze(0).expand(B, nh, nh)
    Wr = model.Wr.weight  # (nh,nh)
    Js = []

    for j in range(1, T + 1):
        xjm1 = X[:, j - 1]
        a = As[:, j - 1]
        D = torch.diag_embed(d_tanh(a))
        DWr = torch.matmul(D, Wr.unsqueeze(0).expand(B, -1, -1))

        if isinstance(model, LeakyRNN) and not isinstance(model, (ScalarGateRNN, MultiGateRNN)):
            J = model.alpha * DWr + (1 - model.alpha) * I

        elif isinstance(model, ScalarGateRNN):
            assert G is not None and AG is not None
            g = G[:, j - 1, :]             # (B,1)
            ag = AG[:, j - 1, :]           # (B,1)
            sigp = torch.sigmoid(ag) * (1 - torch.sigmoid(ag))  # (B,1)
            term1 = g.view(B, 1, 1) * DWr
            term2 = (1 - g).view(B, 1, 1) * I
            phi = torch.tanh(a)
            d = (phi - xjm1).unsqueeze(2)  # (B,nh,1)
            row = sigp.view(B, 1) * model.Wrg.weight  # (B,nh)
            term3 = torch.matmul(d, row.unsqueeze(1))  # (B,nh,nh)
            J = term1 + term2 + term3

        else:  # MultiGateRNN
            assert G is not None and AG is not None
            g = G[:, j - 1, :]             # (B,nh)
            ag = AG[:, j - 1, :]           # (B,nh)
            sigp = torch.sigmoid(ag) * (1 - torch.sigmoid(ag))  # (B,nh)
            term1 = torch.matmul(torch.diag_embed(g), DWr)      # diag(g) * D*Wr
            term2 = torch.diag_embed(1 - g)                     # diag(1-g)
            phi = torch.tanh(a)
            Jg = torch.matmul(torch.diag_embed(sigp),
                              model.Wrg.weight.unsqueeze(0).expand(B, -1, -1))
            term3 = torch.matmul(torch.diag_embed(phi - xjm1), Jg)
            J = term1 + term2 + term3

        Js.append(J)

    return Js


# ----------------------- Sensitivity (power iteration) -------------------- #

def product_sensitivities(Js, iters=5):
    """
    Approximate S_{t,k} = ||∏_{j=k+1}^t J_j||_2 via a few power iterations.
    Returns S of shape (B,T,T) with zeros for k>t.
    """
    B, nh, _ = Js[0].shape
    T = len(Js)
    S = torch.zeros(B, T, T, device=Js[0].device)
    for b in range(B):
        for t in range(T):
            P = torch.eye(nh, device=Js[0].device)
            for j in range(t, 0, -1):  # J_t ... J_1
                P = torch.matmul(Js[j][b], P)
                # Power iteration for norm(P)
                v = torch.randn(nh, device=P.device)
                v = v / (v.norm() + 1e-12)
                for _ in range(iters):
                    v = torch.matmul(P, v)
                    v = v / (v.norm() + 1e-12)
                Sv = torch.matmul(P, v)
                S[b, t, j - 1] = Sv.norm()
    return S


# -------------------------- Gate-product predictor ------------------------ #

def gate_product_predictor(model, G, T, B, device, alpha=None):
    """
    Returns GP of shape (B,T,T):
      - leaky  : GP_{t,k} = α^{t-k}
      - scalar : ∏ scalar g over (k+1..t)
      - multi  : mean over neurons of ∏ g^{(i)} over (k+1..t)
    """
    if isinstance(model, LeakyRNN) and not isinstance(model, (ScalarGateRNN, MultiGateRNN)):
        assert alpha is not None
        GP = torch.zeros(B, T, T, device=device)
        for t in range(T):
            for k in range(t + 1):
                GP[:, t, k] = alpha ** (t - k)
        return GP

    if isinstance(model, ScalarGateRNN):
        assert G is not None
        g = G.squeeze(-1)  # (B,T)
        GP = torch.zeros(B, T, T, device=g.device)
        for t in range(T):
            prod = torch.ones(B, device=g.device)
            for j in range(t, -1, -1):
                prod = prod * g[:, j]
                GP[:, t, j] = prod
        return GP

    # multi-gate
    assert G is not None
    B, T, nh = G.shape
    GP = torch.zeros(B, T, T, device=G.device)
    for t in range(T):
        prod = torch.ones(B, nh, device=G.device)
        for j in range(t, -1, -1):
            prod = prod * G[:, j, :]
            GP[:, t, j] = prod.mean(dim=-1)  # mean across neurons
    return GP


# ------------------------------ Profiling --------------------------------- #

def per_lag_median_profile(S, GP):
    """
    S, GP: tensors (B,T,T). Returns lag-wise medians over (t,k) s.t. t-k=h.
    h ranges 1..T-1. Returns arrays of shape (T-1,) for S_med and GP_med.
    """
    B, T, _ = S.shape
    S_med = []
    GP_med = []
    for h in range(1, T):
        Ss = []
        GPs = []
        for t in range(h, T):
            k = t - h
            Ss.append(S[:, t, k])
            GPs.append(GP[:, t, k])
        S_med.append(torch.median(torch.stack(Ss, dim=-1).reshape(-1)))
        GP_med.append(torch.median(torch.stack(GPs, dim=-1).reshape(-1)))
    return torch.stack(S_med), torch.stack(GP_med)  # (T-1,), (T-1,)


def fit_loglog_slope(S, GP, trim=0.01):
    """
    Fit log S = a + s log GP with central trimming on GP.
    Returns slope s, R^2, low/high trims used.
    """
    Sv = to_numpy(S.reshape(-1))
    GPv = to_numpy(GP.reshape(-1))
    valid = np.isfinite(Sv) & np.isfinite(GPv) & (Sv > 0) & (GPv > 0)
    if valid.sum() < 50:
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = np.quantile(GPv[valid], [trim, 1 - trim])
    central = valid & (GPv >= lo) & (GPv <= hi)
    x = np.log(GPv[central])[:, None]
    y = np.log(Sv[central])[:, None]
    Xmat = np.hstack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    slope = float(beta[1, 0])
    intercept = float(beta[0, 0])
    yhat = Xmat @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, r2, float(lo), float(hi)


# --------------------------------- Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['adding', 'recall', 'sinmix', 'short'],
                    help='sequence task generator')
    ap.add_argument('--model', required=True, choices=['leaky', 'scalar', 'multigate'])
    ap.add_argument('--alpha', type=float, default=0.8, help='used only for leaky')
    ap.add_argument('--T', type=int, default=80)
    ap.add_argument('--Btrain', type=int, default=64)
    ap.add_argument('--Bprobe', type=int, default=12)
    ap.add_argument('--ni', type=int, default=4, help='ni-1 content channels + 1 mask channel')
    ap.add_argument('--nh', type=int, default=64)
    ap.add_argument('--no', type=int, default=1)
    ap.add_argument('--mu', type=float, default=1e-2)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--checkpoints', type=str, default='0,50,200,400,600')
    ap.add_argument('--trim', type=float, default=0.01, help='quantile trim on predictor for slope fit')
    ap.add_argument('--power-iters', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    gen_train = torch.Generator(device=device).manual_seed(args.seed + 123)
    gen_probe = torch.Generator(device=device).manual_seed(args.seed + 999)

    # Build model
    model = build_model(args.model, ni=args.ni, nh=args.nh, no=args.no, alpha=args.alpha).to(device)

    # Optimizer: plain SGD (no momentum, no wd)
    optim = torch.optim.SGD(model.parameters(), lr=args.mu)

    # Parse checkpoints (sorted unique)
    ckpts = sorted(set(int(x) for x in args.checkpoints.split(',')))
    ckpt_set = set(ckpts)

    # Fixed probe batch for consistency over time
    U_probe, Y_probe = make_batch(args.task, args.T, args.Bprobe, args.ni, device, gen_probe)

    # Storage for heatmap rows and slopes
    profile_rows = []
    profile_lags = None
    slope_records = []

    def run_probe_and_plots(ckpt_iter: int):
        nonlocal profile_lags

        # Forward on probe batch
        X, As, Phis, G, AG, yhat = model.forward_full(U_probe)

        # Js, Sensitivity, Predictor
        Js = compute_J_blocks(model, X, As, G, AG)
        S = product_sensitivities(Js, iters=args.power_iters)  # (B,T,T)
        GP = gate_product_predictor(model, G if args.model != 'leaky' else None,
                                    args.T, args.Bprobe, device, alpha=args.alpha)

        # Fit slope on all (t,k) pairs (with trimming)
        s, r2, lo, hi = fit_loglog_slope(S, GP, trim=args.trim)

        # Per-lag medians and normalized profiles
        S_med, GP_med = per_lag_median_profile(S, GP)
        eps = 1e-12
        S_norm = S_med / (S_med[0] + eps)
        P0_norm = GP_med / (GP_med[0] + eps)
        Pf_norm = torch.pow(P0_norm + eps, s if np.isfinite(s) else 1.0)

        # Save per-ckpt plot
        lags = np.arange(1, args.T)
        if profile_lags is None:
            profile_lags = lags

        plt.figure()
        plt.plot(lags, to_numpy(S_norm), lw=2, label=r'Empirical $\tilde{\mu}_{\mathrm{eff}}(h;\ell)$')
        plt.plot(lags, to_numpy(P0_norm), lw=2, linestyle='--',
                 label=r'Zeroth-order $\tilde{\mu}^{(0)}_{\mathrm{pred}}(h;\ell)$')
        plt.plot(lags, to_numpy(Pf_norm), lw=2, linestyle='-.',
                 label=r'Fitted-power $\tilde{\mu}^{(\mathrm{fit})}(h;\ell)$'
                       + f', $s={s:.2f}$, $R^2={r2:.2f}$')
        plt.xlabel('Lag $h$')
        plt.ylabel('Normalized effective LR (lag-wise)')
        plt.title(f'Effective LR profile (task={args.task}, model={args.model}, iter={ckpt_iter})')
        plt.legend(loc='best')
        plt.tight_layout()
        fpath = os.path.join(args.out, f"s1_profile_{args.task}_{args.model}_iter{ckpt_iter}.png")
        plt.savefig(fpath, dpi=300); plt.close()

        # Append to heatmap rows (numpy)
        profile_rows.append(to_numpy(S_norm))

        # Save JSON summary
        summary = {
            "task": args.task,
            "model": args.model,
            "iter": ckpt_iter,
            "slope": None if not np.isfinite(s) else float(s),
            "R2": None if not np.isfinite(r2) else float(r2),
            "predictor_trim_low": None if not np.isfinite(lo) else float(lo),
            "predictor_trim_high": None if not np.isfinite(hi) else float(hi),
            "T": args.T,
            "Bprobe": args.Bprobe,
            "alpha": args.alpha if args.model == 'leaky' else None
        }
        with open(os.path.join(args.out, f"s1_summary_{args.task}_{args.model}_iter{ckpt_iter}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        slope_records.append({"iter": ckpt_iter, "slope": float(s) if np.isfinite(s) else np.nan,
                              "R2": float(r2) if np.isfinite(r2) else np.nan})

        print(f"[PROBE] task={args.task} model={args.model} iter={ckpt_iter}  slope={s:.3f}  R2={r2:.3f}  |  plot: {fpath}")

    # ------------------------- Training loop (SGD) ------------------------- #
    for it in range(args.steps + 1):
        # Probe BEFORE update at scheduled checkpoints
        if it in ckpt_set:
            run_probe_and_plots(it)

        if it == args.steps:
            break

        # Randomized mini-batch (stochastic)
        U_tr, Y_tr = make_batch(args.task, args.T, args.Btrain, args.ni, device, gen_train)
        X, As, Phis, G, AG, y_hat = model.forward_full(U_tr)
        loss = F.mse_loss(y_hat, Y_tr)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # ------------------------- Aggregate heatmap & slope ------------------- #
    if len(profile_rows) >= 2:
        H = np.stack(profile_rows, axis=0)  # (#ckpt, T-1)
        plt.figure(figsize=(8, 3 + 0.12*len(profile_rows)))
        im = plt.imshow(H, aspect='auto', origin='lower',
                        extent=[profile_lags[0], profile_lags[-1], 0, len(profile_rows)-1])
        plt.colorbar(im, label='Normalized effective LR')
        plt.yticks(np.arange(len(profile_rows)), [str(c) for c in ckpts])
        plt.xlabel('Lag $h$')
        plt.ylabel('Checkpoint (iteration)')
        plt.title(f'Effective LR profile heatmap (task={args.task}, model={args.model})')
        plt.tight_layout()
        fpath_hm = os.path.join(args.out, f"s1_heatmap_{args.task}_{args.model}.png")
        plt.savefig(fpath_hm, dpi=300); plt.close()
        print(f"[SAVE] Heatmap -> {fpath_hm}")

    if len(slope_records) >= 2:
        its = [r["iter"] for r in slope_records]
        ss  = [r["slope"] for r in slope_records]
        r2s = [r["R2"] for r in slope_records]
        plt.figure()
        plt.plot(its, ss, marker='o', lw=2, label='slope $s(\\ell)$')
        plt.xlabel('Iteration')
        plt.ylabel('Log–log slope $s(\\ell)$')
        plt.title(f'Log–log slope over training (task={args.task}, model={args.model})')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        fpath_sv = os.path.join(args.out, f"s1_slope_vs_iter_{args.task}_{args.model}.png")
        plt.savefig(fpath_sv, dpi=300); plt.close()
        print(f"[SAVE] Slope-vs-iteration -> {fpath_sv}")


if __name__ == "__main__":
    main()