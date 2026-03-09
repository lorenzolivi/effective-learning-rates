#!/usr/bin/env python3
"""
S2: Joint anisotropy analysis (Jacobian spectra + Gradient covariance)

Single-seed logic for training three RNN variants (plain, scalar-gated,
multi-gated) on canonical sequence tasks and measuring:
  - Propagation anisotropy: singular-value spectra of Jacobian products
  - Update anisotropy: singular-value spectra of gradient covariance matrices

Supported tasks:
  • adding      – classic two-marker adding problem
  • ar2         – AR(2) latent process regression
  • movingavg   – moving average over a sliding window
  • delaysum    – weighted sum at fixed delays
  • narma10     – NARMA-10 nonlinear sequence regression
  • narma20     – NARMA-20 (longer horizon variant)

Examples:
  python s2_joint_anisotropy.py --task adding --out s2/adding --device cpu \
    --iters 1200 --checkpoints 0 400 800 1200

  python s2_joint_anisotropy.py --task narma10 --out s2/narma10 --device cpu \
    --iters 1200 --checkpoints 0 400 800 1200

  python s2_joint_anisotropy.py --task ar2 --out s2/ar2 --device cpu \
    --iters 1200 --checkpoints 0 400 800 1200
"""
import argparse, os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------- utilities -------------------------------- #

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_np(x):
    return x.detach().cpu().numpy()

def d_tanh(a):
    return 1.0 - torch.tanh(a)**2

def median_iqr(vals):
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return (np.nanmedian(vals),
            np.nanpercentile(vals, 25),
            np.nanpercentile(vals, 75))

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[saved] {path}")

def safe_unit(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmax(x)
    if np.isfinite(m) and m > 0:
        return x / m
    return x

# -------------------------------- tasks ----------------------------------- #

@torch.no_grad()
def make_adding_batch(T, B, ni_total, device, seed=None):
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    ni_base = ni_total - 1
    x = torch.rand((B, T, ni_base), generator=g, device=device, dtype=torch.float32)
    m = torch.zeros((B, T, 1), device=device, dtype=torch.float32)
    idx1 = torch.randint(0, T // 2, (B,), generator=g, device=device)
    idx2 = torch.randint(T // 2, T, (B,), generator=g, device=device)
    m[torch.arange(B), idx1, 0] = 1.0
    m[torch.arange(B), idx2, 0] = 1.0
    u = torch.cat([x, m], dim=-1)
    y = (x[..., 0:1] * m).sum(dim=1)
    return u, y

@torch.no_grad()
def make_narma10_batch(T, B, ni_total, device, seed=None):
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    u1 = 0.5 * torch.rand((B, T), generator=g, device=device, dtype=torch.float32)
    u = torch.zeros((B, T, ni_total), device=device, dtype=torch.float32)
    u[:, :, 0] = u1
    y = torch.zeros(B, T+1, device=device, dtype=torch.float32)
    for t in range(T):
        ym1 = y[:, t]
        idx_lo = max(0, t-9)
        s10 = y[:, idx_lo:t+1].sum(dim=1)
        u_t = u1[:, t]
        u_t_9 = u1[:, t-9] if t-9 >= 0 else torch.zeros_like(u_t)
        y[:, t+1] = 0.3*ym1 + 0.05*ym1*s10 + 1.5*u_t_9*u_t + 0.1
    yT = y[:, -1].unsqueeze(1)
    return u, yT

@torch.no_grad()
def make_narma20_batch(T, B, ni_total, device, seed=None):
    """
    One common NARMA20 variant:
      y_{t+1} = 0.1*y_t + 0.04*y_t * sum_{i=0..19} y_{t-i} + 1.5*u_{t-19}*u_t + 0.2
    """
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    u1 = 0.5 * torch.rand((B, T), generator=g, device=device, dtype=torch.float32)
    u = torch.zeros((B, T, ni_total), device=device, dtype=torch.float32)
    u[:, :, 0] = u1
    y = torch.zeros(B, T+1, device=device, dtype=torch.float32)
    for t in range(T):
        ym1 = y[:, t]
        idx_lo = max(0, t-19)
        s20 = y[:, idx_lo:t+1].sum(dim=1)
        u_t = u1[:, t]
        u_t_19 = u1[:, t-19] if t-19 >= 0 else torch.zeros_like(u_t)
        y[:, t+1] = 0.1*ym1 + 0.04*ym1*s20 + 1.5*u_t_19*u_t + 0.2
    yT = y[:, -1].unsqueeze(1)
    return u, yT

@torch.no_grad()
def make_movingavg_batch(T, B, ni_total, device, seed=None, W=8):
    """
    Moving-average regression:
      u[:, :, 0] ~ U(0,1). Target y = mean of last W values: (1/W) * sum_{i=T-W..T-1} u[:, i, 0].
    """
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    u = torch.rand((B, T, ni_total), generator=g, device=device, dtype=torch.float32)
    u[:, :, 1:] = 0.0  # keep only first channel active
    W = min(W, T)
    y = u[:, T-W:T, 0].mean(dim=1, keepdim=True)
    return u, y

@torch.no_grad()
def make_delaysum_batch(T, B, ni_total, device, seed=None, delays=(4,12,24,36), weights=None):
    """
    Delay-sum regression (fixed taps on first channel):
      y = sum_i w_i * u[:, T-1 - d_i, 0], clamped for early t.
    """
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    u = torch.rand((B, T, ni_total), generator=g, device=device, dtype=torch.float32)
    u[:, :, 1:] = 0.0
    d = np.array(delays, dtype=int)
    if weights is None:
        w = np.ones_like(d, dtype=np.float32) / max(1, len(d))
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = w / (np.sum(np.abs(w)) + 1e-8)
    # form targets (handle early indices)
    idxs = np.clip(T-1 - d, 0, T-1)
    y = torch.zeros(B, 1, device=device, dtype=torch.float32)
    for wi, idxt in zip(w, idxs):
        y += wi * u[:, idxt, 0:1]
    return u, y

@torch.no_grad()
def make_ar2_batch(T, B, ni_total, device, seed=None, a1=1.6, a2=-0.7, noise_std=0.05):
    """
    AR(2) latent: x_t = a1*x_{t-1} + a2*x_{t-2} + ε_t, ε_t ~ N(0, σ^2)
    Inputs: i.i.d. uniform noise in channel 0 (not used by generator but acts as distractor);
    Target: x_T.
    Coeffs chosen to be stable (roots inside unit circle).
    """
    g = torch.Generator(device=device)
    if seed is not None: g = g.manual_seed(seed)
    u = torch.rand((B, T, ni_total), generator=g, device=device, dtype=torch.float32)
    x = torch.zeros(B, T+1, device=device, dtype=torch.float32)
    eps = noise_std * torch.randn((B, T+1), generator=g, device=device, dtype=torch.float32)
    x[:, 0] = eps[:, 0]
    x[:, 1] = a1*x[:, 0] + eps[:, 1]
    for t in range(1, T):
        x[:, t+1] = a1*x[:, t] + a2*x[:, t-1] + eps[:, t+1]
    y = x[:, -1].unsqueeze(1)
    return u, y

def make_batch(task, T, B, ni, device, seed=None):
    if task == 'adding':
        return make_adding_batch(T, B, ni, device, seed)
    elif task == 'narma10':
        return make_narma10_batch(T, B, ni, device, seed)
    elif task == 'narma20':
        return make_narma20_batch(T, B, ni, device, seed)
    elif task == 'movingavg':
        return make_movingavg_batch(T, B, ni, device, seed, W=8)
    elif task == 'delaysum':
        return make_delaysum_batch(T, B, ni, device, seed, delays=(4,12,24,36))
    elif task == 'ar2':
        return make_ar2_batch(T, B, ni, device, seed)
    else:
        raise ValueError(f"Unknown task: {task}")

# -------------------------------- models ---------------------------------- #

class BaseRNN(nn.Module):
    def __init__(self, ni, nh, no):
        super().__init__()
        self.Wr = nn.Linear(nh, nh, bias=False)
        self.Wi = nn.Linear(ni, nh, bias=False)
        self.Wo = nn.Linear(nh, no, bias=False)

class PlainRNN(BaseRNN):
    def step(self, x, u):
        a = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        return phi, a, phi
    def forward_buffers(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device, dtype=u.dtype)
        X, As, Phis = [], [], []
        for t in range(T):
            x, a, phi = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi)
        X = torch.stack(X, 1)
        z = self.Wo(X[:, -1])
        return z, (X, torch.stack(As,1), torch.stack(Phis,1))

class ScalarGateRNN(BaseRNN):
    def __init__(self, ni, nh, no):
        super().__init__(ni, nh, no)
        self.Wrg = nn.Linear(nh, 1, bias=False)
        self.Wig = nn.Linear(ni, 1, bias=False)
    def step(self, x, u):
        a  = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        ag = self.Wrg(x) + self.Wig(u)
        g  = torch.sigmoid(ag)
        x_next = g * phi + (1 - g) * x
        return x_next, a, phi, g, ag
    def forward_buffers(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device, dtype=u.dtype)
        X, As, Phis, Gs, AGs = [], [], [], [], []
        for t in range(T):
            x, a, phi, g, ag = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi); Gs.append(g); AGs.append(ag)
        X = torch.stack(X, 1)
        z = self.Wo(X[:, -1])
        return z, (X, torch.stack(As,1), torch.stack(Phis,1), torch.stack(Gs,1), torch.stack(AGs,1))

class MultiGateRNN(BaseRNN):
    def __init__(self, ni, nh, no):
        super().__init__(ni, nh, no)
        self.Wrg = nn.Linear(nh, nh, bias=False)
        self.Wig = nn.Linear(ni, nh, bias=False)
    def step(self, x, u):
        a  = self.Wr(x) + self.Wi(u)
        phi = torch.tanh(a)
        ag = self.Wrg(x) + self.Wig(u)
        g  = torch.sigmoid(ag)
        x_next = g * phi + (1 - g) * x
        return x_next, a, phi, g, ag
    def forward_buffers(self, u):
        B, T, _ = u.shape
        nh = self.Wr.out_features
        x = torch.zeros(B, nh, device=u.device, dtype=u.dtype)
        X, As, Phis, Gs, AGs = [], [], [], [], []
        for t in range(T):
            x, a, phi, g, ag = self.step(x, u[:, t])
            X.append(x); As.append(a); Phis.append(phi); Gs.append(g); AGs.append(ag)
        X = torch.stack(X, 1)
        z = self.Wo(X[:, -1])
        return z, (X, torch.stack(As,1), torch.stack(Phis,1), torch.stack(Gs,1), torch.stack(AGs,1))

# ---------------------------- jacobian builders --------------------------- #

@torch.no_grad()
def jacobian_blocks_plain(model, X, As):
    B, T, nh = X.shape
    Wr = model.Wr.weight
    Js = []
    for j in range(1, T+1):
        a = As[:, j-1]
        D = torch.diag_embed(d_tanh(a))
        J = torch.matmul(D, Wr.unsqueeze(0).expand(B, -1, -1))
        Js.append(J)
    return Js

@torch.no_grad()
def jacobian_blocks_scalar(model, X, As, Gs, AGs):
    B, T, nh = X.shape
    Wr = model.Wr.weight
    I = torch.eye(nh, device=X.device).unsqueeze(0).expand(B,-1,-1)
    Js = []
    for j in range(1, T+1):
        xjm1 = X[:, j-1]
        a    = As[:, j-1]
        g    = Gs[:, j-1, :]
        ag   = AGs[:, j-1, :]
        phi  = torch.tanh(a)
        D    = torch.diag_embed(d_tanh(a))
        DWr  = torch.matmul(D, Wr.unsqueeze(0).expand(B, -1, -1))
        term1 = g.view(B,1,1) * DWr
        term2 = (1 - g).view(B,1,1) * I
        sigp  = torch.sigmoid(ag) * (1 - torch.sigmoid(ag))
        row   = sigp.view(B,1) * model.Wrg.weight
        term3 = torch.matmul((phi - xjm1).unsqueeze(2), row.unsqueeze(1))
        Js.append(term1 + term2 + term3)
    return Js

@torch.no_grad()
def jacobian_blocks_multigate(model, X, As, Gs, AGs):
    B, T, nh = X.shape
    Wr = model.Wr.weight
    Js = []
    for j in range(1, T+1):
        xjm1 = X[:, j-1]
        a    = As[:, j-1]
        g    = Gs[:, j-1, :]
        ag   = AGs[:, j-1, :]
        phi  = torch.tanh(a)
        D    = torch.diag_embed(d_tanh(a))
        DWr  = torch.matmul(D, Wr.unsqueeze(0).expand(B,-1,-1))
        term1 = torch.matmul(torch.diag_embed(g), DWr)
        term2 = torch.diag_embed(1 - g)
        sigp  = torch.sigmoid(ag) * (1 - torch.sigmoid(ag))
        Jg    = torch.matmul(torch.diag_embed(sigp), model.Wrg.weight.unsqueeze(0).expand(B,-1,-1))
        term3 = torch.matmul(torch.diag_embed(phi - xjm1), Jg)
        Js.append(term1 + term2 + term3)
    return Js

# ----------------------- pairs, products, SVD metrics --------------------- #

@torch.no_grad()
def sample_tk_pairs(T, B, lags, per_lag=64, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    pairs = {h: [] for h in lags}
    for h in lags:
        if h <= 0 or h >= T: continue
        cand = []
        for b in range(B):
            for t in range(h, T):
                k = t - h
                cand.append((b,t,k))
        if len(cand) <= per_lag:
            pairs[h] = cand
        else:
            idx = rng.choice(len(cand), size=per_lag, replace=False)
            pairs[h] = [cand[i] for i in idx]
    return pairs

@torch.no_grad()
def product_matrix(Js, b, t, k):
    nh = Js[0].shape[-1]
    M = torch.eye(nh, device=Js[0].device, dtype=Js[0].dtype)
    for j in range(t, k, -1):
        M = torch.matmul(Js[j-1][b], M)
    return M

@torch.no_grad()
def topk_svd(M, k=16):
    M64 = M.to(torch.float64)
    U, S, Vh = torch.linalg.svd(M64, full_matrices=False)
    k = min(k, S.shape[-1])
    return U[:, :k], S[:k], Vh[:k, :]

def anisotropy_index(sigmas, r=10):
    s = to_np(sigmas)
    if s.size == 0: return np.nan
    rr = min(max(1, r), len(s))
    return float(s[0] / (s[rr-1] + 1e-12))

def energy_concentration(sigmas, r=10):
    s = to_np(sigmas)
    if s.size == 0: return np.nan
    rr = min(max(1, r), len(s))
    s2 = (s**2).sum() + 1e-12
    top = (s[:rr]**2).sum()
    return float(top / s2)

# -------------------------------- training -------------------------------- #

def train_with_opt(model, opt, task, T, ni_total, device, iters, B, ckpt_set):
    loss_fn = nn.MSELoss()
    ckpts = {}
    for it in range(iters+1):
        u, y = make_batch(task, T, B, ni_total, device)
        u = u + 1e-6 * torch.randn_like(u)  # tiny jitter for GradCov stability
        yhat, _ = model.forward_buffers(u)
        loss = loss_fn(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it in ckpt_set:
            ckpts[it] = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return ckpts

def train_all_models(args, ckpt_set, device):
    plain  = PlainRNN(args.ni, args.nh, args.no).to(device)
    scalar = ScalarGateRNN(args.ni, args.nh, args.no).to(device)
    multi  = MultiGateRNN(args.ni, args.nh, args.no).to(device)

    ck_plain  = train_with_opt(plain,  optim.Adam(plain.parameters(),  lr=args.lr_adam),
                               args.task, args.T, args.ni, device, args.iters, args.B, ckpt_set)
    ck_scalar = train_with_opt(scalar, optim.SGD(scalar.parameters(), lr=args.lr_sgd),
                               args.task, args.T, args.ni, device, args.iters, args.B, ckpt_set)
    ck_multi  = train_with_opt(multi,  optim.SGD(multi.parameters(),  lr=args.lr_sgd),
                               args.task, args.T, args.ni, device, args.iters, args.B, ckpt_set)
    return plain, scalar, multi, ck_plain, ck_scalar, ck_multi

# ------------------------- Gradient covariance (robust) ------------------- #

def flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.reshape(-1))
    return torch.cat(grads, dim=0)

def collect_grad_matrix(model, task, T, Bprobe, ni_total, M_total, device, base_seed=0):
    model.zero_grad(set_to_none=True)
    loss_fn = nn.MSELoss()
    G_rows = []
    rows_done = 0
    round_size = Bprobe if Bprobe > 0 else 64

    with torch.enable_grad():
        while rows_done < M_total:
            seed = base_seed + 100000 + rows_done
            u, y = make_batch(task, T, round_size, ni_total, device, seed=seed)
            u = u + 1e-6 * torch.randn_like(u)
            for b in range(u.shape[0]):
                model.zero_grad(set_to_none=True)
                yhat, _ = model.forward_buffers(u[b:b+1])
                loss = loss_fn(yhat, y[b:b+1])
                loss.backward()
                g = flatten_grads(model).detach().cpu().to(torch.float64)
                G_rows.append(g)
                rows_done += 1
                if rows_done >= M_total: break

    G = torch.stack(G_rows, dim=0)  # (M,P) float64
    # row-direction only
    row_norm = G.norm(dim=1, keepdim=True) + 1e-12
    G = G / row_norm
    # center columns
    G = G - G.mean(dim=0, keepdim=True)
    # drop near-constant columns
    col_norm = G.norm(dim=0)
    keep = col_norm > (1e-10 * math.sqrt(G.shape[0]))
    if keep.sum() == 0:
        keep = torch.ones_like(col_norm, dtype=torch.bool)
    G = G[:, keep]
    # tiny jitter
    G = G + 1e-8 * torch.randn_like(G)
    return G

def gradcov_metrics_from_G(G, r=10):
    M, P = G.shape
    rr = max(1, min(r, min(M, P)))
    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        S = S.cpu().numpy()
    except Exception:
        if M <= P:
            C = (G @ G.T)
            ridge = (C.trace() / max(1, M)).item() * 1e-10
            C = C + ridge * torch.eye(M, dtype=C.dtype)
            evals, _ = torch.linalg.eigh(C)
        else:
            C = (G.T @ G)
            ridge = (C.trace() / max(1, P)).item() * 1e-10
            C = C + ridge * torch.eye(P, dtype=C.dtype)
            evals, _ = torch.linalg.eigh(C)
        evals = torch.clamp(evals, min=0.0).cpu().numpy()
        S = np.sqrt(np.sort(evals)[::-1])
    if S.size == 0 or S[0] <= 0:
        return float('nan'), float('nan')
    AI = float(S[0] / (S[rr-1] + 1e-12))
    CE = float(np.sum(S[:rr]**2) / (np.sum(S**2) + 1e-12))
    return AI, CE

# ----------------------------------- main --------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--task', choices=['adding','narma10','narma20','movingavg','delaysum','ar2'],
                    default='adding')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=1)

    ap.add_argument('--T', type=int, default=120)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--Bprobe', type=int, default=64)
    ap.add_argument('--M', type=int, default=256, help='rows for GradCov matrix G')

    ap.add_argument('--ni', type=int, default=6, help='total input channels')
    ap.add_argument('--nh', type=int, default=64)
    ap.add_argument('--no', type=int, default=1)

    ap.add_argument('--iters', type=int, default=800)
    ap.add_argument('--checkpoints', type=int, nargs='+',
                    default=[0, 400, 800])

    ap.add_argument('--lags', type=int, nargs='+', default=[1,2,4,8,12,16,24,32,40])
    ap.add_argument('--pairs-per-lag', type=int, default=64)
    ap.add_argument('--svd-k', type=int, default=16)
    ap.add_argument('--ai-r', type=int, default=10)
    ap.add_argument('--ce-r', type=int, default=10)

    ap.add_argument('--lr_adam', type=float, default=1e-3)
    ap.add_argument('--lr_sgd', type=float, default=1e-2)
    args = ap.parse_args()

    ensure_dir(args.out)
    set_seed(args.seed)
    device = torch.device(args.device)

    valid_lags = [h for h in args.lags if 0 < h < args.T]
    if len(valid_lags) == 0:
        raise ValueError("No valid lags (need 0 < h < T).")

    # Fixed probe batch for Jacobians
    u_probe, _ = make_batch(args.task, args.T, args.Bprobe, args.ni, device, seed=args.seed)

    ckpt_list = sorted(set(args.checkpoints))
    ckpt_set  = set(ckpt_list)

    print(f"[train] kind=plain+adam, task={args.task}")
    print(f"[train] kind=scalar+sgd, task={args.task}")
    print(f"[train] kind=multigate+sgd, task={args.task}")
    plain, scalar, multi, ck_plain, ck_scalar, ck_multi = train_all_models(args, ckpt_set, device)

    rng = np.random.RandomState(args.seed)

    all_summary = {
        "task": args.task,
        "checkpoints": ckpt_list,
        "lags": valid_lags,
        "ai_r": args.ai_r,
        "ce_r": args.ce_r,
        "svd_k": args.svd_k,
        "M_gradcov": args.M,
        "per_ckpt": {}
    }

    for it in ckpt_list:
        print(f"\n[ckpt] iter={it}")

        plain.load_state_dict(ck_plain[it])
        scalar.load_state_dict(ck_scalar[it])
        multi.load_state_dict(ck_multi[it])

        # ---------- Jacobians ----------
        with torch.no_grad():
            _, (Xp, Ap, _) = plain.forward_buffers(u_probe)
            Jp = jacobian_blocks_plain(plain, Xp, Ap)
            _, (Xs, As, _, Gs, AGs) = scalar.forward_buffers(u_probe)
            Js = jacobian_blocks_scalar(scalar, Xs, As, Gs, AGs)
            _, (Xm, Am, _, Gm, AGm) = multi.forward_buffers(u_probe)
            Jm = jacobian_blocks_multigate(multi, Xm, Am, Gm, AGm)

        pairs = sample_tk_pairs(args.T, args.Bprobe, valid_lags,
                                per_lag=args.pairs_per_lag, rng=rng)

        def collect_stats(Js):
            AI = {h: [] for h in valid_lags}
            CE = {h: [] for h in valid_lags}
            for h in valid_lags:
                for (b, t, k) in pairs[h]:
                    M = product_matrix(Js, b, t, k)
                    if not torch.isfinite(M).all():
                        continue
                    _, S, _ = topk_svd(M, k=args.svd_k)
                    AI[h].append(anisotropy_index(S, r=args.ai_r))
                    CE[h].append(energy_concentration(S, r=args.ce_r))
            xs, AI_med, CE_med = [], [], []
            for h in valid_lags:
                m_ai, _, _ = median_iqr(AI[h])
                m_ce, _, _ = median_iqr(CE[h])
                if np.isfinite(m_ai) and np.isfinite(m_ce):
                    xs.append(h); AI_med.append(m_ai); CE_med.append(m_ce)
            return np.array(xs), np.array(AI_med), np.array(CE_med)

        h_p, AI_p, CE_p = collect_stats(Jp)
        h_s, AI_s, CE_s = collect_stats(Js)
        h_m, AI_m, CE_m = collect_stats(Jm)

        plt.figure(figsize=(7.4,5.6))
        plt.plot(h_p, AI_p, marker='o', lw=2, label=f'Plain+Adam (r={args.ai_r})')
        plt.plot(h_s, AI_s, marker='^', lw=2, label=f'ScalarGate+SGD (r={args.ai_r})')
        plt.plot(h_m, AI_m, marker='s', lw=2, label=f'MultiGate+SGD (r={args.ai_r})')
        plt.yscale('log'); plt.xlabel('Lag $h$')
        plt.ylabel(r'Jacobian AI $\sigma_1/\sigma_{r}$')
        plt.title(f'Jacobian anisotropy vs lag (task={args.task}, iter={it})')
        plt.legend()
        savefig(os.path.join(args.out, f"s23_Jacobian_AI_vs_lag_{args.task}_iter{it}.png"))

        plt.figure(figsize=(7.4,5.6))
        plt.plot(h_p, CE_p, marker='o', lw=2, label=f'Plain+Adam (r={args.ce_r})')
        plt.plot(h_s, CE_s, marker='^', lw=2, label=f'ScalarGate+SGD (r={args.ce_r})')
        plt.plot(h_m, CE_m, marker='s', lw=2, label=f'MultiGate+SGD (r={args.ce_r})')
        plt.ylim(0.0, 1.02); plt.xlabel('Lag $h$')
        plt.ylabel(fr'Jacobian energy concentration CE$_{{{args.ce_r}}}$')
        plt.title(f'Jacobian energy concentration vs lag (task={args.task}, iter={it})')
        plt.legend()
        savefig(os.path.join(args.out, f"s23_Jacobian_CE_vs_lag_{args.task}_iter{it}.png"))

        # ---------- Gradient covariance ----------
        def compute_gc(model, name):
            G = collect_grad_matrix(
                model, args.task, args.T, args.Bprobe, args.ni,
                args.M, device, base_seed=args.seed + it + (0 if name=='plain' else (1000 if name=='scalar' else 2000))
            )
            AI, CE = gradcov_metrics_from_G(G, r=args.ce_r)
            return AI, CE, G.shape

        AIg_p, CEg_p, shp_p = compute_gc(plain,  'plain')
        AIg_s, CEg_s, shp_s = compute_gc(scalar, 'scalar')
        AIg_m, CEg_m, shp_m = compute_gc(multi,  'multi')

        models_x = np.array([0,1,2], dtype=float)
        labels = ['Plain+Adam', 'ScalarGate+SGD', 'MultiGate+SGD']
        vals_AI = np.array([AIg_p, AIg_s, AIg_m], dtype=float)
        vals_CE = np.array([CEg_p, CEg_s, CEg_m], dtype=float)

        vals_AI_plot = safe_unit(vals_AI)
        vals_CE_plot = safe_unit(vals_CE)
        vals_AI_plot = np.nan_to_num(vals_AI_plot, nan=0.0, posinf=np.nanmax(vals_AI_plot[np.isfinite(vals_AI_plot)]) if np.any(np.isfinite(vals_AI_plot)) else 0.0, neginf=0.0)
        vals_CE_plot = np.nan_to_num(vals_CE_plot, nan=0.0, posinf=1.0, neginf=0.0)

        plt.figure(figsize=(8.8,5.6))
        plt.plot(models_x, vals_AI_plot, marker='o', lw=2, label=f'GradCov AI (r={args.ce_r})')
        plt.plot(models_x, vals_CE_plot, marker='s', lw=2, label=f'GradCov CE (r={args.ce_r})')
        plt.xticks(models_x, labels, rotation=18)
        plt.ylim(0.0, 1.02 if np.nanmax(np.r_[vals_AI_plot, vals_CE_plot]) <= 1.2 else None)
        plt.ylabel('Normalized index (per checkpoint)')
        plt.title(f'Gradient-covariance anisotropy (task={args.task}, iter={it})')
        plt.legend()
        savefig(os.path.join(args.out, f"s23_gradcov_{args.task}_iter{it}.png"))

        all_summary["per_ckpt"][int(it)] = {
            "jacobian": {
                "plain":  {"h": h_p.tolist(), "AI_med": AI_p.tolist(), "CE_med": CE_p.tolist()},
                "scalar": {"h": h_s.tolist(), "AI_med": AI_s.tolist(), "CE_med": CE_s.tolist()},
                "multi":  {"h": h_m.tolist(), "AI_med": AI_m.tolist(), "CE_med": CE_m.tolist()},
            },
            "gradcov": {
                "plain":  {"AI": float(AIg_p), "CE": float(CEg_p), "shape": list(shp_p)},
                "scalar": {"AI": float(AIg_s), "CE": float(CEg_s), "shape": list(shp_s)},
                "multi":  {"AI": float(AIg_m), "CE": float(CEg_m), "shape": list(shp_m)},
                "normalized_plotted": {"AI": vals_AI_plot.tolist(), "CE": vals_CE_plot.tolist()}
            }
        }

    with open(os.path.join(args.out, f"s23_summary_{args.task}.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"[saved] {os.path.join(args.out, f's23_summary_{args.task}.json')}")

if __name__ == "__main__":
    main()