"""
Mamba Hawkes Process for LOBSTER LOB Modeling
==========================================================================

Given the history of LOB events, predict (i) the TYPE of the next
event (12 classes) and (ii) WHEN it occurs (inter-arrival time), trained
jointly with the point-process log-likelihood:

    L = -LL + beta * CE(next type) + gamma * MSE(next dt)

Model:
  - Event-driven selective SSM: the Mamba step size Delta_j is replaced by
    the actual inter-event gap t_j - t_{j-1}, ZOH-discretized.
  - Intensity head lambda_k(t) = softplus(alpha_k (t - t_j) + w_k^T h_j + b_k),
    non-event integral via Monte Carlo.
  - A-MHP ablations: --time-scaling (GRU-gated interval scaling)
    and --dual-channel (content/time-split state transition).
    --model amhp enables both; --model mhp enables neither.

Data: the script auto-discovers LOBSTER `*_message_*.csv` /
`*_orderbook_*.csv` pairs in its own directory (swap in other tickers by
dropping their files alongside this script).

Run from a terminal, e.g.:
    python mhp_lobster.py --model amhp --max-events 100000 --epochs 10
    python mhp_lobster.py --model mhp --dual-channel        # DC ablation
    python mhp_lobster.py --list                            # show data found

Requires: torch, numpy, pandas.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------
# 12-event taxonomy
# ----------------------------------------------------------------------------
EVENT_NAMES = [
    "LB+",  # 0  aggressive limit buy        (mid up)
    "LS-",  # 1  aggressive limit sell       (mid down)
    "MB+",  # 2  aggressive market buy       (mid up)
    "MS-",  # 3  aggressive market sell      (mid down)
    "BC-",  # 4  aggressive buy cancellation (mid down)
    "SC+",  # 5  aggressive sell cancellation(mid up)
    "LB0",  # 6  non-aggressive limit buy
    "LS0",  # 7  non-aggressive limit sell
    "MB0",  # 8  non-aggressive market buy
    "MS0",  # 9  non-aggressive market sell
    "BC0",  # 10 non-aggressive buy cancellation
    "SC0",  # 11 non-aggressive sell cancellation
]
N_EVENT_TYPES = len(EVENT_NAMES)
N_MARKET_STATES = 3  # queue-imbalance state: down/bal/up

DUMMY_ASK = 9_999_999_999
DUMMY_BID = -9_999_999_999
MIN_DT_SECONDS = 1e-9  # floor for zero gaps (same-timestamp bursts)


# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------
def discover_data(data_dir: str, ticker: str | None):
    """Find (message, orderbook) CSV pairs in data_dir."""
    msgs = sorted(glob.glob(os.path.join(data_dir, "*_message_*.csv")))
    pairs = []
    for m in msgs:
        ob = m.replace("_message_", "_orderbook_")
        if os.path.exists(ob):
            pairs.append((m, ob))
    if ticker is not None:
        pairs = [p for p in pairs
                 if os.path.basename(p[0]).upper().startswith(ticker.upper())]
    return pairs


def load_and_label(message_path: str, orderbook_path: str, max_events: int,
                   theta: float = 0.4):
    """Parse LOBSTER files -> (event_types, dts_seconds, market_states, times).

    Event classification uses the midprice move caused by each message
    (orderbook row i vs. i-1). Market state is the 3-way queue-imbalance
    regime of the book *before* each event (theta = 0.4 as in
    Shi & Cartlidge 2022 / Lalor & Swishchuk 2025).
    """
    print(f"[data] message file : {os.path.basename(message_path)}")
    print(f"[data] orderbook    : {os.path.basename(orderbook_path)}")
    nrows = None if max_events <= 0 else max_events + 1
    msg = pd.read_csv(message_path, header=None, nrows=nrows,
                      names=["time", "type", "oid", "size", "price", "dir"],
                      dtype={"time": np.float64, "type": np.int64,
                             "dir": np.int64})
    # only level-1 columns needed: ask1 p/s, bid1 p/s
    ob = pd.read_csv(orderbook_path, header=None, nrows=nrows,
                     usecols=[0, 1, 2, 3],
                     names=["ask1", "asz1", "bid1", "bsz1"])

    t = msg["time"].to_numpy()
    etype = msg["type"].to_numpy()
    edir = msg["dir"].to_numpy()
    esize = np.maximum(msg["size"].to_numpy(dtype=np.float64), 1.0)  # shares
    ask = ob["ask1"].to_numpy(dtype=np.float64)
    bid = ob["bid1"].to_numpy(dtype=np.float64)
    asz = ob["asz1"].to_numpy(dtype=np.float64)
    bsz = ob["bsz1"].to_numpy(dtype=np.float64)

    # dummy levels -> NaN, forward-fill midprice
    bad = (ask >= DUMMY_ASK) | (bid <= DUMMY_BID)
    mid = (ask + bid) / 2.0
    mid[bad] = np.nan
    mid = pd.Series(mid).ffill().bfill().to_numpy()

    dmid = np.empty_like(mid)
    dmid[0] = 0.0
    dmid[1:] = mid[1:] - mid[:-1]

    # queue imbalance of the book BEFORE the event (row i-1)
    imb = (bsz - asz) / np.maximum(bsz + asz, 1.0)
    imb_prev = np.empty_like(imb)
    imb_prev[0] = imb[0]
    imb_prev[1:] = imb[:-1]
    state = np.ones(len(imb_prev), dtype=np.int64)          # balanced
    state[imb_prev < -theta] = 0                            # sell pressure
    state[imb_prev > theta] = 2                             # buy pressure

    up, down = dmid > 0, dmid < 0
    is_limit = etype == 1
    is_cancel = (etype == 2) | (etype == 3)
    is_exec = (etype == 4) | (etype == 5)
    buy, sell = edir == 1, edir == -1

    lab = np.full(len(t), -1, dtype=np.int64)
    lab[is_limit & buy] = np.where(up[is_limit & buy], 0, 6)     # LB+/LB0
    lab[is_limit & sell] = np.where(down[is_limit & sell], 1, 7)  # LS-/LS0
    # execution of a SELL limit order == buyer-initiated (market buy)
    lab[is_exec & sell] = np.where(up[is_exec & sell], 2, 8)      # MB+/MB0
    lab[is_exec & buy] = np.where(down[is_exec & buy], 3, 9)      # MS-/MS0
    lab[is_cancel & buy] = np.where(down[is_cancel & buy], 4, 10)  # BC-/BC0
    lab[is_cancel & sell] = np.where(up[is_cancel & sell], 5, 11)  # SC+/SC0

    keep = lab >= 0  # drops halts (7) and anything unclassified
    t, lab, state, esize = t[keep], lab[keep], state[keep], esize[keep]
    if max_events > 0:
        t, lab, state, esize = (t[:max_events], lab[:max_events],
                                state[:max_events], esize[:max_events])

    dt = np.empty_like(t)
    dt[0] = MIN_DT_SECONDS
    dt[1:] = np.maximum(t[1:] - t[:-1], MIN_DT_SECONDS)

    counts = np.bincount(lab, minlength=N_EVENT_TYPES)
    print(f"[data] events kept  : {len(t):,}  "
          f"(span {t[0]:.1f}s -> {t[-1]:.1f}s after midnight)")
    print("[data] event distribution:")
    for k in range(N_EVENT_TYPES):
        print(f"         {EVENT_NAMES[k]:>4s}: {counts[k]:>8,d} "
              f"({counts[k] / len(t):6.2%})")
    print(f"[data] order size   : median {np.median(esize):,.0f}  "
          f"mean {esize.mean():,.1f}  max {esize.max():,.0f} shares")
    return lab, dt, state, t, mid[keep][:len(t)], esize


class LOBWindowDataset(Dataset):
    """Windows of L+1 consecutive events.

    inputs : types/dts/states at positions 0..L-1
             (dt_j = gap PRECEDING event j -> the SSM step size Delta_j)
    targets: type/dt at positions 1..L (next-event prediction, teacher-forced)
    """

    def __init__(self, types, dts, states, tods, sizes_z, atom_idx,
                 seq_len, stride):
        self.types = types
        self.dts = dts
        self.states = states
        self.tods = tods            # seconds after midnight of each event
        self.sizes_z = sizes_z      # standardized log order size (the mark)
        self.atom_idx = atom_idx    # index into the atom set, -1 = continuous
        self.L = seq_len
        n = len(types) - (seq_len + 1)
        self.starts = np.arange(0, max(n, 0) + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        L = self.L
        return (
            torch.from_numpy(self.types[s:s + L]).long(),
            torch.from_numpy(self.dts[s:s + L]).float(),
            torch.from_numpy(self.states[s:s + L]).long(),
            torch.from_numpy(self.tods[s:s + L]).float(),
            torch.from_numpy(self.sizes_z[s:s + L]).float(),
            torch.from_numpy(self.types[s + 1:s + L + 1]).long(),
            torch.from_numpy(self.dts[s + 1:s + L + 1]).float(),
            torch.from_numpy(self.sizes_z[s + 1:s + L + 1]).float(),
            torch.from_numpy(self.atom_idx[s + 1:s + L + 1]).long(),
        )


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps)


class EventDrivenSSM(nn.Module):
    """Selective SSM whose step size is the true inter-event gap.

    ZOH with diagonal A:  Abar = exp(dt * A),  Bbar = ((Abar - 1)/A) * B(x).
    A per-channel dt scale (softplus-positive, init ~1) lets channels sit at
    different timescales, matching the multi-exponential Hawkes recovery of
    Prop. 1 while keeping the event-driven Delta of the paper.

    dual_channel=True swaps the linear B(x) for the A-MHP dual-channel
    transition: B~ = sigmoid([W_c x ; W_tau dt']) * B_0 with the
.DS_Store
    """

    def __init__(self, d_inner, d_state, dual_channel):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.dual_channel = dual_channel

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        # per-channel dt scale, softplus(0.5413) ~= 1.0
        self.dt_scale = nn.Parameter(torch.full((d_inner,), 0.5413))
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        if dual_channel:
            self.ds_content = (2 * d_state) // 3          # 2:1 split
            self.ds_time = d_state - self.ds_content
            self.W_c = nn.Linear(d_inner, self.ds_content, bias=True)
            self.W_tau = nn.Linear(1, self.ds_time, bias=True)
            self.B0 = nn.Parameter(torch.ones(d_state))
        else:
            self.B_proj = nn.Linear(d_inner, d_state, bias=False)


    def _bmat(self, x, dt):
        """B(t_j): dual-channel (eq. 25) or plain linear projection."""
        if self.dual_channel:
            b_c = self.W_c(x)
            b_t = self.W_tau(dt.unsqueeze(-1))
            return torch.sigmoid(torch.cat([b_c, b_t], -1)) * self.B0
        return self.B_proj(x)

    def step(self, x, dt, h):
        """One event update. x: (B, d_inner), dt: (B,), h: (B, d_inner, N)."""
        A = -torch.exp(self.A_log)                              # (D, N)
        delta = dt.unsqueeze(-1) * F.softplus(self.dt_scale)    # (B, D)
        dA = delta.unsqueeze(-1) * A                            # (B, D, N)
        Abar = torch.exp(dA)
        Bcoef = (Abar - 1.0) / A
        Bmat = self._bmat(x, dt)                                # (B, N)
        Bu = Bcoef * Bmat.unsqueeze(1) * x.unsqueeze(-1)        # (B, D, N)
        h = Abar * h + Bu
        y = torch.einsum("bdn,bn->bd", h, self.C_proj(x)) + self.D * x
        return y, h

    def forward(self, x, dt):
        # x: (B, L, d_inner)   dt: (B, L) positive step sizes
        Bb, L, D = x.shape
        A = -torch.exp(self.A_log)                              # (D, N) < 0
        delta = dt.unsqueeze(-1) * F.softplus(self.dt_scale)    # (B, L, D)
        dA = delta.unsqueeze(-1) * A                            # (B, L, D, N)
        Abar = torch.exp(dA)
        Bbar_coef = (Abar - 1.0) / A                            # ZOH input coef

        Bmat = self._bmat(x, dt)                                # (B, L, N)

        Cmat = self.C_proj(x)                                   # (B, L, N)
        Bu = Bbar_coef * Bmat.unsqueeze(2) * x.unsqueeze(-1)    # (B, L, D, N)

        h = torch.zeros(Bb, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for j in range(L):
            h = Abar[:, j] * h + Bu[:, j]
            ys.append(torch.einsum("bdn,bn->bd", h, Cmat[:, j]))
        y = torch.stack(ys, dim=1)                              # (B, L, D)
        return y + self.D * x


class MambaHawkesBlock(nn.Module):
    """Pre-norm Mamba block: in_proj -> causal conv -> SiLU -> SSM -> gate."""

    def __init__(self, d_model, d_state, expand, d_conv, dual_channel):
        super().__init__()
        d_inner = expand * d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                              padding=d_conv - 1)
        self.ssm = EventDrivenSSM(d_inner, d_state, dual_channel)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)


    def init_state(self, batch, device, dtype):
        d_inner = self.conv.in_channels
        return {
            "conv": torch.zeros(batch, d_inner, self.conv.kernel_size[0],
                                device=device, dtype=dtype),
            "h": torch.zeros(batch, d_inner, self.ssm.d_state,
                             device=device, dtype=dtype),
        }

    def step(self, u, dt, state):
        """One event through the block, carrying conv buffer + SSM hidden."""
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)      # (B, d_inner)
        buf = torch.cat([state["conv"][:, :, 1:], x.unsqueeze(-1)], dim=-1)
        state["conv"] = buf
        w = self.conv.weight.squeeze(1)                          # (d_inner, k)
        x = (buf * w).sum(-1) + self.conv.bias
        x = F.silu(x)
        y, state["h"] = self.ssm.step(x, dt, state["h"])
        y = y * F.silu(z)
        return u + self.out_proj(y), state

    def forward(self, u, dt):
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x, dt) * F.silu(z)
        return u + self.out_proj(y)


class TimeScaling(nn.Module):
    """A-MHP Time-Scaling mechanism: GRU over event embeddings
    produces a positive per-event scaling s_j; the SSM consumes dt' = dt*s_j
    and the time head predicts in scaled units (divide by s_j to report)."""

    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.gru = nn.GRU(d_model, d_hidden, batch_first=True)
        self.w = nn.Linear(d_hidden, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.w.bias, 0.5413)  # softplus -> ~1 at init


    def step(self, x, l):
        """One event. x: (B, d_model), l: GRU hidden (1, B, H) or None."""
        out, l = self.gru(x.unsqueeze(1), l)
        s = F.softplus(self.w(out[:, 0])).squeeze(-1).clamp(1e-3, 1e3)
        return s, l

    def forward(self, x):
        l, _ = self.gru(x)
        return F.softplus(self.w(l)).squeeze(-1).clamp(1e-3, 1e3)  # (B, L)




class EventDrivenSSM3(nn.Module):
    """Mamba-3 selective SSM, event-driven (step size = true inter-event gap).

    Exponential-trapezoidal recurrence with the complex state
    implemented via the RoPE trick: the full complex transition
    M = e^(dA) R(d*theta) multiplies the carried terms (previous state and the
    previous input contribution), while the current input enters un-rotated:

        h_t = M [ h_(t-1) + (1-lam_t) d_t Bu_(t-1) ] + lam_t d_t Bu_t

    lam_t = sigmoid(per-channel affine of x) gates Euler (0) <-> trapezoid (1).
    Rotation angles are d_t * theta_t(x) * inv_freq: phase advances with real
    elapsed time - damped oscillatory kernels, natural for Hawkes excitation.
    B and C get RMS BCNorm + learnable biases (init 1). No short conv.
    d_state must be even (complex pairs).
    """

    def __init__(self, d_inner, d_state, dual_channel):
        super().__init__()
        assert d_state % 2 == 0, "mamba3 backbone needs an even d_state"
        self.d_inner, self.d_state = d_inner, d_state
        self.n_pairs = d_state // 2
        self.dual_channel = dual_channel

        A = torch.arange(1, self.n_pairs + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(d_inner, 1))  # (D, P)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_scale = nn.Parameter(torch.full((d_inner,), 0.5413))
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # data-dependent rotation + trapezoid gate (per-channel affine)
        inv_freq = 1.0 / (100.0 ** (torch.arange(self.n_pairs) / max(self.n_pairs - 1, 1)))
        self.register_buffer("inv_freq", inv_freq)
        self.w_theta = nn.Parameter(torch.zeros(d_inner))
        self.b_theta = nn.Parameter(torch.ones(d_inner))
        self.w_lam = nn.Parameter(torch.zeros(d_inner))
        self.b_lam = nn.Parameter(torch.zeros(d_inner))   # sigmoid(0)=0.5

        # BCNorm + biases (init 1, per reference implementation)
        self.norm_B = RMSNorm(d_state)
        self.norm_C = RMSNorm(d_state)
        self.bias_B = nn.Parameter(torch.ones(d_state))
        self.bias_C = nn.Parameter(torch.ones(d_state))

        if dual_channel:
            self.ds_content = (2 * d_state) // 3
            self.ds_time = d_state - self.ds_content
            self.W_c = nn.Linear(d_inner, self.ds_content, bias=True)
            self.W_tau = nn.Linear(1, self.ds_time, bias=True)
            self.B0 = nn.Parameter(torch.ones(d_state))
        else:
            self.B_proj = nn.Linear(d_inner, d_state, bias=False)

    def _bmat(self, x, dt):
        if self.dual_channel:
            b_c = self.W_c(x)
            b_t = self.W_tau(dt.unsqueeze(-1))
            raw = torch.sigmoid(torch.cat([b_c, b_t], -1)) * self.B0
        else:
            raw = self.B_proj(x)
        return self.norm_B(raw) + self.bias_B

    def _cmat(self, x):
        return self.norm_C(self.C_proj(x)) + self.bias_C

    @staticmethod
    def _rotate(v, cos, sin):
        """Pairwise rotation. v: (B, D, N); cos/sin: (B, D, P)."""
        v = v.view(*v.shape[:-1], -1, 2)
        v1, v2 = v[..., 0], v[..., 1]
        return torch.stack([v1 * cos - v2 * sin,
                            v1 * sin + v2 * cos], dim=-1).flatten(-2)

    def _coeffs(self, x, dt):
        """Shared per-step coefficient computation for forward() and step()."""
        delta = dt.unsqueeze(-1) * F.softplus(self.dt_scale)      # (..., D)
        A = -torch.exp(self.A_log)                                # (D, P)
        decay = torch.exp(delta.unsqueeze(-1) * A)                # (..., D, P)
        decay = decay.repeat_interleave(2, dim=-1)                # (..., D, N)
        theta = x * self.w_theta + self.b_theta                   # (..., D)
        ang = (delta * theta).unsqueeze(-1) * self.inv_freq       # (..., D, P)
        lam = torch.sigmoid(x * self.w_lam + self.b_lam)          # (..., D)
        return delta, decay, torch.cos(ang), torch.sin(ang), lam

    def forward(self, x, dt):
        Bb, L, D = x.shape
        delta, decay, cos, sin, lam = self._coeffs(x, dt)
        Bmat = self._bmat(x, dt)                                  # (B, L, N)
        Cmat = self._cmat(x)                                      # (B, L, N)
        Bu = Bmat.unsqueeze(2) * x.unsqueeze(-1)                  # (B, L, D, N)
        h = torch.zeros(Bb, D, self.d_state, device=x.device, dtype=x.dtype)
        prev = torch.zeros_like(h)
        ys = []
        for j in range(L):
            carry = h + ((1.0 - lam[:, j]) * delta[:, j]).unsqueeze(-1) * prev
            carry = self._rotate(carry, cos[:, j], sin[:, j])
            h = decay[:, j] * carry                 + (lam[:, j] * delta[:, j]).unsqueeze(-1) * Bu[:, j]
            prev = Bu[:, j]
            ys.append(torch.einsum("bdn,bn->bd", h, Cmat[:, j]))
        y = torch.stack(ys, dim=1)
        return y + self.D * x

    def step(self, x, dt, state):
        """One event. x: (B, D), dt: (B,); state carries h and prev Bu."""
        delta, decay, cos, sin, lam = self._coeffs(x, dt)
        Bu = self._bmat(x, dt).unsqueeze(1) * x.unsqueeze(-1)     # (B, D, N)
        carry = state["h"] + ((1.0 - lam) * delta).unsqueeze(-1) * state["prev"]
        carry = self._rotate(carry, cos, sin)
        h = decay * carry + (lam * delta).unsqueeze(-1) * Bu
        state["h"], state["prev"] = h, Bu
        y = torch.einsum("bdn,bn->bd", h, self._cmat(x)) + self.D * x
        return y, state


class Mamba3Block(nn.Module):
    """Mamba-3 block: pre-norm, in_proj -> (x, z), NO short conv (removed in
    Mamba-3; the trapezoidal recurrence + BC biases replace it), SSM, gate by
    SiLU(z), out_proj + residual. Post-gate RMSNorm removed per the paper's
    pure-Mamba-3 finding (BCNorm suffices)."""

    def __init__(self, d_model, d_state, expand, dual_channel):
        super().__init__()
        d_inner = expand * d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.ssm = EventDrivenSSM3(d_inner, d_state, dual_channel)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u, dt):
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        y = self.ssm(x, dt) * F.silu(z)
        return u + self.out_proj(y)

    def init_state(self, batch, device, dtype):
        d_inner = self.out_proj.in_features
        z = torch.zeros(batch, d_inner, self.ssm.d_state,
                        device=device, dtype=dtype)
        return {"h": z, "prev": z.clone()}

    def step(self, u, dt, state):
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        y, state = self.ssm.step(x, dt, state)
        y = y * F.silu(z)
        return u + self.out_proj(y), state


class SwiGLUBlock(nn.Module):
    """Pre-norm SwiGLU MLP block, alternated with Mamba-3 blocks
    (Llama-style layout per the Mamba-3 architecture section)."""

    def __init__(self, d_model, expand=2):
        super().__init__()
        d_h = expand * d_model
        self.norm = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_h, bias=False)
        self.w_up = nn.Linear(d_model, d_h, bias=False)
        self.w_down = nn.Linear(d_h, d_model, bias=False)

    def forward(self, u, dt=None):
        h = self.norm(u)
        return u + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))

    def init_state(self, batch, device, dtype):
        return {}

    def step(self, u, dt, state):
        return self.forward(u), state


MARKET_OPEN_S = 34200.0   # 9:30
DAY_MINUTES = 390.0


class BathtubBaseline(nn.Module):
    """Time-of-day baseline, Van Vliet eq. (7), per event type.

    mu_k(t) = softplus(g_k) + softplus(n0_k) e^(-ul*tau) + softplus(nT_k) e^(-ur*(T-tau))
    with tau in MINUTES since the open and FIXED decay rates ul, ur (per
    minute) - only the per-type levels are learned. Output units are events
    per normalized-time unit, matching the excitation head, so the learnable
    amplitudes absorb the unit conversion. g_k overlaps with the excitation
    b_k by design; harmless redundancy.
    """

    def __init__(self, n_types, ups_l=0.15, ups_r=0.15):
        super().__init__()
        self.ups_l, self.ups_r = ups_l, ups_r
        init = -4.0  # softplus(-4) ~ 0.018: start near-flat
        self.g = nn.Parameter(torch.full((n_types,), init))
        self.n0 = nn.Parameter(torch.full((n_types,), init))
        self.nT = nn.Parameter(torch.full((n_types,), init))

    def forward(self, tod_seconds):
        """tod_seconds: (...,) absolute seconds after midnight -> (..., K)."""
        tau = ((tod_seconds - MARKET_OPEN_S) / 60.0).clamp(0.0, DAY_MINUTES)
        left = torch.exp(-self.ups_l * tau.unsqueeze(-1))
        right = torch.exp(-self.ups_r * (DAY_MINUTES - tau).unsqueeze(-1))
        return (F.softplus(self.g) + F.softplus(self.n0) * left
                + F.softplus(self.nT) * right)


class MambaHawkesProcess(nn.Module):
    def __init__(self, n_types=N_EVENT_TYPES, d_model=64, d_state=16,
                 n_layers=2, expand=2, d_conv=4, time_scaling=False,
                 dual_channel=False, use_market_state=True,
                 baseline="const", bathtub_ups=(0.15, 0.15),
                 backbone="mamba3", size_mark=True, size_mix=5,
                 size_atoms=16):
        super().__init__()
        self.backbone = backbone
        self.baseline_mode = baseline
        self.dt_mean_s = 1.0   # set via set_time_context(); NOT in state_dict
        # mark context: train mean/std of log(size), set via
        # set_size_context(). Plain attributes (not buffers) so
        # --no-size-mark state dicts stay byte-compatible with v4.
        self.size_mark = size_mark
        self.size_mix = size_mix
        self.size_atoms = size_atoms
        self.n_types = n_types
        self.logm_mean = 0.0
        self.logm_std = 1.0
        if baseline == "bathtub":
            self.bathtub = BathtubBaseline(n_types, *bathtub_ups)
        self.time_scaling_on = time_scaling
        self.use_market_state = use_market_state
        self.embed = nn.Embedding(n_types, d_model)
        if use_market_state:
            self.state_embed = nn.Embedding(N_MARKET_STATES, d_model)
        self.time_scaling = TimeScaling(d_model) if time_scaling else None
        if backbone == "mamba3":
            blocks = []
            for _ in range(n_layers):
                blocks.append(Mamba3Block(d_model, d_state, expand,
                                          dual_channel))
                blocks.append(SwiGLUBlock(d_model))
            self.blocks = nn.ModuleList(blocks)
        else:  # "mamba2": the v2/v3 block, checkpoint-compatible
            self.blocks = nn.ModuleList([
                MambaHawkesBlock(d_model, d_state, expand, d_conv,
                                 dual_channel)
                for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)

        # intensity head (eq. 15-16): lambda_k(u|h) = softplus(a_k u + w_k.h + b_k)
        self.alpha = nn.Parameter(torch.zeros(n_types))
        self.w_int = nn.Linear(d_model, n_types)
        # prediction heads (eq. 19)
        self.type_head = nn.Linear(d_model, n_types)
        self.time_head = nn.Linear(d_model, 1)
        # v5 mark modules: input projection of standardized log-size, and a
        # per-type mixture-density head f_k(z|h) on the standardized log-size
        # axis (weights, means, log-stds for size_mix components per type).
        if size_mark:
            self.size_proj = nn.Linear(1, d_model, bias=False)
            self.size_head = nn.Linear(d_model, n_types * size_mix * 3)
            if size_atoms > 0:
                # per-type logits over [atom_1..atom_N, continuous branch]
                self.size_atom_head = nn.Linear(
                    d_model, n_types * (size_atoms + 1))
                # the atom share values (train top-N); buffer -> saved in
                # the checkpoint, so a loaded model is self-contained
                self.register_buffer(
                    "atom_sizes", torch.zeros(size_atoms))

    def set_time_context(self, dt_mean_seconds):
        """Mean training gap in seconds: converts normalized elapsed time
        to wall-clock for the bathtub. Plain attribute (not a buffer) so
        const-mode state dicts stay byte-compatible with v2 checkpoints."""
        self.dt_mean_s = float(dt_mean_seconds)

    def set_size_context(self, logm_mean, logm_std):
        """Train-split mean/std of log(order size): maps shares <-> the
        standardized log axis the mark head lives on."""
        self.logm_mean = float(logm_mean)
        self.logm_std = float(max(logm_std, 1e-6))

    def set_size_atoms(self, atoms, marginal=None):
        """Install the train-split top-N exact sizes as the atom set.
        marginal (optional, len N+1): train frequencies of each atom plus
        the continuous remainder; used to initialize the atom-head bias at
        the empirical log-marginal, so the atom-vs-slab split starts
        calibrated instead of uniform (uniform init puts prior mass
        N/(N+1) on atoms)."""
        with torch.no_grad():
            self.atom_sizes.copy_(torch.as_tensor(
                atoms, dtype=self.atom_sizes.dtype,
                device=self.atom_sizes.device))
            if marginal is not None:
                lp = torch.log(torch.as_tensor(
                    marginal, dtype=self.size_atom_head.bias.dtype,
                    device=self.size_atom_head.bias.device).clamp(min=1e-6))
                self.size_atom_head.bias.copy_(lp.repeat(self.n_types))
                self.size_atom_head.weight.mul_(0.01)  # start near-marginal

    def size_to_z(self, shares):
        return (np.log(np.maximum(shares, 1.0)) - self.logm_mean) / self.logm_std



    def init_sim_state(self, batch, device, dtype=torch.float32):
        return {"blocks": [b.init_state(batch, device, dtype)
                           for b in self.blocks],
                "gru": None}

    def step(self, types, dts, states, sim, sizes_z=None):
        """Process ONE event (type k_j, gap dt_j, mark z_j) and return
        (h_j, s_j, sim). Mirrors forward() exactly, but incrementally."""
        x = self.embed(types)                                    # (B, d_model)
        if self.use_market_state and states is not None:
            x = x + self.state_embed(states)
        if self.size_mark and sizes_z is not None:
            x = x + self.size_proj(sizes_z.unsqueeze(-1))
        if self.time_scaling_on:
            s, sim["gru"] = self.time_scaling.step(x, sim["gru"])
        else:
            s = torch.ones_like(dts)
        dt_eff = dts * s
        for blk, st in zip(self.blocks, sim["blocks"]):
            x, _ = blk.step(x, dt_eff, st)
        return self.norm_f(x), s, sim

    def forward(self, types, dts, states=None, sizes_z=None):
        x = self.embed(types)
        if self.use_market_state and states is not None:
            x = x + self.state_embed(states)
        if self.size_mark and sizes_z is not None:
            x = x + self.size_proj(sizes_z.unsqueeze(-1))
        s = self.time_scaling(x) if self.time_scaling_on \
            else torch.ones_like(dts)
        dt_eff = dts * s
        for blk in self.blocks:
            x = blk(x, dt_eff)
        h = self.norm_f(x)                                   # (B, L, d_model)
        return h, s

    def intensity(self, h, u, tod=None, s=None):
        """lambda_k at elapsed time u after each hidden state.
        h: (B, L, D);  u: (B, L) or (B, L, M) -> returns (..., K)."""
        base = self.w_int(h)                                 # (B, L, K)
        mc = u.dim() != h.dim() - 1
        if not mc:
            arg = self.alpha * u.unsqueeze(-1) + base
        else:                                                # MC samples
            arg = (self.alpha * u.unsqueeze(-1)
                   + base.unsqueeze(2))                      # (B, L, M, K)
        lam = F.softplus(arg, beta=1.0) + 1e-9
        # Bathtub baseline: mu is a REAL-clock intensity; on the scaled axis
        # it contributes mu(t_j + (u/s)*dt_mean)/s (change-of-variables).
        if self.baseline_mode == "bathtub" and tod is not None:
            if s is None:
                s = torch.ones_like(tod)
            if mc:
                t_abs = tod.unsqueeze(-1) + (u / s.unsqueeze(-1)) * self.dt_mean_s
                lam = lam + self.bathtub(t_abs) / s.unsqueeze(-1).unsqueeze(-1)
            else:
                t_abs = tod + (u / s) * self.dt_mean_s
                lam = lam + self.bathtub(t_abs) / s.unsqueeze(-1)
        return lam

    @torch.no_grad()
    def expected_dt(self, h, t_max=50.0, n_grid=200, tod=None, s=None):
        """E[dt] under the learned intensity (scaled-time units, same axis the
        LL was trained on): E = int_0^inf S(u) du with S(u) = exp(-int Lambda).

        Numerical: log-spaced grid on (0, t_max] (gaps concentrate near zero
        but the survival tail is long), trapezoid for both the compensator and
        the outer integral. Truncation at t_max biases E downward only when
        substantial survival mass lies beyond it; with normalized time
        (mean gap = 1), t_max = 50 covers all but pathological states.
        h: (B, L, D) -> returns (B, L).
        """
        dev = h.device
        # build the grid on CPU: aten::logspace has no MPS kernel; the tensor
        # is tiny (n_grid+1 floats) so the transfer is negligible
        grid = torch.logspace(-3, math.log10(t_max), n_grid)
        u = torch.cat([torch.zeros(1), grid]).to(dev)          # (M,)
        B, L, _ = h.shape
        ug = u.view(1, 1, -1).expand(B, L, -1)                 # (B, L, M)
        lam = self.intensity(h, ug, tod, s).sum(-1)            # (B, L, M)
        du = (u[1:] - u[:-1]).view(1, 1, -1)
        comp = torch.cumsum(0.5 * (lam[..., 1:] + lam[..., :-1]) * du, dim=-1)
        S = torch.cat([torch.ones(B, L, 1, device=dev),
                       torch.exp(-comp)], dim=-1)              # (B, L, M)
        return (0.5 * (S[..., 1:] + S[..., :-1]) * du).sum(-1)  # (B, L)

    def log_likelihood(self, h, target_types, target_dts, n_mc=10,
                       tod=None, s=None):
        """Per-window LL (eq. 17) with Monte Carlo non-event integral."""
        lam_at_event = self.intensity(h, target_dts, tod, s)  # (B, L, K)
        log_lam = torch.log(
            lam_at_event.gather(-1, target_types.unsqueeze(-1)).squeeze(-1))
        u = torch.rand(*target_dts.shape, n_mc,
                       device=h.device) * target_dts.unsqueeze(-1)
        lam_total = self.intensity(h, u, tod, s).sum(-1)      # (B, L, M)
        integral = lam_total.mean(-1) * target_dts           # (B, L)
        return log_lam - integral                            # (B, L)

    # ---- v5 mark kernel: f_k(m | h) --------------------------------------
    def _mark_params(self, h):
        """h: (..., D) -> (logw, mu, log_sig), each (..., K, M) on the
        standardized log-size axis."""
        p = self.size_head(h).view(*h.shape[:-1], self.n_types,
                                   self.size_mix, 3)
        logw = F.log_softmax(p[..., 0], dim=-1)
        mu = p[..., 1]
        log_sig = p[..., 2].clamp(-4.0, 4.0)
        return logw, mu, log_sig

    def mark_log_density(self, h, target_types, target_z, target_atom=None):
        """Per-event mark log-likelihood. With atoms (v5.1 hybrid):

            m in atoms:  log p_atom(i | h, k)                (probability)
            m outside :  log p_cont(h, k) + log g_k(log m|h) (density,
                                                              log-share units)

        i.e. the LL under the mixed base measure (counting on atoms +
        Lebesgue elsewhere). Without atoms (size_atoms=0), reduces to the
        v5.0 pure mixture density. target_atom: (B, L) long, -1 = non-atom.
        h: (B, L, D) -> (B, L)."""
        logw, mu, log_sig = self._mark_params(h)              # (B, L, K, M)
        idx = target_types.unsqueeze(-1).unsqueeze(-1).expand(
            *target_types.shape, 1, self.size_mix)
        logw = logw.gather(-2, idx).squeeze(-2)               # (B, L, M)
        mu = mu.gather(-2, idx).squeeze(-2)
        log_sig = log_sig.gather(-2, idx).squeeze(-2)
        z = target_z.unsqueeze(-1)
        comp = (logw - 0.5 * ((z - mu) / log_sig.exp()) ** 2
                - log_sig - 0.5 * math.log(2 * math.pi))
        log_g = torch.logsumexp(comp, dim=-1) - math.log(self.logm_std)
        if self.size_atoms == 0 or target_atom is None:
            return log_g
        # hybrid: per-type log-probs over [atoms..., continuous]
        la = self.size_atom_head(h).view(
            *h.shape[:-1], self.n_types, self.size_atoms + 1)
        idx_t = target_types.unsqueeze(-1).unsqueeze(-1).expand(
            *target_types.shape, 1, self.size_atoms + 1)
        la = F.log_softmax(la.gather(-2, idx_t).squeeze(-2), dim=-1)
        is_atom = target_atom >= 0
        atom_lp = la.gather(-1, target_atom.clamp(min=0)
                            .unsqueeze(-1)).squeeze(-1)
        cont_lp = la[..., -1] + log_g
        return torch.where(is_atom, atom_lp, cont_lp)

    @torch.no_grad()
    def sample_size(self, h, k, rng=None):
        """Draw one order size from f_k(.|h). h: (1, D), k: int.
        Hybrid: first pick atom-vs-slab; atoms emit the exact round lot.
        Returns (z, shares) with shares an integer >= 1."""
        if self.size_atoms > 0:
            la = self.size_atom_head(h).view(
                1, self.n_types, self.size_atoms + 1)[0, k]
            c = int(torch.multinomial(F.softmax(la, -1).cpu(), 1,
                                      generator=rng))
            if c < self.size_atoms:                      # exact atom
                shares = float(self.atom_sizes[c])
                z_fb = (math.log(max(shares, 1.0))
                        - self.logm_mean) / self.logm_std
                return z_fb, shares
        logw, mu, log_sig = self._mark_params(h)              # (1, K, M)
        logw, mu, log_sig = logw[0, k], mu[0, k], log_sig[0, k]
        c = int(torch.multinomial(logw.exp().cpu(), 1, generator=rng))
        eps = torch.randn(1, generator=rng).item()
        z = float(mu[c]) + math.exp(float(log_sig[c])) * eps
        shares = max(1.0, round(math.exp(z * self.logm_std + self.logm_mean)))
        # NOTE: rounding can land the slab draw ON an atom value; that is a
        # harmless simulation-side overlap (the likelihood never sees it).
        # re-standardize the REALIZED (rounded) size for the feedback input
        z_fb = (math.log(shares) - self.logm_mean) / self.logm_std
        return z_fb, shares


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "mhp"
    time_scaling: bool = False
    dual_channel: bool = False
    use_market_state: bool = True
    max_events: int = 100_000
    seq_len: int = 128
    stride: int = 128
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 2
    expand: int = 2
    d_conv: int = 4
    beta: float = 1.0     # event-type CE weight  (eq. 21)
    gamma: float = 0.01   # time MSE weight       (eq. 21)
    n_mc: int = 10
    size_mark: bool = True      # v5: order size as an explicit mark
    size_mix: int = 5           # mixture components of f_k(log m | h)
    size_atoms: int = 16        # v5.1: top-N exact-size point masses (0=off)
    size_weight: float = 1.0    # w_sz on the mark NLL in the loss
    backbone: str = "mamba3"
    baseline: str = "const"
    bathtub_ups_l: float = 0.15
    bathtub_ups_r: float = 0.15
    time_pred: str = "head"
    time_grid_max: float = 50.0
    time_grid_n: int = 200
    seed: int = 42
    device: str = "auto"
    log_interval: int = 50
    num_workers: int = 0


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, cfg, device, optimizer=None, dt_mean=1.0):
    training = optimizer is not None
    model.train(training)
    tot = dict(loss=0.0, nll=0.0, ce=0.0, mse=0.0, correct=0, n=0, se=0.0,
               sz=0.0)
    t0, seen = time.time(), 0
    for bi, (types, dts, states, tods, zs, tt, tdt, tz, ta) in enumerate(loader):
        types, dts, states = types.to(device), dts.to(device), states.to(device)
        tods, tt, tdt = tods.to(device), tt.to(device), tdt.to(device)
        zs, tz, ta = zs.to(device), tz.to(device), ta.to(device)

        with torch.set_grad_enabled(training):
            h, s = model(types, dts, states, zs)
            # scaled target gaps: TS trains/predicts in scaled time (Sec. 6.4)
            tdt_s = tdt * s
            ll = model.log_likelihood(h, tt, tdt_s, cfg.n_mc, tods, s)
            nll = -ll.mean()
            logits = model.type_head(h)
            ce = F.cross_entropy(logits.reshape(-1, N_EVENT_TYPES),
                                 tt.reshape(-1))
            head_dt_s = model.time_head(h).squeeze(-1)
            mse = F.mse_loss(head_dt_s, tdt_s)   # loss always trains the head
            if cfg.time_pred == "intensity" and not training:
                pred_dt_s = model.expected_dt(h, cfg.time_grid_max,
                                              cfg.time_grid_n, tods, s)
            else:
                pred_dt_s = head_dt_s
            if model.size_mark:
                size_nll = -model.mark_log_density(h, tt, tz, ta).mean()
            else:
                size_nll = torch.zeros((), device=h.device)
            loss = (nll + cfg.size_weight * size_nll
                    + cfg.beta * ce + cfg.gamma * mse)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            n = tt.numel()
            tot["loss"] += loss.item() * n
            tot["nll"] += nll.item() * n
            tot["sz"] += size_nll.item() * n
            tot["ce"] += ce.item() * n
            tot["mse"] += mse.item() * n
            tot["correct"] += (logits.argmax(-1) == tt).sum().item()
            # RMSE in real (unscaled, normalized) units, then seconds
            pred_dt = (pred_dt_s / s).clamp(min=0)
            tot["se"] += ((pred_dt - tdt) ** 2).sum().item()
            tot["n"] += n
            seen += n

        if training and (bi + 1) % cfg.log_interval == 0:
            print(f"    batch {bi + 1:>4d}/{len(loader)}  "
                  f"loss {tot['loss'] / tot['n']:.4f}  "
                  f"nll {tot['nll'] / tot['n']:.4f}  "
                  f"acc {tot['correct'] / tot['n']:.4f}  "
                  f"({seen / (time.time() - t0):,.0f} ev/s)")

    n = max(tot["n"], 1)
    return dict(loss=tot["loss"] / n, nll=tot["nll"] / n, ce=tot["ce"] / n,
                acc=tot["correct"] / n,
                rmse_norm=math.sqrt(tot["se"] / n),
                rmse_sec=math.sqrt(tot["se"] / n) * dt_mean,
                ll_per_event=-tot["nll"] / n,           # time+type (v4-comparable)
                size_nll=tot["sz"] / n,                 # mark term, log-share units
                ll_marked_per_event=(-tot["nll"] - tot["sz"]) / n)


@torch.no_grad()
def confusion_report(model, loader, device):
    model.eval()
    cm = np.zeros((N_EVENT_TYPES, N_EVENT_TYPES), dtype=np.int64)
    for types, dts, states, _tods, zs, tt, _, _, _ in loader:
        h, _ = model(types.to(device), dts.to(device), states.to(device),
                     zs.to(device))
        pred = model.type_head(h).argmax(-1).cpu().numpy().ravel()
        true = tt.numpy().ravel()
        np.add.at(cm, (true, pred), 1)
    print("\n[test] per-class next-event accuracy (recall):")
    for k in range(N_EVENT_TYPES):
        row = cm[k].sum()
        acc = cm[k, k] / row if row else float("nan")
        print(f"         {EVENT_NAMES[k]:>4s}: n={row:>8,d}  recall={acc:.4f}")
    return cm



# ----------------------------------------------------------------------------
# Simulation: Ogata's thinning on the learned intensity (Lalor & Swishchuk
# Sec. 3.2, eq. 9), then an event-driven midprice path (their eq. 10-11).
# ----------------------------------------------------------------------------
UP_EVENTS = [0, 2, 5]     # LB+, MB+, SC+  -> midprice up
DOWN_EVENTS = [1, 3, 4]   # LS-, MS-, BC-  -> midprice down


@torch.no_grad()
def _total_intensity(model, h, u, tod=None, s=None):
    """Lambda(u) and per-type lambda_k(u) at scaled elapsed time u after the
    last event. h: (1, D). u: scalar tensor. Returns (float, (K,) tensor)."""
    kw = {}
    if tod is not None:
        kw = dict(tod=torch.full((1, 1), float(tod), device=h.device),
                  s=torch.full((1, 1), float(s), device=h.device))
    lam = model.intensity(h.unsqueeze(1), u.view(1, 1), **kw)[0, 0]   # (K,)
    return float(lam.sum()), lam


@torch.no_grad()
def _bound_on_window(model, h, u0, w, tod=None, s=1.0):
    """A valid upper bound of Lambda on [u0, u0+w] in scaled time.

    Each lambda_k = softplus(alpha_k * u + c_k) is monotone in u (direction =
    sign(alpha_k)), so the windowed supremum is attained at an endpoint per
    type: u0+w where alpha_k > 0, u0 where alpha_k <= 0. Exact, not a heuristic
    safety factor - thinning stays unbiased even with learned alpha_k > 0.
    """
    base = model.w_int(h)[0]                                    # (K,)
    a = model.alpha
    u_star = torch.where(a > 0, torch.full_like(a, u0 + w),
                         torch.full_like(a, u0))
    bound = float((F.softplus(a * u_star + base) + 1e-9).sum())
    if model.baseline_mode == "bathtub" and tod is not None:
        # each wing is monotone in real time: left wing max at the window
        # start, right wing max at the window end; exact endpoint bound.
        bt = model.bathtub
        t0 = tod + (u0 / s) * model.dt_mean_s
        t1 = tod + ((u0 + w) / s) * model.dt_mean_s
        tau0 = max(0.0, min((t0 - MARKET_OPEN_S) / 60.0, DAY_MINUTES))
        tau1 = max(0.0, min((t1 - MARKET_OPEN_S) / 60.0, DAY_MINUTES))
        mu_max = (F.softplus(bt.g) + F.softplus(bt.n0) * math.exp(-bt.ups_l * tau0)
                  + F.softplus(bt.nT) * math.exp(-bt.ups_r * (DAY_MINUTES - tau1)))
        bound += float(mu_max.sum()) / s
    return bound


@torch.no_grad()
def simulate_events(model, seed_types, seed_dts, seed_states, seed_tods,
                    n_events, device, sim_state_mode="balanced", max_gap=1e4,
                    rng=None, seed_sizes_z=None):
    """Generate n_events future MARKED events (type, dt, order size) with
    Ogata's thinning. The mark does not change thinning itself (the ground
    intensity is mark-free); after each accepted (dt, k), a size is drawn
    from f_k(.|h) and fed back into the state autoregressively.

    Warm-up: the seed window is replayed through model.step() so the SSM
    hidden, conv buffers, and (A-MHP) GRU state carry the real history.
    Thinning runs on the SCALED time axis (the axis the likelihood was
    trained on when Time-Scaling is active); accepted scaled waits u are
    converted to real time via dt = u / s_j, where s_j is fixed over the
    interval because it depends only on events up to j.

    Market state is not observable without simulating the whole book, so
    the simulated stream uses a fixed state: "balanced" (imbalance regime 1)
    or "last" (carry the final seed state). Returns (types, dts) as numpy.
    """
    rng = rng or torch.Generator(device="cpu")
    model.eval()
    sim = model.init_sim_state(1, device)
    h = s = None
    last_state = 1
    for j in range(len(seed_types)):
        tt = torch.tensor([seed_types[j]], device=device)
        dd = torch.tensor([seed_dts[j]], dtype=torch.float32, device=device)
        ss = torch.tensor([seed_states[j]], device=device)
        zz = None if seed_sizes_z is None else torch.tensor(
            [seed_sizes_z[j]], dtype=torch.float32, device=device)
        h, s, sim = model.step(tt, dd, ss, sim, zz)
        last_state = int(seed_states[j])
    fixed_state = 1 if sim_state_mode == "balanced" else last_state
    wall = float(seed_tods[-1])          # simulation wall clock (s after midnight)

    out_types, out_dts, out_sizes = [], [], []
    for _ in range(n_events):
        s_j = float(s[0])
        u = 0.0                                     # scaled elapsed time
        while True:
            lam_now, _ = _total_intensity(model, h, torch.tensor(u, device=device),
                                          wall, s_j)
            w = min(max(3.0 / max(lam_now, 1e-8), 1e-3), max_gap)
            lam_bar = _bound_on_window(model, h, u, w, wall, s_j)
            e = -math.log(1.0 - torch.rand(1, generator=rng).item()) / lam_bar
            if e > w:                               # no event in this window
                u += w
                if u / s_j > max_gap:               # intensity died out
                    u = max_gap * s_j
                    lam_at = _total_intensity(model, h, torch.tensor(u, device=device),
                                              wall, s_j)[1]
                    break
                continue
            u += e
            lam_tot, lam_at = _total_intensity(model, h, torch.tensor(u, device=device),
                                              wall, s_j)
            if torch.rand(1, generator=rng).item() * lam_bar <= lam_tot:
                break                               # accepted
        probs = (lam_at / lam_at.sum()).cpu()
        k = int(torch.multinomial(probs, 1, generator=rng))
        dt_real = u / s_j                           # back to real (normalized) time
        wall += dt_real * model.dt_mean_s           # advance the day clock
        if model.size_mark:
            z_fb, shares = model.sample_size(h, k, rng)
            zz = torch.tensor([z_fb], dtype=torch.float32, device=device)
        else:
            zz, shares = None, float("nan")
        out_types.append(k)
        out_dts.append(dt_real)
        out_sizes.append(shares)
        tt = torch.tensor([k], device=device)
        dd = torch.tensor([dt_real], dtype=torch.float32, device=device)
        ss = torch.tensor([fixed_state], device=device)
        h, s, sim = model.step(tt, dd, ss, sim, zz)
    return np.array(out_types), np.array(out_dts), np.array(out_sizes)


def empirical_jump_dists(types, mid):
    """Empirical |midprice jump| distributions for up and down events from the
    real data (Lalor eq. 11: discrete jump sizes with empirical probabilities)."""
    dmid = np.diff(mid, prepend=mid[0])
    dists = {}
    for name, evs, sign in [("up", UP_EVENTS, 1), ("down", DOWN_EVENTS, -1)]:
        mask = np.isin(types, evs)
        jumps = np.abs(dmid[mask])
        jumps = jumps[jumps > 0]
        if len(jumps) == 0:
            jumps = np.array([1.0])
        vals, counts = np.unique(jumps, return_counts=True)
        dists[name] = (vals, counts / counts.sum())
    return dists


def build_midprice(sim_types, v0, jump_dists, rng):
    """V(t+dt) = V(t) + sgn * |jump| with jumps drawn from the empirical
    per-direction distributions; non-aggressive events leave V unchanged."""
    path = np.empty(len(sim_types) + 1)
    path[0] = v0
    v = v0
    for i, k in enumerate(sim_types):
        if k in UP_EVENTS:
            vals, p = jump_dists["up"]
            v += rng.choice(vals, p=p)
        elif k in DOWN_EVENTS:
            vals, p = jump_dists["down"]
            v -= rng.choice(vals, p=p)
        path[i + 1] = v
    return path


def hurst_rs(x):
    """Hurst exponent of a return series via rescaled range over dyadic blocks."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return float("nan")
    sizes, rs = [], []
    n = 16
    while n <= len(x) // 2:
        m = len(x) // n
        chunks = x[:m * n].reshape(m, n)
        d = chunks - chunks.mean(axis=1, keepdims=True)
        z = d.cumsum(axis=1)
        r = z.max(axis=1) - z.min(axis=1)
        sd = chunks.std(axis=1, ddof=1)
        ok = sd > 0
        if ok.sum() > 0:
            sizes.append(n)
            rs.append((r[ok] / sd[ok]).mean())
        n *= 2
    if len(sizes) < 3:
        return float("nan")
    slope = np.polyfit(np.log(sizes), np.log(rs), 1)[0]
    return float(slope)


def stylized_stats(prices):
    """Volatility, absolute skewness, excess kurtosis, Hurst of log returns
    (the Lalor & Swishchuk Table 4 panel)."""
    p = np.asarray(prices, dtype=np.float64)
    p = p[p > 0]
    r = np.diff(np.log(p))
    r = r[np.isfinite(r)]
    if len(r) < 8:
        return dict(vol=float("nan"), abs_skew=float("nan"),
                    ex_kurt=float("nan"), hurst=float("nan"))
    mu, sd = r.mean(), r.std(ddof=1)
    if sd == 0:
        return dict(vol=0.0, abs_skew=0.0, ex_kurt=0.0, hurst=float("nan"))
    z = (r - mu) / sd
    return dict(vol=float(sd), abs_skew=float(abs((z ** 3).mean())),
                ex_kurt=float((z ** 4).mean() - 3.0), hurst=hurst_rs(r))


def run_simulation(model, cfg, types, dts_n, states, tods_abs, mid, i2,
                   dt_mean, device, args, out_prefix, sizes_raw=None,
                   sizes_z=None):
    """Post-training stage: R thinning simulations seeded from real history,
    event-frequency comparison, midprice paths, stylized-fact table."""
    print("\n" + "=" * 68)
    print(f"[sim ] Ogata thinning: {args.n_sims} runs x {args.simulate} events "
          f"(seed prefix: last {cfg.seq_len} events before the "
          f"{'test' if args.sim_from == 'test' else 'train'} boundary)")
    anchor = i2 if args.sim_from == "test" else int(0.6 * len(types))
    s0, s1 = anchor - cfg.seq_len, anchor
    seed_t, seed_d, seed_s = types[s0:s1], dts_n[s0:s1], states[s0:s1]
    seed_tod = tods_abs[s0:s1]
    seed_z = None if sizes_z is None else sizes_z[s0:s1]
    real_sizes = None if sizes_raw is None else sizes_raw[s1:s1 + args.simulate]
    v0 = mid[s1 - 1] / 10000.0                     # LOBSTER price*1e4 -> $
    jump_dists = empirical_jump_dists(types[:s1], mid[:s1] / 10000.0)

    real_slice = types[s1:s1 + args.simulate]
    real_counts = np.bincount(real_slice, minlength=N_EVENT_TYPES)
    real_mid = mid[s1:s1 + args.simulate] / 10000.0

    all_counts = np.zeros(N_EVENT_TYPES)
    sim_stats, sim_paths, sim_dt_means, sim_sizes = [], [], [], []
    np_rng = np.random.default_rng(cfg.seed)
    for r in range(args.n_sims):
        t0 = time.time()
        g = torch.Generator(device="cpu"); g.manual_seed(cfg.seed + r)
        st, sd, sz = simulate_events(model, seed_t, seed_d, seed_s, seed_tod,
                                     args.simulate, device, args.sim_state,
                                     rng=g, seed_sizes_z=seed_z)
        sim_sizes.append(sz)
        all_counts += np.bincount(st, minlength=N_EVENT_TYPES)
        path = build_midprice(st, v0, jump_dists, np_rng)
        sim_paths.append(path)
        sim_stats.append(stylized_stats(path))
        sim_dt_means.append(sd.mean() * dt_mean)
        print(f"[sim ] run {r + 1}/{args.n_sims}: {args.simulate} events in "
              f"{time.time() - t0:.1f}s   mean gap "
              f"{sd.mean() * dt_mean * 1000:.2f} ms (real "
              f"{dts_n[s1:s1 + args.simulate].mean() * dt_mean * 1000:.2f} ms)")

    print("\n[sim ] event-type frequency: simulated vs real "
          f"(next {args.simulate} real events)")
    sim_p = all_counts / all_counts.sum()
    real_p = real_counts / max(real_counts.sum(), 1)
    for k in range(N_EVENT_TYPES):
        print(f"         {EVENT_NAMES[k]:>4s}:  sim {sim_p[k]:7.4f}   "
              f"real {real_p[k]:7.4f}")

    rs = stylized_stats(real_mid)
    keys = ["vol", "abs_skew", "ex_kurt", "hurst"]
    mean_sim = {k: float(np.nanmean([s[k] for s in sim_stats])) for k in keys}
    print("\n[sim ] stylized facts of the midprice (log returns), "
          "real vs simulated (mean over runs):")
    label = {"vol": "Volatility", "abs_skew": "Abs. skewness",
             "ex_kurt": "Excess kurtosis", "hurst": "Hurst exponent"}
    for k in keys:
        print(f"         {label[k]:>16s}:  real {rs[k]:10.6f}   "
              f"sim {mean_sim[k]:10.6f}")

    size_summary = {}
    if model.size_mark and real_sizes is not None:
        allsz = np.concatenate(sim_sizes)
        allsz = allsz[np.isfinite(allsz)]
        size_summary = {
            "real_mean": float(real_sizes.mean()),
            "real_median": float(np.median(real_sizes)),
            "real_p90": float(np.percentile(real_sizes, 90)),
            "sim_mean": float(allsz.mean()),
            "sim_median": float(np.median(allsz)),
            "sim_p90": float(np.percentile(allsz, 90)),
        }
        if model.size_atoms > 0:
            atoms = set(float(a) for a in
                        model.atom_sizes.cpu().numpy().tolist())
            size_summary["atoms"] = sorted(atoms)
            size_summary["real_atom_hit"] = float(np.isin(
                real_sizes.astype(float), list(atoms)).mean())
            size_summary["sim_atom_hit"] = float(np.isin(
                allsz.astype(float), list(atoms)).mean())
        print("\n[sim ] order-size mark, real vs simulated "
              f"(next {args.simulate} real events):")
        print(f"         {'mean':>8s}: real {size_summary['real_mean']:9.1f}"
              f"   sim {size_summary['sim_mean']:9.1f}")
        print(f"         {'median':>8s}: real {size_summary['real_median']:9.1f}"
              f"   sim {size_summary['sim_median']:9.1f}")
        print(f"         {'p90':>8s}: real {size_summary['real_p90']:9.1f}"
              f"   sim {size_summary['sim_p90']:9.1f}")
        if model.size_atoms > 0:
            print(f"         {'atom hit':>8s}: real "
                  f"{size_summary['real_atom_hit']:9.4f}"
                  f"   sim {size_summary['sim_atom_hit']:9.4f}"
                  f"   (top-{model.size_atoms} round lots)")

    np.savez(f"{out_prefix}_sim.npz",
             seed_end_index=s1,
             sim_paths=np.array(sim_paths),
             real_mid=real_mid,
             sim_counts=all_counts, real_counts=real_counts,
             sim_sizes=np.array(sim_sizes),
             real_sizes=(real_sizes if real_sizes is not None
                         else np.array([])))
    summary = {"n_sims": args.n_sims, "n_events": args.simulate,
               "sim_freq": sim_p.tolist(), "real_freq": real_p.tolist(),
               "stylized_real": rs, "stylized_sim_mean": mean_sim,
               "mean_gap_ms_sim": float(np.mean(sim_dt_means) * 1000),
               "size_mark": size_summary}
    with open(f"{out_prefix}_sim.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] paths -> {out_prefix}_sim.npz   summary -> {out_prefix}_sim.json")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            for pth in sim_paths:
                ax[0].plot(pth, lw=0.7, alpha=0.7)
            ax[0].set_title("Simulated midprice paths"); ax[0].set_xlabel("event #")
            ax[1].plot(real_mid, lw=0.8, color="k")
            ax[1].set_title("Real midprice (same horizon)"); ax[1].set_xlabel("event #")
            fig.tight_layout()
            fig.savefig(f"{out_prefix}_sim_paths.png", dpi=130)
            print(f"[save] plot -> {out_prefix}_sim_paths.png")
        except ImportError:
            print("[sim ] matplotlib not installed; skipping --plots")


def main():
    p = argparse.ArgumentParser(
        description="Mamba Hawkes Process on LOBSTER 12-event LOB data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=["mhp", "amhp"], default="mhp",
                   help="mhp = base model; amhp = Time-Scaling + Dual-Channel")
    p.add_argument("--time-scaling", action="store_true",
                   help="ablation: enable Time-Scaling only (on top of MHP)")
    p.add_argument("--dual-channel", action="store_true",
                   help="ablation: enable Dual-Channel only (on top of MHP)")
    p.add_argument("--no-market-state", action="store_true",
                   help="disable the queue-imbalance market-state embedding")
    p.add_argument("--ticker", default=None,
                   help="prefix filter if several tickers' files are present")
    p.add_argument("--data-dir", default=None,
                   help="folder with LOBSTER CSVs (default: script folder)")
    p.add_argument("--list", action="store_true",
                   help="list discovered data pairs and exit")
    p.add_argument("--max-events", type=int, default=100_000,
                   help="cap on events used (0 = full file)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--stride", type=int, default=None,
                   help="window stride (default: seq-len, non-overlapping)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--n-mc", type=int, default=10)
    p.add_argument("--no-size-mark", action="store_true",
                   help="disable the v5 order-size mark (reproduces v4 "
                        "exactly; required to load pre-v5 checkpoints)")
    p.add_argument("--size-mix", type=int, default=5,
                   help="mixture components of the per-type log-size "
                        "density f_k(m|h)")
    p.add_argument("--size-atoms", type=int, default=16,
                   help="hybrid mark head: point masses on the top-N most "
                        "frequent exact train-split sizes (round lots), "
                        "with the mixture as the continuous fallback. "
                        "0 = pure mixture (v5.0 behavior)")
    p.add_argument("--size-weight", type=float, default=1.0,
                   help="weight of the mark NLL in the loss (1.0 = the "
                        "canonical marked-process likelihood)")
    p.add_argument("--backbone", choices=["mamba2", "mamba3"],
                   default="mamba3",
                   help="SSM backbone: mamba3 (exponential-trapezoidal + "
                        "complex state via RoPE trick + BCNorm, no conv; "
                        "default) or mamba2 (the v2/v3 block; use this to "
                        "load pre-v4 checkpoints)")
    p.add_argument("--baseline", choices=["const", "bathtub"],
                   default="const",
                   help="intensity baseline: const (v2 behavior) or the "
                        "Van Vliet eq.-7 bathtub time-of-day baseline with "
                        "fixed shape and learnable per-type levels")
    p.add_argument("--bathtub-ups-l", type=float, default=0.15,
                   help="left-wing decay rate, per minute (fixed, not learned)")
    p.add_argument("--bathtub-ups-r", type=float, default=0.15,
                   help="right-wing decay rate, per minute (fixed, not learned)")
    p.add_argument("--time-pred", choices=["head", "intensity"],
                   default="head",
                   help="next-time point prediction for the RMSE metric: the "
                        "linear head, or E[dt] under the learned intensity "
                        "(inference-only; applied on val/test, training loss "
                        "unchanged)")
    p.add_argument("--time-grid-max", type=float, default=50.0,
                   help="truncation of the survival integral (x mean gap)")
    p.add_argument("--time-grid-n", type=int, default=200,
                   help="grid points for the survival integral")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="auto | cuda | mps | cpu")
    p.add_argument("--simulate", type=int, default=2000,
                   help="events to simulate with Ogata thinning after "
                        "training (0 = train/eval only)")
    p.add_argument("--n-sims", type=int, default=5,
                   help="number of independent simulation runs")
    p.add_argument("--sim-from", choices=["test", "train"], default="test",
                   help="seed the simulation with real history ending at the "
                        "test boundary (default) or the train boundary")
    p.add_argument("--sim-state", choices=["balanced", "last"],
                   default="balanced",
                   help="market-state input during simulation (the book is "
                        "not simulated): fixed balanced regime, or carry the "
                        "last seed state")
    p.add_argument("--plots", action="store_true",
                   help="save midprice path PNG (requires matplotlib)")
    p.add_argument("--load", default=None,
                   help="checkpoint .pt to load; skips training and goes "
                        "straight to evaluation + simulation")
    p.add_argument("--out", default=None,
                   help="output prefix for checkpoint/metrics "
                        "(default: <model>_<ticker>)")
    args = p.parse_args()

    script_dir = args.data_dir or os.path.dirname(os.path.abspath(__file__))
    pairs = discover_data(script_dir, args.ticker)
    if args.list or not pairs:
        print(f"[data] pairs found in {script_dir}:")
        for m, _ in pairs:
            print("   ", os.path.basename(m))
        if not pairs:
            print("   (none) -- place *_message_*.csv and *_orderbook_*.csv "
                  "next to this script")
        sys.exit(0 if args.list else 1)
    if len(pairs) > 1:
        print(f"[data] {len(pairs)} pairs found; using the first "
              f"(use --ticker to select):")
        for m, _ in pairs:
            print("   ", os.path.basename(m))
    msg_path, ob_path = pairs[0]
    ticker = os.path.basename(msg_path).split("_")[0]

    cfg = Config(
        model=args.model,
        time_scaling=args.time_scaling or args.model == "amhp",
        dual_channel=args.dual_channel or args.model == "amhp",
        use_market_state=not args.no_market_state,
        max_events=args.max_events, seq_len=args.seq_len,
        stride=args.stride or args.seq_len, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, d_model=args.d_model,
        d_state=args.d_state, n_layers=args.n_layers, beta=args.beta,
        gamma=args.gamma, n_mc=args.n_mc, seed=args.seed,
        size_mark=not args.no_size_mark, size_mix=args.size_mix,
        size_atoms=(0 if args.no_size_mark else args.size_atoms),
        size_weight=args.size_weight,
        device=args.device, time_pred=args.time_pred,
        backbone=args.backbone,
        baseline=args.baseline, bathtub_ups_l=args.bathtub_ups_l,
        bathtub_ups_r=args.bathtub_ups_r,
        time_grid_max=args.time_grid_max, time_grid_n=args.time_grid_n)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    variant = ("A-MHP" if cfg.time_scaling and cfg.dual_channel else
               "MHP+TS" if cfg.time_scaling else
               "MHP+DC" if cfg.dual_channel else "MHP")
    if cfg.baseline == "bathtub":
        variant += "+BT"
    if cfg.backbone == "mamba3":
        variant += "-M3"
    if cfg.size_mark:
        variant += "+SZ"
    print(f"[cfg ] variant={variant}  device={device}  "
          f"market-state={'on' if cfg.use_market_state else 'off'}")
    print(f"[cfg ] {json.dumps(asdict(cfg))}")

    types, dts, states, tods_abs, mid, sizes_raw = load_and_label(
        msg_path, ob_path, cfg.max_events)
    tods_abs = tods_abs.astype(np.float32)

    # chronological 60/20/20 split 
    n = len(types)
    i1, i2 = int(0.6 * n), int(0.8 * n)
    dt_mean = float(dts[:i1].mean())          # normalize by TRAIN mean gap
    dts_n = (dts / dt_mean).astype(np.float32)
    # mark standardization: TRAIN-split mean/std of log(size)
    logm = np.log(sizes_raw)
    logm_mean = float(logm[:i1].mean())
    logm_std = float(max(logm[:i1].std(), 1e-6))
    sizes_z = ((logm - logm_mean) / logm_std).astype(np.float32)
    # v5.1 atoms: top-N most frequent exact sizes in the TRAIN split
    if cfg.size_mark and cfg.size_atoms > 0:
        vals, cnts = np.unique(sizes_raw[:i1], return_counts=True)
        if len(vals) < cfg.size_atoms:
            cfg.size_atoms = len(vals)
        top = vals[np.argsort(cnts)[::-1][:cfg.size_atoms]]
        atoms = np.sort(top)
        lut = {float(a): i for i, a in enumerate(atoms)}
        atom_idx = np.array([lut.get(float(m), -1) for m in sizes_raw],
                            dtype=np.int64)
        cov = (atom_idx[:i1] >= 0).mean()
        # train marginal over [atoms..., continuous] for atom-head init
        atom_marg = np.array(
            [(atom_idx[:i1] == i).mean() for i in range(cfg.size_atoms)]
            + [(atom_idx[:i1] < 0).mean()])
    else:
        atoms = np.array([])
        atom_idx = np.full(len(sizes_raw), -1, dtype=np.int64)
    print(f"[data] split: train {i1:,} / val {i2 - i1:,} / test {n - i2:,}"
          f"   mean train gap = {dt_mean * 1000:.3f} ms")
    if cfg.size_mark:
        print(f"[data] size mark: log-size mean {logm_mean:.3f} "
              f"std {logm_std:.3f} (train split)")
        if cfg.size_atoms > 0:
            head = ", ".join(f"{a:,.0f}" for a in atoms[:8])
            print(f"[data] size atoms: top-{cfg.size_atoms} exact sizes "
                  f"cover {cov:.1%} of train events "
                  f"({head}{', ...' if len(atoms) > 8 else ''})")
    maj = np.bincount(types[:i1], minlength=N_EVENT_TYPES).argmax()
    maj_acc = (types[i2:] == maj).mean()
    print(f"[data] majority-class baseline on test "
          f"({EVENT_NAMES[maj]}): acc = {maj_acc:.4f}")

    def make(a, b, stride):
        return LOBWindowDataset(types[a:b], dts_n[a:b], states[a:b],
                                tods_abs[a:b], sizes_z[a:b], atom_idx[a:b],
                                cfg.seq_len, stride)

    train_ds = make(0, i1, cfg.stride)
    val_ds = make(i1, i2, cfg.seq_len)
    test_ds = make(i2, n, cfg.seq_len)
    train_ld = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, cfg.batch_size)
    test_ld = DataLoader(test_ds, cfg.batch_size)
    print(f"[data] windows: train {len(train_ds)} / val {len(val_ds)} / "
          f"test {len(test_ds)}  (seq_len={cfg.seq_len})")

    model = MambaHawkesProcess(
        d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers,
        expand=cfg.expand, d_conv=cfg.d_conv, time_scaling=cfg.time_scaling,
        dual_channel=cfg.dual_channel,
        use_market_state=cfg.use_market_state, baseline=cfg.baseline,
        bathtub_ups=(cfg.bathtub_ups_l, cfg.bathtub_ups_r),
        backbone=cfg.backbone, size_mark=cfg.size_mark,
        size_mix=cfg.size_mix, size_atoms=cfg.size_atoms).to(device)
    model.set_time_context(dt_mean)
    model.set_size_context(logm_mean, logm_std)
    if cfg.size_mark and cfg.size_atoms > 0:
        model.set_size_atoms(atoms, atom_marg)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[model] parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    out = args.out or f"{cfg.model}_{ticker}"
    best_val, best_state = float("inf"), None
    history = []

    if args.load:
        ck = torch.load(args.load, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["state_dict"])
        except RuntimeError as e:
            if cfg.size_mark and "size_proj" in str(e):
                sys.exit("[load] checkpoint has no size-mark modules - it "
                         "is pre-v5. Re-run with --no-size-mark to load it.")
            if "atom" in str(e):
                sys.exit("[load] atom-head mismatch: the checkpoint was "
                         "trained with a different --size-atoms (or none). "
                         "Pass the same --size-atoms it was trained with.")
            raise
        if "logm_mean" in ck:
            model.set_size_context(ck["logm_mean"], ck["logm_std"])
        print(f"[load] restored weights from {args.load} "
              f"(trained dt_mean {ck.get('dt_mean', dt_mean) * 1000:.3f} ms) "
              f"- skipping training")
        cfg_epochs_range = []
    else:
        cfg_epochs_range = range(1, cfg.epochs + 1)

    for ep in cfg_epochs_range:
        te = time.time()
        print(f"[train] epoch {ep}/{cfg.epochs}")
        tr = run_epoch(model, train_ld, cfg, device, opt, dt_mean)
        va = run_epoch(model, val_ld, cfg, device, None, dt_mean)
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"[epoch {ep:>2d}] "
              f"train: loss {tr['loss']:.4f} LL/ev {tr['ll_per_event']:.4f} "
              f"acc {tr['acc']:.4f} rmse {tr['rmse_sec']:.4f}s | "
              f"val: loss {va['loss']:.4f} LL/ev {va['ll_per_event']:.4f} "
              f"acc {va['acc']:.4f} rmse {va['rmse_sec']:.4f}s "
              f"({time.time() - te:.1f}s)")
        if va["loss"] < best_val:
            best_val = va["loss"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print(f"          * new best val loss ({best_val:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    ts = run_epoch(model, test_ld, cfg, device, None, dt_mean)
    print("\n" + "=" * 68)
    print(f"[test] {variant} on {ticker}  "
          f"(best-val checkpoint)")
    print(f"[test] log-likelihood / event : {ts['ll_per_event']:.4f}  "
          f"(normalized-time units)")
    print(f"[test] next-type accuracy     : {ts['acc']:.4f}  "
          f"(majority baseline {maj_acc:.4f})")
    print(f"[test] next-time RMSE ({cfg.time_pred:9s}): {ts['rmse_norm']:.4f} "
          f"(x mean gap)  =  {ts['rmse_sec']:.4f} s")
    if cfg.size_mark:
        print(f"[test] size NLL / event       : {ts['size_nll']:.4f}  "
              f"(log-share units)   marked LL/ev: "
              f"{ts['ll_marked_per_event']:.4f}")
    print("=" * 68)
    confusion_report(model, test_ld, device)

    if args.simulate > 0:
        run_simulation(model, cfg, types, dts_n, states, tods_abs, mid, i2,
                       dt_mean, device, args, out,
                       sizes_raw=sizes_raw, sizes_z=sizes_z)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                "dt_mean": dt_mean, "logm_mean": logm_mean,
                "logm_std": logm_std, "ticker": ticker}, f"{out}.pt")
    with open(f"{out}_metrics.json", "w") as f:
        json.dump({"config": asdict(cfg), "ticker": ticker,
                   "majority_baseline_acc": float(maj_acc),
                   "test": ts, "history": history}, f, indent=2)
    print(f"[save] checkpoint -> {out}.pt   metrics -> {out}_metrics.json")


if __name__ == "__main__":
    main()
