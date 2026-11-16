# ---- notebook cell 3: Setup: imports, seeds, shared error-analysis tools ----
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

plt.rcParams.update({"font.size": 11, "figure.dpi": 100, "figure.constrained_layout.use": True})

TSN = "200288141"
SEED = int(TSN)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def rng_for(k):
    """Random generator for my k-th run (seed = SEED + k, see the table above)."""
    return np.random.default_rng(SEED + k)


def save(fig, name):
    """Save a figure for the report as figures/<name>.png."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


VALIDATION = []                                         # (test name, passed?) for the summary at the end


def check(name, ok, detail=""):
    """Record and print one validation test. Prints instead of asserting, so Run All always finishes."""
    VALIDATION.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


# ---------- shared error-analysis tools (Lecture 16, Lab 12) ----------
def autocorrelation(f, max_lag):
    """Normalized autocorrelation rho(t), t = 0..max_lag, with Lecture 16's estimator
    c(t) = (1/N) sum_{i=1}^{N-t} (f_i - fbar)(f_{i+t} - fbar),  rho(t) = c(t) / c(0).
    The sums are done with an FFT, zero-padded to 2N so nothing wraps around: same numbers, much faster."""
    d = np.asarray(f, dtype=float) - np.mean(f)
    n = d.size
    spectrum = np.fft.rfft(d, 2 * n)
    c = np.fft.irfft(spectrum * np.conj(spectrum))[:max_lag + 1] / n
    return c / c[0]


def tau_int(f, return_window=False):
    """Integrated autocorrelation time, Lecture 16 convention: tau(M) = 1 + 2 sum_{t=1}^{M} rho(t),
    with the smallest window M such that M >= 5 tau(M). (Lab 12 calls this g and uses g/2.)"""
    rho = autocorrelation(f, len(f) - 1)
    windows = np.arange(1, len(rho))
    tau_of_M = 1 + 2 * np.cumsum(rho[1:])
    ok = windows >= 5 * tau_of_M
    M = int(windows[np.argmax(ok)]) if ok.any() else int(windows[-1])
    if not ok.any():
        print("  warning: no window with M >= 5 tau; the chain is too short")
    tau = float(tau_of_M[M - 1])
    return (tau, M) if return_window else tau


def naive_error(f):
    """Error on the mean if every sample were independent: s / sqrt(N)."""
    return np.std(f, ddof=1) / np.sqrt(len(f))


def blocking_curve(f, block_lengths):
    """Blocking error for each block length B (Lecture 16): average consecutive blocks of B samples
    (leftover samples at the end are dropped), then sqrt( sum_b (F_b - Fbar)^2 / (N_B (N_B - 1)) )."""
    f = np.asarray(f, dtype=float)
    errors = np.empty(len(block_lengths))
    for k, B in enumerate(block_lengths):
        n_blocks = len(f) // B
        means = f[:n_blocks * B].reshape(n_blocks, B).mean(axis=1)
        errors[k] = np.sqrt(np.sum((means - means.mean()) ** 2) / (n_blocks * (n_blocks - 1)))
    return errors


MIN_BLOCKS = 32          # keep enough blocks for their spread to be meaningful (Lab 12, Lab 14)


def block_lengths_for(n):
    """B = 1, 2, 4, ... while at least MIN_BLOCKS blocks remain."""
    return 2 ** np.arange(int(np.log2(n // MIN_BLOCKS)) + 1)


def blocking_error(f, tau):
    """Plateau value of the blocking curve, read at B* = smallest power of 2 with B* >= 20 tau (B >> tau),
    keeping at least MIN_BLOCKS blocks. Returns (error, B*, enough); enough = False means the chain is
    too short to reach B* >= 20 tau, and the largest allowed B is used instead."""
    lengths = block_lengths_for(len(f))
    enough = bool(lengths[-1] >= 20 * tau)
    B = int(lengths[np.argmax(lengths >= 20 * tau)]) if enough else int(lengths[-1])
    return float(blocking_curve(f, [B])[0]), B, enough


def burn_in(trace, tau_factor=10):
    """Burn-in rule (Lectures 14 and 16): the transient is over when the trace first reaches its
    equilibrium mean (the mean of the second half of the run), coming from the side it started on.
    After that I wait tau_factor * tau_int more sweeps. Returns (burn_in, t_eq, tau_int)."""
    trace = np.asarray(trace, dtype=float)
    target = trace[len(trace) // 2:].mean()
    side = np.sign(trace[0] - target)
    reached = np.flatnonzero(np.sign(trace - target) != side)
    t_eq = int(reached[0]) if reached.size else len(trace)
    tau = tau_int(trace[len(trace) // 2:])
    return int(t_eq + np.ceil(tau_factor * tau)), t_eq, tau


def plot_blocking(ax, f, tau, label=None, color=None):
    """Blocking curve for B = 1, 2, 4, ... (at least 32 blocks); the plateau point B* is circled."""
    lengths = block_lengths_for(len(f))
    ax.semilogx(lengths, blocking_curve(f, lengths), "o-", base=2, ms=4, color=color, label=label)
    err, B, _ = blocking_error(f, tau)
    ax.plot(B, err, "o", ms=12, mfc="none", color="k")
    ax.set(xlabel="block length B (sweeps)", ylabel="blocking error of the mean")


# ---- notebook cell 5: Validation of the error-analysis tools ----
def ar1(n, r, rng):
    """Lab 12's stationary AR(1) toy: known rho(t) = r^t, tau_int = (1 + r) / (1 - r)."""
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = r * x[t - 1] + np.sqrt(1 - r**2) * eps[t]
    return x


def ar1_exact_error(n, r):
    k = np.arange(1, n)
    return np.sqrt((1 + 2 * np.sum((1 - k / n) * r**k)) / n)


N_TOY, R_TOY = 2**17, 0.9
x_toy = ar1(N_TOY, R_TOY, rng_for(900))
x_white = ar1(N_TOY, 0.0, rng_for(901))

tau_toy = tau_int(x_toy)
check("tau_int on AR(1), r = 0.9: exact 19", abs(tau_toy - 19) < 0.2 * 19, f"got {tau_toy:.2f}")
check("tau_int on white noise: exact 1", abs(tau_int(x_white) - 1) < 0.3, f"got {tau_int(x_white):.2f}")

naive = naive_error(x_toy)
check("naive_error = s / sqrt(N)", np.isclose(naive, np.std(x_toy) / np.sqrt(N_TOY), rtol=1e-3),
      f"got {naive:.5f}")
B_TOY = 2 ** np.arange(0, 11)
curve = blocking_curve(x_toy, B_TOY)
check("blocking at B = 1 equals the naive error", np.isclose(curve[0], naive, rtol=1e-3),
      f"{curve[0]:.5f} vs {naive:.5f}")
exact_err = ar1_exact_error(N_TOY, R_TOY)
check("blocking plateau (B = 1024) = exact error of the mean", abs(curve[-1] / exact_err - 1) < 0.25,
      f"{curve[-1]:.5f} vs exact {exact_err:.5f}")
check("blocked / naive ~ sqrt(tau_int)", abs((curve[-1] / naive) ** 2 / 19 - 1) < 0.5,
      f"(blocked/naive)^2 = {(curve[-1] / naive) ** 2:.1f}")
