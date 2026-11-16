# ---- notebook cell 23: Problem 2 functions: action, sampler, exact results ----
DTAU = 0.1
BETA_2A = 1.0
N_SLICES = int(round(BETA_2A / DTAU))     # 10


def action(x, dtau):
    """Full discretized action S_E of the closed path x (eq. 4, V = x^2 / 2), with x_N = x_0.
    Written independently of the sampler so it can test it."""
    return float(np.sum((np.roll(x, -1) - x) ** 2 / (2 * dtau) + dtau * 0.5 * x**2))


@njit
def delta_action(x, i, x_new, dtau):
    """Change in S_E when only slice i moves from x[i] to x_new (Lectures 18-19), periodic neighbors:
    (x_new - x_i)(x_new + x_i - x_{i-1} - x_{i+1}) / dtau + dtau (V(x_new) - V(x_i)).
    Only reports the change; does not move anything."""
    n = x.shape[0]
    left, right = x[(i - 1) % n], x[(i + 1) % n]
    kinetic = (x_new - x[i]) * (x_new + x[i] - left - right) / dtau
    potential = dtau * 0.5 * (x_new**2 - x[i] ** 2)
    return kinetic + potential


@njit
def pimc_sweeps(x, dtau, h, n_sweeps, rng):
    """n_sweeps sweeps in place on the closed path x (length N). One sweep visits every slice once,
    in order (Lab 14), proposing x_i -> x_i + U[-h, h]; accept with min(1, exp(-dS)). The proposal is
    symmetric, so no Hastings factor appears (Lab 10, Part 2).
    Returns f = mean(x^2) after every sweep (length n_sweeps; rejected updates count) and the acceptance."""
    n = x.shape[0]
    f = np.empty(n_sweeps)
    accepted = 0
    for sweep in range(n_sweeps):
        for i in range(n):
            x_new = x[i] + rng.uniform(-h, h)
            dS = delta_action(x, i, x_new, dtau)
            if dS <= 0.0 or rng.random() < np.exp(-dS):
                x[i] = x_new
                accepted += 1
        f[sweep] = np.mean(x**2)
    return f, accepted / (n_sweeps * n)


def x2_exact(beta):
    """Continuum result 1/2 coth(beta/2)."""
    return 0.5 / np.tanh(beta / 2)


def x2_exact_discrete(beta, dtau):
    """Exact <x^2> of the discretized action (Lab 14, slide 22; same as Creutz-Freedman, Lecture 19):
    the mean over normal modes k of 1 / (dtau + (4/dtau) sin^2(pi k / N))."""
    n = int(round(beta / dtau))
    k = np.arange(n)
    return float(np.mean(1 / (dtau + 4 / dtau * np.sin(np.pi * k / n) ** 2)))


# ---- notebook cell 25: Validation of the path-integral sampler ----
# 1. Local action change vs the independent full action, every slice (ring edges included)
rng = rng_for(970)
x = rng.normal(size=N_SLICES)
worst = 0.0
for i in range(N_SLICES):
    x_new = x[i] + rng.uniform(-1, 1)
    x_moved = x.copy()
    x_moved[i] = x_new
    worst = max(worst, abs((action(x_moved, DTAU) - action(x, DTAU)) - delta_action(x, i, x_new, DTAU)))
check("delta_action = S(x') - S(x) for every slice", worst < 1e-12, f"max difference {worst:.1e}")
lab14_path = 0.8 * np.sin(2 * np.pi * np.arange(30) / 30)        # Lab 14, slide 14: dt = 1, slice 15 moved by +0.65
check("Lab 14's check value: delta_action = 0.63375", np.isclose(delta_action(lab14_path, 15, lab14_path[15] + 0.65, 1.0), 0.63375))
d = 0.37
check("uniform shift changes only the potential: S(x + d) - S(x) = beta (d xbar + d^2/2)  [2A(ii)]",
      np.isclose(action(x + d, DTAU) - action(x, DTAU), BETA_2A * (d * x.mean() + d**2 / 2), atol=1e-12))

# 2. Reference values and their limits
check("x2_exact(1) = 1/2 coth(1/2) = 1.081977", np.isclose(x2_exact(1.0), 1.0819767, atol=1e-6))
check("x2_exact_discrete(1, 0.1) = 1.081010 (-0.089 %)", np.isclose(x2_exact_discrete(1.0, 0.1), 1.0810096, atol=1e-6))
check("discrete -> continuum as dtau -> 0", np.isclose(x2_exact_discrete(1.0, 0.001), x2_exact(1.0), rtol=1e-5))
check("beta -> 0: classical 1/beta", np.isclose(0.01 * x2_exact(0.01), 1.0, rtol=1e-4))

# 3. Bookkeeping and acceptance limits
f, acc = pimc_sweeps(np.zeros(N_SLICES), DTAU, 0.6, 100, rng_for(971))
check("one f per sweep", len(f) == 100, f"len {len(f)}")
_, acc_small = pimc_sweeps(np.zeros(N_SLICES), DTAU, 1e-4, 200, rng_for(972))
_, acc_large = pimc_sweeps(np.zeros(N_SLICES), DTAU, 100.0, 200, rng_for(973))
check("acceptance -> 1 as h -> 0 and -> 0 as h -> infinity", acc_small > 0.99 and acc_large < 0.02,
      f"{acc_small:.3f}, {acc_large:.3f}")

# 4. Known answer: <x^2> at beta = 1 from 8 independent chains vs the discretized exact value
exact = x2_exact_discrete(BETA_2A, DTAU)
means = []
for c in range(8):
    f, _ = pimc_sweeps(np.zeros(N_SLICES), DTAU, 0.63, 42_000, rng_for(980 + c))
    means.append(f[2_000:].mean())
mc, err = np.mean(means), np.std(means, ddof=1) / np.sqrt(8)
check(f"<x^2> at beta = 1 from 8 chains: exact (dtau = 0.1) {exact:.4f}", abs(mc - exact) < 4 * err,
      f"MCMC {mc:.4f} +- {err:.4f}")


# ---- notebook cell 26: Problem 2A(i): scan of the hit size h ----
H_SCAN = np.round(np.arange(0.3, 1.0 + 1e-9, 0.1), 1)
scan_acc = np.array([pimc_sweeps(np.zeros(N_SLICES), DTAU, h, 20_000, rng_for(210 + j))[1]
                     for j, h in enumerate(H_SCAN)])
H_2A = float(H_SCAN[np.argmin(np.abs(scan_acc - 0.5))])          # closest to 50 %, inside the 40-60 % window
for h, a in zip(H_SCAN, scan_acc):
    print(f"h = {h:.1f}: acceptance {a:.3f}" + ("   <- chosen" if h == H_2A else ""))

fig, ax = plt.subplots(figsize=(5.5, 3.3))
ax.axhspan(0.4, 0.6, color="C2", alpha=0.15, label="40-60 % window")
ax.plot(H_SCAN, scan_acc, "o-")
ax.axvline(2 * np.sqrt(DTAU), color="gray", ls=":", label=r"$2\sqrt{\delta\tau}$ (Creutz-Freedman)")
ax.set(xlabel="hit size h", ylabel="acceptance", title=r"2A: acceptance vs h ($\beta$ = 1, $\delta\tau$ = 0.1)")
ax.legend(fontsize=9)
save(fig, "2A_h_scan")
plt.show()


# ---- notebook cell 27: Problem 2A(i): production run and error analysis ----
SWEEPS_2A = 1_000_000          # kept sweeps (>= 50,000 required; >= 50 tau_int needed for a reliable tau_int)
START_2A = 50_000              # first stretch of the same chain, long enough to choose the burn-in
rng = rng_for(200)
x = np.zeros(N_SLICES)         # cold start
f_start, acc_start = pimc_sweeps(x, DTAU, H_2A, START_2A, rng)
BURNIN_2A, t_eq_2A, tau_start = burn_in(f_start)
assert BURNIN_2A < START_2A, "burn-in longer than the starting stretch"
n_more = BURNIN_2A + SWEEPS_2A - START_2A
f_more, acc_more = pimc_sweeps(x, DTAU, H_2A, n_more, rng)       # the same chain continues from x
f_all = np.concatenate([f_start, f_more])
f_kept = f_all[BURNIN_2A:]
acceptance_2A = (acc_start * START_2A + acc_more * n_more) / (START_2A + n_more)
print(f"h = {H_2A}, acceptance {acceptance_2A:.3f}; f first reaches its equilibrium mean at sweep {t_eq_2A}; "
      f"tau_int = {tau_start:.0f} -> burn-in = t_eq + 10 tau_int = {BURNIN_2A:,} sweeps; kept {len(f_kept):,} sweeps")

tau_single, M_single = tau_int(f_kept, return_window=True)
x2_single = float(f_kept.mean())
x2_err_single, B_single, enough = blocking_error(f_kept, tau_single)
x2_err_naive = float(naive_error(f_kept))
n_single = len(f_kept)
exact_c, exact_d = x2_exact(BETA_2A), x2_exact_discrete(BETA_2A, DTAU)
print(f"<x^2> = {x2_single:.4f} +- {x2_err_single:.4f} (blocking, B* = {B_single:,})   naive +- {x2_err_naive:.5f}")
print(f"tau_int = {tau_single:.0f} sweeps (window M = {M_single:,}); (blocked/naive)^2 = "
      f"{(x2_err_single / x2_err_naive) ** 2:.0f}; N/tau_int = {n_single / tau_single:,.0f}"
      + ("" if enough else "   [too short for B* >= 20 tau]"))
print(f"exact, dtau = 0.1: {exact_d:.5f} -> difference {(x2_single - exact_d) / x2_err_single:+.2f} sigma")
print(f"exact, dtau -> 0:  {exact_c:.5f} -> difference {(x2_single - exact_c) / x2_err_single:+.2f} sigma")
print(f"2A(ii) estimate 2/dtau^2 = {2 / DTAU**2:.0f} sweeps vs measured tau_int = {tau_single:.0f} sweeps")


# ---- notebook cell 28: Problem 2A(i): plots ----
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6))
first = 3 * BURNIN_2A
ax0.plot(np.arange(first), f_all[:first], lw=0.5, color="C0")
ax0.axvline(BURNIN_2A, color="C3", ls="--", label=f"burn-in = {BURNIN_2A:,} sweeps")
ax0.axhline(exact_d, color="k", ls=":", label=r"exact $\langle x^2\rangle$, $\delta\tau$ = 0.1")
ax0.set(xlim=(0, first), xlabel="sweep", ylabel=r"$\frac{1}{N}\sum_i x_i^2$", title="2A: start of the chain (cold start)")
ax0.legend(fontsize=9)
ax1.plot(np.arange(n_single), f_kept, lw=0.2, color="C0")
ax1.axhline(exact_d, color="k", ls=":")
ax1.set(xlim=(0, n_single), xlabel="kept sweep", ylabel=r"$\frac{1}{N}\sum_i x_i^2$",
        title=f"kept sweeps: mean {x2_single:.4f} $\\pm$ {x2_err_single:.4f}")
save(fig, "2A_trace")
plt.show()

fig, (ax_rho, ax_blk) = plt.subplots(1, 2, figsize=(10, 3.8))
lags = np.arange(int(2 * M_single) + 1)
ax_rho.plot(lags, autocorrelation(f_kept, lags[-1]), color="C0")
ax_rho.axhline(0, color="gray", lw=0.8)
ax_rho.axvline(M_single, color="C3", ls="--", label=f"window M = {M_single:,}")
ax_rho.set(xlabel="lag t (sweeps)", ylabel=r"$\hat\rho(t)$", title=f"autocorrelation: $\\tau_{{int}}$ = {tau_single:.0f} sweeps")
ax_rho.legend(fontsize=9)
plot_blocking(ax_blk, f_kept, tau_single, color="C0")
ax_blk.axhline(x2_err_naive, color="gray", ls=":", label="naive error (B = 1)")
ax_blk.set_title(f"blocking: plateau {x2_err_single:.4f} at $B^*$ = {B_single:,}")
ax_blk.legend(fontsize=9)
save(fig, "2A_autocorr_blocking")
plt.show()
