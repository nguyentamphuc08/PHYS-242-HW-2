# ---- notebook cell 9: Problem 1 functions: Ising energy and Metropolis sampler ----
L = 20
T_C = 2 / np.log(1 + np.sqrt(2))                  # Onsager, ~2.269


def ising_energy(spins):
    """Total energy -J sum_<ij> s_i s_j, each bond once (up and left neighbors, periodic).
    Written independently of the sampler so it can test it. Valid for lattice side >= 3."""
    return -float(np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1))))


@njit
def flip_energy(spins, i, j):
    """Energy change if spin (i, j) were flipped: 2 J s_ij * (sum of its 4 periodic neighbors), J = 1.
    Only reports the cost; does not flip anything."""
    n = spins.shape[0]
    neighbors = (spins[(i - 1) % n, j] + spins[(i + 1) % n, j]
                 + spins[i, (j - 1) % n] + spins[i, (j + 1) % n])
    return 2.0 * spins[i, j] * neighbors


@njit
def ising_sweeps(spins, T, n_sweeps, rng):
    """n_sweeps Metropolis sweeps (L*L attempted flips each) in place.
    Returns m after every sweep (length n_sweeps + 1, m[0] = start) and the acceptance rate.
    Uses flip_energy. A rejected flip must leave the lattice unchanged (Lab 10, Part 3)."""
    n = spins.shape[0]
    w4, w8 = np.exp(-4.0 / T), np.exp(-8.0 / T)   # the only two acceptance probabilities below 1 (Lecture 17)
    m = np.empty(n_sweeps + 1)
    m[0] = spins.mean()
    accepted = 0
    for sweep in range(1, n_sweeps + 1):
        for _ in range(n * n):
            i = rng.integers(0, n)                # site chosen uniformly: symmetric proposal
            j = rng.integers(0, n)
            dE = flip_energy(spins, i, j)
            if dE <= 0.0 or rng.random() < (w4 if dE == 4.0 else w8):
                spins[i, j] = -spins[i, j]
                accepted += 1
            # rejected: the lattice stays as it is, and the attempt still counts
        m[sweep] = spins.mean()
    return m, accepted / (n_sweeps * n * n)


def initial_spins(kind, rng):
    """'up' -> all +1;  'random' -> +1/-1 with probability 1/2 each."""
    if kind == "up":
        return np.ones((L, L), dtype=np.int64)
    return rng.choice(np.array([-1, 1]), size=(L, L)).astype(np.int64)


def yang_m0(T):
    """Spontaneous magnetization of the infinite lattice (Lecture 17): (1 - sinh(2/T)^-4)^(1/8) below T_c, 0 above."""
    T = np.asarray(T, dtype=float)
    inside = np.clip(1 - np.sinh(2 / T) ** -4.0, 0, None)
    return np.where(T < T_C, inside ** 0.125, 0.0)


# ---- notebook cell 11: Validation of the Ising sampler ----
def exact_ising(n, T):
    """Exact <|m|> on an n x n periodic lattice: Boltzmann-weighted sum over all 2^(n*n) configurations."""
    configs = np.array(list(itertools.product([-1, 1], repeat=n * n))).reshape(-1, n, n)
    E = np.array([ising_energy(s) for s in configs])
    weights = np.exp(-(E - E.min()) / T)             # shift by E.min() to avoid overflow at low T
    return np.sum(weights * np.abs(configs.mean(axis=(1, 2)))) / weights.sum()


# 0. The full energy reproduces lecture's example configuration (Lecture 14, Lab 11): E = -152 J
example = np.ones((10, 10), dtype=np.int64)
example[:5, :] = -1
example[1, 3] = 1
check("ising_energy of lecture's 10x10 example = -152", ising_energy(example) == -152.0, f"got {ising_energy(example)}")

# 1. Local energy change vs the independent total energy, at every site (edges included)
s = initial_spins("random", rng_for(910))
E_before, worst, values = ising_energy(s), 0.0, set()
for i in range(L):
    for j in range(L):
        dE = flip_energy(s, i, j)
        s[i, j] *= -1
        worst = max(worst, abs((ising_energy(s) - E_before) - dE))
        s[i, j] *= -1
        values.add(float(dE))
check("flip_energy = E(after) - E(before) at all 400 sites", worst == 0, f"max difference {worst}")
check("dE only takes the values -8, -4, 0, 4, 8", values <= {-8.0, -4.0, 0.0, 4.0, 8.0}, f"saw {sorted(values)}")

# 2. Known answer: 3 x 3 lattice vs exact enumeration, 8 independent chains per temperature
N_CHAINS_V, SWEEPS_V, BURN_V = 8, 20_000, 1_000
for iT, T in enumerate([1.0, 2.0, 2.269, 4.0, 1000.0]):
    means = []
    for c in range(N_CHAINS_V):
        rng = rng_for(920 + 10 * iT + c)
        m, _ = ising_sweeps(rng.choice(np.array([-1, 1]), size=(3, 3)), T, BURN_V + SWEEPS_V, rng)
        means.append(np.abs(m[BURN_V + 1:]).mean())
    mc, err = np.mean(means), np.std(means, ddof=1) / np.sqrt(N_CHAINS_V)
    exact = exact_ising(3, T)
    check(f"3x3 <|m|> at T = {T:g}: exact {exact:.4f}", abs(mc - exact) < max(4 * err, 0.005),
          f"MCMC {mc:.4f} +- {err:.4f}")

# 3. Bookkeeping and limits on the 20 x 20 lattice
m, acc = ising_sweeps(initial_spins("up", rng_for(960)), 2.0, 50, rng_for(961))
check("one m per sweep plus the start, m[0] = 1", len(m) == 51 and m[0] == 1.0, f"len {len(m)}, m[0] = {m[0]}")
m, acc = ising_sweeps(initial_spins("up", rng_for(962)), 0.5, 200, rng_for(963))
check("T -> 0: all-up start stays ordered, almost nothing accepted", m.min() > 0.99 and acc < 1e-3,
      f"min m {m.min():.3f}, acceptance {acc:.1e}")
m, acc = ising_sweeps(initial_spins("up", rng_for(964)), 1e6, 200, rng_for(965))
check("T -> infinity: nearly every flip accepted, m -> 0", acc > 0.99 and abs(m[-50:]).mean() < 0.1,
      f"acceptance {acc:.4f}")


# ---- notebook cell 12: Problem 1A: runs at T = 2 from both starts, burn-in, trace plot ----
T_1A = 2.0
SWEEPS_1A = 100_000                              # per start: far beyond 50 tau_int (checked in the next cell)
runs_1A = {}
for name, k in [("all up", 0), ("random", 1)]:
    rng = rng_for(k)
    m, acc = ising_sweeps(initial_spins("up" if name == "all up" else "random", rng), T_1A, SWEEPS_1A, rng)
    burn, t_eq, tau = burn_in(np.abs(m))
    runs_1A[name] = dict(m=m, acc=acc, burn=burn, t_eq=t_eq)
    print(f"{name:>6} start: acceptance {acc:.3f}; |m| first reaches its equilibrium mean at sweep {t_eq}; "
          f"tau_int = {tau:.1f} sweeps -> burn-in = t_eq + 10 tau_int = {burn} sweeps")
BURNIN_1A = runs_1A["all up"]["burn"]           # used as the minimum burn-in in 1B (same all-up start)

zoom = max(1000, 3 * max(r["burn"] for r in runs_1A.values()))
fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5))
for ax, last in zip(axes, (zoom, SWEEPS_1A)):
    for (name, r), color in zip(runs_1A.items(), ("C0", "C3")):
        ax.plot(np.arange(last + 1), r["m"][:last + 1], lw=0.6, color=color, label=f"{name} start")
        ax.axvline(r["burn"], color=color, ls="--", lw=1)
    for sign in (1, -1):
        ax.axhline(sign * yang_m0(T_1A), color="k", ls=":", lw=1)
    ax.set(xlim=(0, last), ylim=(-1.05, 1.05), xlabel="MCMC time (sweeps of 400 attempted flips)", ylabel="m")
axes[0].set_title(f"T = {T_1A}: first {zoom} sweeps (dashed: burn-in; dotted: $\\pm m_0(T)$ = $\\pm${yang_m0(T_1A):.3f})")
axes[1].set_title(f"whole run ({SWEEPS_1A:,} sweeps)")
axes[0].legend(loc="center right")
save(fig, "1A_trace")
plt.show()


# ---- notebook cell 13: Problem 1A: autocorrelation, tau_int, blocking error ----
fig, (ax_rho, ax_blk) = plt.subplots(1, 2, figsize=(10, 3.8))
for (name, r), color in zip(runs_1A.items(), ("C0", "C3")):
    a = np.abs(r["m"][r["burn"] + 1:])                     # kept sweeps
    tau, M = tau_int(a, return_window=True)
    err, B, enough = blocking_error(a, tau)
    naive = naive_error(a)
    r.update(a=a, tau=tau, err=err)
    print(f"{name:>6} start: {len(a):,} kept sweeps; <|m|> = {a.mean():.4f} +- {err:.4f} (blocking, B* = {B}), "
          f"naive +- {naive:.4f}; tau_int = {tau:.1f} sweeps (window M = {M}); "
          f"(blocked/naive)^2 = {(err / naive) ** 2:.1f}; N/tau_int = {len(a) / tau:,.0f}"
          + ("" if enough else "   [too short for B* >= 20 tau]"))
    lags = np.arange(int(max(50, 4 * M)) + 1)
    ax_rho.plot(lags, autocorrelation(a, lags[-1]), color=color, label=f"{name}: $\\tau_{{int}}$ = {tau:.1f}")
    plot_blocking(ax_blk, a, tau, label=name, color=color)
ax_rho.axhline(0, color="gray", lw=0.8)
ax_rho.set(xlabel="lag t (sweeps)", ylabel=r"$\hat\rho(t)$ of $|m|$", title=f"autocorrelation, T = {T_1A}")
ax_rho.legend()
ax_blk.set_title("blocking (circle: plateau read at $B^*$)")
ax_blk.legend()
save(fig, "1A_autocorr_blocking")
plt.show()

print(f"\nYang m0({T_1A}) = {yang_m0(T_1A):.4f} (infinite lattice)")
for name, r in runs_1A.items():
    var = np.var(r["a"])
    print(f"{name:>6} start: 50 tau_int = {50 * r['tau']:,.0f} sweeps for a reliable tau; "
          f"error 0.001 on <|m|> needs tau Var/0.001^2 = {r['tau'] * var / 1e-6:,.0f} sweeps")
