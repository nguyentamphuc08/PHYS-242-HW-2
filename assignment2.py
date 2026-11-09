"""PHYS 142/242 - Assignment 2: Tunneling in the double potential well.

Same code as "Assignment 2 Homework.ipynb", as a plain script.

Run it
  - whole file:  python assignment2.py   (or the Run button in VS Code)
  - cell by cell: click "Run Cell" above each "# %%" line (needs the Jupyter extension)

Outputs
  - printed results for every part
  - figures saved to ./figures/*.png (all windows are shown at the end)
  - results.json written next to this file
"""


# %% Setup: parameters, potential, grid integral, harmonic ground states
print("\n=== Setup ===")
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg                              # also loads scipy.sparse
from numba import njit

TSN = "200288141"
ALPHA = 0.3 + int(TSN[-3:]) / 5000                     # = 0.3282
BOXSIZE, ND = 8.0, 600                                 # interval [-4, 4], 600 intervals
DELTAX = BOXSIZE / ND
x = np.linspace(-BOXSIZE / 2, BOXSIZE / 2, ND + 1)     # 601 points, both endpoints included
DELTAT = np.pi / 128                                   # time step epsilon
XMIN = 1 / np.sqrt(ALPHA)                              # well minima at +/- XMIN
OMEGA = 2 * np.sqrt(2)                                 # harmonic frequency (V'' = 8 = omega^2)
N_EXPORT = np.arange(0, 2049, 64)                      # results.json steps n = 0, 64, ..., 2048

try:                                                   # folder of this file (cwd in an interactive window)
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def savefig(name):
    """Save the current figure to figures/<name>.png."""
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"saved figures/{name}.png")


@njit
def V(x, alpha):
    """Double-well potential, eq. (1). @njit so the compiled kernel builder can call it."""
    return alpha * x**4 - 2 * x**2 + 1 / alpha


def integrate(f):
    """Grid integral: dx * sum over all 601 points (assignment convention)."""
    return DELTAX * np.sum(f)


def normalize(psi):
    return psi / np.sqrt(integrate(np.abs(psi)**2))


def plot_mean(ax, tt, curves, title):
    """<x>(t) panel; curves = [(values, label, plot-style dict), ...]."""
    for y, label, style in curves:
        ax.plot(tt, y, label=label, **style)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set(xlabel=r"$t$", ylabel=r"$\langle x\rangle(t)$", xlim=(0, tt[-1]), title=title)


psi_plus, psi_minus = ((OMEGA / np.pi)**0.25 * np.exp(-OMEGA * (x - c)**2 / 2) for c in (+XMIN, -XMIN))  # eq. (5)
print(f"alpha = {ALPHA}, x_min = {XMIN:.6f}, V(0) = {V(0.0, ALPHA):.6f}, grid norm of psi_+ = {integrate(psi_plus**2):.8f}")


# %% A(i): path-integral evolution from psi_+ and the tunneling time
print("\n=== A(i) ===")
@njit
def build_kernel(x, dt, alpha):
    """K~_ij = exp{ i[ (x_i - x_j)^2 / (2 dt) - V((x_i + x_j)/2) dt ] }; row i = arrival, column j = start."""
    K = np.empty((x.size, x.size), dtype=np.complex128)
    for i in range(x.size):
        for j in range(x.size):
            K[i, j] = np.exp(1j * (0.5 * (x[i] - x[j])**2 / dt - V(0.5 * (x[i] + x[j]), alpha) * dt))
    return K


@njit
def evolve(psi0, K, dx, nt):
    """nt steps of psi <- dx K psi, renormalized every step. Returns prob[n, i] = |psi(x_i, n dt)|^2."""
    psi = psi0.astype(np.complex128) / np.sqrt(dx * np.sum(np.abs(psi0)**2))
    prob = np.empty((nt + 1, psi.size))
    prob[0] = np.abs(psi)**2
    for n in range(nt):
        psi = dx * (K @ psi)                                    # one step: matrix-vector product
        psi = psi / np.sqrt(dx * np.sum(np.abs(psi)**2))        # renormalize (prefactor dropped)
        prob[n + 1] = np.abs(psi)**2
    return prob


K_tilde = build_kernel(x, DELTAT, ALPHA)                        # built once (first call also compiles)
T_RUN = 48 * np.pi                                              # >= 16 pi and >= 2 pi/dE (checked in B)
t = np.arange(int(np.ceil(T_RUN / DELTAT)) + 1) * DELTAT
x_mean_A = DELTAX * (evolve(psi_plus, K_tilde, DELTAX, t.size - 1) @ x)  # <x>(t) = dx sum_i x_i |psi_i|^2
n_tunnel = np.argmin(x_mean_A)                                  # deepest point = first major minimum
t_tunnel = t[n_tunnel]
x_mean_pi = x_mean_A[N_EXPORT]                                  # for results.json
print(f"K~ symmetric: {np.allclose(K_tilde, K_tilde.T)};  t_tunnel = {t_tunnel:.4f}, <x> = {x_mean_A[n_tunnel]:.4f}")

fig, ax = plt.subplots(figsize=(9, 4))
plot_mean(ax, t, [(x_mean_A, r"path integral, $\psi(x,0)=\psi_+$", dict(lw=1))],
          rf"A(i): mean position, $\alpha$ = {ALPHA}")
ax.plot(t_tunnel, x_mean_A[n_tunnel], "rv", ms=9,
        label=rf"$t_\mathrm{{tunnel}}$ = {t_tunnel:.2f}, $\langle x\rangle$ = {x_mean_A[n_tunnel]:.3f}")
ax.legend(loc="lower right", fontsize=9)
savefig("A_i_mean_position")


# %% B(i): Hamiltonian, shift-invert eigensolver, validation against Banerjee & Bhatnagar
print("\n=== B(i) ===")
def make_hamiltonian(alpha, L=BOXSIZE, nd=ND):
    """Sets the grid (dx = L/nd) and builds the sparse three-point Hamiltonian H."""
    xg = np.linspace(-L / 2, L / 2, nd + 1)
    dx = L / nd                                         # grid spacing
    off = np.full(nd, -1 / (2 * dx**2))                 # H_{i,i+1} = H_{i+1,i} = -1/(2 dx^2)
    H = scipy.sparse.diags([off, 1 / dx**2 + V(xg, alpha), off], [-1, 0, 1], format="csc")  # H_ii = 1/dx^2 + V
    return xg, dx, H


def lowest_two(alpha, L=BOXSIZE, nd=ND):
    """Two lowest eigenpairs by shift-invert (sigma = 0), normalized and sign-fixed."""
    xg, dx, H = make_hamiltonian(alpha, L, nd)
    E, v = scipy.sparse.linalg.eigsh(H, k=2, sigma=0, which="LM")
    order = np.argsort(E)
    E, phi = E[order], v[:, order] / np.sqrt(dx)        # unit vectors -> dx sum |phi|^2 = 1
    phi[:, 0] *= np.sign(np.sum(phi[:, 0]))             # phi0 positive
    phi[:, 1] *= np.sign(np.sum(phi[xg > 0, 1]))        # right lobe of phi1 positive
    return xg, dx, H, E[0], E[1], phi[:, 0], phi[:, 1]


reference = {0.4: (1.2345071627860, 1.3469408689225),  # Banerjee & Bhatnagar, lambda = 0.10
             0.6: (1.0624992479565, 1.4210868905393)}  # lambda = 0.15
print(f"{'alpha':>5} {'n':>2} {'ours (ND=600)':>14} {'reference':>14} {'error':>10} {'error (ND=1200)':>16} {'ratio':>6}")
for a_val, refs in reference.items():
    E_orig, E_half = lowest_two(a_val)[3:5], lowest_two(a_val, nd=2 * ND)[3:5]
    for n in range(2):
        err, err_h = E_orig[n] - refs[n], E_half[n] - refs[n]
        print(f"{a_val:>5} {n:>2} {E_orig[n]:>14.10f} {refs[n]:>14.10f} {err:>10.2e} {err_h:>16.2e} {err / err_h:>6.2f}")


# %% B(ii): convergence (half spacing, larger box) and original-grid results
print("\n=== B(ii) ===")
grids = [("Original", BOXSIZE, ND), ("Half spacing", BOXSIZE, 2 * ND), ("Larger box", 12.0, 900)]
rows = {name: (L, nd) + lowest_two(ALPHA, L, nd)[3:5] for name, L, nd in grids}
dE_orig = rows["Original"][3] - rows["Original"][2]
print(f"{'grid':<13} {'interval':>9} {'ND':>5} {'E0':>14} {'E1':>14} {'dE':>14} {'dE - dE_orig':>13}")
for name, (L, nd, e0, e1) in rows.items():
    print(f"{name:<13} {f'[{-L/2:g},{L/2:g}]':>9} {nd:>5} {e0:>14.10f} {e1:>14.10f} {e1 - e0:>14.10f} {e1 - e0 - dE_orig:>13.2e}")

x_B, _, H, E0, E1, phi0, phi1 = lowest_two(ALPHA)       # original grid: used from here on
DELTA_E = E1 - E0
print(f"\ndE = {DELTA_E:.10f}, pi/dE = {np.pi / DELTA_E:.4f};  A's run length OK (2pi/dE < 48 pi < 3pi/dE):",
      2 * np.pi / DELTA_E < T_RUN < 3 * np.pi / DELTA_E)


# %% B(iii): eigenfunctions
print("\n=== B(iii) ===")
print(f"norms: {integrate(phi0**2):.12f}, {integrate(phi1**2):.12f};  overlap <phi0|phi1> = {integrate(phi0 * phi1):.1e}")
fig, (ax_phi, ax_V) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
for n, (phi, E) in enumerate([(phi0, E0), (phi1, E1)]):
    ax_phi.plot(x, phi, label=rf"$\phi_{n}$, $E_{n}$ = {E:.6f}")
    ax_V.axhline(E, color=f"C{n}", ls="--", lw=1, label=rf"$E_{n}$")
ax_phi.axhline(0, color="gray", lw=0.7)
ax_phi.set(ylabel=r"$\phi_n(x)$", title=rf"B(iii): two lowest eigenstates, $\alpha$ = {ALPHA}")
ax_phi.legend(loc="upper left", fontsize=9)
ax_V.plot(x, V(x, ALPHA), color="black", label=r"$V(x)$")
ax_V.set(xlabel=r"$x$", ylabel=r"$V(x)$", xlim=(x[0], x[-1]), ylim=(0, 6))
ax_V.legend(loc="upper center", fontsize=9, ncol=3)
savefig("B_iii_eigenstates")


# %% C(i): harmonic-approximation energies (Rayleigh quotients with B's H)
print("\n=== C(i) ===")
psi_S = (psi_plus + psi_minus) / np.sqrt(2)             # unnormalized, as defined
psi_A = (psi_plus - psi_minus) / np.sqrt(2)
E_S, E_A = (integrate(p * (H @ p)) / integrate(p * p) for p in (psi_S, psi_A))   # Rayleigh quotients, B's H
print(f"{'':<8} {'harmonic':>11} {'exact (B)':>11} {'signed err':>11} {'rel. err':>9}")
for name, approx, exact in [("E_S, E0", E_S, E0), ("E_A, E1", E_A, E1), ("gap", E_A - E_S, DELTA_E)]:
    print(f"{name:<8} {approx:>11.8f} {exact:>11.8f} {approx - exact:>11.4f} {abs(approx - exact) / exact:>9.2%}")

S = integrate(psi_plus * psi_minus)
print(f"\nS = {S:.6e} (analytic {np.exp(-OMEGA / ALPHA):.6e});  gap {E_A - E_S:.6e} "
      f"(analytic 6S(1/alpha + 1/omega)/(1-S^2) = {6 * S * (1 / ALPHA + 1 / OMEGA) / (1 - S**2):.6e})")


# %% C(ii): harmonic vs exact ground state on a log scale
print("\n=== C(ii) ===")
psi_S_n = normalize(psi_S)
i0 = np.argmin(np.abs(x))                                          # x = 0
x_turn = np.abs(x[(np.abs(x) < XMIN) & (V(x, ALPHA) > E0)]).max()  # inner turning point (barrier edge)
ratio = phi0[i0] / psi_S_n[i0]
print(f"at x = 0: phi0/psi_S = {ratio:.2f};  barrier |x| < {x_turn:.3f};  V(0) = {V(0.0, ALPHA):.2f} vs V_+(0) = {4 * XMIN**2:.2f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(x, np.abs(phi0), label=r"$|\phi_0|$ (exact, B)")
ax.semilogy(x, np.abs(psi_S_n), "--", label=r"$|\psi_S|$ (harmonic, normalized)")
ax.axvspan(-x_turn, x_turn, color="gray", alpha=0.15, label=r"barrier: $V(x) > E_0$")
ax.annotate(rf"$\phi_0/\psi_S \approx$ {ratio:.1f} at $x=0$", xy=(0, psi_S_n[i0]), xytext=(0.6, 2e-3),
            arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.set(xlabel=r"$x$", ylabel="magnitude (log scale)", xlim=(x[0], x[-1]), ylim=(1e-8, 2),
       title=rf"C(ii): harmonic vs exact ground state, $\alpha$ = {ALPHA}")
ax.legend(loc="lower center", fontsize=9)
savefig("C_ii_log_comparison")


# %% D(i)-(ii): exact two-state evolution and path-integral evolution
print("\n=== D(i)-(ii) ===")
psi0_D = (phi0 + phi1) / np.sqrt(2)                     # right-well two-state initial state
T_D = 2 * np.pi / DELTA_E                               # one full oscillation
t_D = np.arange(int(np.ceil(T_D / DELTAT)) + 1) * DELTAT   # up to the first step at or after T
t_tunnel_D = np.pi / DELTA_E

# D(i): eigenstate evolution (exact at any t, no time stepping)
x01 = integrate(phi0 * x * phi1)
x_mean_eig_D = x01 * np.cos(DELTA_E * t_D)              # every step, for the comparison
x_mean_eig = x01 * np.cos(DELTA_E * N_EXPORT * DELTAT)  # 33 export times, for results.json

# D(ii): path integral, same code as A
prob_PI_D = evolve(psi0_D, K_tilde, DELTAX, t_D.size - 1)
x_mean_PI_D = DELTAX * (prob_PI_D @ x)
print(f"x01 = {x01:.6f};  T = {T_D:.4f} ({t_D.size - 1} steps);  t_tunnel = pi/dE = {t_tunnel_D:.4f};  "
      f"PI minimum at t = {t_D[np.argmin(x_mean_PI_D)]:.4f}")


# %% D(iii): comparison of the two methods
print("\n=== D(iii) ===")
diff = x_mean_PI_D - x_mean_eig_D
n_max = np.argmax(np.abs(diff[t_D <= T_D]))             # eq. (12): only steps with n eps <= T
print(f"max |diff| = {abs(diff[n_max]):.4e} at t = {t_D[n_max]:.4f}: <x>_PI = {x_mean_PI_D[n_max]:.6f}, <x>_eig = {x_mean_eig_D[n_max]:.6f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 2]}, constrained_layout=True)
plot_mean(ax1, t_D, [(x_mean_eig_D, r"eigenstates: $x_{01}\cos(\Delta E\,t)$", dict(lw=2, color="0.6")),
                     (x_mean_PI_D, "path integral", dict(lw=1, color="C3"))],
          rf"D(iii): two-state initial state, $\alpha$ = {ALPHA}")
ax1.legend(loc="upper center", fontsize=9)
ax2.plot(t_D, diff, lw=1)
ax2.plot(t_D[n_max], diff[n_max], "ro")                 # maximum discrepancy
ax2.set(xlabel=r"$t$", xlim=(0, t_D[-1]), ylabel=r"$\langle x\rangle_\mathrm{PI}-\langle x\rangle_\mathrm{eig}$",
        title=rf"difference: max $|\Delta|$ = {abs(diff[n_max]):.3f} at $t$ = {t_D[n_max]:.1f} (red dot)")
savefig("D_iii_mean_position")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
for ax, (name, tt) in zip(axes, [("0", 0.0), (r"t_\mathrm{tunnel}", t_tunnel_D), ("T", T_D)]):
    n = np.argmin(np.abs(t_D - tt))                     # nearest time step
    psi_eig = (np.exp(-1j * E0 * t_D[n]) * phi0 + np.exp(-1j * E1 * t_D[n]) * phi1) / np.sqrt(2)   # eq. (11)
    ax.plot(x, np.abs(psi_eig)**2, lw=2.5, color="0.6", label="eigenstates")
    ax.plot(x, prob_PI_D[n], lw=1, color="C3", label="path integral")
    ax.set(title=rf"near ${name}$: step {n}, $t$ = {t_D[n]:.3f}", xlabel=r"$x$", xlim=(x[0], x[-1]))
axes[0].set_ylabel(r"$|\psi(x,t)|^2$")
axes[0].legend(fontsize=9)
savefig("D_iii_density_snapshots")


# %% A(ii): reading the Problem A result (uses B and D)
print("\n=== A(ii) ===")
two_state = sum(integrate(phi * normalize(psi_plus))**2 for phi in (phi0, phi1))   # |c0|^2 + |c1|^2
mins = np.flatnonzero((x_mean_A[1:-1] < x_mean_A[:-2]) & (x_mean_A[1:-1] < x_mean_A[2:])) + 1  # local minima
P_fast = np.median(np.diff(t[mins]))                    # typical spacing = fast-fluctuation period
print(f"weight of psi_+ in phi0 + phi1 = {two_state:.4f};  fast period ~ {P_fast:.2f} (slow: {2 * np.pi / DELTA_E:.1f})")

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_mean(ax, t, [(x_mean_A, r"A: path integral from $\psi_+$", dict(lw=1)),
                  (x01 * np.cos(DELTA_E * t), r"D: two-state $x_{01}\cos(\Delta E\,t)$", dict(lw=2, color="0.6"))],
          rf"A(ii): Gaussian start (A) vs two-state start (D), $\alpha$ = {ALPHA}")
ax.plot(t_tunnel, x_mean_A[n_tunnel], "rv", ms=9, label=rf"A: major minimum $t$ = {t_tunnel:.2f}")
ax.axvline(np.pi / DELTA_E, color="0.3", ls=":", label=rf"$\pi/\Delta E$ = {np.pi / DELTA_E:.2f}")
ax.annotate(rf"fast fluctuation, period $\approx$ {P_fast:.1f}", xy=(t[mins[5]], x_mean_A[mins[5]]),
            xytext=(2, -0.9), arrowprops=dict(arrowstyle="->", color="C2"), color="C2", fontsize=9)
ax.legend(loc="lower right", fontsize=8)
savefig("A_ii_comparison")


# %% results.json
print("\n=== results.json ===")
results = dict(tsn=TSN, alpha=ALPHA, x_mean_pi=x_mean_pi.tolist(), t_tunnel=float(t_tunnel),
               E0=float(E0), E1=float(E1), phi0=phi0.tolist(), phi1=phi1.tolist(),
               E_S=float(E_S), E_A=float(E_A), x_mean_eig=x_mean_eig.tolist())
with open(HERE / "results.json", "w") as f:
    json.dump(results, f)
print(f"wrote {HERE / 'results.json'}")


# %% Show all figures
plt.show()
