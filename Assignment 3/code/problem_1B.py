# ---- notebook cell 17: Problem 1B: pilot and production chains at every temperature ----
T_GRID = np.round(np.arange(1.0, 4.0 + 1e-9, 0.2), 1)    # 16 temperatures
PILOT_SWEEPS = 50_000
N_CHAINS = 4
MIN_SWEEPS_1B = 50_000

scan_1B = []
for iT, T in enumerate(T_GRID):
    # pilot chain (run 300 + iT): tau_int(T) and the burn-in
    rng = rng_for(300 + iT)
    m_pilot, _ = ising_sweeps(initial_spins("up", rng), T, PILOT_SWEEPS, rng)
    burn, t_eq, tau_pilot = burn_in(np.abs(m_pilot))
    burn = max(burn, BURNIN_1A)
    # blocking needs >= MIN_BLOCKS blocks of B* >= 20 tau; size it with a factor-2 margin on the pilot's tau
    B_needed = int(2 ** np.ceil(np.log2(20 * 2 * tau_pilot)))
    n_keep = int(max(MIN_SWEEPS_1B, MIN_BLOCKS * B_needed))

    # production chains (runs 100 + 10 iT + c), all-up start, every kept sweep used (no thinning)
    kept, means, errs, taus, accs, enough_all = [], [], [], [], [], True
    for c in range(N_CHAINS):
        rng = rng_for(100 + 10 * iT + c)
        m, acc = ising_sweeps(initial_spins("up", rng), T, burn + n_keep, rng)
        a = np.abs(m[burn + 1:])
        tau = tau_int(a)
        err, B, enough = blocking_error(a, tau)
        kept.append(a); means.append(a.mean()); errs.append(err); taus.append(tau); accs.append(acc)
        enough_all &= enough and n_keep >= 50 * tau
    kept = np.concatenate(kept)
    scan_1B.append(dict(
        T=T, burn=burn, n_keep=n_keep, mean=np.mean(means), std=np.std(kept),
        err_block=np.sqrt(np.sum(np.square(errs))) / N_CHAINS,      # error of the average of N_CHAINS chain means
        err_spread=np.std(means, ddof=1) / np.sqrt(N_CHAINS),       # same error, from the spread of chain means
        tau=np.mean(taus), acc=np.mean(accs), ok=enough_all))

print(f"{'T':>4} {'burn':>6} {'kept/chain':>10} {'<|m|>':>7} {'std |m|':>8} {'err(block)':>10} "
      f"{'err(spread)':>11} {'tau_int':>8} {'accept':>7} {'m0(T)':>7}  N>=50tau")
for r in scan_1B:
    print(f"{r['T']:4.1f} {r['burn']:6d} {r['n_keep']:10,d} {r['mean']:7.4f} {r['std']:8.4f} {r['err_block']:10.5f} "
          f"{r['err_spread']:11.5f} {r['tau']:8.1f} {r['acc']:7.3f} {float(yang_m0(r['T'])):7.4f}  {r['ok']}")
m_of_T = [r["mean"] for r in scan_1B]          # 16 values of <|m|> for results.json


# ---- notebook cell 18: Problem 1B: plot of <|m|> and tau_int against T ----
T_arr = np.array([r["T"] for r in scan_1B])
mean_arr = np.array([r["mean"] for r in scan_1B])
std_arr = np.array([r["std"] for r in scan_1B])
err_arr = np.array([r["err_block"] for r in scan_1B])
T_fine = np.linspace(1.0, 4.0, 600)

fig, (ax_m, ax_tau) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
ax_m.fill_between(T_arr, mean_arr - std_arr, mean_arr + std_arr, color="C0", alpha=0.2,
                  label=r"$\pm$ standard deviation of $|m|$ (fluctuations)")
ax_m.errorbar(T_arr, mean_arr, yerr=err_arr, fmt="o", ms=4, capsize=3, color="C0",
              label=r"$\langle|m|\rangle$ $\pm$ blocking error of the mean")
ax_m.plot(T_fine, yang_m0(T_fine), "k-", lw=1.2, label=r"infinite lattice $m_0(T)$ (Yang)")
ax_m.axvline(T_C, color="C3", ls="--", lw=1, label=rf"$T_c$ = {T_C:.3f}")
ax_m.set(ylabel="magnetization per spin", ylim=(-0.02, 1.05),
         title=f"1B: {L}x{L} lattice, {N_CHAINS} chains per temperature")
ax_m.legend(fontsize=9, loc="upper right")
ax_tau.semilogy(T_arr, [r["tau"] for r in scan_1B], "o-", ms=4, color="C2")
ax_tau.axvline(T_C, color="C3", ls="--", lw=1)
ax_tau.set(xlabel="temperature T (J = k_B = 1)", ylabel=r"$\tau_{int}$ of $|m|$ (sweeps)",
           title="critical slowing down")
save(fig, "1B_magnetization")
plt.show()
