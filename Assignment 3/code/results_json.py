# ---- notebook cell 31: Validation summary and results.json ----
failed = [name for name, ok in VALIDATION if not ok]
print(f"Validation: {len(VALIDATION) - len(failed)}/{len(VALIDATION)} passed" + (f"; FAILED: {failed}" if failed else ""))

results = {
    "tsn": TSN,
    "seed": SEED,
    "m": [float(v) for v in m_of_T],                                   # 1B: <|m|> at T = 1.0, 1.2, ..., 4.0
    "h": float(H_2A), "acceptance": float(acceptance_2A), "n_sweeps_single": int(n_single),
    "x2_single": float(x2_single), "x2_err_single": float(x2_err_single),
    "x2_err_naive": float(x2_err_naive), "tau_single": float(tau_single),
}                                                                      # bonus 2B not attempted: its fields are left out
with open("results.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
