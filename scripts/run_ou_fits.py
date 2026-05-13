"""Run OU method-of-moments fits + block bootstrap on the curated corpus.

Reads:
  data/raw/markets.json       (curated index from curate_corpus.py)
  data/processed/{token}.csv  (preprocessed bars, with column z)

Pre-committed filter (matches §5.1):
  - n_bars_after_saturation_drop >= 50
  - sample variance of z >= 0.1**2

Writes:
  data/processed/mom_fits.json    (per-market OU fits + bootstrap spreads)
  data/processed/fit_log.json     (drop counts + reasons for §5.1 table)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.method_of_moments import block_bootstrap, fit_ou_mom

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MIN_BARS = 50
MIN_VAR_Z = 0.1 ** 2


def main() -> None:
    with (RAW / "markets.json").open() as f:
        index = json.load(f)
    with (PROC / "summary.json").open() as f:
        summaries = {s["token_id"]: s for s in json.load(f)}

    fits_out = []
    dropped = []
    rng = np.random.default_rng(20260511)

    for i, m in enumerate(index, start=1):
        tok = m["token_id"]
        summ = summaries.get(tok)
        if summ is None:
            dropped.append({"token_id": tok, "question": m["question"][:80],
                            "reason": "no_preprocess_summary"})
            continue

        n_bars = summ["n_bars_after_saturation_drop"]
        if n_bars < MIN_BARS:
            dropped.append({"token_id": tok, "question": m["question"][:80],
                            "reason": f"n_bars_below_threshold ({n_bars} < {MIN_BARS})"})
            continue

        proc = pd.read_csv(PROC / f"{tok}.csv")
        if "z" not in proc.columns or len(proc) < MIN_BARS:
            dropped.append({"token_id": tok, "question": m["question"][:80],
                            "reason": "missing_z_or_too_short"})
            continue

        z = proc["z"].to_numpy()
        var_z = float(np.var(z, ddof=1))
        if var_z < MIN_VAR_Z or not math.isfinite(var_z):
            dropped.append({"token_id": tok, "question": m["question"][:80],
                            "reason": f"var_z_below_threshold ({var_z:.4f} < {MIN_VAR_Z:.4f})"})
            continue

        # MoM fit (dt = 1.0 hour since bars are hourly)
        try:
            fit = fit_ou_mom(z, dt=1.0)
        except ValueError as e:
            dropped.append({"token_id": tok, "question": m["question"][:80],
                            "reason": f"fit_failed: {e}"})
            continue

        # Block bootstrap for psi/kappa/sigma_stat spread
        try:
            boot = block_bootstrap(
                z, dt=1.0, point_estimate=fit, n_bootstrap=400, rng=rng,
            )
        except Exception as e:
            boot = {
                "psi_hat": {}, "kappa_hat": {}, "sigma_z_from_stationary": {},
                "_error": str(e),
            }

        fits_out.append({
            "token_id": tok,
            "question": m["question"],
            "outcome": m["outcome"],
            "n_obs": int(fit.n_obs),
            "psi_hat": float(fit.psi_hat),
            "psi_bootstrap": boot.get("psi_hat", {}),
            "kappa_hat_per_hour": float(fit.kappa_hat),
            "kappa_bootstrap": boot.get("kappa_hat", {}),
            "sigma_z_stationary": float(fit.sigma_z_from_stationary),
            "sigma_z_stationary_bootstrap": boot.get("sigma_z_from_stationary", {}),
            "sigma_z_increments": float(fit.sigma_z_from_increments),
            "log_acf_r2": float(fit.log_acf_r2),
            "fit_lags_used": int(fit.fit_lags_used),
            "var_z": float(fit.var_z),
        })

        if i % 10 == 0:
            print(f"  fit {i}/{len(index)}", flush=True)

    out_path = PROC / "mom_fits.json"
    with out_path.open("w") as f:
        json.dump(fits_out, f, indent=2)

    log_path = PROC / "fit_log.json"
    with log_path.open("w") as f:
        json.dump({
            "thresholds": {"min_bars": MIN_BARS, "min_var_z": MIN_VAR_Z},
            "input_markets": len(index),
            "fitted": len(fits_out),
            "dropped": len(dropped),
            "drops": dropped,
        }, f, indent=2)

    print(f"\nFitted {len(fits_out)}/{len(index)} markets; "
          f"dropped {len(dropped)}. Wrote {out_path} and {log_path}.")

    # Quick summary
    if fits_out:
        kappas = np.array([f["kappa_hat_per_hour"] for f in fits_out])
        taus = 1.0 / kappas[kappas > 0]
        r2s = np.array([f["log_acf_r2"] for f in fits_out])
        sig_mismatch = [
            abs(f["sigma_z_stationary"] - f["sigma_z_increments"]) /
            max(abs(f["sigma_z_stationary"]), 1e-9)
            for f in fits_out
        ]
        print(f"  tau median={np.median(taus):.1f}h, p10={np.quantile(taus,0.1):.1f}h, p90={np.quantile(taus,0.9):.1f}h")
        print(f"  log_acf_r2: n>=0.94: {(r2s>=0.94).sum()}, median={np.median(r2s):.3f}")
        print(f"  sigma_mismatch <=15%: {sum(1 for x in sig_mismatch if x<=0.15)}/{len(sig_mismatch)}")


if __name__ == "__main__":
    main()
