"""Rolling-window OU fits across the Polymarket corpus.

For each market with enough bars, slide a window of width ``W`` with stride
``S`` across the price-logit series and fit the OU process on each window.
The question this experiment answers: does the mean-reversion rate
$\\hat\\kappa(t)$ grow over a market's lifetime, as the DeGroot framework
predicts via $|A(t)|$ growth implying $\\rho(D(t))$ growth implying
$\\kappa_{\\text{eff}}(t) = 1 - \\rho(D(t))$ growth?

The matrix-spectral story predicts a grow-then-plateau trajectory
(Seeding -> Discovery -> Consensus, model formulation §9). Two alternative
OU-producing models predict different behavior:

  - Representative-agent slow Bayesian: roughly constant $\\hat\\kappa(t)$
    across the market lifetime.
  - Microstructure noise + exogenous fundamental: $\\hat\\kappa$ tied to
    news arrival rate, not to participation, so no systematic growth.

So a systematic $\\hat\\kappa(t)$ growth pattern across the corpus would
discriminate the DeGroot framework from these alternatives; absence of such
growth would refute the time-varying-topology distinctive claim and limit
the empirical content of the bridge to "data are OU-consistent."

Pre-committed thresholds (committed before any fits are run on real data):
  W = 72 bars (3 days; ~7× the corpus-median timescale of 10.5 h)
  S = 24 bars (1 day)
  MIN_BARS = 144 (≥ 2W, so at least 4 windows per market)
  MIN_VAR_Z = 0.01 (same as the static fit; rules out flat segments)
  R²_min = 0.50 (drop window fits whose log-ACF is clearly not exponential)

Outputs:
  data/processed/rolling_ou.json   (per-window fits, indexed by token)
  data/processed/rolling_summary.json (aggregate κ̂(t) statistics)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.method_of_moments import fit_ou_mom

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

W = 72
S = 24
MIN_BARS = 2 * W
MIN_VAR_Z = 0.01
R2_MIN = 0.50


def rolling_fits(z: np.ndarray, *, dt: float = 1.0) -> list[dict]:
    """Slide a window of width W with stride S across z and fit OU on each.

    Returns a list of per-window dicts with center, span, kappa_hat, psi_hat,
    sigma_hat, log_acf_r2, var_z, and a flag for whether the fit was kept.
    """
    n = z.size
    out: list[dict] = []
    if n < W:
        return out
    centers = list(range(W // 2, n - W // 2 + 1, S))
    for c in centers:
        lo, hi = c - W // 2, c + W // 2
        z_win = z[lo:hi]
        var_z = float(np.var(z_win, ddof=1))
        record: dict = {
            "center_idx": int(c),
            "lo_idx": int(lo),
            "hi_idx": int(hi),
            "u_norm": (c - W // 2) / (n - W),   # 0 at first window, 1 at last
            "var_z": var_z,
            "kept": False,
            "kappa_hat_per_hour": float("nan"),
            "psi_hat": float("nan"),
            "sigma_z_stationary": float("nan"),
            "log_acf_r2": float("nan"),
        }
        if var_z < MIN_VAR_Z:
            record["drop_reason"] = "var_z_below_threshold"
            out.append(record)
            continue
        try:
            fit = fit_ou_mom(z_win, dt=dt)
        except ValueError as e:
            record["drop_reason"] = f"fit_failed: {e}"
            out.append(record)
            continue
        record["kappa_hat_per_hour"] = float(fit.kappa_hat)
        record["psi_hat"] = float(fit.psi_hat)
        record["sigma_z_stationary"] = float(fit.sigma_z_from_stationary)
        record["log_acf_r2"] = float(fit.log_acf_r2)
        if fit.log_acf_r2 < R2_MIN or fit.kappa_hat <= 0:
            record["drop_reason"] = (
                f"poor_r2 ({fit.log_acf_r2:.2f})" if fit.log_acf_r2 < R2_MIN
                else f"non_positive_kappa ({fit.kappa_hat:.4f})"
            )
            out.append(record)
            continue
        record["kept"] = True
        out.append(record)
    return out


def main() -> None:
    with (PROC / "mom_fits.json").open() as f:
        static_fits = json.load(f)

    per_market: list[dict] = []
    dropped: list[dict] = []

    for entry in static_fits:
        tok = entry["token_id"]
        proc = pd.read_csv(PROC / f"{tok}.csv")
        z = proc["z"].to_numpy()
        n_bars = z.size

        if n_bars < MIN_BARS:
            dropped.append({
                "token_id": tok,
                "question": entry["question"][:80],
                "reason": f"n_bars_below_threshold ({n_bars} < {MIN_BARS})",
            })
            continue

        windows = rolling_fits(z, dt=1.0)
        kept = [w for w in windows if w["kept"]]
        if len(kept) < 3:
            dropped.append({
                "token_id": tok,
                "question": entry["question"][:80],
                "n_bars": int(n_bars),
                "n_windows": len(windows),
                "n_kept": len(kept),
                "reason": f"too_few_kept_windows ({len(kept)} < 3)",
            })
            continue

        per_market.append({
            "token_id": tok,
            "question": entry["question"],
            "outcome": entry["outcome"],
            "static_kappa_hat_per_hour": entry["kappa_hat_per_hour"],
            "static_psi_hat": entry["psi_hat"],
            "static_log_acf_r2": entry["log_acf_r2"],
            "n_bars": int(n_bars),
            "n_windows": len(windows),
            "n_kept": len(kept),
            "windows": windows,
        })

    out_path = PROC / "rolling_ou.json"
    with out_path.open("w") as f:
        json.dump({
            "params": {
                "W_bars": W, "stride_bars": S, "min_bars": MIN_BARS,
                "min_var_z": MIN_VAR_Z, "r2_min": R2_MIN,
            },
            "n_input_markets": len(static_fits),
            "n_fitted_markets": len(per_market),
            "dropped": dropped,
            "per_market": per_market,
        }, f, indent=2)

    print(f"Rolling-window OU fits: {len(per_market)}/{len(static_fits)} markets "
          f"(dropped {len(dropped)}). Wrote {out_path.relative_to(ROOT)}.")
    if per_market:
        n_kept = [m["n_kept"] for m in per_market]
        print(f"  per-market kept windows: median={np.median(n_kept):.1f}, "
              f"min={min(n_kept)}, max={max(n_kept)}, total={sum(n_kept)}")


if __name__ == "__main__":
    main()
