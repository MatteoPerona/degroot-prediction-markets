"""Aggregate analysis of rolling-window OU fits.

Inputs:
  data/processed/rolling_ou.json   (output of run_rolling_ou.py)

Computes:
  Per-market summaries
    - first/last kept-window kappa, log-ratio log(kappa_last/kappa_first)
    - OLS slope of log(kappa_hat) on u_norm (early-to-late trend)
  Aggregate
    - distribution of per-market log-ratios; median + bootstrap CI
    - distribution of per-market slopes; median + bootstrap CI; share > 0
    - binned κ̂(u) across markets at u in [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]

The DeGroot/topology-growth prediction is:
  median log-ratio > 0   (κ̂ at the end of a market exceeds κ̂ at the start)
  median slope     > 0
  share of slopes > 0    > 0.5 (binomial against null of no trend)

Writes:
  data/processed/rolling_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def per_market_trend(windows: list[dict]) -> dict:
    kept = [w for w in windows if w["kept"]]
    if len(kept) < 3:
        return {}
    u = np.array([w["u_norm"] for w in kept], dtype=float)
    k = np.array([w["kappa_hat_per_hour"] for w in kept], dtype=float)
    if not (np.all(np.isfinite(k)) and np.all(k > 0)):
        return {}
    log_k = np.log(k)

    # OLS slope of log_k on u_norm
    u_c = u - u.mean()
    lk_c = log_k - log_k.mean()
    denom = float(np.dot(u_c, u_c))
    if denom <= 0:
        return {}
    slope = float(np.dot(u_c, lk_c) / denom)
    intercept = float(log_k.mean() - slope * u.mean())
    # R² of the linear fit
    ss_res = float(np.sum((log_k - (intercept + slope * u)) ** 2))
    ss_tot = float(np.sum((log_k - log_k.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_kept": len(kept),
        "u_first": float(u[0]),
        "u_last": float(u[-1]),
        "kappa_first": float(k[0]),
        "kappa_last": float(k[-1]),
        "log_ratio_last_first": float(log_k[-1] - log_k[0]),
        "slope_log_kappa_vs_u": slope,
        "slope_r2": r2,
        "kappa_median": float(np.median(k)),
    }


def bootstrap_median(x: np.ndarray, *, n_boot: int = 5000,
                     rng_seed: int = 20260511) -> dict:
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    medians = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medians[i] = np.median(x[idx])
    return {
        "median": float(np.median(x)),
        "n": int(n),
        "p2.5": float(np.quantile(medians, 0.025)),
        "p97.5": float(np.quantile(medians, 0.975)),
        "se": float(medians.std(ddof=1)),
        "share_positive": float(np.mean(x > 0)),
    }


def main() -> None:
    with (PROC / "rolling_ou.json").open() as f:
        data = json.load(f)

    per_market_out: list[dict] = []
    for m in data["per_market"]:
        trend = per_market_trend(m["windows"])
        if not trend:
            continue
        per_market_out.append({
            "token_id": m["token_id"],
            "question": m["question"],
            "outcome": m["outcome"],
            "n_bars": m["n_bars"],
            "static_kappa_hat_per_hour": m["static_kappa_hat_per_hour"],
            "static_log_acf_r2": m["static_log_acf_r2"],
            **trend,
        })

    log_ratios = np.array([m["log_ratio_last_first"] for m in per_market_out])
    slopes = np.array([m["slope_log_kappa_vs_u"] for m in per_market_out])

    summary = {
        "params": data["params"],
        "n_markets": len(per_market_out),
        "log_ratio_last_first": bootstrap_median(log_ratios),
        "slope_log_kappa_vs_u": bootstrap_median(slopes),
        "per_market": per_market_out,
    }

    # Aggregate κ̂(u) by u-bin across all kept windows
    bins = [0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9]
    bin_labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0]"]
    bin_stats = []
    for lo, hi, label in zip(bins[:-1], bins[1:], bin_labels):
        ks: list[float] = []
        for m in data["per_market"]:
            for w in m["windows"]:
                if not w["kept"]:
                    continue
                if lo <= w["u_norm"] < hi:
                    ks.append(w["kappa_hat_per_hour"])
        ks_arr = np.asarray(ks, dtype=float)
        bin_stats.append({
            "bin": label,
            "n_windows": int(ks_arr.size),
            "median_kappa": float(np.median(ks_arr)) if ks_arr.size else float("nan"),
            "p25_kappa": float(np.quantile(ks_arr, 0.25)) if ks_arr.size else float("nan"),
            "p75_kappa": float(np.quantile(ks_arr, 0.75)) if ks_arr.size else float("nan"),
            "median_tau_hours": float(1.0 / np.median(ks_arr)) if ks_arr.size and np.median(ks_arr) > 0 else float("nan"),
        })
    summary["bin_stats"] = bin_stats

    out_path = PROC / "rolling_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print(f"Rolling-window aggregate analysis ({summary['n_markets']} markets):\n")
    print("Per-market log(κ̂_last / κ̂_first):")
    lr = summary["log_ratio_last_first"]
    print(f"  median = {lr['median']:+.3f}  (n={lr['n']}, 95% CI [{lr['p2.5']:+.3f}, {lr['p97.5']:+.3f}])")
    print(f"  share with κ̂_last > κ̂_first: {lr['share_positive']:.2%}")
    print()
    print("Per-market OLS slope of log(κ̂) on u_norm:")
    sl = summary["slope_log_kappa_vs_u"]
    print(f"  median = {sl['median']:+.3f}  (n={sl['n']}, 95% CI [{sl['p2.5']:+.3f}, {sl['p97.5']:+.3f}])")
    print(f"  share with positive slope: {sl['share_positive']:.2%}")
    print()
    print("Aggregate κ̂ (per-hour) by u-bin:")
    for b in bin_stats:
        print(f"  {b['bin']:>14s}  n={b['n_windows']:4d}  "
              f"median κ̂={b['median_kappa']:.4f} (τ={b['median_tau_hours']:.1f}h)  "
              f"IQR=[{b['p25_kappa']:.4f},{b['p75_kappa']:.4f}]")


if __name__ == "__main__":
    main()
