"""Predictive validity on the expanded corpus.

For each fitted market, compute three probability predictors:
  - p_ou:      σ(ψ̂)  — OU lifetime attractor
  - p_last:    last observed bar
  - p_uniform: 0.5

Compare against realized binary outcome θ ∈ {0, 1}.

Outputs:
  data/processed/predictive_validity.json
    per_market: list of (token_id, question, theta, p_ou, p_last, p_uniform, sigma_mismatch, r2)
    aggregate:  metrics for each predictor
    bootstrap:  paired bootstrap of p_ou vs p_uniform (and vs p_last) on Brier and log-loss
    by_sigma_tier: aggregate metrics stratified by sigma-mismatch quartile
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.predictive_validity import (
    expit, metrics, paired_bootstrap_difference,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main() -> None:
    with (RAW / "markets.json").open() as f:
        index = {m["token_id"]: m for m in json.load(f)}
    with (PROC / "mom_fits.json").open() as f:
        fits = json.load(f)

    rows = []
    for fit in fits:
        tok = fit["token_id"]
        meta = index.get(tok)
        if meta is None:
            continue
        theta = 1.0 if meta["outcome"] == "YES" else 0.0

        proc = pd.read_csv(PROC / f"{tok}.csv")
        if len(proc) == 0:
            continue
        p_close = proc["p_close"].iloc[-1]
        p_ou = float(expit(fit["psi_hat"]))
        sigma_mismatch = (
            abs(fit["sigma_z_stationary"] - fit["sigma_z_increments"])
            / max(abs(fit["sigma_z_stationary"]), 1e-9)
        )
        rows.append({
            "token_id": tok,
            "question": meta["question"],
            "theta": theta,
            "outcome": meta["outcome"],
            "p_ou": p_ou,
            "p_last": float(p_close),
            "p_uniform": 0.5,
            "psi_hat": fit["psi_hat"],
            "kappa_hat_per_hour": fit["kappa_hat_per_hour"],
            "sigma_mismatch": float(sigma_mismatch),
            "log_acf_r2": fit["log_acf_r2"],
            "n_obs": fit["n_obs"],
        })

    print(f"Markets in predictive set: {len(rows)}")

    theta = np.array([r["theta"] for r in rows])
    p_ou = np.array([r["p_ou"] for r in rows])
    p_last = np.array([r["p_last"] for r in rows])
    p_uni = np.array([r["p_uniform"] for r in rows])

    agg = {
        "pi_ou": metrics(p_ou, theta),
        "pi_last": metrics(p_last, theta),
        "pi_uniform": metrics(p_uni, theta),
    }

    rng = np.random.default_rng(20260511)
    bootstrap = {
        "ou_vs_uniform_brier": paired_bootstrap_difference(
            p_ou, p_uni, theta, metric="brier", n_bootstrap=4000, rng=rng,
        ),
        "ou_vs_uniform_log_loss": paired_bootstrap_difference(
            p_ou, p_uni, theta, metric="log_loss", n_bootstrap=4000, rng=rng,
        ),
        "ou_vs_last_brier": paired_bootstrap_difference(
            p_ou, p_last, theta, metric="brier", n_bootstrap=4000, rng=rng,
        ),
    }

    # Stratify by sigma-mismatch quartile
    sigma_mm = np.array([r["sigma_mismatch"] for r in rows])
    quartiles = np.quantile(sigma_mm, [0.25, 0.50, 0.75])
    def tier(x):
        if x <= quartiles[0]: return "Q1_lowest_mismatch"
        if x <= quartiles[1]: return "Q2"
        if x <= quartiles[2]: return "Q3"
        return "Q4_highest_mismatch"
    tiers = [tier(x) for x in sigma_mm]
    by_tier = {}
    for q in ["Q1_lowest_mismatch", "Q2", "Q3", "Q4_highest_mismatch"]:
        mask = np.array([t == q for t in tiers])
        if mask.sum() == 0:
            continue
        by_tier[q] = {
            "n": int(mask.sum()),
            "sigma_mismatch_range": [
                float(sigma_mm[mask].min()), float(sigma_mm[mask].max())
            ],
            "pi_ou": metrics(p_ou[mask], theta[mask]),
            "pi_last": metrics(p_last[mask], theta[mask]),
            "pi_uniform": metrics(p_uni[mask], theta[mask]),
        }

    # Stratify by log_acf_r2 (poor fit vs good fit)
    r2 = np.array([r["log_acf_r2"] for r in rows])
    by_fit_quality = {}
    for label, mask in [
        ("good_fit_r2>=0.94", r2 >= 0.94),
        ("poor_fit_r2<0.94", r2 < 0.94),
    ]:
        if mask.sum() == 0:
            continue
        by_fit_quality[label] = {
            "n": int(mask.sum()),
            "pi_ou": metrics(p_ou[mask], theta[mask]),
            "pi_uniform": metrics(p_uni[mask], theta[mask]),
        }

    out = {
        "per_market": rows,
        "aggregate": agg,
        "bootstrap": bootstrap,
        "by_sigma_mismatch_quartile": by_tier,
        "by_fit_quality": by_fit_quality,
        "sigma_mismatch_quartile_boundaries": [float(q) for q in quartiles],
    }
    with (PROC / "predictive_validity.json").open("w") as f:
        json.dump(out, f, indent=2)

    # Echo headline numbers
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: brier={v['brier']:.4f}, log_loss={v['log_loss']:.4f}, "
              f"accuracy={v['accuracy']:.3f}")

    print("\n=== OU vs uniform (Brier) ===")
    b = bootstrap["ou_vs_uniform_brier"]
    print(f"  point diff = {b['point_difference']:+.4f}  (negative = OU better)")
    print(f"  95% CI on diff = [{b['p2.5']:+.4f}, {b['p97.5']:+.4f}]")
    print(f"  share OU beats uniform = {b['share_a_beats_b']*100:.1f}%")

    print("\n=== OU vs uniform (log-loss) ===")
    b = bootstrap["ou_vs_uniform_log_loss"]
    print(f"  point diff = {b['point_difference']:+.4f}")
    print(f"  95% CI on diff = [{b['p2.5']:+.4f}, {b['p97.5']:+.4f}]")
    print(f"  share OU beats uniform = {b['share_a_beats_b']*100:.1f}%")

    print("\n=== Stratification by sigma-mismatch quartile ===")
    print(f"  quartile boundaries: {quartiles}")
    for q, d in by_tier.items():
        print(f"  {q}: n={d['n']}, range=[{d['sigma_mismatch_range'][0]:.3f}, "
              f"{d['sigma_mismatch_range'][1]:.3f}], "
              f"OU Brier={d['pi_ou']['brier']:.4f}, acc={d['pi_ou']['accuracy']:.2f}")

    print("\n=== Stratification by log-ACF fit quality ===")
    for k, d in by_fit_quality.items():
        print(f"  {k}: n={d['n']}, OU Brier={d['pi_ou']['brier']:.4f}, "
              f"acc={d['pi_ou']['accuracy']:.2f}")


if __name__ == "__main__":
    main()
