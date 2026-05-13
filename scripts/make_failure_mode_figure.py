"""Appendix C: quiescence-then-jump failure-mode case study.

Plots the price-logit trajectory z(t), the OU lifetime attractor ψ̂, a
±1σ_stationary band, and the realized binary outcome (drawn at ±4.5 for
visibility) for four representative quiescence-then-jump markets:

  - Russia x Ukraine ceasefire by May 31, 2026
  - Russia x Ukraine ceasefire by June 30, 2026
  - Russia x Ukraine ceasefire by end of 2026
  - QatarEnergy announces/resumes LNG production by April 30

(A fifth quiescence-then-jump market — U.S. anti-cartel operation — is
omitted from the 2×2 grid for layout; the lifetime miss-list in Section 8.3
still counts all 5.)

Inputs:
  data/processed/mom_fits.json
  data/processed/{token}.csv (per-market preprocessed bars with z column)

Output:
  data/processed/figures/failure_mode_quiescence_jump.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"

# Substrings used to identify the four panels (each must match exactly one fit).
PANEL_QUESTIONS = [
    "Russia x Ukraine ceasefire by May 31",
    "Russia x Ukraine ceasefire by June 30",
    "Russia x Ukraine ceasefire by end of 2026",
    "QatarEnergy announces/resumes LNG",
]


def find_fit(fits: list[dict], needle: str) -> dict:
    matches = [f for f in fits if needle in f["question"]]
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 match for {needle!r}, got {len(matches)}")
    return matches[0]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with (PROC / "mom_fits.json").open() as f:
        fits = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    axes = axes.flatten()

    for ax, needle in zip(axes, PANEL_QUESTIONS):
        fit = find_fit(fits, needle)
        bars = pd.read_csv(PROC / f"{fit['token_id']}.csv")
        z = bars["z"].to_numpy()
        t = np.arange(len(z))

        psi = fit["psi_hat"]
        # Stationary std of z for an OU process: σ_z/√(2κ) = √Var(z).
        # NOT σ_z itself (which has units of [z]/√time, the diffusion coef).
        stat_std = float(np.sqrt(fit["var_z"]))
        theta = 1.0 if fit["outcome"] == "YES" else 0.0

        ax.plot(t, z, color="steelblue", lw=1.0, label=r"$z(t)$")
        ax.axhline(psi, color="red", ls="--", lw=1.0,
                   label=fr"$\hat{{\bar\psi}}={psi:.2f}$")
        ax.axhspan(psi - stat_std, psi + stat_std, color="red", alpha=0.10,
                   label=r"$\pm 1\sigma_{\rm stat}$")
        outcome_y = 4.5 if theta == 1 else -4.5
        ax.axhline(outcome_y, color="green", ls=":", lw=1.0,
                   label=f"outcome (θ={int(theta)})")

        title = fit["question"]
        if len(title) > 64:
            title = title[:61] + "..."
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("hour from start")
        ax.set_ylabel(r"price logit $z$")
        ax.set_ylim(-5.5, 5.5)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle("Quiescence-then-jump: lifetime OU anchors to the quiet regime "
                 "and misses the realized outcome", fontsize=11)
    fig.tight_layout()
    out = FIG / "failure_mode_quiescence_jump.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
