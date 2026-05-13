"""Domain stratification of the predictive-validity corpus.

Reads:
  data/processed/predictive_validity.json   (per-market predictors + outcomes)

Classifies each market into one of three domains:
  - single_game:  NBA "Team A vs. Team B" markets and single-match soccer
                  ("Will <Team> win on YYYY-MM-DD?"). One decisive event
                  determines the outcome; the framework's source-weighted
                  attractor near 0.5 is correct but uninformative about one
                  high-variance binary realization.
  - sports_other: Multi-game series and tournament markets (NBA Finals,
                  Conference Finals, Stanley Cup, PGL Bucharest grand final).
  - non_sports:   Everything else (geopolitical, Fed-rate, crypto/commodity,
                  political).

Stratification metrics (Brier, log-loss, accuracy) are computed for:
  - single_game
  - sports_all  = single_game + sports_other
  - non_sports

Writes:
  data/processed/domain_stratification.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.predictive_validity import metrics

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

NBA_VS = re.compile(r"\bvs\.\s", re.IGNORECASE)
SOCCER_SINGLE_MATCH = re.compile(r"\bwill\b.+\bwin on \d{4}-\d{2}-\d{2}", re.IGNORECASE)
SPORTS_OTHER = re.compile(
    r"NBA Finals|Western Conference Finals|Eastern Conference Finals|"
    r"Stanley Cup|PGL Bucharest",
    re.IGNORECASE,
)


def classify_domain(question: str) -> str:
    if NBA_VS.search(question) or SOCCER_SINGLE_MATCH.search(question):
        return "single_game"
    if SPORTS_OTHER.search(question):
        return "sports_other"
    return "non_sports"


def main() -> None:
    with (PROC / "predictive_validity.json").open() as f:
        pv = json.load(f)

    rows = pv["per_market"]
    domains = [classify_domain(r["question"]) for r in rows]
    theta = np.array([r["theta"] for r in rows])
    p_ou = np.array([r["p_ou"] for r in rows])
    p_last = np.array([r["p_last"] for r in rows])
    p_uni = np.array([r["p_uniform"] for r in rows])

    strata = {
        "single_game": np.array([d == "single_game" for d in domains]),
        "sports_other": np.array([d == "sports_other" for d in domains]),
        "sports_all": np.array([d in ("single_game", "sports_other") for d in domains]),
        "non_sports": np.array([d == "non_sports" for d in domains]),
    }

    out: dict = {
        "classifier_rules": {
            "single_game": "regex /\\bvs\\.\\s/ OR /\\bwill\\b.+\\bwin on \\d{4}-\\d{2}-\\d{2}/",
            "sports_other": "regex /NBA Finals|Western Conference Finals|"
                            "Eastern Conference Finals|Stanley Cup|PGL Bucharest/",
            "non_sports": "default (no rule matches)",
        },
        "per_market_domain": [
            {"token_id": r["token_id"], "question": r["question"], "domain": d}
            for r, d in zip(rows, domains)
        ],
        "strata": {},
    }

    print(f"{'stratum':<15s}  {'n':>3s}  {'Brier_OU':>9s}  {'Brier_uni':>9s}  "
          f"{'acc_OU':>7s}  {'hits':>7s}")
    for name, mask in strata.items():
        n = int(mask.sum())
        if n == 0:
            continue
        m_ou = metrics(p_ou[mask], theta[mask])
        m_uni = metrics(p_uni[mask], theta[mask])
        m_last = metrics(p_last[mask], theta[mask])
        hits = int(((p_ou[mask] >= 0.5).astype(float) == theta[mask]).sum())
        out["strata"][name] = {
            "n": n,
            "pi_ou": m_ou,
            "pi_uniform": m_uni,
            "pi_last": m_last,
            "hits_ou": hits,
        }
        print(f"{name:<15s}  {n:>3d}  {m_ou['brier']:>9.4f}  {m_uni['brier']:>9.4f}  "
              f"{m_ou['accuracy']:>7.3f}  {hits:>3d}/{n}")

    # Miss attribution for Sec 8.3
    miss = ((p_ou >= 0.5).astype(float) != theta)
    total_misses = int(miss.sum())
    single_misses = int((miss & strata["single_game"]).sum())
    sports_other_misses = int((miss & strata["sports_other"]).sum())
    non_sports_misses = int((miss & strata["non_sports"]).sum())
    out["miss_attribution"] = {
        "total_misses": total_misses,
        "single_game_misses": single_misses,
        "sports_other_misses": sports_other_misses,
        "non_sports_misses": non_sports_misses,
    }
    print(f"\nMisses: total={total_misses}, single_game={single_misses}, "
          f"sports_other={sports_other_misses}, non_sports={non_sports_misses}")

    out_path = PROC / "domain_stratification.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
