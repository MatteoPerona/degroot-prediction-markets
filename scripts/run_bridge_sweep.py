"""Sec 5.1 bridge sweep: β_price ∈ {0.1, ..., 0.8} × 5 seeds.

Thin wrapper around fit.sim_to_ou_bridge.sweep() that serializes results to
data/processed/bridge_results.json (consumed by scripts/make_sim_figures.py).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit.sim_to_ou_bridge import sweep

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seeds = [0, 1, 2, 3, 4]
    results = sweep(betas, seeds=seeds)

    out_path = PROC / "bridge_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Wrote {out_path.relative_to(ROOT)} ({len(results)} runs)")


if __name__ == "__main__":
    main()
