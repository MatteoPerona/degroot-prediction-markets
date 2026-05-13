"""Curate an expanded Polymarket corpus for the OU empirical study.

Pre-committed filter chain (decided BEFORE looking at OU fits, to avoid
p-hacking):

  Stage 1 — Gamma metadata:
    - closed = true
    - end_date_min = 2026-03-01 (CLOB ~60-day history window)
    - has clobTokenIds, resolved to YES or NO
    - volume >= MIN_VOLUME ($5M default)

  Stage 2 — De-duplicate multi-candidate events:
    - Group by events[0].id (fallback to events[0].slug, then own id)
    - Per group, keep the highest-volume token (ex-ante criterion).
    - This prevents inflating N with correlated sub-tokens of a single
      multi-way question (Fed chair candidates, NBA Finals teams, etc.).

  Stage 3 — CLOB history:
    - Fetch /prices-history (interval=max) and require >= MIN_HISTORY_PTS
      raw points (100; the §5.1 paper draft used 200, but with 1h bars
      100 raw points is usually enough for a 50-bar OU fit).

  Stage 4 — Preprocessed bars (downstream, in preprocess.py):
    - After saturated-tail truncation and 1h bar resample,
      require n_bars >= MIN_BARS (50) and Var(z) >= MIN_VAR_Z (0.1**2).

Output:
  data/raw/markets.json       — overwritten with the expanded curated set,
                                including the existing 12 markets if they
                                still pass the filters.
  data/raw/{token_id}.csv     — raw histories for each market (existing
                                files reused, new ones fetched).
  data/processed/curation_log.json — drop counts at each stage and the
                                identity of dropped markets, for the §5.1
                                transparency table.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

# Make the data/ package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.polymarket import MarketMetadata, fetch_price_history, save_market

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "Mozilla/5.0 (compatible; degroot-prediction-markets research/0.2)"}

# Pre-committed thresholds
END_DATE_MIN = "2026-03-01"
MIN_VOLUME = 5_000_000.0
MIN_HISTORY_PTS = 100
PAGE_SIZE = 100
MAX_PAGES = 50  # 5000 candidates max — plenty given the 5092 universe seen


def _get_json(url: str, *, timeout: float = 20.0, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def fetch_gamma_universe() -> list[dict]:
    """Page through all closed markets in the CLOB window."""
    out: list[dict] = []
    for page_idx in range(MAX_PAGES):
        params = {
            "closed": "true",
            "limit": str(PAGE_SIZE),
            "offset": str(page_idx * PAGE_SIZE),
            "order": "volumeNum",
            "ascending": "false",
            "end_date_min": END_DATE_MIN,
        }
        url = f"{GAMMA}/markets?" + urllib.parse.urlencode(params)
        page = _get_json(url)
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < PAGE_SIZE:
            break
        time.sleep(0.3)
    return out


def stage1_structural_filter(raw: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep only resolved YES/NO markets with token_ids and volume >= MIN_VOLUME."""
    kept, dropped_reasons = [], []
    for m in raw:
        try:
            ids = json.loads(m.get("clobTokenIds") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
        except json.JSONDecodeError:
            dropped_reasons.append("malformed_json")
            continue
        if not ids or len(prices) < 2:
            dropped_reasons.append("no_token_or_prices")
            continue
        try:
            vol = float(m.get("volume") or 0)
        except (TypeError, ValueError):
            dropped_reasons.append("bad_volume")
            continue
        if vol < MIN_VOLUME:
            dropped_reasons.append("volume_below_threshold")
            continue
        # Resolved YES/NO; for multi-outcome markets (rare) we skip
        if prices[0] == "1":
            outcome = "YES"
        elif prices[1] == "1":
            outcome = "NO"
        else:
            dropped_reasons.append("unresolved_or_multi_outcome")
            continue
        kept.append({**m, "_outcome": outcome, "_volume": vol, "_token": ids[0]})
    return kept, dropped_reasons


def stage2_dedupe_events(markets: list[dict]) -> tuple[list[dict], dict]:
    """Group by event id/slug; keep highest-volume token per group.

    Returns (deduped_markets, log_dict) where log_dict maps each kept market's
    token to the list of dropped sibling markets in the same event.
    """
    def event_key(m: dict) -> str:
        evs = m.get("events") or []
        if isinstance(evs, list) and evs:
            ev = evs[0]
            if isinstance(ev, dict):
                eid = ev.get("id") or ev.get("slug")
                if eid:
                    return f"event:{eid}"
        # Fallback: market is its own group
        return f"market:{m.get('id') or m.get('_token')}"

    groups: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        groups[event_key(m)].append(m)

    deduped: list[dict] = []
    siblings_log: dict[str, list[dict]] = {}
    for k, members in groups.items():
        members.sort(key=lambda x: -x["_volume"])
        winner = members[0]
        deduped.append(winner)
        if len(members) > 1:
            siblings_log[winner["_token"]] = [
                {
                    "question": s.get("question", ""),
                    "token_id": s["_token"],
                    "volume": s["_volume"],
                    "outcome": s["_outcome"],
                }
                for s in members[1:]
            ]
    return deduped, siblings_log


def stage3_clob_history(markets: list[dict]) -> tuple[list[MarketMetadata], list[dict]]:
    """Fetch CLOB history and require >= MIN_HISTORY_PTS raw points."""
    kept, dropped = [], []
    for i, m in enumerate(markets, start=1):
        token = m["_token"]
        # Reuse cached raw CSV if present
        cached = RAW_DIR / f"{token}.csv"
        try:
            if cached.exists():
                # Count rows in cached file (minus header)
                with cached.open() as f:
                    n_pts = sum(1 for _ in f) - 1
            else:
                history = fetch_price_history(token, interval="max")
                n_pts = len(history)
                if n_pts >= MIN_HISTORY_PTS:
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    save_market(
                        MarketMetadata(
                            question=m.get("question", ""),
                            token_id=token,
                            volume=m["_volume"],
                            closed_time=(m.get("closedTime") or "")[:19],
                            outcome=m["_outcome"],
                            n_history_points=n_pts,
                        ),
                        history,
                        RAW_DIR,
                    )
        except Exception as e:
            dropped.append({
                "question": m.get("question", "")[:80],
                "token_id": token,
                "reason": f"clob_fetch_error: {e!r}",
            })
            continue

        if n_pts < MIN_HISTORY_PTS:
            dropped.append({
                "question": m.get("question", "")[:80],
                "token_id": token,
                "reason": f"history_too_short ({n_pts} < {MIN_HISTORY_PTS})",
                "n_points": n_pts,
            })
            continue

        kept.append(
            MarketMetadata(
                question=m.get("question", ""),
                token_id=token,
                volume=m["_volume"],
                closed_time=(m.get("closedTime") or "")[:19],
                outcome=m["_outcome"],
                n_history_points=n_pts,
            )
        )
        if i % 10 == 0:
            print(f"  CLOB stage: {i}/{len(markets)} processed ({len(kept)} kept)", flush=True)
        time.sleep(0.2)  # gentle on the CLOB endpoint

    return kept, dropped


def main() -> None:
    print("Stage 0: paging Gamma API for resolved markets...")
    raw = fetch_gamma_universe()
    print(f"  Gamma returned {len(raw)} closed-market rows")

    print(f"\nStage 1: structural filter (volume >= ${MIN_VOLUME:,.0f}, resolved YES/NO)...")
    s1, s1_drop_reasons = stage1_structural_filter(raw)
    print(f"  kept: {len(s1)}")
    drop_counts = defaultdict(int)
    for r in s1_drop_reasons:
        drop_counts[r] += 1
    for k, v in sorted(drop_counts.items(), key=lambda x: -x[1]):
        print(f"    dropped ({k}): {v}")

    print("\nStage 2: de-duplicate multi-candidate events (one canonical token per event)...")
    s2, siblings_log = stage2_dedupe_events(s1)
    print(f"  events with siblings: {sum(1 for v in siblings_log.values() if v)}")
    print(f"  total sibling tokens dropped: {sum(len(v) for v in siblings_log.values())}")
    print(f"  kept (one per event): {len(s2)}")

    print("\nStage 3: fetch CLOB histories and require >= "
          f"{MIN_HISTORY_PTS} raw points...")
    s3, s3_drop = stage3_clob_history(s2)
    print(f"  kept: {len(s3)}")
    print(f"  dropped: {len(s3_drop)}")

    # Write curated index
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "markets.json"
    with out_path.open("w") as f:
        json.dump([asdict(m) for m in s3], f, indent=2)
    print(f"\nWrote {out_path} with {len(s3)} markets.")

    # Write curation log
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROCESSED_DIR / "curation_log.json"
    with log_path.open("w") as f:
        json.dump({
            "thresholds": {
                "end_date_min": END_DATE_MIN,
                "min_volume": MIN_VOLUME,
                "min_history_points": MIN_HISTORY_PTS,
            },
            "stage0_gamma_rows": len(raw),
            "stage1_after_structural_filter": len(s1),
            "stage1_drop_counts": dict(drop_counts),
            "stage2_after_event_dedup": len(s2),
            "stage2_siblings_dropped": {
                token: siblings for token, siblings in siblings_log.items() if siblings
            },
            "stage3_after_clob_history": len(s3),
            "stage3_drops": s3_drop,
        }, f, indent=2)
    print(f"Wrote curation log: {log_path}")


if __name__ == "__main__":
    main()
