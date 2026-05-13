"""Fetch resolved Polymarket markets and their CLOB price histories.

Two endpoints are used:
  - Gamma API (markets metadata): https://gamma-api.polymarket.com/markets
  - CLOB API (prices history):    https://clob.polymarket.com/prices-history

Closed markets older than a few months tend to have empty histories on the
CLOB endpoint, so we restrict to recently-resolved markets (default cutoff
is ~60 days before today).
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
USER_AGENT = "Mozilla/5.0 (compatible; degroot-prediction-markets research/0.1)"


def _get_json(url: str, *, timeout: float = 15.0, retries: int = 3) -> dict | list:
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


@dataclass
class MarketMetadata:
    question: str
    token_id: str
    volume: float
    closed_time: str
    outcome: str  # "YES" or "NO" — which side resolved to 1
    n_history_points: int


def list_resolved_markets(
    *,
    end_date_min: str = "2026-03-01",
    min_volume: float = 1_000_000.0,
    min_history_points: int = 200,
    max_candidates: int = 100,
    target: int = 15,
) -> list[MarketMetadata]:
    """Return resolved Polymarket markets that have queryable CLOB history.

    Filters in order:
      1. Gamma: closed=true, end_date_min, ordered by volume descending.
      2. Local: clob_token_ids non-empty, outcomePrices known.
      3. CLOB: at least ``min_history_points`` price points returned.
    """
    params = {
        "closed": "true",
        "limit": str(max_candidates),
        "order": "volumeNum",
        "ascending": "false",
        "end_date_min": end_date_min,
    }
    url = f"{GAMMA_BASE}/markets?" + urllib.parse.urlencode(params)
    raw = _get_json(url)
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected gamma response: {raw!r}")

    out: list[MarketMetadata] = []
    for m in raw:
        try:
            ids = json.loads(m.get("clobTokenIds") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
        except json.JSONDecodeError:
            continue
        if not ids or len(prices) < 2:
            continue
        try:
            volume = float(m.get("volume") or 0)
        except (TypeError, ValueError):
            continue
        if volume < min_volume:
            continue

        token = ids[0]
        try:
            body = _get_json(
                f"{CLOB_BASE}/prices-history?market={token}&interval=max"
            )
        except Exception:
            continue
        history = body.get("history", []) if isinstance(body, dict) else []
        if len(history) < min_history_points:
            continue

        outcome = "YES" if prices[0] == "1" else "NO" if prices[1] == "1" else "UNKNOWN"
        if outcome == "UNKNOWN":
            continue

        out.append(
            MarketMetadata(
                question=m.get("question", ""),
                token_id=token,
                volume=volume,
                closed_time=(m.get("closedTime") or "")[:19],
                outcome=outcome,
                n_history_points=len(history),
            )
        )
        if len(out) >= target:
            break

    return out


def fetch_price_history(token_id: str, *, interval: str = "max") -> list[dict]:
    """Return [{'t': unix_ts, 'p': price}, ...] for one CLOB token."""
    body = _get_json(
        f"{CLOB_BASE}/prices-history?market={token_id}&interval={interval}"
    )
    if not isinstance(body, dict):
        raise RuntimeError(f"unexpected CLOB response: {body!r}")
    return body.get("history", [])


def save_market(
    meta: MarketMetadata,
    history: list[dict],
    out_dir: Path,
) -> Path:
    """Write one market's history to {out_dir}/{token_id}.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{meta.token_id}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "p"])
        for pt in history:
            writer.writerow([pt["t"], pt["p"]])
    return csv_path


def save_index(metas: list[MarketMetadata], out_dir: Path) -> Path:
    """Write a markets.json index alongside the per-market CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "markets.json"
    with index_path.open("w") as f:
        json.dump([asdict(m) for m in metas], f, indent=2)
    return index_path
