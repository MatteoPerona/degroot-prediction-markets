"""Preprocess raw Polymarket price histories into clean OU-fitting inputs.

Pipeline per market:
  1. Load raw (t, p) rows from data/raw/{token_id}.csv.
  2. Truncate the saturated tail (drop everything after the market last sat
     in a non-saturated band [in_band_low, in_band_high]).
  3. Resample to a regular time grid (default 1h bars, last observation per bar).
  4. Forward-fill missing bars.
  5. Clip prices to [eps, 1-eps] and compute z = logit(p).

Output: data/processed/{token_id}.csv with columns (t_bar, p_close, p_clipped, z),
plus data/processed/summary.json with per-market diagnostics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass
class PreprocessSummary:
    token_id: str
    question: str
    outcome: str
    n_raw: int
    n_after_tail_truncation: int
    n_bars_resampled: int
    n_bars_after_saturation_drop: int
    bar_size: str
    span_hours: float
    z_min: float
    z_max: float
    z_mean: float
    z_std: float
    saturation_dropped_hours: float


def _logit(p: pd.Series, eps: float = 1e-3) -> pd.Series:
    pc = p.clip(eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def truncate_saturated_tail(
    df: pd.DataFrame,
    *,
    in_band_low: float = 0.02,
    in_band_high: float = 0.98,
) -> pd.DataFrame:
    """Drop trailing rows once the price has saturated for good.

    Find the latest index where price was inside [in_band_low, in_band_high];
    keep everything up to and including that index. If the market never left
    the band, return the dataframe unchanged.
    """
    in_band = (df["p"] >= in_band_low) & (df["p"] <= in_band_high)
    if not in_band.any():
        return df.copy()
    last_in_band_idx = in_band[in_band].index.max()
    return df.loc[:last_in_band_idx].copy()


def resample_to_bars(
    df: pd.DataFrame,
    *,
    bar_size: str = "1h",
) -> pd.DataFrame:
    """Bin (t, p) into regular bars, take last observation per bar, ffill gaps."""
    s = df.set_index(pd.to_datetime(df["t"], unit="s"))["p"]
    bars = s.resample(bar_size).last().ffill()
    bars = bars.dropna()  # leading NaNs if first bar has no obs
    # Coerce datetime index to unix seconds; pandas 3.x stores datetime64[s]
    # so .astype('int64') already gives seconds, but earlier versions give ns.
    t_seconds = bars.index.astype("datetime64[s]").astype("int64")
    out = pd.DataFrame({
        "t_bar": t_seconds,
        "p_close": bars.values,
    })
    return out.reset_index(drop=True)


def preprocess_one(
    token_id: str,
    *,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    bar_size: str = "1h",
    tail_in_band_low: float = 0.02,
    tail_in_band_high: float = 0.98,
    drop_bar_low: float = 0.005,
    drop_bar_high: float = 0.995,
    eps: float = 1e-6,
    metadata: dict | None = None,
) -> PreprocessSummary:
    raw_path = raw_dir / f"{token_id}.csv"
    df_raw = pd.read_csv(raw_path).sort_values("t").reset_index(drop=True)
    n_raw = len(df_raw)

    df_trunc = truncate_saturated_tail(
        df_raw, in_band_low=tail_in_band_low, in_band_high=tail_in_band_high
    )
    n_after_trunc = len(df_trunc)
    sat_dropped_hours = (
        (df_raw["t"].iloc[-1] - df_trunc["t"].iloc[-1]) / 3600.0 if n_after_trunc else 0.0
    )

    df_bars = resample_to_bars(df_trunc, bar_size=bar_size)
    n_bars_resampled = len(df_bars)
    # Drop bars where the close price is essentially resolved (saturation
    # spikes that survived the raw-tail truncation). This is intentionally
    # tighter than the tail-truncation band: we accept moderate-confidence
    # bars (e.g. p=0.95) but drop "as-good-as-resolved" ones.
    keep = (df_bars["p_close"] >= drop_bar_low) & (df_bars["p_close"] <= drop_bar_high)
    df_bars = df_bars.loc[keep].reset_index(drop=True)
    df_bars["z"] = _logit(df_bars["p_close"], eps=eps)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_bars.to_csv(out_dir / f"{token_id}.csv", index=False)

    span_h = (
        (df_bars["t_bar"].iloc[-1] - df_bars["t_bar"].iloc[0]) / 3600.0
        if len(df_bars) > 1 else 0.0
    )

    return PreprocessSummary(
        token_id=token_id,
        question=(metadata or {}).get("question", "")[:80],
        outcome=(metadata or {}).get("outcome", ""),
        n_raw=n_raw,
        n_after_tail_truncation=n_after_trunc,
        n_bars_resampled=n_bars_resampled,
        n_bars_after_saturation_drop=len(df_bars),
        bar_size=bar_size,
        span_hours=span_h,
        z_min=float(df_bars["z"].min()) if len(df_bars) else float("nan"),
        z_max=float(df_bars["z"].max()) if len(df_bars) else float("nan"),
        z_mean=float(df_bars["z"].mean()) if len(df_bars) else float("nan"),
        z_std=float(df_bars["z"].std()) if len(df_bars) else float("nan"),
        saturation_dropped_hours=sat_dropped_hours,
    )


def preprocess_all(
    *,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    bar_size: str = "1h",
) -> list[PreprocessSummary]:
    index_path = raw_dir / "markets.json"
    with index_path.open() as f:
        metas = json.load(f)

    summaries = []
    for m in metas:
        summary = preprocess_one(
            m["token_id"],
            raw_dir=raw_dir,
            out_dir=out_dir,
            bar_size=bar_size,
            metadata=m,
        )
        summaries.append(summary)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)

    return summaries
