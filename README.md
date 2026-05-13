# 📈 Prediction Markets as DeGroot Dynamics with a Price Channel

We derive a mean-field reduction from a DeGroot opinion-dynamics model with a
price-feedback channel to a scalar Ornstein–Uhlenbeck (OU) SDE for the price
logit `z(t) = logit π(t)`, validate the bridge in simulation, and fit the
resulting estimator to resolved Polymarket markets.

📂 What's in the repo

| Path                                  | Purpose                                                                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [prediction_market/](prediction_market/) | Discrete-event simulator: `SourceLayer`, `Trader`, `Market`, `PredictionMarketSimulation`.                                                                                            |
| [fit/](fit/)                             | OU method-of-moments estimator ([`method_of_moments.py`](fit/method_of_moments.py)), simulator→OU bridge ([`sim_to_ou_bridge.py`](fit/sim_to_ou_bridge.py)), and predictive-validity metrics. |
| [data/](data/)                           | Polymarket Gamma + CLOB fetchers and the preprocessing pipeline (saturated-tail truncation, hourly bars, logit transform).                                                                   |
| [scripts/](scripts/)                     | End-to-end pipeline runners (see below).                                                                                                                                                     |
| [notebooks/](notebooks/)                 | `explore_market_data.ipynb` for the real-data side, `explore_simulation.ipynb` for the simulator.                                                                                        |
| [paper/](paper/)                         | Working draft + Pandoc/LaTeX build instructions.                                                                                                                                             |

`data/raw/` and `data/processed/` are git-ignored; rerun the pipeline below to
regenerate them.

## 🛠️ Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy`, `pandas`, `matplotlib`, and Jupyter — no heavy ML dependencies.

## 🔁 Reproducing the results

Each script writes JSON / pickle / CSV artifacts to `data/processed/` that the
next stage reads. Two parallel pipelines:

### Simulator validation (Section 5)

```sh
# 5.1: β_price sweep — bridge between DeGroot D(t) and OU κ_eff.
python scripts/run_bridge_sweep.py

# 5.2: frozen-topology sweep over |A| ∈ {3, 5, 8, 12, 16, 20, 25, 30}.
python scripts/run_topology_sweep.py

# 5.3: information-matrix rank sweep over rank(Φ) ∈ {1, 2, 3, 5}.
python scripts/run_rank_sweep.py

# Plots for Section 5 + Appendix C.
python scripts/make_sim_figures.py
python scripts/make_failure_mode_figure.py
```

### Empirical pipeline (Sections 6 + 7)

```sh
# 1. Curate a corpus of resolved Polymarket markets (Gamma + CLOB).
#    Pre-committed filter chain: closed, recent, volume >= $5M,
#    deduplicated by event, >= 100 raw history points.
python scripts/curate_corpus.py

# 2. Fit the OU method-of-moments estimator + block bootstrap on each market.
python scripts/run_ou_fits.py

# 3. Score the lifetime-attractor predictor against realized outcomes.
python scripts/run_predictive_validity.py

# 4. Domain stratification (single-game / sports-other / non-sports).
python scripts/classify_markets.py

# 5. Rolling-window OU fits, to test whether κ̂(t) grows over a market's
#    lifetime (the framework's distinctive time-varying-topology claim).
python scripts/run_rolling_ou.py
python scripts/analyze_rolling_ou.py

# 6. Plots for Sections 6 + 7.
python scripts/make_figures.py
python scripts/make_rolling_figure.py
```

## 🔬 Headline results

- ✅ **Bridge holds in simulation.** Across a `β_price` sweep, predicted vs.
  fitted `κ_eff` agree on a 45° line (slope 1.00, ρ = 0.93); predicted vs.
  fitted attractor agree at ρ = 0.9995.
- ✅ **OU shape is consistent on real markets.** 68 markets curated from
  ~5,000 closed Polymarket markets; median mean-reversion timescale 10.5 h,
  log-ACF R² ≥ 0.94 on 62%.
- 🎯 **Lifetime-attractor predictor beats the uninformative null** on the
  OU-shape-consistent stratum (Brier 0.199 vs. 0.250 on 42 good-shape
  markets; 0.204 vs. 0.250 on 24 non-sports markets), but is indistinguishable
  from chance on single-game sports markets (Brier 0.264 on n = 39).
- ❌ **Topology-growth prediction is falsified on this corpus.** Rolling-window
  `κ̂(t)` does not systematically grow over a market's lifetime: median
  per-market slope of `log κ̂` on normalized lifetime is −0.31 (95 % CI
  [−0.57, +0.23]); only 43 % of markets have positive slopes. The forward
  direction DeGroot ⇒ OU is well supported, but identifying DeGroot *from*
  observed OU is not.
