"""Minimal prediction-market dynamics (DeGroot-style with price feedback)."""

from prediction_market.market import Market
from prediction_market.simulation import PredictionMarketSimulation, SimulationResult
from prediction_market.sources import SourceLayer
from prediction_market.trader import Trader

__all__ = [
    "Market",
    "PredictionMarketSimulation",
    "SimulationResult",
    "SourceLayer",
    "Trader",
]
