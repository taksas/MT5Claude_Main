"""
Components package for MT5 Trading System
"""

from .ultra_trading_engine import UltraTradingEngine, Signal, SignalType, Trade
from .trading_config import CONFIG, HIGH_PROFIT_SYMBOLS, get_symbol_config

__all__ = [
    'UltraTradingEngine',
    'Signal',
    'SignalType',
    'Trade',
    'CONFIG',
    'HIGH_PROFIT_SYMBOLS',
    'get_symbol_config'
]