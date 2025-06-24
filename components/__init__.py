"""
Components package for MT5 Trading System
"""

from .engine_core import UltraTradingEngine
from .trading_components import Signal, SignalType, Trade, CONFIG, HIGH_PROFIT_SYMBOLS, get_symbol_config

__all__ = [
    'UltraTradingEngine',
    'Signal',
    'SignalType',
    'Trade',
    'CONFIG',
    'HIGH_PROFIT_SYMBOLS',
    'get_symbol_config'
]