#!/usr/bin/env python3
"""
Ultra Trading Engine - Refactored Version
This module now serves as a wrapper that uses the modularized components
"""

# Re-export the main engine class for backward compatibility
from .engine_core import UltraTradingEngine

# Re-export commonly used models and utilities
from .trading_models import Signal, SignalType, Trade
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