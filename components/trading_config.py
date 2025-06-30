#!/usr/bin/env python3
"""
Trading Configuration Module
Contains all configuration constants and symbol configurations
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, List


# High-profit symbols configuration
HIGH_PROFIT_SYMBOLS = {
    
    # TAIKINJI - NERUMAE
    "EURUSD": {"avg_daily_range": 80, "typical_spread": 1, "risk_factor": 0.015, "profit_potential": "high"},

    # HIRUYASUMI - YUUGATA
    #"USDJPY": {"avg_daily_range": 100, "typical_spread": 2, "risk_factor": 0.01, "profit_potential": "high"},

}


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Signal:
    type: SignalType
    confidence: float
    entry: float
    sl: float
    tp: float
    reason: str
    strategies: Optional[Dict[str, float]] = None
    quality: float = 0.0

@dataclass 
class Trade:
    ticket: int
    symbol: str
    type: str
    entry_price: float
    sl: float
    tp: float
    volume: float
    entry_time: datetime


class SymbolUtils:
    def __init__(self):
        self.exotic_currencies = CONFIG["EXOTIC_CURRENCIES"]
        self.metal_symbols = CONFIG["METAL_SYMBOLS"]
        self.crypto_symbols = CONFIG["CRYPTO_SYMBOLS"]
        self.index_symbols = CONFIG["INDEX_SYMBOLS"]
    
    def is_exotic_pair(self, symbol: str) -> bool:
        """Check if symbol is an exotic currency pair"""
        return any(exotic in symbol for exotic in self.exotic_currencies)
    
    def is_metal_pair(self, symbol: str) -> bool:
        """Check if symbol is a metal pair"""
        return any(metal in symbol for metal in self.metal_symbols)
    
    def is_crypto_pair(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency pair"""
        return any(crypto in symbol for crypto in self.crypto_symbols)
    
    def is_index_pair(self, symbol: str) -> bool:
        """Check if symbol is an index"""
        return any(index in symbol for index in self.index_symbols)
    
    def get_instrument_type(self, symbol: str) -> str:
        """Get the instrument type of a symbol"""
        if self.is_crypto_pair(symbol):
            return 'crypto'
        elif self.is_metal_pair(symbol):
            return 'metal'
        elif self.is_index_pair(symbol):
            return 'index'
        elif self.is_exotic_pair(symbol):
            return 'exotic'
        else:
            return 'major'
    
    def filter_symbols(self, symbols: List[str], include_types: List[str] = None) -> List[str]:
        """Filter symbols by instrument type"""
        if include_types is None:
            include_types = ['major']
        
        filtered = []
        for symbol in symbols:
            instrument_type = self.get_instrument_type(symbol)
            if instrument_type in include_types:
                filtered.append(symbol)
        
        return filtered
    
    def get_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol"""
        # Most forex pairs are 6 characters (EURUSD)
        # Some might be longer (EURUSD.m, EURUSD_i, etc.)
        clean_symbol = symbol.split('.')[0].split('_')[0]
        
        if len(clean_symbol) >= 6:
            return clean_symbol[:3]
        return ""
    
    def get_quote_currency(self, symbol: str) -> str:
        """Extract quote currency from symbol"""
        clean_symbol = symbol.split('.')[0].split('_')[0]
        
        if len(clean_symbol) >= 6:
            return clean_symbol[3:6]
        return ""
    
    def is_jpy_pair(self, symbol: str) -> bool:
        """Check if symbol involves JPY"""
        return 'JPY' in symbol.upper()
    
    def get_pip_value(self, symbol: str, digits: int) -> float:
        """Get pip value for a symbol based on digits"""
        if self.is_jpy_pair(symbol) and not self.is_metal_pair(symbol):
            # JPY pairs typically have 3 digits, pip is 0.01
            return 0.01
        elif digits == 3 or digits == 5:
            # 5 digit broker, pip is 0.0001
            return 0.0001
        elif digits == 2 or digits == 4:
            # 4 digit broker, pip is 0.01 or 0.0001
            return 0.01 if digits == 2 else 0.0001
        else:
            # Default
            return 0.0001



# ULTRA Aggressive Configuration - FORCE TRADES
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "SYMBOLS": [],  # Will be populated dynamically
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.15,  # 15% minimum confidence (aggressive for more signals)
    "MIN_QUALITY": 0.25,     # 25% quality threshold
    "MIN_STRATEGIES": 3,     # At least 3 strategies must agree (lowered for debugging)
    "MAX_SPREAD_PIPS": 3.0,  # Maximum 3 pips spread for majors
    "MAX_SPREAD_EXOTIC": 15.0,  # Maximum 15 pips for exotic pairs
    "RISK_PER_TRADE": 0.01,  # 1% risk per trade
    "RISK_PER_EXOTIC": 0.005,  # 0.5% risk for exotic pairs
    "MAX_DAILY_LOSS": 0.05,  # 5% max daily loss
    "MAX_CONCURRENT": 5,     # Maximum 5 concurrent positions (increased for diversification)
    "MIN_RR_RATIO": 1.0,     # 1:1 minimum risk-reward ratio (lowered for more trades)
    "MIN_RR_EXOTIC": 1.2,    # 1.2:1 for exotic pairs (lowered for more trades)
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY",
    "SYMBOL_FILTER": "FOREX",
    "MIN_VOLUME": 0.01,
    "AGGRESSIVE_MODE": True,
    "POSITION_INTERVAL": 600,   # 10 minutes between trades per symbol
    "MAX_SYMBOLS": 25,         # Increased to include all high-profit pairs
    "FORCE_TRADE_INTERVAL": 120,  # Force a trade if none in 2 minutes
    "IGNORE_SPREAD": True,     # Ignore spread check for debugging
    "MAX_SPREAD": 999.0,       # Allow any spread for debugging
    "MIN_INDICATORS": 2,       # At least 2 indicators must be positive (lowered for debugging)
    "EXOTIC_CURRENCIES": [],
    "METAL_SYMBOLS": [],
    "CRYPTO_SYMBOLS": [],
    "INDEX_SYMBOLS": [],
    "MAX_SPREAD_METAL": 30.0,  # Gold can have 30 pip spreads
    "MAX_SPREAD_CRYPTO": 50.0,  # Crypto can have very wide spreads
    "MAX_SPREAD_INDEX": 5.0,   # Indices usually 3-5 points
    "RISK_PER_METAL": 0.007,  # 0.7% risk for metals
    "RISK_PER_CRYPTO": 0.003,  # 0.3% risk for crypto due to extreme volatility
    "RISK_PER_INDEX": 0.008,  # 0.8% risk for indices (high profit potential)
    "LOOP_DELAY": 1,  # Main loop delay in seconds
}

def get_symbol_config(symbol):
    """Get configuration for a specific symbol"""
    # Remove # suffix to match config keys
    symbol_base = symbol.rstrip('#')
    return HIGH_PROFIT_SYMBOLS.get(symbol_base, {
        "avg_daily_range": 100,
        "typical_spread": 2,
        "risk_factor": 0.01,
        "profit_potential": "medium"
    })