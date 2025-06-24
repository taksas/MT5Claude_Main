#!/usr/bin/env python3
"""
Trading Configuration Module
Contains all configuration constants and symbol configurations
"""

# Tradable symbols only - exact list from MT5 API
HIGH_PROFIT_SYMBOLS = {
    "CADJPY": {"priority": 1, "diversification_weight": 1.0},
    "CHFJPY": {"priority": 1, "diversification_weight": 1.0},
    "EURCAD": {"priority": 2, "diversification_weight": 1.2},
    "EURCHF": {"priority": 2, "diversification_weight": 1.1},
    "EURGBP": {"priority": 2, "diversification_weight": 1.1},
    "EURJPY": {"priority": 1, "diversification_weight": 1.0},
    "CADCHF": {"priority": 3, "diversification_weight": 1.3},
    "EURUSD": {"priority": 1, "diversification_weight": 1.0},
    "USDJPY": {"priority": 1, "diversification_weight": 1.0},
    "GBPCAD": {"priority": 2, "diversification_weight": 1.2},
    "GBPCHF": {"priority": 3, "diversification_weight": 1.3},
    "GBPJPY": {"priority": 1, "diversification_weight": 1.0},
    "GBPUSD": {"priority": 1, "diversification_weight": 1.0},
    "USDCAD": {"priority": 2, "diversification_weight": 1.1},
    "USDCHF": {"priority": 2, "diversification_weight": 1.1}
}

# Symbol-specific overrides (only when different from defaults)
SYMBOL_OVERRIDES = {
    "GBPJPY": {"typical_spread": 2, "target_rr_ratio": 3.0, "diversification_weight": 1.0},
    "GBPUSD": {"target_rr_ratio": 2.5, "diversification_weight": 1.0},
    "EURCAD": {"typical_spread": 3, "diversification_weight": 1.2},
    "GBPCAD": {"typical_spread": 3, "target_rr_ratio": 2.5, "diversification_weight": 1.2},
    "GBPCHF": {"typical_spread": 3, "target_rr_ratio": 2.2, "diversification_weight": 1.3},
    "CADCHF": {"typical_spread": 2, "diversification_weight": 1.3}
}

# ULTRA Aggressive Configuration - FORCE TRADES WITH DIVERSIFICATION
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.30,
    "MAX_SPREAD_PIPS": 3.0,
    "RISK_PER_TRADE": 0.01,
    "RISK_PER_EXOTIC": 0.005,
    "RISK_PER_CRYPTO": 0.003,
    "RISK_PER_METAL": 0.008,
    "RISK_PER_INDEX": 0.01,
    "MAX_DAILY_LOSS": 0.30,
    "MAX_CONCURRENT": 5,
    "MIN_RR_RATIO": 0.8,
    "MIN_RR_EXOTIC": 1.2,
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY",
    "MIN_VOLUME": 0.01,
    "POSITION_INTERVAL": 600,
    "FORCE_TRADE_INTERVAL": 120,
    "MIN_SL_DISTANCE_PERCENT": 0.002,
    "MAX_SL_DISTANCE_PERCENT": 0.02,
    "AGGRESSIVE_MODE": True,
    "IGNORE_SPREAD": False,
    "SYMBOL_FILTER": "ALL",
    "MAX_SYMBOLS": 15,
    "DIVERSIFICATION_MODE": True,
    "PROACTIVE_POSITION_SEEKING": True
}

def get_symbol_config(symbol):
    """Get configuration for a specific symbol with defaults and diversification settings"""
    symbol_base = symbol.rstrip('#')
    
    # Default configuration
    config = {
        "typical_spread": 1,
        "min_rr_ratio": 1.5,
        "target_rr_ratio": 2.0,
        "max_rr_ratio": 3.0,
        "diversification_weight": 1.0,
        "priority": 2
    }
    
    # Apply symbol-specific overrides if they exist
    if symbol_base in SYMBOL_OVERRIDES:
        config.update(SYMBOL_OVERRIDES[symbol_base])
    
    # Apply diversification settings from HIGH_PROFIT_SYMBOLS
    if symbol_base in HIGH_PROFIT_SYMBOLS:
        config.update(HIGH_PROFIT_SYMBOLS[symbol_base])
        
    return config