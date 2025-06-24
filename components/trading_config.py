#!/usr/bin/env python3
"""
Trading Configuration Module
Contains all configuration constants and symbol configurations
"""

# High-profit symbols configuration - Only tradable symbols as per UPDATE_INSTRUCTIONS.md
# NOTE: Symbol order DOES NOT affect trading priority! The engine shuffles symbols randomly on each trading loop
# All symbols have equal opportunity to be traded based on market conditions, not their position in this dict
HIGH_PROFIT_SYMBOLS = {
    # Major USD pairs - Equal priority, balanced parameters
    "USDJPY": {
        "avg_daily_range": 100, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "high", "min_rr_ratio": 1.5, "target_rr_ratio": 2.0, "max_rr_ratio": 3.0
    },
    "GBPUSD": {
        "avg_daily_range": 100, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "high", "min_rr_ratio": 1.5, "target_rr_ratio": 2.5, "max_rr_ratio": 4.0
    },
    "EURUSD": {
        "avg_daily_range": 80, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "high", "min_rr_ratio": 1.5, "target_rr_ratio": 2.0, "max_rr_ratio": 3.0
    },
    "USDCAD": {
        "avg_daily_range": 80, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "medium_high", "min_rr_ratio": 1.2, "target_rr_ratio": 1.8, "max_rr_ratio": 2.5
    },
    "USDCHF": {
        "avg_daily_range": 70, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "medium", "min_rr_ratio": 1.0, "target_rr_ratio": 1.5, "max_rr_ratio": 2.0
    },
    
    # JPY Cross pairs - Equal opportunity based on volatility, not order
    "GBPJPY": {
        "avg_daily_range": 150, "typical_spread": 1, "risk_factor": 0.007, 
        "profit_potential": "very_high", "min_rr_ratio": 2.0, "target_rr_ratio": 3.0, "max_rr_ratio": 5.0
    },
    "EURJPY": {
        "avg_daily_range": 100, "typical_spread": 1, "risk_factor": 0.009, 
        "strategy": "risk_sentiment", "min_rr_ratio": 1.5, "target_rr_ratio": 2.5, "max_rr_ratio": 4.0
    },
    "CADJPY": {
        "avg_daily_range": 90, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "medium_high", "min_rr_ratio": 1.5, "target_rr_ratio": 2.0, "max_rr_ratio": 3.0
    },
    "CHFJPY": {
        "avg_daily_range": 80, "typical_spread": 1, "risk_factor": 0.01, 
        "profit_potential": "medium", "min_rr_ratio": 1.2, "target_rr_ratio": 1.8, "max_rr_ratio": 2.5
    },
    
    # Other Cross pairs - Each evaluated independently on market conditions
    "EURGBP": {
        "avg_daily_range": 60, "typical_spread": 1, "risk_factor": 0.010, 
        "strategy": "range_trading", "min_rr_ratio": 1.0, "target_rr_ratio": 1.5, "max_rr_ratio": 2.0
    },
    "EURCAD": {
        "avg_daily_range": 90, "typical_spread": 3, "risk_factor": 0.009, 
        "profit_potential": "medium_high", "min_rr_ratio": 1.2, "target_rr_ratio": 2.0, "max_rr_ratio": 3.0
    },
    "EURCHF": {
        "avg_daily_range": 50, "typical_spread": 1, "risk_factor": 0.010, 
        "profit_potential": "medium", "min_rr_ratio": 0.8, "target_rr_ratio": 1.2, "max_rr_ratio": 1.8
    },
    "GBPCAD": {
        "avg_daily_range": 120, "typical_spread": 3, "risk_factor": 0.008, 
        "profit_potential": "high", "min_rr_ratio": 1.8, "target_rr_ratio": 2.5, "max_rr_ratio": 4.0
    },
    "GBPCHF": {
        "avg_daily_range": 100, "typical_spread": 3, "risk_factor": 0.008, 
        "profit_potential": "high", "min_rr_ratio": 1.5, "target_rr_ratio": 2.2, "max_rr_ratio": 3.5
    },
    "CADCHF": {
        "avg_daily_range": 60, "typical_spread": 2, "risk_factor": 0.010, 
        "profit_potential": "medium", "min_rr_ratio": 1.0, "target_rr_ratio": 1.5, "max_rr_ratio": 2.0
    }
}

# ULTRA Aggressive Configuration - FORCE TRADES
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "SYMBOLS": [],  # Will be populated dynamically
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.30,  # 30% minimum confidence (aggressive for more signals)
    "MIN_QUALITY": 0.25,     # 25% quality threshold
    "MIN_STRATEGIES": 3,     # At least 3 strategies must agree (lowered for debugging)
    "MAX_SPREAD_PIPS": 3.0,  # Maximum 3 pips spread for majors
    "MAX_SPREAD_EXOTIC": 15.0,  # Maximum 15 pips for exotic pairs
    "RISK_PER_TRADE": 0.01,  # 1% risk per trade
    "RISK_PER_EXOTIC": 0.005,  # 0.5% risk for exotic pairs
    "MAX_DAILY_LOSS": 0.30,  # 30% max daily loss
    "MAX_CONCURRENT": 5,     # Maximum 5 concurrent positions (increased for diversification)
    "MIN_RR_RATIO": 0.8,     # 0.8:1 minimum risk-reward ratio (conservative for quick wins)
    "MIN_RR_EXOTIC": 1.0,    # 1:1 for exotic pairs (balanced risk-reward)
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
    "EXOTIC_CURRENCIES": ['TRY', 'ZAR', 'MXN', 'PLN', 'HUF', 'SEK', 'NOK', 'DKK', 
                         'SGD', 'HKD', 'THB', 'CNH', 'RUB', 'BRL', 'INR', 'KRW',
                         'ILS', 'AED', 'SAR', 'PHP', 'IDR', 'MYR', 'CZK', 'RON'],
    "METAL_SYMBOLS": ['XAU', 'XAG', 'XPT', 'XPD'],
    "CRYPTO_SYMBOLS": ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA', 'DOT'],
    "INDEX_SYMBOLS": ['US30', 'USTEC', 'NAS100', 'GER40', 'DAX', 'US500', 'UK100', 'JP225'],
    "MAX_SPREAD_METAL": 30.0,  # Gold can have 30 pip spreads
    "MAX_SPREAD_CRYPTO": 50.0,  # Crypto can have very wide spreads
    "MAX_SPREAD_INDEX": 5.0,   # Indices usually 3-5 points
    "RISK_PER_METAL": 0.007,  # 0.7% risk for metals
    "RISK_PER_CRYPTO": 0.003,  # 0.3% risk for crypto due to extreme volatility
    "RISK_PER_INDEX": 0.008,  # 0.8% risk for indices (high profit potential)
    "MIN_SL_DISTANCE_PERCENT": 0.002,  # Minimum 0.2% SL distance to prevent margin calls
    "MAX_SL_DISTANCE_PERCENT": 0.02,   # Maximum 2% SL distance for risk control
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