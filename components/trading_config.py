#!/usr/bin/env python3
"""
Trading Configuration Module
Contains all configuration constants and symbol configurations
"""

# High-profit symbols configuration
HIGH_PROFIT_SYMBOLS = {
    # Priority symbol as requested
    "USDJPY": {"avg_daily_range": 100, "typical_spread": 2, "risk_factor": 0.01, "profit_potential": "high"},
    # Exotic Currency Pairs - Extreme Volatility
    "USDTRY": {"avg_daily_range": 2000, "typical_spread": 50, "risk_factor": 0.005, "profit_potential": "extreme"},
    "EURTRY": {"avg_daily_range": 2500, "typical_spread": 80, "risk_factor": 0.004, "profit_potential": "extreme"},
    "USDZAR": {"avg_daily_range": 1834, "typical_spread": 100, "risk_factor": 0.005, "profit_potential": "very_high"},
    "USDMXN": {"avg_daily_range": 600, "typical_spread": 50, "risk_factor": 0.007, "profit_potential": "high"},
    "EURNOK": {"avg_daily_range": 800, "typical_spread": 30, "risk_factor": 0.008, "profit_potential": "high"},
    "EURSEK": {"avg_daily_range": 600, "typical_spread": 30, "risk_factor": 0.008, "profit_potential": "medium_high"},
    
    # High-Profit Cross Currency Pairs
    "EURGBP": {"avg_daily_range": 60, "typical_spread": 2, "risk_factor": 0.010, "strategy": "range_trading"},
    "EURAUD": {"avg_daily_range": 110, "typical_spread": 3, "risk_factor": 0.009, "strategy": "trend_following"},
    "EURNZD": {"avg_daily_range": 140, "typical_spread": 4, "risk_factor": 0.008, "strategy": "volatility_breakout"},
    "EURJPY": {"avg_daily_range": 100, "typical_spread": 2, "risk_factor": 0.009, "strategy": "risk_sentiment"},
    "GBPJPY": {"avg_daily_range": 150, "typical_spread": 3, "risk_factor": 0.007, "profit_potential": "very_high"},
    "GBPAUD": {"avg_daily_range": 160, "typical_spread": 4, "risk_factor": 0.007, "profit_potential": "very_high"},
    "GBPNZD": {"avg_daily_range": 200, "typical_spread": 5, "risk_factor": 0.006, "profit_potential": "extreme"},
    "AUDJPY": {"avg_daily_range": 80, "typical_spread": 2, "risk_factor": 0.009, "strategy": "carry_trade"},
    "NZDJPY": {"avg_daily_range": 85, "typical_spread": 3, "risk_factor": 0.009, "strategy": "carry_trade"},
    "AUDNZD": {"avg_daily_range": 50, "typical_spread": 3, "risk_factor": 0.010, "strategy": "mean_reversion"},
    
    # Exotic Metals
    "XPDUSD": {"avg_daily_range": 50, "typical_spread": 100, "risk_factor": 0.005, "volatility": "4%_daily"},
    "XPTUSD": {"avg_daily_range": 30, "typical_spread": 50, "risk_factor": 0.007, "volatility": "3%_daily"},
    
    # Commodities
    "NATGAS": {"avg_daily_range": 0.1, "typical_spread": 10, "risk_factor": 0.005, "seasonal": "Dec-Feb,Jun-Aug"},
    "WHEAT": {"avg_daily_range": 10, "typical_spread": 5, "risk_factor": 0.007, "seasonal": "Mar-May,Jul-Sep"},
    "COPPER": {"avg_daily_range": 0.05, "typical_spread": 5, "risk_factor": 0.008, "correlation": "China_data"}
}

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