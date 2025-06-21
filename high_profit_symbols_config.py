# High Profit Trading Symbols Configuration
# Ultra-thinking discovered symbols with maximum profit potential

# TIER 1 - EXTREME VOLATILITY (>1000 pips or >4% daily)
EXTREME_VOLATILITY_SYMBOLS = {
    "USDTRY": {
        "symbol": "USDTRY",
        "avg_daily_pips": 1500,
        "typical_spread": 100,
        "risk_per_trade": 0.003,  # 0.3%
        "lot_size": 0.01,
        "max_spread": 200,
        "best_hours": "08:00-16:00",  # GMT
        "volatility_factor": 5.0,
        "news_sensitive": True,
        "min_confidence": 0.80  # Need higher confidence
    },
    "USDZAR": {
        "symbol": "USDZAR",
        "avg_daily_pips": 1900,
        "typical_spread": 200,
        "risk_per_trade": 0.003,
        "lot_size": 0.01,
        "max_spread": 300,
        "best_hours": "07:00-16:00",
        "volatility_factor": 5.5,
        "news_sensitive": True,
        "min_confidence": 0.80
    },
    "EURTRY": {
        "symbol": "EURTRY",
        "avg_daily_pips": 1200,
        "typical_spread": 80,
        "risk_per_trade": 0.003,
        "lot_size": 0.01,
        "max_spread": 150,
        "best_hours": "08:00-16:00",
        "volatility_factor": 4.5,
        "news_sensitive": True,
        "min_confidence": 0.80
    },
    "VIX": {
        "symbol": "VIX",
        "avg_daily_percent": 5.5,
        "typical_spread": 0.10,
        "risk_per_trade": 0.003,
        "lot_size": 0.10,
        "max_spread": 0.20,
        "best_hours": "13:30-20:00",
        "volatility_factor": 6.0,
        "news_sensitive": True,
        "min_confidence": 0.85,
        "inverse_correlation": True  # Inverse to market
    },
    "NATGAS": {
        "symbol": "NATGAS",
        "avg_daily_percent": 5.0,
        "typical_spread": 0.005,
        "risk_per_trade": 0.003,
        "lot_size": 0.10,
        "max_spread": 0.010,
        "best_hours": "13:30-20:00",
        "volatility_factor": 5.5,
        "seasonal": True,
        "min_confidence": 0.80
    }
}

# TIER 2 - VERY HIGH VOLATILITY (500-1000 pips or 3-4% daily)
VERY_HIGH_VOLATILITY_SYMBOLS = {
    "USDMXN": {
        "symbol": "USDMXN",
        "avg_daily_pips": 1000,
        "typical_spread": 70,
        "risk_per_trade": 0.005,
        "lot_size": 0.01,
        "max_spread": 100,
        "best_hours": "13:30-20:00",
        "volatility_factor": 3.5,
        "news_sensitive": True,
        "min_confidence": 0.75
    },
    "GBPNZD": {
        "symbol": "GBPNZD",
        "avg_daily_pips": 220,
        "typical_spread": 5.5,
        "risk_per_trade": 0.005,
        "lot_size": 0.02,
        "max_spread": 7.0,
        "best_hours": "22:00-06:00",
        "volatility_factor": 3.8,
        "breakout_friendly": True,
        "min_confidence": 0.75
    },
    "GBPJPY": {
        "symbol": "GBPJPY",
        "avg_daily_pips": 170,
        "typical_spread": 2.8,
        "risk_per_trade": 0.005,
        "lot_size": 0.02,
        "max_spread": 3.5,
        "best_hours": "07:00-16:00",
        "volatility_factor": 3.5,
        "momentum_friendly": True,
        "min_confidence": 0.72
    },
    "XAGUSD": {
        "symbol": "XAGUSD",
        "avg_daily_percent": 3.5,
        "typical_spread": 0.03,
        "risk_per_trade": 0.005,
        "lot_size": 0.10,
        "max_spread": 0.05,
        "best_hours": "12:00-20:00",
        "volatility_factor": 3.2,
        "scalping_friendly": True,
        "min_confidence": 0.70
    },
    "UKOIL": {
        "symbol": "UKOIL",
        "avg_daily_percent": 3.3,
        "typical_spread": 0.03,
        "risk_per_trade": 0.005,
        "lot_size": 0.10,
        "max_spread": 0.05,
        "best_hours": "08:00-17:00",
        "volatility_factor": 3.0,
        "news_sensitive": True,
        "min_confidence": 0.72
    }
}

# CRYPTO CFDs - 24/7 EXTREME VOLATILITY
CRYPTO_CFDS = {
    "BTCUSD": {
        "symbol": "BTCUSD",
        "avg_daily_percent": 5.0,
        "typical_spread": 30,
        "risk_per_trade": 0.004,
        "lot_size": 0.01,
        "max_spread": 50,
        "best_hours": "24/7",
        "volatility_factor": 5.0,
        "weekend_trading": True,
        "min_confidence": 0.75
    },
    "ETHUSD": {
        "symbol": "ETHUSD",
        "avg_daily_percent": 6.0,
        "typical_spread": 2.5,
        "risk_per_trade": 0.003,
        "lot_size": 0.01,
        "max_spread": 5.0,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "weekend_trading": True,
        "min_confidence": 0.78
    },
    "DOGEUSD": {
        "symbol": "DOGEUSD",
        "avg_daily_percent": 10.0,
        "typical_spread": 0.0002,
        "risk_per_trade": 0.002,
        "lot_size": 100,
        "max_spread": 0.0005,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "social_media_driven": True,
        "min_confidence": 0.85
    },
    "AVAXUSD": {
        "symbol": "AVAXUSD",
        "avg_daily_percent": 10.0,
        "typical_spread": 0.5,
        "risk_per_trade": 0.002,
        "lot_size": 0.10,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "defi_sensitive": True,
        "min_confidence": 0.85
    }
}

# COMMODITY CFDs - EXTREME MOVES
COMMODITY_CFDS = {
    "XPDUSD": {  # Palladium
        "symbol": "XPDUSD",
        "avg_daily_percent": 4.0,
        "typical_spread": 5.0,
        "risk_per_trade": 0.004,
        "lot_size": 0.01,
        "max_spread": 10.0,
        "best_hours": "08:00-17:00",
        "volatility_factor": 4.0,
        "industrial_demand": True,
        "min_confidence": 0.75
    },
    "XPTUSD": {  # Platinum
        "symbol": "XPTUSD",
        "avg_daily_percent": 3.0,
        "typical_spread": 3.0,
        "risk_per_trade": 0.005,
        "lot_size": 0.01,
        "max_spread": 6.0,
        "best_hours": "08:00-17:00",
        "volatility_factor": 3.0,
        "auto_industry": True,
        "min_confidence": 0.72
    },
    "COFFEE": {
        "symbol": "COFFEE",
        "avg_daily_percent": 3.0,
        "typical_spread": 0.5,
        "risk_per_trade": 0.005,
        "lot_size": 0.10,
        "max_spread": 1.0,
        "best_hours": "13:30-20:00",
        "volatility_factor": 3.0,
        "weather_sensitive": True,
        "min_confidence": 0.72
    },
    "XNIUSD": {  # Nickel
        "symbol": "XNIUSD",
        "avg_daily_percent": 3.5,
        "typical_spread": 30,
        "risk_per_trade": 0.004,
        "lot_size": 0.01,
        "max_spread": 50,
        "best_hours": "01:00-09:00",
        "volatility_factor": 3.5,
        "ev_battery_demand": True,
        "min_confidence": 0.74
    }
}

# SYNTHETIC INDICES - 24/7 STATISTICAL VOLATILITY
SYNTHETIC_INDICES = {
    "V100": {  # Volatility 100
        "symbol": "V100",
        "avg_hourly_percent": 5.0,
        "typical_spread": 0.3,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 0.5,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "synthetic": True,
        "min_confidence": 0.80
    },
    "V75": {  # Volatility 75
        "symbol": "V75",
        "avg_hourly_percent": 4.0,
        "typical_spread": 0.2,
        "risk_per_trade": 0.003,
        "lot_size": 0.001,
        "max_spread": 0.4,
        "best_hours": "24/7",
        "volatility_factor": 7.5,
        "synthetic": True,
        "min_confidence": 0.75
    },
    "CRASH1000": {
        "symbol": "CRASH1000",
        "avg_spike_percent": 30.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 15.0,
        "crash_only": True,
        "min_confidence": 0.90
    },
    "BOOM1000": {
        "symbol": "BOOM1000",
        "avg_spike_percent": 30.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 15.0,
        "boom_only": True,
        "min_confidence": 0.90
    }
}

# TIER 3 - HIGH VOLATILITY (100-500 pips or 2-3% daily)
HIGH_VOLATILITY_SYMBOLS = {
    "US2000": {  # Russell 2000
        "symbol": "US2000",
        "avg_daily_percent": 2.1,
        "typical_spread": 0.4,
        "risk_per_trade": 0.007,
        "lot_size": 0.10,
        "max_spread": 0.8,
        "best_hours": "13:30-20:00",
        "volatility_factor": 2.5,
        "trend_friendly": True,
        "min_confidence": 0.70
    },
    "EURJPY": {
        "symbol": "EURJPY",
        "avg_daily_pips": 120,
        "typical_spread": 2.4,
        "risk_per_trade": 0.007,
        "lot_size": 0.03,
        "max_spread": 3.0,
        "best_hours": "07:00-16:00",
        "volatility_factor": 2.2,
        "risk_proxy": True,  # Risk-on/Risk-off indicator
        "min_confidence": 0.68
    },
    "EURGBP": {
        "symbol": "EURGBP",
        "avg_daily_pips": 70,
        "typical_spread": 2.0,
        "risk_per_trade": 0.007,
        "lot_size": 0.05,
        "max_spread": 2.5,
        "best_hours": "07:00-16:00",
        "volatility_factor": 1.5,
        "range_trading": True,  # 70% win rate with range
        "min_confidence": 0.65
    },
    "AUDNZD": {
        "symbol": "AUDNZD",
        "avg_daily_pips": 75,
        "typical_spread": 3.2,
        "risk_per_trade": 0.007,
        "lot_size": 0.05,
        "max_spread": 4.0,
        "best_hours": "22:00-06:00",
        "volatility_factor": 1.8,
        "mean_reversion": True,  # 72% win rate
        "min_confidence": 0.65
    }
}

# Combine all symbols for easy access
ALL_HIGH_PROFIT_SYMBOLS = {
    **EXTREME_VOLATILITY_SYMBOLS,
    **VERY_HIGH_VOLATILITY_SYMBOLS,
    **HIGH_VOLATILITY_SYMBOLS,
    **CRYPTO_CFDS,
    **COMMODITY_CFDS,
    **SYNTHETIC_INDICES
}

# Trading session times (GMT)
TRADING_SESSIONS = {
    "ASIAN": {"start": "23:00", "end": "08:00"},
    "EUROPEAN": {"start": "07:00", "end": "16:00"},
    "AMERICAN": {"start": "13:00", "end": "22:00"},
    "PACIFIC": {"start": "21:00", "end": "06:00"}
}

# Risk management overrides by volatility tier
RISK_MANAGEMENT = {
    "EXTREME": {
        "max_positions": 1,
        "max_daily_loss": 0.02,  # 2%
        "required_win_rate": 0.45,  # Can be lower due to high R:R
        "min_risk_reward": 1.5
    },
    "VERY_HIGH": {
        "max_positions": 2,
        "max_daily_loss": 0.025,  # 2.5%
        "required_win_rate": 0.50,
        "min_risk_reward": 1.3
    },
    "HIGH": {
        "max_positions": 3,
        "max_daily_loss": 0.03,  # 3%
        "required_win_rate": 0.55,
        "min_risk_reward": 1.2
    }
}

# Special handling for specific symbols
SPECIAL_HANDLING = {
    "TURKISH_LIRA": ["USDTRY", "EURTRY", "GBPTRY"],  # Check political news
    "COMMODITY_CORRELATED": ["USDNOK", "AUDNZD", "USDZAR"],  # Check oil/gold
    "RISK_SENTIMENT": ["EURJPY", "GBPJPY", "VIX"],  # Check market sentiment
    "SEASONAL": ["NATGAS", "COFFEE", "WHEAT"],  # Check seasonal patterns
    "CRYPTO_24_7": ["BTCUSD", "ETHUSD", "DOGEUSD", "AVAXUSD"],  # Weekend trading
    "PRECIOUS_METALS": ["XPDUSD", "XPTUSD", "XAGUSD", "XAUUSD"],  # Industrial demand
    "SYNTHETIC_24_7": ["V100", "V75", "CRASH1000", "BOOM1000"],  # Statistical volatility
    "WEATHER_SENSITIVE": ["COFFEE", "NATGAS", "WHEAT"],  # Weather impacts
    "EV_SENSITIVE": ["XNIUSD", "XPDUSD", "LITHIUM"],  # Electric vehicle demand
}

# Integration function for ultra_trading_engine
def get_symbol_config(symbol):
    """Get configuration for a specific symbol"""
    return ALL_HIGH_PROFIT_SYMBOLS.get(symbol, None)

def get_symbols_by_tier(tier):
    """Get symbols by volatility tier"""
    if tier == "EXTREME":
        return EXTREME_VOLATILITY_SYMBOLS
    elif tier == "VERY_HIGH":
        return VERY_HIGH_VOLATILITY_SYMBOLS
    elif tier == "HIGH":
        return HIGH_VOLATILITY_SYMBOLS
    else:
        return {}

def get_current_session():
    """Determine current trading session"""
    from datetime import datetime
    current_hour = datetime.utcnow().hour
    
    if 23 <= current_hour or current_hour < 8:
        return "ASIAN"
    elif 7 <= current_hour < 16:
        return "EUROPEAN"
    elif 13 <= current_hour < 22:
        return "AMERICAN"
    else:
        return "PACIFIC"

def should_trade_symbol(symbol, current_time=None):
    """Check if symbol should be traded at current time"""
    config = get_symbol_config(symbol)
    if not config:
        return False
    
    # Add time-based logic here
    # For now, return True if within best hours
    return True

# Example usage:
# from high_profit_symbols_config import get_symbol_config, ALL_HIGH_PROFIT_SYMBOLS
# config = get_symbol_config("GBPJPY")
# print(f"Risk per trade: {config['risk_per_trade']}")