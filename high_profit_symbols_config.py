"""
High-Profit Trading Symbols Configuration
Based on deep market research for maximum profit potential
"""

# Exotic Currency Pairs - Highest Volatility
EXOTIC_CURRENCY_PAIRS = {
    # Turkish Lira Pairs - Extreme Volatility
    "USDTRY": {
        "avg_daily_range": 2000,
        "typical_spread": 50,
        "best_hours": "08:00-12:00 ET",
        "risk_factor": 0.005,  # 0.5% due to extreme volatility
        "min_confidence": 0.35,
        "profit_potential": "extreme"
    },
    "EURTRY": {
        "avg_daily_range": 2500,
        "typical_spread": 80,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.004,
        "min_confidence": 0.40,
        "profit_potential": "extreme"
    },
    
    # South African Rand - Commodity Correlation
    "USDZAR": {
        "avg_daily_range": 1834,
        "typical_spread": 100,
        "best_hours": "03:00-17:00 ET",
        "risk_factor": 0.005,
        "min_confidence": 0.35,
        "profit_potential": "very_high"
    },
    
    # Mexican Peso - Oil Correlation
    "USDMXN": {
        "avg_daily_range": 600,
        "typical_spread": 50,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.30,
        "profit_potential": "high"
    },
    
    # Nordic Currencies - Oil & Commodity Plays
    "EURNOK": {
        "avg_daily_range": 800,
        "typical_spread": 30,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "high"
    },
    "EURSEK": {
        "avg_daily_range": 600,
        "typical_spread": 30,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "medium_high"
    },
    
    # Russian Ruble - Oil Correlation (if available)
    "USDRUB": {
        "avg_daily_range": 4500,
        "typical_spread": 200,
        "best_hours": "02:00-10:00 ET",
        "risk_factor": 0.003,
        "min_confidence": 0.45,
        "profit_potential": "extreme"
    }
}

# High-Profit Cross Currency Pairs
CROSS_CURRENCY_PAIRS = {
    # EUR Crosses
    "EURGBP": {
        "avg_daily_range": 60,
        "typical_spread": 2,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.010,
        "min_confidence": 0.25,
        "profit_potential": "medium",
        "strategy": "range_trading"
    },
    "EURAUD": {
        "avg_daily_range": 110,
        "typical_spread": 3,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.28,
        "profit_potential": "high",
        "strategy": "trend_following"
    },
    "EURNZD": {
        "avg_daily_range": 140,
        "typical_spread": 4,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "high",
        "strategy": "volatility_breakout"
    },
    "EURJPY": {
        "avg_daily_range": 100,
        "typical_spread": 2,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.25,
        "profit_potential": "high",
        "strategy": "risk_sentiment"
    },
    
    # GBP Crosses - The Volatility Kings
    "GBPJPY": {
        "avg_daily_range": 150,
        "typical_spread": 3,
        "best_hours": "03:00-12:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.32,
        "profit_potential": "very_high",
        "strategy": "momentum"
    },
    "GBPAUD": {
        "avg_daily_range": 160,
        "typical_spread": 4,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.32,
        "profit_potential": "very_high",
        "strategy": "commodity_divergence"
    },
    "GBPNZD": {
        "avg_daily_range": 200,
        "typical_spread": 5,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.006,
        "min_confidence": 0.35,
        "profit_potential": "extreme",
        "strategy": "volatility_capture"
    },
    
    # JPY Crosses - Risk Sentiment
    "AUDJPY": {
        "avg_daily_range": 80,
        "typical_spread": 2,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.28,
        "profit_potential": "high",
        "strategy": "carry_trade"
    },
    "NZDJPY": {
        "avg_daily_range": 85,
        "typical_spread": 3,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.28,
        "profit_potential": "high",
        "strategy": "carry_trade"
    },
    "CADJPY": {
        "avg_daily_range": 75,
        "typical_spread": 3,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.28,
        "profit_potential": "medium_high",
        "strategy": "oil_correlation"
    },
    
    # Commodity Currency Crosses
    "AUDNZD": {
        "avg_daily_range": 50,
        "typical_spread": 3,
        "best_hours": "19:00-03:00 ET",
        "risk_factor": 0.010,
        "min_confidence": 0.25,
        "profit_potential": "medium",
        "strategy": "mean_reversion"
    },
    "AUDCAD": {
        "avg_daily_range": 70,
        "typical_spread": 3,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.009,
        "min_confidence": 0.28,
        "profit_potential": "medium_high",
        "strategy": "commodity_divergence"
    }
}

# Exotic Metals
EXOTIC_METALS = {
    "XPDUSD": {  # Palladium
        "avg_daily_range": 50,
        "typical_spread": 100,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.005,
        "min_confidence": 0.35,
        "profit_potential": "extreme",
        "volatility": "4%_daily"
    },
    "XPTUSD": {  # Platinum
        "avg_daily_range": 30,
        "typical_spread": 50,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.32,
        "profit_potential": "very_high",
        "volatility": "3%_daily"
    }
}

# Exotic Indices
EXOTIC_INDICES = {
    "MDAX": {  # German Mid-Cap
        "avg_daily_range": 150,
        "typical_spread": 5,
        "best_hours": "03:00-11:30 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "high"
    },
    "KOSPI200": {  # Korean Tech
        "avg_daily_range": 10,
        "typical_spread": 3,
        "best_hours": "20:00-02:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "high"
    },
    "RUSSELL2K": {  # US Small Cap
        "avg_daily_range": 20,
        "typical_spread": 3,
        "best_hours": "09:30-16:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.28,
        "profit_potential": "very_high"
    },
    "HSCEI": {  # Hang Seng China Enterprise
        "avg_daily_range": 200,
        "typical_spread": 8,
        "best_hours": "21:00-04:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.32,
        "profit_potential": "very_high"
    }
}

# Commodity Futures
COMMODITY_FUTURES = {
    "NATGAS": {  # Natural Gas
        "avg_daily_range": 0.1,
        "typical_spread": 10,
        "best_hours": "09:00-14:30 ET",
        "risk_factor": 0.005,
        "min_confidence": 0.35,
        "profit_potential": "extreme",
        "seasonal": "Dec-Feb, Jun-Aug"
    },
    "WHEAT": {
        "avg_daily_range": 10,
        "typical_spread": 5,
        "best_hours": "09:30-14:00 ET",
        "risk_factor": 0.007,
        "min_confidence": 0.32,
        "profit_potential": "high",
        "seasonal": "Mar-May, Jul-Sep"
    },
    "COPPER": {
        "avg_daily_range": 0.05,
        "typical_spread": 5,
        "best_hours": "08:00-17:00 ET",
        "risk_factor": 0.008,
        "min_confidence": 0.30,
        "profit_potential": "high",
        "correlation": "China_data"
    }
}

# Master list of all high-profit symbols
ALL_HIGH_PROFIT_SYMBOLS = []

# Add all exotic currency pairs
ALL_HIGH_PROFIT_SYMBOLS.extend(list(EXOTIC_CURRENCY_PAIRS.keys()))

# Add all cross currency pairs
ALL_HIGH_PROFIT_SYMBOLS.extend(list(CROSS_CURRENCY_PAIRS.keys()))

# Add exotic metals
ALL_HIGH_PROFIT_SYMBOLS.extend(list(EXOTIC_METALS.keys()))

# Add exotic indices
ALL_HIGH_PROFIT_SYMBOLS.extend(list(EXOTIC_INDICES.keys()))

# Add commodity futures
ALL_HIGH_PROFIT_SYMBOLS.extend(list(COMMODITY_FUTURES.keys()))

# Symbol type identification function
def get_symbol_type(symbol):
    """Identify the type of trading symbol"""
    if symbol in EXOTIC_CURRENCY_PAIRS:
        return "exotic_currency"
    elif symbol in CROSS_CURRENCY_PAIRS:
        return "cross_currency"
    elif symbol in EXOTIC_METALS:
        return "exotic_metal"
    elif symbol in EXOTIC_INDICES:
        return "exotic_index"
    elif symbol in COMMODITY_FUTURES:
        return "commodity"
    elif any(metal in symbol for metal in ['XAU', 'XAG']):
        return "precious_metal"
    elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC']):
        return "crypto"
    elif any(index in symbol for index in ['US30', 'NAS100', 'DAX']):
        return "index"
    else:
        return "major_forex"

# Get symbol-specific configuration
def get_symbol_config(symbol):
    """Get configuration for a specific symbol"""
    symbol_type = get_symbol_type(symbol)
    
    if symbol_type == "exotic_currency":
        return EXOTIC_CURRENCY_PAIRS.get(symbol, {})
    elif symbol_type == "cross_currency":
        return CROSS_CURRENCY_PAIRS.get(symbol, {})
    elif symbol_type == "exotic_metal":
        return EXOTIC_METALS.get(symbol, {})
    elif symbol_type == "exotic_index":
        return EXOTIC_INDICES.get(symbol, {})
    elif symbol_type == "commodity":
        return COMMODITY_FUTURES.get(symbol, {})
    else:
        # Default configuration for other symbols
        return {
            "avg_daily_range": 100,
            "typical_spread": 2,
            "best_hours": "08:00-17:00 ET",
            "risk_factor": 0.01,
            "min_confidence": 0.30,
            "profit_potential": "medium"
        }

# Priority ranking for symbol selection
SYMBOL_PRIORITY = {
    "extreme": 1,
    "very_high": 2,
    "high": 3,
    "medium_high": 4,
    "medium": 5
}

def get_priority_symbols(max_symbols=20):
    """Get highest priority symbols based on profit potential"""
    all_symbols = []
    
    # Collect all symbols with their priority
    for symbol, config in EXOTIC_CURRENCY_PAIRS.items():
        all_symbols.append((symbol, config.get("profit_potential", "medium")))
    
    for symbol, config in CROSS_CURRENCY_PAIRS.items():
        all_symbols.append((symbol, config.get("profit_potential", "medium")))
    
    for symbol, config in EXOTIC_METALS.items():
        all_symbols.append((symbol, config.get("profit_potential", "medium")))
    
    for symbol, config in EXOTIC_INDICES.items():
        all_symbols.append((symbol, config.get("profit_potential", "medium")))
    
    for symbol, config in COMMODITY_FUTURES.items():
        all_symbols.append((symbol, config.get("profit_potential", "medium")))
    
    # Sort by priority
    all_symbols.sort(key=lambda x: SYMBOL_PRIORITY.get(x[1], 99))
    
    # Return top symbols
    return [symbol for symbol, _ in all_symbols[:max_symbols]]

# Trading session definitions
TRADING_SESSIONS = {
    "ASIAN": {"start": "00:00", "end": "09:00", "timezone": "Asia/Tokyo"},
    "EUROPEAN": {"start": "08:00", "end": "17:00", "timezone": "Europe/London"},
    "AMERICAN": {"start": "08:00", "end": "17:00", "timezone": "America/New_York"},
    "PACIFIC": {"start": "17:00", "end": "02:00", "timezone": "Pacific/Auckland"}
}

def get_active_session_symbols(current_hour_utc):
    """Get symbols that are most active in current session"""
    active_symbols = []
    
    # Determine active sessions based on UTC hour
    if 0 <= current_hour_utc < 9:  # Asian session
        active_symbols.extend(['USDJPY', 'EURJPY', 'AUDJPY', 'NZDJPY', 'KOSPI200'])
    elif 7 <= current_hour_utc < 16:  # European session
        active_symbols.extend(['EURGBP', 'EURAUD', 'EURNOK', 'EURSEK', 'MDAX', 'DAX'])
    elif 13 <= current_hour_utc < 22:  # American session
        active_symbols.extend(['USDMXN', 'USDCAD', 'RUSSELL2K', 'NATGAS', 'COPPER'])
    elif 21 <= current_hour_utc or current_hour_utc < 6:  # Pacific session
        active_symbols.extend(['AUDNZD', 'AUDUSD', 'NZDUSD', 'HSCEI'])
    
    return active_symbols

print(f"Loaded {len(ALL_HIGH_PROFIT_SYMBOLS)} high-profit trading symbols")