"""
Ultra Rare High-Profit Trading Symbols Configuration
Based on deep market research for maximum profit potential
"""

# Exotic Forex Pairs - Highest Volatility (200+ pips daily)
EXOTIC_FOREX_ULTRA = {
    "USDZAR": {  # South African Rand - 1000-1834 pips daily
        "symbol": "USDZAR",
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 100,  # Wide spread for exotic
        "profit_target_pips": 300,
        "stop_loss_pips": 150,
        "trading_hours": "13:00-17:00",  # London/NY overlap
        "volatility_factor": 2.5,
        "risk_level": "extreme"
    },
    "USDTRY": {  # Turkish Lira - 1000-2000 pips daily
        "symbol": "USDTRY", 
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 120,
        "profit_target_pips": 400,
        "stop_loss_pips": 200,
        "trading_hours": "08:00-16:00",  # European session
        "volatility_factor": 3.0,
        "risk_level": "extreme"
    },
    "USDBRL": {  # Brazilian Real - 400-800 pips daily
        "symbol": "USDBRL",
        "timeframes": ["M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 80,
        "profit_target_pips": 250,
        "stop_loss_pips": 125,
        "trading_hours": "13:00-21:00",  # NY session
        "volatility_factor": 2.0,
        "risk_level": "high"
    },
    "USDMXN": {  # Mexican Peso - 500-800 pips (can spike to 5000+)
        "symbol": "USDMXN",
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 60,
        "profit_target_pips": 200,
        "stop_loss_pips": 100,
        "trading_hours": "13:00-21:00",  # NY session
        "volatility_factor": 2.2,
        "risk_level": "high"
    },
    "USDNGN": {  # Nigerian Naira - 200-400 pips daily
        "symbol": "USDNGN",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "max_spread": 150,
        "profit_target_pips": 150,
        "stop_loss_pips": 75,
        "trading_hours": "08:00-16:00",  # London session
        "volatility_factor": 1.8,
        "risk_level": "high"
    }
}

# Synthetic Indices (Deriv Broker Specific)
SYNTHETIC_INDICES = {
    "BOOM1000": {  # Boom 1000 Index
        "symbol": "Boom 1000 Index",
        "timeframes": ["M1", "M5"],
        "lot_size": 0.20,
        "strategy": "spike_catcher",
        "avg_spike_interval": 1000,
        "profit_per_spike": 100,
        "risk_level": "extreme"
    },
    "CRASH1000": {  # Crash 1000 Index
        "symbol": "Crash 1000 Index",
        "timeframes": ["M1", "M5"],
        "lot_size": 0.20,
        "strategy": "drop_catcher",
        "avg_drop_interval": 1000,
        "profit_per_drop": 100,
        "risk_level": "extreme"
    },
    "V75": {  # Volatility 75 Index
        "symbol": "Volatility 75 Index",
        "timeframes": ["M5", "M15"],
        "lot_size": 0.10,
        "volatility": 75,
        "avg_daily_range": 500,
        "risk_level": "high"
    },
    "V100": {  # Volatility 100 Index
        "symbol": "Volatility 100 Index",
        "timeframes": ["M5", "M15"],
        "lot_size": 0.05,
        "volatility": 100,
        "avg_daily_range": 800,
        "risk_level": "extreme"
    }
}

# High-Volatility Commodities
RARE_COMMODITIES = {
    "OJ": {  # Orange Juice - Most volatile commodity
        "symbol": "OJ",
        "name": "Orange Juice Futures",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "seasonal_peak": "June-November",  # Hurricane season
        "avg_monthly_volatility": 20,  # Percentage
        "2024_range": [230, 589],  # Price range
        "risk_level": "extreme"
    },
    "KC": {  # Coffee Arabica - All-time highs
        "symbol": "KC",
        "name": "Coffee Futures",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "seasonal_peak": "May-August",  # Brazil frost
        "current_price": 349.58,  # Record high
        "avg_monthly_volatility": 12,
        "risk_level": "high"
    },
    "CC": {  # Cocoa - Record highs Q4 2024
        "symbol": "CC",
        "name": "Cocoa Futures",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "seasonal_peak": "Oct-Mar",  # Harvest
        "avg_monthly_volatility": 15,
        "risk_level": "high"
    },
    "NG": {  # Natural Gas - Energy play
        "symbol": "NG",
        "name": "Natural Gas Futures",
        "timeframes": ["M30", "H1"],
        "lot_size": 0.01,
        "seasonal_peak": "Dec-Feb",  # Winter
        "contract_size": 10000,  # MMBtu
        "risk_level": "high"
    },
    "PA": {  # Palladium - Precious metal alternative
        "symbol": "PA",
        "name": "Palladium Futures",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "2024_range": [900, 1100],
        "contract_size": 100,  # Troy ounces
        "risk_level": "medium"
    }
}

# Master configuration combining all high-profit symbols
ULTRA_PROFIT_SYMBOLS = {
    **EXOTIC_FOREX_ULTRA,
    **SYNTHETIC_INDICES,
    **RARE_COMMODITIES
}

# Trading parameters optimized for high volatility
ULTRA_PROFIT_SETTINGS = {
    "risk_per_trade": 0.005,  # 0.5% for extreme volatility
    "max_concurrent_trades": 2,
    "confidence_threshold": 80,  # Higher threshold for exotics
    "avoid_hours": ["03:00-07:00"],  # Low liquidity JST
    "news_filter": True,
    "correlation_check": True,
    "max_daily_loss": 0.02,  # 2% daily loss limit
    "profit_factor_target": 2.0,
    "use_trailing_stop": True,
    "trailing_stop_distance": 50,
    "scale_in_enabled": False,  # Too risky for exotics
    "martingale_enabled": False  # Never for high volatility
}

# Special strategies for different symbol types
SYMBOL_STRATEGIES = {
    "exotic_forex": {
        "primary": "trend_following",
        "secondary": "breakout",
        "indicators": ["ATR", "Bollinger", "RSI"],
        "news_sensitivity": "extreme"
    },
    "synthetic": {
        "primary": "spike_trading",
        "secondary": "mean_reversion",
        "indicators": ["Custom", "Momentum"],
        "news_sensitivity": "none"  # Unaffected by news
    },
    "commodities": {
        "primary": "seasonal_patterns",
        "secondary": "supply_demand",
        "indicators": ["Volume", "COT", "Seasonality"],
        "news_sensitivity": "high"
    }
}

# Risk warnings and requirements
REQUIREMENTS = {
    "min_account_balance": 1000,  # USD
    "recommended_balance": 5000,
    "broker_requirements": [
        "Wide spread tolerance",
        "Exotic pairs availability",
        "High leverage support",
        "No dealing desk"
    ],
    "warnings": [
        "Extreme volatility - use minimal lot sizes",
        "Wide spreads require larger profit targets",
        "Political/economic events cause massive moves",
        "Liquidity can disappear during crisis",
        "Not suitable for beginners"
    ]
}