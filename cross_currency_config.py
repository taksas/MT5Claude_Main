#!/usr/bin/env python3
"""
Cross-Currency Pair Configuration for Ultra Trading Engine
High-profit cross pairs with specific trading parameters
"""

# Cross-Currency Trading Configuration
CROSS_CURRENCY_CONFIG = {
    # EUR Crosses - Premium European Pairs
    "EURGBP": {
        "category": "range",
        "daily_range_pips": [40, 80],
        "atr_range": [0.0045, 0.0065],
        "volatility": 6.5,  # Annual %
        "optimal_hours": [7, 16],  # GMT
        "strategy": "range_trading",
        "sl_pips": 30,
        "tp_pips": 45,
        "risk_percent": 1.0,
        "min_confidence": 0.55,
        "key_levels": [0.8300, 0.8500, 0.8700, 0.9000],
        "correlation": {"GBPUSD": -0.65},
        "spread_limit": 2.5,
        "filters": {
            "rsi_range": [30, 70],
            "bb_squeeze": True,
            "avoid_news": True
        }
    },
    
    "EURAUD": {
        "category": "trend",
        "daily_range_pips": [80, 140],
        "atr_range": [0.0090, 0.0120],
        "volatility": 9.2,
        "optimal_hours": [22, 2],  # Sydney-London overlap
        "strategy": "trend_following",
        "sl_pips": 70,
        "tp_pips": 150,
        "risk_percent": 0.8,
        "min_confidence": 0.60,
        "seasonal_strength": ["Q4"],
        "correlation": {"XAUUSD": 0.45},
        "spread_limit": 4.0,
        "filters": {
            "adx_min": 25,
            "ema_cross": [20, 50],
            "trend_strength": 0.7
        }
    },
    
    "EURNZD": {
        "category": "breakout",
        "daily_range_pips": [100, 180],
        "atr_range": [0.0140, 0.0180],
        "volatility": 11.5,
        "optimal_hours": [22, 10],
        "strategy": "volatility_breakout",
        "sl_pips": 90,
        "tp_pips": 200,
        "risk_percent": 0.7,
        "min_confidence": 0.65,
        "correlation": {"dairy_futures": -0.72},
        "spread_limit": 5.0,
        "filters": {
            "volume_surge": 2.0,
            "breakout_confirmation": True,
            "false_breakout_check": True
        }
    },
    
    "EURJPY": {
        "category": "momentum",
        "daily_range_pips": [80, 150],
        "atr_range": [0.95, 1.40],
        "volatility": 10.8,
        "optimal_hours": [0, 9],  # Tokyo-London overlap
        "strategy": "risk_sentiment",
        "sl_pips": 60,
        "tp_pips": 120,
        "risk_percent": 0.8,
        "min_confidence": 0.60,
        "correlation": {"SP500": 0.80, "risk_on": True},
        "spread_limit": 3.0,
        "filters": {
            "stock_correlation": 0.75,
            "macd_confluence": True,
            "session_momentum": True
        }
    },
    
    # GBP Crosses - High Volatility Opportunities
    "GBPJPY": {
        "category": "beast",
        "daily_range_pips": [100, 200],
        "atr_range": [1.20, 1.80],
        "volatility": 13.5,
        "optimal_hours": [7, 16],
        "strategy": "volatility_explosion",
        "sl_pips": 90,
        "tp_pips": 200,
        "risk_percent": 0.5,  # Lower risk due to high volatility
        "min_confidence": 0.70,
        "correlation": {"VIX": -0.85},
        "spread_limit": 4.0,
        "filters": {
            "atr_expansion": True,
            "range_break_hours": 2,
            "max_positions": 1,  # Only one position at a time
            "strict_risk": True
        }
    },
    
    "GBPAUD": {
        "category": "swing",
        "daily_range_pips": [120, 200],
        "atr_range": [0.0150, 0.0200],
        "volatility": 12.2,
        "optimal_hours": [8, 16],
        "strategy": "swing_trading",
        "sl_pips": 110,
        "tp_pips": 250,
        "risk_percent": 0.6,
        "min_confidence": 0.65,
        "correlation": {"copper": -0.60},
        "spread_limit": 5.0,
        "filters": {
            "daily_patterns": True,
            "commodity_divergence": True,
            "uk_data_impact": True
        }
    },
    
    "GBPNZD": {
        "category": "extreme",
        "daily_range_pips": [150, 250],
        "atr_range": [0.0200, 0.0280],
        "volatility": 14.8,
        "optimal_hours": [8, 16],
        "strategy": "news_momentum",
        "sl_pips": 135,
        "tp_pips": 325,
        "risk_percent": 0.4,  # Lowest risk due to extreme volatility
        "min_confidence": 0.75,
        "correlation": {"interest_differential": True},
        "spread_limit": 6.0,
        "filters": {
            "news_required": True,
            "post_news_momentum": True,
            "max_daily_trades": 2
        }
    },
    
    # JPY Commodity Crosses
    "AUDJPY": {
        "category": "carry",
        "daily_range_pips": [60, 120],
        "atr_range": [0.75, 1.10],
        "volatility": 10.2,
        "optimal_hours": [22, 6],  # Sydney session
        "strategy": "carry_trade",
        "sl_pips": 55,
        "tp_pips": 125,
        "risk_percent": 0.8,
        "min_confidence": 0.55,
        "correlation": {"copper": 0.82, "SPX": 0.75},
        "spread_limit": 3.0,
        "filters": {
            "interest_positive": True,
            "china_data": True,
            "trend_alignment": True
        }
    },
    
    "NZDJPY": {
        "category": "yield",
        "daily_range_pips": [60, 100],
        "atr_range": [0.65, 0.95],
        "volatility": 9.8,
        "optimal_hours": [21, 1],
        "strategy": "position_trading",
        "sl_pips": 70,
        "tp_pips": 160,
        "risk_percent": 0.9,
        "min_confidence": 0.55,
        "correlation": {"milk_futures": 0.70},
        "spread_limit": 3.0,
        "seasonal": ["Q2", "Q3"],
        "filters": {
            "weekly_trend": True,
            "carry_differential": 0.5,
            "hold_time_days": 2
        }
    },
    
    "CADJPY": {
        "category": "oil",
        "daily_range_pips": [50, 90],
        "atr_range": [0.60, 0.85],
        "volatility": 8.5,
        "optimal_hours": [13, 21],  # US session
        "strategy": "correlation_trading",
        "sl_pips": 45,
        "tp_pips": 100,
        "risk_percent": 1.0,
        "min_confidence": 0.55,
        "correlation": {"WTI": 0.85},
        "spread_limit": 2.5,
        "filters": {
            "oil_divergence": True,
            "min_oil_move": 1.5,  # %
            "api_wednesdays": True
        }
    },
    
    # Special Mean Reversion Pairs
    "AUDNZD": {
        "category": "mean_reversion",
        "daily_range_pips": [40, 70],
        "atr_range": [0.0050, 0.0070],
        "volatility": 5.8,
        "optimal_hours": [22, 6],
        "strategy": "range_bound",
        "sl_pips": 35,
        "tp_pips": 50,
        "risk_percent": 1.2,  # Higher risk for stable pair
        "min_confidence": 0.50,
        "range": [1.0000, 1.1500],
        "spread_limit": 2.0,
        "filters": {
            "bb_2sd": True,
            "rsi_extreme": [20, 80],
            "range_percent": 0.8
        }
    },
    
    "AUDCAD": {
        "category": "commodity",
        "daily_range_pips": [50, 90],
        "atr_range": [0.0065, 0.0085],
        "volatility": 7.2,
        "optimal_hours": [13, 21],
        "strategy": "commodity_divergence",
        "sl_pips": 50,
        "tp_pips": 110,
        "risk_percent": 0.9,
        "min_confidence": 0.60,
        "correlation": {"gold_oil_ratio": True},
        "spread_limit": 3.0,
        "filters": {
            "weekly_reversal": True,
            "round_numbers": True,
            "commodity_spread": 2.0
        }
    },
    
    # Exotic Crosses
    "GBPCHF": {
        "category": "safe_haven",
        "daily_range_pips": [80, 140],
        "atr_range": [0.0100, 0.0140],
        "volatility": 9.5,
        "optimal_hours": [8, 16],
        "strategy": "risk_sentiment",
        "sl_pips": 80,
        "tp_pips": 170,
        "risk_percent": 0.7,
        "min_confidence": 0.65,
        "correlation": {"SMI": -0.70},
        "spread_limit": 4.0,
        "filters": {
            "brexit_filter": True,
            "snb_watch": True,
            "risk_transitions": True
        }
    },
    
    "EURCHF": {
        "category": "policy",
        "daily_range_pips": [30, 60],
        "atr_range": [0.0035, 0.0055],
        "volatility": 4.5,
        "optimal_hours": [8, 16],
        "strategy": "policy_divergence",
        "sl_pips": 25,
        "tp_pips": 40,
        "risk_percent": 1.0,
        "min_confidence": 0.60,
        "floor": 1.0500,  # Historical SNB floor
        "spread_limit": 2.0,
        "filters": {
            "ecb_snb_divergence": True,
            "low_volatility": True,
            "technical_only": False
        }
    }
}

# Trading Strategy Definitions
CROSS_STRATEGIES = {
    "range_trading": {
        "indicators": ["bollinger_bands", "rsi", "support_resistance"],
        "entry_rules": {
            "bb_touch": True,
            "rsi_divergence": True,
            "sr_bounce": True
        },
        "exit_rules": {
            "bb_middle": True,
            "time_based": 240  # minutes
        }
    },
    
    "trend_following": {
        "indicators": ["ema_cross", "adx", "macd"],
        "entry_rules": {
            "ema_alignment": True,
            "adx_above": 25,
            "macd_confirmation": True
        },
        "exit_rules": {
            "trailing_stop": True,
            "reverse_signal": True
        }
    },
    
    "volatility_breakout": {
        "indicators": ["atr", "volume", "range_break"],
        "entry_rules": {
            "range_hours": 4,
            "volume_spike": 2.0,
            "atr_expansion": True
        },
        "exit_rules": {
            "profit_target": True,
            "volatility_contraction": True
        }
    },
    
    "mean_reversion": {
        "indicators": ["bollinger_bands", "rsi", "zscore"],
        "entry_rules": {
            "bb_2sd": True,
            "rsi_extreme": [25, 75],
            "zscore_extreme": [-2, 2]
        },
        "exit_rules": {
            "mean_touch": True,
            "opposite_extreme": True
        }
    },
    
    "carry_trade": {
        "indicators": ["interest_differential", "trend", "momentum"],
        "entry_rules": {
            "positive_carry": True,
            "trend_alignment": True,
            "no_reversal_pattern": True
        },
        "exit_rules": {
            "weekly_close": True,
            "carry_change": True
        }
    }
}

# Correlation Matrix for Risk Management
CROSS_CORRELATIONS = {
    "EURGBP": {
        "GBPUSD": -0.65,
        "EURUSD": 0.45,
        "GBPJPY": -0.55
    },
    "GBPJPY": {
        "USDJPY": 0.70,
        "EURJPY": 0.85,
        "risk_on": 0.80
    },
    "AUDNZD": {
        "AUDUSD": 0.30,
        "NZDUSD": -0.30,
        "commodity": 0.50
    }
}

# Session Optimization
SESSION_PREFERENCES = {
    "asian": ["AUDJPY", "NZDJPY", "AUDNZD", "EURAUD"],
    "european": ["EURGBP", "EURJPY", "GBPCHF", "EURCHF"],
    "london": ["GBPJPY", "GBPAUD", "GBPNZD", "EURNZD"],
    "newyork": ["CADJPY", "AUDCAD", "USDCAD", "GBPUSD"]
}

# Risk Scaling by Volatility
VOLATILITY_RISK_SCALE = {
    "low": {"volatility_max": 7.0, "risk_multiplier": 1.2},
    "medium": {"volatility_max": 10.0, "risk_multiplier": 1.0},
    "high": {"volatility_max": 13.0, "risk_multiplier": 0.7},
    "extreme": {"volatility_max": 100.0, "risk_multiplier": 0.5}
}

def get_cross_config(symbol: str) -> dict:
    """Get configuration for a specific cross pair"""
    return CROSS_CURRENCY_CONFIG.get(symbol.replace("#", "").replace(".", ""), {})

def get_optimal_crosses_for_session(hour_gmt: int) -> list:
    """Get best cross pairs for current session"""
    optimal_pairs = []
    for symbol, config in CROSS_CURRENCY_CONFIG.items():
        start, end = config["optimal_hours"]
        if start <= hour_gmt <= end or (start > end and (hour_gmt >= start or hour_gmt <= end)):
            optimal_pairs.append(symbol)
    return optimal_pairs

def calculate_cross_portfolio_risk(positions: dict) -> float:
    """Calculate total portfolio risk considering correlations"""
    total_risk = 0
    symbols = list(positions.keys())
    
    # Direct risk
    for symbol, position in positions.items():
        total_risk += position['risk']
    
    # Correlation risk
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            if sym1 in CROSS_CORRELATIONS and sym2 in CROSS_CORRELATIONS[sym1]:
                correlation = CROSS_CORRELATIONS[sym1][sym2]
                risk_adjustment = abs(correlation) * 0.5
                total_risk *= (1 + risk_adjustment)
    
    return total_risk

# Example usage in trading engine
if __name__ == "__main__":
    # Get current optimal crosses
    from datetime import datetime
    current_hour = datetime.utcnow().hour
    optimal_crosses = get_optimal_crosses_for_session(current_hour)
    print(f"Optimal crosses for hour {current_hour} GMT: {optimal_crosses}")
    
    # Get specific pair config
    gbpjpy_config = get_cross_config("GBPJPY")
    print(f"\nGBPJPY Configuration: {gbpjpy_config}")