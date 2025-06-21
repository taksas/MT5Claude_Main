# Cross-Currency Pairs Configuration for MT5 Trading
# Selected based on profitability, liquidity, and automated trading suitability

CROSS_CURRENCY_PAIRS = {
    # EUR Crosses - Excellent for various strategies
    "EURGBP": {
        "symbol": "EURGBP",
        "strategy": "mean_reversion",
        "avg_daily_range": 70,  # pips
        "typical_spread": 2.0,
        "best_hours": [(8, 16)],  # GMT
        "max_spread": 3.0,
        "risk_multiplier": 1.2,  # Lower volatility = slightly higher risk
        "win_rate_target": 0.70,
        "take_profit": 25,
        "stop_loss": 20,
        "notes": "Excellent range-bound behavior, high win rate"
    },
    
    "EURJPY": {
        "symbol": "EURJPY",
        "strategy": "trend_following",
        "avg_daily_range": 120,
        "typical_spread": 2.5,
        "best_hours": [(7, 16), (0, 2)],  # European session and Tokyo overlap
        "max_spread": 4.0,
        "risk_multiplier": 1.0,
        "win_rate_target": 0.68,
        "take_profit": 40,
        "stop_loss": 30,
        "notes": "Best risk-on/risk-off proxy, follows equities"
    },
    
    "EURAUD": {
        "symbol": "EURAUD",
        "strategy": "breakout",
        "avg_daily_range": 140,
        "typical_spread": 3.0,
        "best_hours": [(0, 2), (7, 9)],
        "max_spread": 5.0,
        "risk_multiplier": 0.8,  # Higher volatility = lower risk
        "win_rate_target": 0.60,
        "take_profit": 60,
        "stop_loss": 40,
        "notes": "Large ranges, clear trends, weekend gaps"
    },
    
    "EURNZD": {
        "symbol": "EURNZD",
        "strategy": "breakout",
        "avg_daily_range": 160,
        "typical_spread": 4.0,
        "best_hours": [(22, 6), (7, 9)],  # Asian session focus
        "max_spread": 6.0,
        "risk_multiplier": 0.7,
        "win_rate_target": 0.58,
        "take_profit": 80,
        "stop_loss": 50,
        "notes": "Highest EUR cross volatility, strong trends"
    },
    
    # GBP Crosses - High volatility, high reward
    "GBPJPY": {
        "symbol": "GBPJPY",
        "strategy": "momentum",
        "avg_daily_range": 170,
        "typical_spread": 2.8,
        "best_hours": [(8, 16)],
        "max_spread": 4.5,
        "risk_multiplier": 0.6,  # "The Dragon" - manage risk carefully
        "win_rate_target": 0.65,
        "take_profit": 70,
        "stop_loss": 50,
        "notes": "Extreme volatility, excellent momentum"
    },
    
    "GBPAUD": {
        "symbol": "GBPAUD",
        "strategy": "trend_following",
        "avg_daily_range": 190,
        "typical_spread": 4.0,
        "best_hours": [(22, 0), (8, 10)],
        "max_spread": 6.0,
        "risk_multiplier": 0.5,
        "win_rate_target": 0.64,
        "take_profit": 90,
        "stop_loss": 60,
        "notes": "Massive ranges, strong trends"
    },
    
    # JPY Crosses - Risk sentiment plays
    "AUDJPY": {
        "symbol": "AUDJPY",
        "strategy": "carry_trade",
        "avg_daily_range": 100,
        "typical_spread": 2.5,
        "best_hours": [(0, 9)],  # Tokyo session
        "max_spread": 4.0,
        "risk_multiplier": 1.0,
        "win_rate_target": 0.66,
        "take_profit": 35,
        "stop_loss": 25,
        "notes": "Commodity/risk correlation, carry potential"
    },
    
    "NZDJPY": {
        "symbol": "NZDJPY",
        "strategy": "trend_following",
        "avg_daily_range": 110,
        "typical_spread": 2.8,
        "best_hours": [(22, 2), (0, 9)],
        "max_spread": 4.5,
        "risk_multiplier": 0.9,
        "win_rate_target": 0.62,
        "take_profit": 45,
        "stop_loss": 35,
        "notes": "Higher volatility than AUDJPY"
    },
    
    # Special Pairs - Unique opportunities
    "AUDNZD": {
        "symbol": "AUDNZD",
        "strategy": "mean_reversion",
        "avg_daily_range": 75,
        "typical_spread": 3.0,
        "best_hours": [(22, 7)],  # Sydney session
        "max_spread": 5.0,
        "risk_multiplier": 1.3,
        "win_rate_target": 0.72,
        "take_profit": 30,
        "stop_loss": 25,
        "notes": "Best mean reversion pair, 80% range-bound"
    },
    
    "EURCHF": {
        "symbol": "EURCHF",
        "strategy": "grid_trading",
        "avg_daily_range": 50,
        "typical_spread": 2.0,
        "best_hours": [(7, 16)],
        "max_spread": 3.5,
        "risk_multiplier": 1.5,  # Very stable
        "win_rate_target": 0.68,
        "take_profit": 20,
        "stop_loss": 15,
        "notes": "SNB influence, extremely stable"
    }
}

# Strategy-specific settings
STRATEGY_SETTINGS = {
    "mean_reversion": {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "min_range_pips": 50
    },
    
    "trend_following": {
        "ema_fast": 20,
        "ema_slow": 50,
        "adx_period": 14,
        "adx_threshold": 25,
        "momentum_period": 10
    },
    
    "breakout": {
        "lookback_periods": 20,
        "atr_multiplier": 1.5,
        "volume_confirmation": True,
        "min_range_break": 1.2  # Times average range
    },
    
    "momentum": {
        "rsi_period": 9,
        "rsi_momentum": 60,  # Above for long, below 40 for short
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9
    },
    
    "carry_trade": {
        "min_interest_differential": 1.0,  # Percent
        "trend_filter": True,
        "ema_period": 50,
        "hold_days": 5  # Minimum hold period
    },
    
    "grid_trading": {
        "grid_size": 20,  # Pips
        "max_positions": 5,
        "tp_per_level": 20,
        "martingale": False,  # Safer without
        "range_bound_check": True
    }
}

# Risk management overrides for cross currencies
CROSS_RISK_MANAGEMENT = {
    "max_correlation_exposure": 0.7,  # Max correlation between open positions
    "max_currency_exposure": 2,  # Max positions with same currency
    "volatility_adjustment": True,  # Adjust position size by volatility
    "news_filter": True,  # Avoid trading during high-impact news
    "session_overlap_bonus": 1.2,  # Increase confidence during overlaps
    "weekend_gap_protection": True,  # Close positions before weekend
    "max_spread_multiplier": 1.5,  # During low liquidity
}

# Correlation matrix for risk management
CORRELATION_MATRIX = {
    "EURGBP": {"GBPUSD": -0.7, "EURUSD": 0.5},
    "EURJPY": {"USDJPY": 0.6, "RISK_ON": 0.8},
    "GBPJPY": {"RISK_ON": 0.85, "EURJPY": 0.7},
    "AUDJPY": {"COMMODITIES": 0.8, "NZDJPY": 0.9},
    "AUDNZD": {"AUDUSD": 0.3, "NZDUSD": -0.3}
}

# Time zone conversions (to server time)
TIME_ZONES = {
    "GMT": 0,
    "Tokyo": 9,
    "Sydney": 11,
    "London": 0,  # GMT
    "NewYork": -5
}

# News events to avoid (high impact)
NEWS_BLACKOUT = {
    "ECB": {"days": ["THU"], "time": (12, 45)},  # ECB meetings
    "BOE": {"days": ["THU"], "time": (12, 0)},   # BOE meetings
    "RBA": {"days": ["TUE"], "time": (4, 30)},   # RBA meetings
    "RBNZ": {"days": ["WED"], "time": (2, 0)},   # RBNZ meetings
    "BOJ": {"days": ["FRI"], "time": (3, 0)},    # BOJ meetings
}

def get_optimal_pairs_for_session(current_hour_gmt):
    """Return the best pairs to trade for the current session"""
    optimal_pairs = []
    
    for pair, config in CROSS_CURRENCY_PAIRS.items():
        for time_range in config["best_hours"]:
            if time_range[0] <= current_hour_gmt < time_range[1]:
                optimal_pairs.append(pair)
                break
    
    return optimal_pairs

def calculate_position_size(pair_config, base_lot_size=0.01):
    """Calculate position size based on pair volatility and risk multiplier"""
    return base_lot_size * pair_config["risk_multiplier"]

def check_correlation_limit(open_positions, new_pair):
    """Check if adding new pair would exceed correlation limits"""
    if new_pair not in CORRELATION_MATRIX:
        return True
    
    total_correlation = 0
    for pos in open_positions:
        if pos in CORRELATION_MATRIX.get(new_pair, {}):
            total_correlation += abs(CORRELATION_MATRIX[new_pair][pos])
    
    return total_correlation < CROSS_RISK_MANAGEMENT["max_correlation_exposure"]