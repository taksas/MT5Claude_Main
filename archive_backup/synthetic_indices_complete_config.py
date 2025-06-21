# Complete Synthetic Indices Configuration
# All volatility products and synthetic indices for 24/7 trading

# VOLATILITY INDICES - STATISTICAL MARKETS
VOLATILITY_INDICES = {
    "V10": {
        "symbol": "V10",
        "alt_symbols": ["V10(1s)", "Volatility 10"],
        "name": "Volatility 10 Index",
        "volatility_percent": 10,
        "avg_hourly_percent": 0.4,
        "typical_spread": 0.03,
        "risk_per_trade": 0.008,
        "lot_size": 0.01,
        "max_spread": 0.05,
        "best_hours": "24/7",
        "volatility_factor": 1.0,
        "update_frequency": "1 second",
        "min_confidence": 0.65,
        "profit_characteristics": "Steady movements, ideal for beginners"
    },
    "V25": {
        "symbol": "V25",
        "alt_symbols": ["V25(1s)", "Volatility 25"],
        "name": "Volatility 25 Index",
        "volatility_percent": 25,
        "avg_hourly_percent": 1.0,
        "typical_spread": 0.08,
        "risk_per_trade": 0.007,
        "lot_size": 0.005,
        "max_spread": 0.12,
        "best_hours": "24/7",
        "volatility_factor": 2.5,
        "update_frequency": "1 second",
        "min_confidence": 0.68,
        "profit_characteristics": "Moderate swings, good for swing trading"
    },
    "V50": {
        "symbol": "V50",
        "alt_symbols": ["V50(1s)", "Volatility 50"],
        "name": "Volatility 50 Index",
        "volatility_percent": 50,
        "avg_hourly_percent": 2.0,
        "typical_spread": 0.15,
        "risk_per_trade": 0.005,
        "lot_size": 0.003,
        "max_spread": 0.25,
        "best_hours": "24/7",
        "volatility_factor": 5.0,
        "update_frequency": "1 second",
        "min_confidence": 0.70,
        "profit_characteristics": "Active movements, manageable risk"
    },
    "V75": {
        "symbol": "V75",
        "alt_symbols": ["V75(1s)", "Volatility 75"],
        "name": "Volatility 75 Index",
        "volatility_percent": 75,
        "avg_hourly_percent": 3.0,
        "typical_spread": 0.2,
        "risk_per_trade": 0.003,
        "lot_size": 0.001,
        "max_spread": 0.4,
        "best_hours": "24/7",
        "volatility_factor": 7.5,
        "update_frequency": "1 second",
        "min_confidence": 0.75,
        "profit_characteristics": "High volatility, experienced traders"
    },
    "V100": {
        "symbol": "V100",
        "alt_symbols": ["V100(1s)", "Volatility 100"],
        "name": "Volatility 100 Index",
        "volatility_percent": 100,
        "avg_hourly_percent": 4.0,
        "typical_spread": 0.3,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 0.5,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "update_frequency": "1 second",
        "min_confidence": 0.80,
        "profit_characteristics": "Very high volatility, tight risk control needed"
    },
    "V200": {
        "symbol": "V200",
        "alt_symbols": ["V200(1s)", "Volatility 200"],
        "name": "Volatility 200 Index",
        "volatility_percent": 200,
        "avg_hourly_percent": 8.0,
        "typical_spread": 0.5,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 0.8,
        "best_hours": "24/7",
        "volatility_factor": 20.0,
        "update_frequency": "1 second",
        "min_confidence": 0.85,
        "profit_characteristics": "Extreme volatility for scalpers"
    },
    "V300": {
        "symbol": "V300",
        "alt_symbols": ["V300(1s)", "Volatility 300"],
        "name": "Volatility 300 Index",
        "volatility_percent": 300,
        "avg_hourly_percent": 12.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.2,
        "best_hours": "24/7",
        "volatility_factor": 30.0,
        "update_frequency": "1 second",
        "min_confidence": 0.90,
        "profit_characteristics": "Maximum volatility, expert scalpers only"
    }
}

# JUMP INDICES - PRICE JUMP MARKETS
JUMP_INDICES = {
    "JUMP10": {
        "symbol": "Jump 10",
        "name": "Jump 10 Index",
        "jump_probability": 0.10,
        "avg_jump_size": 0.3,
        "typical_spread": 0.03,
        "risk_per_trade": 0.005,
        "lot_size": 0.01,
        "max_spread": 0.05,
        "best_hours": "24/7",
        "volatility_factor": 2.0,
        "min_confidence": 0.70,
        "profit_characteristics": "Jumps every ~10 ticks, breakout friendly"
    },
    "JUMP25": {
        "symbol": "Jump 25",
        "name": "Jump 25 Index",
        "jump_probability": 0.25,
        "avg_jump_size": 0.4,
        "typical_spread": 0.04,
        "risk_per_trade": 0.004,
        "lot_size": 0.01,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 3.0,
        "min_confidence": 0.72,
        "profit_characteristics": "More frequent jumps, higher volatility"
    },
    "JUMP50": {
        "symbol": "Jump 50",
        "name": "Jump 50 Index",
        "jump_probability": 0.50,
        "avg_jump_size": 0.5,
        "typical_spread": 0.05,
        "risk_per_trade": 0.003,
        "lot_size": 0.005,
        "max_spread": 0.08,
        "best_hours": "24/7",
        "volatility_factor": 4.0,
        "min_confidence": 0.75,
        "profit_characteristics": "50% jump chance, momentum trading"
    },
    "JUMP75": {
        "symbol": "Jump 75",
        "name": "Jump 75 Index",
        "jump_probability": 0.75,
        "avg_jump_size": 0.6,
        "typical_spread": 0.06,
        "risk_per_trade": 0.002,
        "lot_size": 0.003,
        "max_spread": 0.10,
        "best_hours": "24/7",
        "volatility_factor": 5.0,
        "min_confidence": 0.78,
        "profit_characteristics": "Near-constant jumping, experienced traders"
    },
    "JUMP100": {
        "symbol": "Jump 100",
        "name": "Jump 100 Index",
        "jump_probability": 1.00,
        "avg_jump_size": 0.8,
        "typical_spread": 0.08,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 0.12,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "min_confidence": 0.82,
        "profit_characteristics": "Continuous jumps, maximum volatility"
    }
}

# STEP INDICES - FIXED STEP MOVEMENTS
STEP_INDICES = {
    "STEP": {
        "symbol": "Step Index",
        "name": "Step Index",
        "step_size": 0.1,
        "movement_probability": 0.5,
        "typical_spread": 0.01,
        "risk_per_trade": 0.006,
        "lot_size": 0.10,
        "max_spread": 0.02,
        "best_hours": "24/7",
        "volatility_factor": 1.5,
        "min_confidence": 0.65,
        "profit_characteristics": "Predictable steps, algorithmic friendly"
    },
    "STEP200": {
        "symbol": "Step 200",
        "name": "Step Index 200",
        "step_size": 0.2,
        "movement_probability": 0.5,
        "typical_spread": 0.02,
        "risk_per_trade": 0.005,
        "lot_size": 0.05,
        "max_spread": 0.03,
        "best_hours": "24/7",
        "volatility_factor": 2.0,
        "min_confidence": 0.68,
        "profit_characteristics": "Larger steps, higher profit per move"
    },
    "STEP300": {
        "symbol": "Step 300",
        "name": "Step Index 300",
        "step_size": 0.3,
        "movement_probability": 0.5,
        "typical_spread": 0.03,
        "risk_per_trade": 0.004,
        "lot_size": 0.03,
        "max_spread": 0.04,
        "best_hours": "24/7",
        "volatility_factor": 2.5,
        "min_confidence": 0.70,
        "profit_characteristics": "Medium steps, balanced risk-reward"
    },
    "STEP400": {
        "symbol": "Step 400",
        "name": "Step Index 400",
        "step_size": 0.4,
        "movement_probability": 0.5,
        "typical_spread": 0.04,
        "risk_per_trade": 0.003,
        "lot_size": 0.02,
        "max_spread": 0.05,
        "best_hours": "24/7",
        "volatility_factor": 3.0,
        "min_confidence": 0.72,
        "profit_characteristics": "Large steps, higher volatility"
    },
    "STEP500": {
        "symbol": "Step 500",
        "name": "Step Index 500",
        "step_size": 0.5,
        "movement_probability": 0.5,
        "typical_spread": 0.05,
        "risk_per_trade": 0.002,
        "lot_size": 0.01,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 3.5,
        "min_confidence": 0.75,
        "profit_characteristics": "Maximum step size, highest profit potential"
    },
    "MULTISTEP2": {
        "symbol": "Multi Step 2",
        "name": "Multi Step 2 Index",
        "step_sizes": [0.1, 0.3],
        "movement_probability": 0.5,
        "typical_spread": 0.03,
        "risk_per_trade": 0.004,
        "lot_size": 0.03,
        "max_spread": 0.05,
        "best_hours": "24/7",
        "volatility_factor": 2.2,
        "min_confidence": 0.70,
        "profit_characteristics": "Variable steps, more complex patterns"
    },
    "MULTISTEP4": {
        "symbol": "Multi Step 4",
        "name": "Multi Step 4 Index",
        "step_sizes": [0.1, 0.2, 0.3, 0.4],
        "movement_probability": 0.5,
        "typical_spread": 0.04,
        "risk_per_trade": 0.003,
        "lot_size": 0.02,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 2.8,
        "min_confidence": 0.72,
        "profit_characteristics": "Most complex step pattern, varied opportunities"
    }
}

# CRASH INDICES - DOWNWARD SPIKE MARKETS
CRASH_INDICES = {
    "CRASH300": {
        "symbol": "Crash 300",
        "name": "Crash 300 Index",
        "avg_ticks_between_crashes": 300,
        "avg_crash_size_percent": 3.0,
        "typical_spread": 0.6,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 8.0,
        "min_confidence": 0.82,
        "profit_characteristics": "Frequent crashes, high opportunity"
    },
    "CRASH500": {
        "symbol": "Crash 500",
        "name": "Crash 500 Index",
        "avg_ticks_between_crashes": 500,
        "avg_crash_size_percent": 4.0,
        "typical_spread": 0.7,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 1.2,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "min_confidence": 0.85,
        "profit_characteristics": "Moderate frequency, larger crashes"
    },
    "CRASH600": {
        "symbol": "Crash 600",
        "name": "Crash 600 Index",
        "avg_ticks_between_crashes": 600,
        "avg_crash_size_percent": 5.0,
        "typical_spread": 0.7,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.3,
        "best_hours": "24/7",
        "volatility_factor": 11.0,
        "min_confidence": 0.86,
        "profit_characteristics": "Less frequent, bigger crashes"
    },
    "CRASH900": {
        "symbol": "Crash 900",
        "name": "Crash 900 Index",
        "avg_ticks_between_crashes": 900,
        "avg_crash_size_percent": 7.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.4,
        "best_hours": "24/7",
        "volatility_factor": 12.0,
        "min_confidence": 0.88,
        "profit_characteristics": "Rare but massive crashes"
    },
    "CRASH1000": {
        "symbol": "Crash 1000",
        "name": "Crash 1000 Index",
        "avg_ticks_between_crashes": 1000,
        "avg_crash_size_percent": 8.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 15.0,
        "min_confidence": 0.90,
        "profit_characteristics": "Maximum crash size, patient traders"
    }
}

# BOOM INDICES - UPWARD SPIKE MARKETS
BOOM_INDICES = {
    "BOOM300": {
        "symbol": "Boom 300",
        "name": "Boom 300 Index",
        "avg_ticks_between_booms": 300,
        "avg_boom_size_percent": 3.0,
        "typical_spread": 0.6,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 8.0,
        "min_confidence": 0.82,
        "profit_characteristics": "Frequent spikes upward"
    },
    "BOOM500": {
        "symbol": "Boom 500",
        "name": "Boom 500 Index",
        "avg_ticks_between_booms": 500,
        "avg_boom_size_percent": 4.0,
        "typical_spread": 0.7,
        "risk_per_trade": 0.002,
        "lot_size": 0.001,
        "max_spread": 1.2,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "min_confidence": 0.85,
        "profit_characteristics": "Balanced frequency and size"
    },
    "BOOM600": {
        "symbol": "Boom 600",
        "name": "Boom 600 Index",
        "avg_ticks_between_booms": 600,
        "avg_boom_size_percent": 5.0,
        "typical_spread": 0.7,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.3,
        "best_hours": "24/7",
        "volatility_factor": 11.0,
        "min_confidence": 0.86,
        "profit_characteristics": "Less frequent, bigger booms"
    },
    "BOOM900": {
        "symbol": "Boom 900",
        "name": "Boom 900 Index",
        "avg_ticks_between_booms": 900,
        "avg_boom_size_percent": 7.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.4,
        "best_hours": "24/7",
        "volatility_factor": 12.0,
        "min_confidence": 0.88,
        "profit_characteristics": "Rare but significant spikes"
    },
    "BOOM1000": {
        "symbol": "Boom 1000",
        "name": "Boom 1000 Index",
        "avg_ticks_between_booms": 1000,
        "avg_boom_size_percent": 8.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.001,
        "lot_size": 0.001,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 15.0,
        "min_confidence": 0.90,
        "profit_characteristics": "Maximum spike size"
    }
}

# RANGE BREAK INDICES - BREAKOUT MARKETS
RANGE_BREAK_INDICES = {
    "RB100": {
        "symbol": "RB100",
        "name": "Range Break 100 Index",
        "avg_touches_before_break": 100,
        "typical_break_size": 2.0,
        "typical_spread": 0.03,
        "risk_per_trade": 0.004,
        "lot_size": 0.01,
        "max_spread": 0.05,
        "best_hours": "24/7",
        "volatility_factor": 3.0,
        "min_confidence": 0.72,
        "profit_characteristics": "Frequent breakouts, range + breakout strategies"
    },
    "RB200": {
        "symbol": "RB200",
        "name": "Range Break 200 Index",
        "avg_touches_before_break": 200,
        "typical_break_size": 3.0,
        "typical_spread": 0.04,
        "risk_per_trade": 0.003,
        "lot_size": 0.01,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 3.5,
        "min_confidence": 0.75,
        "profit_characteristics": "More stable ranges, larger breakouts"
    }
}

# VOLATILITY ETFs/ETNs (MARKET HOURS ONLY)
VOLATILITY_ETFS = {
    "VXX": {
        "symbol": "VXX",
        "name": "iPath S&P 500 VIX Short-Term Futures ETN",
        "tracking": "VIX futures 1-2 month",
        "typical_spread": 0.02,
        "risk_per_trade": 0.003,
        "lot_size": 1.0,
        "max_spread": 0.05,
        "best_hours": "13:30-20:00",
        "market_hours_only": True,
        "volatility_factor": 4.0,
        "inverse_sp500": True,
        "min_confidence": 0.75,
        "profit_characteristics": "15% gain on 5% S&P drop"
    },
    "UVXY": {
        "symbol": "UVXY",
        "name": "ProShares Ultra VIX Short-Term Futures ETF",
        "tracking": "1.5x VIX futures",
        "typical_spread": 0.03,
        "risk_per_trade": 0.002,
        "lot_size": 0.5,
        "max_spread": 0.08,
        "best_hours": "13:30-20:00",
        "market_hours_only": True,
        "volatility_factor": 6.0,
        "leveraged": 1.5,
        "min_confidence": 0.78,
        "profit_characteristics": "Amplified VIX moves, extreme volatility"
    },
    "SVXY": {
        "symbol": "SVXY",
        "name": "ProShares Short VIX Short-Term Futures ETF",
        "tracking": "-0.5x inverse VIX",
        "typical_spread": 0.05,
        "risk_per_trade": 0.004,
        "lot_size": 0.5,
        "max_spread": 0.10,
        "best_hours": "13:30-20:00",
        "market_hours_only": True,
        "volatility_factor": 3.0,
        "inverse_vix": True,
        "min_confidence": 0.72,
        "profit_characteristics": "Profits from declining volatility"
    },
    "VIXY": {
        "symbol": "VIXY",
        "name": "ProShares VIX Short-Term Futures ETF",
        "tracking": "1x VIX futures",
        "typical_spread": 0.03,
        "risk_per_trade": 0.003,
        "lot_size": 0.8,
        "max_spread": 0.06,
        "best_hours": "13:30-20:00",
        "market_hours_only": True,
        "volatility_factor": 4.0,
        "min_confidence": 0.74,
        "profit_characteristics": "Direct VIX exposure, volatility hedge"
    }
}

# Combine all synthetic indices
ALL_SYNTHETIC_INDICES = {
    **VOLATILITY_INDICES,
    **JUMP_INDICES,
    **STEP_INDICES,
    **CRASH_INDICES,
    **BOOM_INDICES,
    **RANGE_BREAK_INDICES,
    **VOLATILITY_ETFS
}

# Trading strategies for synthetic indices
SYNTHETIC_STRATEGIES = {
    "scalping": {
        "best_indices": ["V200", "V300", "Jump 75", "Jump 100"],
        "timeframe": "M1, M5",
        "approach": "Quick in-out trades on momentum",
        "risk_per_trade": 0.001,
        "target_pips": 5-10
    },
    "range_trading": {
        "best_indices": ["V10", "V25", "Step indices", "RB100", "RB200"],
        "timeframe": "M15, H1",
        "approach": "Trade bounces at support/resistance",
        "risk_per_trade": 0.003,
        "target_pips": 20-50
    },
    "crash_boom_trading": {
        "best_indices": ["All Crash/Boom indices"],
        "timeframe": "M5, M15",
        "approach": "Trade recovery after crashes/booms",
        "risk_per_trade": 0.001,
        "target_percent": "2-5%"
    },
    "trend_following": {
        "best_indices": ["V50", "V75", "V100"],
        "timeframe": "H1, H4",
        "approach": "Follow strong directional moves",
        "risk_per_trade": 0.002,
        "target_pips": 50-100
    }
}

# Risk management for synthetics
SYNTHETIC_RISK_MANAGEMENT = {
    "high_volatility": {
        "indices": ["V200", "V300", "Crash/Boom 1000"],
        "max_positions": 1,
        "max_risk": 0.001,
        "stop_type": "tight_trailing"
    },
    "medium_volatility": {
        "indices": ["V50", "V75", "V100", "Jump indices"],
        "max_positions": 2,
        "max_risk": 0.003,
        "stop_type": "standard"
    },
    "low_volatility": {
        "indices": ["V10", "V25", "Step indices"],
        "max_positions": 3,
        "max_risk": 0.005,
        "stop_type": "wider_stops"
    }
}

# Special characteristics
SYNTHETIC_CHARACTERISTICS = {
    "24_7_trading": {
        "all_synthetics_except": ["VXX", "UVXY", "SVXY", "VIXY"],
        "advantage": "Trade anytime, no gaps",
        "consideration": "Manage fatigue, set schedules"
    },
    "algorithm_based": {
        "all_synthetics": True,
        "advantage": "No manipulation, pure statistics",
        "consideration": "Technical analysis more reliable"
    },
    "no_fundamentals": {
        "all_synthetics": True,
        "advantage": "No news surprises",
        "consideration": "Pure price action trading"
    }
}

def get_synthetics_by_volatility(min_volatility):
    """Get synthetic indices by minimum volatility"""
    high_vol = {}
    for symbol, config in ALL_SYNTHETIC_INDICES.items():
        vol = config.get("volatility_percent", 0) or config.get("volatility_factor", 0) * 10
        if vol >= min_volatility:
            high_vol[symbol] = config
    return high_vol

def get_24_7_synthetics():
    """Get all 24/7 tradeable synthetic indices"""
    return {k: v for k, v in ALL_SYNTHETIC_INDICES.items() 
            if not v.get("market_hours_only", False)}

def get_crash_boom_indices():
    """Get all crash and boom indices"""
    return {**CRASH_INDICES, **BOOM_INDICES}