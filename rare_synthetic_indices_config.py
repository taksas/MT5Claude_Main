"""
Rare Synthetic Indices Configuration
Exotic and specialized synthetic products for MT5
Focus on DEX, Drift Switching, and other rare indices
"""

# DEX Indices - Directional Extreme Indices
DEX_INDICES = {
    "DEX600UP": {
        "symbol": "DEX 600 UP",
        "alt_symbols": ["DEX600UP", "DEX_600_UP"],
        "name": "DEX 600 UP Index",
        "spike_interval_seconds": 600,
        "spike_direction": "up",
        "avg_spike_size_percent": 5.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.002,
        "lot_size": 0.01,
        "max_spread": 1.2,
        "best_hours": "24/7",
        "volatility_factor": 8.0,
        "min_confidence": 0.85,
        "entry_window_seconds": 60,
        "profit_characteristics": "Predictable upward spikes every 10 minutes"
    },
    "DEX900UP": {
        "symbol": "DEX 900 UP",
        "alt_symbols": ["DEX900UP", "DEX_900_UP"],
        "name": "DEX 900 UP Index",
        "spike_interval_seconds": 900,
        "spike_direction": "up",
        "avg_spike_size_percent": 6.0,
        "typical_spread": 0.9,
        "risk_per_trade": 0.002,
        "lot_size": 0.01,
        "max_spread": 1.3,
        "best_hours": "24/7",
        "volatility_factor": 9.0,
        "min_confidence": 0.86,
        "entry_window_seconds": 90,
        "profit_characteristics": "Larger spikes every 15 minutes"
    },
    "DEX1500UP": {
        "symbol": "DEX 1500 UP",
        "alt_symbols": ["DEX1500UP", "DEX_1500_UP"],
        "name": "DEX 1500 UP Index",
        "spike_interval_seconds": 1500,
        "spike_direction": "up",
        "avg_spike_size_percent": 8.0,
        "typical_spread": 1.0,
        "risk_per_trade": 0.001,
        "lot_size": 0.01,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "min_confidence": 0.88,
        "entry_window_seconds": 120,
        "profit_characteristics": "Maximum spikes every 25 minutes"
    },
    "DEX600DOWN": {
        "symbol": "DEX 600 DOWN",
        "alt_symbols": ["DEX600DOWN", "DEX_600_DOWN"],
        "name": "DEX 600 DOWN Index",
        "spike_interval_seconds": 600,
        "spike_direction": "down",
        "avg_spike_size_percent": 5.0,
        "typical_spread": 0.8,
        "risk_per_trade": 0.002,
        "lot_size": 0.01,
        "max_spread": 1.2,
        "best_hours": "24/7",
        "volatility_factor": 8.0,
        "min_confidence": 0.85,
        "entry_window_seconds": 60,
        "profit_characteristics": "Predictable downward drops every 10 minutes"
    },
    "DEX900DOWN": {
        "symbol": "DEX 900 DOWN",
        "alt_symbols": ["DEX900DOWN", "DEX_900_DOWN"],
        "name": "DEX 900 DOWN Index",
        "spike_interval_seconds": 900,
        "spike_direction": "down",
        "avg_spike_size_percent": 6.0,
        "typical_spread": 0.9,
        "risk_per_trade": 0.002,
        "lot_size": 0.01,
        "max_spread": 1.3,
        "best_hours": "24/7",
        "volatility_factor": 9.0,
        "min_confidence": 0.86,
        "entry_window_seconds": 90,
        "profit_characteristics": "Larger drops every 15 minutes"
    },
    "DEX1500DOWN": {
        "symbol": "DEX 1500 DOWN",
        "alt_symbols": ["DEX1500DOWN", "DEX_1500_DOWN"],
        "name": "DEX 1500 DOWN Index",
        "spike_interval_seconds": 1500,
        "spike_direction": "down",
        "avg_spike_size_percent": 8.0,
        "typical_spread": 1.0,
        "risk_per_trade": 0.001,
        "lot_size": 0.01,
        "max_spread": 1.5,
        "best_hours": "24/7",
        "volatility_factor": 10.0,
        "min_confidence": 0.88,
        "entry_window_seconds": 120,
        "profit_characteristics": "Maximum drops every 25 minutes"
    }
}

# Drift Switching Indices
DRIFT_SWITCHING_INDICES = {
    "DS10": {
        "symbol": "Drift Switch 10",
        "alt_symbols": ["DS10", "DRIFT10"],
        "name": "Drift Switch 10 Index",
        "phase_duration_minutes": 10,
        "phases": ["upward", "downward", "sideways"],
        "avg_phase_movement_percent": 3.0,
        "typical_spread": 0.5,
        "risk_per_trade": 0.003,
        "lot_size": 0.02,
        "max_spread": 0.8,
        "best_hours": "24/7",
        "volatility_factor": 4.0,
        "min_confidence": 0.75,
        "transition_duration_seconds": 30,
        "profit_characteristics": "Quick phase changes, scalping friendly"
    },
    "DS20": {
        "symbol": "Drift Switch 20",
        "alt_symbols": ["DS20", "DRIFT20"],
        "name": "Drift Switch 20 Index",
        "phase_duration_minutes": 20,
        "phases": ["upward", "downward", "sideways"],
        "avg_phase_movement_percent": 5.0,
        "typical_spread": 0.6,
        "risk_per_trade": 0.003,
        "lot_size": 0.02,
        "max_spread": 0.9,
        "best_hours": "24/7",
        "volatility_factor": 5.0,
        "min_confidence": 0.78,
        "transition_duration_seconds": 45,
        "profit_characteristics": "Balanced phase duration, trend following"
    },
    "DS30": {
        "symbol": "Drift Switch 30",
        "alt_symbols": ["DS30", "DRIFT30"],
        "name": "Drift Switch 30 Index",
        "phase_duration_minutes": 30,
        "phases": ["upward", "downward", "sideways"],
        "avg_phase_movement_percent": 8.0,
        "typical_spread": 0.7,
        "risk_per_trade": 0.002,
        "lot_size": 0.015,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "min_confidence": 0.80,
        "transition_duration_seconds": 60,
        "profit_characteristics": "Longer trends, position trading"
    }
}

# Hybrid Indices - Complex market behaviors
HYBRID_INDICES = {
    "HYBRID1": {
        "symbol": "Hybrid Index 1",
        "alt_symbols": ["HYB1", "HYBRID_1"],
        "name": "Hybrid Index Type 1",
        "behavior_mix": ["volatility", "trend", "range"],
        "typical_spread": 0.6,
        "risk_per_trade": 0.003,
        "lot_size": 0.02,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 5.0,
        "min_confidence": 0.76,
        "profit_characteristics": "Mixed market conditions, adaptive strategies"
    },
    "HYBRID2": {
        "symbol": "Hybrid Index 2",
        "alt_symbols": ["HYB2", "HYBRID_2"],
        "name": "Hybrid Index Type 2",
        "behavior_mix": ["jump", "drift", "volatility"],
        "typical_spread": 0.7,
        "risk_per_trade": 0.002,
        "lot_size": 0.015,
        "max_spread": 1.1,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "min_confidence": 0.78,
        "profit_characteristics": "Complex patterns, experienced traders"
    }
}

# Skew Step Indices - Asymmetric movements
SKEW_STEP_INDICES = {
    "SKEWSTEP": {
        "symbol": "Skew Step Index",
        "alt_symbols": ["SSTEP", "SKEW_STEP"],
        "name": "Skew Step Index",
        "up_step_size": 0.3,
        "down_step_size": 0.2,
        "up_probability": 0.55,
        "down_probability": 0.45,
        "typical_spread": 0.04,
        "risk_per_trade": 0.004,
        "lot_size": 0.05,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 3.0,
        "min_confidence": 0.72,
        "profit_characteristics": "Bullish bias, trend strategies"
    },
    "SKEWSTEP2": {
        "symbol": "Skew Step Index 2",
        "alt_symbols": ["SSTEP2", "SKEW_STEP_2"],
        "name": "Skew Step Index Type 2",
        "up_step_size": 0.2,
        "down_step_size": 0.3,
        "up_probability": 0.45,
        "down_probability": 0.55,
        "typical_spread": 0.04,
        "risk_per_trade": 0.004,
        "lot_size": 0.05,
        "max_spread": 0.06,
        "best_hours": "24/7",
        "volatility_factor": 3.0,
        "min_confidence": 0.72,
        "profit_characteristics": "Bearish bias, counter-trend opportunities"
    }
}

# Cryptocurrency Volatility Indices
CRYPTO_VOLATILITY_INDICES = {
    "CVI": {
        "symbol": "CVI",
        "name": "Crypto Volatility Index",
        "tracking": "Overall crypto market volatility",
        "typical_spread": 0.5,
        "risk_per_trade": 0.003,
        "lot_size": 0.1,
        "max_spread": 1.0,
        "best_hours": "24/7",
        "volatility_factor": 8.0,
        "correlation": "inverse_crypto_market",
        "min_confidence": 0.80,
        "profit_characteristics": "Spikes during crypto crashes, 20%+ moves"
    },
    "BVIV": {
        "symbol": "BVIV",
        "name": "Bitcoin Implied Volatility",
        "tracking": "30-day BTC implied volatility",
        "typical_spread": 0.3,
        "risk_per_trade": 0.003,
        "lot_size": 0.2,
        "max_spread": 0.6,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "min_confidence": 0.78,
        "profit_characteristics": "BTC volatility exposure"
    },
    "EVIV": {
        "symbol": "EVIV",
        "name": "Ethereum Implied Volatility",
        "tracking": "30-day ETH implied volatility",
        "typical_spread": 0.4,
        "risk_per_trade": 0.003,
        "lot_size": 0.2,
        "max_spread": 0.7,
        "best_hours": "24/7",
        "volatility_factor": 7.0,
        "min_confidence": 0.78,
        "profit_characteristics": "ETH volatility exposure"
    }
}

# Basket Indices
BASKET_INDICES = {
    "TECHBASKET": {
        "symbol": "TECH_BASKET",
        "name": "Technology Volatility Basket",
        "components": ["NASDAQ", "Tech stocks volatility"],
        "typical_spread": 1.0,
        "risk_per_trade": 0.002,
        "lot_size": 0.1,
        "max_spread": 2.0,
        "best_hours": "13:30-20:00",
        "market_hours_only": True,
        "volatility_factor": 5.0,
        "min_confidence": 0.75,
        "profit_characteristics": "Tech sector volatility plays"
    },
    "COMMODITYBASKET": {
        "symbol": "COMM_BASKET",
        "name": "Commodity Volatility Basket",
        "components": ["Gold", "Oil", "Copper volatility"],
        "typical_spread": 1.2,
        "risk_per_trade": 0.002,
        "lot_size": 0.1,
        "max_spread": 2.5,
        "best_hours": "24/7",
        "volatility_factor": 6.0,
        "min_confidence": 0.76,
        "profit_characteristics": "Commodity sector volatility"
    }
}

# Combine all rare indices
ALL_RARE_INDICES = {
    **DEX_INDICES,
    **DRIFT_SWITCHING_INDICES,
    **HYBRID_INDICES,
    **SKEW_STEP_INDICES,
    **CRYPTO_VOLATILITY_INDICES,
    **BASKET_INDICES
}

# Trading strategies for rare indices
RARE_INDEX_STRATEGIES = {
    "dex_timing": {
        "indices": list(DEX_INDICES.keys()),
        "approach": "Enter 30-60 seconds before spike",
        "exit": "Trail stop after spike begins",
        "risk_reward": "1:5 minimum",
        "win_rate_target": 0.70,
        "daily_profit_target": "50-100%"
    },
    "drift_phase_trading": {
        "indices": list(DRIFT_SWITCHING_INDICES.keys()),
        "approach": "Enter at phase transition",
        "exit": "80% of phase duration",
        "risk_reward": "1:3",
        "win_rate_target": 0.65,
        "daily_profit_target": "30-60%"
    },
    "skew_momentum": {
        "indices": list(SKEW_STEP_INDICES.keys()),
        "approach": "Trade with the bias direction",
        "exit": "Opposite bias signals",
        "risk_reward": "1:2",
        "win_rate_target": 0.60,
        "daily_profit_target": "20-40%"
    },
    "volatility_spike": {
        "indices": list(CRYPTO_VOLATILITY_INDICES.keys()),
        "approach": "Long on market fear signals",
        "exit": "Volatility normalization",
        "risk_reward": "1:4",
        "win_rate_target": 0.55,
        "daily_profit_target": "40-80%"
    }
}

# Risk management for rare indices
RARE_INDEX_RISK_MANAGEMENT = {
    "extreme_indices": {
        "indices": ["DEX1500UP", "DEX1500DOWN", "CVI"],
        "max_positions": 1,
        "max_risk": 0.001,
        "stop_type": "time_based",
        "max_holding_time": 300  # seconds
    },
    "timed_indices": {
        "indices": list(DEX_INDICES.keys()) + list(DRIFT_SWITCHING_INDICES.keys()),
        "max_positions": 2,
        "max_risk": 0.002,
        "stop_type": "hybrid_time_price",
        "entry_timing_critical": True
    },
    "volatility_indices": {
        "indices": list(CRYPTO_VOLATILITY_INDICES.keys()),
        "max_positions": 1,
        "max_risk": 0.003,
        "stop_type": "volatility_adjusted",
        "correlation_check": True
    }
}

# Special features and characteristics
RARE_INDEX_FEATURES = {
    "dex_indices": {
        "unique_feature": "Predictable spike timing",
        "best_use": "Timer-based entries",
        "avoid": "Trading between spikes",
        "profit_potential": "50-100% daily"
    },
    "drift_switching": {
        "unique_feature": "Phase-based trending",
        "best_use": "Trend following in phases",
        "avoid": "Phase transitions",
        "profit_potential": "30-60% daily"
    },
    "hybrid_indices": {
        "unique_feature": "Mixed market behaviors",
        "best_use": "Adaptive strategies",
        "avoid": "Fixed strategy approaches",
        "profit_potential": "20-40% daily"
    },
    "crypto_volatility": {
        "unique_feature": "Crypto fear gauge",
        "best_use": "Market crash hedging",
        "avoid": "Calm market periods",
        "profit_potential": "100%+ on events"
    }
}

# Broker and platform requirements
PLATFORM_REQUIREMENTS = {
    "deriv_exclusives": {
        "indices": list(DEX_INDICES.keys()) + list(DRIFT_SWITCHING_INDICES.keys()),
        "platform": "Deriv MT5 only",
        "account_type": "Synthetic Indices",
        "min_deposit": 10,
        "leverage": "up to 1:1000",
        "availability": "24/7"
    },
    "crypto_volatility": {
        "indices": list(CRYPTO_VOLATILITY_INDICES.keys()),
        "platforms": ["Various CFD brokers"],
        "account_type": "Standard CFD",
        "min_deposit": 100,
        "leverage": "varies",
        "availability": "24/7 or market hours"
    }
}

def get_indices_by_profit_potential(min_daily_profit_percent):
    """Get indices by minimum daily profit potential"""
    high_profit = {}
    
    # DEX indices - 50-100% potential
    if min_daily_profit_percent <= 100:
        high_profit.update(DEX_INDICES)
    
    # Drift switching - 30-60% potential
    if min_daily_profit_percent <= 60:
        high_profit.update(DRIFT_SWITCHING_INDICES)
    
    # Crypto volatility - event-based 100%+
    if min_daily_profit_percent <= 100:
        high_profit.update(CRYPTO_VOLATILITY_INDICES)
    
    return high_profit

def get_timed_indices():
    """Get all indices with predictable timing patterns"""
    return {
        **DEX_INDICES,
        **DRIFT_SWITCHING_INDICES
    }

def get_24_7_indices():
    """Get all 24/7 tradeable rare indices"""
    return {k: v for k, v in ALL_RARE_INDICES.items() 
            if not v.get("market_hours_only", False)}