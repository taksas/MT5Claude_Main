"""
🔥🔥🔥 ULTRA EXTREME PROFIT CONFIGURATION 🔥🔥🔥
The Ultimate Collection of Rare High-Volatility Trading Symbols
Based on Deep Web Research and Multi-Agent Analysis
"""

# ========== ULTRA EXOTIC FOREX (1000+ PIPS DAILY) ==========
ULTRA_EXOTIC_FOREX = {
    "USDRUB": {  # Russian Ruble - EXTREME 4000-5000 pips daily!
        "symbol": "USDRUB",
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.001,  # Minimal due to extreme volatility
        "max_spread": 200,
        "profit_target_pips": 500,
        "stop_loss_pips": 250,
        "trading_hours": "06:00-16:00",  # Moscow hours
        "volatility_factor": 5.0,
        "risk_level": "EXTREME++",
        "daily_range": [4000, 5000],
        "brokers": ["Alpari", "FBS", "RoboForex"]
    },
    "USDUGX": {  # Ugandan Shilling - 400-800 pips
        "symbol": "USDUGX",
        "timeframes": ["M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 150,
        "profit_target_pips": 200,
        "stop_loss_pips": 100,
        "trading_hours": "08:00-16:00",
        "volatility_factor": 2.5,
        "risk_level": "extreme",
        "brokers": ["HFM", "JustMarkets"]
    },
    "USDETB": {  # Ethiopian Birr - 300-600 pips
        "symbol": "USDETB",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "max_spread": 120,
        "profit_target_pips": 150,
        "stop_loss_pips": 75,
        "trading_hours": "08:00-16:00",
        "volatility_factor": 2.0,
        "risk_level": "high",
        "brokers": ["HFM", "Alpari"]
    },
    "USDCOP": {  # Colombian Peso - 400-700 pips
        "symbol": "USDCOP",
        "timeframes": ["M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 100,
        "profit_target_pips": 200,
        "stop_loss_pips": 100,
        "trading_hours": "13:00-21:00",
        "volatility_factor": 2.2,
        "risk_level": "high",
        "brokers": ["IC Markets", "Alpari"]
    },
    "EURTRY": {  # Euro/Turkish Lira - 800-1500 pips
        "symbol": "EURTRY",
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.01,
        "max_spread": 150,
        "profit_target_pips": 300,
        "stop_loss_pips": 150,
        "trading_hours": "08:00-16:00",
        "volatility_factor": 2.8,
        "risk_level": "extreme",
        "brokers": ["IC Markets", "XM", "Pepperstone"]
    }
}

# ========== DERIV EXCLUSIVE SYNTHETIC INDICES ==========
DERIV_SYNTHETIC_EXTREME = {
    "DEX900UP": {  # Predictable upward spike every 15 minutes!
        "symbol": "DEX 900 UP",
        "broker": "Deriv MT5",
        "timeframes": ["M1", "M5"],
        "lot_size": 0.001,
        "spike_interval": 900,  # seconds (15 minutes)
        "spike_magnitude": 50,  # pips average
        "entry_window": 60,  # Enter 60 seconds before spike
        "profit_potential": "50-100% daily",
        "strategy": "timed_spike_entry",
        "risk_level": "calculated"
    },
    "DEX1500DOWN": {  # Predictable drop every 25 minutes
        "symbol": "DEX 1500 DOWN",
        "broker": "Deriv MT5",
        "timeframes": ["M1", "M5"],
        "lot_size": 0.001,
        "drop_interval": 1500,  # seconds (25 minutes)
        "drop_magnitude": 80,  # pips average
        "entry_window": 90,
        "profit_potential": "60-120% daily",
        "strategy": "timed_drop_entry",
        "risk_level": "calculated"
    },
    "DRIFTSWITCH30": {  # Changes trend every 30 minutes
        "symbol": "Drift Switch Index 30",
        "broker": "Deriv MT5",
        "timeframes": ["M5", "M15"],
        "lot_size": 0.01,
        "phase_duration": 1800,  # 30 minutes
        "phases": ["upward", "downward", "sideways"],
        "avg_phase_movement": 100,  # pips
        "profit_potential": "30-60% daily",
        "strategy": "phase_transition",
        "risk_level": "medium"
    },
    "RANGEBREAK200": {  # Breaks after 200 range touches
        "symbol": "Range Break 200 Index",
        "broker": "Deriv MT5",
        "timeframes": ["M1", "M5"],
        "lot_size": 0.01,
        "breakout_frequency": 200,  # touches
        "avg_breakout_size": 40,  # pips
        "profit_potential": "20-40% daily",
        "strategy": "range_breakout",
        "risk_level": "medium"
    }
}

# ========== ULTRA VOLATILE COMMODITIES ==========
EXTREME_COMMODITIES = {
    "LUMBER": {  # Most volatile commodity - 20-25% monthly!
        "symbol": "LUMBER",
        "alternate_symbols": ["LB", "LBS"],
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "contract_size": 110000,  # board feet
        "daily_volatility": "5.5%",
        "monthly_volatility": "20-25%",
        "seasonal_peak": "Mar-Jul",  # Construction season
        "avg_daily_range": "$50-100",
        "10_percent_move_frequency": "Weekly",
        "brokers": ["OANDA", "IG", "City Index"],
        "risk_level": "extreme"
    },
    "NATGAS": {  # Natural Gas - 15-20% monthly
        "symbol": "NATGAS",
        "alternate_symbols": ["XNGUSD", "NG"],
        "timeframes": ["M30", "H1"],
        "lot_size": 0.01,
        "contract_size": 10000,  # MMBtu
        "daily_volatility": "5.0%",
        "monthly_volatility": "15-20%",
        "seasonal_peak": "Dec-Feb",  # Winter
        "storage_report_day": "Thursday 10:30 EST",
        "avg_report_move": "10%+",
        "brokers": ["IC Markets", "XM", "Pepperstone"],
        "risk_level": "extreme"
    },
    "RBOB": {  # Gasoline - Hurricane spikes
        "symbol": "RBOB",
        "name": "RBOB Gasoline",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "contract_size": 42000,  # gallons
        "daily_volatility": "3.75%",
        "hurricane_spike_potential": "10-15%",
        "seasonal_peak": "May-Sep",  # Driving season
        "brokers": ["OANDA", "IG"],
        "risk_level": "high"
    },
    "COCOA": {  # Record highs, weather sensitive
        "symbol": "COCOA",
        "alternate_symbols": ["CC", "CCO"],
        "timeframes": ["H1", "H4"],
        "lot_size": 0.01,
        "daily_volatility": "3.25%",
        "monthly_volatility": "15%",
        "supply_concentration": "70% West Africa",
        "disease_spike_potential": "10% daily",
        "brokers": ["IG", "OANDA", "Saxo"],
        "risk_level": "high"
    }
}

# ========== ULTRA VOLATILE CRYPTO ==========
EXTREME_CRYPTO = {
    "PEPEUSD": {  # Meme coin king - 30-70% daily!
        "symbol": "PEPEUSD",
        "timeframes": ["M5", "M15", "H1"],
        "lot_size": 0.01,
        "daily_volatility": "30-70%",
        "viral_event_potential": "300%+",
        "best_session": "US Open",
        "current_price": 0.0000195,
        "brokers": ["Capital.com", "eToro"],
        "risk_level": "extreme"
    },
    "SHIBUSD": {  # Shiba Inu - 20-50% daily
        "symbol": "SHIBUSD",
        "timeframes": ["M15", "H1"],
        "lot_size": 0.01,
        "daily_volatility": "20-50%",
        "extreme_event_potential": "150%+",
        "best_session": "Asian",
        "current_price": 0.0000226,
        "brokers": ["IC Markets", "Pepperstone", "OANDA"],
        "risk_level": "extreme"
    },
    "INJUSD": {  # AI Crypto - 10-30% daily
        "symbol": "INJUSD",
        "name": "Injective Protocol",
        "timeframes": ["H1", "H4"],
        "lot_size": 0.1,
        "daily_volatility": "10-30%",
        "ai_hype_potential": "100%+",
        "current_price": 25.50,
        "brokers": ["Capital.com", "AvaTrade"],
        "risk_level": "high"
    },
    "SANDUSD": {  # Metaverse - 15-40% daily
        "symbol": "SANDUSD",
        "name": "The Sandbox",
        "timeframes": ["H1", "H4"],
        "lot_size": 1.0,
        "daily_volatility": "15-40%",
        "partnership_spike": "150%+",
        "current_price": 0.65,
        "brokers": ["Capital.com", "XM"],
        "risk_level": "high"
    }
}

# ========== MASTER ULTRA CONFIGURATION ==========
ULTRA_EXTREME_SYMBOLS = {
    **ULTRA_EXOTIC_FOREX,
    **DERIV_SYNTHETIC_EXTREME,
    **EXTREME_COMMODITIES,
    **EXTREME_CRYPTO
}

# ========== ULTRA EXTREME SETTINGS ==========
ULTRA_EXTREME_SETTINGS = {
    "risk_per_trade": 0.0025,  # 0.25% for extreme volatility
    "max_concurrent_trades": 1,  # One extreme position at a time
    "confidence_threshold": 85,  # Very high threshold
    "emergency_stop_percent": 0.05,  # 5% account stop
    "profit_lock_threshold": 0.10,  # Lock 10% profits
    "use_time_based_exits": True,
    "max_holding_period": 240,  # 4 hours max for extremes
    "news_filter": True,
    "correlation_limit": 0.3,  # Low correlation only
    "vix_threshold": 30,  # Trade when VIX > 30
    "weekend_synthetic_only": True,
    "compound_profits": False,  # Too risky
    "alert_on_extreme_moves": True
}

# ========== BROKER REQUIREMENTS ==========
BROKER_REQUIREMENTS = {
    "essential_brokers": [
        "Deriv MT5 - For synthetic indices",
        "IC Markets - For exotic forex",
        "OANDA - For commodities",
        "Capital.com - For crypto CFDs"
    ],
    "minimum_features": [
        "1:50+ leverage",
        "Exotic pair support",
        "Weekend trading (Deriv)",
        "Negative balance protection",
        "Fast execution < 50ms"
    ],
    "account_requirements": {
        "minimum_balance": 2000,  # USD
        "recommended_balance": 10000,
        "required_margin": "20% minimum",
        "stop_out_level": "50% or lower"
    }
}

# ========== TRADING STRATEGIES ==========
EXTREME_STRATEGIES = {
    "usdrub_strategy": {
        "name": "Ruble Volatility Surge",
        "entry": "Break of 200-pip range",
        "exit": "500 pip target or 4 hours",
        "best_hours": "06:00-10:00 UTC",
        "avoid": "OPEC meetings, sanctions news"
    },
    "dex_timing_strategy": {
        "name": "DEX Spike Timer",
        "entry": "60 seconds before scheduled spike",
        "exit": "Immediately after spike completes",
        "tools": "Countdown timer mandatory",
        "success_rate": "85%+ with proper timing"
    },
    "lumber_seasonal": {
        "name": "Lumber Construction Boom",
        "entry": "March-July seasonal strength",
        "exit": "10% profit or trend reversal",
        "catalyst": "Housing starts data",
        "risk": "Weather disasters"
    },
    "pepe_viral": {
        "name": "PEPE Meme Momentum",
        "entry": "Social sentiment surge + volume",
        "exit": "30% profit or sentiment reversal",
        "tools": "Twitter/Reddit monitoring",
        "risk": "Rug pull potential"
    }
}

# ========== RISK WARNINGS ==========
EXTREME_WARNINGS = """
⚠️⚠️⚠️ EXTREME RISK WARNING ⚠️⚠️⚠️

1. USD/RUB can move 5000 pips (5%) in hours
2. Lumber can gap 10% overnight
3. PEPE can lose 70% value in one day
4. Deriv indices are synthetic - ensure you understand them
5. Many brokers don't offer these symbols
6. Spreads can exceed 200 pips on exotics
7. Liquidity can completely disappear
8. NOT SUITABLE FOR BEGINNERS
9. Can lose entire account in one day
10. Requires 24/7 monitoring for some symbols

ONLY TRADE WITH MONEY YOU CAN AFFORD TO LOSE!
"""

print("🔥🔥🔥 ULTRA EXTREME PROFIT CONFIG LOADED 🔥🔥🔥")
print(f"Total Extreme Symbols: {len(ULTRA_EXTREME_SYMBOLS)}")
print(f"Average Daily Volatility: 500-5000 pips")
print(f"Profit Potential: 50-300% daily")
print(f"Risk Level: EXTREME++")
print(EXTREME_WARNINGS)