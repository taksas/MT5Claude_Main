"""
Ultra Exotic Forex Pairs Configuration
======================================
The rarest and most volatile forex pairs available on MT5 brokers
with extreme profit potential and 300+ pip daily movements.

Created: 2025-06-21
"""

ULTRA_EXOTIC_SYMBOLS = {
    # AFRICAN EXOTIC PAIRS - Extreme Volatility
    "USD/ZAR": {
        "description": "US Dollar/South African Rand",
        "daily_volatility_pips": "1000-1500",
        "average_daily_range": 2595,
        "volatility_percentage": "1.40%",
        "characteristics": [
            "Most volatile exotic pair globally",
            "Driven by gold exports and political developments",
            "Extreme movements during market stress",
            "Can exceed 1500 pips in single day"
        ],
        "best_session": "London/New York overlap",
        "typical_spread": "50-150 pips",
        "recommended_leverage": "1:50 or lower"
    },
    
    "USD/NGN": {
        "description": "US Dollar/Nigerian Naira",
        "daily_volatility_pips": "300-500",
        "current_rate": "1547.78",
        "characteristics": [
            "Oil price correlation",
            "Political instability impact",
            "Central bank interventions",
            "Limited liquidity periods"
        ],
        "best_session": "London",
        "typical_spread": "100-300 pips",
        "recommended_leverage": "1:20"
    },
    
    "USD/KES": {
        "description": "US Dollar/Kenyan Shilling",
        "daily_volatility_pips": "200-400",
        "current_rate": "129.25",
        "characteristics": [
            "Most stable East African currency",
            "Agricultural export influence",
            "Tourism sector impact",
            "Regional trade hub currency"
        ],
        "best_session": "London",
        "typical_spread": "50-150 pips"
    },
    
    "USD/ETB": {
        "description": "US Dollar/Ethiopian Birr",
        "daily_volatility_pips": "300-600",
        "current_rate": "127.75",
        "characteristics": [
            "High inflation impact",
            "Limited market access",
            "Government controls",
            "Extreme volatility spikes"
        ],
        "best_session": "London",
        "typical_spread": "200-500 pips"
    },
    
    "USD/UGX": {
        "description": "US Dollar/Ugandan Shilling",
        "daily_volatility_pips": "400-800",
        "current_rate": "3666.36",
        "characteristics": [
            "Coffee export correlation",
            "Regional instability impact",
            "Low liquidity periods",
            "Sharp intraday movements"
        ],
        "best_session": "London"
    },
    
    # LATIN AMERICAN EXOTIC PAIRS - High Volatility
    "USD/BRL": {
        "description": "US Dollar/Brazilian Real",
        "daily_volatility_pips": "800-1200",
        "average_daily_range": 591.78,
        "volatility_percentage": "1.13%",
        "characteristics": [
            "Commodity export driven",
            "Political volatility impact",
            "Emerging market bellwether",
            "High carry trade interest"
        ],
        "best_session": "New York",
        "typical_spread": "20-50 pips"
    },
    
    "USD/COP": {
        "description": "US Dollar/Colombian Peso",
        "daily_volatility_pips": "400-700",
        "characteristics": [
            "Oil price correlation",
            "Exceeded 4000 pesos/dollar milestone",
            "Among worst performing LatAm currencies",
            "Political instability premium"
        ],
        "best_session": "New York",
        "typical_spread": "50-200 pips"
    },
    
    "USD/CLP": {
        "description": "US Dollar/Chilean Peso",
        "daily_volatility_pips": "300-600",
        "characteristics": [
            "Copper price correlation",
            "Lost 20% value in 3 weeks during crisis",
            "Reached 1060 pesos all-time high",
            "Mining sector influence"
        ],
        "best_session": "New York",
        "typical_spread": "30-100 pips"
    },
    
    "USD/PEN": {
        "description": "US Dollar/Peruvian Sol",
        "daily_volatility_pips": "200-400",
        "characteristics": [
            "Best performing LatAm currency in crisis",
            "Mining export driven",
            "Relatively stable vs peers",
            "Central bank interventions"
        ],
        "best_session": "New York"
    },
    
    # ASIAN EXOTIC PAIRS - Extreme Movements
    "USD/KRW": {
        "description": "US Dollar/South Korean Won",
        "daily_volatility_pips": "500-800",
        "characteristics": [
            "Technology sector correlation",
            "Export economy sensitivity",
            "Geopolitical risk premium",
            "Sharp overnight gaps"
        ],
        "best_session": "Asian",
        "typical_spread": "10-30 pips"
    },
    
    "USD/IDR": {
        "description": "US Dollar/Indonesian Rupiah",
        "daily_volatility_pips": "300-500",
        "current_rate": "15000+",
        "characteristics": [
            "Largest Southeast Asian economy",
            "Commodity export driven",
            "High volatility warning",
            "Political instability impact"
        ],
        "best_session": "Asian",
        "typical_spread": "50-150 pips",
        "risk_warning": "Multi-decade depreciation history"
    },
    
    "USD/THB": {
        "description": "US Dollar/Thai Baht",
        "daily_volatility_pips": "200-400",
        "volatility_rating": "0.64%",
        "characteristics": [
            "Tourism sector impact",
            "Regional trade hub",
            "Central bank interventions",
            "Pandemic volatility increase"
        ],
        "best_session": "Asian",
        "typical_spread": "20-50 pips"
    },
    
    "USD/VND": {
        "description": "US Dollar/Vietnamese Dong",
        "daily_volatility_pips": "100-300",
        "current_rate": "24000+",
        "characteristics": [
            "Export manufacturing growth",
            "Government controls",
            "Limited free float",
            "Depreciation trend"
        ],
        "best_session": "Asian",
        "risk_warning": "Long-term depreciation risk"
    },
    
    # MIDDLE EASTERN EXOTIC PAIRS
    "USD/ILS": {
        "description": "US Dollar/Israeli Shekel",
        "daily_volatility_pips": "100-300",
        "characteristics": [
            "Tech sector correlation",
            "Geopolitical risk spikes",
            "Central bank interventions",
            "Weekend gap risk"
        ],
        "best_session": "London",
        "typical_spread": "20-50 pips"
    },
    
    "USD/AED": {
        "description": "US Dollar/UAE Dirham",
        "daily_volatility_pips": "50-150",
        "characteristics": [
            "USD pegged (3.67)",
            "Oil price indirect impact",
            "Limited volatility due to peg",
            "Regional hub currency"
        ],
        "best_session": "London",
        "note": "Pegged currency - limited movement"
    },
    
    "USD/JOD": {
        "description": "US Dollar/Jordanian Dinar",
        "daily_volatility_pips": "50-100",
        "monthly_volatility": "1.03%",
        "characteristics": [
            "USD soft peg",
            "Regional stability premium",
            "Limited free float",
            "Low volatility"
        ],
        "best_session": "London"
    },
    
    # ULTRA RARE EXOTIC PAIRS (Limited Availability)
    "USD/RUB": {
        "description": "US Dollar/Russian Ruble",
        "daily_volatility_pips": "4000-5000",
        "characteristics": [
            "Extreme volatility",
            "Sanctions impact",
            "Oil price correlation",
            "Geopolitical risk"
        ],
        "risk_warning": "Check broker availability and restrictions"
    },
    
    "USD/SEK": {
        "description": "US Dollar/Swedish Krona",
        "daily_volatility_pips": "500-600",
        "characteristics": [
            "Safe haven flows",
            "European exposure",
            "Central bank policy impact",
            "Moderate exotic volatility"
        ],
        "best_session": "London"
    },
    
    "EUR/TRY": {
        "description": "Euro/Turkish Lira",
        "daily_volatility_pips": "800-1500",
        "characteristics": [
            "Extreme inflation impact",
            "Political instability",
            "Central bank credibility issues",
            "Massive daily swings"
        ],
        "best_session": "London",
        "risk_warning": "Extreme volatility - position sizing critical"
    },
    
    "USD/CNH": {
        "description": "US Dollar/Offshore Chinese Yuan",
        "daily_volatility_pips": "100-300",
        "characteristics": [
            "Trade war impact",
            "Government interventions",
            "Different from onshore CNY",
            "Policy announcement gaps"
        ],
        "best_session": "Asian"
    }
}

# Brokers offering exotic pairs on MT5
RECOMMENDED_BROKERS = {
    "HFM": {
        "regulation": ["FCA", "CySEC", "DFSA", "FSCA"],
        "exotic_pairs": ["African currencies", "KES", "ETB", "UGX", "NGN"],
        "min_spread": "0.1 pips",
        "max_leverage": "1:1000",
        "platform": "MT5"
    },
    
    "JustMarkets": {
        "special_feature": "Account denominations in IDR, THB, VND",
        "pairs_count": "66+ Forex pairs",
        "platform": "MT4/MT5",
        "account_types": 7
    },
    
    "Tickmill": {
        "features": "Best MT5 execution",
        "exotic_focus": "Emerging markets",
        "platform": "MT5"
    },
    
    "Alpari International": {
        "latin_pairs": ["USD/COP", "USD/CLP", "USD/PEN"],
        "platform": "MT5"
    },
    
    "IC Markets": {
        "features": "Raw spreads",
        "exotic_pairs": "Full range",
        "platform": "MT5"
    },
    
    "Exness": {
        "features": "Instant execution",
        "exotic_pairs": ["USD/COP", "USD/BRL"],
        "platform": "MT5"
    }
}

# Trading Guidelines
TRADING_GUIDELINES = {
    "risk_management": {
        "position_sizing": "Use 1/5 of normal major pair size",
        "stop_loss": "Wider stops required - 200-500 pips minimum",
        "leverage": "Maximum 1:50 for exotics, preferably 1:20",
        "correlation": "Check commodity/political correlations"
    },
    
    "best_practices": {
        "liquidity": "Trade during main market sessions only",
        "news": "Monitor political/economic events closely",
        "gaps": "Beware of weekend and overnight gaps",
        "spreads": "Account for wide spreads in profit calculations"
    },
    
    "profit_targets": {
        "scalping": "50-100 pips (account for spreads)",
        "day_trading": "200-500 pips",
        "swing_trading": "500-1500 pips",
        "position_trading": "1000-3000 pips"
    }
}

# Most Profitable Setups
PROFITABLE_SETUPS = {
    "extreme_volatility": ["USD/ZAR", "USD/BRL", "EUR/TRY", "USD/RUB"],
    "high_carry_trades": ["USD/BRL", "USD/MXN", "USD/ZAR"],
    "commodity_correlation": ["USD/CLP (copper)", "USD/COP (oil)", "USD/ZAR (gold)"],
    "crisis_plays": ["USD/TRY", "USD/ARS", "USD/ZAR"],
    "asian_growth": ["USD/IDR", "USD/VND", "USD/KRW"]
}

# Session-based Trading
OPTIMAL_SESSIONS = {
    "Asian": ["USD/KRW", "USD/IDR", "USD/THB", "USD/VND", "USD/CNH"],
    "London": ["USD/ZAR", "USD/NGN", "USD/KES", "EUR/TRY", "USD/ILS"],
    "New York": ["USD/BRL", "USD/COP", "USD/CLP", "USD/MXN", "USD/PEN"],
    "24h_volatile": ["USD/ZAR", "USD/BRL", "EUR/TRY", "USD/RUB"]
}