"""
Commodity CFDs Configuration for MT5 - High Profit Potential
Focus on commodities with high intraday volatility suitable for automated trading
"""

COMMODITY_CFDS = {
    # PRECIOUS METALS (beyond XAUUSD)
    "XAGUSD": {  # Silver
        "name": "Silver vs USD",
        "category": "precious_metal",
        "avg_daily_range": 0.80,  # $0.80 movement
        "typical_spread": 3.5,    # 3.5 pips
        "volatility": "3.5%",     # Daily volatility
        "best_sessions": ["London", "New_York"],
        "seasonal_patterns": {
            "strong_months": ["Jan", "Feb", "Aug", "Sep"],
            "weak_months": ["May", "Jun", "Oct"]
        },
        "profit_potential": "very_high",
        "risk_factor": 0.008,
        "correlation": "inverse_USD, positive_XAUUSD",
        "news_sensitivity": "high",
        "typical_lot_size": 5000  # 5000 oz
    },
    
    "XPDUSD": {  # Palladium
        "name": "Palladium vs USD", 
        "category": "precious_metal",
        "avg_daily_range": 50.00,  # $50 movement
        "typical_spread": 100,     # 100 pips
        "volatility": "4.0%",      # Daily volatility
        "best_sessions": ["London", "New_York"],
        "seasonal_patterns": {
            "strong_months": ["Jan", "Sep", "Oct"],
            "weak_months": ["Jun", "Jul", "Aug"]
        },
        "profit_potential": "extreme",
        "risk_factor": 0.005,
        "correlation": "auto_industry, positive_XPTUSD",
        "industrial_use": "catalytic_converters",
        "typical_lot_size": 100   # 100 oz
    },
    
    "XPTUSD": {  # Platinum
        "name": "Platinum vs USD",
        "category": "precious_metal", 
        "avg_daily_range": 30.00,  # $30 movement
        "typical_spread": 50,      # 50 pips
        "volatility": "3.0%",      # Daily volatility
        "best_sessions": ["London", "New_York"],
        "seasonal_patterns": {
            "strong_months": ["Jan", "Mar", "Sep"],
            "weak_months": ["May", "Aug", "Dec"]
        },
        "profit_potential": "high",
        "risk_factor": 0.007,
        "correlation": "auto_industry, jewelry_demand",
        "industrial_use": "catalytic_converters, jewelry",
        "typical_lot_size": 100   # 100 oz
    },
    
    # ENERGY COMMODITIES
    "USOIL": {  # WTI Crude Oil
        "name": "US Crude Oil (WTI)",
        "category": "energy",
        "avg_daily_range": 2.50,   # $2.50 movement
        "typical_spread": 3,       # 3 pips
        "volatility": "3.2%",      # Daily volatility
        "best_sessions": ["New_York", "London"],
        "seasonal_patterns": {
            "strong_months": ["Mar", "Apr", "Aug", "Sep"],  # Driving season
            "weak_months": ["Dec", "Jan", "Feb"]
        },
        "profit_potential": "very_high",
        "risk_factor": 0.007,
        "correlation": "USD_negative, CAD_positive, stocks_positive",
        "news_events": ["EIA_inventory", "OPEC_meetings"],
        "typical_lot_size": 1000   # 1000 barrels
    },
    
    "UKOIL": {  # Brent Crude Oil
        "name": "UK Crude Oil (Brent)",
        "category": "energy",
        "avg_daily_range": 2.80,   # $2.80 movement
        "typical_spread": 5,       # 5 pips
        "volatility": "3.3%",      # Daily volatility
        "best_sessions": ["London", "New_York"],
        "seasonal_patterns": {
            "strong_months": ["Mar", "Apr", "Sep", "Oct"],
            "weak_months": ["Dec", "Jan", "Feb"]
        },
        "profit_potential": "very_high",
        "risk_factor": 0.007,
        "correlation": "USOIL_high, EUR_positive, geopolitical",
        "news_events": ["EIA_inventory", "OPEC_meetings", "Middle_East_news"],
        "typical_lot_size": 1000   # 1000 barrels
    },
    
    "NATGAS": {  # Natural Gas
        "name": "Natural Gas",
        "category": "energy",
        "avg_daily_range": 0.150,  # $0.15 movement
        "typical_spread": 10,      # 10 pips
        "volatility": "5.0%",      # Daily volatility - EXTREME
        "best_sessions": ["New_York"],
        "seasonal_patterns": {
            "strong_months": ["Dec", "Jan", "Feb"],  # Winter heating
            "weak_months": ["Apr", "May", "Sep"]    # Shoulder months
        },
        "profit_potential": "extreme",
        "risk_factor": 0.005,
        "correlation": "weather_dependent, storage_reports",
        "news_events": ["EIA_storage", "weather_forecasts"],
        "typical_lot_size": 10000  # 10,000 MMBtu
    },
    
    # AGRICULTURAL COMMODITIES
    "WHEAT": {  # Wheat
        "name": "Wheat",
        "category": "agricultural",
        "avg_daily_range": 15.00,  # 15 cents movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.5%",      # Daily volatility
        "best_sessions": ["Chicago", "London"],
        "seasonal_patterns": {
            "strong_months": ["Mar", "Apr", "May"],  # Planting
            "weak_months": ["Aug", "Sep", "Oct"]     # Harvest
        },
        "profit_potential": "high",
        "risk_factor": 0.007,
        "correlation": "weather, USD_negative, corn_positive",
        "news_events": ["USDA_reports", "weather_updates"],
        "typical_lot_size": 5000   # 5000 bushels
    },
    
    "CORN": {  # Corn
        "name": "Corn", 
        "category": "agricultural",
        "avg_daily_range": 10.00,  # 10 cents movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.2%",      # Daily volatility
        "best_sessions": ["Chicago", "London"],
        "seasonal_patterns": {
            "strong_months": ["Apr", "May", "Jun"],  # Planting
            "weak_months": ["Oct", "Nov", "Dec"]     # Harvest
        },
        "profit_potential": "medium_high",
        "risk_factor": 0.008,
        "correlation": "ethanol_demand, wheat_positive",
        "news_events": ["USDA_reports", "weather_updates"],
        "typical_lot_size": 5000   # 5000 bushels
    },
    
    "SOYBEAN": {  # Soybeans
        "name": "Soybeans",
        "category": "agricultural",
        "avg_daily_range": 25.00,  # 25 cents movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.3%",      # Daily volatility
        "best_sessions": ["Chicago", "London"],
        "seasonal_patterns": {
            "strong_months": ["May", "Jun", "Jul"],  # Growing season
            "weak_months": ["Nov", "Dec", "Jan"]     # Post-harvest
        },
        "profit_potential": "high",
        "risk_factor": 0.007,
        "correlation": "China_demand, weather, corn_positive",
        "news_events": ["USDA_reports", "China_trade"],
        "typical_lot_size": 5000   # 5000 bushels
    },
    
    "COFFEE": {  # Coffee
        "name": "Coffee",
        "category": "agricultural",
        "avg_daily_range": 3.00,   # 3 cents movement
        "typical_spread": 10,      # 10 pips
        "volatility": "3.0%",      # Daily volatility
        "best_sessions": ["New_York", "London"],
        "seasonal_patterns": {
            "strong_months": ["May", "Jun", "Jul"],  # Frost season Brazil
            "weak_months": ["Oct", "Nov", "Dec"]
        },
        "profit_potential": "high",
        "risk_factor": 0.006,
        "correlation": "Brazil_weather, USD_negative",
        "news_events": ["Brazil_weather", "inventory_reports"],
        "typical_lot_size": 37500  # 37,500 lbs
    },
    
    "SUGAR": {  # Sugar
        "name": "Sugar",
        "category": "agricultural",
        "avg_daily_range": 0.40,   # 0.40 cents movement
        "typical_spread": 3,       # 3 pips
        "volatility": "2.8%",      # Daily volatility
        "best_sessions": ["New_York", "London"],
        "seasonal_patterns": {
            "strong_months": ["Feb", "Mar", "Apr"],  # Pre-harvest
            "weak_months": ["Sep", "Oct", "Nov"]     # Harvest
        },
        "profit_potential": "medium_high",
        "risk_factor": 0.007,
        "correlation": "Brazil_weather, ethanol_prices, oil_positive",
        "news_events": ["Brazil_production", "India_policy"],
        "typical_lot_size": 112000 # 112,000 lbs
    },
    
    "COCOA": {  # Cocoa
        "name": "Cocoa",
        "category": "agricultural",
        "avg_daily_range": 50.00,  # $50 movement
        "typical_spread": 10,      # 10 pips
        "volatility": "2.5%",      # Daily volatility
        "best_sessions": ["New_York", "London"],
        "seasonal_patterns": {
            "strong_months": ["Sep", "Oct", "Nov"],  # Main crop
            "weak_months": ["Apr", "May", "Jun"]     # Mid crop
        },
        "profit_potential": "high",
        "risk_factor": 0.007,
        "correlation": "West_Africa_weather, GBP_positive",
        "news_events": ["Ivory_Coast_weather", "Ghana_production"],
        "typical_lot_size": 10     # 10 metric tons
    },
    
    # INDUSTRIAL METALS
    "COPPER": {  # Copper
        "name": "Copper",
        "category": "industrial_metal",
        "avg_daily_range": 0.0500, # $0.05 movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.0%",      # Daily volatility
        "best_sessions": ["London", "Shanghai", "New_York"],
        "seasonal_patterns": {
            "strong_months": ["Jan", "Feb", "Sep", "Oct"],  # Construction season
            "weak_months": ["Jun", "Jul", "Aug"]
        },
        "profit_potential": "high",
        "risk_factor": 0.008,
        "correlation": "China_PMI, construction, USD_negative",
        "news_events": ["China_data", "Chile_production"],
        "typical_lot_size": 25000  # 25,000 lbs
    },
    
    "ALUMINUM": {  # Aluminum
        "name": "Aluminum",
        "category": "industrial_metal",
        "avg_daily_range": 30.00,  # $30 movement
        "typical_spread": 5,       # 5 pips
        "volatility": "1.8%",      # Daily volatility
        "best_sessions": ["London", "Shanghai"],
        "seasonal_patterns": {
            "strong_months": ["Mar", "Apr", "Sep", "Oct"],
            "weak_months": ["Dec", "Jan", "Jul"]
        },
        "profit_potential": "medium_high",
        "risk_factor": 0.009,
        "correlation": "China_demand, energy_prices",
        "news_events": ["LME_inventory", "China_production"],
        "typical_lot_size": 25     # 25 metric tons
    },
    
    "ZINC": {  # Zinc
        "name": "Zinc",
        "category": "industrial_metal",
        "avg_daily_range": 50.00,  # $50 movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.2%",      # Daily volatility
        "best_sessions": ["London", "Shanghai"],
        "seasonal_patterns": {
            "strong_months": ["Feb", "Mar", "Sep", "Oct"],
            "weak_months": ["Jun", "Jul", "Dec"]
        },
        "profit_potential": "high",
        "risk_factor": 0.008,
        "correlation": "galvanizing_demand, construction",
        "news_events": ["LME_inventory", "China_demand"],
        "typical_lot_size": 25     # 25 metric tons
    },
    
    "NICKEL": {  # Nickel
        "name": "Nickel",
        "category": "industrial_metal",
        "avg_daily_range": 300.00, # $300 movement
        "typical_spread": 20,      # 20 pips
        "volatility": "3.5%",      # Daily volatility
        "best_sessions": ["London", "Shanghai"],
        "seasonal_patterns": {
            "strong_months": ["Jan", "Feb", "Sep", "Oct"],
            "weak_months": ["Jun", "Jul", "Aug"]
        },
        "profit_potential": "very_high",
        "risk_factor": 0.006,
        "correlation": "stainless_steel, EV_batteries, Indonesia_policy",
        "news_events": ["Indonesia_export_policy", "EV_demand"],
        "typical_lot_size": 6      # 6 metric tons
    },
    
    "LEAD": {  # Lead
        "name": "Lead",
        "category": "industrial_metal",
        "avg_daily_range": 30.00,  # $30 movement
        "typical_spread": 5,       # 5 pips
        "volatility": "2.0%",      # Daily volatility
        "best_sessions": ["London", "Shanghai"],
        "seasonal_patterns": {
            "strong_months": ["Sep", "Oct", "Nov"],  # Battery season
            "weak_months": ["Apr", "May", "Jun"]
        },
        "profit_potential": "medium_high",
        "risk_factor": 0.008,
        "correlation": "battery_demand, auto_production",
        "news_events": ["LME_inventory", "battery_demand"],
        "typical_lot_size": 25     # 25 metric tons
    }
}

# Helper functions for commodity trading
def get_commodity_config(symbol):
    """Get configuration for a specific commodity"""
    return COMMODITY_CFDS.get(symbol, {
        "avg_daily_range": 50,
        "typical_spread": 10,
        "risk_factor": 0.008,
        "profit_potential": "medium"
    })

def get_commodities_by_category(category):
    """Get all commodities in a specific category"""
    return {symbol: config for symbol, config in COMMODITY_CFDS.items() 
            if config.get("category") == category}

def get_high_profit_commodities():
    """Get commodities with high or extreme profit potential"""
    return {symbol: config for symbol, config in COMMODITY_CFDS.items() 
            if config.get("profit_potential") in ["high", "very_high", "extreme"]}

def get_seasonal_commodities(month):
    """Get commodities that are seasonally strong in a given month"""
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    month_name = month_map.get(month, "Jan")
    
    seasonal_commodities = {}
    for symbol, config in COMMODITY_CFDS.items():
        if "seasonal_patterns" in config:
            if month_name in config["seasonal_patterns"].get("strong_months", []):
                seasonal_commodities[symbol] = config
    
    return seasonal_commodities

# Trading recommendations by volatility level
VOLATILITY_BASED_RECOMMENDATIONS = {
    "extreme": {  # >4% daily volatility
        "symbols": ["NATGAS", "XPDUSD"],
        "strategy": "scalping_with_tight_stops",
        "risk_per_trade": 0.003,  # 0.3% risk
        "take_profit_multiplier": 2.5,
        "notes": "Trade during high liquidity sessions only"
    },
    "very_high": {  # 3-4% daily volatility
        "symbols": ["XAGUSD", "USOIL", "UKOIL", "NICKEL", "COFFEE"],
        "strategy": "trend_following_with_volatility_filters",
        "risk_per_trade": 0.005,  # 0.5% risk
        "take_profit_multiplier": 2.0,
        "notes": "Best during trending markets"
    },
    "high": {  # 2-3% daily volatility
        "symbols": ["XPTUSD", "WHEAT", "COCOA", "COPPER", "SUGAR", "SOYBEAN"],
        "strategy": "breakout_trading",
        "risk_per_trade": 0.007,  # 0.7% risk
        "take_profit_multiplier": 1.5,
        "notes": "Trade breakouts of key levels"
    },
    "medium": {  # <2% daily volatility
        "symbols": ["CORN", "ALUMINUM", "ZINC", "LEAD"],
        "strategy": "range_trading_mean_reversion",
        "risk_per_trade": 0.008,  # 0.8% risk
        "take_profit_multiplier": 1.3,
        "notes": "Best in ranging markets"
    }
}

# Key trading sessions for commodities
COMMODITY_TRADING_SESSIONS = {
    "London": {
        "open": "08:00",
        "close": "17:00",
        "timezone": "Europe/London",
        "best_for": ["metals", "energy", "agricultural"]
    },
    "New_York": {
        "open": "09:30",
        "close": "16:00", 
        "timezone": "America/New_York",
        "best_for": ["energy", "agricultural", "precious_metals"]
    },
    "Shanghai": {
        "open": "09:00",
        "close": "15:00",
        "timezone": "Asia/Shanghai",
        "best_for": ["industrial_metals", "agricultural"]
    },
    "Chicago": {
        "open": "08:30",
        "close": "13:15",
        "timezone": "America/Chicago",
        "best_for": ["agricultural"]
    }
}