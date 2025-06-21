# Rare and Ultra-High-Profit Commodities Configuration
# Discovered through ultra-thinking analysis for maximum volatility trading

# AGRICULTURAL COMMODITIES - WEATHER SENSITIVE
AGRICULTURAL_COMMODITIES = {
    "COCOA": {
        "symbol": "COCOA",
        "mt5_symbols": ["COCOA", "CC", "COCOA-C"],
        "name": "Cocoa Futures",
        "daily_volatility_percent": 3.25,
        "typical_spread": 4.0,
        "risk_per_trade": 0.004,
        "lot_size": 0.10,
        "max_spread": 6.0,
        "best_hours": "13:30-20:00",
        "best_months": "Sep-Nov",
        "volatility_factor": 3.5,
        "seasonal_pattern": "harvest_disruption",
        "weather_sensitive": True,
        "supply_concentration": ["Ivory Coast", "Ghana"],
        "min_confidence": 0.74,
        "profit_characteristics": "Pod disease outbreaks create 10% daily moves"
    },
    "SUGAR": {
        "symbol": "SUGAR",
        "mt5_symbols": ["SUGAR", "SB", "SUGAR11", "SUGAR-C"],
        "name": "Sugar #11 Futures",
        "daily_volatility_percent": 2.75,
        "typical_spread": 0.02,
        "risk_per_trade": 0.005,
        "lot_size": 1.0,
        "max_spread": 0.04,
        "best_hours": "13:30-20:00",
        "best_months": "May-Jul, Oct-Dec",
        "volatility_factor": 3.0,
        "dual_use": ["food", "ethanol"],
        "brazil_dependent": True,
        "oil_correlation": 0.65,
        "min_confidence": 0.72,
        "profit_characteristics": "Brazil ethanol policy swings create arbitrage"
    },
    "COTTON": {
        "symbol": "COTTON",
        "mt5_symbols": ["COTTON", "CT", "COTTON-C"],
        "name": "Cotton Futures",
        "daily_volatility_percent": 2.25,
        "typical_spread": 0.10,
        "risk_per_trade": 0.005,
        "lot_size": 0.50,
        "max_spread": 0.15,
        "best_hours": "13:30-20:00",
        "best_months": "Jul-Sep",
        "volatility_factor": 2.5,
        "weather_markets": ["US", "India", "China"],
        "fashion_cycle": True,
        "min_confidence": 0.70,
        "profit_characteristics": "US weather + China demand = volatility"
    },
    "WHEAT": {
        "symbol": "WHEAT",
        "mt5_symbols": ["WHEAT", "ZW", "WHEAT-C", "W"],
        "name": "Wheat Futures",
        "daily_volatility_percent": 3.0,
        "typical_spread": 0.75,
        "risk_per_trade": 0.004,
        "lot_size": 0.20,
        "max_spread": 1.25,
        "best_hours": "13:30-20:00",
        "best_months": "Mar-May, Jul-Aug",
        "volatility_factor": 3.5,
        "geopolitical_sensitive": True,
        "black_sea_dependent": True,
        "export_ban_risk": True,
        "min_confidence": 0.73,
        "profit_characteristics": "Ukraine conflicts create 8% daily spikes"
    },
    "CORN": {
        "symbol": "CORN",
        "mt5_symbols": ["CORN", "ZC", "CORN-C", "C"],
        "name": "Corn Futures",
        "daily_volatility_percent": 2.65,
        "typical_spread": 0.50,
        "risk_per_trade": 0.005,
        "lot_size": 0.25,
        "max_spread": 0.75,
        "best_hours": "13:30-20:00",
        "best_months": "Jun-Aug",
        "volatility_factor": 3.0,
        "pollination_critical": True,
        "ethanol_demand": True,
        "usda_reports": True,
        "min_confidence": 0.71,
        "profit_characteristics": "Pollination weather creates 5% moves"
    },
    "SOYBEANS": {
        "symbol": "SOYBEAN",
        "mt5_symbols": ["SOYBEAN", "ZS", "SOYBEANS-C", "S"],
        "name": "Soybean Futures",
        "daily_volatility_percent": 2.25,
        "typical_spread": 1.25,
        "risk_per_trade": 0.005,
        "lot_size": 0.10,
        "max_spread": 2.0,
        "best_hours": "13:30-20:00",
        "best_months": "Jul-Aug, Mar-Apr",
        "volatility_factor": 2.5,
        "us_china_trade": True,
        "dual_hemisphere": True,
        "protein_demand": True,
        "min_confidence": 0.70,
        "profit_characteristics": "Trade war news creates instant 3% moves"
    }
}

# ENERGY PRODUCTS - SEASONAL VOLATILITY
ENERGY_PRODUCTS = {
    "GASOLINE": {
        "symbol": "RBOB",
        "mt5_symbols": ["RBOB", "RB", "GASOLINE", "XRB"],
        "name": "RBOB Gasoline Futures",
        "daily_volatility_percent": 3.75,
        "typical_spread": 0.0030,
        "risk_per_trade": 0.004,
        "lot_size": 0.10,
        "max_spread": 0.0050,
        "best_hours": "13:30-20:00",
        "best_months": "May-Sep",
        "volatility_factor": 4.0,
        "driving_season": True,
        "hurricane_sensitive": True,
        "refinery_risk": True,
        "min_confidence": 0.74,
        "profit_characteristics": "Hurricane season creates 10% daily moves"
    },
    "HEATINGOIL": {
        "symbol": "HO",
        "mt5_symbols": ["HO", "HEATINGOIL", "HEATOIL"],
        "name": "Heating Oil Futures",
        "daily_volatility_percent": 3.0,
        "typical_spread": 0.0025,
        "risk_per_trade": 0.004,
        "lot_size": 0.10,
        "max_spread": 0.0040,
        "best_hours": "13:30-20:00",
        "best_months": "Nov-Mar",
        "volatility_factor": 3.5,
        "winter_demand": True,
        "northeast_weather": True,
        "inventory_sensitive": True,
        "min_confidence": 0.72,
        "profit_characteristics": "Polar vortex events create 8% spikes"
    }
}

# LIVESTOCK - DISEASE AND FEED VOLATILITY
LIVESTOCK_COMMODITIES = {
    "LIVECATTLE": {
        "symbol": "LE",
        "mt5_symbols": ["LE", "LC", "LIVECATTLE"],
        "name": "Live Cattle Futures",
        "daily_volatility_percent": 2.0,
        "typical_spread": 0.025,
        "risk_per_trade": 0.006,
        "lot_size": 0.20,
        "max_spread": 0.040,
        "best_hours": "13:30-20:00",
        "best_months": "May-Sep",
        "volatility_factor": 2.2,
        "grilling_season": True,
        "feed_cost_sensitive": True,
        "disease_risk": True,
        "min_confidence": 0.68,
        "profit_characteristics": "BBQ season + feed costs drive prices"
    },
    "FEEDERCATTLE": {
        "symbol": "GF",
        "mt5_symbols": ["GF", "FC", "FEEDERCATTLE"],
        "name": "Feeder Cattle Futures",
        "daily_volatility_percent": 2.4,
        "typical_spread": 0.050,
        "risk_per_trade": 0.005,
        "lot_size": 0.15,
        "max_spread": 0.075,
        "best_hours": "13:30-20:00",
        "best_months": "During droughts",
        "volatility_factor": 2.8,
        "pasture_conditions": True,
        "corn_correlation": 0.75,
        "min_confidence": 0.70,
        "profit_characteristics": "More volatile than live cattle"
    },
    "LEANHOGS": {
        "symbol": "HE",
        "mt5_symbols": ["HE", "LH", "LEANHOGS"],
        "name": "Lean Hogs Futures",
        "daily_volatility_percent": 3.0,
        "typical_spread": 0.050,
        "risk_per_trade": 0.004,
        "lot_size": 0.20,
        "max_spread": 0.075,
        "best_hours": "13:30-20:00",
        "best_months": "May-Jul, Nov-Dec",
        "volatility_factor": 3.5,
        "china_exports": True,
        "disease_outbreaks": True,
        "pork_cycle": True,
        "min_confidence": 0.72,
        "profit_characteristics": "China demand swings create 6% moves"
    }
}

# SOFT COMMODITIES - EXTREME WEATHER PLAYS
SOFT_COMMODITIES = {
    "ORANGEJUICE": {
        "symbol": "OJ",
        "mt5_symbols": ["OJ", "FCOJ", "OJUICE"],
        "name": "Orange Juice Futures",
        "daily_volatility_percent": 4.5,
        "typical_spread": 0.50,
        "risk_per_trade": 0.003,
        "lot_size": 0.10,
        "max_spread": 1.00,
        "best_hours": "13:30-20:00",
        "best_months": "Jun-Nov",
        "volatility_factor": 5.0,
        "hurricane_extreme": True,
        "citrus_disease": True,
        "florida_dependent": True,
        "min_confidence": 0.78,
        "profit_characteristics": "Hurricane threats create 15% daily moves"
    },
    "LUMBER": {
        "symbol": "LUMBER",
        "mt5_symbols": ["LUMBER", "LB", "LBS"],
        "name": "Lumber Futures",
        "daily_volatility_percent": 5.5,
        "typical_spread": 2.0,
        "risk_per_trade": 0.003,
        "lot_size": 0.05,
        "max_spread": 4.0,
        "best_hours": "13:30-20:00",
        "best_months": "Apr-Aug",
        "volatility_factor": 6.0,
        "housing_sensitive": True,
        "interest_rate_impact": True,
        "wildfire_risk": True,
        "min_confidence": 0.80,
        "profit_characteristics": "Can move 10%+ daily, extreme volatility"
    }
}

# RARE METALS (LIMITED AVAILABILITY)
RARE_METALS_INFO = {
    "LITHIUM": {
        "availability": "Not available as direct CFD",
        "alternatives": ["LIT ETF", "Lithium mining stocks"],
        "volatility": "5-8% daily in related stocks",
        "key_drivers": "EV demand, battery technology"
    },
    "COBALT": {
        "availability": "Not available as direct CFD",
        "alternatives": ["Cobalt mining stocks", "Battery metal ETFs"],
        "volatility": "4-6% daily in related stocks",
        "key_drivers": "Battery demand, Congo supply"
    },
    "URANIUM": {
        "availability": "Limited availability",
        "alternatives": ["URA ETF", "Uranium mining stocks"],
        "volatility": "3-5% daily",
        "key_drivers": "Nuclear policy, reactor construction"
    },
    "RHODIUM": {
        "availability": "Not available on MT5",
        "alternatives": ["Precious metals miners"],
        "volatility": "Extreme when available",
        "key_drivers": "Auto catalyst demand"
    }
}

# Combine all rare commodities
ALL_RARE_COMMODITIES = {
    **AGRICULTURAL_COMMODITIES,
    **ENERGY_PRODUCTS,
    **LIVESTOCK_COMMODITIES,
    **SOFT_COMMODITIES
}

# Seasonal trading calendar
COMMODITY_SEASONALITY = {
    "Q1": {
        "strong": ["HEATINGOIL", "NATGAS", "COCOA"],
        "weak": ["GASOLINE", "CORN"]
    },
    "Q2": {
        "strong": ["GASOLINE", "CORN", "WHEAT", "SOYBEANS"],
        "weak": ["HEATINGOIL", "NATGAS"]
    },
    "Q3": {
        "strong": ["CORN", "WHEAT", "LIVECATTLE", "GASOLINE"],
        "weak": ["HEATINGOIL", "COCOA"]
    },
    "Q4": {
        "strong": ["HEATINGOIL", "NATGAS", "LEANHOGS", "COCOA"],
        "weak": ["GASOLINE", "CORN"]
    }
}

# Weather event trading
WEATHER_EVENT_TRADING = {
    "hurricane": {
        "commodities": ["ORANGEJUICE", "GASOLINE", "NATGAS"],
        "typical_move": "5-15% daily",
        "season": "June-November"
    },
    "drought": {
        "commodities": ["CORN", "WHEAT", "SOYBEANS", "FEEDERCATTLE"],
        "typical_move": "3-8% daily",
        "season": "June-August"
    },
    "cold_snap": {
        "commodities": ["NATGAS", "HEATINGOIL", "ORANGEJUICE"],
        "typical_move": "5-10% daily",
        "season": "December-February"
    }
}

# Risk management for commodities
COMMODITY_RISK_MANAGEMENT = {
    "agricultural": {
        "max_positions": 2,
        "position_size": 0.005,
        "stop_loss": "2-3% from entry",
        "weather_hedge": True
    },
    "energy": {
        "max_positions": 1,
        "position_size": 0.004,
        "stop_loss": "3-4% from entry",
        "inventory_monitor": True
    },
    "livestock": {
        "max_positions": 2,
        "position_size": 0.006,
        "stop_loss": "2% from entry",
        "disease_monitor": True
    },
    "soft": {
        "max_positions": 1,
        "position_size": 0.003,
        "stop_loss": "4-5% from entry",
        "weather_extreme": True
    }
}

def get_commodities_by_volatility(min_volatility):
    """Get commodities with minimum daily volatility"""
    high_vol = {}
    for symbol, config in ALL_RARE_COMMODITIES.items():
        if config.get("daily_volatility_percent", 0) >= min_volatility:
            high_vol[symbol] = config
    return high_vol

def get_seasonal_commodities(quarter):
    """Get commodities strong in specific quarter"""
    season = COMMODITY_SEASONALITY.get(quarter, {})
    seasonal_commodities = {}
    for commodity in season.get("strong", []):
        if commodity in ALL_RARE_COMMODITIES:
            seasonal_commodities[commodity] = ALL_RARE_COMMODITIES[commodity]
    return seasonal_commodities

def get_weather_sensitive_commodities():
    """Get all weather-sensitive commodities"""
    weather_commodities = {}
    for symbol, config in ALL_RARE_COMMODITIES.items():
        if any(key in config for key in ["weather_sensitive", "hurricane_sensitive", 
                                          "weather_markets", "winter_demand", 
                                          "hurricane_extreme", "weather_extreme"]):
            weather_commodities[symbol] = config
    return weather_commodities