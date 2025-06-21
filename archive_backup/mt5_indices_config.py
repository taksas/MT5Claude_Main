"""
MT5 High-Profit Indices Configuration
Tradeable indices with high volatility and profit potential
"""

# Major US Indices
US_INDICES = {
    "US30": {  # Dow Jones Industrial Average
        "name": "Dow Jones 30",
        "avg_daily_movement": 1.2,  # Percentage
        "volatility": "Medium-High",
        "trading_hours": "23:00-21:15 GMT (with breaks)",
        "spread_range": "2-5 points",
        "characteristics": [
            "Blue-chip stocks focus",
            "Less volatile than NASDAQ",
            "Good for trend following",
            "Strong correlation with US economy"
        ],
        "best_sessions": ["US Open", "US Close"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "US100": {  # NASDAQ 100
        "name": "NASDAQ 100",
        "avg_daily_movement": 1.8,
        "volatility": "High",
        "trading_hours": "23:00-21:15 GMT",
        "spread_range": "1-3 points",
        "characteristics": [
            "Tech-heavy index",
            "Higher volatility than US30",
            "Excellent for scalping",
            "Sensitive to tech earnings"
        ],
        "best_sessions": ["US Pre-market", "US Open"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "US500": {  # S&P 500
        "name": "S&P 500",
        "avg_daily_movement": 1.4,
        "volatility": "Medium-High",
        "trading_hours": "23:00-21:15 GMT",
        "spread_range": "0.3-1 points",
        "characteristics": [
            "Most liquid US index",
            "Broad market representation",
            "Tightest spreads",
            "Best for large positions"
        ],
        "best_sessions": ["US Open", "FOMC days"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "RUSSELL2K": {  # Russell 2000
        "name": "Russell 2000",
        "avg_daily_movement": 2.1,
        "volatility": "Very High",
        "trading_hours": "23:00-21:15 GMT",
        "spread_range": "0.3-0.8 points",
        "characteristics": [
            "Small-cap index",
            "Highest volatility US index",
            "Great for aggressive trading",
            "Less correlated with major indices"
        ],
        "best_sessions": ["US Open", "Economic data releases"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    }
}

# European Indices
EUROPEAN_INDICES = {
    "DAX40": {  # German DAX
        "name": "Germany 40",
        "avg_daily_movement": 1.5,
        "volatility": "High",
        "trading_hours": "07:00-21:00 GMT",
        "spread_range": "0.8-2 points",
        "characteristics": [
            "Europe's most traded index",
            "Auto sector sensitive",
            "High volatility during EU session",
            "Correlates with EUR strength"
        ],
        "best_sessions": ["EU Open", "US Open overlap"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "FTSE100": {  # UK FTSE 100
        "name": "UK 100",
        "avg_daily_movement": 1.2,
        "volatility": "Medium",
        "trading_hours": "08:00-21:00 GMT",
        "spread_range": "1-2 points",
        "characteristics": [
            "GBP sensitive",
            "Mining and banking heavy",
            "Less volatile than DAX",
            "Brexit news sensitive"
        ],
        "best_sessions": ["London Open", "UK data releases"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "STOXX50": {  # Euro STOXX 50
        "name": "Europe 50",
        "avg_daily_movement": 1.3,
        "volatility": "Medium-High",
        "trading_hours": "07:00-21:00 GMT",
        "spread_range": "1-3 points",
        "characteristics": [
            "Pan-European blue chips",
            "EUR correlation",
            "ECB policy sensitive",
            "Good diversification"
        ],
        "best_sessions": ["EU Open", "ECB announcements"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "MDAX": {  # German Mid-Cap
        "name": "Germany Mid-Cap 50",
        "avg_daily_movement": 1.8,
        "volatility": "High",
        "trading_hours": "07:00-17:30 GMT",
        "spread_range": "3-8 points",
        "characteristics": [
            "German mid-cap companies",
            "Higher volatility than DAX",
            "Less liquid, wider spreads",
            "Good for volatility plays"
        ],
        "best_sessions": ["EU Open", "German data"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    }
}

# Asian Indices
ASIAN_INDICES = {
    "JP225": {  # Nikkei 225
        "name": "Japan 225",
        "avg_daily_movement": 1.6,
        "volatility": "High",
        "trading_hours": "23:00-21:15 GMT",
        "spread_range": "3-10 points",
        "characteristics": [
            "JPY correlation",
            "Tech and export heavy",
            "BOJ policy sensitive",
            "Weekend gap opportunities"
        ],
        "best_sessions": ["Tokyo Open", "US market correlation"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "HK50": {  # Hang Seng
        "name": "Hong Kong 50",
        "avg_daily_movement": 2.0,
        "volatility": "Very High",
        "trading_hours": "01:15-04:00, 05:00-08:30 GMT",
        "spread_range": "5-15 points",
        "characteristics": [
            "China proxy play",
            "Finance sector heavy",
            "Political risk sensitive",
            "High overnight volatility"
        ],
        "best_sessions": ["HK Open", "China data releases"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    },
    
    "AUS200": {  # ASX 200
        "name": "Australia 200",
        "avg_daily_movement": 1.3,
        "volatility": "Medium",
        "trading_hours": "22:50-21:00 GMT",
        "spread_range": "1-3 points",
        "characteristics": [
            "Commodity sensitive",
            "Mining sector heavy",
            "AUD correlation",
            "China demand proxy"
        ],
        "best_sessions": ["Sydney Open", "China PMI days"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    }
}

# Volatility Indices
VOLATILITY_INDICES = {
    "VIX": {  # CBOE Volatility Index
        "name": "Volatility Index",
        "avg_daily_movement": 5.5,
        "volatility": "Extreme",
        "trading_hours": "23:00-21:15 GMT",
        "spread_range": "0.05-0.15 points",
        "characteristics": [
            "Fear gauge",
            "Inverse market correlation",
            "Spikes during uncertainty",
            "Mean reverting"
        ],
        "best_sessions": ["Market stress periods", "Fed days"],
        "min_lot": 0.1,
        "contract_size": 1000,
        "margin_requirement": "High"
    },
    
    "VXX": {  # VIX ETN
        "name": "VIX Short-Term Futures",
        "avg_daily_movement": 3.8,
        "volatility": "Very High",
        "trading_hours": "Market hours",
        "spread_range": "0.01-0.05 points",
        "characteristics": [
            "VIX futures tracker",
            "Decay over time",
            "Volatility of volatility",
            "Short-term hedging tool"
        ],
        "best_sessions": ["High volatility days"],
        "min_lot": 1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    }
}

# Emerging Market Indices
EMERGING_INDICES = {
    "BVSP": {  # Brazil Bovespa
        "name": "Brazil Index",
        "avg_daily_movement": 2.3,
        "volatility": "Very High",
        "trading_hours": "13:00-20:00 GMT",
        "spread_range": "20-50 points",
        "characteristics": [
            "BRL sensitive",
            "Commodity correlation",
            "Political risk premium",
            "High carry trade impact"
        ],
        "best_sessions": ["Brazil Open", "FOMC impact"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "High"
    },
    
    "MEX35": {  # Mexico IPC
        "name": "Mexico 35",
        "avg_daily_movement": 1.9,
        "volatility": "High",
        "trading_hours": "14:30-21:00 GMT",
        "spread_range": "15-40 points",
        "characteristics": [
            "USMCA sensitive",
            "Peso correlation",
            "Oil price impact",
            "US economy proxy"
        ],
        "best_sessions": ["Mexico Open", "US data"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    },
    
    "INDIA50": {  # Nifty 50
        "name": "India 50",
        "avg_daily_movement": 1.7,
        "volatility": "High",
        "trading_hours": "03:45-10:00 GMT",
        "spread_range": "2-8 points",
        "characteristics": [
            "IT sector heavy",
            "Monsoon sensitive",
            "FII flow dependent",
            "Rupee correlation"
        ],
        "best_sessions": ["Mumbai Open", "RBI policy days"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Medium"
    }
}

# Sector Indices
SECTOR_INDICES = {
    "TECH100": {  # Technology Sector
        "name": "Tech Sector Index",
        "avg_daily_movement": 2.2,
        "volatility": "Very High",
        "trading_hours": "Market hours",
        "spread_range": "2-5 points",
        "characteristics": [
            "FAANG heavy",
            "Earnings volatility",
            "Growth sensitive",
            "Rate sensitive"
        ],
        "best_sessions": ["Tech earnings season"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    },
    
    "BANK": {  # Banking Sector
        "name": "Banking Index",
        "avg_daily_movement": 1.8,
        "volatility": "High",
        "trading_hours": "Market hours",
        "spread_range": "1-3 points",
        "characteristics": [
            "Interest rate sensitive",
            "Regulation impact",
            "Credit cycle proxy",
            "Dividend plays"
        ],
        "best_sessions": ["Fed days", "Bank earnings"],
        "min_lot": 0.1,
        "contract_size": 1,
        "margin_requirement": "Low"
    }
}

# Recommended High-Profit Index Combinations
HIGH_PROFIT_COMBINATIONS = [
    {
        "name": "Volatility Breakout Strategy",
        "indices": ["RUSSELL2K", "VIX", "HK50"],
        "timeframe": "M5-M15",
        "best_hours": "Market open/close",
        "expected_daily_movement": "2-5%"
    },
    {
        "name": "Cross-Continental Arbitrage",
        "indices": ["DAX40", "US100", "JP225"],
        "timeframe": "H1-H4",
        "best_hours": "Session overlaps",
        "expected_daily_movement": "1.5-3%"
    },
    {
        "name": "Emerging Market Momentum",
        "indices": ["BVSP", "INDIA50", "MEX35"],
        "timeframe": "M30-H1",
        "best_hours": "Local market hours",
        "expected_daily_movement": "2-4%"
    },
    {
        "name": "Small-Cap Volatility Play",
        "indices": ["RUSSELL2K", "MDAX", "AUS200"],
        "timeframe": "M15-M30",
        "best_hours": "Economic releases",
        "expected_daily_movement": "1.8-3.5%"
    }
]

# Trading Recommendations by Market Condition
MARKET_CONDITIONS = {
    "high_volatility": {
        "preferred_indices": ["VIX", "RUSSELL2K", "HK50", "BVSP"],
        "strategy": "Breakout with tight stops",
        "position_size": "Reduced (50%)",
        "timeframe": "M5-M15"
    },
    "trending": {
        "preferred_indices": ["US100", "DAX40", "JP225"],
        "strategy": "Trend following with pyramiding",
        "position_size": "Normal",
        "timeframe": "H1-H4"
    },
    "ranging": {
        "preferred_indices": ["US500", "FTSE100", "AUS200"],
        "strategy": "Support/Resistance trading",
        "position_size": "Increased (150%)",
        "timeframe": "M30-H1"
    },
    "risk_off": {
        "preferred_indices": ["VIX", "JP225", "US30"],
        "strategy": "Long volatility, short risk",
        "position_size": "Defensive (75%)",
        "timeframe": "H1-D1"
    }
}

# Risk Management for Indices
INDEX_RISK_PARAMS = {
    "max_position_size": {
        "high_volatility": 0.5,  # Half normal size
        "medium_volatility": 1.0,  # Normal size
        "low_volatility": 1.5  # 1.5x normal size
    },
    "stop_loss_points": {
        "US30": 30,
        "US100": 40,
        "US500": 10,
        "RUSSELL2K": 15,
        "DAX40": 30,
        "JP225": 50,
        "VIX": 0.5,
        "default": 25
    },
    "take_profit_multiplier": 2.0,  # 2x stop loss
    "max_daily_trades_per_index": 3,
    "correlation_limit": 0.7  # Max correlation between positions
}

# Integration with MT5
MT5_INDEX_SYMBOLS = {
    # US Indices
    "US30": ["US30", "US30Cash", "US30.cash", "DJ30", "DJI"],
    "US100": ["US100", "US100Cash", "NAS100", "NASDAQ", "NDX"],
    "US500": ["US500", "US500Cash", "SP500", "SPX", "S&P500"],
    "RUSSELL2K": ["RUSSELL2000", "US2000", "RUT", "Russell2K"],
    
    # European Indices
    "DAX40": ["DAX40", "DAX30", "GER40", "GER30", "DE40"],
    "FTSE100": ["FTSE100", "UK100", "FTSE", "UKX"],
    "STOXX50": ["STOXX50", "EU50", "ESTX50", "EUR50"],
    "MDAX": ["MDAX", "MDAX50", "GERMid50"],
    
    # Asian Indices
    "JP225": ["JP225", "JPN225", "Nikkei225", "N225"],
    "HK50": ["HK50", "HongKong50", "HSI", "HangSeng"],
    "AUS200": ["AUS200", "Australia200", "ASX200", "AUS200Cash"],
    
    # Other Indices
    "VIX": ["VIX", "Volatility", "VIX.cash", "CBOE_VIX"]
}

def get_mt5_symbol(index_key, broker="default"):
    """Get the correct MT5 symbol for an index based on broker"""
    if index_key in MT5_INDEX_SYMBOLS:
        return MT5_INDEX_SYMBOLS[index_key][0]  # Return first/default symbol
    return None

def get_index_config(symbol):
    """Get configuration for a specific index"""
    all_indices = {
        **US_INDICES,
        **EUROPEAN_INDICES,
        **ASIAN_INDICES,
        **VOLATILITY_INDICES,
        **EMERGING_INDICES,
        **SECTOR_INDICES
    }
    return all_indices.get(symbol, None)

def get_high_volatility_indices():
    """Get list of high volatility indices for aggressive trading"""
    high_vol = []
    all_indices = {
        **US_INDICES,
        **EUROPEAN_INDICES,
        **ASIAN_INDICES,
        **VOLATILITY_INDICES,
        **EMERGING_INDICES,
        **SECTOR_INDICES
    }
    
    for symbol, config in all_indices.items():
        if config["avg_daily_movement"] >= 1.8:
            high_vol.append(symbol)
    
    return high_vol

# Example usage for integration
if __name__ == "__main__":
    print("High Volatility Indices for MT5 Trading:")
    print("=" * 50)
    
    high_vol = get_high_volatility_indices()
    for symbol in high_vol:
        config = get_index_config(symbol)
        mt5_symbol = get_mt5_symbol(symbol)
        print(f"\n{symbol} ({config['name']}):")
        print(f"  MT5 Symbol: {mt5_symbol}")
        print(f"  Avg Daily Move: {config['avg_daily_movement']}%")
        print(f"  Volatility: {config['volatility']}")
        print(f"  Best Sessions: {', '.join(config['best_sessions'])}")