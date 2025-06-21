"""
Cryptocurrency CFDs Configuration for MT5 Trading
High-volatility crypto assets for profitable trading opportunities
"""

CRYPTO_CFDS = {
    # Major Cryptocurrencies
    "BTCUSD": {
        "name": "Bitcoin vs USD",
        "category": "Major Crypto",
        "daily_volatility": "3-5%",
        "high_volatility_range": "5-15%",
        "typical_spread": "15-30 USD",
        "best_sessions": ["Asian", "European", "US"],
        "24_7_trading": True,
        "mt5_symbols": ["BTCUSD", "BTCUSD.m", "Bitcoin"],
        "key_characteristics": [
            "Most liquid crypto",
            "Strong trend movements",
            "High correlation with crypto market",
            "Institutional interest"
        ],
        "trading_tips": [
            "Watch for breakouts above/below key levels",
            "Higher volatility during US market hours",
            "Weekend gaps common"
        ]
    },
    
    "ETHUSD": {
        "name": "Ethereum vs USD",
        "category": "Major Crypto",
        "daily_volatility": "4-6%",
        "high_volatility_range": "7-20%",
        "typical_spread": "2-5 USD",
        "best_sessions": ["European", "US"],
        "24_7_trading": True,
        "mt5_symbols": ["ETHUSD", "ETHUSD.m", "Ethereum"],
        "key_characteristics": [
            "DeFi ecosystem leader",
            "Higher volatility than BTC",
            "Strong technical patterns",
            "Smart contract platform"
        ],
        "trading_tips": [
            "More volatile than Bitcoin",
            "Strong moves during DeFi news",
            "Watch ETH/BTC ratio"
        ]
    },
    
    "XRPUSD": {
        "name": "Ripple vs USD",
        "category": "Major Crypto",
        "daily_volatility": "5-8%",
        "high_volatility_range": "10-25%",
        "typical_spread": "0.005-0.01 USD",
        "best_sessions": ["Asian", "European"],
        "24_7_trading": True,
        "mt5_symbols": ["XRPUSD", "XRPUSD.m", "Ripple"],
        "key_characteristics": [
            "Banking partnerships",
            "Regulatory news sensitive",
            "High liquidity",
            "Quick price spikes"
        ],
        "trading_tips": [
            "Very news-sensitive",
            "Watch for SEC developments",
            "Strong Asian market interest"
        ]
    },
    
    "LTCUSD": {
        "name": "Litecoin vs USD",
        "category": "Major Crypto",
        "daily_volatility": "4-7%",
        "high_volatility_range": "8-20%",
        "typical_spread": "0.5-1 USD",
        "best_sessions": ["European", "US"],
        "24_7_trading": True,
        "mt5_symbols": ["LTCUSD", "LTCUSD.m", "Litecoin"],
        "key_characteristics": [
            "Bitcoin alternative",
            "Halving cycles",
            "Payment focus",
            "Established crypto"
        ],
        "trading_tips": [
            "Often follows Bitcoin",
            "Watch halving events",
            "Good for range trading"
        ]
    },
    
    # High-Volatility Altcoins
    "DOGEUSD": {
        "name": "Dogecoin vs USD",
        "category": "Meme Coin",
        "daily_volatility": "7-12%",
        "high_volatility_range": "15-40%",
        "typical_spread": "0.001-0.002 USD",
        "best_sessions": ["US", "Asian"],
        "24_7_trading": True,
        "mt5_symbols": ["DOGEUSD", "DOGEUSD.m", "Dogecoin"],
        "key_characteristics": [
            "Social media driven",
            "Extreme volatility",
            "Retail favorite",
            "Elon Musk influence"
        ],
        "trading_tips": [
            "Monitor social media sentiment",
            "Extreme intraday swings",
            "High risk/reward"
        ]
    },
    
    "ADAUSD": {
        "name": "Cardano vs USD",
        "category": "Smart Contract Platform",
        "daily_volatility": "5-9%",
        "high_volatility_range": "10-30%",
        "typical_spread": "0.01-0.02 USD",
        "best_sessions": ["European", "US"],
        "24_7_trading": True,
        "mt5_symbols": ["ADAUSD", "ADAUSD.m", "Cardano"],
        "key_characteristics": [
            "Academic approach",
            "Proof of Stake",
            "Development milestones",
            "Strong community"
        ],
        "trading_tips": [
            "Watch development updates",
            "Strong technical patterns",
            "Good for swing trading"
        ]
    },
    
    "DOTUSD": {
        "name": "Polkadot vs USD",
        "category": "Interoperability",
        "daily_volatility": "6-10%",
        "high_volatility_range": "12-35%",
        "typical_spread": "0.05-0.1 USD",
        "best_sessions": ["European", "US"],
        "24_7_trading": True,
        "mt5_symbols": ["DOTUSD", "DOTUSD.m", "Polkadot"],
        "key_characteristics": [
            "Cross-chain focus",
            "Parachain auctions",
            "High volatility",
            "Technical innovation"
        ],
        "trading_tips": [
            "Parachain news impacts price",
            "Strong breakout potential",
            "Watch ecosystem growth"
        ]
    },
    
    "LINKUSD": {
        "name": "Chainlink vs USD",
        "category": "Oracle/DeFi",
        "daily_volatility": "6-11%",
        "high_volatility_range": "12-40%",
        "typical_spread": "0.05-0.1 USD",
        "best_sessions": ["US", "European"],
        "24_7_trading": True,
        "mt5_symbols": ["LINKUSD", "LINKUSD.m", "Chainlink"],
        "key_characteristics": [
            "Oracle leader",
            "DeFi infrastructure",
            "Partnership driven",
            "High volatility"
        ],
        "trading_tips": [
            "Partnership announcements key",
            "DeFi sector correlation",
            "Strong trending behavior"
        ]
    },
    
    "AVAXUSD": {
        "name": "Avalanche vs USD",
        "category": "Smart Contract Platform",
        "daily_volatility": "7-13%",
        "high_volatility_range": "15-45%",
        "typical_spread": "0.1-0.2 USD",
        "best_sessions": ["US", "Asian"],
        "24_7_trading": True,
        "mt5_symbols": ["AVAXUSD", "AVAXUSD.m", "Avalanche"],
        "key_characteristics": [
            "High-speed transactions",
            "4,500 TPS capability",
            "DeFi ecosystem",
            "Extreme volatility"
        ],
        "trading_tips": [
            "Very high volatility",
            "Strong momentum moves",
            "Watch DeFi TVL metrics"
        ]
    },
    
    # DeFi Tokens
    "UNIUSD": {
        "name": "Uniswap vs USD",
        "category": "DeFi/DEX",
        "daily_volatility": "8-15%",
        "high_volatility_range": "15-50%",
        "typical_spread": "0.05-0.15 USD",
        "best_sessions": ["US", "European"],
        "24_7_trading": True,
        "mt5_symbols": ["UNIUSD", "UNIUSD.m", "Uniswap"],
        "key_characteristics": [
            "Leading DEX",
            "DeFi bluechip",
            "Volume driven",
            "Governance token"
        ],
        "trading_tips": [
            "DeFi sector leader",
            "Watch DEX volumes",
            "Regulatory news sensitive"
        ]
    },
    
    "SOLUSD": {
        "name": "Solana vs USD",
        "category": "Smart Contract Platform",
        "daily_volatility": "8-14%",
        "high_volatility_range": "15-50%",
        "typical_spread": "0.1-0.3 USD",
        "best_sessions": ["US", "Asian"],
        "24_7_trading": True,
        "mt5_symbols": ["SOLUSD", "SOLUSD.m", "Solana"],
        "key_characteristics": [
            "High-speed blockchain",
            "NFT ecosystem",
            "Network outage risk",
            "Extreme volatility"
        ],
        "trading_tips": [
            "Network stability issues",
            "Strong trending moves",
            "NFT market correlation"
        ]
    },
    
    "MATICUSD": {
        "name": "Polygon vs USD",
        "category": "Layer 2/Scaling",
        "daily_volatility": "7-12%",
        "high_volatility_range": "15-40%",
        "typical_spread": "0.01-0.02 USD",
        "best_sessions": ["Asian", "European"],
        "24_7_trading": True,
        "mt5_symbols": ["MATICUSD", "MATICUSD.m", "Polygon"],
        "key_characteristics": [
            "Ethereum scaling",
            "Enterprise adoption",
            "High volatility",
            "Partnership driven"
        ],
        "trading_tips": [
            "ETH correlation",
            "Partnership news key",
            "Good breakout trades"
        ]
    }
}

# Top MT5 Brokers for Crypto CFDs
MT5_CRYPTO_BROKERS = {
    "IC Markets": {
        "crypto_pairs": 23,
        "leverage": "1:200 (MT5), 1:5 (cTrader)",
        "min_spread_btc": "$7.32",
        "regulation": ["ASIC", "CySEC"],
        "features": ["Tight spreads", "Fast execution", "Multiple platforms"]
    },
    
    "Pepperstone": {
        "crypto_pairs": 15,
        "leverage": "1:200",
        "platforms": ["MT4", "MT5", "cTrader"],
        "regulation": ["FCA", "ASIC", "CySEC"],
        "features": ["1200+ instruments", "Fast execution", "Expert Advisors"]
    },
    
    "FP Markets": {
        "crypto_pairs": 12,
        "leverage": "1:200",
        "platforms": ["MT4", "MT5", "IRESS"],
        "regulation": ["ASIC", "CySEC"],
        "features": ["Expert Advisors", "Fast execution", "Low spreads"]
    },
    
    "AvaTrade": {
        "crypto_pairs": 20,
        "leverage": "1:100",
        "platforms": ["MT4", "MT5", "AvaTradeGO"],
        "regulation": ["Central Bank of Ireland", "ASIC", "CySEC"],
        "features": ["Established broker", "Multiple regulations", "Educational resources"]
    },
    
    "Capital.com": {
        "crypto_pairs": 141,
        "leverage": "1:100",
        "platforms": ["MT4", "Proprietary"],
        "regulation": ["CySEC", "FCA", "ASIC"],
        "features": ["Largest crypto selection", "AI-powered insights", "No commissions"]
    },
    
    "Bybit MT5": {
        "crypto_pairs": "Multiple",
        "leverage": "Up to 500x",
        "platforms": ["MT5"],
        "features": ["High leverage", "Comprehensive instruments", "24/7 trading"]
    }
}

# Trading Session Characteristics
CRYPTO_TRADING_SESSIONS = {
    "Asian": {
        "hours": "00:00-08:00 UTC",
        "characteristics": [
            "Lower volatility typically",
            "XRP and Asian projects active",
            "Weekend trading continuation",
            "Arbitrage opportunities"
        ]
    },
    
    "European": {
        "hours": "08:00-16:00 UTC",
        "characteristics": [
            "Increasing volatility",
            "Institutional activity",
            "Major news releases",
            "Strong trending moves"
        ]
    },
    
    "US": {
        "hours": "16:00-00:00 UTC",
        "characteristics": [
            "Highest volatility",
            "Major price moves",
            "Institutional and retail mix",
            "News-driven volatility"
        ]
    },
    
    "Weekend": {
        "hours": "Saturday-Sunday",
        "characteristics": [
            "Lower liquidity",
            "Gap risk",
            "Retail-driven moves",
            "Technical setups"
        ]
    }
}

# High-Profit Trading Strategies for Crypto CFDs
CRYPTO_TRADING_STRATEGIES = {
    "breakout_volatility": {
        "description": "Trade breakouts during high volatility periods",
        "suitable_for": ["BTCUSD", "ETHUSD", "AVAXUSD", "SOLUSD"],
        "timeframes": ["M15", "H1", "H4"],
        "indicators": ["ATR", "Bollinger Bands", "Volume"],
        "entry_rules": [
            "Wait for consolidation",
            "Volume spike on breakout",
            "ATR > 2x average"
        ]
    },
    
    "momentum_scalping": {
        "description": "Quick trades on momentum surges",
        "suitable_for": ["DOGEUSD", "AVAXUSD", "LINKUSD", "UNIUSD"],
        "timeframes": ["M1", "M5", "M15"],
        "indicators": ["RSI", "MACD", "Volume Profile"],
        "entry_rules": [
            "Strong directional move",
            "Volume confirmation",
            "Quick profit targets"
        ]
    },
    
    "range_trading": {
        "description": "Trade established ranges in consolidation",
        "suitable_for": ["LTCUSD", "XRPUSD", "ADAUSD"],
        "timeframes": ["H1", "H4", "D1"],
        "indicators": ["Support/Resistance", "RSI", "Stochastic"],
        "entry_rules": [
            "Clear range boundaries",
            "Oversold/overbought conditions",
            "Volume at extremes"
        ]
    },
    
    "news_trading": {
        "description": "Trade on crypto news and announcements",
        "suitable_for": ["All pairs"],
        "timeframes": ["M5", "M15", "H1"],
        "indicators": ["Volume", "Price Action", "News Calendar"],
        "entry_rules": [
            "Major news release",
            "Immediate price reaction",
            "Volume surge"
        ]
    }
}

# Risk Management for Crypto CFDs
CRYPTO_RISK_MANAGEMENT = {
    "position_sizing": {
        "max_risk_per_trade": "1-2%",
        "leverage_usage": "Conservative (1:10-1:50)",
        "volatility_adjustment": "Reduce size in extreme volatility"
    },
    
    "stop_loss_guidelines": {
        "scalping": "0.5-1% from entry",
        "day_trading": "2-3% from entry",
        "swing_trading": "5-7% from entry",
        "weekend_positions": "Wider stops or close before weekend"
    },
    
    "profit_targets": {
        "scalping": "1:1 to 1:2 risk/reward",
        "day_trading": "1:2 to 1:3 risk/reward",
        "swing_trading": "1:3 to 1:5 risk/reward"
    },
    
    "max_exposure": {
        "total_crypto_exposure": "20-30% of portfolio",
        "single_crypto_exposure": "5-10% of portfolio",
        "correlation_management": "Diversify across different crypto categories"
    }
}

def get_high_volatility_cryptos():
    """Return cryptos with daily volatility > 5%"""
    high_vol = {}
    for symbol, data in CRYPTO_CFDS.items():
        # Extract minimum volatility percentage
        vol_range = data["daily_volatility"]
        min_vol = float(vol_range.split("-")[0].replace("%", ""))
        if min_vol >= 5:
            high_vol[symbol] = data
    return high_vol

def get_best_session_cryptos(session):
    """Return cryptos best traded in specific session"""
    session_cryptos = {}
    for symbol, data in CRYPTO_CFDS.items():
        if session in data["best_sessions"]:
            session_cryptos[symbol] = data
    return session_cryptos

def get_defi_tokens():
    """Return DeFi-related crypto CFDs"""
    defi_tokens = {}
    for symbol, data in CRYPTO_CFDS.items():
        if "DeFi" in data["category"] or "DEX" in data["category"]:
            defi_tokens[symbol] = data
    return defi_tokens

# Example usage
if __name__ == "__main__":
    print("=== HIGH VOLATILITY CRYPTOS (>5% daily) ===")
    high_vol = get_high_volatility_cryptos()
    for symbol, data in high_vol.items():
        print(f"{symbol}: {data['name']} - Volatility: {data['daily_volatility']}")
    
    print("\n=== BEST US SESSION CRYPTOS ===")
    us_cryptos = get_best_session_cryptos("US")
    for symbol in us_cryptos:
        print(f"{symbol}: {us_cryptos[symbol]['name']}")
    
    print("\n=== DEFI TOKENS ===")
    defi = get_defi_tokens()
    for symbol in defi:
        print(f"{symbol}: {defi[symbol]['name']} - Category: {defi[symbol]['category']}")