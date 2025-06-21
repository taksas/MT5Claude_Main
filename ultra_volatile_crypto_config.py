"""
Ultra Volatile Cryptocurrency CFDs Configuration for MT5
Focus on extreme volatility tokens with 50%+ daily movement potential
Based on 2024-2025 market research and broker availability
"""

# Extreme Volatility Meme Coins (Available on select MT5 brokers)
MEME_COINS_ULTRA = {
    "DOGEUSD": {
        "symbol": "DOGEUSD",
        "name": "Dogecoin vs USD",
        "category": "Meme Coin",
        "daily_volatility": "15-40%",
        "extreme_volatility_events": "50-100%+",
        "lot_size": 0.10,
        "max_spread": 0.003,
        "profit_target_percent": 10,
        "stop_loss_percent": 5,
        "best_trading_hours": "13:00-21:00 UTC",  # US session
        "social_media_driven": True,
        "mt5_availability": ["IC Markets", "Pepperstone", "OANDA", "Capital.com"],
        "key_drivers": [
            "Elon Musk tweets",
            "Social media virality",
            "Celebrity endorsements",
            "Reddit momentum"
        ],
        "risk_level": "extreme"
    },
    "SHIBUSD": {
        "symbol": "SHIBUSD",
        "name": "Shiba Inu vs USD",
        "category": "Meme Coin",
        "daily_volatility": "20-50%",
        "extreme_volatility_events": "70-150%+",
        "lot_size": 0.20,
        "max_spread": 0.00001,
        "profit_target_percent": 15,
        "stop_loss_percent": 7,
        "best_trading_hours": "00:00-08:00 UTC",  # Asian session
        "mt5_availability": ["Capital.com", "AvaTrade"],
        "key_drivers": [
            "Burn events",
            "Exchange listings",
            "Whale movements",
            "Community hype"
        ],
        "risk_level": "extreme"
    },
    "PEPEUSD": {
        "symbol": "PEPEUSD",
        "name": "Pepe vs USD",
        "category": "Meme Coin",
        "daily_volatility": "30-70%",
        "extreme_volatility_events": "100-300%+",
        "lot_size": 0.50,
        "max_spread": 0.0000001,
        "profit_target_percent": 20,
        "stop_loss_percent": 10,
        "best_trading_hours": "16:00-00:00 UTC",  # US session peak
        "mt5_availability": ["Limited brokers"],
        "key_drivers": [
            "Viral memes",
            "New exchange listings",
            "Influencer mentions",
            "FOMO rallies"
        ],
        "risk_level": "extreme"
    }
}

# AI-Related Crypto Projects (Growing sector)
AI_CRYPTO_PROJECTS = {
    "INJUSD": {
        "symbol": "INJUSD",
        "name": "Injective vs USD",
        "category": "AI/DeFi Infrastructure",
        "daily_volatility": "10-25%",
        "extreme_volatility_events": "40-80%+",
        "lot_size": 0.05,
        "max_spread": 0.10,
        "profit_target_percent": 12,
        "stop_loss_percent": 6,
        "best_trading_hours": "08:00-16:00 UTC",  # European session
        "mt5_availability": ["Select brokers"],
        "key_features": [
            "Layer 1 blockchain",
            "AI-powered DEX",
            "Cross-chain capabilities",
            "High-performance trading"
        ],
        "upcoming_catalysts": [
            "AI integration updates",
            "Partnership announcements",
            "DeFi TVL growth",
            "Mainnet upgrades"
        ],
        "risk_level": "high"
    },
    "RNDR": {
        "symbol": "RNDRUSD",
        "name": "Render Token vs USD",
        "category": "AI/GPU Computing",
        "daily_volatility": "12-30%",
        "extreme_volatility_events": "50-100%+",
        "lot_size": 0.10,
        "max_spread": 0.05,
        "profit_target_percent": 15,
        "stop_loss_percent": 7,
        "best_trading_hours": "16:00-00:00 UTC",
        "mt5_availability": ["Limited availability"],
        "key_features": [
            "GPU rendering network",
            "AI computation market",
            "Metaverse rendering",
            "Decentralized cloud"
        ],
        "risk_level": "high"
    }
}

# DeFi and Yield Farming Tokens (Limited MT5 availability)
DEFI_YIELD_TOKENS = {
    "AAVEUSD": {
        "symbol": "AAVEUSD",
        "name": "Aave vs USD",
        "category": "DeFi Lending",
        "daily_volatility": "8-20%",
        "extreme_volatility_events": "30-60%+",
        "lot_size": 0.02,
        "max_spread": 0.50,
        "profit_target_percent": 10,
        "stop_loss_percent": 5,
        "best_trading_hours": "12:00-20:00 UTC",
        "mt5_availability": ["Exness", "XM", "Pepperstone"],
        "defi_metrics": {
            "tvl_importance": "critical",
            "yield_sensitivity": "high",
            "protocol_risk": "medium"
        },
        "key_drivers": [
            "TVL changes",
            "Interest rate updates",
            "Protocol upgrades",
            "Market liquidations"
        ],
        "risk_level": "high"
    },
    "UNIUSD": {
        "symbol": "UNIUSD",
        "name": "Uniswap vs USD",
        "category": "DeFi DEX",
        "daily_volatility": "10-25%",
        "extreme_volatility_events": "40-80%+",
        "lot_size": 0.05,
        "max_spread": 0.10,
        "profit_target_percent": 12,
        "stop_loss_percent": 6,
        "best_trading_hours": "14:00-22:00 UTC",
        "mt5_availability": ["OANDA", "Capital.com", "AvaTrade"],
        "defi_metrics": {
            "volume_sensitivity": "extreme",
            "fee_revenue_impact": "high",
            "governance_news": "significant"
        },
        "key_drivers": [
            "DEX volume spikes",
            "V3 adoption metrics",
            "Regulatory news",
            "Layer 2 expansion"
        ],
        "risk_level": "high"
    },
    "LINKUSD": {
        "symbol": "LINKUSD",
        "name": "Chainlink vs USD",
        "category": "DeFi Oracle",
        "daily_volatility": "12-30%",
        "extreme_volatility_events": "50-100%+",
        "lot_size": 0.05,
        "max_spread": 0.15,
        "profit_target_percent": 15,
        "stop_loss_percent": 7,
        "best_trading_hours": "16:00-00:00 UTC",
        "mt5_availability": ["OANDA", "XM", "Pepperstone", "Capital.com"],
        "key_features": [
            "Oracle network leader",
            "Critical DeFi infrastructure",
            "Partnership driven",
            "Cross-chain data feeds"
        ],
        "upcoming_catalysts": [
            "Major protocol integrations",
            "CCIP expansion",
            "Enterprise partnerships",
            "Staking implementation"
        ],
        "risk_level": "high"
    }
}

# Gaming and Metaverse Tokens (Very limited on MT5)
GAMING_METAVERSE_TOKENS = {
    "MANAUSD": {
        "symbol": "MANAUSD",
        "name": "Decentraland vs USD",
        "category": "Metaverse",
        "daily_volatility": "15-35%",
        "extreme_volatility_events": "60-120%+",
        "lot_size": 0.20,
        "max_spread": 0.02,
        "profit_target_percent": 18,
        "stop_loss_percent": 9,
        "best_trading_hours": "Global 24/7",
        "mt5_availability": ["Very limited"],
        "metaverse_metrics": {
            "active_users": "critical",
            "land_sales_volume": "high",
            "partnership_news": "significant"
        },
        "key_drivers": [
            "Metaverse adoption news",
            "Major brand partnerships",
            "Virtual land sales",
            "Gaming updates"
        ],
        "risk_level": "extreme"
    },
    "SANDUSD": {
        "symbol": "SANDUSD",
        "name": "The Sandbox vs USD",
        "category": "Gaming/Metaverse",
        "daily_volatility": "18-40%",
        "extreme_volatility_events": "70-150%+",
        "lot_size": 0.30,
        "max_spread": 0.03,
        "profit_target_percent": 20,
        "stop_loss_percent": 10,
        "best_trading_hours": "Asian/US overlap",
        "mt5_availability": ["Very limited"],
        "gaming_metrics": {
            "player_count": "critical",
            "nft_volume": "high",
            "game_releases": "significant"
        },
        "risk_level": "extreme"
    }
}

# Cross-Chain and Bridge Tokens
CROSS_CHAIN_TOKENS = {
    "DOTUSD": {
        "symbol": "DOTUSD",
        "name": "Polkadot vs USD",
        "category": "Cross-Chain/Interoperability",
        "daily_volatility": "10-25%",
        "extreme_volatility_events": "40-80%+",
        "lot_size": 0.05,
        "max_spread": 0.15,
        "profit_target_percent": 12,
        "stop_loss_percent": 6,
        "best_trading_hours": "08:00-16:00 UTC",
        "mt5_availability": ["Pepperstone", "Capital.com", "XM"],
        "cross_chain_metrics": {
            "parachain_auctions": "critical",
            "ecosystem_growth": "high",
            "technical_updates": "significant"
        },
        "key_drivers": [
            "Parachain slot auctions",
            "Cross-chain activity",
            "Developer adoption",
            "Technical milestones"
        ],
        "risk_level": "high"
    },
    "ATOMUSD": {
        "symbol": "ATOMUSD",
        "name": "Cosmos vs USD",
        "category": "Cross-Chain/IBC",
        "daily_volatility": "12-28%",
        "extreme_volatility_events": "45-90%+",
        "lot_size": 0.08,
        "max_spread": 0.10,
        "profit_target_percent": 14,
        "stop_loss_percent": 7,
        "best_trading_hours": "12:00-20:00 UTC",
        "mt5_availability": ["Limited brokers"],
        "ibc_metrics": {
            "ibc_volume": "critical",
            "chain_connections": "high",
            "ecosystem_tvl": "significant"
        },
        "risk_level": "high"
    }
}

# Ultra High-Risk Emerging Tokens (Rarely on MT5)
EMERGING_ULTRA_VOLATILE = {
    "SOLAXY": {
        "symbol": "SOLX",
        "name": "Solaxy",
        "category": "Layer 2/Cross-Chain",
        "status": "Presale",
        "potential": "1000x+",
        "daily_volatility": "Expected 50-100%+",
        "mt5_availability": ["Not yet available"],
        "key_features": [
            "Solana Layer 2",
            "Cross-chain bridges",
            "DeFi integration",
            "Meme coin support"
        ],
        "risk_level": "extreme"
    },
    "FARTCOIN": {
        "symbol": "FARTCOIN",
        "name": "Fartcoin",
        "category": "Meme Coin",
        "market_cap": "$1.46B+",
        "daily_volatility": "40-80%+",
        "extreme_volatility_events": "100-300%+",
        "mt5_availability": ["Not available"],
        "key_features": [
            "Viral meme potential",
            "High volume spikes",
            "Community driven",
            "Extreme volatility"
        ],
        "risk_level": "extreme"
    }
}

# Master Ultra Volatile Configuration
ULTRA_VOLATILE_CRYPTOS = {
    **MEME_COINS_ULTRA,
    **AI_CRYPTO_PROJECTS,
    **DEFI_YIELD_TOKENS,
    **GAMING_METAVERSE_TOKENS,
    **CROSS_CHAIN_TOKENS
}

# Trading Settings for Ultra Volatile Cryptos
ULTRA_VOLATILE_SETTINGS = {
    "risk_per_trade": 0.0025,  # 0.25% - Very conservative due to extreme volatility
    "max_concurrent_trades": 1,  # Only one ultra volatile position at a time
    "confidence_threshold": 85,  # Very high threshold
    "scalping_timeframes": ["M1", "M5"],
    "day_trading_timeframes": ["M15", "H1"],
    "avoid_times": {
        "low_liquidity": ["03:00-07:00 UTC"],
        "weekend_gaps": ["Friday 21:00 - Sunday 21:00 UTC"]
    },
    "volatility_filters": {
        "min_atr_multiplier": 2.5,
        "max_spread_atr_ratio": 0.1,
        "volume_spike_threshold": 3.0
    },
    "risk_management": {
        "use_trailing_stop": True,
        "trailing_distance_percent": 5,
        "partial_profit_levels": [25, 50, 75],  # Take profit at these percentages
        "max_daily_loss_percent": 1.0,
        "emergency_stop_percent": 15
    }
}

# Volatility-Based Trading Strategies
VOLATILITY_STRATEGIES = {
    "meme_coin_spike": {
        "description": "Catch viral social media driven spikes",
        "entry_conditions": [
            "Social sentiment surge",
            "Volume spike > 300%",
            "Breaking key resistance",
            "RSI not overbought (< 70)"
        ],
        "exit_conditions": [
            "Trailing stop hit",
            "Volume exhaustion",
            "Sentiment reversal",
            "Technical breakdown"
        ],
        "timeframes": ["M1", "M5", "M15"],
        "risk_reward": "1:3 minimum"
    },
    "defi_tvl_momentum": {
        "description": "Trade on DeFi TVL and volume changes",
        "entry_conditions": [
            "TVL increase > 10%",
            "Volume above average",
            "Technical breakout",
            "Positive funding rates"
        ],
        "timeframes": ["H1", "H4"],
        "risk_reward": "1:2.5 minimum"
    },
    "ai_news_catalyst": {
        "description": "Trade AI token news and partnerships",
        "entry_conditions": [
            "Major partnership announcement",
            "Technical breakthrough news",
            "Volume surge confirmation",
            "Market sentiment positive"
        ],
        "timeframes": ["M15", "H1"],
        "risk_reward": "1:3 minimum"
    }
}

# Broker-Specific Crypto Offerings
MT5_BROKER_CRYPTO_SUMMARY = {
    "comprehensive_offerings": {
        "Capital.com": {
            "crypto_pairs": 141,
            "includes": ["Major", "DeFi", "Some meme coins"],
            "leverage": "Up to 1:100",
            "key_advantage": "Largest selection"
        },
        "Exness": {
            "crypto_pairs": 27,
            "includes": ["BTC", "ETH", "AAVE", "ADA", "BNB", "DOGE"],
            "leverage": "Varies",
            "key_advantage": "Good DeFi selection"
        },
        "Pepperstone": {
            "crypto_pairs": 15,
            "includes": ["Major cryptos", "DOT", "LINK"],
            "leverage": "Up to 1:200",
            "key_advantage": "Fast execution"
        }
    },
    "limited_offerings": {
        "OANDA": {
            "crypto_pairs": 8,
            "includes": ["ETH", "LTC", "BNB", "ADA", "LINK", "DOGE", "DOT", "UNI"],
            "leverage": "Varies",
            "key_advantage": "Regulated, reliable"
        },
        "IC Markets": {
            "crypto_pairs": 23,
            "leverage": "1:200 (MT5), 1:5 (cTrader)",
            "key_advantage": "Tight spreads"
        }
    }
}

# Risk Warnings and Requirements
EXTREME_RISK_WARNINGS = {
    "volatility_risks": [
        "50%+ daily price swings possible",
        "100%+ intraday moves during extreme events",
        "Liquidity can disappear instantly",
        "Spreads widen dramatically during volatility",
        "Gap risk especially on weekends"
    ],
    "specific_risks": {
        "meme_coins": "Social media manipulation, pump and dump schemes",
        "defi_tokens": "Smart contract bugs, protocol hacks, impermanent loss",
        "ai_tokens": "Hype-driven valuations, technology risk",
        "gaming_tokens": "User adoption risk, game failure risk",
        "cross_chain": "Bridge hacks, technical complexity"
    },
    "minimum_requirements": {
        "account_balance": 5000,  # USD minimum
        "recommended_balance": 10000,
        "experience_level": "Advanced only",
        "risk_tolerance": "Extreme",
        "monitoring": "24/7 position monitoring required"
    },
    "critical_rules": [
        "Never risk more than 0.25% per trade",
        "Always use stop losses",
        "Monitor social media sentiment",
        "Be aware of funding rates",
        "Check broker-specific crypto availability",
        "Understand CFDs don't give token ownership"
    ]
}

def get_extreme_volatility_cryptos(min_volatility=50):
    """Return cryptos with potential for extreme volatility moves"""
    extreme_cryptos = {}
    
    all_cryptos = {
        **ULTRA_VOLATILE_CRYPTOS,
        **EMERGING_ULTRA_VOLATILE
    }
    
    for symbol, data in all_cryptos.items():
        if "extreme_volatility_events" in data:
            # Extract max volatility from range
            max_vol = data["extreme_volatility_events"].replace("%", "").replace("+", "")
            max_vol_num = float(max_vol.split("-")[-1]) if "-" in max_vol else float(max_vol)
            
            if max_vol_num >= min_volatility:
                extreme_cryptos[symbol] = {
                    "name": data["name"],
                    "category": data["category"],
                    "extreme_volatility": data["extreme_volatility_events"],
                    "mt5_availability": data.get("mt5_availability", ["Not available"]),
                    "risk_level": data.get("risk_level", "extreme")
                }
    
    return extreme_cryptos

def get_mt5_available_volatile_cryptos():
    """Return only cryptos available on MT5 brokers"""
    mt5_cryptos = {}
    
    for symbol, data in ULTRA_VOLATILE_CRYPTOS.items():
        if "mt5_availability" in data and data["mt5_availability"][0] not in ["Not available", "Very limited", "Limited brokers", "Not yet available"]:
            mt5_cryptos[symbol] = data
    
    return mt5_cryptos

# Example usage
if __name__ == "__main__":
    print("=== EXTREME VOLATILITY CRYPTOS (50%+ Potential) ===")
    extreme = get_extreme_volatility_cryptos(50)
    for symbol, data in extreme.items():
        print(f"\n{symbol}:")
        print(f"  Name: {data['name']}")
        print(f"  Category: {data['category']}")
        print(f"  Extreme Volatility: {data['extreme_volatility']}")
        print(f"  MT5 Availability: {', '.join(data['mt5_availability'])}")
        print(f"  Risk Level: {data['risk_level']}")
    
    print("\n\n=== MT5 AVAILABLE HIGH VOLATILITY CRYPTOS ===")
    mt5_available = get_mt5_available_volatile_cryptos()
    for symbol, data in mt5_available.items():
        print(f"\n{symbol}: {data['name']}")
        print(f"  Daily Volatility: {data['daily_volatility']}")
        print(f"  Brokers: {', '.join(data['mt5_availability'])}")