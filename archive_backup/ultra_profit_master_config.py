# Ultra Profit Master Configuration
# Integrates all high-profit symbols discovered through ultra-thinking

import os
import sys
from datetime import datetime

# Import all configuration modules
from high_profit_symbols_config import ALL_HIGH_PROFIT_SYMBOLS
from exotic_forex_ultra_config import ALL_EXOTIC_FOREX
from rare_commodities_config import ALL_RARE_COMMODITIES
from synthetic_indices_complete_config import ALL_SYNTHETIC_INDICES
from crypto_cfds_config import CRYPTO_CFDS

# MASTER SYMBOL DATABASE
ULTRA_PROFIT_SYMBOLS = {
    **ALL_HIGH_PROFIT_SYMBOLS,  # Original high-profit symbols
    **ALL_EXOTIC_FOREX,         # New exotic forex pairs
    **ALL_RARE_COMMODITIES,     # Rare commodities
    **ALL_SYNTHETIC_INDICES,    # Complete synthetic indices
    **CRYPTO_CFDS              # Crypto CFDs
}

# VOLATILITY TIERS (Updated with new symbols)
VOLATILITY_TIERS = {
    "EXTREME": {  # >4% daily or >1000 pips
        "symbols": [
            # Forex
            "USDTRY", "USDZAR", "EURTRY", "USDBRL", "USDNGN", "USDSEK",
            "TRYJPY", "ZARJPY", "USDHUF", "USDCLP",
            # Commodities  
            "ORANGEJUICE", "LUMBER", "GASOLINE",
            # Crypto
            "DOGEUSD", "AVAXUSD", "SOLUSD", "UNIUSD",
            # Synthetics
            "V200", "V300", "CRASH1000", "BOOM1000", "JUMP100"
        ],
        "min_confidence": 0.80,
        "max_positions": 1,
        "risk_per_trade": 0.001
    },
    "VERY_HIGH": {  # 3-4% daily or 500-1000 pips
        "symbols": [
            # Forex
            "USDMXN", "EURNOK", "USDNOK", "USDCOP", "EURHUF", "USDPLN",
            "USDGHS", "USDKES", "USDTHB", "USDPHP",
            # Commodities
            "COCOA", "WHEAT", "HEATINGOIL", "LEANHOGS",
            # Crypto
            "BTCUSD", "ETHUSD", "LINKUSD", "DOTUSD",
            # Synthetics
            "V100", "V75", "CRASH500", "BOOM500", "JUMP75"
        ],
        "min_confidence": 0.75,
        "max_positions": 2,
        "risk_per_trade": 0.003
    },
    "HIGH": {  # 2-3% daily or 100-500 pips
        "symbols": [
            # Forex
            "GBPNZD", "GBPJPY", "EURJPY", "EURPLN", "EURCZK", "USDPEN",
            "USDIDR", "NOKSEK",
            # Commodities
            "SUGAR", "COTTON", "CORN", "SOYBEANS", "FEEDERCATTLE", "LIVECATTLE",
            # Crypto
            "XRPUSD", "LTCUSD", "ADAUSD", "MATICUSD",
            # Synthetics
            "V50", "V25", "JUMP50", "JUMP25", "RB100", "RB200"
        ],
        "min_confidence": 0.70,
        "max_positions": 3,
        "risk_per_trade": 0.005
    }
}

# REGIONAL TRADING SESSIONS (Extended)
GLOBAL_SESSIONS = {
    "ASIAN": {
        "hours": "23:00-08:00 GMT",
        "best_symbols": [
            "USDJPY", "AUDJPY", "NZDJPY", "USDTHB", "USDPHP", "USDIDR",
            "USDSGD", "XRPUSD", "V75", "V100"
        ],
        "avoid": ["EURGBP", "USDCAD", "USDMXN"]
    },
    "EUROPEAN": {
        "hours": "07:00-16:00 GMT",
        "best_symbols": [
            "EURUSD", "GBPUSD", "EURGBP", "EURJPY", "EURNOK", "EURPLN",
            "EURHUF", "EURCZK", "UKOIL", "XAUUSD"
        ],
        "avoid": ["AUDUSD", "NZDUSD", "USDTHB"]
    },
    "AMERICAN": {
        "hours": "13:00-22:00 GMT",
        "best_symbols": [
            "EURUSD", "GBPUSD", "USDCAD", "USDMXN", "USDBRL", "USDCLP",
            "USDCOP", "US2000", "GASOLINE", "CORN", "WHEAT"
        ],
        "avoid": ["AUDJPY", "NZDJPY", "USDTHB"]
    },
    "SYNTHETIC_24_7": {
        "hours": "24/7",
        "best_symbols": [
            "V10", "V25", "V50", "V75", "V100", "V200", "V300",
            "CRASH1000", "BOOM1000", "JUMP100", "STEP500"
        ],
        "avoid": []
    }
}

# CORRELATION GROUPS (For diversification)
CORRELATION_GROUPS = {
    "COMMODITY_CURRENCIES": {
        "symbols": ["AUDUSD", "NZDUSD", "USDCAD", "USDNOK", "USDZAR"],
        "correlated_with": ["Gold", "Oil", "Copper"]
    },
    "RISK_CURRENCIES": {
        "symbols": ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"],
        "correlated_with": ["Stock indices", "Risk sentiment"]
    },
    "EMERGING_MARKETS": {
        "symbols": ["USDTRY", "USDZAR", "USDBRL", "USDMXN", "USDHUF"],
        "correlated_with": ["USD strength", "Risk-off events"]
    },
    "CRYPTO_CORRELATED": {
        "symbols": ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD"],
        "correlated_with": ["Tech stocks", "Risk sentiment"]
    }
}

# SEASONAL TRADING CALENDAR
SEASONAL_OPPORTUNITIES = {
    "JANUARY": {
        "strong": ["NATGAS", "HEATINGOIL", "COCOA", "VIX"],
        "weak": ["GASOLINE", "CORN"],
        "events": ["New Year positioning", "Winter demand"]
    },
    "MAY": {
        "strong": ["GASOLINE", "CORN", "WHEAT", "LIVECATTLE"],
        "weak": ["NATGAS", "HEATINGOIL"],
        "events": ["Summer driving season", "Planting season"]
    },
    "SEPTEMBER": {
        "strong": ["VIX", "COCOA", "COFFEE", "ORANGEJUICE"],
        "weak": ["GASOLINE"],
        "events": ["Market volatility", "Hurricane season"]
    },
    "DECEMBER": {
        "strong": ["LEANHOGS", "RETAIL stocks", "NATGAS"],
        "weak": ["Agricultural commodities"],
        "events": ["Holiday demand", "Year-end positioning"]
    }
}

# ULTRA-PROFIT TRADING STRATEGIES
ULTRA_STRATEGIES = {
    "volatility_breakout": {
        "best_symbols": ["LUMBER", "ORANGEJUICE", "V300", "CRASH1000"],
        "entry": "ATR > 2x average, volume spike",
        "exit": "Trailing stop 1.5x ATR",
        "timeframe": "M15, H1",
        "win_rate_target": 0.45,
        "risk_reward": 1.5
    },
    "commodity_correlation": {
        "best_symbols": ["USDNOK", "USDCOP", "USDCLP", "EURNOK"],
        "entry": "Commodity divergence > 2%",
        "exit": "Convergence or fixed target",
        "timeframe": "H1, H4",
        "win_rate_target": 0.65,
        "risk_reward": 1.2
    },
    "synthetic_scalping": {
        "best_symbols": ["V200", "V300", "JUMP100", "STEP500"],
        "entry": "Momentum burst, RSI extreme",
        "exit": "Quick 10-20 pip target",
        "timeframe": "M1, M5",
        "win_rate_target": 0.70,
        "risk_reward": 1.0
    },
    "crash_recovery": {
        "best_symbols": ["CRASH300", "CRASH500", "CRASH1000"],
        "entry": "After crash completion",
        "exit": "50% recovery of crash",
        "timeframe": "M5, M15",
        "win_rate_target": 0.75,
        "risk_reward": 2.0
    }
}

# RISK MANAGEMENT MATRIX
RISK_MATRIX = {
    "account_size": {
        "small": {  # <$5000
            "max_risk_per_trade": 0.005,
            "max_daily_risk": 0.02,
            "max_positions": 2,
            "avoid_tiers": ["EXTREME"]
        },
        "medium": {  # $5000-$25000
            "max_risk_per_trade": 0.01,
            "max_daily_risk": 0.03,
            "max_positions": 3,
            "avoid_tiers": []
        },
        "large": {  # >$25000
            "max_risk_per_trade": 0.015,
            "max_daily_risk": 0.05,
            "max_positions": 5,
            "avoid_tiers": []
        }
    }
}

# INTEGRATION FUNCTIONS
def get_current_best_symbols(session=None, volatility_min=None, strategy=None):
    """Get best symbols based on current conditions"""
    current_hour = datetime.utcnow().hour
    best_symbols = []
    
    # Determine session if not provided
    if not session:
        if 23 <= current_hour or current_hour < 8:
            session = "ASIAN"
        elif 7 <= current_hour < 16:
            session = "EUROPEAN"
        elif 13 <= current_hour < 22:
            session = "AMERICAN"
        else:
            session = "SYNTHETIC_24_7"
    
    # Get session-appropriate symbols
    session_symbols = GLOBAL_SESSIONS.get(session, {}).get("best_symbols", [])
    
    # Filter by volatility if specified
    if volatility_min:
        for symbol, config in ULTRA_PROFIT_SYMBOLS.items():
            vol = config.get("daily_volatility_percent", 0) or \
                  config.get("avg_daily_percent", 0) or \
                  config.get("volatility_percent", 0) or 0
            if vol >= volatility_min and symbol in session_symbols:
                best_symbols.append(symbol)
    else:
        best_symbols = session_symbols
    
    # Filter by strategy if specified
    if strategy and strategy in ULTRA_STRATEGIES:
        strategy_symbols = ULTRA_STRATEGIES[strategy].get("best_symbols", [])
        best_symbols = [s for s in best_symbols if s in strategy_symbols]
    
    return best_symbols

def calculate_position_size(symbol, account_balance, risk_percentage=0.01):
    """Calculate optimal position size for a symbol"""
    config = ULTRA_PROFIT_SYMBOLS.get(symbol, {})
    
    # Get symbol-specific risk override
    symbol_risk = config.get("risk_per_trade", risk_percentage)
    
    # Adjust for volatility tier
    for tier, tier_config in VOLATILITY_TIERS.items():
        if symbol in tier_config["symbols"]:
            symbol_risk = min(symbol_risk, tier_config["risk_per_trade"])
            break
    
    # Calculate position size
    risk_amount = account_balance * symbol_risk
    lot_size = config.get("lot_size", 0.01)
    
    return {
        "risk_amount": risk_amount,
        "lot_size": lot_size,
        "risk_percentage": symbol_risk
    }

def get_symbol_trading_conditions(symbol):
    """Get current trading conditions for a symbol"""
    config = ULTRA_PROFIT_SYMBOLS.get(symbol)
    if not config:
        return None
    
    current_hour = datetime.utcnow().hour
    current_time = f"{current_hour:02d}:00"
    
    # Check if within best hours
    best_hours = config.get("best_hours", "24/7")
    if best_hours == "24/7":
        within_best_hours = True
    else:
        # Parse best hours (format: "HH:MM-HH:MM")
        try:
            start, end = best_hours.split("-")
            start_hour = int(start.split(":")[0])
            end_hour = int(end.split(":")[0])
            
            if start_hour <= end_hour:
                within_best_hours = start_hour <= current_hour <= end_hour
            else:  # Crosses midnight
                within_best_hours = current_hour >= start_hour or current_hour <= end_hour
        except:
            within_best_hours = True
    
    return {
        "symbol": symbol,
        "tradeable": within_best_hours,
        "current_time": current_time,
        "best_hours": best_hours,
        "typical_spread": config.get("typical_spread", "Unknown"),
        "max_spread": config.get("max_spread", "Unknown"),
        "volatility_factor": config.get("volatility_factor", 1.0),
        "min_confidence": config.get("min_confidence", 0.70)
    }

# MASTER CONTROL FUNCTION
def get_ultra_profit_recommendations(account_balance=10000, risk_tolerance="medium"):
    """Get personalized trading recommendations"""
    current_session = get_current_best_symbols()
    risk_config = RISK_MATRIX["account_size"][risk_tolerance]
    
    recommendations = {
        "current_session": current_session,
        "top_opportunities": [],
        "risk_parameters": risk_config,
        "warnings": []
    }
    
    # Get top 5 opportunities
    for symbol in current_session[:5]:
        conditions = get_symbol_trading_conditions(symbol)
        if conditions and conditions["tradeable"]:
            position = calculate_position_size(symbol, account_balance, 
                                             risk_config["max_risk_per_trade"])
            recommendations["top_opportunities"].append({
                "symbol": symbol,
                "position_size": position,
                "conditions": conditions
            })
    
    # Add warnings for extreme volatility
    extreme_symbols = VOLATILITY_TIERS["EXTREME"]["symbols"]
    if any(s in current_session for s in extreme_symbols):
        recommendations["warnings"].append(
            "Extreme volatility symbols detected. Use tight stops!"
        )
    
    return recommendations

# Example usage
if __name__ == "__main__":
    print("=== ULTRA PROFIT MASTER CONFIGURATION LOADED ===")
    print(f"Total symbols available: {len(ULTRA_PROFIT_SYMBOLS)}")
    
    # Get current recommendations
    recs = get_ultra_profit_recommendations(account_balance=10000)
    print(f"\nCurrent session best symbols: {recs['current_session'][:5]}")
    print(f"Risk parameters: {recs['risk_parameters']}")
    
    # Show volatility distribution
    print("\n=== VOLATILITY DISTRIBUTION ===")
    for tier, config in VOLATILITY_TIERS.items():
        print(f"{tier}: {len(config['symbols'])} symbols")