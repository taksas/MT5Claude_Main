#!/usr/bin/env python3
"""
Test script for Ultra Rare Profit Configuration
"""

try:
    from ultra_rare_profit_config import (
        ULTRA_PROFIT_SYMBOLS,
        ULTRA_PROFIT_SETTINGS,
        SYMBOL_STRATEGIES,
        EXOTIC_FOREX_ULTRA,
        SYNTHETIC_INDICES,
        RARE_COMMODITIES
    )
    
    print("🔥🔥 ULTRA RARE PROFIT CONFIGURATION LOADED SUCCESSFULLY! 🔥🔥")
    print("=" * 60)
    
    print(f"\n💎 EXOTIC FOREX PAIRS ({len(EXOTIC_FOREX_ULTRA)} symbols):")
    for symbol, config in EXOTIC_FOREX_ULTRA.items():
        print(f"  - {symbol}: {config.get('volatility_factor', 0)}x volatility, "
              f"{config.get('stop_loss_pips', 0)}-{config.get('profit_target_pips', 0)} pips TP/SL")
    
    print(f"\n🎯 SYNTHETIC INDICES ({len(SYNTHETIC_INDICES)} symbols):")
    for symbol, config in SYNTHETIC_INDICES.items():
        print(f"  - {symbol}: {config.get('symbol', '')} - {config.get('strategy', '')}")
    
    print(f"\n🌟 RARE COMMODITIES ({len(RARE_COMMODITIES)} symbols):")
    for symbol, config in RARE_COMMODITIES.items():
        print(f"  - {symbol}: {config.get('name', '')} - "
              f"{config.get('avg_monthly_volatility', 0)}% monthly volatility")
    
    print(f"\n⚡ TOTAL ULTRA SYMBOLS: {len(ULTRA_PROFIT_SYMBOLS)}")
    
    print(f"\n⚙️  ULTRA PROFIT SETTINGS:")
    for key, value in ULTRA_PROFIT_SETTINGS.items():
        print(f"  - {key}: {value}")
    
    print(f"\n✅ Configuration is ready to use!")
    print(f"💡 To activate: Run ultra_trading_engine.py")
    
except ImportError as e:
    print(f"❌ Error loading ultra_rare_profit_config: {e}")
    print("Make sure ultra_rare_profit_config.py exists in the same directory")