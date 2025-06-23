#!/usr/bin/env python3
"""
Comprehensive signal analysis debug
"""

import pandas as pd
from components.mt5_api_client import MT5APIClient
from components.market_data import MarketData
from components.trading_strategy import TradingStrategy
from components.trading_config import CONFIG

print("=" * 80)
print("SIGNAL GENERATION DEBUG ANALYSIS")
print("=" * 80)

# Initialize components
api_client = MT5APIClient(CONFIG["API_BASE"])
market_data = MarketData(api_client)
strategy = TradingStrategy()

# Get symbols
symbols = api_client.discover_symbols()
print(f"\nFound {len(symbols)} symbols")

# Test first 5 symbols
test_symbols = symbols[:5]
print(f"\nTesting symbols: {test_symbols}")

for symbol in test_symbols:
    print(f"\n{'='*60}")
    print(f"ANALYZING: {symbol}")
    print(f"{'='*60}")
    
    # Get market data
    df = market_data.get_market_data(symbol, count=100)
    if df is None or len(df) < 50:
        print(f"❌ Insufficient data: {len(df) if df is not None else 0} candles")
        continue
    
    print(f"✓ Got {len(df)} candles")
    print(f"  Latest: {df.index[-1]}")
    print(f"  Close: {df['close'].iloc[-1]}")
    
    # Get current price
    current_price = market_data.get_current_price(symbol)
    if not current_price:
        print("❌ No current price available")
        continue
    
    print(f"✓ Current price: {current_price}")
    
    # Check spread
    spread_ok, spread = market_data.check_spread(symbol)
    print(f"✓ Spread: {spread:.1f} pips ({'OK' if spread_ok else 'TOO WIDE'})")
    
    # Analyze with strategy
    try:
        signal = strategy.analyze_ultra(symbol, df, current_price)
        
        if signal:
            print(f"\n🎯 SIGNAL GENERATED!")
            print(f"  Type: {signal.type.value}")
            print(f"  Confidence: {signal.confidence:.1%}")
            print(f"  Quality: {signal.quality:.1%}")
            print(f"  Entry: {signal.entry}")
            print(f"  SL: {signal.sl}")
            print(f"  TP: {signal.tp}")
            print(f"  Reasons: {', '.join(signal.reasons[:3])}")
        else:
            print(f"\n⚪ No signal generated")
            
            # Let's analyze why - check indicators directly
            print("\nDirect indicator analysis:")
            
            # Check RSI
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                print(f"  RSI: {rsi:.1f}")
            
            # Check MACD
            if 'macd' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
                print(f"  MACD: {macd:.5f}, Signal: {macd_signal:.5f}")
            
            # Force a detailed analysis
            print("\nForcing strategy analysis with debug mode...")
            # This will help us see what's happening inside
            
    except Exception as e:
        print(f"❌ Error analyzing: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("CONFIGURATION CHECK")
print("="*80)
print(f"MIN_CONFIDENCE: {CONFIG.get('MIN_CONFIDENCE', 0.7)}")
print(f"AGGRESSIVE_MODE: {CONFIG.get('AGGRESSIVE_MODE', False)}")
print(f"RISK_PER_TRADE: {CONFIG.get('RISK_PER_TRADE', 0.01)}")
print(f"MAX_SPREAD: {CONFIG.get('MAX_SPREAD', 2.5)}")

# Let's also check if the strategy thresholds are too high
print("\nChecking strategy internal thresholds...")
print("If confidence threshold is too high, no signals will be generated")