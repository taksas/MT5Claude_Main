#!/usr/bin/env python3
"""
Test engine with proper initialization
"""

import time
import threading
from components.engine_core import UltraTradingEngine
from components.trading_config import CONFIG

print("Testing engine with proper initialization...")
print(f"Settings: MIN_CONFIDENCE={CONFIG['MIN_CONFIDENCE']}, AGGRESSIVE={CONFIG['AGGRESSIVE_MODE']}")

# Create and initialize engine
engine = UltraTradingEngine()

# Initialize properly (without starting the full loop)
print("\nInitializing engine...")
if not engine.api_client.check_connection():
    print("✗ Cannot connect to API")
    exit(1)

engine.balance = engine.api_client.get_balance()
if not engine.balance:
    print("✗ Cannot get account balance")
    exit(1)

print(f"✓ Balance: ¥{engine.balance:,.0f}")

# Discover symbols
engine.tradable_symbols = engine._discover_symbols()
if not engine.tradable_symbols:
    print("✗ No tradable symbols found")
    exit(1)

print(f"✓ Found {len(engine.tradable_symbols)} symbols: {', '.join(engine.tradable_symbols[:5])}...")

# Set up signal capture
signals_captured = []

def capture_signal(symbol, signal):
    print(f"\n🎯 SIGNAL FOUND for {symbol}!")
    print(f"  Type: {signal.type.value}")
    print(f"  Confidence: {signal.confidence:.1%}")
    print(f"  Entry: {signal.entry}, SL: {signal.sl}, TP: {signal.tp}")
    signals_captured.append((symbol, signal))

original_execute = engine._execute_signal
engine._execute_signal = capture_signal

# Test specific symbols
print("\nAnalyzing top 10 symbols...")
for i, symbol in enumerate(engine.tradable_symbols[:10]):
    print(f"\n[{i+1}/10] Analyzing {symbol}...", end=" ")
    try:
        signal = engine._analyze_symbol(symbol)
        if signal:
            print("SIGNAL FOUND!")
            capture_signal(symbol, signal)
        else:
            print("no signal")
    except Exception as e:
        print(f"error: {e}")

# If no signals, try force trade
if not signals_captured and CONFIG['AGGRESSIVE_MODE']:
    print("\n\nNo signals found. Attempting force trade...")
    try:
        engine._force_trade()
    except Exception as e:
        print(f"Force trade error: {e}")

print(f"\n\nTotal signals found: {len(signals_captured)}")

# If still no signals, check why
if not signals_captured:
    print("\nDebugging first symbol in detail...")
    symbol = engine.tradable_symbols[0]
    from components.market_data import MarketData
    from components.trading_strategy import TradingStrategy
    
    market_data = MarketData(engine.api_client)
    strategy = TradingStrategy()
    
    df = market_data.get_market_data(symbol)
    if df is not None:
        print(f"  Data points: {len(df)}")
        current_price = market_data.get_current_price(symbol)
        print(f"  Current price: {current_price}")
        
        if current_price:
            print("  Forcing analysis with debug...")
            signal = strategy.analyze_ultra(symbol, df, current_price)
            print(f"  Result: {'Signal' if signal else 'No signal'}")