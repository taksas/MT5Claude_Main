#!/usr/bin/env python3
"""
Force signal generation test
"""

import pandas as pd
import numpy as np
from components.trading_strategy import TradingStrategy
from components.trading_models import SignalType
from components.trading_config import CONFIG

print("Testing forced signal generation...")
print(f"MIN_CONFIDENCE: {CONFIG['MIN_CONFIDENCE']}")
print(f"AGGRESSIVE_MODE: {CONFIG['AGGRESSIVE_MODE']}")
print(f"IGNORE_SPREAD: {CONFIG['IGNORE_SPREAD']}")

# Create dummy market data
dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
dummy_data = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 101,
    'low': np.random.randn(100).cumsum() + 99,
    'close': np.random.randn(100).cumsum() + 100,
    'tick_volume': np.random.randint(10, 100, 100),
    'spread': np.random.randint(1, 10, 100),
    'real_volume': 0
}, index=dates)

# Make sure we have proper OHLC relationships
for i in range(len(dummy_data)):
    dummy_data.loc[dummy_data.index[i], 'high'] = max(
        dummy_data.loc[dummy_data.index[i], 'open'],
        dummy_data.loc[dummy_data.index[i], 'high'],
        dummy_data.loc[dummy_data.index[i], 'close']
    )
    dummy_data.loc[dummy_data.index[i], 'low'] = min(
        dummy_data.loc[dummy_data.index[i], 'open'],
        dummy_data.loc[dummy_data.index[i], 'low'],
        dummy_data.loc[dummy_data.index[i], 'close']
    )

print(f"\nCreated dummy data with {len(dummy_data)} candles")

# Initialize strategy
strategy = TradingStrategy()

# Test analyze_ultra
print("\nTesting analyze_ultra...")
current_price = dummy_data['close'].iloc[-1]
signal = strategy.analyze_ultra("EURUSD", dummy_data, current_price)

if signal:
    print(f"\n✅ SIGNAL GENERATED!")
    print(f"Type: {signal.type.value}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Entry: {signal.entry}")
    print(f"SL: {signal.sl}")
    print(f"TP: {signal.tp}")
else:
    print("\n❌ No signal generated")
    
# Test force_trade_signal
print("\n\nTesting force_trade_signal...")
forced_signal = strategy.force_trade_signal("EURUSD", dummy_data, current_price)

if forced_signal:
    print(f"\n✅ FORCED SIGNAL GENERATED!")
    print(f"Type: {forced_signal.type.value}")
    print(f"Confidence: {forced_signal.confidence:.1%}")
    print(f"Entry: {forced_signal.entry}")
    print(f"SL: {forced_signal.sl}")
    print(f"TP: {forced_signal.tp}")
else:
    print("\n❌ No forced signal generated")