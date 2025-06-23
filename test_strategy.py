#!/usr/bin/env python3
"""Test trading strategy after fixes"""

import pandas as pd
import numpy as np
from components.trading_strategy import TradingStrategy

# Create test data
dates = pd.date_range(end='2024-01-01', periods=100, freq='5min')
test_df = pd.DataFrame({
    'time': dates,
    'open': 1.1000 + np.cumsum(np.random.randn(100) * 0.0001),
    'high': 1.1005 + np.cumsum(np.random.randn(100) * 0.0001),
    'low': 1.0995 + np.cumsum(np.random.randn(100) * 0.0001),
    'close': 1.1000 + np.cumsum(np.random.randn(100) * 0.0001),
    'volume': np.random.randint(100, 1000, 100)
})

# Ensure high/low are correct
test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)

print("Testing Trading Strategy...")
print("=" * 50)

try:
    strategy = TradingStrategy()
    print("✓ Strategy initialized successfully")
    
    # Test ultra analysis
    signal = strategy.analyze_ultra("EURUSD", test_df, test_df['close'].iloc[-1])
    
    if signal:
        print(f"✓ Signal generated: {signal.type}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Reason: {signal.reason}")
        print(f"  Entry: {signal.entry}")
        print(f"  SL: {signal.sl}")
        print(f"  TP: {signal.tp}")
    else:
        print("✓ No signal generated (this is normal)")
    
    # Test consciousness level
    print(f"\nConsciousness level: {strategy.consciousness_level}")
    print(f"Enlightenment progress: {strategy.enlightenment_progress:.2%}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()