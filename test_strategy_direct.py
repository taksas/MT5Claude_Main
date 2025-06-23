#!/usr/bin/env python3
"""Direct test of strategy with minimal setup"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

sys.path.append('.')

from components.trading_strategy import TradingStrategy
from components.trading_models import SignalType

# Enable all logging
logging.basicConfig(level=logging.DEBUG)

def create_test_data():
    """Create synthetic test data"""
    # Create 100 candles of data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    
    # Create trending data
    base_price = 1.1000
    trend = np.linspace(0, 0.01, 100)  # Uptrend
    noise = np.random.normal(0, 0.0005, 100)
    
    close_prices = base_price + trend + noise
    
    # Create OHLC data
    df = pd.DataFrame({
        'open': close_prices - 0.0002,
        'high': close_prices + 0.0003,
        'low': close_prices - 0.0003,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    return df

def test_strategy():
    print("=== Testing Strategy Directly ===\n")
    
    # Create strategy
    strategy = TradingStrategy()
    
    # Create test data
    df = create_test_data()
    current_price = df['close'].iloc[-1]
    
    print(f"Test data created:")
    print(f"  Candles: {len(df)}")
    print(f"  Current price: {current_price:.5f}")
    print(f"  Price change: {((current_price / df['close'].iloc[0]) - 1) * 100:.2f}%\n")
    
    # Test 1: Force trade signal (simplest)
    print("1. Testing force_trade_signal...")
    try:
        signal = strategy.force_trade_signal("EURUSD", df, current_price)
        if signal:
            print(f"✓ Force signal generated:")
            print(f"  Type: {signal.type}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Entry: {signal.entry}")
            print(f"  SL: {signal.sl}")
            print(f"  TP: {signal.tp}")
        else:
            print("✗ No force signal generated")
    except Exception as e:
        print(f"✗ Error in force signal: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Ultra analyze (complex)
    print("\n2. Testing analyze_ultra...")
    try:
        # First, let's check if indicators can be calculated
        print("  Calculating indicators...")
        from components.indicators import QuantumUltraIntelligentIndicators
        indicators = QuantumUltraIntelligentIndicators()
        
        analysis = indicators.calculate_ultra_indicators(df, current_price)
        print(f"  ✓ Indicators calculated")
        print(f"    Regime: {analysis.get('regime', 'Unknown')}")
        print(f"    Composite signal: {analysis.get('composite_signal', 0):.3f}")
        
        # Now try full analysis
        signal = strategy.analyze_ultra("EURUSD", df, current_price)
        if signal:
            print(f"\n  ✓ Ultra signal generated:")
            print(f"    Type: {signal.type}")
            print(f"    Confidence: {signal.confidence:.2%}")
            print(f"    Reason: {signal.reason}")
        else:
            print("\n  ✗ No ultra signal generated")
            
    except Exception as e:
        print(f"\n  ✗ Error in ultra analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check minimum requirements
    print("\n3. Checking signal generation requirements...")
    from components.trading_config import CONFIG
    print(f"  MIN_CONFIDENCE: {CONFIG['MIN_CONFIDENCE']}")
    print(f"  MAX_SPREAD: {CONFIG['MAX_SPREAD']}")
    print(f"  AGGRESSIVE_MODE: {CONFIG['AGGRESSIVE_MODE']}")

if __name__ == "__main__":
    test_strategy()