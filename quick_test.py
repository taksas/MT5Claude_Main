#!/usr/bin/env python3
"""
Quick test of trading strategies
"""

from improved_strategies import ImprovedStrategyEnsemble
from datetime import datetime
import numpy as np
import random

def generate_quick_test_data(bars=50):
    """Generate quick test data"""
    data = []
    base_price = 1.0850
    
    for i in range(bars):
        price_change = np.random.normal(0, 0.0004)
        base_price += price_change
        
        candle = {
            'time': datetime.utcnow().isoformat(),
            'open': round(base_price, 5),
            'high': round(base_price + abs(np.random.normal(0, 0.0002)), 5),
            'low': round(base_price - abs(np.random.normal(0, 0.0002)), 5),
            'close': round(base_price, 5),
            'tick_volume': random.randint(300, 800)
        }
        
        # Ensure OHLC consistency
        candle['high'] = max(candle['open'], candle['high'], candle['low'], candle['close'])
        candle['low'] = min(candle['open'], candle['high'], candle['low'], candle['close'])
        
        data.append(candle)
    
    return data

def main():
    print("🚀 Quick Strategy Test")
    print("="*40)
    
    # Test enhanced strategies
    ensemble = ImprovedStrategyEnsemble()
    
    # Generate test data
    test_data = generate_quick_test_data(100)
    print(f"Generated {len(test_data)} test candles")
    
    # Test strategy analysis
    try:
        signal = ensemble.get_ensemble_signal(test_data)
        
        if signal:
            print(f"✅ Signal Generated!")
            print(f"   Type: {signal.signal.value}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Entry: {signal.entry_price:.5f}")
            print(f"   Stop Loss: {signal.stop_loss:.5f}")
            print(f"   Take Profit: {signal.take_profit:.5f}")
            print(f"   Reason: {signal.reason}")
        else:
            print("❌ No signal generated")
        
        print("\n✅ Strategy ensemble working correctly!")
        
    except Exception as e:
        print(f"❌ Error in strategy: {e}")
    
    print("="*40)

if __name__ == "__main__":
    main()