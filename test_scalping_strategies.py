#!/usr/bin/env python3
"""
Test script to verify scalping strategies work without errors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ultra_scalping_strategies import UltraScalpingEnsemble, SignalType

def generate_test_data(num_bars=50, trend="neutral"):
    """Generate synthetic market data for testing"""
    times = [datetime.now() - timedelta(minutes=i) for i in range(num_bars, 0, -1)]
    
    # Base price
    base_price = 1.0800
    
    # Generate OHLCV data
    data = []
    for i, time in enumerate(times):
        # Add trend
        if trend == "up":
            trend_component = i * 0.00001
        elif trend == "down":
            trend_component = -i * 0.00001
        else:
            trend_component = 0
        
        # Add noise
        noise = np.random.normal(0, 0.00005)
        
        open_price = base_price + trend_component + noise
        high_price = open_price + abs(np.random.normal(0, 0.00003))
        low_price = open_price - abs(np.random.normal(0, 0.00003))
        close_price = np.random.uniform(low_price, high_price)
        volume = np.random.randint(100, 1000)
        
        data.append({
            'time': time.isoformat(),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': volume,
            'volume': volume * 100,
            'spread': 1
        })
    
    return data

def test_edge_cases():
    """Test edge cases that might cause errors"""
    print("Testing edge cases...")
    
    ensemble = UltraScalpingEnsemble()
    
    # Test 1: Flat market (no price movement)
    print("\n1. Testing flat market...")
    flat_data = []
    for i in range(30):
        flat_data.append({
            'time': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'open': 1.0800,
            'high': 1.0801,
            'low': 1.0799,
            'close': 1.0800,
            'tick_volume': 100,
            'volume': 10000,
            'spread': 1
        })
    
    try:
        signal = ensemble.get_ensemble_signal(flat_data)
        print(f"   Result: {signal.signal.value if signal else 'No signal'} ✓")
    except Exception as e:
        print(f"   Error: {e} ✗")
    
    # Test 2: Zero volume
    print("\n2. Testing zero volume...")
    zero_vol_data = generate_test_data()
    for bar in zero_vol_data:
        bar['tick_volume'] = 0
    
    try:
        signal = ensemble.get_ensemble_signal(zero_vol_data)
        print(f"   Result: {signal.signal.value if signal else 'No signal'} ✓")
    except Exception as e:
        print(f"   Error: {e} ✗")
    
    # Test 3: Single price (high = low = open = close)
    print("\n3. Testing single price bars...")
    single_price_data = []
    for i in range(30):
        price = 1.0800 + i * 0.00001
        single_price_data.append({
            'time': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'tick_volume': 100,
            'volume': 10000,
            'spread': 1
        })
    
    try:
        signal = ensemble.get_ensemble_signal(single_price_data)
        print(f"   Result: {signal.signal.value if signal else 'No signal'} ✓")
    except Exception as e:
        print(f"   Error: {e} ✗")
    
    # Test 4: Extreme volatility
    print("\n4. Testing extreme volatility...")
    volatile_data = generate_test_data()
    for i, bar in enumerate(volatile_data):
        bar['high'] = bar['open'] + 0.01  # 100 pips range
        bar['low'] = bar['open'] - 0.01
        bar['close'] = np.random.uniform(bar['low'], bar['high'])
    
    try:
        signal = ensemble.get_ensemble_signal(volatile_data)
        print(f"   Result: {signal.signal.value if signal else 'No signal'} ✓")
    except Exception as e:
        print(f"   Error: {e} ✗")

def test_normal_scenarios():
    """Test normal trading scenarios"""
    print("\nTesting normal scenarios...")
    
    ensemble = UltraScalpingEnsemble()
    
    # Test uptrend
    print("\n1. Testing uptrend...")
    up_data = generate_test_data(trend="up")
    try:
        signal = ensemble.get_ensemble_signal(up_data)
        if signal:
            print(f"   Signal: {signal.signal.value}")
            print(f"   Confidence: {signal.confidence:.2%}")
            print(f"   Reason: {signal.reason}")
        else:
            print("   No signal generated")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test downtrend
    print("\n2. Testing downtrend...")
    down_data = generate_test_data(trend="down")
    try:
        signal = ensemble.get_ensemble_signal(down_data)
        if signal:
            print(f"   Signal: {signal.signal.value}")
            print(f"   Confidence: {signal.confidence:.2%}")
            print(f"   Reason: {signal.reason}")
        else:
            print("   No signal generated")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    print("="*60)
    print("SCALPING STRATEGIES TEST")
    print("="*60)
    
    test_edge_cases()
    test_normal_scenarios()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()