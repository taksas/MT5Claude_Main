#!/usr/bin/env python3
"""
Test stop loss calculation algorithm
"""

from live_trading_engine import LiveTradingEngine
import numpy as np
from datetime import datetime, timedelta

def generate_test_market_data(base_price=1.0800, volatility=0.0005, trend=0, bars=100):
    """Generate synthetic market data for testing"""
    data = []
    for i in range(bars):
        noise = np.random.normal(0, volatility)
        trend_component = trend * i * 0.00001
        
        open_price = base_price + trend_component + noise
        high = open_price + abs(np.random.normal(0, volatility * 0.5))
        low = open_price - abs(np.random.normal(0, volatility * 0.5))
        close = np.random.uniform(low, high)
        
        data.append({
            'time': (datetime.now() - timedelta(minutes=bars-i)).isoformat(),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': np.random.randint(100, 1000),
            'volume': np.random.randint(10000, 100000),
            'spread': 1
        })
    
    return data

def test_stop_loss_calculation():
    """Test stop loss calculation with various market conditions"""
    print("="*60)
    print("STOP LOSS CALCULATION TEST")
    print("="*60)
    
    engine = LiveTradingEngine()
    
    # Test scenarios
    scenarios = [
        ("Normal Market", 1.0800, 0.0003, 0),
        ("High Volatility", 1.0800, 0.0010, 0),
        ("Low Volatility", 1.0800, 0.0001, 0),
        ("Uptrend", 1.0800, 0.0003, 1),
        ("Downtrend", 1.0800, 0.0003, -1),
        ("USDJPY", 145.50, 0.05, 0),  # JPY pair
    ]
    
    for scenario_name, base_price, volatility, trend in scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 40)
        
        # Generate test data
        market_data = generate_test_market_data(base_price, volatility, trend)
        
        # Determine symbol
        symbol = "USDJPY#" if base_price > 100 else "EURUSD#"
        
        # Test BUY order
        sl_buy, tp_buy, method_buy = engine.calculate_safe_stop_loss(
            symbol, "BUY", base_price, market_data
        )
        
        # Test SELL order
        sl_sell, tp_sell, method_sell = engine.calculate_safe_stop_loss(
            symbol, "SELL", base_price, market_data
        )
        
        # Calculate pip distances
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        
        buy_sl_pips = (base_price - sl_buy) / pip_value
        buy_tp_pips = (tp_buy - base_price) / pip_value
        buy_rr_ratio = buy_tp_pips / buy_sl_pips if buy_sl_pips > 0 else 0
        
        sell_sl_pips = (sl_sell - base_price) / pip_value
        sell_tp_pips = (base_price - tp_sell) / pip_value
        sell_rr_ratio = sell_tp_pips / sell_sl_pips if sell_sl_pips > 0 else 0
        
        print(f"Symbol: {symbol}")
        print(f"Entry Price: {base_price}")
        print(f"\nBUY Order:")
        print(f"  Stop Loss: {sl_buy:.5f} ({buy_sl_pips:.1f} pips) - {method_buy}")
        print(f"  Take Profit: {tp_buy:.5f} ({buy_tp_pips:.1f} pips)")
        print(f"  Risk/Reward: 1:{buy_rr_ratio:.1f}")
        print(f"\nSELL Order:")
        print(f"  Stop Loss: {sl_sell:.5f} ({sell_sl_pips:.1f} pips) - {method_sell}")
        print(f"  Take Profit: {tp_sell:.5f} ({sell_tp_pips:.1f} pips)")
        print(f"  Risk/Reward: 1:{sell_rr_ratio:.1f}")
        
        # Verify stop loss is within limits (with small tolerance for floating point)
        tolerance = 0.001
        assert engine.min_sl_pips - tolerance <= buy_sl_pips <= engine.max_sl_pips + tolerance, \
            f"BUY SL out of range: {buy_sl_pips}"
        assert engine.min_sl_pips - tolerance <= sell_sl_pips <= engine.max_sl_pips + tolerance, \
            f"SELL SL out of range: {sell_sl_pips}"
        
        print("✅ Stop loss within safety limits")
    
    print("\n" + "="*60)
    print("All tests passed! Stop loss calculation is working correctly.")
    print("="*60)

if __name__ == "__main__":
    test_stop_loss_calculation()