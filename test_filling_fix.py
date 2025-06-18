#!/usr/bin/env python3
"""Test the filling mode fix with proper order parameters"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def get_current_price(symbol):
    """Get current bid/ask prices"""
    response = requests.get(f"{API_BASE}/market/tick/{symbol}")
    if response.status_code == 200:
        tick = response.json()
        return tick.get('bid'), tick.get('ask')
    return None, None

def test_order_with_auto_filling(symbol="EURUSD#"):
    """Test order placement with auto-detected filling mode"""
    print(f"\n🧪 Testing order for {symbol} with auto-detected filling mode")
    
    # Get current prices
    bid, ask = get_current_price(symbol)
    if not bid or not ask:
        print(f"❌ Could not get prices for {symbol}")
        return
    
    print(f"Current prices - Bid: {bid}, Ask: {ask}")
    
    # Calculate proper stop loss and take profit
    pip_value = 0.0001 if "JPY" not in symbol else 0.01
    sl_distance = 10 * pip_value  # 10 pips
    tp_distance = 15 * pip_value  # 15 pips
    
    # BUY order parameters
    entry_price = ask
    stop_loss = round(entry_price - sl_distance, 5)
    take_profit = round(entry_price + tp_distance, 5)
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": stop_loss,
        "tp": take_profit,
        "comment": "Test auto filling"
        # Note: NOT specifying type_filling - let API auto-detect
    }
    
    print(f"\nOrder details:")
    print(f"  Entry: {entry_price}")
    print(f"  Stop Loss: {stop_loss}")
    print(f"  Take Profit: {take_profit}")
    print(f"\nOrder data: {json.dumps(order_data, indent=2)}")
    
    # Send order
    response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
    print(f"\nResponse status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 201:
        print("✅ Order placed successfully!")
        result = response.json()
        print(f"Order ticket: {result.get('order')}")
        
        # Wait a moment then close the position
        time.sleep(2)
        ticket = result.get('order')
        if ticket:
            print(f"\n🔄 Closing position {ticket}...")
            close_response = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
            print(f"Close response: {close_response.status_code}")
    else:
        print("❌ Order failed")

if __name__ == "__main__":
    print("="*60)
    print("FILLING MODE FIX TEST")
    print("="*60)
    
    # Test with EURUSD#
    test_order_with_auto_filling("EURUSD#")
    
    # Test with another symbol
    print("\n" + "="*60)
    test_order_with_auto_filling("USDJPY#")