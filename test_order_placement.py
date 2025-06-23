#!/usr/bin/env python3
"""Test order placement directly"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

def test_order():
    print("Testing order placement...\n")
    
    # First, get a valid symbol
    response = requests.get(f"{API_BASE}/market/symbols/tradable")
    symbols = response.json()
    test_symbol = symbols[0] if symbols else "EURUSD"
    
    print(f"Using symbol: {test_symbol}")
    
    # Get symbol info for proper pricing
    from urllib.parse import quote
    encoded_symbol = quote(test_symbol)
    response = requests.get(f"{API_BASE}/market/symbols/{encoded_symbol}")
    if response.status_code == 200:
        symbol_info = response.json()
        current_price = symbol_info.get('ask', 1.0)
        min_volume = symbol_info.get('volume_min', 0.01)
        digits = symbol_info.get('digits', 5)
        
        print(f"Current price: {current_price}")
        print(f"Min volume: {min_volume}")
        print(f"Digits: {digits}\n")
    else:
        print("Failed to get symbol info")
        current_price = 1.0
        min_volume = 0.01
        digits = 5
    
    # Create a test order
    order = {
        "action": 1,  # TRADE_ACTION_DEAL
        "symbol": test_symbol,
        "volume": min_volume,
        "type": 0,  # ORDER_TYPE_BUY
        "price": current_price,
        "sl": round(current_price * 0.995, digits),  # 0.5% stop loss
        "tp": round(current_price * 1.01, digits),   # 1% take profit
        "comment": "Test order",
        "deviation": 20,
        "magic": 123456
    }
    
    print("Order details:")
    print(json.dumps(order, indent=2))
    
    # Try to place the order
    print("\nPlacing order...")
    try:
        response = requests.post(
            f"{API_BASE}/trading/orders",
            json=order,
            timeout=10
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"\n✓ Order placed successfully!")
            print(f"Order ticket: {data.get('order')}")
            print(f"Deal ticket: {data.get('deal')}")
        else:
            print(f"\n✗ Order failed")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    test_order()