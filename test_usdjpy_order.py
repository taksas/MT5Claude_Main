#!/usr/bin/env python3
"""Test order placement with USDJPY# symbol"""

import requests
import json
from urllib.parse import quote

API_BASE = "http://172.28.144.1:8000"

def test_usdjpy_order():
    print("=== Testing USDJPY# Order Placement ===\n")
    
    # Check if USDJPY# is available
    test_symbol = "USDJPY#"
    
    # Get symbol info
    print(f"1. Getting symbol info for {test_symbol}...")
    encoded_symbol = quote(test_symbol)
    
    try:
        response = requests.get(f"{API_BASE}/market/symbols/{encoded_symbol}")
        if response.status_code == 200:
            symbol_info = response.json()
            print(f"✓ Symbol found: {symbol_info.get('name')}")
            print(f"  Bid: {symbol_info.get('bid')}")
            print(f"  Ask: {symbol_info.get('ask')}")
            print(f"  Spread: {symbol_info.get('spread')}")
            print(f"  Min volume: {symbol_info.get('volume_min')}")
            print(f"  Digits: {symbol_info.get('digits')}\n")
            
            current_price = symbol_info.get('ask', 150.0)
            min_volume = symbol_info.get('volume_min', 0.01)
            digits = symbol_info.get('digits', 3)
        else:
            print(f"✗ Symbol not found: {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"✗ Error getting symbol info: {e}")
        return
    
    # Create test order
    print("2. Creating test order...")
    
    # For USDJPY, typical stop loss is 50-100 pips (0.50-1.00 price units)
    sl_distance = 0.50  # 50 pips
    tp_distance = 1.00  # 100 pips
    
    order = {
        "action": 1,  # TRADE_ACTION_DEAL
        "symbol": test_symbol,
        "volume": min_volume,
        "type": 0,  # ORDER_TYPE_BUY
        "sl": round(current_price - sl_distance, digits),
        "tp": round(current_price + tp_distance, digits),
        "comment": "Test USDJPY order",
        "deviation": 20,
        "magic": 100100
    }
    
    print("Order details:")
    print(json.dumps(order, indent=2))
    
    # Place order
    print("\n3. Placing order...")
    try:
        response = requests.post(
            f"{API_BASE}/trading/orders",
            json=order,
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"\n✓ Order placed successfully!")
            print(f"  Order ticket: {data.get('order')}")
            print(f"  Deal ticket: {data.get('deal')}")
            print(f"  Executed price: {data.get('price')}")
        else:
            print(f"\n✗ Order failed with status {response.status_code}")
            
    except Exception as e:
        print(f"\n✗ Error placing order: {e}")

if __name__ == "__main__":
    test_usdjpy_order()