#!/usr/bin/env python3
"""Test to understand MT5 filling mode constants"""

import requests

API_BASE = "http://172.28.144.1:8000"

# Test all possible filling mode values
print("Testing all filling mode values from 0 to 10:\n")

# Get current price first
response = requests.post(f"{API_BASE}/market/history", 
                       json={"symbol": "EURUSD#", "timeframe": "M1", "count": 1})
if response.status_code == 200:
    history = response.json()
    if history:
        current_price = history[0]['close']
        pip = 0.0001
        
        for mode in range(11):
            order_data = {
                "action": 1,  # DEAL
                "symbol": "EURUSD#",
                "volume": 0.01,
                "type": 0,  # BUY
                "sl": round(current_price - 20 * pip, 5),
                "tp": round(current_price + 30 * pip, 5),
                "type_filling": mode,
                "comment": f"Test mode {mode}"
            }
            
            response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
            print(f"Mode {mode}: {response.status_code} - {response.text[:100]}...")
            
            if response.status_code == 201:
                print(f"  ✅ SUCCESS! Mode {mode} works!")
                # Close the position
                result = response.json()
                ticket = result.get('order')
                if ticket:
                    import time
                    time.sleep(1)
                    close_resp = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
                break