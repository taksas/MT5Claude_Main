#!/usr/bin/env python3
"""Test to see what's happening with order check"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

# Get current price
response = requests.post(f"{API_BASE}/market/history", 
                       json={"symbol": "EURUSD#", "timeframe": "M1", "count": 1})
if response.status_code == 200:
    history = response.json()
    if history:
        current_price = history[0]['close']
        print(f"Current price: {current_price}")
        
        # Get current positions to see account status
        pos_response = requests.get(f"{API_BASE}/trading/positions")
        print(f"\nCurrent positions: {len(pos_response.json() if pos_response.status_code == 200 else [])}")
        
        # Get account info
        acc_response = requests.get(f"{API_BASE}/account/")
        if acc_response.status_code == 200:
            account = acc_response.json()
            print(f"Account balance: {account.get('balance')}")
            print(f"Account currency: {account.get('currency')}")
            print(f"Free margin: {account.get('margin_free')}")
        
        pip = 0.0001
        
        # Try a very conservative order
        order_data = {
            "action": 1,  # DEAL
            "symbol": "EURUSD#",
            "volume": 0.01,
            "type": 0,  # BUY
            "sl": round(current_price - 50 * pip, 5),  # 50 pips SL
            "tp": round(current_price + 50 * pip, 5),  # 50 pips TP
            "type_filling": 1,  # IOC
            "comment": "Test IOC order"
        }
        
        print(f"\nOrder details:")
        print(f"  Entry: Market")
        print(f"  SL: {order_data['sl']} ({50} pips)")
        print(f"  TP: {order_data['tp']} ({50} pips)")
        print(f"  Volume: {order_data['volume']}")
        print(f"  Filling: IOC (1)")
        
        print(f"\nSending order...")
        response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
        print(f"Response: {response.status_code}")
        print(f"Body: {response.text}")
        
        # If it's a validation error, try without SL/TP
        if "Done (retcode: 0)" in response.text:
            print("\n🔄 Trying without SL/TP...")
            order_data.pop('sl', None)
            order_data.pop('tp', None)
            response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
            print(f"Response: {response.status_code}")
            print(f"Body: {response.text}")