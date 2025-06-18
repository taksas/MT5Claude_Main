#!/usr/bin/env python3
"""Test position modification with correct PATCH method"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def test_position_modification():
    """Test modifying position SL/TP"""
    print("="*60)
    print("TESTING POSITION MODIFICATION")
    print("="*60)
    
    # Check for open positions
    response = requests.get(f"{API_BASE}/trading/positions")
    if response.status_code != 200:
        print("❌ Failed to get positions")
        return
        
    positions = response.json()
    if not positions:
        print("No open positions to modify")
        print("\nCreating a test position...")
        
        # Place a test order
        symbol = "EURUSD#"
        response = requests.post(f"{API_BASE}/market/history", 
                               json={"symbol": symbol, "timeframe": "M1", "count": 1})
        if response.status_code == 200:
            history = response.json()
            if history:
                price = history[0]['close']
                pip = 0.0001
                
                order_data = {
                    "action": 1,
                    "symbol": symbol,
                    "volume": 0.01,
                    "type": 0,  # BUY
                    "sl": round(price - 50 * pip, 5),
                    "tp": round(price + 50 * pip, 5),
                    "comment": "Test for modify"
                }
                
                response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
                if response.status_code == 201:
                    print("✅ Test position created")
                    time.sleep(1)
                    # Get positions again
                    response = requests.get(f"{API_BASE}/trading/positions")
                    positions = response.json() if response.status_code == 200 else []
                else:
                    print(f"❌ Failed to create test position: {response.text}")
                    return
    
    if positions:
        position = positions[0]
        ticket = position['ticket']
        current_sl = position.get('sl', 0)
        current_tp = position.get('tp', 0)
        price_open = position.get('price_open', 0)
        symbol = position.get('symbol', '')
        
        print(f"\n📊 Position found:")
        print(f"   Ticket: {ticket}")
        print(f"   Symbol: {symbol}")
        print(f"   Current SL: {current_sl}")
        print(f"   Current TP: {current_tp}")
        
        # Calculate new SL/TP
        pip = 0.0001 if "JPY" not in symbol else 0.01
        new_sl = round(price_open - 30 * pip, 5)
        new_tp = round(price_open + 40 * pip, 5)
        
        print(f"\n🔧 Modifying position:")
        print(f"   New SL: {new_sl}")
        print(f"   New TP: {new_tp}")
        
        # Test with PATCH method
        modify_data = {"sl": new_sl, "tp": new_tp}
        response = requests.patch(f"{API_BASE}/trading/positions/{ticket}", json=modify_data)
        
        print(f"\n📥 Response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Position modified successfully!")
            result = response.json()
            print(f"   Result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Failed: {response.text}")
            
        # Clean up - close the position
        print("\n🧹 Closing test position...")
        close_response = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
        if close_response.status_code == 200:
            print("✅ Position closed")

if __name__ == "__main__":
    test_position_modification()