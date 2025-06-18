#!/usr/bin/env python3
"""Test order placement with fixed return code handling"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def test_order_placement():
    """Test order placement with the fixed API"""
    print("="*60)
    print("TESTING FIXED ORDER PLACEMENT")
    print("="*60)
    
    # Check API status
    response = requests.get(f"{API_BASE}/status/mt5")
    if response.status_code != 200:
        print("❌ API connection failed")
        return
        
    status = response.json()
    print(f"✅ API connected: {status['connected']}")
    print(f"✅ Trading allowed: {status['trade_allowed']}")
    
    # Get account info
    response = requests.get(f"{API_BASE}/account/")
    if response.status_code == 200:
        account = response.json()
        print(f"✅ Account balance: {account.get('balance')} {account.get('currency')}")
    
    # Test with EURUSD#
    symbol = "EURUSD#"
    print(f"\n🧪 Testing order for {symbol}")
    
    # Get market data
    response = requests.post(f"{API_BASE}/market/history", 
                           json={"symbol": symbol, "timeframe": "M1", "count": 1})
    if response.status_code != 200:
        print(f"❌ Failed to get market data: {response.status_code}")
        return
        
    history = response.json()
    if not history:
        print("❌ No price data available")
        return
        
    current_price = history[0]['close']
    print(f"📊 Current price: {current_price}")
    
    # Place order
    pip = 0.0001
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": round(current_price - 50 * pip, 5),
        "tp": round(current_price + 50 * pip, 5),
        "comment": "Fixed API test"
    }
    
    print(f"\n📤 Placing order...")
    print(f"   SL: {order_data['sl']} (-50 pips)")
    print(f"   TP: {order_data['tp']} (+50 pips)")
    
    response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
    print(f"\n📥 Response: {response.status_code}")
    
    if response.status_code == 201:
        print("✅ ORDER PLACED SUCCESSFULLY!")
        result = response.json()
        print(f"   Order ID: {result.get('order')}")
        print(f"   Price: {result.get('price')}")
        print(f"   Volume: {result.get('volume')}")
        
        # Wait and close
        ticket = result.get('order')
        if ticket:
            print("\n⏳ Waiting 3 seconds before closing...")
            time.sleep(3)
            
            print("🔄 Closing position...")
            close_response = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
            if close_response.status_code == 200:
                print("✅ Position closed successfully")
            else:
                print(f"❌ Failed to close: {close_response.text}")
                
    else:
        print(f"❌ Order failed: {response.text}")
        
        # If it's still about symbol visibility
        if "visible" in response.text or "Market Watch" in response.text:
            print("\n⚠️  SYMBOL VISIBILITY ISSUE")
            print("Please add symbols to Market Watch in MT5:")
            print("1. Open MT5")
            print("2. Press Ctrl+M for Market Watch")
            print("3. Right-click and select 'Show All'")
            print("4. Find and enable EURUSD#")

if __name__ == "__main__":
    test_order_placement()