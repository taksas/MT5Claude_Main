#!/usr/bin/env python3
"""Force a test order to verify filling mode fix"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def get_market_history(symbol):
    """Get market history to find current price"""
    data = {"symbol": symbol, "timeframe": 1, "count": 1}
    response = requests.post(f"{API_BASE}/market/history", json=data)
    if response.status_code == 200:
        history = response.json()
        if history and len(history) > 0:
            return history[0]['close']
    return None

def test_order_placement(symbol="EURUSD#"):
    """Test order placement with the fixed filling mode"""
    print(f"\n🧪 Testing order for {symbol}")
    
    # Get current price from history
    current_price = get_market_history(symbol)
    if not current_price:
        print(f"❌ Could not get price for {symbol}")
        return
    
    print(f"Current price: {current_price}")
    
    # Calculate stop loss and take profit
    pip_value = 0.0001 if "JPY" not in symbol else 0.01
    sl_distance = 20 * pip_value  # 20 pips for safety
    tp_distance = 30 * pip_value  # 30 pips
    
    # BUY order parameters
    stop_loss = round(current_price - sl_distance, 5)
    take_profit = round(current_price + tp_distance, 5)
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": stop_loss,
        "tp": take_profit,
        "comment": "Filling mode test"
        # NOT specifying type_filling - let API auto-detect
    }
    
    print(f"\nOrder details:")
    print(f"  Entry: Market price")
    print(f"  Stop Loss: {stop_loss} ({sl_distance/pip_value:.0f} pips)")
    print(f"  Take Profit: {take_profit} ({tp_distance/pip_value:.0f} pips)")
    
    # Send order
    print(f"\nSending order...")
    response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 201:
        print("✅ Order placed successfully! Filling mode fix is working!")
        result = response.json()
        order_id = result.get('order')
        print(f"Order ID: {order_id}")
        
        # Wait then close
        print("\n⏳ Waiting 3 seconds before closing...")
        time.sleep(3)
        
        if order_id:
            print(f"🔄 Closing position {order_id}...")
            close_response = requests.delete(f"{API_BASE}/trading/positions/{order_id}")
            print(f"Close status: {close_response.status_code}")
            if close_response.status_code == 200:
                print("✅ Position closed successfully")
    else:
        print("❌ Order failed - filling mode issue still exists")
        
def check_positions():
    """Check current open positions"""
    response = requests.get(f"{API_BASE}/trading/positions")
    if response.status_code == 200:
        positions = response.json()
        print(f"\n📊 Open positions: {len(positions)}")
        for pos in positions:
            print(f"  - {pos['symbol']}: {pos['type_str']} {pos['volume']} @ {pos['price_open']}")

if __name__ == "__main__":
    print("="*60)
    print("FILLING MODE FIX VERIFICATION")
    print("="*60)
    
    # Check API status
    response = requests.get(f"{API_BASE}/status/mt5")
    if response.status_code == 200:
        status = response.json()
        print(f"✅ API connected: {status['connected']}")
        print(f"✅ Trading allowed: {status['trade_allowed']}")
    
    # Check current positions
    check_positions()
    
    # Test order placement
    test_order_placement("EURUSD#")
    
    # Check positions again
    check_positions()