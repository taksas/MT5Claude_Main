#!/usr/bin/env python3
"""Debug script to understand the filling mode issue"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

def debug_symbol_info(symbol):
    """Get detailed symbol information and debug filling modes"""
    print(f"\n🔍 Debugging symbol: {symbol}")
    print("="*60)
    
    # Get symbol info
    response = requests.get(f"{API_BASE}/market/symbols/{symbol}")
    if response.status_code != 200:
        print(f"❌ Failed to get symbol info: {response.status_code}")
        return
    
    info = response.json()
    filling_mode = info.get('filling_mode', 'Not found')
    
    print(f"📊 Symbol filling_mode value: {filling_mode}")
    
    # Decode the filling mode bits
    if isinstance(filling_mode, int):
        print("\nFilling mode breakdown:")
        print(f"  - Binary: {bin(filling_mode)}")
        print(f"  - FOK (0x01):    {'✓' if filling_mode & 1 else '✗'}")
        print(f"  - IOC (0x02):    {'✓' if filling_mode & 2 else '✗'}")
        print(f"  - RETURN (0x04): {'✓' if filling_mode & 4 else '✗'}")
    
    return filling_mode

def test_order_with_mode(symbol, mode):
    """Test an order with a specific filling mode"""
    print(f"\n📝 Testing order with filling mode {mode}")
    
    # Get current price
    response = requests.post(f"{API_BASE}/market/history", 
                           json={"symbol": symbol, "timeframe": "M1", "count": 1})
    if response.status_code != 200:
        print(f"❌ Failed to get price: {response.status_code}")
        return
    
    history = response.json()
    if not history:
        print("❌ No price data available")
        return
    
    current_price = history[0]['close']
    pip = 0.0001 if "JPY" not in symbol else 0.01
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": round(current_price - 20 * pip, 5),
        "tp": round(current_price + 30 * pip, 5),
        "type_filling": mode,
        "comment": f"Test mode {mode}"
    }
    
    print(f"Order data: {json.dumps(order_data, indent=2)}")
    
    response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
    print(f"Response: {response.status_code} - {response.text}")
    
    if response.status_code == 201:
        print("✅ Order successful!")
        result = response.json()
        ticket = result.get('order')
        if ticket:
            # Close the position
            import time
            time.sleep(2)
            close_resp = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
            print(f"Position closed: {close_resp.status_code}")

def test_order_without_mode(symbol):
    """Test an order without specifying filling mode"""
    print(f"\n📝 Testing order WITHOUT filling mode (auto-detect)")
    
    # Get current price
    response = requests.post(f"{API_BASE}/market/history", 
                           json={"symbol": symbol, "timeframe": "M1", "count": 1})
    if response.status_code != 200:
        print(f"❌ Failed to get price: {response.status_code}")
        return
    
    history = response.json()
    if not history:
        print("❌ No price data available")
        return
    
    current_price = history[0]['close']
    pip = 0.0001 if "JPY" not in symbol else 0.01
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": round(current_price - 20 * pip, 5),
        "tp": round(current_price + 30 * pip, 5),
        # NOT specifying type_filling
        "comment": "Test auto mode"
    }
    
    print(f"Order data: {json.dumps(order_data, indent=2)}")
    
    response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
    print(f"Response: {response.status_code} - {response.text}")
    
    if response.status_code == 201:
        print("✅ Order successful!")
        result = response.json()
        ticket = result.get('order')
        if ticket:
            # Close the position
            import time
            time.sleep(2)
            close_resp = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
            print(f"Position closed: {close_resp.status_code}")

if __name__ == "__main__":
    print("="*60)
    print("FILLING MODE DEBUG SCRIPT")
    print("="*60)
    
    # Check API status
    response = requests.get(f"{API_BASE}/status/mt5")
    if response.status_code == 200:
        status = response.json()
        print(f"✅ API connected: {status['connected']}")
        print(f"✅ Trading allowed: {status['trade_allowed']}")
    else:
        print("❌ API connection failed")
        exit(1)
    
    # Test with EURUSD#
    symbol = "EURUSD#"
    filling_mode = debug_symbol_info(symbol)
    
    # Test without specifying mode (should auto-detect)
    test_order_without_mode(symbol)
    
    # If we know the filling mode, test with specific modes
    if isinstance(filling_mode, int):
        if filling_mode == 1:
            test_order_with_mode(symbol, 0)  # Try FOK
        elif filling_mode == 2:
            test_order_with_mode(symbol, 1)  # Try IOC
        elif filling_mode == 4:
            test_order_with_mode(symbol, 2)  # Try RETURN
        else:
            # Try different modes based on bits
            if filling_mode & 1:
                test_order_with_mode(symbol, 0)  # FOK
            elif filling_mode & 2:
                test_order_with_mode(symbol, 1)  # IOC
            elif filling_mode & 4:
                test_order_with_mode(symbol, 2)  # RETURN