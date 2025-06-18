#!/usr/bin/env python3
"""Comprehensive diagnostic for order placement issues"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def get_tick_info(symbol):
    """Get current tick information"""
    # Try getting tick data through history
    response = requests.post(f"{API_BASE}/market/history", 
                           json={"symbol": symbol, "timeframe": "M1", "count": 1})
    if response.status_code == 200:
        history = response.json()
        if history:
            return history[0]
    return None

def diagnose_symbol(symbol):
    """Run comprehensive diagnostics on a symbol"""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {symbol}")
    print('='*60)
    
    # 1. Get symbol info
    response = requests.get(f"{API_BASE}/market/symbols/{symbol}")
    if response.status_code != 200:
        print(f"❌ Failed to get symbol info: {response.status_code}")
        return
    
    symbol_info = response.json()
    print("\n📊 Symbol Information:")
    print(f"  Name: {symbol_info.get('name')}")
    print(f"  Visible: {symbol_info.get('visible')}")
    print(f"  Trade mode: {symbol_info.get('trade_mode')} ({symbol_info.get('trade_mode_description', 'Unknown')})")
    print(f"  Filling mode: {symbol_info.get('filling_mode')} (binary: {bin(symbol_info.get('filling_mode', 0))})")
    print(f"  Min volume: {symbol_info.get('volume_min')}")
    print(f"  Max volume: {symbol_info.get('volume_max')}")
    print(f"  Volume step: {symbol_info.get('volume_step')}")
    print(f"  Digits: {symbol_info.get('digits')}")
    print(f"  Point: {symbol_info.get('point')}")
    print(f"  Spread: {symbol_info.get('spread')}")
    print(f"  Stops level: {symbol_info.get('trade_stops_level')}")
    
    # 2. Get current prices
    tick = get_tick_info(symbol)
    if not tick:
        print("\n❌ No price data available")
        return
    
    current_price = tick.get('close', 0)
    print(f"\n💹 Current Price: {current_price}")
    
    # 3. Account info
    acc_response = requests.get(f"{API_BASE}/account/")
    if acc_response.status_code == 200:
        account = acc_response.json()
        print(f"\n💰 Account Info:")
        print(f"  Balance: {account.get('balance')} {account.get('currency')}")
        print(f"  Equity: {account.get('equity')}")
        print(f"  Free margin: {account.get('margin_free')}")
        print(f"  Leverage: {account.get('leverage')}")
        print(f"  Trade allowed: {account.get('trade_allowed')}")
        print(f"  Trade expert: {account.get('trade_expert')}")
    
    # 4. Calculate order parameters
    digits = symbol_info.get('digits', 5)
    point = symbol_info.get('point', 0.00001)
    stops_level = symbol_info.get('trade_stops_level', 0)
    min_distance = max(stops_level * point, 10 * point)  # At least 10 points
    
    print(f"\n📏 Calculated Parameters:")
    print(f"  Minimum stop distance: {min_distance:.{digits}f} ({stops_level} points)")
    
    # 5. Test orders with different configurations
    print(f"\n🧪 Testing Order Configurations:")
    
    test_configs = [
        {
            "name": "Auto filling mode",
            "order": {
                "action": 1,
                "symbol": symbol,
                "volume": symbol_info.get('volume_min', 0.01),
                "type": 0,  # BUY
                "sl": round(current_price - min_distance * 5, digits),
                "tp": round(current_price + min_distance * 5, digits),
                "comment": "Test auto"
            }
        },
        {
            "name": "Filling mode 1 (IOC)",
            "order": {
                "action": 1,
                "symbol": symbol,
                "volume": symbol_info.get('volume_min', 0.01),
                "type": 0,  # BUY
                "sl": round(current_price - min_distance * 5, digits),
                "tp": round(current_price + min_distance * 5, digits),
                "type_filling": 1,
                "comment": "Test IOC"
            }
        },
        {
            "name": "No SL/TP",
            "order": {
                "action": 1,
                "symbol": symbol,
                "volume": symbol_info.get('volume_min', 0.01),
                "type": 0,  # BUY
                "comment": "Test no SLTP"
            }
        },
        {
            "name": "Market SELL",
            "order": {
                "action": 1,
                "symbol": symbol,
                "volume": symbol_info.get('volume_min', 0.01),
                "type": 1,  # SELL
                "sl": round(current_price + min_distance * 5, digits),
                "tp": round(current_price - min_distance * 5, digits),
                "comment": "Test SELL"
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n  Testing: {config['name']}")
        print(f"  Order: {json.dumps(config['order'], indent=4)}")
        
        response = requests.post(f"{API_BASE}/trading/orders", json=config['order'])
        print(f"  Response: {response.status_code}")
        
        if response.status_code == 201:
            print(f"  ✅ SUCCESS!")
            result = response.json()
            ticket = result.get('order')
            print(f"  Order ticket: {ticket}")
            print(f"  Price: {result.get('price')}")
            
            # Close the position
            if ticket:
                time.sleep(2)
                close_resp = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
                print(f"  Position closed: {close_resp.status_code}")
            break
        else:
            error_text = response.text[:200]
            print(f"  ❌ Error: {error_text}")
            
            # Parse error details
            if "retcode: 0" in error_text:
                print("  ⚠️  Retcode 0 usually means success, but order_check is failing")
                print("  ⚠️  This might indicate:")
                print("     - Market is closed")
                print("     - Symbol is not tradable")
                print("     - Account restrictions")
                print("     - Insufficient margin")

def main():
    print("="*60)
    print("MT5 ORDER PLACEMENT DIAGNOSTIC")
    print("="*60)
    
    # Check API status
    response = requests.get(f"{API_BASE}/status/mt5")
    if response.status_code != 200:
        print("❌ API connection failed")
        return
        
    status = response.json()
    print(f"✅ API connected: {status['connected']}")
    print(f"✅ Trading allowed: {status['trade_allowed']}")
    if 'trade_expert' in status:
        print(f"✅ Expert allowed: {status['trade_expert']}")
    
    # Test multiple symbols
    symbols = ["EURUSD#", "USDJPY#", "GBPUSD#"]
    
    for symbol in symbols:
        diagnose_symbol(symbol)
        print("\n" + "-"*60)

if __name__ == "__main__":
    main()