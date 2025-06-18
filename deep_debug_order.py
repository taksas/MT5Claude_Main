#!/usr/bin/env python3
"""Deep debugging for order check issue"""

import requests
import json
import time
from datetime import datetime

API_BASE = "http://172.28.144.1:8000"

def check_market_status():
    """Check if market is open"""
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    
    print(f"\n🕐 Current time: {now}")
    print(f"   Weekday: {weekday} (0=Monday, 6=Sunday)")
    print(f"   Hour: {hour}")
    
    # Forex market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
    if weekday == 5:  # Saturday
        print("   ⚠️ Market is CLOSED (Saturday)")
        return False
    elif weekday == 6:  # Sunday
        if hour < 22:
            print("   ⚠️ Market is CLOSED (Sunday before 22:00)")
            return False
    elif weekday == 4 and hour >= 22:  # Friday after 22:00
        print("   ⚠️ Market is CLOSED (Friday after 22:00)")
        return False
    
    print("   ✅ Market should be OPEN")
    return True

def test_with_different_symbols():
    """Test different symbol variations"""
    symbols_to_test = [
        "EURUSD#",
        "EURUSD",
        "USDJPY#", 
        "USDJPY",
        "GBPUSD#",
        "GBPUSD"
    ]
    
    for symbol in symbols_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {symbol}")
        print('='*60)
        
        # Get symbol info
        response = requests.get(f"{API_BASE}/market/symbols/{symbol}")
        if response.status_code != 200:
            print(f"❌ Symbol not found")
            continue
            
        info = response.json()
        print(f"✅ Symbol found:")
        print(f"   Name: {info.get('name')}")
        print(f"   Visible: {info.get('visible')}")
        print(f"   Trade mode: {info.get('trade_mode')}")
        print(f"   Filling mode: {info.get('filling_mode')}")
        
        if not info.get('visible'):
            print(f"   ⚠️ Symbol is not visible in Market Watch!")
            continue
            
        # Try to place order
        response = requests.post(f"{API_BASE}/market/history", 
                               json={"symbol": symbol, "timeframe": "M1", "count": 1})
        if response.status_code == 200:
            history = response.json()
            if history:
                price = history[0]['close']
                pip = 0.0001 if "JPY" not in symbol else 0.01
                
                order_data = {
                    "action": 1,
                    "symbol": symbol,
                    "volume": 0.01,
                    "type": 0,
                    "sl": round(price - 50 * pip, 5),
                    "tp": round(price + 50 * pip, 5),
                    "comment": f"Test {symbol}"
                }
                
                print(f"\n   Attempting order at price: {price}")
                response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
                
                if response.status_code == 201:
                    print(f"   ✅ SUCCESS! Order placed!")
                    result = response.json()
                    ticket = result.get('order')
                    if ticket:
                        time.sleep(2)
                        requests.delete(f"{API_BASE}/trading/positions/{ticket}")
                    return True
                else:
                    print(f"   ❌ Failed: {response.text[:100]}")
    
    return False

def test_with_price_parameter():
    """Test with explicit price parameter"""
    print("\n" + "="*60)
    print("Testing with explicit price parameter")
    print("="*60)
    
    symbol = "EURUSD#"
    
    # Get current tick
    response = requests.post(f"{API_BASE}/market/history", 
                           json={"symbol": symbol, "timeframe": "M1", "count": 1})
    if response.status_code == 200:
        history = response.json()
        if history:
            last_close = history[0]['close']
            
            # Try with explicit price
            order_data = {
                "action": 1,
                "symbol": symbol,
                "volume": 0.01,
                "type": 0,  # BUY
                "price": last_close,  # Add explicit price
                "sl": round(last_close - 0.0050, 5),
                "tp": round(last_close + 0.0050, 5),
                "deviation": 50,  # Larger deviation
                "comment": "Test with price"
            }
            
            print(f"Order with price: {json.dumps(order_data, indent=2)}")
            response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
            print(f"Response: {response.status_code} - {response.text[:200]}")

def check_account_details():
    """Check detailed account information"""
    print("\n" + "="*60)
    print("Account Details Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/account/")
    if response.status_code == 200:
        account = response.json()
        
        # Check for demo/real account
        print(f"Login: {account.get('login')}")
        print(f"Server: {account.get('server', 'N/A')}")
        print(f"Company: {account.get('company', 'N/A')}")
        print(f"Name: {account.get('name', 'N/A')}")
        print(f"Trade mode: {account.get('trade_mode', 'N/A')}")
        print(f"Limit orders: {account.get('limit_orders', 'N/A')}")
        
        # Check if it's a demo account
        if 'demo' in str(account.get('server', '')).lower():
            print("\n✅ This appears to be a DEMO account")
        else:
            print("\n⚠️ This might be a REAL account - be careful!")

def main():
    print("="*60)
    print("DEEP ORDER DEBUG")
    print("="*60)
    
    # 1. Check market status
    market_open = check_market_status()
    
    # 2. Check account details
    check_account_details()
    
    # 3. Test different symbols
    if test_with_different_symbols():
        print("\n✅ Found a working symbol!")
    else:
        print("\n❌ No symbols worked")
        
        # 4. Try additional tests
        test_with_price_parameter()
        
        print("\n" + "="*60)
        print("POSSIBLE ISSUES:")
        print("="*60)
        print("1. Market might be closed (check market hours)")
        print("2. Symbols need to be visible in MT5 Market Watch")
        print("3. Account might have restrictions")
        print("4. MT5 terminal might need 'Algo Trading' enabled")
        print("5. The broker might require different order parameters")
        
        if not market_open:
            print("\n⚠️ IMPORTANT: The market appears to be CLOSED!")
            print("   Forex market hours: Sunday 22:00 UTC - Friday 22:00 UTC")

if __name__ == "__main__":
    main()