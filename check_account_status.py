#!/usr/bin/env python3
"""Check current account status and positions"""

import requests

API_BASE = "http://172.28.144.1:8000"

print("=== Account Status Check ===\n")

# Get account info
try:
    response = requests.get(f"{API_BASE}/account/")
    if response.status_code == 200:
        account = response.json()
        print("Account Information:")
        print(f"  Balance: ¥{account.get('balance', 0):,.0f}")
        print(f"  Equity: ¥{account.get('equity', 0):,.0f}")
        print(f"  Margin: ¥{account.get('margin', 0):,.0f}")
        print(f"  Free Margin: ¥{account.get('margin_free', 0):,.0f}")
        print(f"  Profit/Loss: ¥{account.get('profit', 0):,.0f}")
        
        # Calculate drawdown
        balance = account.get('balance', 1)
        equity = account.get('equity', balance)
        drawdown = ((balance - equity) / balance) * 100
        print(f"  Drawdown: {drawdown:.1f}%\n")
except Exception as e:
    print(f"Error getting account info: {e}\n")

# Get open positions
try:
    response = requests.get(f"{API_BASE}/trading/positions")
    if response.status_code == 200:
        positions = response.json()
        print(f"Open Positions: {len(positions)}")
        
        if positions:
            total_profit = 0
            for pos in positions:
                print(f"\n  Position {pos.get('ticket')}:")
                print(f"    Symbol: {pos.get('symbol')}")
                print(f"    Type: {pos.get('type_description', pos.get('type'))}")
                print(f"    Volume: {pos.get('volume')}")
                print(f"    Open Price: {pos.get('price_open')}")
                print(f"    Current Price: {pos.get('price_current')}")
                print(f"    Profit: ¥{pos.get('profit', 0):,.0f}")
                print(f"    Comment: {pos.get('comment', '')}")
                total_profit += pos.get('profit', 0)
            
            print(f"\n  Total P/L from positions: ¥{total_profit:,.0f}")
except Exception as e:
    print(f"Error getting positions: {e}")

print("\n=== Recommendations ===")
print("If drawdown is high:")
print("1. Close losing positions manually in MT5")
print("2. Wait for equity to recover")
print("3. Reduce risk settings in trading_config.py")
print("4. Consider using smaller position sizes")