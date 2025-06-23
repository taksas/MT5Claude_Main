#!/usr/bin/env python3
"""Close all open positions to reset the account"""

import requests
import time

API_BASE = "http://172.28.144.1:8000"

print("=== Closing All Open Positions ===\n")

# Get open positions
try:
    response = requests.get(f"{API_BASE}/trading/positions")
    if response.status_code == 200:
        positions = response.json()
        print(f"Found {len(positions)} open positions\n")
        
        for pos in positions:
            ticket = pos.get('ticket')
            symbol = pos.get('symbol')
            volume = pos.get('volume')
            position_type = pos.get('type')
            
            print(f"Closing position {ticket} ({symbol})...")
            
            # Create close order
            close_order = {
                "action": 1,  # TRADE_ACTION_DEAL
                "position": ticket,
                "symbol": symbol,
                "volume": volume,
                "type": 1 if position_type == 0 else 0,  # Opposite type to close
                "deviation": 50,
                "magic": 123456,
                "comment": "Close position"
            }
            
            try:
                response = requests.post(
                    f"{API_BASE}/trading/orders",
                    json=close_order,
                    timeout=10
                )
                
                if response.status_code in [200, 201]:
                    print(f"✓ Position {ticket} closed successfully")
                else:
                    print(f"✗ Failed to close position {ticket}: {response.text}")
                    
                time.sleep(1)  # Small delay between orders
                
            except Exception as e:
                print(f"✗ Error closing position {ticket}: {e}")
        
        print("\n=== All positions processed ===")
        
        # Check final account status
        time.sleep(2)
        response = requests.get(f"{API_BASE}/account/")
        if response.status_code == 200:
            account = response.json()
            print(f"\nFinal account status:")
            print(f"  Balance: ¥{account.get('balance', 0):,.0f}")
            print(f"  Equity: ¥{account.get('equity', 0):,.0f}")
            
except Exception as e:
    print(f"Error: {e}")