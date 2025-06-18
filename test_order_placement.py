#!/usr/bin/env python3
"""
Test order placement with proper filling modes
"""

import requests
import json

def test_order_placement(api_base="http://172.28.144.1:8000"):
    """Test placing an order with different configurations"""
    
    print("="*60)
    print("ORDER PLACEMENT TEST")
    print("="*60)
    
    # Test symbol
    symbol = "USDJPY#"
    
    # Get current price
    try:
        response = requests.get(f"{api_base}/market/symbols/{symbol}")
        if response.status_code != 200:
            print(f"Failed to get symbol info: {response.text}")
            return
            
        symbol_info = response.json()
        print(f"\nSymbol: {symbol}")
        print(f"Bid: {symbol_info.get('bid', 'N/A')}")
        print(f"Ask: {symbol_info.get('ask', 'N/A')}")
        print(f"Filling mode: {symbol_info.get('filling_mode', 'N/A')}")
        
        current_price = symbol_info.get('ask', 0)  # Use ask for BUY orders
        
        # Calculate stop loss and take profit
        if "JPY" in symbol:
            pip_value = 0.01
            sl_distance = 10 * pip_value  # 10 pips
            tp_distance = 15 * pip_value  # 15 pips
        else:
            pip_value = 0.0001
            sl_distance = 10 * pip_value
            tp_distance = 15 * pip_value
        
        stop_loss = round(current_price - sl_distance, 3)
        take_profit = round(current_price + tp_distance, 3)
        
        print(f"\nOrder Parameters:")
        print(f"Entry: {current_price}")
        print(f"Stop Loss: {stop_loss} ({10} pips)")
        print(f"Take Profit: {take_profit} ({15} pips)")
        
        # Test different filling modes
        filling_modes = [
            (2, "RETURN - Standard execution"),
            (1, "IOC - Immediate or Cancel"),
            (0, "FOK - Fill or Kill")
        ]
        
        print("\nTesting filling modes:")
        print("-"*40)
        
        for mode, description in filling_modes:
            print(f"\nTesting mode {mode} ({description})...")
            
            order_data = {
                "action": 1,  # DEAL
                "symbol": symbol,
                "volume": 0.01,
                "type": 0,  # BUY
                "type_filling": mode,
                "sl": stop_loss,
                "tp": take_profit,
                "comment": f"Test order - mode {mode}"
            }
            
            print(f"Order data: {json.dumps(order_data, indent=2)}")
            
            # Note: This would actually place an order!
            # Uncomment only if you want to test with real orders
            # response = requests.post(f"{api_base}/trading/orders", json=order_data)
            # print(f"Response: {response.status_code}")
            # print(f"Result: {response.text}")
            
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print("1. Use filling mode 2 (RETURN) as default")
        print("2. Have fallback to mode 1 (IOC) if RETURN fails")
        print("3. Ensure stop loss is always set")
        print("4. Log all order parameters for debugging")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_order_placement()