#!/usr/bin/env python3
"""Test API connectivity and basic functionality"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

def test_api():
    print("Testing MT5 API connectivity...\n")
    
    # Test 1: Basic connection
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        print(f"✓ API connection: {response.status_code}")
        print(f"  Response: {response.text}\n")
    except Exception as e:
        print(f"✗ API connection failed: {e}\n")
        return
    
    # Test 2: Account info
    try:
        response = requests.get(f"{API_BASE}/account", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Account info:")
            print(f"  Balance: ¥{data.get('balance', 0):,.0f}")
            print(f"  Equity: ¥{data.get('equity', 0):,.0f}")
            print(f"  Server: {data.get('server', 'Unknown')}\n")
        else:
            print(f"✗ Account info failed: {response.status_code}\n")
    except Exception as e:
        print(f"✗ Account info error: {e}\n")
    
    # Test 3: Discover symbols
    try:
        response = requests.get(f"{API_BASE}/market/symbols/tradable", timeout=10)
        if response.status_code == 200:
            symbols = response.json()
            print(f"✓ Discovered {len(symbols)} symbols")
            # Show first 10 symbols
            print(f"  First 10: {symbols[:10]}\n")
            
            # Check for # suffix symbols
            hash_symbols = [s for s in symbols if s.endswith('#')]
            print(f"  Symbols with #: {len(hash_symbols)}")
            if hash_symbols:
                print(f"  Examples: {hash_symbols[:5]}\n")
        else:
            print(f"✗ Symbol discovery failed: {response.status_code}\n")
    except Exception as e:
        print(f"✗ Symbol discovery error: {e}\n")
    
    # Test 4: Get prices for a test symbol
    # Use actual symbols from the discovered list
    test_symbols = ["EURNOK#", "EURTRY#", "USDMXN#", "CADJPY#", "EURCAD#"]
    for symbol in test_symbols:
        try:
            response = requests.get(f"{API_BASE}/market/price/{symbol}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Price for {symbol}:")
                print(f"  Bid: {data.get('bid', 0)}")
                print(f"  Ask: {data.get('ask', 0)}")
                print(f"  Spread: {data.get('spread', 0)}\n")
                break  # Found a working symbol
            else:
                print(f"  {symbol} not available")
        except Exception as e:
            print(f"  {symbol} error: {e}")
    
    # Test 5: Get open positions
    try:
        response = requests.get(f"{API_BASE}/trading/positions", timeout=5)
        if response.status_code == 200:
            positions = response.json()
            print(f"✓ Open positions: {len(positions)}")
            if positions:
                for pos in positions[:3]:  # Show first 3
                    print(f"  {pos.get('symbol')}: {pos.get('type')} {pos.get('volume')} lots")
        else:
            print(f"✗ Positions query failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Positions query error: {e}")

if __name__ == "__main__":
    test_api()