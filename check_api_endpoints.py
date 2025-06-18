#!/usr/bin/env python3
"""
Check available API endpoints and get tradable symbols
"""

import requests
import json

def check_api_endpoints(api_base="http://172.28.144.1:8000"):
    """Check various possible endpoints for symbols"""
    
    print("Checking API endpoints...")
    print("="*60)
    
    # List of possible endpoints to try
    endpoints = [
        "/status/mt5",
        "/market/symbols",
        "/symbols",
        "/trading/symbols",
        "/market/info",
        "/account/",
        "/market/symbol_info",
        "/market/tick",
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{api_base}{endpoint}"
            print(f"\nTrying {url}...")
            response = requests.get(url, timeout=5)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"Found {len(data)} items")
                    # Check if these are symbols
                    if isinstance(data[0], str):
                        symbols = [s for s in data if "#" in s]
                        if symbols:
                            print(f"Tradable symbols found: {symbols[:5]}...")
                            return symbols
                elif isinstance(data, dict):
                    print(f"Response: {json.dumps(data, indent=2)[:200]}...")
                    
        except Exception as e:
            print(f"Error: {e}")
    
    # Try getting symbol info for known symbols
    print("\n" + "="*60)
    print("Checking known symbols directly...")
    
    known_symbols = ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#", "GBPJPY#", "USDCAD#"]
    working_symbols = []
    
    for symbol in known_symbols:
        try:
            # Try symbol info endpoint
            response = requests.get(f"{api_base}/market/symbols/{symbol}")
            if response.status_code == 200:
                data = response.json()
                if 'bid' in data and data['bid'] > 0:
                    print(f"✓ {symbol} - Bid: {data['bid']}, Ask: {data['ask']}")
                    working_symbols.append(symbol)
            else:
                # Try tick endpoint
                response = requests.post(f"{api_base}/market/tick", json={"symbol": symbol})
                if response.status_code == 200:
                    data = response.json()
                    if 'bid' in data and data['bid'] > 0:
                        print(f"✓ {symbol} - Bid: {data['bid']}, Ask: {data['ask']}")
                        working_symbols.append(symbol)
                        
        except Exception as e:
            print(f"✗ {symbol} - Error: {e}")
    
    return working_symbols

def main():
    print("="*60)
    print("MT5 API ENDPOINT CHECK")
    print("="*60)
    
    symbols = check_api_endpoints()
    
    if symbols:
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"\nFound {len(symbols)} tradable symbols:")
        for symbol in symbols:
            print(f"  - {symbol}")
        
        print(f"\nFor live_trading_engine.py, update to:")
        print(f"self.symbols = {symbols[:4]}")
    else:
        print("\nNo symbols found via API. Using defaults.")
        print("self.symbols = ['EURUSD#', 'USDJPY#', 'GBPUSD#', 'EURJPY#']")

if __name__ == "__main__":
    main()