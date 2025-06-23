#!/usr/bin/env python3
"""
Direct API testing to debug market data issues
"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

def test_market_history():
    """Test market history endpoint directly"""
    print("Testing Market History Endpoint")
    print("=" * 50)
    
    # Test with a known symbol
    test_symbol = "EURNOK#"
    
    payload = {
        "symbol": test_symbol,
        "timeframe": "M5",
        "count": 10
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE}/market/history", json=payload, timeout=5)
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
                
                # Print first few items if it's a list
                if 'candles' in data:
                    print(f"\nCandles type: {type(data['candles'])}")
                    print(f"Number of candles: {len(data['candles'])}")
                    if len(data['candles']) > 0:
                        print(f"\nFirst candle: {data['candles'][0]}")
            elif isinstance(data, list):
                print(f"Response is a list with {len(data)} items")
                if len(data) > 0:
                    print(f"\nFirst item: {data[0]}")
                    print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_symbol_info():
    """Test symbol info endpoint"""
    print("\n\nTesting Symbol Info Endpoint")
    print("=" * 50)
    
    test_symbol = "EURNOK#"
    
    try:
        response = requests.get(f"{API_BASE}/market/symbols/{test_symbol}", timeout=5)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nSymbol info keys: {list(data.keys())}")
            print(f"Full response: {json.dumps(data, indent=2)}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_market_history()
    test_symbol_info()