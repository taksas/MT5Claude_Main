#!/usr/bin/env python3
"""
Test positions API endpoint
"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

print("Testing positions endpoint...")
print("=" * 50)

try:
    response = requests.get(f"{API_BASE}/trading/positions", timeout=5)
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Response is a list with {len(data)} positions")
            if len(data) > 0:
                print(f"\nFirst position structure:")
                print(json.dumps(data[0], indent=2))
        elif isinstance(data, dict):
            print(f"Response is a dict with keys: {list(data.keys())}")
        else:
            print(f"Unexpected response type: {data}")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")

# Now test with the updated API client
print("\n\nTesting with API client...")
from components.mt5_api_client import MT5APIClient

client = MT5APIClient(API_BASE)
positions = client.get_positions()
print(f"API client returned {len(positions)} positions")
print("✅ Positions API working correctly!" if isinstance(positions, list) else "❌ Still having issues")