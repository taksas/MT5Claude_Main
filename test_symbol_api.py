#!/usr/bin/env python3
"""
Test symbol API with and without # and URL encoding
"""

import requests
from urllib.parse import quote

API_BASE = "http://172.28.144.1:8000"

print("Testing Symbol API Endpoints")
print("=" * 60)

# Test different symbol formats
test_cases = [
    ("EURUSD", "Plain symbol without #"),
    ("EURUSD#", "Symbol with # (not encoded)"),
    (quote("EURUSD#"), "Symbol with # (URL encoded)"),
    ("EURUSD%23", "Symbol with # (manually encoded)"),
]

print("\n1. Testing /market/symbols/{symbol} endpoint:")
for symbol, description in test_cases:
    url = f"{API_BASE}/market/symbols/{symbol}"
    print(f"\n  Testing: {description}")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, timeout=2)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Success! Got data for: {data.get('name', 'unknown')}")
            print(f"    Bid: {data.get('bid', 'N/A')}, Ask: {data.get('ask', 'N/A')}")
        else:
            print(f"  ✗ Failed: {response.text[:100]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Test market history endpoint
print("\n\n2. Testing /market/history endpoint:")
test_symbols = ["EURUSD", "EURUSD#"]
for symbol in test_symbols:
    print(f"\n  Testing: {symbol}")
    try:
        response = requests.post(
            f"{API_BASE}/market/history",
            json={"symbol": symbol, "timeframe": "M1", "count": 1},
            timeout=2
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                print(f"  ✓ Success! Got {len(data)} candles")
                print(f"    Latest close: {data[0].get('close', 'N/A')}")
        else:
            print(f"  ✗ Failed: {response.text[:100]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Get list of tradable symbols to see format
print("\n\n3. Checking tradable symbols format:")
response = requests.get(f"{API_BASE}/market/symbols/tradable")
if response.status_code == 200:
    symbols = response.json()
    print(f"  First 5 symbols: {symbols[:5]}")
    has_hash = all('#' in s for s in symbols[:10])
    print(f"  All symbols have #: {has_hash}")