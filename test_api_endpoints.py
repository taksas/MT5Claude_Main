#!/usr/bin/env python3
"""
Test available API endpoints
"""

import requests

API_BASE = "http://172.28.144.1:8000"

def test_api_docs():
    """Check API documentation"""
    print("Checking API documentation...")
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=5)
        print(f"Docs endpoint status: {response.status_code}")
        
        # Also try OpenAPI schema
        response = requests.get(f"{API_BASE}/openapi.json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\nAvailable endpoints:")
            for path, methods in data.get('paths', {}).items():
                for method in methods:
                    print(f"  {method.upper()} {path}")
    except Exception as e:
        print(f"Error: {e}")

def test_symbol_endpoints():
    """Test various ways to get symbol data"""
    print("\n\nTesting symbol data endpoints...")
    test_symbol = "EURNOK#"
    
    # Try different endpoints
    endpoints = [
        ("GET", f"/market/symbols/{test_symbol}"),
        ("GET", f"/market/symbols/{test_symbol.rstrip('#')}"),
        ("GET", f"/market/{test_symbol}/tick"),
        ("GET", f"/market/{test_symbol.rstrip('#')}/tick"),
    ]
    
    for method, endpoint in endpoints:
        try:
            url = f"{API_BASE}{endpoint}"
            print(f"\nTrying {method} {endpoint}")
            
            if method == "GET":
                response = requests.get(url, timeout=2)
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_api_docs()
    test_symbol_endpoints()