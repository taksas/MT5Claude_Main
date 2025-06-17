#!/usr/bin/env python3
"""
Script to test which MT5 symbols are actually accessible via the API
"""

import json
import urllib.request
import urllib.error

def test_symbol(base_url, symbol):
    """Test if a symbol is accessible via the API"""
    url = f"{base_url}/market/symbols/{symbol}"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return True, data.get('name', symbol)
            else:
                return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)

def main():
    base_url = "http://172.28.144.1:8000"
    
    # Get the full list of tradable symbols
    print("Getting tradable symbols list...")
    try:
        with urllib.request.urlopen(f"{base_url}/market/symbols/tradable") as response:
            symbols = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error getting tradable symbols: {e}")
        return
    
    print(f"Found {len(symbols)} tradable symbols")
    print("\nTesting individual symbol access...")
    
    # Focus on forex pairs with # suffix for initial testing
    forex_symbols = [s for s in symbols if s.endswith('#') and len(s) == 7]  # XXXYYY# format
    print(f"Testing {len(forex_symbols)} forex pairs...")
    
    working_symbols = []
    broken_symbols = []
    
    for symbol in forex_symbols:
        success, result = test_symbol(base_url, symbol)
        if success:
            working_symbols.append(symbol)
            print(f"✓ {symbol} - OK")
        else:
            broken_symbols.append((symbol, result))
            print(f"✗ {symbol} - {result}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Working symbols: {len(working_symbols)}")
    print(f"Broken symbols: {len(broken_symbols)}")
    
    if working_symbols:
        print(f"\nWorking symbols: {working_symbols}")
    
    if broken_symbols:
        print(f"\nBroken symbols:")
        for symbol, error in broken_symbols:
            print(f"  {symbol}: {error}")
    
    # Save results to JSON file
    results = {
        'working_symbols': working_symbols,
        'broken_symbols': [{'symbol': s, 'error': e} for s, e in broken_symbols],
        'total_tested': len(forex_symbols)
    }
    
    with open('/home/takumi/MT5Claude_Main/symbol_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to symbol_test_results.json")

if __name__ == "__main__":
    main()