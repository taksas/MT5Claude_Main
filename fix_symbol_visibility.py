#!/usr/bin/env python3
"""Script to make symbols visible in MT5 Market Watch"""

import requests
import json
import time

API_BASE = "http://172.28.144.1:8000"

def get_all_symbols():
    """Get all available symbols"""
    response = requests.get(f"{API_BASE}/market/symbols/all")
    if response.status_code == 200:
        return response.json()
    return []

def select_symbol(symbol):
    """Try to select a symbol in Market Watch"""
    # This would need to be done through the MT5 API
    # For now, we'll document what needs to be done
    print(f"  → Symbol {symbol} needs to be added to Market Watch in MT5")

def main():
    print("="*60)
    print("SYMBOL VISIBILITY CHECK")
    print("="*60)
    
    # Get tradable symbols
    response = requests.get(f"{API_BASE}/market/symbols/tradable")
    if response.status_code != 200:
        print("❌ Failed to get tradable symbols")
        return
        
    tradable_symbols = response.json()
    print(f"\nFound {len(tradable_symbols)} tradable symbols")
    
    # Check each symbol
    invisible_symbols = []
    for symbol in tradable_symbols[:10]:  # Check first 10
        response = requests.get(f"{API_BASE}/market/symbols/{symbol}")
        if response.status_code == 200:
            info = response.json()
            if not info.get('visible', False):
                invisible_symbols.append(symbol)
                print(f"❌ {symbol} - Not visible in Market Watch")
            else:
                print(f"✅ {symbol} - Visible")
    
    if invisible_symbols:
        print(f"\n⚠️  {len(invisible_symbols)} symbols are not visible in Market Watch!")
        print("\n📝 SOLUTION:")
        print("1. Open MT5 terminal on Windows")
        print("2. Go to View → Market Watch (or press Ctrl+M)")
        print("3. Right-click in Market Watch window")
        print("4. Select 'Symbols' or 'Show All'")
        print("5. Find and add these symbols:")
        for symbol in invisible_symbols:
            print(f"   - {symbol}")
        print("\nAlternatively, right-click and select 'Show All' to show all available symbols")
        
        print("\n🔧 MANUAL FIX IN MT5:")
        print("The symbols ending with '#' might be in a different category.")
        print("Look for them under:")
        print("  - Forex")
        print("  - Forex Majors")
        print("  - Or your broker's specific category")
    else:
        print("\n✅ All checked symbols are visible!")

if __name__ == "__main__":
    main()