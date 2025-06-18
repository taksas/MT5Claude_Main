#!/usr/bin/env python3
"""
Check supported filling modes for tradable symbols
"""

import requests

def check_filling_modes(api_base="http://172.28.144.1:8000"):
    """Check filling modes for all tradable symbols"""
    
    # Filling mode explanations
    FILLING_MODES = {
        0: "FILL_OR_KILL (FOK) - All or nothing, immediate",
        1: "IMMEDIATE_OR_CANCEL (IOC) - Fill what's possible, cancel rest", 
        2: "RETURN - Standard market execution",
        3: "FILL_OR_KILL (alternative)"
    }
    
    print("="*80)
    print("SYMBOL FILLING MODES CHECK")
    print("="*80)
    
    # Get tradable symbols
    try:
        response = requests.get(f"{api_base}/market/symbols/tradable")
        if response.status_code != 200:
            print("Failed to get tradable symbols")
            return
        
        symbols = response.json()
        print(f"Found {len(symbols)} tradable symbols\n")
        
        # Check each symbol
        filling_stats = {}
        
        for symbol in symbols[:10]:  # Check first 10 as sample
            try:
                response = requests.get(f"{api_base}/market/symbols/{symbol}")
                if response.status_code == 200:
                    info = response.json()
                    filling_mode = info.get('filling_mode', 'Unknown')
                    
                    if filling_mode not in filling_stats:
                        filling_stats[filling_mode] = []
                    filling_stats[filling_mode].append(symbol)
                    
                    print(f"{symbol:<10} - Filling mode: {filling_mode} "
                          f"({FILLING_MODES.get(filling_mode, 'Unknown type')})")
            except Exception as e:
                print(f"{symbol:<10} - Error: {e}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        
        for mode, symbols in filling_stats.items():
            print(f"\nFilling mode {mode} ({FILLING_MODES.get(mode, 'Unknown')}):")
            print(f"  Symbols: {', '.join(symbols)}")
            print(f"  Count: {len(symbols)}")
        
        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION:")
        print("="*80)
        
        if filling_stats:
            most_common = max(filling_stats.keys(), key=lambda k: len(filling_stats[k]))
            print(f"Most common filling mode: {most_common}")
            print(f"Description: {FILLING_MODES.get(most_common, 'Unknown')}")
            print(f"\nUse 'type_filling': {most_common} in order data")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_filling_modes()