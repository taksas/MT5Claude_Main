#!/usr/bin/env python3
"""
Get all tradable symbols from MT5 API
"""

import requests
import json

def get_tradable_symbols(api_base="http://172.28.144.1:8000"):
    """Fetch all tradable symbols with # suffix"""
    try:
        # Get all symbols
        response = requests.get(f"{api_base}/market/symbols")
        if response.status_code != 200:
            print(f"Failed to get symbols: {response.status_code}")
            return []
        
        all_symbols = response.json()
        tradable_symbols = []
        
        print("Checking tradable symbols...")
        print("-" * 60)
        
        for symbol in all_symbols:
            if "#" in symbol:  # Only symbols with # are tradable
                # Get symbol info to check if it's really tradable
                info_response = requests.get(f"{api_base}/market/symbols/{symbol}")
                if info_response.status_code == 200:
                    symbol_info = info_response.json()
                    
                    # Check if symbol is visible and has valid pricing
                    if (symbol_info.get('visible', False) and 
                        symbol_info.get('bid', 0) > 0 and 
                        symbol_info.get('ask', 0) > 0):
                        
                        spread = symbol_info['ask'] - symbol_info['bid']
                        spread_points = spread / symbol_info.get('point', 0.00001)
                        
                        tradable_symbols.append({
                            'symbol': symbol,
                            'bid': symbol_info['bid'],
                            'ask': symbol_info['ask'],
                            'spread_points': spread_points,
                            'digits': symbol_info.get('digits', 5)
                        })
                        
                        print(f"✓ {symbol:<15} Bid: {symbol_info['bid']:<10.5f} "
                              f"Ask: {symbol_info['ask']:<10.5f} Spread: {spread_points:.1f} points")
        
        print("-" * 60)
        print(f"Found {len(tradable_symbols)} tradable symbols")
        
        # Sort by spread (best spreads first)
        tradable_symbols.sort(key=lambda x: x['spread_points'])
        
        # Get symbol names only
        symbol_names = [s['symbol'] for s in tradable_symbols]
        
        # Prioritize major pairs
        major_pairs = ['EURUSD#', 'USDJPY#', 'GBPUSD#', 'USDCHF#', 'AUDUSD#', 'USDCAD#', 'NZDUSD#']
        minor_pairs = ['EURJPY#', 'GBPJPY#', 'EURGBP#', 'EURAUD#', 'EURCAD#', 'GBPAUD#', 'GBPCAD#']
        
        # Build final list with majors first
        final_symbols = []
        
        # Add major pairs that are available
        for symbol in major_pairs:
            if symbol in symbol_names:
                final_symbols.append(symbol)
        
        # Add minor pairs that are available
        for symbol in minor_pairs:
            if symbol in symbol_names and symbol not in final_symbols:
                final_symbols.append(symbol)
        
        # Add any other symbols not yet included
        for symbol in symbol_names:
            if symbol not in final_symbols:
                final_symbols.append(symbol)
        
        print("\nRecommended symbols for trading (ordered by priority):")
        for i, symbol in enumerate(final_symbols[:10], 1):
            print(f"{i}. {symbol}")
        
        return final_symbols
        
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []

def main():
    print("="*60)
    print("MT5 TRADABLE SYMBOLS CHECK")
    print("="*60)
    
    symbols = get_tradable_symbols()
    
    if symbols:
        print("\nPython list format for live_trading_engine.py:")
        print(f"self.symbols = {symbols[:5]}")  # Top 5 symbols
        
        print("\nAll tradable symbols:")
        print(f"all_symbols = {symbols}")
    else:
        print("\nNo tradable symbols found!")
        print("Using default: ['EURUSD#', 'USDJPY#', 'GBPUSD#']")

if __name__ == "__main__":
    main()