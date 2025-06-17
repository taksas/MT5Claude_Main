#!/usr/bin/env python3
"""
MT5 Symbol Discovery Tool
Discovers available trading symbols and validates them for automated trading
"""

import json
from datetime import datetime
from mt5_client import MT5Client
import logging

def discover_and_validate_symbols():
    """Comprehensive symbol discovery and validation"""
    
    print("🔍 MT5 SYMBOL DISCOVERY AND VALIDATION")
    print("="*60)
    
    client = MT5Client()
    
    # Test connection first
    if not client.ping():
        print("❌ MT5 API server not responding")
        print("Please ensure MT5 Bridge API is running on Windows host")
        return None
    
    print("✅ MT5 API connection successful")
    
    try:
        # Get account info
        account_info = client.get_account_info()
        print(f"📊 Account: {account_info.get('login', 'N/A')}")
        print(f"💰 Balance: {account_info.get('balance', 0):.2f} {account_info.get('currency', 'USD')}")
        print()
        
        # Discover all symbols
        all_symbols = client.discover_available_symbols()
        print(f"📈 Total Available Symbols: {len(all_symbols)}")
        
        # Categorize symbols
        hash_symbols = [s for s in all_symbols if s.endswith('#')]
        forex_symbols = client.get_forex_symbols()
        
        print(f"🎯 Hash Symbols (#): {len(hash_symbols)}")
        print(f"💱 Forex Symbols: {len(forex_symbols)}")
        print()
        
        # Validate symbols for trading
        validated_symbols = {}
        
        print("🔎 VALIDATING SYMBOLS FOR TRADING:")
        print("-" * 40)
        
        # Prioritize hash symbols (as per requirements)
        symbols_to_check = hash_symbols if hash_symbols else forex_symbols[:10]
        
        for symbol in symbols_to_check:
            try:
                symbol_info = client.get_symbol_info(symbol)
                
                validation = {
                    'symbol': symbol,
                    'name': symbol_info.get('description', 'N/A'),
                    'trade_mode': symbol_info.get('trade_mode_description', 'UNKNOWN'),
                    'point': symbol_info.get('point', 0.0001),
                    'min_lot': symbol_info.get('volume_min', 0.01),
                    'max_lot': symbol_info.get('volume_max', 100.0),
                    'lot_step': symbol_info.get('volume_step', 0.01),
                    'spread': symbol_info.get('spread', 0),
                    'margin_required': symbol_info.get('margin_required', 0),
                    'tradable': symbol_info.get('trade_mode_description') == 'FULL'
                }
                
                validated_symbols[symbol] = validation
                
                # Display validation result
                status = "✅" if validation['tradable'] else "❌"
                print(f"{status} {symbol:<12} | {validation['name']:<25} | Mode: {validation['trade_mode']}")
                print(f"   Point: {validation['point']:<8} | Lot: {validation['min_lot']}-{validation['max_lot']} | Spread: {validation['spread']}")
                
            except Exception as e:
                print(f"❌ {symbol:<12} | Error: {e}")
        
        # Select best symbols for trading
        tradable_symbols = {k: v for k, v in validated_symbols.items() if v['tradable']}
        
        print()
        print("🎯 RECOMMENDED TRADING SYMBOLS:")
        print("-" * 40)
        
        if tradable_symbols:
            # Prioritize major forex pairs
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
            recommended = []
            
            for symbol, info in tradable_symbols.items():
                # Check if it's a major pair
                is_major = any(pair in symbol.upper() for pair in major_pairs)
                
                # Good trading conditions
                good_conditions = (
                    info['min_lot'] <= 0.01 and  # Can trade 0.01 lots
                    info['spread'] < 50 and      # Reasonable spread
                    info['point'] > 0            # Valid point value
                )
                
                if is_major and good_conditions:
                    recommended.append(symbol)
                    print(f"🚀 {symbol} - EXCELLENT (Major pair, good conditions)")
                elif good_conditions:
                    recommended.append(symbol)
                    print(f"✅ {symbol} - GOOD (Tradable with good conditions)")
                elif is_major:
                    print(f"⚠️  {symbol} - MAJOR PAIR (Check trading conditions)")
                else:
                    print(f"📊 {symbol} - AVAILABLE (Monitor performance)")
        
        else:
            print("❌ No fully tradable symbols found")
            print("Please check:")
            print("- MT5 account permissions")
            print("- Symbol availability in your broker")
            print("- Account type restrictions")
        
        # Save results
        results = {
            'discovery_timestamp': datetime.utcnow().isoformat(),
            'account_info': {
                'login': account_info.get('login'),
                'balance': account_info.get('balance'),
                'currency': account_info.get('currency')
            },
            'total_symbols': len(all_symbols),
            'hash_symbols_count': len(hash_symbols),
            'forex_symbols_count': len(forex_symbols),
            'validated_symbols': validated_symbols,
            'recommended_symbols': recommended if tradable_symbols else [],
            'hash_symbols_list': hash_symbols,
            'forex_symbols_list': forex_symbols
        }
        
        with open('symbol_discovery.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print()
        print("="*60)
        print(f"📁 Results saved to: symbol_discovery.json")
        print(f"🎯 Recommended symbols: {len(recommended) if tradable_symbols else 0}")
        print(f"📊 Total validated: {len(validated_symbols)}")
        
        if recommended:
            print(f"🚀 Ready for trading with: {', '.join(recommended[:5])}")
            return recommended
        else:
            print("⚠️  No recommended symbols - check account settings")
            return []
        
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return None

def main():
    """Main symbol discovery function"""
    try:
        symbols = discover_and_validate_symbols()
        
        if symbols:
            print(f"\n✅ Symbol discovery complete!")
            print(f"🎯 Found {len(symbols)} symbols ready for automated trading")
        else:
            print(f"\n❌ Symbol discovery failed or no symbols available")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()