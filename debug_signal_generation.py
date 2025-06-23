#!/usr/bin/env python3
"""Debug script to test signal generation flow"""

import logging
import sys
import pandas as pd
from datetime import datetime

# Add components to path
sys.path.append('.')

from components.mt5_api_client import MT5APIClient
from components.market_data import MarketData
from components.trading_strategy import TradingStrategy
from components.indicators import QuantumUltraIntelligentIndicators
from components.trading_config import CONFIG

# Configure logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_signal_generation():
    print("=== Testing Signal Generation Flow ===\n")
    
    # 1. Initialize components
    api_client = MT5APIClient(CONFIG["API_BASE"])
    market_data = MarketData(api_client)
    strategy = TradingStrategy()
    
    # 2. Check connection
    print("1. Testing API connection...")
    if not api_client.check_connection():
        print("✗ API connection failed!")
        return
    print("✓ API connected\n")
    
    # 3. Get account info
    print("2. Getting account info...")
    account = api_client.get_account_info()
    if account:
        print(f"✓ Balance: ¥{account.get('balance', 0):,.0f}")
        print(f"  Equity: ¥{account.get('equity', 0):,.0f}\n")
    else:
        print("✗ Failed to get account info\n")
    
    # 4. Discover symbols
    print("3. Discovering symbols...")
    symbols = api_client.discover_symbols()
    print(f"✓ Found {len(symbols)} symbols")
    print(f"  First 5: {symbols[:5]}\n")
    
    # 5. Test with first available symbol
    if not symbols:
        print("✗ No symbols available!")
        return
        
    test_symbol = symbols[0]  # Use first symbol
    print(f"4. Testing with symbol: {test_symbol}")
    
    # 6. Get symbol info
    print("   Getting symbol info...")
    symbol_info = api_client.get_symbol_info(test_symbol)
    if symbol_info:
        print(f"   ✓ Symbol: {symbol_info.get('name')}")
        print(f"     Digits: {symbol_info.get('digits')}")
        print(f"     Min volume: {symbol_info.get('volume_min')}")
        print(f"     Bid: {symbol_info.get('bid', 0)}")
        print(f"     Ask: {symbol_info.get('ask', 0)}\n")
    else:
        print("   ✗ Failed to get symbol info\n")
    
    # 7. Get market data
    print("5. Getting market data...")
    df = market_data.get_market_data(test_symbol)
    if df is not None and len(df) > 0:
        print(f"   ✓ Got {len(df)} candles")
        print(f"     Latest: {df.index[-1]}")
        print(f"     Close: {df['close'].iloc[-1]}\n")
    else:
        print("   ✗ Failed to get market data\n")
        return
    
    # 8. Get current price
    print("6. Getting current price...")
    current_price = market_data.get_current_price(test_symbol)
    if current_price:
        print(f"   ✓ Current price: {current_price}\n")
    else:
        print("   ✗ Failed to get current price\n")
        return
    
    # 9. Check spread
    print("7. Checking spread...")
    spread_ok, spread = market_data.check_spread(test_symbol)
    print(f"   Spread: {spread} pips")
    print(f"   Spread OK: {spread_ok}\n")
    
    # 10. Generate signal
    print("8. Generating trading signal...")
    print(f"   MIN_CONFIDENCE required: {CONFIG['MIN_CONFIDENCE']}")
    
    try:
        signal = strategy.analyze_ultra(test_symbol, df, current_price)
        
        if signal:
            print(f"\n   ✓ SIGNAL GENERATED!")
            print(f"     Type: {signal.type}")
            print(f"     Confidence: {signal.confidence:.2%}")
            print(f"     Entry: {signal.entry}")
            print(f"     SL: {signal.sl}")
            print(f"     TP: {signal.tp}")
            print(f"     Reason: {signal.reason}")
        else:
            print("\n   ✗ No signal generated")
            print("   Possible reasons:")
            print("   - Confidence below threshold")
            print("   - No clear market direction")
            print("   - Spread too high")
            print("   - Risk parameters not met")
            
            # Try forced signal to see if strategy works at all
            print("\n9. Testing forced signal generation...")
            forced_signal = strategy.force_trade_signal(test_symbol, df, current_price)
            if forced_signal:
                print(f"   ✓ FORCED SIGNAL GENERATED!")
                print(f"     Type: {forced_signal.type}")
                print(f"     Confidence: {forced_signal.confidence:.2%}")
            else:
                print("   ✗ Even forced signal failed")
                
    except Exception as e:
        print(f"\n   ✗ Error generating signal: {e}")
        import traceback
        traceback.print_exc()
    
    # 11. Test multiple symbols
    print("\n10. Testing multiple symbols for signals...")
    signals_found = 0
    for symbol in symbols[:10]:  # Test first 10 symbols
        try:
            df = market_data.get_market_data(symbol)
            if df is None or len(df) < 50:
                continue
                
            current_price = market_data.get_current_price(symbol)
            if not current_price:
                continue
                
            signal = strategy.analyze_ultra(symbol, df, current_price)
            if signal:
                signals_found += 1
                print(f"   ✓ Signal for {symbol}: {signal.type} @ {signal.confidence:.2%}")
        except Exception as e:
            print(f"   ✗ Error with {symbol}: {e}")
    
    print(f"\n   Total signals found: {signals_found}/10")

if __name__ == "__main__":
    test_signal_generation()