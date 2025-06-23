#!/usr/bin/env python3
"""
Test harmonic pattern fix
"""

import pandas as pd
from components.indicators import QuantumUltraIntelligentIndicators
from components.mt5_api_client import MT5APIClient
from components.market_data import MarketData
from components.trading_config import CONFIG

# Initialize components
api_client = MT5APIClient(CONFIG["API_BASE"])
market_data = MarketData(api_client)
indicators = QuantumUltraIntelligentIndicators()

# Get some real market data
symbols = api_client.discover_symbols()
if symbols:
    test_symbol = symbols[0]
    print(f"Testing with {test_symbol}")
    
    df = market_data.get_market_data(test_symbol, count=100)
    if df is not None and len(df) >= 100:
        current_price = market_data.get_current_price(test_symbol)
        if current_price:
            print(f"Data: {len(df)} candles, Current price: {current_price}")
            
            # Test harmonic patterns specifically
            try:
                patterns = indicators._detect_harmonic_patterns(df, current_price)
                print(f"✅ Harmonic pattern detection successful! Found {len(patterns)} patterns")
            except Exception as e:
                print(f"❌ Harmonic pattern error: {e}")
            
            # Test full pattern detection
            try:
                all_patterns = indicators._detect_chart_patterns(df, current_price)
                print(f"✅ Full pattern detection successful! Found {len(all_patterns)} patterns")
            except Exception as e:
                print(f"❌ Full pattern detection error: {e}")
        else:
            print("No current price available")
    else:
        print("Insufficient market data")
else:
    print("No symbols available")