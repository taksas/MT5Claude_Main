#!/usr/bin/env python3
"""
Simple test script to verify the MT5 trader can connect and fetch data
"""

import asyncio
import sys
import json
from automated_trader import MT5TradingBot

async def test_connection():
    """Test basic MT5 trader functionality"""
    print("Testing MT5 Trading Bot connection...")
    
    async with MT5TradingBot() as bot:
        # Test API connection
        print("1. Testing API connection...")
        if not await bot.check_connection():
            print("❌ API connection failed")
            return False
        print("✅ API connection successful")
        
        # Test MT5 status
        print("2. Testing MT5 status...")
        if not await bot.check_mt5_status():
            print("❌ MT5 status failed")
            return False
        print("✅ MT5 status successful")
        
        # Test account info
        print("3. Testing account info...")
        account_info = await bot.get_account_info()
        if account_info:
            print(f"✅ Account Balance: {account_info.get('balance')} {account_info.get('currency')}")
            print(f"   Free Margin: {account_info.get('margin_free')}")
        else:
            print("❌ Failed to get account info")
            return False
        
        # Test symbols
        print("4. Testing symbol list...")
        symbols = await bot.get_tradable_symbols()
        print(f"✅ Found {len(symbols)} verified working symbols")
        
        # Test individual symbol info
        print("5. Testing individual symbol info...")
        test_symbols = symbols[:3]  # Test first 3 symbols
        for symbol in test_symbols:
            symbol_info = await bot.get_symbol_info(symbol)
            if symbol_info:
                print(f"✅ {symbol}: {symbol_info.get('description', 'N/A')} - Spread: {symbol_info.get('spread', 'N/A')}")
            else:
                print(f"❌ Failed to get info for {symbol}")
                return False
        
        # Test historical data
        print("6. Testing historical data...")
        hist_data = await bot.get_historical_data(symbols[0], "M5", 10)
        if hist_data and len(hist_data) > 0:
            print(f"✅ Retrieved {len(hist_data)} historical bars for {symbols[0]}")
            latest = hist_data[-1]
            print(f"   Latest: Close={latest.get('close')}, Time={latest.get('time')}")
        else:
            print(f"❌ Failed to get historical data for {symbols[0]}")
            return False
        
        # Test technical indicators calculation
        print("7. Testing technical indicators...")
        indicators = bot.calculate_technical_indicators(hist_data)
        if indicators:
            print(f"✅ Calculated indicators for {symbols[0]}:")
            print(f"   Current Price: {indicators.get('current_price')}")
            print(f"   RSI: {indicators.get('rsi')}")
            print(f"   SMA5: {indicators.get('sma_5')}")
        else:
            print("❌ Failed to calculate indicators")
            return False
        
        print("\n🎉 All tests passed! MT5 Trading Bot is ready for operation.")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)