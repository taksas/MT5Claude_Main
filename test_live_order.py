#!/usr/bin/env python3
"""Test live order placement through the trading engine"""

from live_trading_engine import LiveTradingEngine
import logging

logging.basicConfig(level=logging.INFO)

def test_single_order():
    """Test placing a single order through the live trading engine"""
    engine = LiveTradingEngine()
    
    # Check API connection
    if not engine.check_api_connection():
        print("❌ API connection failed")
        return
    
    print("✅ API connected")
    
    # Get account info
    account = engine.get_account_info()
    if account:
        print(f"📊 Account balance: ${account.get('balance', 0):.2f}")
    
    # Test with a specific symbol
    symbol = "EURUSD#"
    print(f"\n🧪 Testing order placement for {symbol}")
    
    # Get market data
    market_data = engine.get_market_data(symbol)
    if not market_data:
        print(f"❌ Could not get market data for {symbol}")
        return
    
    print(f"✅ Got market data: {len(market_data)} candles")
    
    # Use the engine's analyze_and_trade method
    print(f"\n🔍 Analyzing {symbol}...")
    engine.analyze_and_trade(symbol)
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_single_order()