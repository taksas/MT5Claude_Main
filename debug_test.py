#!/usr/bin/env python3
"""
Debug test script to identify trading system issues
"""

import logging
import json
from components.mt5_api_client import MT5APIClient
from components.trading_config import CONFIG

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('DebugTest')

def test_api_connection():
    """Test API connection"""
    logger.info("Testing API connection...")
    api_client = MT5APIClient(CONFIG["API_BASE"])
    
    # Test connection
    is_connected = api_client.check_connection()
    logger.info(f"API Connection: {'✓ SUCCESS' if is_connected else '✗ FAILED'}")
    
    if is_connected:
        # Test account info
        account_info = api_client.get_account_info()
        if account_info:
            logger.info(f"Account Balance: ¥{account_info.get('balance', 0):,.0f}")
            logger.info(f"Account Equity: ¥{account_info.get('equity', 0):,.0f}")
        else:
            logger.error("Failed to get account info")
        
        # Test symbols
        try:
            response = api_client.session.get(f"{api_client.base_url}/symbols/")
            if response.status_code == 200:
                symbols = response.json()
                logger.info(f"Available symbols: {len(symbols) if symbols else 0}")
                if symbols and len(symbols) > 0:
                    logger.info(f"First 5 symbols: {list(symbols.keys())[:5]}")
            else:
                logger.warning("Failed to get symbols")
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
    
    return is_connected

def test_market_data():
    """Test market data retrieval"""
    logger.info("\nTesting market data...")
    api_client = MT5APIClient(CONFIG["API_BASE"])
    
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    for symbol in test_symbols:
        try:
            # Get tick data
            tick = api_client.get_tick(symbol)
            if tick:
                logger.info(f"{symbol} - Bid: {tick.get('bid', 0)}, Ask: {tick.get('ask', 0)}")
            else:
                logger.warning(f"{symbol} - No tick data")
            
            # Get price history
            bars = api_client.get_price_history(symbol, "M5", 100)
            if bars:
                logger.info(f"{symbol} - Got {len(bars)} bars of history")
            else:
                logger.warning(f"{symbol} - No price history")
                
        except Exception as e:
            logger.error(f"{symbol} - Error: {e}")

def test_indicators():
    """Test indicator calculations"""
    logger.info("\nTesting indicators...")
    
    try:
        from components.indicators import quantum_indicators
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dates = pd.date_range(end='2024-01-01', periods=100, freq='5min')
        dummy_data = pd.DataFrame({
            'time': dates,
            'open': 1.1000 + np.random.randn(100) * 0.001,
            'high': 1.1005 + np.random.randn(100) * 0.001,
            'low': 1.0995 + np.random.randn(100) * 0.001,
            'close': 1.1000 + np.random.randn(100) * 0.001,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        # Calculate indicators
        result = quantum_indicators.calculate_ultra_indicators(dummy_data, 1.1000)
        
        logger.info(f"Indicator calculation: ✓ SUCCESS")
        logger.info(f"Regime: {result.get('regime', {}).get('trend', 'unknown')}")
        logger.info(f"Quantum state coherence: {result.get('quantum_state', {}).get('coherence', 0)}")
        logger.info(f"Composite signal: {result.get('composite_signal', {}).get('signal', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Indicator test failed: {e}")
        import traceback
        traceback.print_exc()

def test_trading_strategy():
    """Test trading strategy"""
    logger.info("\nTesting trading strategy...")
    
    try:
        from components.trading_strategy import TradingStrategy
        import pandas as pd
        import numpy as np
        
        strategy = TradingStrategy()
        
        # Create dummy data
        dates = pd.date_range(end='2024-01-01', periods=100, freq='5min')
        dummy_data = pd.DataFrame({
            'time': dates,
            'open': 1.1000 + np.random.randn(100) * 0.001,
            'high': 1.1005 + np.random.randn(100) * 0.001,
            'low': 1.0995 + np.random.randn(100) * 0.001,
            'close': 1.1000 + np.random.randn(100) * 0.001,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        # Test signal generation
        signal = strategy.analyze_ultra("EURUSD", dummy_data, 1.1000)
        
        if signal:
            logger.info(f"Signal generated: ✓ {signal.type}")
            logger.info(f"Confidence: {signal.confidence:.2%}")
            logger.info(f"Reason: {signal.reason}")
        else:
            logger.info("No signal generated (normal behavior)")
            
    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debug tests"""
    logger.info("=" * 60)
    logger.info("ULTRA TRADING SYSTEM DEBUG TEST")
    logger.info("=" * 60)
    
    # Test 1: API Connection
    if not test_api_connection():
        logger.error("\n⚠️  API CONNECTION FAILED - Check if MT5 Bridge is running")
        logger.error("Expected API endpoint: http://172.28.144.1:8000")
        return
    
    # Test 2: Market Data
    test_market_data()
    
    # Test 3: Indicators
    test_indicators()
    
    # Test 4: Trading Strategy
    test_trading_strategy()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEBUG TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()