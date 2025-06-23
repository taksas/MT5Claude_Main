#!/usr/bin/env python3
"""
Comprehensive System Debug Script
Tests all components of the MT5 Trading System
"""

import sys
import time
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SystemDebug')

def test_imports():
    """Test all imports"""
    print("\n=== Testing Imports ===")
    imports_ok = True
    
    try:
        print("✓ Importing main...")
        from main import UltraTradingSystem
        print("  ✓ UltraTradingSystem imported")
    except Exception as e:
        print(f"  ✗ Failed to import main: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing engine_core...")
        from components.engine_core import UltraTradingEngine
        print("  ✓ UltraTradingEngine imported")
    except Exception as e:
        print(f"  ✗ Failed to import engine_core: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing visualizer...")
        from components.visualizer import TradingVisualizer
        print("  ✓ TradingVisualizer imported")
    except Exception as e:
        print(f"  ✗ Failed to import visualizer: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing mt5_api_client...")
        from components.mt5_api_client import MT5APIClient
        print("  ✓ MT5APIClient imported")
    except Exception as e:
        print(f"  ✗ Failed to import mt5_api_client: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing trading_strategy...")
        from components.trading_strategy import TradingStrategy
        print("  ✓ TradingStrategy imported")
    except Exception as e:
        print(f"  ✗ Failed to import trading_strategy: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing indicators...")
        from components.indicators import QuantumUltraIntelligentIndicators
        print("  ✓ QuantumUltraIntelligentIndicators imported")
    except Exception as e:
        print(f"  ✗ Failed to import indicators: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing trading_config...")
        from components.trading_config import CONFIG, HIGH_PROFIT_SYMBOLS
        print("  ✓ CONFIG and HIGH_PROFIT_SYMBOLS imported")
    except Exception as e:
        print(f"  ✗ Failed to import trading_config: {e}")
        imports_ok = False
    
    try:
        print("✓ Importing trading_models...")
        from components.trading_models import Trade, Signal, SignalType
        print("  ✓ Trading models imported")
    except Exception as e:
        print(f"  ✗ Failed to import trading_models: {e}")
        imports_ok = False
    
    return imports_ok

def test_api_connection():
    """Test API connection"""
    print("\n=== Testing API Connection ===")
    try:
        from components.mt5_api_client import MT5APIClient
        from components.trading_config import CONFIG
        
        api_client = MT5APIClient(CONFIG["API_BASE"])
        print(f"✓ API Base URL: {CONFIG['API_BASE']}")
        
        # Test connection
        connected = api_client.check_connection()
        if connected:
            print("  ✓ API connection successful")
        else:
            print("  ✗ API connection failed")
            return False
        
        # Test account info
        account_info = api_client.get_account_info()
        if account_info:
            print(f"  ✓ Account Balance: ¥{account_info.get('balance', 0):,.0f}")
            print(f"  ✓ Account Equity: ¥{account_info.get('equity', 0):,.0f}")
            print(f"  ✓ Account Currency: {account_info.get('currency', 'N/A')}")
        else:
            print("  ✗ Failed to get account info")
            return False
        
        # Test symbol discovery
        symbols = api_client.discover_symbols()
        if symbols:
            print(f"  ✓ Found {len(symbols)} symbols")
            print(f"    Sample symbols: {', '.join(symbols[:5])}")
        else:
            print("  ✗ Failed to discover symbols")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        traceback.print_exc()
        return False

def test_engine_initialization():
    """Test engine initialization"""
    print("\n=== Testing Engine Initialization ===")
    try:
        from components.engine_core import UltraTradingEngine
        
        engine = UltraTradingEngine()
        print("✓ Engine created successfully")
        
        # Test component initialization
        print("  ✓ API Client:", type(engine.api_client).__name__)
        print("  ✓ Market Data:", type(engine.market_data).__name__)
        print("  ✓ Strategy:", type(engine.strategy).__name__)
        print("  ✓ Risk Manager:", type(engine.risk_manager).__name__)
        print("  ✓ Order Manager:", type(engine.order_manager).__name__)
        
        # Test symbol discovery
        engine.tradable_symbols = engine._discover_symbols()
        if engine.tradable_symbols:
            print(f"  ✓ Discovered {len(engine.tradable_symbols)} tradable symbols")
            print(f"    Top symbols: {', '.join(engine.tradable_symbols[:10])}")
        else:
            print("  ✗ No tradable symbols found")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Engine initialization failed: {e}")
        traceback.print_exc()
        return False

def test_visualizer():
    """Test visualizer component"""
    print("\n=== Testing Visualizer ===")
    try:
        from components.visualizer import TradingVisualizer
        
        visualizer = TradingVisualizer()
        print("✓ Visualizer created successfully")
        
        # Test data fetching
        account = visualizer.get_account_info()
        if account:
            print("  ✓ Can fetch account info")
        else:
            print("  ✗ Failed to fetch account info")
        
        positions = visualizer.get_positions()
        print(f"  ✓ Found {len(positions)} open positions")
        
        history = visualizer.get_history()
        print(f"  ✓ Found {len(history)} historical trades today")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Visualizer test failed: {e}")
        traceback.print_exc()
        return False

def test_strategy():
    """Test strategy component"""
    print("\n=== Testing Strategy ===")
    try:
        from components.trading_strategy import TradingStrategy
        from components.mt5_api_client import MT5APIClient
        from components.market_data import MarketData
        from components.trading_config import CONFIG
        import pandas as pd
        
        # Initialize components
        api_client = MT5APIClient(CONFIG["API_BASE"])
        market_data = MarketData(api_client)
        strategy = TradingStrategy()
        
        print("✓ Strategy initialized")
        
        # Get a test symbol
        symbols = api_client.discover_symbols()
        if not symbols:
            print("  ✗ No symbols available for testing")
            return False
        
        test_symbol = symbols[0]
        print(f"  Testing with symbol: {test_symbol}")
        
        # Get market data
        df = market_data.get_market_data(test_symbol)
        if df is None or len(df) == 0:
            print("  ✗ Failed to get market data")
            return False
        
        print(f"  ✓ Got {len(df)} candles of market data")
        
        # Get current price
        current_price = market_data.get_current_price(test_symbol)
        if not current_price:
            print("  ✗ Failed to get current price")
            return False
        
        print(f"  ✓ Current price: {current_price}")
        
        # Test signal generation
        try:
            signal = strategy.analyze_ultra(test_symbol, df, current_price)
            if signal:
                print(f"  ✓ Generated signal: {signal.type.value} with confidence {signal.confidence:.1%}")
            else:
                print("  ✓ No signal generated (normal behavior)")
        except Exception as e:
            print(f"  ✗ Signal generation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_full_system():
    """Test full system integration"""
    print("\n=== Testing Full System Integration ===")
    try:
        from main import UltraTradingSystem
        
        # Test system creation
        system = UltraTradingSystem(mode='engine', visualize=False)
        print("✓ System created successfully")
        
        # Test signal handler
        system.signal_handler(None, None)
        print("  ✓ Signal handler works")
        system.shutdown = False  # Reset
        
        return True
        
    except Exception as e:
        print(f"  ✗ System integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("=" * 80)
    print("MT5 TRADING SYSTEM - COMPREHENSIVE DEBUG")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
        print("\n⚠️  Import tests failed - fixing imports is critical")
    
    if not test_api_connection():
        all_tests_passed = False
        print("\n⚠️  API connection failed - check if MT5 Bridge API is running")
    
    if not test_engine_initialization():
        all_tests_passed = False
        print("\n⚠️  Engine initialization failed")
    
    if not test_visualizer():
        all_tests_passed = False
        print("\n⚠️  Visualizer tests failed")
    
    if not test_strategy():
        all_tests_passed = False
        print("\n⚠️  Strategy tests failed")
    
    if not test_full_system():
        all_tests_passed = False
        print("\n⚠️  System integration tests failed")
    
    # Summary
    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - System is ready to run!")
        print("\nTo start trading:")
        print("  python3 main.py                  # Run engine only")
        print("  python3 main.py --visualize      # Run engine with visualizer")
        print("  python3 main.py --mode both      # Run engine and visualizer separately")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        print("\nCommon issues:")
        print("  1. MT5 Bridge API not running (check http://172.28.144.1:8000/docs)")
        print("  2. Missing dependencies (run: pip install -r requirements.txt)")
        print("  3. Import errors (check file paths and module names)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()