#!/usr/bin/env python3
"""
System Health Check Script
Performs comprehensive testing of all components
"""

import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SystemHealthCheck')

def check_imports():
    """Check all required imports"""
    logger.info("Checking imports...")
    try:
        import pandas
        import numpy
        import requests
        import scipy
        from components.mt5_api_client import MT5APIClient
        from components.trading_strategy import TradingStrategy
        from components.indicators import quantum_indicators
        from components.engine_core import UltraTradingEngine
        from components.visualizer import TradingVisualizer
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def check_api_connection():
    """Check API connectivity"""
    logger.info("Checking API connection...")
    try:
        from components.mt5_api_client import MT5APIClient
        from components.trading_config import CONFIG
        
        client = MT5APIClient(CONFIG["API_BASE"])
        if client.check_connection():
            logger.info("✅ API connection successful")
            
            # Check account info
            balance = client.get_balance()
            if balance:
                logger.info(f"✅ Account balance: ¥{balance:,.0f}")
            else:
                logger.warning("⚠️ Could not retrieve balance")
            
            return True
        else:
            logger.error("❌ API connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ API check error: {e}")
        return False

def check_strategy():
    """Check trading strategy initialization"""
    logger.info("Checking trading strategy...")
    try:
        from components.trading_strategy import TradingStrategy
        import pandas as pd
        import numpy as np
        
        strategy = TradingStrategy()
        logger.info("✅ Strategy initialized")
        
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        test_df = pd.DataFrame({
            'time': dates,
            'open': 1.1000 + np.cumsum(np.random.randn(100) * 0.0001),
            'high': 1.1005 + np.cumsum(np.random.randn(100) * 0.0001),
            'low': 1.0995 + np.cumsum(np.random.randn(100) * 0.0001),
            'close': 1.1000 + np.cumsum(np.random.randn(100) * 0.0001),
            'volume': np.random.randint(100, 1000, 100)
        })
        
        # Ensure high/low are correct
        test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
        test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)
        
        # Test signal generation
        signal = strategy.analyze_ultra("EURUSD", test_df, test_df['close'].iloc[-1])
        
        if signal:
            logger.info(f"✅ Signal generated: {signal.type} (confidence: {signal.confidence:.2%})")
        else:
            logger.info("✅ No signal generated (normal)")
        
        logger.info(f"✅ Consciousness level: {strategy.consciousness_level}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Strategy check error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_indicators():
    """Check indicator calculations"""
    logger.info("Checking indicators...")
    try:
        from components.indicators import quantum_indicators
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        test_df = pd.DataFrame({
            'time': dates,
            'open': 1.1000 + np.random.randn(100) * 0.001,
            'high': 1.1005 + np.random.randn(100) * 0.001,
            'low': 1.0995 + np.random.randn(100) * 0.001,
            'close': 1.1000 + np.random.randn(100) * 0.001,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        # Calculate indicators
        result = quantum_indicators.calculate_ultra_indicators(test_df, 1.1000)
        
        logger.info(f"✅ Indicators calculated")
        logger.info(f"✅ Regime: {result.get('regime', {}).get('trend', 'unknown')}")
        logger.info(f"✅ Quantum coherence: {result.get('quantum_state', {}).get('coherence', 0)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Indicator check error: {e}")
        return False

def check_engine():
    """Check engine initialization"""
    logger.info("Checking engine...")
    try:
        from components.engine_core import UltraTradingEngine
        
        engine = UltraTradingEngine()
        logger.info("✅ Engine initialized")
        
        # Check components
        if hasattr(engine, 'api_client'):
            logger.info("✅ API client present")
        if hasattr(engine, 'strategy'):
            logger.info("✅ Strategy present")
        if hasattr(engine, 'risk_manager'):
            logger.info("✅ Risk manager present")
        
        return True
    except Exception as e:
        logger.error(f"❌ Engine check error: {e}")
        return False

def check_visualizer():
    """Check visualizer initialization"""
    logger.info("Checking visualizer...")
    try:
        from components.visualizer import TradingVisualizer
        
        viz = TradingVisualizer()
        logger.info("✅ Visualizer initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ Visualizer check error: {e}")
        return False

def main():
    """Run all health checks"""
    print("=" * 80)
    print("ULTRA TRADING SYSTEM HEALTH CHECK")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    checks = [
        ("Imports", check_imports),
        ("API Connection", check_api_connection),
        ("Strategy", check_strategy),
        ("Indicators", check_indicators),
        ("Engine", check_engine),
        ("Visualizer", check_visualizer)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            result = check_func()
            results.append((name, result))
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Unexpected error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("✅ ALL SYSTEMS OPERATIONAL")
    else:
        print("❌ SOME SYSTEMS NEED ATTENTION")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    sys.exit(0 if main() else 1)