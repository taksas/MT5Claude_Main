#!/usr/bin/env python3
"""Simplified test to find why no signals are generated"""

import sys
import time
import logging

sys.path.append('.')

from components.engine_core import UltraTradingEngine

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug for specific modules
logging.getLogger('UltraTradingEngine').setLevel(logging.DEBUG)
logging.getLogger('TradingStrategy').setLevel(logging.DEBUG)
logging.getLogger('MarketData').setLevel(logging.DEBUG)

def test_engine():
    print("=== Testing Engine Signal Generation ===\n")
    
    # Create engine
    engine = UltraTradingEngine()
    
    # Start engine but only run once
    print("Starting engine...")
    if not engine.api_client.check_connection():
        print("✗ Cannot connect to API")
        return
    
    engine.balance = engine.api_client.get_balance()
    if not engine.balance:
        print("✗ Cannot get balance")
        return
    
    # Discover symbols
    engine.tradable_symbols = engine._discover_symbols()
    if not engine.tradable_symbols:
        print("✗ No tradable symbols found")
        return
    
    print(f"✓ Engine initialized")
    print(f"  Balance: ¥{engine.balance:,.0f}")
    print(f"  Symbols: {len(engine.tradable_symbols)}")
    print(f"  First 5: {engine.tradable_symbols[:5]}\n")
    
    engine.running = True
    
    # Run the main loop a few times
    print("Running trading loop 5 times...\n")
    for i in range(5):
        print(f"--- Iteration {i+1} ---")
        engine.run_once()
        print(f"Active trades: {len(engine.active_trades)}")
        print(f"Daily trades: {engine.daily_trades}")
        print(f"Force trade attempts: {engine.force_trade_attempts}")
        
        # Check if should force trade
        time_since_last = time.time() - engine.last_global_trade_time
        print(f"Time since last trade: {time_since_last:.1f}s")
        print(f"Force trade threshold: {engine.config['FORCE_TRADE_INTERVAL']}s")
        print()
        
        time.sleep(1)  # Small delay between iterations
    
    print("\n=== Summary ===")
    print(f"Total trades opened: {engine.daily_trades}")
    print(f"Force trade attempts: {engine.force_trade_attempts}")
    print(f"Active positions: {len(engine.active_trades)}")
    
    # Check last signals
    if engine.last_signals:
        print("\nLast signals:")
        for symbol, sig_data in list(engine.last_signals.items())[:5]:
            signal = sig_data['signal']
            print(f"  {symbol}: {signal.type} @ {signal.confidence:.2%}")

if __name__ == "__main__":
    test_engine()