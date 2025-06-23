#!/usr/bin/env python3
"""
Test engine directly
"""

from components.engine_core import UltraTradingEngine

print("Creating engine...")
engine = UltraTradingEngine()

# Check initialization
print(f"API connected: {engine.api_client.check_connection()}")
print(f"Balance: ¥{engine.balance}")
print(f"Tradable symbols: {len(engine.tradable_symbols)}")
print(f"First 5 symbols: {engine.tradable_symbols[:5]}")

# Test trading hours
print(f"\nTrading hours check: {engine._is_trading_hours()}")

# Manually run one iteration
print("\nRunning one iteration...")
try:
    engine.running = True
    engine.run_once()
    print("✓ Iteration completed")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test analyze symbol directly
if engine.tradable_symbols:
    print(f"\nTesting direct analysis of {engine.tradable_symbols[0]}...")
    try:
        signal = engine._analyze_symbol(engine.tradable_symbols[0])
        if signal:
            print(f"✅ Signal found! Type: {signal.type.value}, Confidence: {signal.confidence:.1%}")
        else:
            print("⚪ No signal from analysis")
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        import traceback
        traceback.print_exc()