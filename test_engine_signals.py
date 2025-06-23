#!/usr/bin/env python3
"""
Test engine signal generation
"""

import time
import threading
from components.engine_core import UltraTradingEngine
from components.trading_config import CONFIG

print("Testing engine with lowered thresholds...")
print(f"MIN_CONFIDENCE: {CONFIG['MIN_CONFIDENCE']}")
print(f"MIN_STRATEGIES: {CONFIG['MIN_STRATEGIES']}")
print(f"MIN_INDICATORS: {CONFIG['MIN_INDICATORS']}")
print(f"AGGRESSIVE_MODE: {CONFIG['AGGRESSIVE_MODE']}")
print(f"IGNORE_SPREAD: {CONFIG['IGNORE_SPREAD']}")

# Create engine
engine = UltraTradingEngine()

# Override to capture signals
original_execute = engine._execute_signal
signals_found = []

def capture_signal(symbol, signal):
    print(f"\n🎯 SIGNAL CAPTURED for {symbol}!")
    print(f"  Type: {signal.type.value}")
    print(f"  Confidence: {signal.confidence:.1%}")
    print(f"  Quality: {signal.quality:.1%}")
    print(f"  Entry: {signal.entry}")
    print(f"  SL: {signal.sl}")
    print(f"  TP: {signal.tp}")
    print(f"  Reasons: {', '.join(signal.reasons[:3])}")
    signals_found.append((symbol, signal))
    # Don't actually execute
    return

engine._execute_signal = capture_signal

# Also capture analysis results
original_analyze = engine._analyze_symbol
analysis_count = 0

def track_analysis(symbol):
    global analysis_count
    analysis_count += 1
    if analysis_count % 10 == 0:
        print(f".", end="", flush=True)
    return original_analyze(symbol)

engine._analyze_symbol = track_analysis

print("\nRunning engine for 20 seconds...")
print("Progress: ", end="")

# Run in thread
engine.running = True
engine_thread = threading.Thread(target=lambda: [engine.run_once() for _ in range(10)])
engine_thread.start()

# Wait
time.sleep(20)
engine.running = False
engine_thread.join(timeout=5)

print(f"\n\nAnalyzed {analysis_count} symbol checks")
print(f"Found {len(signals_found)} signals")

if not signals_found and CONFIG['AGGRESSIVE_MODE']:
    print("\nNo signals found. Testing force trade...")
    engine._force_trade()
    if signals_found:
        print("✅ Force trade generated a signal!")
    else:
        print("❌ Even force trade didn't generate signals")