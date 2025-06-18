#!/usr/bin/env python3
"""
Test script to demonstrate parallel trading capabilities
"""

import time
from live_trading_engine import LiveTradingEngine

def test_parallel_performance():
    """Test and display parallel trading performance"""
    print("="*60)
    print("PARALLEL TRADING ENGINE TEST")
    print("="*60)
    
    # Initialize engine
    engine = LiveTradingEngine(use_scalping=True)
    
    print(f"\nConfiguration:")
    print(f"- Total symbols to monitor: {len(engine.symbols)}")
    print(f"- Max concurrent trades: {engine.max_concurrent_trades}")
    print(f"- Max trades per symbol: {engine.max_trades_per_symbol}")
    print(f"- Thread pool workers: 10")
    print(f"- Analysis interval: {engine.analysis_interval} seconds")
    
    print(f"\nSymbols being monitored:")
    for i, symbol in enumerate(engine.symbols, 1):
        print(f"  {i:2d}. {symbol}", end="")
        if i % 4 == 0:
            print()
    print()
    
    print("\nPerformance comparison:")
    print("-"*40)
    
    # Simulate sequential analysis time
    seq_time = len(engine.symbols) * 0.5  # Assume 0.5s per symbol
    print(f"Sequential analysis (estimated): {seq_time:.1f} seconds")
    
    # Test parallel analysis
    print("Parallel analysis (actual): ", end="", flush=True)
    start = time.time()
    engine.analyze_symbols_parallel()
    parallel_time = time.time() - start
    print(f"{parallel_time:.1f} seconds")
    
    speedup = seq_time / parallel_time if parallel_time > 0 else 1
    print(f"\nSpeedup factor: {speedup:.1f}x faster")
    print(f"Symbols analyzed per second: {len(engine.symbols)/parallel_time:.1f}")
    
    # Benefits summary
    print("\n" + "="*60)
    print("BENEFITS OF PARALLEL TRADING:")
    print("="*60)
    print("1. Monitor all 28 forex pairs simultaneously")
    print("2. Never miss trading opportunities")
    print("3. Diversified risk across multiple symbols")
    print("4. Optimal resource utilization")
    print("5. Real-time analysis of entire market")
    print("="*60)
    
    # Cleanup
    engine.stop_trading()

if __name__ == "__main__":
    test_parallel_performance()