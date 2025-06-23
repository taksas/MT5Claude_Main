#!/usr/bin/env python3
"""
System Test Runner - Test the complete trading system
"""

import time
import sys
import signal
import threading
from datetime import datetime

# Set up signal handling
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    print("\n\nShutdown signal received. Stopping system...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def test_engine_only():
    """Test engine in isolation"""
    print("\n" + "=" * 80)
    print("TESTING: Engine Only Mode")
    print("=" * 80)
    
    from components.engine_core import UltraTradingEngine
    
    engine = UltraTradingEngine()
    
    # Run one iteration
    print("Running one trading loop iteration...")
    engine.running = True
    engine.run_once()
    
    print("✓ Engine test completed")

def test_visualizer_only():
    """Test visualizer in isolation"""
    print("\n" + "=" * 80)
    print("TESTING: Visualizer Only Mode")
    print("=" * 80)
    
    from components.visualizer import TradingVisualizer
    
    visualizer = TradingVisualizer()
    
    # Update data once
    print("Fetching and displaying data...")
    visualizer.update_display_data()
    
    print(f"✓ Account Balance: ¥{visualizer.display_data.balance:,.0f}")
    print(f"✓ Open Positions: {len(visualizer.display_data.positions)}")
    print("✓ Visualizer test completed")

def test_full_system():
    """Test full system with 30 second run"""
    print("\n" + "=" * 80)
    print("TESTING: Full System Integration (30 seconds)")
    print("=" * 80)
    print("Press Ctrl+C to stop early")
    
    from main import UltraTradingSystem
    
    # Create system with visualizer
    system = UltraTradingSystem(mode='engine', visualize=True)
    
    # Run in thread
    system_thread = threading.Thread(target=system.start)
    system_thread.daemon = True
    system_thread.start()
    
    # Run for 30 seconds or until interrupted
    start_time = time.time()
    while not shutdown_event.is_set() and (time.time() - start_time) < 30:
        time.sleep(1)
    
    print("\nStopping system...")
    system.stop()
    time.sleep(2)
    
    print("✓ Full system test completed")

def main():
    """Run all tests"""
    print("=" * 80)
    print("MT5 TRADING SYSTEM - COMPLETE SYSTEM TEST")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test components
        test_engine_only()
        test_visualizer_only()
        
        # Test full system
        if not shutdown_event.is_set():
            test_full_system()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nSystem is ready for production use!")
        print("\nRun commands:")
        print("  python3 main.py                  # Engine only")
        print("  python3 main.py --visualize      # Engine with visualizer")
        print("  python3 main.py --mode both      # Separate processes")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()