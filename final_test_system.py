#!/usr/bin/env python3
"""Final comprehensive test of the trading system"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("   TRADING SYSTEM DIAGNOSTIC TEST")
print("=" * 70)
print("\nTesting all components to identify signal generation issues...\n")

# Run the main trading system
print("Starting trading engine with current configuration...")
print("This will run for 30 seconds to observe behavior.\n")

try:
    from main import UltraTradingSystem
    
    # Create and start system
    system = UltraTradingSystem(mode='engine', visualize=False)
    
    # Monkey patch to run for limited time
    import time
    import threading
    
    def stop_after_delay():
        time.sleep(30)  # Run for 30 seconds
        print("\n\n=== TEST COMPLETE ===")
        print("Stopping system after 30 seconds...")
        system.shutdown = True
        system.stop()
        
    # Start timer thread
    timer = threading.Thread(target=stop_after_delay)
    timer.daemon = True
    timer.start()
    
    # Start the system
    system.start()
    
except KeyboardInterrupt:
    print("\nTest interrupted by user.")
except Exception as e:
    print(f"\nError during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY:")
print("- If no trades opened, check MIN_CONFIDENCE and RR ratio settings")
print("- Current MIN_CONFIDENCE: 15%")
print("- Current MIN_RR_EXOTIC: 1.2")
print("- System should force trades every 2 minutes if no signals")
print("=" * 70)