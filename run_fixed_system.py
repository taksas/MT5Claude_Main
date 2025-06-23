#!/usr/bin/env python3
"""Run the fixed trading system"""

print("=" * 70)
print("   FIXED TRADING SYSTEM - READY TO TRADE")
print("=" * 70)
print("\nChanges made:")
print("1. ✓ Fixed order placement format (action, type, magic)")
print("2. ✓ Fixed API response handling (201 status)")
print("3. ✓ Fixed active_trades dictionary iteration")
print("4. ✓ Adjusted risk-reward ratios")
print("5. ✓ Set USDJPY# as priority symbol")
print("\nStarting trading system...\n")

import subprocess
import sys

# Run the main trading system
try:
    subprocess.run([sys.executable, "main.py"], check=True)
except KeyboardInterrupt:
    print("\nSystem stopped by user.")
except Exception as e:
    print(f"\nError: {e}")