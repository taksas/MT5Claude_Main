#!/usr/bin/env python3
"""Final verification that the system is working"""

print("=" * 70)
print("   FINAL SYSTEM VERIFICATION")
print("=" * 70)
print("\nRunning main.py for 30 seconds to verify trades are opening...\n")

import subprocess
import time
import threading

# Run main.py in a subprocess
process = subprocess.Popen(
    ["python3", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Kill after 30 seconds
def kill_after_delay():
    time.sleep(30)
    process.terminate()

timer = threading.Thread(target=kill_after_delay)
timer.daemon = True
timer.start()

# Read output and look for key events
trades_opened = 0
signals_generated = 0
errors = 0

try:
    for line in process.stdout:
        print(line.rstrip())
        
        if "Order placed successfully" in line or "✅" in line:
            trades_opened += 1
        if "SIGNAL GENERATED" in line:
            signals_generated += 1
        if "Error" in line or "Failed" in line:
            errors += 1
            
    process.wait()
    
except KeyboardInterrupt:
    process.terminate()

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY:")
print(f"- Trades opened: {trades_opened}")
print(f"- Signals generated: {signals_generated}")
print(f"- Errors encountered: {errors}")
print("=" * 70)

if trades_opened > 0:
    print("\n✓ SUCCESS: System is now opening trades!")
else:
    print("\n✗ ISSUE: No trades opened yet. Check logs for details.")