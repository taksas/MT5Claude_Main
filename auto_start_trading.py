#!/usr/bin/env python3
"""
Auto-Start Trading System
Monitors for MT5 API availability and automatically starts live trading
"""

import time
import subprocess
import requests
from datetime import datetime

def check_api_availability():
    """Check if MT5 API is available"""
    try:
        response = requests.get("http://172.28.144.1:8000/status/ping", timeout=3)
        return response.status_code == 200
    except:
        return False

def discover_symbols_and_start_trading():
    """Discover symbols and start live trading"""
    print("🔍 DISCOVERING AVAILABLE SYMBOLS...")
    
    # Run symbol discovery
    result = subprocess.run(['python3', 'discover_symbols.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Symbol discovery completed")
        print(result.stdout)
        
        # Start live trading
        print("\n🚀 STARTING LIVE TRADING WITH DISCOVERED SYMBOLS...")
        subprocess.run(['python3', 'main.py', '--mode', 'live'])
        
    else:
        print("❌ Symbol discovery failed")
        print(result.stderr)

def main():
    """Main monitoring and auto-start function"""
    print("🤖 MT5 CLAUDE AUTO-START TRADING MONITOR")
    print("="*50)
    print(f"Started: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    print("Waiting for MT5 Bridge API to become available...")
    print("(Checking every 10 seconds)")
    print()
    
    check_count = 0
    
    while True:
        check_count += 1
        
        if check_api_availability():
            print(f"\n✅ MT5 API DETECTED! (Check #{check_count})")
            print("🚀 STARTING AUTOMATED TRADING SEQUENCE...")
            print()
            
            # Give API a moment to fully initialize
            time.sleep(2)
            
            # Discover symbols and start trading
            discover_symbols_and_start_trading()
            break
            
        else:
            if check_count % 6 == 0:  # Status update every minute
                print(f"⏳ Still waiting... (Check #{check_count}) - {datetime.utcnow().strftime('%H:%M:%S')}")
                print("   Ensure MT5 Bridge API is running: uvicorn main:app --host 0.0.0.0 --port 8000")
            
            time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Auto-start monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")