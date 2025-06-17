#!/usr/bin/env python3
"""
Trading Session Monitor - Real-time monitoring of autonomous trading
"""

import time
import subprocess
import os
from datetime import datetime

def check_session_status():
    """Check if trading session is still running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'final_trader.py'], 
                              capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def get_latest_logs(lines=10):
    """Get latest log entries"""
    try:
        with open('trading_session.log', 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
    except:
        return "No log file found"

def main():
    print("🔍 === AUTONOMOUS TRADING SESSION MONITOR ===")
    print(f"⏰ Monitor started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Checking session every 30 seconds...")
    print("🛑 Press Ctrl+C to stop monitoring\n")
    
    session_start = datetime.now()
    
    try:
        while True:
            current_time = datetime.now()
            elapsed = current_time - session_start
            
            # Check if session is running
            is_running = check_session_status()
            status = "🟢 RUNNING" if is_running else "🔴 STOPPED"
            
            print(f"📅 {current_time.strftime('%H:%M:%S')} | Monitor: {elapsed} | Session: {status}")
            
            if not is_running:
                print("\n⚠️  Trading session appears to have stopped!")
                print("📋 Last log entries:")
                print(get_latest_logs(5))
                break
            
            # Show recent activity every 5 minutes
            if elapsed.total_seconds() % 300 < 30:  # Every 5 minutes (with 30s tolerance)
                print("\n📊 Recent Activity:")
                recent_logs = get_latest_logs(3)
                for line in recent_logs.split('\n')[-3:]:
                    if line.strip():
                        print(f"   {line}")
                print()
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitor stopped at {datetime.now().strftime('%H:%M:%S')}")
        
        if check_session_status():
            print("✅ Trading session is still running in background")
            print("📋 Check logs: tail -f trading_session.log")
            print("🛑 Stop session: pkill -f final_trader.py")
        else:
            print("ℹ️  Trading session has completed or stopped")

if __name__ == "__main__":
    main()