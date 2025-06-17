#!/usr/bin/env python3
"""
Instant Live Trader - Starts immediately when API is available
"""

import time
import requests
from datetime import datetime
from mt5_client import MT5Client
from trading_engine import TradingEngine
import logging

def start_instant_trading():
    """Start trading immediately with available symbols"""
    
    print("🚀 INSTANT LIVE TRADING SYSTEM ACTIVATED")
    print("="*50)
    print(f"Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    # Bypassing API check as requested
    print("⚠️  Bypassing API connection check")
    
    # Initialize trading engine with enhanced symbol discovery
    engine = TradingEngine()
    
    if not engine.initialize():
        print("❌ Failed to initialize trading engine")
        return False
    
    print("✅ Trading engine initialized")
    print(f"🎯 Approved symbols: {list(engine.approved_symbols)}")
    
    if not engine.approved_symbols:
        print("❌ No approved symbols found")
        return False
    
    # Start live trading
    print("\n🚀 STARTING LIVE AUTOMATED TRADING...")
    print("   - Real money trading with discovered symbols")
    print("   - Enhanced 2025 strategies active")
    print("   - Risk management: 1.5% per trade")
    print("   - Position holds: 5-30 minutes")
    print("   - Automatic stop losses")
    print()
    
    try:
        engine.start_trading()
        
        # Keep monitoring
        while engine.is_running:
            time.sleep(60)  # Status check every minute
            status = engine.get_status()
            
            print(f"📊 Status: {len(status['active_positions'])} active positions")
            
            if status['risk_status']['emergency_stop']:
                print("🛑 Emergency stop triggered")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
        engine.stop_trading()
    
    return True

if __name__ == "__main__":
    # Start trading without connection check
    print("🚀 STARTING TRADING WITHOUT CONNECTION CHECK")
    start_instant_trading()