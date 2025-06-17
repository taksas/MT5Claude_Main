#!/usr/bin/env python3
"""
MT5 Claude Automated Forex Trading System
Main orchestrator script for automated short-term forex trading
"""

import sys
import time
import logging
import argparse
from datetime import datetime
from mt5_client import MT5Client, test_connection
from trading_engine import TradingEngine
from paper_trading import PaperTradingEngine
from backtesting import Backtester
from trading_strategies import StrategyEnsemble

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'trading_log_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_market_hours() -> bool:
    """Check if markets are open for trading"""
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour
    
    # Forex markets are closed on weekends
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        if weekday == 6 and hour < 22:  # Sunday before 22:00 UTC
            return False
        elif weekday == 5 and hour >= 22:  # Friday after 22:00 UTC
            return False
    
    return True

def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("="*60)
    print("MT5 CLAUDE TRADING SYSTEM DIAGNOSTICS")
    print("="*60)
    
    # Test MT5 API connection
    print("\n1. Testing MT5 API Connection...")
    if test_connection():
        print("✅ MT5 API connection successful")
        api_available = True
    else:
        print("❌ MT5 API connection failed")
        print("   Please ensure:")
        print("   - MT5 Bridge API server is running on Windows host")
        print("   - Windows firewall allows port 8000")
        print("   - MT5 terminal is logged in and algorithmic trading is enabled")
        api_available = False
    
    # Test strategy components
    print("\n2. Testing Strategy Components...")
    try:
        ensemble = StrategyEnsemble()
        print("✅ Strategy ensemble initialized")
        
        # Test with sample data
        sample_data = []
        for i in range(100):
            sample_data.append({
                'time': datetime.utcnow().isoformat(),
                'open': 1.1000 + i * 0.0001,
                'high': 1.1005 + i * 0.0001,
                'low': 1.0995 + i * 0.0001,
                'close': 1.1002 + i * 0.0001,
                'tick_volume': 100
            })
        
        signal = ensemble.get_ensemble_signal(sample_data)
        print("✅ Strategy analysis working")
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
    
    # Test backtesting system
    print("\n3. Testing Backtesting System...")
    try:
        backtester = Backtester()
        print("✅ Backtesting system ready")
    except Exception as e:
        print(f"❌ Backtesting system failed: {e}")
    
    print("\n" + "="*60)
    return api_available

def run_paper_trading(duration_hours: int = 2):
    """Run paper trading simulation"""
    print(f"\nStarting {duration_hours}-hour paper trading simulation...")
    
    test_symbols = ["EURUSD#", "GBPUSD#", "USDJPY#", "USDCAD#"]
    paper_engine = PaperTradingEngine(initial_balance=10000.0)
    
    try:
        paper_engine.start_paper_trading(test_symbols, duration_hours)
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user")

def run_live_trading():
    """Run live trading engine"""
    if not check_market_hours():
        print("Markets are currently closed. Live trading will start when markets open.")
        return
    
    print("\nInitializing live trading engine...")
    
    engine = TradingEngine()
    
    if not engine.initialize():
        print("❌ Failed to initialize trading engine")
        return
    
    print("✅ Trading engine initialized successfully")
    print("🚀 Starting automated forex trading...")
    print("   - Position hold time: 5-30 minutes")
    print("   - Risk per trade: 2% of account")
    print("   - Stop losses: Automatic")
    print("   - Lot size: 0.01 only")
    print("\nPress Ctrl+C to stop trading safely...")
    
    try:
        engine.start_trading()
        
        # Monitor and report status
        while True:
            time.sleep(300)  # 5-minute status updates
            status = engine.get_status()
            
            print(f"\n--- Status Update {datetime.utcnow().strftime('%H:%M:%S')} ---")
            print(f"Active positions: {status['active_positions']}")
            print(f"Approved symbols: {len(status['approved_symbols'])}")
            
            risk_status = status['risk_status']
            print(f"Trades today: {risk_status['trades_today']}")
            print(f"Daily P&L: ${risk_status['daily_pnl']:.2f}")
            
            if risk_status['emergency_stop']:
                print("⚠️  Emergency stop conditions detected:")
                for reason in risk_status['emergency_reasons']:
                    print(f"   - {reason}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping trading engine safely...")
        engine.stop_trading()
        print("✅ Trading engine stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MT5 Claude Automated Trading System')
    parser.add_argument('--mode', choices=['diagnostic', 'paper', 'live'], 
                       default='diagnostic', help='Operation mode')
    parser.add_argument('--paper-hours', type=int, default=2,
                       help='Paper trading duration in hours')
    parser.add_argument('--force', action='store_true',
                       help='Force live trading even if diagnostics fail')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("MT5 CLAUDE AUTOMATED FOREX TRADING SYSTEM")
    print("Short-term trading with 5-30 minute position holds")
    print(f"Started at: {datetime.utcnow().isoformat()} UTC")
    
    # Always run diagnostics first
    api_available = run_system_diagnostics()
    
    if args.mode == 'diagnostic':
        print("\nDiagnostic complete. Use --mode paper or --mode live to start trading.")
        
    elif args.mode == 'paper':
        print(f"\nRunning paper trading for {args.paper_hours} hours...")
        run_paper_trading(args.paper_hours)
        
    elif args.mode == 'live':
        if not api_available and not args.force:
            print("\n❌ Cannot start live trading - MT5 API not available")
            print("   Use --force to override (not recommended)")
            sys.exit(1)
        
        if not args.force:
            confirm = input("\n⚠️  Start LIVE trading with REAL money? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Live trading cancelled")
                sys.exit(0)
        
        run_live_trading()

if __name__ == "__main__":
    main()