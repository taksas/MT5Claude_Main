#!/usr/bin/env python3
"""
Real-time monitoring dashboard for parallel trading across all symbols
"""

import requests
import time
import json
from datetime import datetime
from collections import defaultdict
import os

class ParallelTradingMonitor:
    def __init__(self, api_base="http://172.28.144.1:8000"):
        self.api_base = api_base
        self.symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
        self.start_time = datetime.now()
        
    def get_positions(self):
        """Get all open positions"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []
    
    def get_account_info(self):
        """Get account information"""
        try:
            response = requests.get(f"{self.api_base}/account/")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_symbol_info(self, symbol):
        """Get current price info for a symbol"""
        try:
            response = requests.get(f"{self.api_base}/market/symbols/{symbol}")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def display_dashboard(self):
        """Display the monitoring dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("="*80)
        print("PARALLEL FOREX TRADING MONITOR".center(80))
        print("="*80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours")
        print("-"*80)
        
        # Account info
        account = self.get_account_info()
        if account:
            print(f"Balance: ${account['balance']:.2f} | "
                  f"Equity: ${account['equity']:.2f} | "
                  f"Margin: ${account['margin']:.2f} | "
                  f"Free: ${account['margin_free']:.2f}")
            print("-"*80)
        
        # Open positions
        positions = self.get_positions()
        if positions:
            print(f"\nOPEN POSITIONS ({len(positions)}):")
            print(f"{'Symbol':<10} {'Type':<5} {'Entry':<10} {'Current':<10} {'P/L':<10} {'Pips':<8} {'Time':<8}")
            print("-"*80)
            
            total_pnl = 0
            for pos in sorted(positions, key=lambda x: x.get('profit', 0), reverse=True):
                symbol = pos.get('symbol', 'N/A')
                pos_type = 'BUY' if pos.get('type') == 0 else 'SELL'
                entry = pos.get('price_open', 0)
                current = pos.get('price_current', 0)
                profit = pos.get('profit', 0)
                total_pnl += profit
                
                # Calculate pips
                pip_value = 0.01 if "JPY" in symbol else 0.0001
                if pos.get('type') == 0:  # BUY
                    pips = (current - entry) / pip_value
                else:  # SELL
                    pips = (entry - current) / pip_value
                
                # Calculate time in position
                try:
                    open_time = datetime.fromisoformat(pos.get('time', '').replace('Z', '+00:00'))
                    duration = (datetime.now(open_time.tzinfo) - open_time).total_seconds() / 60
                    time_str = f"{duration:.0f}m"
                except:
                    time_str = "N/A"
                
                # Color coding for P/L
                color = '\033[92m' if profit > 0 else '\033[91m' if profit < 0 else '\033[0m'
                reset = '\033[0m'
                
                print(f"{symbol:<10} {pos_type:<5} {entry:<10.5f} {current:<10.5f} "
                      f"{color}${profit:<9.2f}{reset} {pips:<7.1f} {time_str:<8}")
            
            print("-"*80)
            print(f"Total P/L: ${total_pnl:.2f}")
        else:
            print("\nNo open positions")
        
        # Symbol performance summary
        print("\nSYMBOL ACTIVITY:")
        active_symbols = set()
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol:
                active_symbols.add(symbol)
                self.symbol_stats[symbol]['trades'] += 0  # Will be updated from history
        
        if active_symbols:
            print(f"Active in {len(active_symbols)} symbols: {', '.join(sorted(active_symbols))}")
        
        # Trading statistics
        print("\nTRADING STATISTICS:")
        print(f"Total symbols available: 28")
        print(f"Max concurrent trades: 3")
        print(f"Analysis interval: 10 seconds")
        print(f"Trading hours: Active (avoiding 03:00-08:00 JST)")
        
        print("="*80)
    
    def run(self, refresh_interval=5):
        """Run the monitoring dashboard"""
        print("Starting parallel trading monitor...")
        print("Press Ctrl+C to exit")
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")

def main():
    monitor = ParallelTradingMonitor()
    monitor.run(refresh_interval=3)

if __name__ == "__main__":
    main()