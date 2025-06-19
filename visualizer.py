#!/usr/bin/env python3
"""
Trading Visualizer - Real-time monitoring dashboard
Shows account balance, profit, positions, and strategy confidence
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Visualizer')

# Configuration
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "REFRESH_RATE": 1,  # seconds
    "ACCOUNT_CURRENCY": "JPY"
}

@dataclass
class DisplayData:
    """Data structure for display"""
    balance: float = 0
    equity: float = 0
    profit: float = 0
    daily_pnl: float = 0
    positions: List[Dict] = None
    closed_today: List[Dict] = None
    strategy_signals: Dict[str, Dict] = None
    last_update: datetime = None

class TradingVisualizer:
    def __init__(self, data_queue: Optional[queue.Queue] = None):
        self.api_base = CONFIG["API_BASE"]
        self.data_queue = data_queue
        self.running = False
        self.display_data = DisplayData()
        
        # Track closed positions
        self.known_positions = set()
        self.closed_positions = []
        
        # Initialize strategy signals for all symbols
        self.display_data.strategy_signals = {
            "EURUSD#": {"type": "WAITING", "confidence": 0, "strategies": {}},
            "USDJPY#": {"type": "WAITING", "confidence": 0, "strategies": {}},
            "GBPUSD#": {"type": "WAITING", "confidence": 0, "strategies": {}}
        }
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def format_currency(self, amount: float) -> str:
        """Format currency for display"""
        return f"¥{amount:,.0f}"
        
    def format_pips(self, points: float, symbol: str) -> str:
        """Convert points to pips"""
        if "JPY" in symbol:
            return f"{points/100:.1f}"
        else:
            return f"{points/10:.1f}"
            
    def get_account_info(self) -> Optional[Dict]:
        """Fetch account information"""
        try:
            resp = requests.get(f"{self.api_base}/account/", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None
        
    def get_positions(self) -> List[Dict]:
        """Fetch current positions"""
        try:
            resp = requests.get(f"{self.api_base}/trading/positions", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return []
        
    def get_history(self) -> List[Dict]:
        """Fetch today's trade history"""
        try:
            resp = requests.get(f"{self.api_base}/trading/history", timeout=2)
            if resp.status_code == 200:
                # Filter for today's trades
                today = datetime.now().date()
                history = resp.json()
                today_trades = []
                for trade in history:
                    try:
                        trade_date = datetime.fromisoformat(trade['time_done'].replace('Z', '+00:00')).date()
                        if trade_date == today:
                            today_trades.append(trade)
                    except:
                        pass
                return today_trades
        except:
            pass
        return []
        
    def update_display_data(self):
        """Update all display data"""
        # Account info
        account = self.get_account_info()
        if account:
            self.display_data.balance = account.get('balance', 0)
            self.display_data.equity = account.get('equity', 0)
            self.display_data.profit = account.get('profit', 0)
            
        # Positions
        positions = self.get_positions()
        self.display_data.positions = positions
        
        # Track closed positions
        current_tickets = {p['ticket'] for p in positions}
        newly_closed = self.known_positions - current_tickets
        
        if newly_closed:
            # Fetch history to get details of closed positions
            history = self.get_history()
            for trade in history:
                if trade.get('position_id') in newly_closed:
                    self.closed_positions.append(trade)
                    
        self.known_positions = current_tickets
        
        # Calculate daily P&L from closed positions
        daily_pnl = sum(trade.get('profit', 0) for trade in self.closed_positions)
        self.display_data.daily_pnl = daily_pnl
        
        # Get strategy signals from queue if available
        if self.data_queue:
            # Initialize strategy signals if needed
            if self.display_data.strategy_signals is None:
                self.display_data.strategy_signals = {}
                
            # Process all available signals in the queue
            while not self.data_queue.empty():
                try:
                    signals = self.data_queue.get_nowait()
                    # Update signals for each symbol
                    for symbol, data in signals.items():
                        self.display_data.strategy_signals[symbol] = data
                except:
                    break
                
        self.display_data.last_update = datetime.now()
        
    def display_header(self):
        """Display header information"""
        print("=" * 80)
        print("                        ULTIMATE TRADING SYSTEM MONITOR")
        print("=" * 80)
        
    def display_account(self):
        """Display account information"""
        print("\n📊 ACCOUNT STATUS")
        print("-" * 40)
        print(f"Balance:    {self.format_currency(self.display_data.balance)}")
        print(f"Equity:     {self.format_currency(self.display_data.equity)}")
        print(f"Floating:   {self.format_currency(self.display_data.profit)} "
              f"({self.display_data.profit/self.display_data.balance*100:.2f}%)" if self.display_data.balance > 0 else "")
        print(f"Daily P&L:  {self.format_currency(self.display_data.daily_pnl)} "
              f"({self.display_data.daily_pnl/self.display_data.balance*100:.2f}%)" if self.display_data.balance > 0 else "")
        
    def display_positions(self):
        """Display open positions"""
        print("\n📈 OPEN POSITIONS")
        print("-" * 80)
        
        if not self.display_data.positions:
            print("No open positions")
        else:
            print(f"{'Ticket':<10} {'Symbol':<10} {'Type':<6} {'Volume':<8} "
                  f"{'Entry':<10} {'Current':<10} {'P&L':<12} {'Time':<8}")
            print("-" * 80)
            
            for pos in self.display_data.positions:
                ticket = pos['ticket']
                symbol = pos['symbol']
                type_str = "BUY" if pos['type'] == 0 else "SELL"
                volume = pos['volume']
                entry = pos['price_open']
                current = pos['price_current']
                profit = pos['profit']
                
                # Calculate time held
                try:
                    open_time = datetime.fromtimestamp(pos['time'])
                    duration = datetime.now() - open_time
                    time_str = f"{duration.seconds//60}m"
                except:
                    time_str = "N/A"
                
                print(f"{ticket:<10} {symbol:<10} {type_str:<6} {volume:<8.2f} "
                      f"{entry:<10.5f} {current:<10.5f} {self.format_currency(profit):<12} {time_str:<8}")
                      
    def display_closed_positions(self):
        """Display closed positions from today"""
        print("\n📉 CLOSED TODAY")
        print("-" * 80)
        
        if not self.closed_positions:
            print("No closed positions today")
        else:
            print(f"{'Symbol':<10} {'Type':<6} {'Volume':<8} {'Entry':<10} "
                  f"{'Exit':<10} {'P&L':<12} {'Time':<12}")
            print("-" * 80)
            
            # Show last 5 closed positions
            for trade in self.closed_positions[-5:]:
                symbol = trade.get('symbol', 'N/A')
                type_str = "BUY" if trade.get('type', 0) == 0 else "SELL"
                volume = trade.get('volume', 0)
                entry = trade.get('price', 0)
                exit_price = trade.get('price', 0)  # This might need adjustment based on API
                profit = trade.get('profit', 0)
                
                try:
                    close_time = datetime.fromisoformat(trade['time_done'].replace('Z', '+00:00'))
                    time_str = close_time.strftime("%H:%M:%S")
                except:
                    time_str = "N/A"
                
                print(f"{symbol:<10} {type_str:<6} {volume:<8.2f} {entry:<10.5f} "
                      f"{exit_price:<10.5f} {self.format_currency(profit):<12} {time_str:<12}")
                      
    def display_strategy_signals(self):
        """Display strategy confidence levels"""
        print("\n🎯 STRATEGY SIGNALS")
        print("-" * 80)
        
        # Define the symbols we're monitoring
        monitored_symbols = ["EURUSD#", "USDJPY#", "GBPUSD#"]
        
        if not self.display_data.strategy_signals:
            print("Waiting for signals...")
        else:
            # Display in a compact format for all symbols
            print(f"{'Symbol':<10} {'Signal':<10} {'Conf':<8} {'Trend':<8} {'RSI':<8} {'BB':<8} {'Mom':<8} {'Vol':<8}")
            print("-" * 80)
            
            for symbol in monitored_symbols:
                if symbol in self.display_data.strategy_signals:
                    data = self.display_data.strategy_signals[symbol]
                    signal_type = data.get('type', 'NONE')
                    confidence = data.get('confidence', 0)
                    strategies = data.get('strategies', {})
                    
                    # Format each strategy score
                    trend = f"{strategies.get('Trend', 0):.0%}"
                    rsi = f"{strategies.get('RSI', 0):.0%}"
                    bb = f"{strategies.get('Bollinger', 0):.0%}"
                    mom = f"{strategies.get('Momentum', 0):.0%}"
                    vol = f"{strategies.get('Volume', 0):.0%}"
                    
                    # Color coding for signals (terminal colors)
                    if signal_type == "BUY":
                        signal_display = f"\033[92m{signal_type:<10}\033[0m"  # Green
                    elif signal_type == "SELL":
                        signal_display = f"\033[91m{signal_type:<10}\033[0m"  # Red
                    else:
                        signal_display = f"{signal_type:<10}"
                    
                    print(f"{symbol:<10} {signal_display} {confidence:<8.1%} {trend:<8} {rsi:<8} {bb:<8} {mom:<8} {vol:<8}")
                else:
                    print(f"{symbol:<10} {'-':<10} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8}")
                    
            # Show detailed reasons for high confidence signals
            print("\n📊 Active Signals:")
            for symbol, data in self.display_data.strategy_signals.items():
                if data.get('confidence', 0) >= 0.6 and data.get('type') != 'NONE':
                    reasons = data.get('reasons', [])
                    print(f"  {symbol}: {data.get('type')} - {', '.join(reasons[:3])}")
                        
    def display_footer(self):
        """Display footer information"""
        print("\n" + "=" * 80)
        print(f"Last Update: {self.display_data.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to exit")
        
    def run_display_loop(self):
        """Main display loop"""
        while self.running:
            try:
                # Update data
                self.update_display_data()
                
                # Clear and redraw
                self.clear_screen()
                
                # Display sections
                self.display_header()
                self.display_account()
                self.display_positions()
                self.display_closed_positions()
                self.display_strategy_signals()
                self.display_footer()
                
                # Wait before refresh
                time.sleep(CONFIG["REFRESH_RATE"])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(1)
                
    def start(self):
        """Start the visualizer"""
        self.running = True
        logger.info("Starting Trading Visualizer")
        
        # Check API connection
        account = self.get_account_info()
        if not account:
            logger.error("Cannot connect to API")
            return
            
        # Run display loop
        self.run_display_loop()
        
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        logger.info("Stopping Trading Visualizer")

def main():
    """Run standalone visualizer"""
    visualizer = TradingVisualizer()
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.stop()

if __name__ == "__main__":
    main()