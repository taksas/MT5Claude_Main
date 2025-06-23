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
        
        # Initialize strategy signals (will be populated dynamically)
        self.display_data.strategy_signals = {}
        
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
        # Note: History endpoint not available in current API
        # This would need to be implemented in the MT5 Bridge API
        # For now, return empty list
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
        print("=" * 100)
        print("                              ULTRA TRADING SYSTEM MONITOR")
        print("=" * 100)
        
    def display_account(self):
        """Display account information in compact format"""
        print("\n💰 ACCOUNT: ", end="")
        print(f"Balance: {self.format_currency(self.display_data.balance)} | "
              f"Equity: {self.format_currency(self.display_data.equity)} | "
              f"Floating: {self.format_currency(self.display_data.profit)} "
              f"({self.display_data.profit/self.display_data.balance*100:+.1f}%) | " if self.display_data.balance > 0 else " | ", end="")
        print(f"Daily: {self.format_currency(self.display_data.daily_pnl)} "
              f"({self.display_data.daily_pnl/self.display_data.balance*100:+.1f}%)" if self.display_data.balance > 0 else "")
        
    def display_positions(self):
        """Display open positions in compact format"""
        if self.display_data.positions:
            print(f"\n📈 POSITIONS ({len(self.display_data.positions)}): ", end="")
            total_profit = sum(pos.get('profit', 0) for pos in self.display_data.positions)
            
            # Show summary
            for i, pos in enumerate(self.display_data.positions[:5]):  # Show first 5
                symbol = pos['symbol'].replace('#', '')
                type_icon = "🟢" if pos['type'] == 0 else "🔴"
                profit = pos['profit']
                profit_str = f"+{profit:.0f}" if profit > 0 else f"{profit:.0f}"
                print(f"{type_icon}{symbol}:{profit_str}¥", end=" ")
                
            if len(self.display_data.positions) > 5:
                print(f"(+{len(self.display_data.positions)-5} more)", end=" ")
                
            print(f"| Total: {self.format_currency(total_profit)}")
                      
    def display_closed_positions(self):
        """Display closed positions summary"""
        if self.closed_positions:
            wins = sum(1 for t in self.closed_positions if t.get('profit', 0) > 0)
            total_closed_pnl = sum(t.get('profit', 0) for t in self.closed_positions)
            win_rate = wins / len(self.closed_positions) * 100 if self.closed_positions else 0
            
            print(f"\n📊 CLOSED TODAY ({len(self.closed_positions)}): "
                  f"Wins: {wins} | Win Rate: {win_rate:.0f}% | "
                  f"P&L: {self.format_currency(total_closed_pnl)}")
                      
    def display_strategy_signals(self):
        """Display strategy confidence levels with expanded 2-line format per symbol"""
        print("\n🎯 STRATEGY SIGNALS (EXPANDED VIEW)")
        print("=" * 100)
        
        if not self.display_data.strategy_signals:
            print("Waiting for signals...")
        else:
            # Sort symbols by confidence for better visibility
            sorted_symbols = sorted(self.display_data.strategy_signals.items(), 
                                  key=lambda x: x[1].get('confidence', 0), reverse=True)
            
            # Display all symbols with stored signals
            for idx, (symbol, data) in enumerate(sorted_symbols):
                signal_type = data.get('type', 'NONE')
                confidence = data.get('confidence', 0)
                quality = data.get('quality', 0)
                strategies = data.get('strategies', {})
                reasons = data.get('reasons', [])
                
                # Skip very weak signals unless they're recent
                if confidence < 0.1 and signal_type == 'NONE':
                    continue
                
                # Symbol header with signal type
                if signal_type == "BUY":
                    signal_icon = "🟢"
                    signal_color = "\033[92m"
                elif signal_type == "SELL":
                    signal_icon = "🔴"
                    signal_color = "\033[91m"
                else:
                    signal_icon = "⚪"
                    signal_color = "\033[90m"
                
                # Line 1: Symbol, Signal, Confidence, Quality, Main Reasons
                print(f"\n{signal_icon} {signal_color}{symbol:<10}\033[0m Signal: {signal_color}{signal_type:<6}\033[0m "
                      f"Conf: {confidence:>5.1%} | Qual: {quality:>5.1%} | {', '.join(reasons[:3]) if reasons else 'Analyzing...'}")
                
                # Line 2: All active indicators with their values
                if strategies:
                    # Collect all active indicators
                    active_indicators = []
                    
                    # For Ultra engine with many indicators
                    indicator_categories = {
                        'PA': ['PriceAction', 'pin_bar', 'engulfing', 'hammer', 'doji'],
                        'CP': ['ChartPatterns', 'double_top', 'double_bottom', 'triangle'],
                        'MA': ['Trend', 'MACD', 'SMA', 'EMA'],
                        'MO': ['RSI', 'Stochastic', 'Momentum', 'CCI', 'Williams'],
                        'VO': ['Volume', 'MFI', 'OBV', 'Chaikin'],
                        'ST': ['Structure', 'Support', 'Resistance', 'ADX'],
                        'VL': ['Bollinger', 'ATR', 'Keltner', 'Volatility'],
                        'TM': ['Time', 'Session', 'Pattern'],
                        'MS': ['Statistical', 'ZScore', 'Divergence'],
                        'IC': ['Ichimoku', 'Cloud']
                    }
                    
                    # Build indicator string
                    for category, indicators in indicator_categories.items():
                        for indicator in indicators:
                            if indicator in strategies and strategies[indicator] > 0:
                                score = strategies[indicator]
                                if score >= 0.8:
                                    active_indicators.append(f"{indicator}:{score:.0%}*")
                                elif score >= 0.5:
                                    active_indicators.append(f"{indicator}:{score:.0%}")
                                else:
                                    active_indicators.append(f"{indicator}:{score:.0%}")
                    
                    # If no categorized indicators, show all
                    if not active_indicators:
                        for strat, score in strategies.items():
                            if score > 0:
                                active_indicators.append(f"{strat}:{score:.0%}")
                    
                    # Display indicators (limit to fit on screen)
                    indicator_str = " | ".join(active_indicators[:12])
                    if len(active_indicators) > 12:
                        indicator_str += f" (+{len(active_indicators)-12} more)"
                    
                    print(f"   └─ Indicators: {indicator_str}")
                else:
                    print(f"   └─ Indicators: Calculating...")
                
                # Add separator every 5 symbols for readability
                if (idx + 1) % 5 == 0:
                    print("   " + "-" * 95)
                    
            # Summary footer
            print("\n" + "=" * 100)
            total_symbols = len(self.display_data.strategy_signals)
            buy_count = sum(1 for s in self.display_data.strategy_signals.values() if s.get('type') == 'BUY')
            sell_count = sum(1 for s in self.display_data.strategy_signals.values() if s.get('type') == 'SELL')
            high_conf = sum(1 for s in self.display_data.strategy_signals.values() if s.get('confidence', 0) >= 0.5)
            
            print(f"📊 Summary: {total_symbols} symbols | "
                  f"🟢 {buy_count} BUY | 🔴 {sell_count} SELL | "
                  f"⭐ {high_conf} High Confidence (>50%)")
                        
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