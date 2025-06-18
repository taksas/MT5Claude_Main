#!/usr/bin/env python3
"""
Real-time performance monitor for scalping trades
Shows live statistics and recommendations
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

class ScalpingMonitor:
    def __init__(self, api_base="http://172.28.144.1:8000"):
        self.api_base = api_base
        self.trade_history = []
        self.session_start = datetime.now()
        self.initial_balance = None
        
    def get_account_info(self):
        """Get current account information"""
        try:
            response = requests.get(f"{self.api_base}/account/")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_positions(self):
        """Get open positions"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []
    
    def get_history(self):
        """Get recent trade history"""
        try:
            response = requests.get(f"{self.api_base}/trading/history")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []
    
    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate trading statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pips': 0,
                'avg_duration': 0
            }
        
        wins = [t for t in trades if t.get('profit', 0) > 0]
        losses = [t for t in trades if t.get('profit', 0) < 0]
        
        total_trades = len(trades)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['profit'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['profit']) for t in losses]) if losses else 0
        
        gross_profit = sum([t['profit'] for t in wins])
        gross_loss = abs(sum([t['profit'] for t in losses]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate pips
        total_pips = 0
        durations = []
        
        for trade in trades:
            if 'entry_price' in trade and 'exit_price' in trade:
                pip_value = 0.0001 if "JPY" not in trade.get('symbol', '') else 0.01
                if trade.get('type') == 0:  # BUY
                    pips = (trade['exit_price'] - trade['entry_price']) / pip_value
                else:  # SELL
                    pips = (trade['entry_price'] - trade['exit_price']) / pip_value
                total_pips += pips
            
            # Duration
            if 'time' in trade and 'close_time' in trade:
                try:
                    open_time = datetime.fromisoformat(trade['time'].replace('Z', '+00:00'))
                    close_time = datetime.fromisoformat(trade['close_time'].replace('Z', '+00:00'))
                    duration = (close_time - open_time).total_seconds() / 60  # minutes
                    durations.append(duration)
                except:
                    pass
        
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pips': total_pips,
            'avg_duration': avg_duration,
            'winning_trades': len(wins),
            'losing_trades': len(losses)
        }
    
    def display_dashboard(self):
        """Display live trading dashboard"""
        print("\033[2J\033[H")  # Clear screen
        print("="*70)
        print("ULTRA-SCALPING PERFORMANCE MONITOR")
        print("="*70)
        print(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {(datetime.now() - self.session_start).total_seconds() / 3600:.1f} hours")
        print("-"*70)
        
        # Account info
        account = self.get_account_info()
        if account:
            if self.initial_balance is None:
                self.initial_balance = account['balance']
            
            session_pl = account['balance'] - self.initial_balance
            session_return = (session_pl / self.initial_balance * 100) if self.initial_balance > 0 else 0
            
            print(f"Balance: ${account['balance']:.2f}")
            print(f"Session P/L: ${session_pl:.2f} ({session_return:+.2f}%)")
            print(f"Free Margin: ${account['margin_free']:.2f}")
            print("-"*70)
        
        # Open positions
        positions = self.get_positions()
        if positions:
            print(f"\nOPEN POSITIONS ({len(positions)}):")
            print(f"{'Symbol':<10} {'Type':<6} {'Entry':<10} {'Current':<10} {'P/L':<10} {'Duration':<10}")
            print("-"*70)
            
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                pos_type = 'BUY' if pos.get('type') == 0 else 'SELL'
                entry = pos.get('price_open', 0)
                current = pos.get('price_current', 0)
                profit = pos.get('profit', 0)
                
                # Calculate duration
                open_time = datetime.now()  # Would need actual open time
                duration = "N/A"
                
                print(f"{symbol:<10} {pos_type:<6} {entry:<10.5f} {current:<10.5f} "
                      f"${profit:<9.2f} {duration:<10}")
        else:
            print("\nNo open positions")
        
        print("-"*70)
        
        # Historical performance
        history = self.get_history()
        if history:
            # Filter today's trades
            today_trades = []
            for trade in history:
                try:
                    if 'close_time' in trade:
                        close_time = datetime.fromisoformat(trade['close_time'].replace('Z', '+00:00'))
                        if close_time.date() == datetime.now().date():
                            today_trades.append(trade)
                except:
                    pass
            
            stats = self.calculate_statistics(today_trades)
            
            print(f"\nTODAY'S PERFORMANCE:")
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Winning Trades: {stats['winning_trades']}")
            print(f"Losing Trades: {stats['losing_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Average Win: ${stats['avg_win']:.2f}")
            print(f"Average Loss: ${stats['avg_loss']:.2f}")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Total Pips: {stats['total_pips']:.1f}")
            print(f"Avg Duration: {stats['avg_duration']:.1f} minutes")
        
        print("="*70)
        
        # Recommendations
        self.print_recommendations(stats if history else None, positions)
    
    def print_recommendations(self, stats: Dict, positions: List):
        """Print trading recommendations based on performance"""
        print("\nRECOMMENDATIONS:")
        
        if not stats or stats['total_trades'] == 0:
            print("• No trades yet today - waiting for optimal setups")
            return
        
        # Win rate analysis
        if stats['win_rate'] < 60:
            print("• Win rate below target (60%) - consider tightening entry criteria")
        elif stats['win_rate'] > 80:
            print("• Excellent win rate! Maintain current strategy")
        
        # Duration analysis
        if stats['avg_duration'] > 10:
            print("• Average trade duration exceeds 10 minutes - consider quicker exits")
        elif stats['avg_duration'] < 3:
            print("• Very short trade duration - ensure proper setup confirmation")
        
        # Profit factor
        if stats['profit_factor'] < 1.5:
            print("• Profit factor needs improvement - focus on risk/reward ratio")
        
        # Position management
        if len(positions) > 0:
            for pos in positions:
                if pos.get('profit', 0) > 8:
                    print(f"• Consider taking profit on {pos['symbol']} (${pos['profit']:.2f})")
                elif pos.get('profit', 0) < -5:
                    print(f"• Review stop loss on {pos['symbol']} (${pos['profit']:.2f})")
    
    def run_monitor(self, refresh_interval=5):
        """Run the monitoring dashboard"""
        print("Starting scalping monitor... Press Ctrl+C to exit")
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")

def main():
    monitor = ScalpingMonitor()
    monitor.run_monitor(refresh_interval=5)

if __name__ == "__main__":
    main()