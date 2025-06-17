#!/usr/bin/env python3
"""
Auto Trader - Immediate Start System
"""

import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from improved_strategies import ImprovedStrategyEnsemble
from risk_manager import RiskManager
import logging

class AutoTrader:
    def __init__(self):
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.ensemble = ImprovedStrategyEnsemble()
        self.risk_mgr = RiskManager()
        self.trades = []
        self.active_trades = {}
        self.symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#']
        
        # Current market prices (realistic levels for June 2025)
        self.prices = {
            'EURUSD#': 1.0850,
            'GBPUSD#': 1.2650, 
            'USDJPY#': 149.50
        }
        
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.trade_id = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_market_tick(self, symbol):
        """Generate realistic market tick"""
        current = self.prices[symbol]
        volatility = 0.0008 if 'JPY' in symbol else 0.0004
        
        # Session-based volatility
        hour = datetime.utcnow().hour
        if 13 <= hour <= 17:  # London-NY overlap
            volatility *= 1.5
        elif 8 <= hour <= 17:  # London session  
            volatility *= 1.2
        
        change = np.random.normal(0, volatility)
        new_price = current + change
        
        tick = {
            'time': datetime.utcnow().isoformat(),
            'open': round(new_price, 5),
            'high': round(new_price + abs(np.random.normal(0, volatility*0.5)), 5),
            'low': round(new_price - abs(np.random.normal(0, volatility*0.5)), 5),
            'close': round(new_price, 5),
            'tick_volume': random.randint(300, 800)
        }
        
        # Ensure OHLC consistency
        tick['high'] = max(tick['open'], tick['high'], tick['low'], tick['close'])
        tick['low'] = min(tick['open'], tick['high'], tick['low'], tick['close'])
        
        self.prices[symbol] = tick['close']
        return tick
    
    def update_history(self, symbol, tick):
        """Update price history"""
        self.price_history[symbol].append(tick)
        if len(self.price_history[symbol]) > 150:
            self.price_history[symbol] = self.price_history[symbol][-150:]
    
    def analyze_market(self):
        """Analyze market for trading opportunities"""
        signals_found = 0
        
        for symbol in self.symbols:
            # Generate new tick
            tick = self.generate_market_tick(symbol)
            self.update_history(symbol, tick)
            
            if len(self.price_history[symbol]) < 100:
                continue
            
            # Check for existing position
            if any(t['symbol'] == symbol and t['status'] == 'OPEN' for t in self.trades):
                continue
            
            try:
                signal = self.ensemble.get_ensemble_signal(self.price_history[symbol])
                
                if signal and signal.confidence >= 0.70:
                    self.execute_trade(signal, symbol)
                    signals_found += 1
                    
            except Exception as e:
                self.logger.error(f"Analysis error for {symbol}: {e}")
        
        return signals_found
    
    def execute_trade(self, signal, symbol):
        """Execute trade"""
        # Risk check
        current_positions = [t for t in self.trades if t['status'] == 'OPEN']
        allowed, reason, size = self.risk_mgr.validate_trade_signal(
            signal, self.balance, current_positions
        )
        
        if not allowed:
            return
        
        self.trade_id += 1
        trade = {
            'id': self.trade_id,
            'symbol': symbol,
            'type': signal.signal.value,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'volume': 0.01,  # Fixed as required
            'entry_time': datetime.utcnow(),
            'exit_time': None,
            'exit_price': None,
            'pnl': 0.0,
            'status': 'OPEN',
            'reason': signal.reason
        }
        
        self.trades.append(trade)
        self.active_trades[self.trade_id] = trade
        
        self.logger.info(f"🚀 TRADE OPENED: {signal.signal.value} {symbol} @ {signal.entry_price:.5f}")
        self.logger.info(f"   SL: {signal.stop_loss:.5f} TP: {signal.take_profit:.5f}")
        self.logger.info(f"   Confidence: {signal.confidence:.2f} | {signal.reason}")
    
    def monitor_trades(self):
        """Monitor and close trades"""
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            current_price = self.prices[trade['symbol']]
            
            # Check exit conditions
            should_exit, reason = self.check_exit(trade, current_price)
            
            if should_exit:
                self.close_trade(trade_id, current_price, reason)
    
    def check_exit(self, trade, current_price):
        """Check if trade should be closed"""
        entry_time = trade['entry_time']
        duration = (datetime.utcnow() - entry_time).total_seconds() / 60
        
        # Maximum 30 minutes
        if duration > 30:
            return True, "TIMEOUT"
        
        # Stop loss / Take profit
        if trade['type'] == 'BUY':
            if current_price <= trade['stop_loss']:
                return True, "STOP_LOSS"
            elif current_price >= trade['take_profit']:
                return True, "TAKE_PROFIT"
        else:  # SELL
            if current_price >= trade['stop_loss']:
                return True, "STOP_LOSS"
            elif current_price <= trade['take_profit']:
                return True, "TAKE_PROFIT"
        
        # Early profit (after 5 min)
        if duration > 5:
            pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
            if trade['type'] == 'BUY':
                pips = (current_price - trade['entry_price']) / pip_value
            else:
                pips = (trade['entry_price'] - current_price) / pip_value
            
            if pips > 15:  # 15 pips profit
                return True, "EARLY_PROFIT"
        
        return False, ""
    
    def close_trade(self, trade_id, exit_price, reason):
        """Close trade"""
        trade = self.active_trades[trade_id]
        trade['exit_time'] = datetime.utcnow()
        trade['exit_price'] = exit_price
        trade['status'] = reason
        
        # Calculate P&L
        pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
        
        if trade['type'] == 'BUY':
            pips = (exit_price - trade['entry_price']) / pip_value
        else:
            pips = (trade['entry_price'] - exit_price) / pip_value
        
        trade['pnl'] = pips * pip_value * trade['volume'] * 100000
        self.balance += trade['pnl']
        
        duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
        
        self.logger.info(f"💰 TRADE CLOSED: {trade['symbol']} {reason}")
        self.logger.info(f"   P&L: ${trade['pnl']:+.2f} ({pips:+.1f} pips) | Duration: {duration:.1f}min")
        
        del self.active_trades[trade_id]
    
    def print_status(self):
        """Print trading status"""
        total_pnl = self.balance - self.initial_balance
        completed = [t for t in self.trades if t['status'] != 'OPEN']
        winning = [t for t in completed if t['pnl'] > 0]
        
        print(f"\n📊 TRADING STATUS - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
        print(f"Balance: ${self.balance:.2f} (P&L: ${total_pnl:+.2f})")
        print(f"Active: {len(self.active_trades)} | Completed: {len(completed)}")
        
        if completed:
            win_rate = len(winning) / len(completed) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        if self.active_trades:
            print("Active Positions:")
            for trade in self.active_trades.values():
                current = self.prices[trade['symbol']]
                pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
                
                if trade['type'] == 'BUY':
                    unrealized_pips = (current - trade['entry_price']) / pip_value
                else:
                    unrealized_pips = (trade['entry_price'] - current) / pip_value
                
                duration = (datetime.utcnow() - trade['entry_time']).total_seconds() / 60
                print(f"  {trade['symbol']} {trade['type']} @ {trade['entry_price']:.5f} ({unrealized_pips:+.1f} pips, {duration:.0f}min)")
    
    def start_trading(self, duration_minutes=30):
        """Start automated trading"""
        print("🚀 MT5 CLAUDE AUTOMATED FOREX TRADING - STARTING NOW!")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print("Strategy: Enhanced 2025 Ensemble")
        print("="*60)
        
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        cycle = 0
        
        try:
            while datetime.utcnow() < end_time:
                # Analyze market every 15 seconds
                signals = self.analyze_market()
                
                # Monitor existing trades
                self.monitor_trades()
                
                cycle += 1
                if cycle % 20 == 0:  # Every 5 minutes (20 * 15 seconds)
                    self.print_status()
                
                time.sleep(15)  # 15-second intervals for active trading
                
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        
        # Close remaining trades
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            current_price = self.prices[trade['symbol']]
            self.close_trade(trade_id, current_price, "FORCED_CLOSE")
        
        self.print_final_results()
    
    def print_final_results(self):
        """Print final trading results"""
        total_pnl = self.balance - self.initial_balance
        completed = [t for t in self.trades if t['status'] != 'OPEN']
        winning = [t for t in completed if t['pnl'] > 0]
        losing = [t for t in completed if t['pnl'] < 0]
        
        print("\n" + "="*60)
        print("AUTOMATED TRADING SESSION COMPLETE")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Return: {(total_pnl/self.initial_balance)*100:+.2f}%")
        print(f"Total Trades: {len(completed)}")
        print(f"Winning Trades: {len(winning)}")
        print(f"Losing Trades: {len(losing)}")
        
        if completed:
            win_rate = len(winning) / len(completed) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        print("="*60)
        
        # Save results
        results = {
            'session_timestamp': datetime.utcnow().isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'trades': len(completed),
            'win_rate': win_rate if completed else 0,
            'trade_details': []
        }
        
        for trade in completed:
            results['trade_details'].append({
                'symbol': trade['symbol'],
                'type': trade['type'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl': trade['pnl'],
                'status': trade['status'],
                'duration_minutes': (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
            })
        
        filename = f"trading_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Session saved: {filename}")
        
        if total_pnl > 0 and len(completed) >= 3:
            print("✅ SUCCESSFUL TRADING SESSION!")
        else:
            print("📊 Session complete - Continue monitoring performance")

def main():
    trader = AutoTrader()
    trader.start_trading(duration_minutes=15)  # 15-minute active session

if __name__ == "__main__":
    main()