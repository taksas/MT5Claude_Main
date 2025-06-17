#!/usr/bin/env python3
"""
Live Trading Simulation Engine
Simulates live trading with real-time decision making
"""

import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from improved_strategies import ImprovedStrategyEnsemble
from risk_manager import RiskManager
import logging

@dataclass
class LiveTrade:
    id: int
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    entry_time: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "OPEN"
    reason: str = ""

class LiveTradingSimulator:
    """
    Simulates live trading environment with real-time price feeds
    """
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.strategy_ensemble = ImprovedStrategyEnsemble()
        self.risk_manager = RiskManager()
        
        self.active_trades = {}
        self.completed_trades = []
        self.trade_counter = 0
        self.is_running = False
        
        # Market simulation parameters
        self.symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#']
        self.current_prices = {
            'EURUSD#': 1.0850,
            'GBPUSD#': 1.2650,
            'USDJPY#': 149.50
        }
        
        # Price history for strategy analysis
        self.price_history = {symbol: [] for symbol in self.symbols}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_real_time_price(self, symbol: str) -> Dict:
        """Generate realistic real-time price tick"""
        current_price = self.current_prices[symbol]
        
        # Simulate realistic price movement
        volatility = 0.0008 if 'JPY' in symbol else 0.0004
        
        # Add session-based volatility
        hour = datetime.utcnow().hour
        if 13 <= hour <= 17:  # London-NY overlap
            volatility *= 1.5
        elif 8 <= hour <= 17:  # London session
            volatility *= 1.2
        
        # Generate price movement
        price_change = np.random.normal(0, volatility)
        new_price = current_price + price_change
        
        # Generate OHLC tick
        spread = volatility * 0.3
        tick = {
            'time': datetime.utcnow().isoformat(),
            'open': round(new_price, 5),
            'high': round(new_price + abs(np.random.normal(0, spread)), 5),
            'low': round(new_price - abs(np.random.normal(0, spread)), 5),
            'close': round(new_price, 5),
            'tick_volume': random.randint(200, 800)
        }
        
        # Ensure OHLC consistency
        tick['high'] = max(tick['open'], tick['high'], tick['low'], tick['close'])
        tick['low'] = min(tick['open'], tick['high'], tick['low'], tick['close'])
        
        self.current_prices[symbol] = tick['close']
        return tick
    
    def update_price_history(self, symbol: str, tick: Dict):
        """Update price history for strategy analysis"""
        self.price_history[symbol].append(tick)
        
        # Keep only last 200 ticks for strategy analysis
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
    
    def analyze_trading_opportunities(self):
        """Analyze all symbols for trading opportunities"""
        for symbol in self.symbols:
            try:
                # Generate new price tick
                tick = self.generate_real_time_price(symbol)
                self.update_price_history(symbol, tick)
                
                # Need sufficient history for analysis
                if len(self.price_history[symbol]) < 100:
                    continue
                
                # Get trading signal
                signal = self.strategy_ensemble.get_ensemble_signal(self.price_history[symbol])
                
                if signal and signal.confidence >= 0.70:
                    self.process_trading_signal(signal, symbol)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
    
    def process_trading_signal(self, signal, symbol: str):
        """Process and potentially execute trading signal"""
        try:
            # Check if we already have a position in this symbol
            symbol_trades = [t for t in self.active_trades.values() if t.symbol == symbol]
            if symbol_trades:
                return  # Skip if already have position
            
            # Validate with risk manager
            current_positions = list(self.active_trades.values())
            allowed, reason, position_size = self.risk_manager.validate_trade_signal(
                signal, self.current_balance, current_positions
            )
            
            if not allowed:
                self.logger.info(f"Trade rejected for {symbol}: {reason}")
                return
            
            # Execute trade
            self.execute_trade(signal, symbol, position_size)
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    def execute_trade(self, signal, symbol: str, position_size: float):
        """Execute a new trade"""
        self.trade_counter += 1
        
        trade = LiveTrade(
            id=self.trade_counter,
            symbol=symbol,
            signal_type=signal.signal.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=position_size,
            entry_time=datetime.utcnow().isoformat(),
            reason=signal.reason
        )
        
        self.active_trades[self.trade_counter] = trade
        
        # Update risk manager
        risk_amount = self.current_balance * 0.015
        self.risk_manager.update_trade_opened(position_size, risk_amount)
        
        self.logger.info(f"🚀 Trade opened: {signal.signal.value} {symbol} "
                        f"@ {signal.entry_price:.5f} Size: {position_size:.2f} "
                        f"Confidence: {signal.confidence:.2f}")
    
    def monitor_active_trades(self):
        """Monitor and manage active trades"""
        for trade_id, trade in list(self.active_trades.items()):
            try:
                current_price = self.current_prices[trade.symbol]
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(trade, current_price)
                
                if should_exit:
                    self.close_trade(trade_id, current_price, exit_reason)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring trade {trade_id}: {e}")
    
    def check_exit_conditions(self, trade: LiveTrade, current_price: float) -> tuple:
        """Check if trade should be closed"""
        entry_time = datetime.fromisoformat(trade.entry_time)
        time_elapsed = (datetime.utcnow() - entry_time).total_seconds() / 60
        
        # Maximum hold time: 30 minutes
        if time_elapsed > 30:
            return True, "TIMEOUT"
        
        # Stop loss and take profit
        if trade.signal_type == "BUY":
            if current_price <= trade.stop_loss:
                return True, "STOP_LOSS"
            elif current_price >= trade.take_profit:
                return True, "TAKE_PROFIT"
        else:  # SELL
            if current_price >= trade.stop_loss:
                return True, "STOP_LOSS"
            elif current_price <= trade.take_profit:
                return True, "TAKE_PROFIT"
        
        # Early profit taking after 5 minutes
        if time_elapsed > 5:
            pip_value = 0.01 if 'JPY' in trade.symbol else 0.0001
            
            if trade.signal_type == "BUY":
                pips_profit = (current_price - trade.entry_price) / pip_value
            else:
                pips_profit = (trade.entry_price - current_price) / pip_value
            
            if pips_profit > 15:  # 15 pips profit
                return True, "EARLY_PROFIT"
        
        return False, ""
    
    def close_trade(self, trade_id: int, exit_price: float, reason: str):
        """Close an active trade"""
        trade = self.active_trades[trade_id]
        
        trade.exit_time = datetime.utcnow().isoformat()
        trade.exit_price = exit_price
        trade.status = reason
        
        # Calculate P&L
        pip_value = 0.01 if 'JPY' in trade.symbol else 0.0001
        
        if trade.signal_type == "BUY":
            pnl_pips = (exit_price - trade.entry_price) / pip_value
        else:
            pnl_pips = (trade.entry_price - exit_price) / pip_value
        
        trade.pnl = pnl_pips * pip_value * trade.volume * 100000
        
        # Update balance
        self.current_balance += trade.pnl
        
        # Update risk manager
        risk_amount = self.initial_balance * 0.015
        self.risk_manager.update_trade_closed(trade.pnl, risk_amount)
        
        # Move to completed trades
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]
        
        # Calculate trade duration
        entry_time = datetime.fromisoformat(trade.entry_time)
        exit_time = datetime.fromisoformat(trade.exit_time)
        duration = (exit_time - entry_time).total_seconds() / 60
        
        self.logger.info(f"💰 Trade closed: {trade.symbol} {reason} "
                        f"P&L: ${trade.pnl:.2f} ({pnl_pips:.1f} pips) "
                        f"Duration: {duration:.1f}min")
    
    def print_status(self):
        """Print current trading status"""
        total_pnl = sum(t.pnl for t in self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])
        total_trades = len(self.completed_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n📊 LIVE TRADING STATUS - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
        print(f"Balance: ${self.current_balance:.2f} (${total_pnl:+.2f})")
        print(f"Active Trades: {len(self.active_trades)}")
        print(f"Completed Trades: {total_trades} (Win Rate: {win_rate:.1f}%)")
        
        if self.active_trades:
            print("Active Positions:")
            for trade in self.active_trades.values():
                current_price = self.current_prices[trade.symbol]
                if trade.signal_type == "BUY":
                    unrealized_pips = (current_price - trade.entry_price) / (0.01 if 'JPY' in trade.symbol else 0.0001)
                else:
                    unrealized_pips = (trade.entry_price - current_price) / (0.01 if 'JPY' in trade.symbol else 0.0001)
                
                entry_time = datetime.fromisoformat(trade.entry_time)
                duration = (datetime.utcnow() - entry_time).total_seconds() / 60
                
                print(f"  {trade.symbol} {trade.signal_type} @ {trade.entry_price:.5f} "
                      f"({unrealized_pips:+.1f} pips, {duration:.0f}min)")
    
    def start_live_simulation(self, duration_minutes: int = 60):
        """Start live trading simulation"""
        self.is_running = True
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        print(f"🚀 Starting Live Trading Simulation")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Strategy: Enhanced 2025 Ensemble")
        print("="*50)
        
        try:
            cycle_count = 0
            while self.is_running and datetime.utcnow() < end_time:
                # Analyze trading opportunities every 10 seconds
                self.analyze_trading_opportunities()
                
                # Monitor active trades
                self.monitor_active_trades()
                
                # Print status every 5 minutes
                cycle_count += 1
                if cycle_count % 30 == 0:  # Every 5 minutes (30 * 10 seconds)
                    self.print_status()
                
                time.sleep(10)  # 10-second intervals
                
        except KeyboardInterrupt:
            print("\n🛑 Live simulation stopped by user")
        finally:
            self.stop_simulation()
    
    def stop_simulation(self):
        """Stop simulation and close all trades"""
        self.is_running = False
        
        # Close all remaining trades
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            current_price = self.current_prices[trade.symbol]
            self.close_trade(trade_id, current_price, "FORCED_CLOSE")
        
        self.print_final_results()
        self.save_simulation_results()
    
    def print_final_results(self):
        """Print final simulation results"""
        total_pnl = sum(t.pnl for t in self.completed_trades)
        winning_trades = [t for t in self.completed_trades if t.pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.pnl < 0]
        
        print("\n" + "="*60)
        print("LIVE TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Return: {(total_pnl/self.initial_balance)*100:+.2f}%")
        print(f"Total Trades: {len(self.completed_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        
        if self.completed_trades:
            win_rate = len(winning_trades) / len(self.completed_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
            
            if winning_trades:
                avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                print(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                print(f"Average Loss: ${avg_loss:.2f}")
        
        print("="*60)
        
        # Performance assessment
        if (total_pnl > 0 and len(self.completed_trades) >= 5 and
            len(winning_trades) / len(self.completed_trades) >= 0.4):
            print("✅ SIMULATION SUCCESSFUL - Strategy ready for live deployment!")
        else:
            print("⚠️  Strategy needs further optimization before live trading")
    
    def save_simulation_results(self):
        """Save simulation results"""
        results = {
            'simulation_timestamp': datetime.utcnow().isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_pnl': self.current_balance - self.initial_balance,
            'total_trades': len(self.completed_trades),
            'strategy_used': 'enhanced_2025_ensemble',
            'trades': [asdict(trade) for trade in self.completed_trades]
        }
        
        filename = f"live_simulation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to {filename}")

def main():
    """Main function to run live trading simulation"""
    print("🤖 MT5 Claude Live Trading Simulator")
    print("⚡ Enhanced 2025 Forex Strategies")
    print()
    
    simulator = LiveTradingSimulator(initial_balance=10000.0)
    
    try:
        # Run 30-minute simulation
        simulator.start_live_simulation(duration_minutes=30)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

if __name__ == "__main__":
    main()