#!/usr/bin/env python3
"""
Ultra-Short-Term Scalping Trading Engine
Optimized for 1-10 minute trades with high win rate focus
"""

import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading
from ultra_scalping_strategies import UltraScalpingEnsemble
from trading_strategies import SignalType, TradingSignal

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScalpTrade:
    ticket: int
    symbol: str
    signal_type: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float = 0.01
    status: str = "OPEN"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    confidence: float = 0.0
    max_profit: float = 0.0  # Track maximum profit reached
    time_in_profit: int = 0  # Seconds spent in profit
    entry_spread: float = 0.0

class SpreadMonitor:
    """Monitors and tracks spread conditions"""
    
    def __init__(self, api_base: str):
        self.api_base = api_base
        self.spread_history = {}
        self.max_spread_ratio = 0.0015  # Max 0.15% spread for scalping
        
    def check_spread(self, symbol: str) -> Tuple[bool, float]:
        """Check if spread is acceptable for scalping"""
        try:
            response = requests.get(f"{self.api_base}/market/symbols/{symbol}")
            if response.status_code == 200:
                data = response.json()
                if 'ask' in data and 'bid' in data:
                    spread = data['ask'] - data['bid']
                    spread_ratio = spread / data['bid']
                    
                    # Track spread history
                    if symbol not in self.spread_history:
                        self.spread_history[symbol] = deque(maxlen=10)
                    self.spread_history[symbol].append(spread_ratio)
                    
                    # Check if current spread is acceptable
                    if spread_ratio <= self.max_spread_ratio:
                        return True, spread
                    else:
                        logger.debug(f"Spread too wide for {symbol}: {spread_ratio:.4%}")
                        return False, spread
        except Exception as e:
            logger.error(f"Error checking spread for {symbol}: {e}")
        
        return False, 0.0

class ExitManager:
    """Manages trade exits for optimal profit taking"""
    
    def __init__(self):
        self.profit_threshold = 5.0  # Start monitoring after $5 profit
        self.trailing_distance = 3.0  # Trail by $3
        self.time_based_exit = 600  # Exit after 10 minutes max
        self.quick_profit_exit = 10.0  # Take quick profit at $10
        
    def should_exit_trade(self, trade: ScalpTrade, current_price: float, 
                         elapsed_seconds: int) -> Tuple[bool, str]:
        """Determine if trade should be exited"""
        
        # Calculate current profit
        if trade.signal_type == "BUY":
            current_profit = (current_price - trade.entry_price) * 10000  # Approximate USD profit
        else:
            current_profit = (trade.entry_price - current_price) * 10000
        
        # Quick profit taking
        if current_profit >= self.quick_profit_exit:
            return True, "Quick profit target reached"
        
        # Time-based exit
        if elapsed_seconds >= self.time_based_exit:
            if current_profit > 0:
                return True, "Time limit reached with profit"
            elif current_profit < -5:
                return True, "Time limit reached with loss"
        
        # Dynamic trailing stop
        if current_profit > self.profit_threshold:
            if trade.max_profit - current_profit > self.trailing_distance:
                return True, "Trailing stop triggered"
        
        # Cut losses quickly
        if current_profit < -8:  # $8 loss
            return True, "Stop loss optimization"
        
        return False, ""

class UltraScalpingEngine:
    """Ultra-short-term scalping engine for 1-10 minute trades"""
    
    def __init__(self, api_base="http://172.28.144.1:8000"):
        self.api_base = api_base
        self.strategy_ensemble = UltraScalpingEnsemble()
        self.spread_monitor = SpreadMonitor(api_base)
        self.exit_manager = ExitManager()
        
        # Optimized for scalping
        self.symbols = []  # Will be dynamically selected
        self.active_trades = {}
        self.trade_history = []
        self.running = False
        
        # Scalping parameters
        self.max_concurrent_trades = 1  # Focus on one trade at a time
        self.min_time_between_trades = 30  # 30 seconds between trades
        self.analysis_interval = 5  # Check every 5 seconds
        # Use confirmed tradable symbols with best liquidity
        self.preferred_symbols = ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#", 
                                 "GBPJPY#", "USDCAD#", "EURGBP#"]
        
        # Performance tracking
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'total_pips': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_trade_duration': 0.0,
            'current_streak': 0
        }
        
        # Session tracking
        self.session_start_balance = None
        self.last_trade_time = {}
        
    def select_best_symbol(self) -> Optional[str]:
        """Select the best symbol for current market conditions"""
        best_symbol = None
        best_score = 0
        
        for symbol in self.preferred_symbols:
            try:
                # Check spread first
                spread_ok, spread = self.spread_monitor.check_spread(symbol)
                if not spread_ok:
                    continue
                
                # Get recent data
                data = self.get_market_data(symbol, "M1", 30)
                if not data:
                    continue
                
                df = pd.DataFrame(data)
                
                # Calculate volatility score
                volatility = df['close'].pct_change().std()
                
                # Calculate momentum score
                momentum = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                
                # Calculate volume score
                volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
                
                # Combined score (higher is better)
                score = (volatility * 1000) + (momentum * 100) + (volume_ratio * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
                    
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
        
        return best_symbol
    
    def check_api_connection(self):
        """Verify API connection and MT5 status"""
        try:
            response = requests.get(f"{self.api_base}/status/mt5")
            if response.status_code == 200:
                status = response.json()
                if status.get('connected') and status.get('trade_allowed'):
                    logger.info("✅ MT5 API connected and trading allowed")
                    return True
                else:
                    logger.error("❌ MT5 not ready for trading")
                    return False
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
    
    def get_account_info(self):
        """Get current account information"""
        try:
            response = requests.get(f"{self.api_base}/account/")
            if response.status_code == 200:
                account = response.json()
                if self.session_start_balance is None:
                    self.session_start_balance = account['balance']
                
                session_profit = account['balance'] - self.session_start_balance
                logger.info(f"Balance: {account['balance']} | Session P/L: {session_profit:.2f} {account['currency']}")
                return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
        return None
    
    def get_market_data(self, symbol: str, timeframe: str = "M1", count: int = 30):
        """Get market data optimized for scalping"""
        url = f"{self.api_base}/market/history"
        data = {"symbol": symbol, "timeframe": timeframe, "count": count}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
        return None
    
    def place_scalp_order(self, signal: TradingSignal, symbol: str) -> Optional[ScalpTrade]:
        """Place a scalping order with tight stops"""
        # Final spread check before entry
        spread_ok, current_spread = self.spread_monitor.check_spread(symbol)
        if not spread_ok:
            logger.info(f"Order cancelled - spread too wide for {symbol}")
            return None
        
        if signal.signal == SignalType.BUY:
            order_type = 0  # BUY
        else:
            order_type = 1  # SELL
        
        order_data = {
            "action": 1,  # DEAL (market order)
            "symbol": symbol,
            "volume": 0.01,  # Fixed lot size per CLAUDE.md
            "type": order_type,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "comment": f"Scalp {signal.confidence:.0%}"
        }
        
        try:
            response = requests.post(f"{self.api_base}/trading/orders", json=order_data)
            if response.status_code == 201:
                result = response.json()
                logger.info(f"✅ Scalp order placed: {result}")
                
                # Create trade record
                trade = ScalpTrade(
                    ticket=result.get('order', 0),
                    symbol=symbol,
                    signal_type=signal.signal.value,
                    entry_time=datetime.now().isoformat(),
                    entry_price=result.get('price', signal.entry_price),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence=signal.confidence,
                    entry_spread=current_spread
                )
                
                self.active_trades[trade.ticket] = trade
                self.stats['total_trades'] += 1
                self.last_trade_time[symbol] = time.time()
                
                logger.info(f"🎯 Scalp opened: {symbol} {signal.signal.value} @ {trade.entry_price}")
                logger.info(f"   SL: {trade.stop_loss} | TP: {trade.take_profit} | Spread: {current_spread:.5f}")
                
                return trade
            else:
                logger.error(f"❌ Order failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
        return None
    
    def manage_open_trades(self):
        """Actively manage open trades for optimal exits"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                positions = response.json()
                current_tickets = {pos['ticket'] for pos in positions}
                
                # Check closed trades
                for ticket, trade in list(self.active_trades.items()):
                    if ticket not in current_tickets:
                        # Trade was closed
                        self._process_closed_trade(trade)
                
                # Manage open trades
                for position in positions:
                    ticket = position['ticket']
                    if ticket in self.active_trades:
                        trade = self.active_trades[ticket]
                        current_price = position.get('price_current', trade.entry_price)
                        current_profit = position.get('profit', 0)
                        
                        # Update max profit
                        trade.max_profit = max(trade.max_profit, current_profit)
                        if current_profit > 0:
                            trade.time_in_profit += self.analysis_interval
                        
                        # Calculate elapsed time
                        entry_time = datetime.fromisoformat(trade.entry_time)
                        elapsed = (datetime.now() - entry_time).total_seconds()
                        
                        # Check exit conditions
                        should_exit, reason = self.exit_manager.should_exit_trade(
                            trade, current_price, elapsed
                        )
                        
                        if should_exit:
                            logger.info(f"🎯 Closing trade {ticket}: {reason}")
                            self._close_position(ticket)
                        elif elapsed > 60 and current_profit > 3:  # After 1 minute with profit
                            # Move stop loss to breakeven
                            self._move_to_breakeven(trade, position)
                
        except Exception as e:
            logger.error(f"Error managing trades: {e}")
    
    def _process_closed_trade(self, trade: ScalpTrade):
        """Process a closed trade and update statistics"""
        trade.status = "CLOSED"
        trade.exit_time = datetime.now().isoformat()
        
        # Calculate duration
        entry_time = datetime.fromisoformat(trade.entry_time)
        exit_time = datetime.fromisoformat(trade.exit_time)
        duration = (exit_time - entry_time).total_seconds()
        
        # Update stats
        if trade.profit and trade.profit > 0:
            self.stats['winning_trades'] += 1
            self.stats['current_streak'] = max(0, self.stats['current_streak']) + 1
        else:
            self.stats['current_streak'] = min(0, self.stats['current_streak']) - 1
        
        if trade.profit:
            self.stats['total_profit'] += trade.profit
            self.stats['best_trade'] = max(self.stats['best_trade'], trade.profit)
            self.stats['worst_trade'] = min(self.stats['worst_trade'], trade.profit)
        
        # Calculate pips
        pip_value = 0.0001 if "JPY" not in trade.symbol else 0.01
        if trade.exit_price and trade.profit:
            if trade.signal_type == "BUY":
                pips = (trade.exit_price - trade.entry_price) / pip_value
            else:
                pips = (trade.entry_price - trade.exit_price) / pip_value
            self.stats['total_pips'] += pips
        
        self.trade_history.append(trade)
        del self.active_trades[trade.ticket]
        
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
        logger.info(f"📊 Trade closed: {trade.symbol} | P/L: ${trade.profit:.2f} | "
                   f"Duration: {duration:.0f}s | Win Rate: {win_rate:.1f}%")
    
    def _close_position(self, ticket: int):
        """Close a specific position"""
        try:
            close_data = {"ticket": ticket}
            response = requests.delete(f"{self.api_base}/trading/positions/{ticket}", json=close_data)
            if response.status_code == 200:
                logger.info(f"Position {ticket} closed successfully")
            else:
                logger.error(f"Failed to close position {ticket}: {response.text}")
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _move_to_breakeven(self, trade: ScalpTrade, position: Dict):
        """Move stop loss to breakeven for risk-free trade"""
        if trade.signal_type == "BUY" and trade.stop_loss < trade.entry_price:
            new_sl = trade.entry_price + 0.00002  # Small buffer for spread
            self._modify_position(position['ticket'], new_sl, trade.take_profit)
            trade.stop_loss = new_sl
            logger.info(f"🔒 Moved to breakeven: {trade.symbol}")
        elif trade.signal_type == "SELL" and trade.stop_loss > trade.entry_price:
            new_sl = trade.entry_price - 0.00002
            self._modify_position(position['ticket'], new_sl, trade.take_profit)
            trade.stop_loss = new_sl
            logger.info(f"🔒 Moved to breakeven: {trade.symbol}")
    
    def _modify_position(self, ticket: int, sl: float, tp: float):
        """Modify an existing position"""
        modify_data = {"sl": sl, "tp": tp}
        
        try:
            response = requests.patch(f"{self.api_base}/trading/positions/{ticket}", json=modify_data)
            if response.status_code != 200:
                logger.error(f"Failed to modify position {ticket}: {response.text}")
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
    
    def analyze_and_trade(self):
        """Main trading logic for scalping"""
        # Skip if we have an active trade
        if self.active_trades:
            return
        
        # Select best symbol
        symbol = self.select_best_symbol()
        if not symbol:
            logger.debug("No suitable symbol found for trading")
            return
        
        # Check cooldown
        if symbol in self.last_trade_time:
            time_since_last = time.time() - self.last_trade_time[symbol]
            if time_since_last < self.min_time_between_trades:
                return
        
        # Get fresh market data
        market_data = self.get_market_data(symbol, "M1", 30)
        if not market_data:
            return
        
        try:
            # Get scalping signal
            signal = self.strategy_ensemble.get_ensemble_signal(market_data)
            
            if signal and signal.confidence >= 0.76:  # High confidence for scalping
                logger.info(f"💡 Scalping opportunity: {symbol} {signal.signal.value} "
                           f"(confidence: {signal.confidence:.2f})")
                
                # Place the trade
                trade = self.place_scalp_order(signal, symbol)
                
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
    
    def run_scalping_session(self, duration_minutes=300):
        """Run ultra-short-term scalping session"""
        if not self.check_api_connection():
            logger.error("Cannot start trading - API connection failed")
            return
        
        account = self.get_account_info()
        if not account:
            logger.error("Cannot start trading - account info unavailable")
            return
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        logger.info(f"🚀 Starting Ultra-Scalping Session")
        logger.info(f"Duration: {duration_minutes} minutes | Target: 1-10 minute trades")
        logger.info(f"Symbols: {', '.join(self.preferred_symbols)}")
        
        try:
            cycle = 0
            while self.running and datetime.now() < end_time:
                cycle += 1
                
                # Manage existing trades first
                self.manage_open_trades()
                
                # Look for new opportunities
                self.analyze_and_trade()
                
                # Brief pause
                time.sleep(self.analysis_interval)
                
                # Status update every minute
                if cycle % 12 == 0:  # 12 * 5 seconds = 1 minute
                    self._log_session_status()
                
                # Account check every 5 minutes
                if cycle % 60 == 0:  # 60 * 5 seconds = 5 minutes
                    self.get_account_info()
                    
        except KeyboardInterrupt:
            logger.info("🛑 Scalping session interrupted by user")
        except Exception as e:
            logger.error(f"❌ Session error: {e}")
        finally:
            self.running = False
            
        # Final cleanup
        self.manage_open_trades()
        self.print_session_summary()
    
    def _log_session_status(self):
        """Log current session status"""
        if self.stats['total_trades'] > 0:
            win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            avg_duration = sum([(datetime.fromisoformat(t.exit_time) - datetime.fromisoformat(t.entry_time)).total_seconds() 
                               for t in self.trade_history if t.exit_time]) / max(len(self.trade_history), 1)
            
            logger.info(f"📈 Status: Trades: {self.stats['total_trades']} | "
                       f"Win Rate: {win_rate:.1f}% | P/L: ${self.stats['total_profit']:.2f} | "
                       f"Pips: {self.stats['total_pips']:.1f} | Avg Duration: {avg_duration:.0f}s")
    
    def print_session_summary(self):
        """Print detailed scalping session summary"""
        print("\n" + "="*70)
        print("ULTRA-SCALPING SESSION SUMMARY")
        print("="*70)
        
        if self.stats['total_trades'] > 0:
            win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            # Calculate average trade duration
            durations = []
            for trade in self.trade_history:
                if trade.exit_time:
                    duration = (datetime.fromisoformat(trade.exit_time) - 
                              datetime.fromisoformat(trade.entry_time)).total_seconds()
                    durations.append(duration)
            
            avg_duration = np.mean(durations) if durations else 0
            
            print(f"Total Trades: {self.stats['total_trades']}")
            print(f"Winning Trades: {self.stats['winning_trades']}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total P/L: ${self.stats['total_profit']:.2f}")
            print(f"Total Pips: {self.stats['total_pips']:.1f}")
            print(f"Best Trade: ${self.stats['best_trade']:.2f}")
            print(f"Worst Trade: ${self.stats['worst_trade']:.2f}")
            print(f"Avg Trade Duration: {avg_duration:.0f} seconds ({avg_duration/60:.1f} minutes)")
            print(f"Current Streak: {self.stats['current_streak']}")
            
            if self.session_start_balance:
                session_return = ((self.stats['total_profit'] / self.session_start_balance) * 100)
                print(f"Session Return: {session_return:.2f}%")
            
            # Symbol performance
            print("\nPerformance by Symbol:")
            symbol_stats = {}
            for trade in self.trade_history:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {'count': 0, 'profit': 0, 'wins': 0}
                symbol_stats[trade.symbol]['count'] += 1
                if trade.profit:
                    symbol_stats[trade.symbol]['profit'] += trade.profit
                    if trade.profit > 0:
                        symbol_stats[trade.symbol]['wins'] += 1
            
            for symbol, stats in symbol_stats.items():
                sym_win_rate = (stats['wins'] / stats['count']) * 100
                print(f"  {symbol}: {stats['count']} trades, {sym_win_rate:.0f}% win rate, ${stats['profit']:.2f}")
        else:
            print("No trades executed in this session")
        
        print("="*70)
    
    def stop_trading(self):
        """Stop the scalping session"""
        self.running = False
        logger.info("🛑 Scalping engine stopped")

def main():
    """Main execution function"""
    engine = UltraScalpingEngine()
    
    logger.info("🎯 Starting Ultra-Short-Term Forex Scalping System")
    logger.info("Target: 1-10 minute trades with high win rate")
    logger.info("Strategy: Multi-strategy ensemble with tight risk management")
    
    try:
        # Run for 5 hours as specified
        engine.run_scalping_session(duration_minutes=300)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    logger.info("🏁 Scalping session completed")

if __name__ == "__main__":
    main()