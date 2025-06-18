import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from trading_strategies import StrategyEnsemble, SignalType
from improved_strategies import ImprovedStrategyEnsemble
from ultra_scalping_strategies import UltraScalpingEnsemble
from dataclasses import dataclass
from symbol_manager import SymbolManager
import pytz
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LiveTrade:
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

class LiveTradingEngine:
    def __init__(self, api_base="http://172.28.144.1:8000", use_scalping=True):
        self.api_base = api_base
        # Use ultra-scalping strategies for 1-10 minute trades
        if use_scalping:
            self.strategy_ensemble = UltraScalpingEnsemble()
            self.timeframe = "M1"  # 1-minute timeframe for scalping
            self.analysis_interval = 10  # Check every 10 seconds
            self.min_confidence = 0.76  # Higher threshold for scalping
        else:
            self.strategy_ensemble = ImprovedStrategyEnsemble()
            self.timeframe = "M5"
            self.analysis_interval = 15
            self.min_confidence = 0.72
            
        # Initialize symbol manager for dynamic symbol selection
        self.symbol_manager = SymbolManager(api_base)
        
        # Get ALL tradable symbols for parallel monitoring
        try:
            all_symbols = self.symbol_manager.fetch_tradable_symbols()
            self.symbols = all_symbols if all_symbols else ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#"]
        except:
            self.symbols = ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#"]
            
        logger.info(f"Monitoring {len(self.symbols)} symbols: {self.symbols[:5]}..." if len(self.symbols) > 5 else f"Monitoring symbols: {self.symbols}")
        
        self.active_trades = {}
        self.max_concurrent_trades = 3  # Allow more concurrent trades across different symbols
        self.max_trades_per_symbol = 1  # Max 1 trade per symbol
        self.trade_history = []
        self.running = False
        self.min_time_between_trades = 30  # 30 seconds between trades for same symbol
        self.last_trade_time = {}
        self.max_trade_duration = 600  # Exit after 10 minutes max
        self.symbol_refresh_counter = 0  # Counter to refresh symbols periodically
        
        # Threading components
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.analysis_queue = queue.Queue()
        self.symbol_locks = {symbol: threading.Lock() for symbol in self.symbols}
        self.trade_lock = threading.Lock()  # Global lock for trade operations
        
        # Trading hours configuration (avoid 3 AM - 8 AM due to wide spreads)
        self.no_trade_start_hour = 3   # 3 AM
        self.no_trade_end_hour = 8     # 8 AM
        self.timezone = pytz.timezone('Asia/Tokyo')  # Adjust to your broker's timezone
        
        # Stop loss configuration for safety
        self.min_sl_pips = 5      # Minimum 5 pips stop loss
        self.max_sl_pips = 20     # Maximum 20 pips stop loss for scalping
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.use_atr_multiplier = 1.5  # ATR multiplier for dynamic SL
        
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
                logger.info(f"Account Balance: {account['balance']} {account['currency']}")
                logger.info(f"Free Margin: {account['margin_free']} {account['currency']}")
                return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
        return None
    
    def is_trading_allowed(self) -> bool:
        """Check if current time is within allowed trading hours"""
        current_time = datetime.now(self.timezone)
        current_hour = current_time.hour
        
        # Check if we're in the no-trade window (3 AM - 8 AM)
        if self.no_trade_start_hour <= current_hour < self.no_trade_end_hour:
            logger.info(f"🚫 Trading paused - Current time {current_hour}:00 is outside trading hours "
                       f"(avoiding {self.no_trade_start_hour}:00-{self.no_trade_end_hour}:00 due to wide spreads)")
            return False
        
        # Also avoid weekends (Saturday and Sunday)
        if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            logger.info("🚫 Trading paused - Market closed (weekend)")
            return False
            
        return True
    
    def calculate_safe_stop_loss(self, symbol: str, signal_type: str, entry_price: float, 
                                market_data: List[Dict]) -> Tuple[float, float, str]:
        """
        Calculate safe stop loss and take profit levels
        Returns: (stop_loss, take_profit, calculation_method)
        """
        # Determine pip value based on symbol
        if "JPY" in symbol:
            pip_value = 0.01
            pip_digits = 2
        else:
            pip_value = 0.0001
            pip_digits = 4
        
        # Calculate ATR-based stop loss
        df = pd.DataFrame(market_data)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        # Method 1: ATR-based stop loss
        atr_sl_distance = atr * self.use_atr_multiplier
        atr_sl_pips = atr_sl_distance / pip_value
        atr_sl_pips = max(self.min_sl_pips, min(atr_sl_pips, self.max_sl_pips))
        
        # Method 2: Recent swing points (with limits)
        recent_data = df.tail(20)
        if signal_type == "BUY":
            swing_low = recent_data['low'].min()
            swing_sl_distance = entry_price - swing_low
        else:
            swing_high = recent_data['high'].max()
            swing_sl_distance = swing_high - entry_price
        
        # Ensure swing-based stop loss is within reasonable limits
        swing_sl_pips = swing_sl_distance / pip_value
        swing_sl_pips = max(self.min_sl_pips, min(swing_sl_pips, self.max_sl_pips))
        
        # Method 3: Volatility-based stop loss
        volatility = df['close'].pct_change().std() * np.sqrt(12)  # 5-min bars
        vol_sl_pips = max(self.min_sl_pips, min(volatility * 1000, self.max_sl_pips))
        
        # Choose the most appropriate stop loss
        sl_pips_options = [
            (atr_sl_pips, "ATR"),
            (swing_sl_pips, "Swing"),
            (vol_sl_pips, "Volatility")
        ]
        
        # Filter out invalid options and choose the best
        valid_options = [(pips, method) for pips, method in sl_pips_options 
                        if self.min_sl_pips <= pips <= self.max_sl_pips]
        
        if not valid_options:
            # If no valid option, use default
            final_sl_pips = self.min_sl_pips * 2  # 10 pips default
            method = "Default"
        else:
            # Choose median value for balanced approach
            valid_options.sort(key=lambda x: x[0])
            final_sl_pips, method = valid_options[len(valid_options)//2]
        
        # Calculate actual stop loss price
        if signal_type == "BUY":
            stop_loss = entry_price - (final_sl_pips * pip_value)
            # For scalping, use 1.5:1 risk-reward ratio
            take_profit = entry_price + (final_sl_pips * pip_value * 1.5)
        else:
            stop_loss = entry_price + (final_sl_pips * pip_value)
            take_profit = entry_price - (final_sl_pips * pip_value * 1.5)
        
        # Round to appropriate decimal places
        stop_loss = round(stop_loss, pip_digits + 1)
        take_profit = round(take_profit, pip_digits + 1)
        
        logger.debug(f"Stop loss calculation for {symbol}: {final_sl_pips:.1f} pips ({method})")
        
        return stop_loss, take_profit, f"{method} ({final_sl_pips:.1f} pips)"
        
    def get_market_data(self, symbol, timeframe=None, count=100):
        """Get current market data for analysis"""
        if timeframe is None:
            timeframe = self.timeframe
        url = f"{self.api_base}/market/history"
        data = {"symbol": symbol, "timeframe": timeframe, "count": count}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
        return None
        
    def place_order(self, signal, symbol, market_data):
        """Place a market order with safe stop loss calculation"""
        if signal.signal == SignalType.BUY:
            order_type = 0  # BUY
            signal_type = "BUY"
        else:
            order_type = 1  # SELL
            signal_type = "SELL"
        
        # Calculate safe stop loss and take profit
        safe_sl, safe_tp, sl_method = self.calculate_safe_stop_loss(
            symbol, signal_type, signal.entry_price, market_data
        )
        
        # Verify stop loss is set
        if safe_sl is None or safe_sl == 0:
            logger.error(f"❌ Cannot place order - stop loss calculation failed")
            return None
            
        order_data = {
            "action": 1,  # DEAL (market order)
            "symbol": symbol,
            "volume": 0.01,  # Fixed lot size
            "type": order_type,
            # Don't specify type_filling - let API auto-determine
            "sl": safe_sl,
            "tp": safe_tp,
            "comment": f"SL: {sl_method}"
        }
        
        logger.info(f"📊 Order details: {signal_type} {symbol} | SL: {safe_sl} | TP: {safe_tp} | Method: {sl_method}")
        logger.debug(f"Order data: {order_data}")
        
        try:
            response = requests.post(f"{self.api_base}/trading/orders", json=order_data)
            if response.status_code == 201:
                result = response.json()
                logger.info(f"✅ Order placed successfully: {result}")
                
                # Create trade record with verified stop loss
                trade = LiveTrade(
                    ticket=result.get('order', 0),
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_time=datetime.now().isoformat(),
                    entry_price=result.get('price', signal.entry_price),
                    stop_loss=safe_sl,  # Use calculated safe stop loss
                    take_profit=safe_tp  # Use calculated take profit
                )
                
                self.active_trades[trade.ticket] = trade
                logger.info(f"✅ Trade opened: {symbol} {signal_type} at {trade.entry_price}")
                logger.info(f"   Protected by stop loss: {safe_sl} ({sl_method})")
                return trade
            else:
                error_detail = response.text
                logger.error(f"❌ Order failed: {error_detail}")
                
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
        return None
        
    def check_open_positions(self):
        """Check and update status of open positions with active management (thread-safe)"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                positions = response.json()
                current_tickets = {pos['ticket'] for pos in positions}
                
                # Check for closed trades (thread-safe)
                with self.trade_lock:
                    for ticket, trade in list(self.active_trades.items()):
                        if ticket not in current_tickets:
                            # Trade was closed
                            trade.status = "CLOSED"
                            trade.exit_time = datetime.now().isoformat()
                            self.trade_history.append(trade)
                            del self.active_trades[ticket]
                            logger.info(f"🔄 Trade {ticket} ({trade.symbol}) closed")
                
                # Update and manage open trades
                for position in positions:
                    ticket = position['ticket']
                    if ticket in self.active_trades:
                        trade = self.active_trades[ticket]
                        current_profit = position.get('profit', 0)
                        trade.profit = current_profit
                        
                        # CRITICAL: Verify stop loss is set
                        self._verify_stop_loss(trade, position)
                        
                        # Calculate trade duration
                        entry_time = datetime.fromisoformat(trade.entry_time)
                        duration = (datetime.now() - entry_time).total_seconds()
                        
                        # Scalping exit rules
                        if self.timeframe == "M1":
                            # Quick profit taking
                            if current_profit >= 8:  # $8 profit
                                logger.info(f"💰 Taking quick profit on {trade.symbol}: ${current_profit:.2f}")
                                self._close_position(ticket)
                            # Time-based exit
                            elif duration > self.max_trade_duration:
                                if current_profit > 0:
                                    logger.info(f"⏰ Time exit with profit on {trade.symbol}: ${current_profit:.2f}")
                                    self._close_position(ticket)
                                elif current_profit < -5:
                                    logger.info(f"⏰ Time exit with loss on {trade.symbol}: ${current_profit:.2f}")
                                    self._close_position(ticket)
                            # Move to breakeven after 1 minute with $3+ profit
                            elif duration > 60 and current_profit >= 3:
                                self._move_to_breakeven(trade, position)
                        
                logger.debug(f"📈 Active trades: {len(self.active_trades)}")
                return positions
        except Exception as e:
            logger.error(f"Failed to check positions: {e}")
        return []
    
    def _close_position(self, ticket: int):
        """Close a specific position"""
        try:
            close_data = {"ticket": ticket}
            response = requests.delete(f"{self.api_base}/trading/positions/{ticket}", json=close_data)
            if response.status_code == 200:
                logger.debug(f"Position {ticket} closed successfully")
            else:
                logger.error(f"Failed to close position {ticket}: {response.text}")
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _move_to_breakeven(self, trade: LiveTrade, position: Dict):
        """Move stop loss to breakeven for risk-free trade"""
        pip_value = 0.01 if "JPY" in trade.symbol else 0.0001
        buffer = pip_value * 0.2  # 0.2 pip buffer for spread
        
        if trade.signal_type == "BUY" and trade.stop_loss < trade.entry_price:
            new_sl = trade.entry_price + buffer
            self._modify_position(position['ticket'], new_sl, trade.take_profit)
            trade.stop_loss = new_sl
            logger.info(f"🔒 Moved to breakeven: {trade.symbol} (SL: {new_sl})")
        elif trade.signal_type == "SELL" and trade.stop_loss > trade.entry_price:
            new_sl = trade.entry_price - buffer
            self._modify_position(position['ticket'], new_sl, trade.take_profit)
            trade.stop_loss = new_sl
            logger.info(f"🔒 Moved to breakeven: {trade.symbol} (SL: {new_sl})")
    
    def _verify_stop_loss(self, trade: LiveTrade, position: Dict):
        """Verify stop loss is properly set and fix if needed"""
        position_sl = position.get('sl', 0)
        
        # Check if stop loss is missing or zero
        if position_sl == 0 or position_sl is None:
            logger.warning(f"⚠️ Missing stop loss for {trade.symbol}, setting it now!")
            self._modify_position(position['ticket'], trade.stop_loss, trade.take_profit)
            return False
        
        # Verify stop loss matches our records (within small tolerance)
        tolerance = 0.00005
        if abs(position_sl - trade.stop_loss) > tolerance:
            logger.warning(f"⚠️ Stop loss mismatch for {trade.symbol}: "
                         f"Expected {trade.stop_loss}, Found {position_sl}")
            # Update to our calculated stop loss
            self._modify_position(position['ticket'], trade.stop_loss, trade.take_profit)
            return False
            
        return True
    
    def _modify_position(self, ticket: int, sl: float, tp: float):
        """Modify position parameters"""
        modify_data = {"sl": sl, "tp": tp}  # Remove ticket from body - it's in the URL
        try:
            response = requests.patch(f"{self.api_base}/trading/positions/{ticket}", json=modify_data)
            if response.status_code != 200:
                logger.error(f"Failed to modify position {ticket}: {response.text}")
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
        
    def analyze_and_trade(self, symbol):
        """Analyze market and place trades if signal is strong (thread-safe)"""
        # Check if trading is allowed at current time
        if not self.is_trading_allowed():
            return
        
        # Use symbol-specific lock to prevent concurrent analysis of same symbol
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()
            
        with self.symbol_locks[symbol]:
            # Check global trade limit
            with self.trade_lock:
                if len(self.active_trades) >= self.max_concurrent_trades:
                    return
                
                # Check if we already have a position in this symbol
                symbol_positions = [t for t in self.active_trades.values() if t.symbol == symbol]
                if len(symbol_positions) >= self.max_trades_per_symbol:
                    return
            
            # Check if we traded this symbol recently
            if symbol in self.last_trade_time:
                time_since_last_trade = time.time() - self.last_trade_time[symbol]
                if time_since_last_trade < self.min_time_between_trades:
                    return
            
        # Get market data
        market_data = self.get_market_data(symbol, self.timeframe, 50 if self.timeframe == "M1" else 100)
        if not market_data:
            return
        
        # For non-scalping, also get M1 data for confirmation
        market_data_m1 = None
        if self.timeframe != "M1":
            market_data_m1 = self.get_market_data(symbol, "M1", 20)
            
        try:
            # Get trading signal
            signal = self.strategy_ensemble.get_ensemble_signal(market_data)
            
            if signal and signal.confidence >= self.min_confidence:
                # Confirm with M1 timeframe
                if market_data_m1:
                    m1_signal = self.strategy_ensemble.get_ensemble_signal(market_data_m1)
                    if m1_signal and m1_signal.signal != signal.signal:
                        logger.info(f"Signal conflict between M5 and M1 for {symbol}, skipping")
                        return
                
                logger.info(f"🎯 Strong signal detected: {symbol} {signal.signal.value} "
                           f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
                
                # Place the trade with safe stop loss (thread-safe)
                with self.trade_lock:
                    if len(self.active_trades) < self.max_concurrent_trades:
                        trade = self.place_order(signal, symbol, market_data)
                        if trade:
                            self.last_trade_time[symbol] = time.time()
                    
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
    
    def analyze_symbols_parallel(self):
        """Analyze all symbols in parallel using thread pool"""
        futures = []
        
        # Submit analysis tasks for all symbols
        for symbol in self.symbols:
            future = self.thread_pool.submit(self.analyze_and_trade, symbol)
            futures.append((future, symbol))
        
        # Wait for all analyses to complete (with timeout)
        completed = 0
        for future, symbol in futures:
            try:
                future.result(timeout=10)  # 10 second timeout per symbol
                completed += 1
            except Exception as e:
                logger.error(f"Parallel analysis failed for {symbol}: {e}")
        
        logger.debug(f"Completed parallel analysis for {completed}/{len(self.symbols)} symbols")
            
    def run_trading_session(self, duration_minutes=None):
        """Run a continuous 24/7 trading session"""
        if not self.check_api_connection():
            logger.error("Cannot start trading - API connection failed")
            return
            
        account = self.get_account_info()
        if not account:
            logger.error("Cannot start trading - account info unavailable")
            return
            
        self.running = True
        start_time = datetime.now()
        
        if duration_minutes:
            end_time = start_time + timedelta(minutes=duration_minutes)
            logger.info(f"🚀 Starting live trading session for {duration_minutes} minutes")
        else:
            end_time = None
            logger.info("🚀 Starting continuous 24/7 trading session")
            logger.info(f"Trading hours: Avoiding {self.no_trade_start_hour}:00-{self.no_trade_end_hour}:00 {self.timezone}")
            
        logger.info(f"Trading symbols: {self.symbols}")
        
        try:
            while self.running and (end_time is None or datetime.now() < end_time):
                # Check existing positions
                self.check_open_positions()
                
                # Refresh symbol list every 60 cycles (10 minutes at 10-second intervals)
                self.symbol_refresh_counter += 1
                if self.symbol_refresh_counter >= 60:
                    self.symbol_refresh_counter = 0
                    try:
                        new_symbols = self.symbol_manager.fetch_tradable_symbols()
                        if new_symbols and new_symbols != self.symbols:
                            self.symbols = new_symbols
                            # Update symbol locks for new symbols
                            for symbol in new_symbols:
                                if symbol not in self.symbol_locks:
                                    self.symbol_locks[symbol] = threading.Lock()
                            logger.info(f"🔄 Updated to {len(self.symbols)} tradable symbols")
                    except Exception as e:
                        logger.error(f"Failed to refresh symbols: {e}")
                
                # Analyze all symbols in parallel
                if self.is_trading_allowed():
                    start_analysis = time.time()
                    self.analyze_symbols_parallel()
                    analysis_time = time.time() - start_analysis
                    if analysis_time > 5:
                        logger.warning(f"Parallel analysis took {analysis_time:.1f}s for {len(self.symbols)} symbols")
                
                # Wait before next analysis cycle
                time.sleep(self.analysis_interval)
                
                # Log status every 5 minutes
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    if duration_minutes:
                        logger.info(f"⏰ Session running: {elapsed:.1f}/{duration_minutes} minutes")
                    else:
                        hours = elapsed / 60
                        logger.info(f"⏰ Continuous session running: {hours:.1f} hours | "
                                   f"Trades: {len(self.trade_history)} | Active: {len(self.active_trades)}")
                    
                # Add periodic connection check for 24/7 operation
                if int(elapsed) % 30 == 0 and elapsed > 0:  # Every 30 minutes
                    if not self.check_api_connection():
                        logger.warning("⚠️ API connection lost, waiting 60 seconds before retry...")
                        time.sleep(60)
                        continue
                    
        except KeyboardInterrupt:
            logger.info("🛑 Trading session interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Trading session error: {e}")
            if end_time is None:  # Continuous mode
                logger.info("🔄 Restarting in 60 seconds...")
                time.sleep(60)
                # Recursive restart for 24/7 operation
                self.run_trading_session(duration_minutes=None)
            else:
                self.running = False
            
        # Final status check
        final_positions = self.check_open_positions()
        
        logger.info("📊 Trading session completed")
        logger.info(f"Active trades: {len(self.active_trades)}")
        logger.info(f"Completed trades: {len(self.trade_history)}")
        
        # Summary
        self.print_session_summary()
        
    def print_session_summary(self):
        """Print trading session summary with symbol breakdown"""
        print("\n" + "="*70)
        print("PARALLEL TRADING SESSION SUMMARY")
        print("="*70)
        
        total_trades = len(self.trade_history) + len(self.active_trades)
        completed_trades = len(self.trade_history)
        
        if completed_trades > 0:
            profits = [trade.profit for trade in self.trade_history if trade.profit is not None]
            total_profit = sum(profits) if profits else 0
            winning_trades = len([p for p in profits if p > 0])
            win_rate = (winning_trades / completed_trades) * 100
            
            print(f"Total Symbols Monitored: {len(self.symbols)}")
            print(f"Total Trades: {total_trades}")
            print(f"Completed Trades: {completed_trades}")
            print(f"Active Trades: {len(self.active_trades)}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total Profit: ${total_profit:.2f}")
            
            # Symbol breakdown
            symbol_stats = {}
            for trade in self.trade_history:
                symbol = trade.symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'trades': 0, 'wins': 0, 'profit': 0}
                symbol_stats[symbol]['trades'] += 1
                if trade.profit and trade.profit > 0:
                    symbol_stats[symbol]['wins'] += 1
                if trade.profit:
                    symbol_stats[symbol]['profit'] += trade.profit
            
            if symbol_stats:
                print("\nPerformance by Symbol:")
                sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
                for symbol, stats in sorted_symbols[:10]:  # Top 10 symbols
                    win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                    print(f"  {symbol:<10} Trades: {stats['trades']:<3} Win Rate: {win_rate:>5.1f}%  P/L: ${stats['profit']:>8.2f}")
        else:
            print(f"Monitoring {len(self.symbols)} symbols")
            print("No completed trades in this session yet")
            
        # Show active positions
        if self.active_trades:
            print(f"\nActive Positions ({len(self.active_trades)}):")
            for ticket, trade in list(self.active_trades.items())[:5]:  # Show first 5
                print(f"  {trade.symbol} {trade.signal_type} - Entry: {trade.entry_price}")
            
        print("="*70)
        
    def stop_trading(self):
        """Gracefully stop trading and cleanup resources"""
        self.running = False
        logger.info("🛑 Stopping trading engine...")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")
        
        logger.info("🛑 Trading engine stopped")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Short-Term Forex Trading Bot')
    parser.add_argument('--duration', type=int, default=None, 
                       help='Session duration in minutes (omit for 24/7 operation)')
    parser.add_argument('--no-scalping', action='store_true', 
                       help='Use standard strategies instead of scalping')
    args = parser.parse_args()
    
    # Use ultra-scalping mode by default
    engine = LiveTradingEngine(use_scalping=not args.no_scalping)
    
    logger.info("🎯 Starting Parallel Ultra-Short-Term Forex Trading Bot")
    logger.info("Mode: 24/7 Continuous Operation" if args.duration is None else f"Duration: {args.duration} minutes")
    logger.info(f"Parallel Monitoring: {len(engine.symbols)} symbols simultaneously")
    logger.info("Strategy: Multi-strategy scalping ensemble on M1 timeframe")
    logger.info(f"Trading Hours: Active except {engine.no_trade_start_hour}:00-{engine.no_trade_end_hour}:00 {engine.timezone}")
    logger.info(f"Risk: 0.01 lot per trade, max {engine.max_concurrent_trades} concurrent positions")
    logger.info("Features: Parallel analysis, thread-safe execution, automatic stop loss")
    
    try:
        # Run trading session (continuous if duration not specified)
        engine.run_trading_session(duration_minutes=args.duration)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # In 24/7 mode, restart after fatal error
        if args.duration is None:
            logger.info("Restarting in 5 minutes...")
            time.sleep(300)
            main()  # Restart
    
    if args.duration:
        logger.info("🏁 Trading session completed")
    else:
        logger.info("🏁 Trading bot stopped")

if __name__ == "__main__":
    main()