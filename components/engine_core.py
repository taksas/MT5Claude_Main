#!/usr/bin/env python3
"""
Engine Core Module
Main orchestrator that brings all components together
"""

import logging
import time
import pytz
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from .trading_components import CONFIG, HIGH_PROFIT_SYMBOLS, Trade, Signal, MT5APIClient, MarketData, SymbolUtils
from .trading_strategy import TradingStrategy
from .risk_management import RiskManagement
from .trading_components import OrderManagement

logger = logging.getLogger('UltraTradingEngine')

class UltraTradingEngine:
    def __init__(self, signal_queue: Optional[queue.Queue] = None):
        # Initialize components
        self.api_client = MT5APIClient(CONFIG["API_BASE"])
        self.market_data = MarketData(self.api_client)
        self.symbol_utils = SymbolUtils()
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManagement()
        self.order_manager = OrderManagement(self.api_client)
        
        # Pass api_client to risk_manager if needed for dynamic data
        self.risk_manager.api_client = self.api_client

        # Configuration
        self.config = CONFIG
        self.timezone = pytz.timezone(CONFIG["TIMEZONE"])
        
        # Communication queue for visualizer
        self.signal_queue = signal_queue
        
        # State management
        self.active_trades = {}
        self.pending_orders = {}
        self.symbol_positions = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_time = {}
        self.last_global_trade_time = 0
        self.balance = None
        self.account_info = None
        self.running = False
        self.trades_this_hour = 0
        self.force_trade_attempts = 0
        
        self.day_start_balance = None
        self.daily_closed_pnl = 0
        self.current_day = None
        
        self.tradable_symbols = []
        
        self.last_signals = {}
        
        self.last_pnl_share_time = 0
        self.last_shared_pnl = None
        self.account_safety_warning_logged = False
        
    def start(self):
        """Initialize and start trading"""
        if not self.api_client.check_connection():
            logger.error("Cannot connect to API")
            return False
        
        self.balance = self.api_client.get_balance()
        if not self.balance:
            logger.error("Cannot get account balance")
            return False
        
        self.pending_orders.clear()
        logger.info("Cleared pending orders from previous session")
        
        self.tradable_symbols = self._discover_symbols()
        if not self.tradable_symbols:
            logger.error("No tradable symbols found")
            return False
        
        logger.info(f"🚀 Starting Ultra Trading Engine with Enhanced Diversification")
        logger.info(f"💰 Balance: ¥{self.balance:,.0f}")
        logger.info(f"📊 Monitoring {len(self.tradable_symbols)} symbols for maximum diversification")
        logger.info(f"🎯 Diversification Policy: One position per symbol maximum")
        
        self.running = True
        self.run()
        return True
    
    def run(self):
        """Main trading loop"""
        logger.info("📈 Starting main trading loop...")
        
        while self.running:
            try:
                self.run_once()
                time.sleep(CONFIG.get("LOOP_DELAY", 1))
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("📉 Main trading loop stopped")
    
    def stop(self):
        """Stop trading engine"""
        self.running = False
        logger.info("🛑 Stopping Ultra Trading Engine")
    
    def run_once(self):
        """Run one iteration of the trading loop"""
        if not self.running: return
        
        try:
            self._cleanup_pending_orders()
            account_info = self.api_client.get_account_info()
            if account_info:
                self.account_info = account_info
                self.balance = account_info.get('balance', self.balance)
                equity = account_info.get('equity', self.balance)
                margin = account_info.get('margin', 0)
                
                current_date = datetime.now(self.timezone).date()
                if self.current_day != current_date:
                    self.current_day = current_date
                    self.day_start_balance = self.balance
                    self.daily_closed_pnl = 0
                    logger.info(f"🌅 New trading day: {current_date}, Starting balance: {self.balance:.2f}")
                
                if self.day_start_balance is None: self.day_start_balance = self.balance
                
                positions = self.api_client.get_positions()
                open_pnl = sum(pos.get('profit', 0) for pos in positions)
                self.daily_pnl = self.daily_closed_pnl + open_pnl
                
                safe, reason = self.risk_manager.check_account_safety(self.day_start_balance, equity, margin)
                if not safe:
                    # まだ警告ログを出力していない場合のみ、ログを出力してフラグを立てる
                    if not self.account_safety_warning_logged:
                        logger.warning(f"⚠️ Account not safe: {reason}. Trading will be paused.")
                        self.account_safety_warning_logged = True
                    return # 取引は行わずにループを抜ける
                else:
                    # アカウントが安全な状態に復帰した場合
                    if self.account_safety_warning_logged:
                        logger.info("✅ Account safety restored. Resuming normal operations.")
                        self.account_safety_warning_logged = False # フラグをリセット
            
            self._manage_positions()
            
            if not self._is_trading_hours(): return
            
            if self._should_force_trade(): self._force_trade()
            
            available_symbols = self.risk_manager.get_available_symbols_for_trading(self.active_trades, self.tradable_symbols)
            
            should_seek, seek_reason = self.risk_manager.should_seek_new_positions(self.account_info, self.active_trades) if self.account_info else (False, "No account info")
            
            if should_seek and available_symbols:
                symbols_to_analyze = available_symbols[:10]
            else:
                symbols_to_analyze = self.tradable_symbols[:self.config["MAX_SYMBOLS"]]
                import random
                random.shuffle(symbols_to_analyze)
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._analyze_symbol, symbol): symbol for symbol in symbols_to_analyze}
                for future in futures:
                    symbol = futures[future]
                    try:
                        signal = future.result()
                        if signal:
                            # Pass the correct symbol from the future mapping
                            self._execute_signal(symbol, signal)
                    except Exception as e:
                        logger.error(f"Error processing analysis for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}", exc_info=True)
    
    def _discover_symbols(self) -> List[str]:
        """Discover and filter tradable symbols with diversification priority"""
        try:
            all_symbols = self.api_client.discover_symbols()
            if not all_symbols: return []
            
            # Sort by priority from config
            sorted_priority = sorted(HIGH_PROFIT_SYMBOLS.items(), key=lambda item: item[1].get('priority', 99))
            
            final_symbols = [s + "#" for s, _ in sorted_priority if (s + "#") in all_symbols]
            
            # Add other symbols that might exist on broker but not in our priority list
            for symbol in all_symbols:
                if symbol not in final_symbols and not any(x in symbol.lower() for x in ['_i', '.i', 'mini', 'micro']):
                    final_symbols.append(symbol)

            result = final_symbols[:self.config["MAX_SYMBOLS"]]
            logger.info(f"📊 Symbol discovery: {len(result)} symbols selected for monitoring.")
            return result
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []
    
    def _is_trading_hours(self) -> bool: return True
    
    def _should_force_trade(self) -> bool:
        if not self.config["AGGRESSIVE_MODE"]: return False
        if not self.active_trades and self.last_global_trade_time == 0: # First run
             self.last_global_trade_time = time.time()
        time_since_last = time.time() - self.last_global_trade_time
        return time_since_last > self.config["FORCE_TRADE_INTERVAL"]
    
    def _force_trade(self):
        """Force a trade on the best opportunity with diversification priority"""
        logger.info("🔥 Forcing trade due to inactivity...")
        available_symbols = self.risk_manager.get_available_symbols_for_trading(self.active_trades, self.tradable_symbols)
        symbols_to_check = available_symbols[:10] if available_symbols else self.tradable_symbols[:10]
        
        # FIXED: Store both the best signal and its corresponding symbol
        best_signal = None
        best_symbol = None
        
        for symbol in symbols_to_check:
            df = self.market_data.get_market_data(symbol)
            if df is None or len(df) < 50: continue
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price: continue
            
            signal = self.strategy.force_trade_signal(symbol, df, current_price)
            if signal:
                # Compare quality to find the best signal
                if best_signal is None or signal.quality > best_signal.quality:
                    best_signal = signal
                    best_symbol = symbol # Store the symbol along with the signal
        
        if best_signal and best_symbol:
            # FIXED: Use the stored 'best_symbol' for logging and execution
            logger.info(f"🎯 Force trade selected: {best_symbol}")
            self._execute_signal(best_symbol, best_signal)
        else:
            logger.warning("🚫 No suitable symbols found for forced trade.")
    
    def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """Analyze a symbol for trading opportunities"""
        try:
            # Step 1: Preliminary checks (spread) before heavy analysis
            spread_ok, _ = self.market_data.check_spread(symbol)
            if not self.config["IGNORE_SPREAD"] and not spread_ok:
                if self.signal_queue:
                    self.signal_queue.put({symbol: {'type': 'NONE', 'confidence': 0.0, 'quality': 0.0, 'reasons': ['High spread'], 'strategies': {}}})
                return None

            # Step 2: Get market data
            df = self.market_data.get_market_data(symbol, count=250)
            if df is None or len(df) < 200:
                if self.signal_queue:
                    self.signal_queue.put({symbol: {'type': 'NONE', 'confidence': 0.0, 'quality': 0.0, 'reasons': ['Not enough data'], 'strategies': {}}})
                return None
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price: return None
            
            # Step 3: Always perform analysis and get a potential signal
            signal = self.strategy.analyze_ultra(symbol, df, current_price)
            
            # Step 4: IMMEDIATELY send analysis result to the visualizer queue
            if self.signal_queue:
                viz_data = {}
                if signal:
                    # A potential trade signal was found
                    viz_data = {
                        'type': signal.type.value,
                        'confidence': signal.confidence,
                        'quality': signal.quality,
                        'reasons': [signal.reason],
                        'strategies': signal.strategies
                    }
                else:
                    # Analysis was done, but no clear signal emerged
                    viz_data = {
                        'type': 'NONE',
                        'confidence': 0.0,
                        'quality': 0.0,
                        'reasons': ['No clear signal'],
                        'strategies': {}
                    }
                self.signal_queue.put({symbol: viz_data})
            
            # Step 5: Now, check if we are allowed to trade on this signal
            if not signal:
                return None # No signal to execute

            can_trade, reason = self.risk_manager.can_trade_symbol(symbol, self.active_trades, self.last_trade_time, self.pending_orders)
            if not can_trade:
                if "position already open" not in reason.lower() and "pending order" not in reason.lower():
                    logger.debug(f"Signal for {symbol} found, but cannot trade: {reason}")
                return None # Signal exists but cannot be executed right now
            
            # If all checks pass, return the signal for execution
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            # Notify visualizer of the error
            if self.signal_queue:
                self.signal_queue.put({symbol: {'type': 'ERROR', 'confidence': 0.0, 'reasons': [str(e)]}})
            return None
    
    def _execute_signal(self, symbol: str, signal: Signal):
        """Execute trading signal"""
        try:
            if symbol in self.pending_orders:
                if time.time() - self.pending_orders[symbol]['timestamp'] < 60:
                    logger.warning(f"Pending order already exists for {symbol}, skipping.")
                    return
            
            self.pending_orders[symbol] = {'signal': signal, 'timestamp': time.time()}

            symbol_info = self.api_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Cannot get symbol info for {symbol}")
                del self.pending_orders[symbol]
                return

            # ARCHITECTURE CHANGE: Adjust parameters for risk BEFORE validating them
            adjusted_signal, volume = self.risk_manager.adjust_trade_parameters_for_risk(
                signal, self.balance, symbol_info
            )

            if volume <= 0:
                logger.error(f"Trade rejected for {symbol} due to zero or negative volume calculation.")
                del self.pending_orders[symbol]
                return

            # Now validate the ADJUSTED parameters
            valid, reason = self.risk_manager.validate_trade_parameters(
                symbol, adjusted_signal.type.value, adjusted_signal.entry, adjusted_signal.sl, adjusted_signal.tp
            )
            if not valid:
                logger.warning(f"Invalid adjusted trade parameters for {symbol}: {reason}")
                del self.pending_orders[symbol]
                return
            
            if self.account_info:
                sl_distance = abs(adjusted_signal.entry - adjusted_signal.sl)
                margin_safe, margin_reason = self.risk_manager.check_margin_safety_before_trade(
                    symbol, volume, adjusted_signal.entry, sl_distance, self.account_info
                )
                if not margin_safe:
                    logger.warning(f"⚠️ Margin safety check failed for {symbol}: {margin_reason}")
                    del self.pending_orders[symbol]
                    return
            
            ticket = self.order_manager.place_order(adjusted_signal, symbol, volume)
            if ticket:
                trade = Trade(
                    ticket=ticket, symbol=symbol, type=adjusted_signal.type.value,
                    entry_price=adjusted_signal.entry, sl=adjusted_signal.sl, tp=adjusted_signal.tp,
                    volume=volume, entry_time=datetime.now()
                )
                self.active_trades[ticket] = trade
                self.symbol_positions[symbol] = adjusted_signal.type.value
                self.last_trade_time[symbol] = time.time()
                self.last_global_trade_time = time.time()
                self.daily_trades += 1
                
            del self.pending_orders[symbol]

        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)
            if symbol in self.pending_orders:
                del self.pending_orders[symbol]
    
    def _manage_positions(self):
        """Manage open positions"""
        try:
            current_positions = self.api_client.get_positions()
            if current_positions is None:
                logger.warning("Could not retrieve current positions.")
                return

            current_tickets = {pos.get('ticket') for pos in current_positions}
            
            closed_tickets = set(self.active_trades.keys()) - current_tickets
            for ticket in closed_tickets:
                trade = self.active_trades.pop(ticket, None)
                if trade:
                    logger.info(f"🎯 Position {ticket} ({trade.symbol}) closed (detected by disappearance).")
                    if trade.symbol in self.symbol_positions:
                        del self.symbol_positions[trade.symbol]

            # Update PnL based on balance changes
            if self.day_start_balance is not None and self.balance is not None:
                balance_change = self.balance - self.day_start_balance
                open_pnl = sum(pos.get('profit', 0) for pos in current_positions)
                self.daily_closed_pnl = balance_change - open_pnl

            # Manage remaining open positions
            management_results = self.order_manager.manage_positions(self.active_trades)
            for ticket in management_results.get('closed', []):
                if ticket in self.active_trades:
                    self.active_trades.pop(ticket)

        except Exception as e:
            logger.error(f"Error managing positions: {e}", exc_info=True)
    
    def _cleanup_pending_orders(self):
        """Clean up expired pending orders"""
        expired = [s for s, d in list(self.pending_orders.items()) if time.time() - d['timestamp'] > 60]
        for s in expired:
            logger.warning(f"Cleaning up expired pending order for {s}")
            del self.pending_orders[s]