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

from .trading_config import CONFIG, HIGH_PROFIT_SYMBOLS
from .trading_models import Trade, Signal
from .mt5_api_client import MT5APIClient
from .market_data import MarketData
from .symbol_utils import SymbolUtils
from .trading_strategy import TradingStrategy
from .risk_management import RiskManagement
from .order_management import OrderManagement

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
        
        # Configuration
        self.config = CONFIG
        self.timezone = pytz.timezone(CONFIG["TIMEZONE"])
        
        # Communication queue for visualizer
        self.signal_queue = signal_queue
        
        # State management
        self.active_trades = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_time = {}
        self.last_global_trade_time = 0
        self.balance = None
        self.running = False
        self.trades_this_hour = 0
        self.force_trade_attempts = 0
        
        # Symbol management
        self.tradable_symbols = []
        
        # Performance tracking
        self.last_signals = {}
        
    def start(self):
        """Initialize and start trading"""
        if not self.api_client.check_connection():
            logger.error("Cannot connect to API")
            return False
        
        self.balance = self.api_client.get_balance()
        if not self.balance:
            logger.error("Cannot get account balance")
            return False
        
        # Discover tradable symbols
        self.tradable_symbols = self._discover_symbols()
        if not self.tradable_symbols:
            logger.error("No tradable symbols found")
            return False
        
        logger.info(f"🚀 Starting Ultra Trading Engine")
        logger.info(f"💰 Balance: ¥{self.balance:,.0f}")
        logger.info(f"📊 Monitoring {len(self.tradable_symbols)} symbols")
        
        self.running = True
        
        # Start the main trading loop
        self.run()
        
        return True
    
    def run(self):
        """Main trading loop"""
        logger.info("📈 Starting main trading loop...")
        
        while self.running:
            try:
                self.run_once()
                time.sleep(CONFIG.get("LOOP_DELAY", 1))  # Default 1 second delay
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("📉 Main trading loop stopped")
    
    def stop(self):
        """Stop trading engine"""
        self.running = False
        logger.info("🛑 Stopping Ultra Trading Engine")
    
    def run_once(self):
        """Run one iteration of the trading loop"""
        if not self.running:
            return
        
        try:
            # Update account info
            account_info = self.api_client.get_account_info()
            if account_info:
                self.balance = account_info.get('balance', self.balance)
                equity = account_info.get('equity', self.balance)
                margin = account_info.get('margin', 0)
                
                # Check account safety
                safe, reason = self.risk_manager.check_account_safety(
                    self.balance, equity, margin, self.daily_pnl
                )
                if not safe:
                    logger.warning(f"⚠️ Account not safe: {reason}")
                    return
            
            # Manage existing positions
            self._manage_positions()
            
            # Check trading hours
            if not self._is_trading_hours():
                return
            
            # Check for forced trade
            if self._should_force_trade():
                self._force_trade()
            
            # Analyze symbols for opportunities
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for symbol in self.tradable_symbols[:self.config["MAX_SYMBOLS"]]:
                    future = executor.submit(self._analyze_symbol, symbol)
                    futures.append((symbol, future))
                
                for symbol, future in futures:
                    try:
                        signal = future.result(timeout=5)
                        if signal:
                            self._execute_signal(symbol, signal)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
    
    def _discover_symbols(self) -> List[str]:
        """Discover and filter tradable symbols"""
        try:
            all_symbols = self.api_client.discover_symbols()
            
            # Start with high-profit symbols (add # suffix)
            priority_symbols = [s + "#" for s in HIGH_PROFIT_SYMBOLS.keys()]
            
            # Filter all symbols
            filtered_symbols = []
            for symbol in all_symbols:
                # Skip if not a trading symbol
                if any(x in symbol.lower() for x in ['_i', '.i', 'mini', 'micro']):
                    continue
                
                # Get instrument type (strip # for checking)
                symbol_base = symbol.rstrip('#')
                instrument_type = self.symbol_utils.get_instrument_type(symbol_base)
                
                # Add based on configuration
                if self.config["SYMBOL_FILTER"] == "ALL":
                    filtered_symbols.append(symbol)
                elif self.config["SYMBOL_FILTER"] == "FOREX" and instrument_type in ['major', 'exotic']:
                    filtered_symbols.append(symbol)
                elif symbol in priority_symbols or symbol_base in HIGH_PROFIT_SYMBOLS:
                    filtered_symbols.append(symbol)
            
            # Combine priority and filtered symbols
            final_symbols = []
            # First add priority symbols that exist in all_symbols
            for symbol in priority_symbols:
                if symbol in all_symbols:
                    final_symbols.append(symbol)
            # Then add other filtered symbols
            for symbol in filtered_symbols:
                if symbol not in final_symbols:
                    final_symbols.append(symbol)
            
            # Limit to max symbols
            return final_symbols[:self.config["MAX_SYMBOLS"]]
            
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        # Trading hours check disabled - trade 24/7
        return True
    
    def _should_force_trade(self) -> bool:
        """Check if we should force a trade"""
        if not self.config["AGGRESSIVE_MODE"]:
            return False
        
        time_since_last = time.time() - self.last_global_trade_time
        return time_since_last > self.config["FORCE_TRADE_INTERVAL"]
    
    def _force_trade(self):
        """Force a trade on the best opportunity"""
        logger.info("🔥 Forcing trade due to inactivity")
        self.force_trade_attempts += 1
        
        best_symbol = None
        best_signal = None
        best_score = 0
        
        for symbol in self.tradable_symbols[:10]:  # Check top 10 symbols
            df = self.market_data.get_market_data(symbol)
            if df is None or len(df) < 50:
                continue
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price:
                continue
            
            signal = self.strategy.force_trade_signal(symbol, df, current_price)
            if signal and signal.confidence > best_score:
                best_symbol = symbol
                best_signal = signal
                best_score = signal.confidence
        
        if best_symbol and best_signal:
            self._execute_signal(best_symbol, best_signal)
    
    def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """Analyze a symbol for trading opportunities"""
        try:
            # Check if we can trade this symbol
            can_trade, reason = self.risk_manager.can_trade_symbol(
                symbol, self.active_trades, self.last_trade_time
            )
            if not can_trade:
                return None
            
            # Check spread
            spread_ok, spread = self.market_data.check_spread(symbol)
            if not spread_ok and not self.config["IGNORE_SPREAD"]:
                return None
            
            # Get market data
            df = self.market_data.get_market_data(symbol)
            if df is None or len(df) < 50:
                return None
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price:
                return None
            
            # Generate signal
            signal = self.strategy.analyze_ultra(symbol, df, current_price)
            
            # Cache signal for visualization
            if signal:
                self.last_signals[symbol] = {
                    'signal': signal,
                    'timestamp': time.time()
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _execute_signal(self, symbol: str, signal: Signal):
        """Execute trading signal"""
        try:
            # Get symbol info
            symbol_info = self.api_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Cannot get symbol info for {symbol}")
                return
            
            # Calculate position size
            sl_distance = abs(signal.entry - signal.sl)
            volume = self.risk_manager.calculate_position_size(
                symbol, sl_distance, signal.entry, self.balance, symbol_info
            )
            
            # Validate trade parameters
            valid, reason = self.risk_manager.validate_trade_parameters(
                symbol, signal.type.value, signal.entry, signal.sl, signal.tp
            )
            if not valid:
                logger.warning(f"Invalid trade parameters for {symbol}: {reason}")
                return
            
            # Place order
            ticket = self.order_manager.place_order(signal, symbol, volume)
            if ticket:
                # Record trade
                trade = Trade(
                    ticket=ticket,
                    symbol=symbol,
                    type=signal.type.value,
                    entry_price=signal.entry,
                    sl=signal.sl,
                    tp=signal.tp,
                    volume=volume,
                    entry_time=datetime.now()
                )
                
                self.active_trades[ticket] = trade
                self.last_trade_time[symbol] = time.time()
                self.last_global_trade_time = time.time()
                self.daily_trades += 1
                self.trades_this_hour += 1
                
                # Send to visualizer
                if self.signal_queue:
                    self.signal_queue.put({
                        'type': 'trade',
                        'symbol': symbol,
                        'signal': signal,
                        'volume': volume,
                        'ticket': ticket,
                        'timestamp': time.time()
                    })
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
    
    def _manage_positions(self):
        """Manage open positions"""
        try:
            results = self.order_manager.manage_positions(self.active_trades)
            
            # Remove closed positions
            for ticket in results['closed']:
                if ticket in self.active_trades:
                    del self.active_trades[ticket]
            
            # Update daily P&L
            positions = self.api_client.get_positions()
            current_pnl = sum(pos.get('profit', 0) for pos in positions)
            self.daily_pnl = current_pnl
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")