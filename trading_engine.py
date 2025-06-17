import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
from mt5_client import MT5Client
from trading_strategies import StrategyEnsemble, SignalType, TradingSignal
from risk_manager import RiskManager, RiskLimits
from backtesting import Backtester

class TradingEngine:
    def __init__(self, mt5_host: str = "172.28.144.1", mt5_port: int = 8000):
        self.mt5_client = MT5Client(mt5_host, mt5_port)
        self.strategy_ensemble = StrategyEnsemble()
        self.risk_manager = RiskManager()
        self.backtester = Backtester()
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.active_positions = {}
        self.position_monitors = {}
        self.last_analysis_time = {}
        
        self.approved_symbols = set()
        self.symbol_pip_values = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_engine.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize(self) -> bool:
        """Initialize the trading engine and validate connection"""
        try:
            if not self.mt5_client.ping():
                self.logger.error("MT5 Bridge API is not responding")
                return False
            
            mt5_status = self.mt5_client.check_mt5_status()
            if not mt5_status.get("connected") or not mt5_status.get("trade_allowed"):
                self.logger.error("MT5 not connected or trading not allowed")
                return False
            
            account_info = self.mt5_client.get_account_info()
            self.logger.info(f"Account Balance: {account_info.get('balance', 0):.2f}")
            
            self._discover_tradable_symbols()
            self._validate_strategies()
            
            self.logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    def _discover_tradable_symbols(self):
        """Discover and validate available forex symbols"""
        try:
            # Get all available symbols
            all_symbols = self.mt5_client.discover_available_symbols()
            self.logger.info(f"Found {len(all_symbols)} total symbols")
            
            # First try hash symbols (as specified in requirements)
            hash_symbols = [s for s in all_symbols if s.endswith('#')]
            if hash_symbols:
                self.logger.info(f"Found {len(hash_symbols)} hash symbols - using those")
                candidate_symbols = hash_symbols[:5]  # Limit to first 5
            else:
                # Fall back to forex symbols
                forex_symbols = self.mt5_client.get_forex_symbols()
                self.logger.info(f"No hash symbols found, using {len(forex_symbols)} forex symbols")
                
                # Prioritize major pairs
                priority_symbols = []
                for symbol in forex_symbols:
                    if any(pair in symbol.upper() for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']):
                        priority_symbols.append(symbol)
                
                candidate_symbols = priority_symbols[:5] if priority_symbols else forex_symbols[:5]
            
            # Validate each symbol
            for symbol in candidate_symbols:
                try:
                    symbol_info = self.mt5_client.get_symbol_info(symbol)
                    trade_mode_num = symbol_info.get('trade_mode', 0)
                    
                    # MT5 trade modes: 0=DISABLED, 1=LONGONLY, 2=SHORTONLY, 3=CLOSEONLY, 4=FULL
                    if trade_mode_num == 4:
                        self.approved_symbols.add(symbol)
                        
                        # Calculate pip value
                        point = symbol_info.get('point', 0.0001)
                        if 'JPY' in symbol.upper():
                            point = 0.01  # JPY pairs typically use 2 decimal places
                        
                        self.symbol_pip_values[symbol] = point
                        
                        self.logger.info(f"✅ Approved symbol: {symbol} (pip: {point}, mode: {trade_mode_num})")
                    else:
                        self.logger.warning(f"❌ Symbol {symbol} not fully tradable (mode: {trade_mode_num})")
                        
                except Exception as e:
                    self.logger.warning(f"Error checking symbol {symbol}: {e}")
            
            if not self.approved_symbols:
                self.logger.error("❌ No approved symbols found - cannot start trading")
            else:
                self.logger.info(f"🎯 Using {len(self.approved_symbols)} approved symbols: {list(self.approved_symbols)}")
            
        except Exception as e:
            self.logger.error(f"Error discovering symbols: {e}")
    
    def _validate_strategies(self):
        """Validate strategies with backtesting"""
        if not self.approved_symbols:
            return
        
        self.logger.info("Validating strategies with backtesting...")
        
        for symbol in list(self.approved_symbols)[:2]:  # Test first 2 symbols
            try:
                # Get historical data for backtesting
                historical_data = self.mt5_client.get_historical_data(symbol, "M5", 500)
                
                if len(historical_data) < 200:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Run backtest
                pip_value = self.symbol_pip_values.get(symbol, 0.0001)
                results = self.backtester.run_backtest(historical_data, symbol, pip_value)
                
                self.logger.info(f"Backtest {symbol}: {results.total_trades} trades, "
                               f"{results.win_rate:.1f}% win rate, "
                               f"${results.total_pnl:.2f} PnL")
                
                # Remove symbol if backtest performance is poor
                if (results.total_pnl < 0 or results.win_rate < 40 or 
                    results.max_drawdown > 25):
                    self.approved_symbols.discard(symbol)
                    self.logger.warning(f"Removed {symbol} due to poor backtest results")
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {symbol}: {e}")
                self.approved_symbols.discard(symbol)
    
    def start_trading(self):
        """Start the automated trading engine"""
        if not self.approved_symbols:
            self.logger.error("No approved symbols available for trading")
            return
        
        self.is_running = True
        self.logger.info("Starting automated trading engine...")
        
        # Start main trading loop
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()
        
        # Start position monitoring
        monitor_thread = threading.Thread(target=self._position_monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Trading engine started successfully")
    
    def stop_trading(self):
        """Stop the trading engine"""
        self.is_running = False
        self.logger.info("Trading engine stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Check risk management status
                if self.risk_manager.should_emergency_stop():
                    reasons = self.risk_manager.get_emergency_stop_reasons()
                    self.logger.warning(f"Emergency stop triggered: {reasons}")
                    time.sleep(300)  # Wait 5 minutes before retrying
                    continue
                
                # Analyze each approved symbol
                for symbol in self.approved_symbols:
                    try:
                        self._analyze_and_trade_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol}: {e}")
                
                # Wait before next analysis
                time.sleep(60)  # Analyze every minute
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(30)
    
    def _analyze_and_trade_symbol(self, symbol: str):
        """Analyze a symbol and potentially place trades"""
        current_time = datetime.utcnow()
        
        # Check if enough time has passed since last analysis
        if (symbol in self.last_analysis_time and 
            (current_time - self.last_analysis_time[symbol]).total_seconds() < 300):
            return
        
        # Get recent market data
        historical_data = self.mt5_client.get_historical_data(symbol, "M5", 100)
        
        if len(historical_data) < 50:
            return
        
        # Get trading signal
        signal = self.strategy_ensemble.get_ensemble_signal(historical_data)
        
        if signal and signal.signal != SignalType.HOLD:
            self._process_trading_signal(signal, symbol)
        
        self.last_analysis_time[symbol] = current_time
    
    def _process_trading_signal(self, signal: TradingSignal, symbol: str):
        """Process a trading signal and potentially execute trade"""
        try:
            # Get account info and current positions
            account_info = self.mt5_client.get_account_info()
            current_positions = self.mt5_client.get_positions()
            
            # Validate signal with risk manager
            allowed, reason, position_size = self.risk_manager.validate_trade_signal(
                signal, account_info.get('balance', 0), current_positions
            )
            
            if not allowed:
                self.logger.info(f"Trade rejected for {symbol}: {reason}")
                return
            
            # Execute trade
            self._execute_trade(signal, symbol, position_size)
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _execute_trade(self, signal: TradingSignal, symbol: str, position_size: float):
        """Execute a trade based on the signal"""
        try:
            # Determine order type
            order_type = 0 if signal.signal == SignalType.BUY else 1
            
            # Place market order
            result = self.mt5_client.place_order(
                action=1,  # Market order
                symbol=symbol,
                volume=position_size,
                order_type=order_type,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"AI_{signal.reason[:20]}"
            )
            
            if result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                position_ticket = result.get('order')
                
                self.active_positions[position_ticket] = {
                    'symbol': symbol,
                    'signal': signal,
                    'position_size': position_size,
                    'entry_time': datetime.utcnow(),
                    'entry_price': result.get('price', signal.entry_price)
                }
                
                # Update risk manager
                account_balance = self.mt5_client.get_account_info().get('balance', 0)
                risk_amount = account_balance * 0.02
                self.risk_manager.update_trade_opened(position_size, risk_amount)
                
                self.logger.info(f"Trade executed: {signal.signal.value} {symbol} "
                               f"Size: {position_size} Price: {result.get('price'):.5f} "
                               f"SL: {signal.stop_loss:.5f} TP: {signal.take_profit:.5f}")
                
                # Start monitoring this position
                self._start_position_monitor(position_ticket)
                
            else:
                self.logger.error(f"Trade execution failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def _start_position_monitor(self, ticket: int):
        """Start monitoring a position for early exit conditions"""
        def monitor():
            while ticket in self.active_positions and self.is_running:
                try:
                    position = self.mt5_client.get_positions()
                    active_position = next((p for p in position if p.get('ticket') == ticket), None)
                    
                    if not active_position:
                        # Position closed
                        self._handle_position_closed(ticket)
                        break
                    
                    # Check for early exit conditions
                    if self._should_exit_early(ticket, active_position):
                        self._close_position_early(ticket)
                        break
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring position {ticket}: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.position_monitors[ticket] = monitor_thread
    
    def _should_exit_early(self, ticket: int, position_data: Dict) -> bool:
        """Determine if position should be closed early"""
        if ticket not in self.active_positions:
            return False
        
        trade_info = self.active_positions[ticket]
        entry_time = trade_info['entry_time']
        current_time = datetime.utcnow()
        
        # Close after 30 minutes max (as per requirements)
        if (current_time - entry_time).total_seconds() > 1800:  # 30 minutes
            return True
        
        # Close if in profit and time > 5 minutes (minimum hold time)
        trade_duration = (current_time - entry_time).total_seconds()
        current_profit = position_data.get('profit', 0)
        
        if trade_duration > 300 and current_profit > 10:  # 5 minutes and $10 profit
            return True
        
        return False
    
    def _close_position_early(self, ticket: int):
        """Close position early"""
        try:
            result = self.mt5_client.close_position(ticket)
            self.logger.info(f"Position {ticket} closed early: {result}")
            self._handle_position_closed(ticket)
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
    
    def _handle_position_closed(self, ticket: int):
        """Handle position closure cleanup"""
        if ticket in self.active_positions:
            trade_info = self.active_positions[ticket]
            
            # Update risk manager (will need actual PnL from position data)
            risk_amount = 100  # Placeholder
            self.risk_manager.update_trade_closed(0, risk_amount)
            
            del self.active_positions[ticket]
            
            if ticket in self.position_monitors:
                del self.position_monitors[ticket]
    
    def _position_monitor_loop(self):
        """Monitor all positions periodically"""
        while self.is_running:
            try:
                if self.active_positions:
                    positions = self.mt5_client.get_positions()
                    active_tickets = {p.get('ticket') for p in positions}
                    
                    # Check for closed positions
                    closed_tickets = set(self.active_positions.keys()) - active_tickets
                    for ticket in closed_tickets:
                        self._handle_position_closed(ticket)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                time.sleep(60)
    
    def get_status(self) -> Dict:
        """Get current trading engine status"""
        risk_status = self.risk_manager.get_status_report()
        
        return {
            "is_running": self.is_running,
            "approved_symbols": list(self.approved_symbols),
            "active_positions": len(self.active_positions),
            "risk_status": risk_status,
            "last_update": datetime.utcnow().isoformat()
        }

def main():
    """Main function to run the trading engine"""
    engine = TradingEngine()
    
    if not engine.initialize():
        print("Failed to initialize trading engine")
        return
    
    try:
        engine.start_trading()
        
        # Keep running and show status
        while True:
            time.sleep(300)  # Status update every 5 minutes
            status = engine.get_status()
            print(f"\nStatus: {json.dumps(status, indent=2)}")
            
    except KeyboardInterrupt:
        print("\nShutting down trading engine...")
        engine.stop_trading()
    except Exception as e:
        print(f"Error in main: {e}")
        engine.stop_trading()

if __name__ == "__main__":
    main()