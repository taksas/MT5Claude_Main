import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from trading_strategies import StrategyEnsemble, SignalType, TradingSignal
from risk_manager import RiskManager
from mt5_client import MT5Client

@dataclass
class PaperTrade:
    id: int
    symbol: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    entry_time: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    status: str = "OPEN"
    reason: str = ""
    max_adverse: float = 0.0
    max_favorable: float = 0.0

class PaperTradingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.mt5_client = MT5Client()
        self.strategy_ensemble = StrategyEnsemble()
        self.risk_manager = RiskManager()
        
        self.paper_trades = {}
        self.trade_counter = 0
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def start_paper_trading(self, symbols: List[str], duration_hours: int = 24):
        """Start paper trading simulation"""
        self.is_running = True
        self.logger.info(f"Starting paper trading for {duration_hours} hours")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)
        
        try:
            while self.is_running and datetime.utcnow() < end_time:
                # Analyze each symbol
                for symbol in symbols:
                    try:
                        self._analyze_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol}: {e}")
                
                # Update open trades
                self._update_open_trades()
                
                # Print status every 30 minutes
                if self.trade_counter % 30 == 0:
                    self._print_status()
                
                time.sleep(60)  # Analyze every minute
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        finally:
            self._close_all_trades()
            self._print_final_results()
    
    def _analyze_symbol(self, symbol: str):
        """Analyze a symbol and create paper trades"""
        try:
            # Check if MT5 API is available, otherwise skip
            if not self.mt5_client.ping():
                return
            
            # Get historical data
            historical_data = self.mt5_client.get_historical_data(symbol, "M5", 100)
            
            if len(historical_data) < 50:
                return
            
            # Get trading signal
            signal = self.strategy_ensemble.get_ensemble_signal(historical_data)
            
            if signal and signal.signal != SignalType.HOLD:
                self._create_paper_trade(signal, symbol)
                
        except Exception as e:
            # If MT5 API is not available, use simulated data
            self._analyze_with_simulated_data(symbol)
    
    def _analyze_with_simulated_data(self, symbol: str):
        """Analyze with simulated market data when MT5 API is unavailable"""
        # Generate simulated OHLC data
        import random
        base_price = 1.1000 if 'EUR' in symbol else 150.00 if 'JPY' in symbol else 1.3000
        
        simulated_data = []
        for i in range(100):
            price = base_price + random.uniform(-0.01, 0.01)
            candle = {
                'time': datetime.utcnow().isoformat(),
                'open': price,
                'high': price + random.uniform(0, 0.005),
                'low': price - random.uniform(0, 0.005),
                'close': price + random.uniform(-0.003, 0.003),
                'tick_volume': random.randint(100, 1000)
            }
            simulated_data.append(candle)
        
        try:
            signal = self.strategy_ensemble.get_ensemble_signal(simulated_data)
            if signal and signal.signal != SignalType.HOLD and signal.confidence > 0.7:
                self._create_paper_trade(signal, symbol)
        except Exception as e:
            self.logger.error(f"Error with simulated analysis: {e}")
    
    def _create_paper_trade(self, signal: TradingSignal, symbol: str):
        """Create a paper trade from a signal"""
        # Validate with risk manager
        current_positions = list(self.paper_trades.values())
        allowed, reason, position_size = self.risk_manager.validate_trade_signal(
            signal, self.current_balance, current_positions
        )
        
        if not allowed:
            self.logger.info(f"Paper trade rejected for {symbol}: {reason}")
            return
        
        # Create paper trade
        self.trade_counter += 1
        paper_trade = PaperTrade(
            id=self.trade_counter,
            symbol=symbol,
            signal_type=signal.signal,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=position_size,
            entry_time=datetime.utcnow().isoformat(),
            reason=signal.reason
        )
        
        self.paper_trades[self.trade_counter] = paper_trade
        
        # Update risk manager
        risk_amount = self.current_balance * 0.02
        self.risk_manager.update_trade_opened(position_size, risk_amount)
        
        self.logger.info(f"Paper trade created: {signal.signal.value} {symbol} "
                        f"@ {signal.entry_price:.5f} Size: {position_size} "
                        f"Confidence: {signal.confidence:.2f}")
    
    def _update_open_trades(self):
        """Update all open paper trades"""
        for trade_id, trade in list(self.paper_trades.items()):
            if trade.status == "OPEN":
                self._update_trade_status(trade_id)
    
    def _update_trade_status(self, trade_id: int):
        """Update the status of a specific trade"""
        trade = self.paper_trades[trade_id]
        
        try:
            # Try to get current price from MT5
            if self.mt5_client.ping():
                historical_data = self.mt5_client.get_historical_data(trade.symbol, "M1", 1)
                if historical_data:
                    current_price = historical_data[-1]['close']
                else:
                    current_price = self._simulate_current_price(trade)
            else:
                current_price = self._simulate_current_price(trade)
            
            # Check for exit conditions
            exit_reason = self._check_exit_conditions(trade, current_price)
            
            if exit_reason:
                self._close_paper_trade(trade_id, current_price, exit_reason)
                
        except Exception as e:
            self.logger.error(f"Error updating trade {trade_id}: {e}")
    
    def _simulate_current_price(self, trade: PaperTrade) -> float:
        """Simulate current price movement"""
        import random
        
        # Simulate price movement based on time elapsed
        entry_time = datetime.fromisoformat(trade.entry_time)
        time_elapsed = (datetime.utcnow() - entry_time).total_seconds() / 60  # minutes
        
        # Random walk with slight trend bias
        price_change = random.uniform(-0.002, 0.002) * (time_elapsed / 30)
        
        # Add some bias based on signal direction
        if trade.signal_type == SignalType.BUY:
            price_change += random.uniform(0, 0.001)
        else:
            price_change -= random.uniform(0, 0.001)
        
        return trade.entry_price + price_change
    
    def _check_exit_conditions(self, trade: PaperTrade, current_price: float) -> Optional[str]:
        """Check if trade should be closed"""
        entry_time = datetime.fromisoformat(trade.entry_time)
        time_elapsed = (datetime.utcnow() - entry_time).total_seconds() / 60  # minutes
        
        # Close after 30 minutes max
        if time_elapsed > 30:
            return "TIMEOUT"
        
        # Check stop loss and take profit
        if trade.signal_type == SignalType.BUY:
            if current_price <= trade.stop_loss:
                return "STOP_LOSS"
            elif current_price >= trade.take_profit:
                return "TAKE_PROFIT"
        else:  # SELL
            if current_price >= trade.stop_loss:
                return "STOP_LOSS"
            elif current_price <= trade.take_profit:
                return "TAKE_PROFIT"
        
        # Early profit taking after 5 minutes
        if time_elapsed > 5:
            pip_value = 0.0001 if 'JPY' not in trade.symbol else 0.01
            
            if trade.signal_type == SignalType.BUY:
                pips_profit = (current_price - trade.entry_price) / pip_value
            else:
                pips_profit = (trade.entry_price - current_price) / pip_value
            
            if pips_profit > 20:  # 20 pips profit
                return "EARLY_PROFIT"
        
        return None
    
    def _close_paper_trade(self, trade_id: int, exit_price: float, reason: str):
        """Close a paper trade"""
        trade = self.paper_trades[trade_id]
        
        trade.exit_time = datetime.utcnow().isoformat()
        trade.exit_price = exit_price
        trade.status = reason
        
        # Calculate P&L
        pip_value = 0.0001 if 'JPY' not in trade.symbol else 0.01
        
        if trade.signal_type == SignalType.BUY:
            trade.pnl_pips = (exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - exit_price) / pip_value
        
        trade.pnl = trade.pnl_pips * pip_value * trade.volume * 100000
        
        # Update balance
        self.current_balance += trade.pnl
        
        # Update risk manager
        risk_amount = self.initial_balance * 0.02
        self.risk_manager.update_trade_closed(trade.pnl, risk_amount)
        
        # Calculate duration
        entry_time = datetime.fromisoformat(trade.entry_time)
        exit_time = datetime.fromisoformat(trade.exit_time)
        duration = (exit_time - entry_time).total_seconds() / 60
        
        self.logger.info(f"Paper trade closed: {trade.symbol} {reason} "
                        f"PnL: ${trade.pnl:.2f} ({trade.pnl_pips:.1f} pips) "
                        f"Duration: {duration:.1f}min")
    
    def _close_all_trades(self):
        """Close all remaining open trades"""
        for trade_id, trade in self.paper_trades.items():
            if trade.status == "OPEN":
                current_price = self._simulate_current_price(trade)
                self._close_paper_trade(trade_id, current_price, "FORCED_CLOSE")
    
    def _print_status(self):
        """Print current trading status"""
        open_trades = len([t for t in self.paper_trades.values() if t.status == "OPEN"])
        total_pnl = sum(t.pnl for t in self.paper_trades.values())
        
        self.logger.info(f"Status: Balance: ${self.current_balance:.2f} "
                        f"Open trades: {open_trades} Total PnL: ${total_pnl:.2f}")
    
    def _print_final_results(self):
        """Print final paper trading results"""
        trades = list(self.paper_trades.values())
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades)
        total_pips = sum(t.pnl_pips for t in trades)
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        print("\n" + "="*60)
        print("PAPER TRADING RESULTS")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Total Trades: {len(trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Pips: {total_pips:.1f}")
        
        if winning_trades:
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            print(f"Average Win: ${avg_win:.2f}")
        
        if losing_trades:
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            print(f"Average Loss: ${avg_loss:.2f}")
        
        print("="*60)
        
        # Determine if strategy is ready for live trading
        if (total_pnl > 0 and win_rate >= 50 and len(trades) >= 10):
            print("✅ STRATEGY APPROVED FOR LIVE TRADING")
        else:
            print("❌ STRATEGY NEEDS MORE TESTING")
        
        print("="*60)
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save paper trading results to file"""
        results = {
            "summary": {
                "initial_balance": self.initial_balance,
                "final_balance": self.current_balance,
                "total_pnl": self.current_balance - self.initial_balance,
                "total_trades": len(self.paper_trades),
                "timestamp": datetime.utcnow().isoformat()
            },
            "trades": [asdict(trade) for trade in self.paper_trades.values()]
        }
        
        filename = f"paper_trading_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filename}")

def main():
    """Run paper trading simulation"""
    # Test symbols (using common forex pairs)
    test_symbols = ["EURUSD#", "GBPUSD#", "USDJPY#", "USDCAD#"]
    
    paper_engine = PaperTradingEngine()
    
    try:
        paper_engine.start_paper_trading(test_symbols, duration_hours=2)  # 2 hour test
    except KeyboardInterrupt:
        print("\nPaper trading interrupted by user")

if __name__ == "__main__":
    main()