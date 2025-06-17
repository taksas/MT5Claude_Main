import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from trading_strategies import StrategyEnsemble, TradingSignal, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: str
    exit_time: Optional[str] = None
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.01
    pnl: float = 0.0
    pnl_pips: float = 0.0
    status: str = "OPEN"
    reason: str = ""
    duration_minutes: int = 0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0

@dataclass
class BacktestResults:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pips: float = 0.0
    avg_trade_duration: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)

class Backtester:
    def __init__(self, initial_balance: float = 5644.0, risk_per_trade: float = 0.02, api_base="http://172.28.144.1:8000"):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.strategy_ensemble = StrategyEnsemble()
        self.logger = logging.getLogger(__name__)
        self.api_base = api_base
        
    def get_historical_data(self, symbol, timeframe="M5", count=500):
        """Get historical data from MT5 API"""
        url = f"{self.api_base}/market/history"
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                data_list = response.json()
                return data_list
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        return None
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              account_balance: float, pip_value: float = 0.0001) -> float:
        risk_amount = account_balance * self.risk_per_trade
        
        if entry_price == 0 or stop_loss == 0:
            return 0.01
        
        price_diff = abs(entry_price - stop_loss)
        pips_at_risk = price_diff / pip_value
        
        if pips_at_risk == 0:
            return 0.01
        
        position_size = risk_amount / (pips_at_risk * pip_value * 100000)
        
        return max(0.01, min(1.0, round(position_size, 2)))
    
    def simulate_trade_execution(self, signal: TradingSignal, data: List[Dict], 
                               start_idx: int, pip_value: float = 0.0001) -> Trade:
        trade = Trade(
            entry_time=signal.timestamp,
            symbol="",
            signal_type=signal.signal,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=0.01,
            reason=signal.reason
        )
        
        max_adverse = 0.0
        max_favorable = 0.0
        
        for i in range(start_idx + 1, min(start_idx + 31, len(data))):
            candle = data[i]
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']
            
            if signal.signal == SignalType.BUY:
                current_pnl = (close_price - signal.entry_price) / pip_value
                adverse_move = (low_price - signal.entry_price) / pip_value
                favorable_move = (high_price - signal.entry_price) / pip_value
                
                max_adverse = min(max_adverse, adverse_move)
                max_favorable = max(max_favorable, favorable_move)
                
                if low_price <= signal.stop_loss:
                    trade.exit_price = signal.stop_loss
                    trade.exit_time = candle['time']
                    trade.status = "STOPPED"
                    trade.pnl_pips = (signal.stop_loss - signal.entry_price) / pip_value
                    break
                elif high_price >= signal.take_profit:
                    trade.exit_price = signal.take_profit
                    trade.exit_time = candle['time']
                    trade.status = "TARGET"
                    trade.pnl_pips = (signal.take_profit - signal.entry_price) / pip_value
                    break
            
            elif signal.signal == SignalType.SELL:
                current_pnl = (signal.entry_price - close_price) / pip_value
                adverse_move = (high_price - signal.entry_price) / pip_value
                favorable_move = (signal.entry_price - low_price) / pip_value
                
                max_adverse = min(max_adverse, -adverse_move)
                max_favorable = max(max_favorable, favorable_move)
                
                if high_price >= signal.stop_loss:
                    trade.exit_price = signal.stop_loss
                    trade.exit_time = candle['time']
                    trade.status = "STOPPED"
                    trade.pnl_pips = (signal.entry_price - signal.stop_loss) / pip_value
                    break
                elif low_price <= signal.take_profit:
                    trade.exit_price = signal.take_profit
                    trade.exit_time = candle['time']
                    trade.status = "TARGET"
                    trade.pnl_pips = (signal.entry_price - signal.take_profit) / pip_value
                    break
        
        if trade.status == "OPEN":
            trade.exit_price = data[min(start_idx + 30, len(data) - 1)]['close']
            trade.exit_time = data[min(start_idx + 30, len(data) - 1)]['time']
            trade.status = "TIMEOUT"
            
            if signal.signal == SignalType.BUY:
                trade.pnl_pips = (trade.exit_price - signal.entry_price) / pip_value
            else:
                trade.pnl_pips = (signal.entry_price - trade.exit_price) / pip_value
        
        trade.max_adverse_excursion = max_adverse
        trade.max_favorable_excursion = max_favorable
        
        if trade.exit_time:
            entry_dt = datetime.fromisoformat(trade.entry_time.replace('Z', '+00:00'))
            exit_dt = datetime.fromisoformat(trade.exit_time.replace('Z', '+00:00'))
            trade.duration_minutes = int((exit_dt - entry_dt).total_seconds() / 60)
        
        trade.pnl = trade.pnl_pips * pip_value * trade.volume * 100000
        
        return trade
    
    def run_backtest(self, historical_data: List[Dict], symbol: str = "EURUSD", 
                    pip_value: float = 0.0001, lookback_window: int = 100) -> BacktestResults:
        trades = []
        balance = self.initial_balance
        equity_curve = [balance]
        peak_balance = balance
        max_drawdown = 0.0
        
        self.logger.info(f"Starting backtest with {len(historical_data)} data points")
        
        for i in range(lookback_window, len(historical_data) - 30):
            window_data = historical_data[i-lookback_window:i+1]
            
            try:
                signal = self.strategy_ensemble.get_ensemble_signal(window_data)
                
                if signal and signal.confidence >= 0.6:
                    position_size = self.calculate_position_size(
                        signal.entry_price, signal.stop_loss, balance, pip_value
                    )
                    
                    trade = self.simulate_trade_execution(signal, historical_data, i, pip_value)
                    trade.symbol = symbol
                    trade.volume = position_size
                    trade.pnl = trade.pnl_pips * pip_value * position_size * 100000
                    
                    trades.append(trade)
                    balance += trade.pnl
                    equity_curve.append(balance)
                    
                    if balance > peak_balance:
                        peak_balance = balance
                    
                    current_drawdown = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                    self.logger.info(f"Trade {len(trades)}: {trade.signal_type.value} "
                                   f"{trade.status} PnL: {trade.pnl:.2f} "
                                   f"({trade.pnl_pips:.1f} pips)")
            
            except Exception as e:
                self.logger.error(f"Error processing data at index {i}: {e}")
                continue
        
        return self._calculate_results(trades, max_drawdown)
    
    def _calculate_results(self, trades: List[Trade], max_drawdown: float) -> BacktestResults:
        if not trades:
            return BacktestResults()
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pips = sum(t.pnl_pips for t in trades)
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade_duration = sum(t.duration_minutes for t in trades) / len(trades)
        
        pnl_series = np.array([t.pnl for t in trades])
        sharpe_ratio = (np.mean(pnl_series) / np.std(pnl_series)) * np.sqrt(252) if np.std(pnl_series) > 0 else 0
        
        results = BacktestResults(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades) * 100,
            total_pnl=total_pnl,
            total_pnl_pips=total_pnl_pips,
            max_drawdown=max_drawdown * 100,
            avg_trade_duration=avg_trade_duration,
            avg_winning_trade=gross_profit / len(winning_trades) if winning_trades else 0,
            avg_losing_trade=gross_loss / len(losing_trades) if losing_trades else 0,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            trades=trades
        )
        
        return results
    
    def print_results(self, results: BacktestResults):
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.1f}%")
        print(f"Total P&L: ${results.total_pnl:.2f}")
        print(f"Total P&L (Pips): {results.total_pnl_pips:.1f}")
        print(f"Max Drawdown: {results.max_drawdown:.1f}%")
        print(f"Average Trade Duration: {results.avg_trade_duration:.1f} minutes")
        print(f"Average Winning Trade: ${results.avg_winning_trade:.2f}")
        print(f"Average Losing Trade: ${results.avg_losing_trade:.2f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print("="*60)
        
        if results.total_pnl > 0 and results.win_rate >= 50 and results.max_drawdown < 20:
            print("✅ STRATEGY APPROVED FOR LIVE TRADING")
        else:
            print("❌ STRATEGY NEEDS OPTIMIZATION")
        print("="*60)
        
    def run_comprehensive_backtest(self, symbols=None):
        """Run backtest on all available symbols"""
        if symbols is None:
            symbols = ["EURCAD#", "USDJPY#", "AUDCAD#", "EURAUD#", "NZDCHF#", "NZDJPY#"]
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")
            
            # Determine pip value based on symbol
            if "JPY" in symbol:
                pip_value = 0.01
            else:
                pip_value = 0.0001
                
            historical_data = self.get_historical_data(symbol, "M5", 500)
            
            if historical_data:
                results = self.run_backtest(historical_data, symbol, pip_value)
                all_results[symbol] = results
                
                print(f"\n--- Results for {symbol} ---")
                self.print_results(results)
            else:
                logger.warning(f"No data available for {symbol}")
        
        return all_results
        
    def save_results(self, results_dict, filename=None):
        """Save backtest results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for symbol, results in results_dict.items():
            json_results[symbol] = {
                "total_trades": results.total_trades,
                "winning_trades": results.winning_trades,
                "losing_trades": results.losing_trades,
                "win_rate": results.win_rate,
                "total_pnl": results.total_pnl,
                "total_pnl_pips": results.total_pnl_pips,
                "max_drawdown": results.max_drawdown,
                "avg_trade_duration": results.avg_trade_duration,
                "avg_winning_trade": results.avg_winning_trade,
                "avg_losing_trade": results.avg_losing_trade,
                "profit_factor": float(results.profit_factor) if results.profit_factor != float('inf') else 999.0,
                "sharpe_ratio": float(results.sharpe_ratio),
                "approved_for_live": bool(results.total_pnl > 0 and results.win_rate >= 50 and results.max_drawdown < 20)
            }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename

if __name__ == "__main__":
    backtester = Backtester()
    results = backtester.run_comprehensive_backtest()
    backtester.save_results(results)