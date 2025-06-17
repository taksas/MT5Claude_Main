#!/usr/bin/env python3
"""
Automated Forex Trading System with Multiple Strategies
- RSI Strategy: Buy when oversold (RSI < 30), sell when overbought (RSI > 70)
- MACD Strategy: Buy on bullish crossover, sell on bearish crossover
- Bollinger Bands Strategy: Buy at lower band, sell at upper band
- Moving Average Strategy: Buy when price crosses above MA, sell when crosses below
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5API:
    def __init__(self, host: str = "172.28.144.1", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        
    def get_status(self) -> Dict:
        response = requests.get(f"{self.base_url}/status/mt5")
        response.raise_for_status()
        return response.json()
    
    def get_account_info(self) -> Dict:
        response = requests.get(f"{self.base_url}/account/")
        response.raise_for_status()
        return response.json()
        
    def get_tradable_symbols(self) -> List[str]:
        response = requests.get(f"{self.base_url}/market/symbols/tradable")
        response.raise_for_status()
        return response.json()
        
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        response = requests.post(f"{self.base_url}/market/history", json=data)
        response.raise_for_status()
        return response.json()
        
    def place_order(self, symbol: str, order_type: int, volume: float, sl: float = None, tp: float = None) -> Dict:
        order_data = {
            "action": "TRADE_ACTION_DEAL",
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "comment": "Automated trading"
        }
        if sl:
            order_data["sl"] = sl
        if tp:
            order_data["tp"] = tp
            
        response = requests.post(f"{self.base_url}/trading/orders", json=order_data)
        response.raise_for_status()
        return response.json()
    
    def get_positions(self) -> List[Dict]:
        response = requests.get(f"{self.base_url}/trading/positions")
        response.raise_for_status()
        return response.json()
    
    def close_position(self, ticket: int) -> Dict:
        response = requests.delete(f"{self.base_url}/trading/positions/{ticket}", 
                                 json={"deviation": 20})
        response.raise_for_status()
        return response.json()

class TechnicalIndicators:
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.values, sma.values, lower_band.values
    
    @staticmethod
    def moving_average(prices: np.ndarray, period: int = 20) -> np.ndarray:
        return pd.Series(prices).rolling(window=period).mean().values

class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
        
    def should_buy(self, data: pd.DataFrame) -> bool:
        raise NotImplementedError
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        raise NotImplementedError
    
    def calculate_stop_loss(self, entry_price: float, is_buy: bool) -> float:
        # Default 1% stop loss
        if is_buy:
            return entry_price * 0.99
        else:
            return entry_price * 1.01
    
    def calculate_take_profit(self, entry_price: float, is_buy: bool) -> float:
        # Default 2% take profit (2:1 risk-reward ratio)
        if is_buy:
            return entry_price * 1.02
        else:
            return entry_price * 0.98

class RSIStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("RSI Strategy")
        
    def should_buy(self, data: pd.DataFrame) -> bool:
        if len(data) < 15:
            return False
        rsi = TechnicalIndicators.rsi(data['close'].values)
        return rsi[-1] < 30 and rsi[-2] >= 30  # RSI just crossed below 30
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        if len(data) < 15:
            return False
        rsi = TechnicalIndicators.rsi(data['close'].values)
        return rsi[-1] > 70 and rsi[-2] <= 70  # RSI just crossed above 70

class MACDStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MACD Strategy")
        
    def should_buy(self, data: pd.DataFrame) -> bool:
        if len(data) < 27:
            return False
        macd, signal, _ = TechnicalIndicators.macd(data['close'].values)
        return macd[-1] > signal[-1] and macd[-2] <= signal[-2]  # MACD crossed above signal
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        if len(data) < 27:
            return False
        macd, signal, _ = TechnicalIndicators.macd(data['close'].values)
        return macd[-1] < signal[-1] and macd[-2] >= signal[-2]  # MACD crossed below signal

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Bollinger Bands Strategy")
        
    def should_buy(self, data: pd.DataFrame) -> bool:
        if len(data) < 21:
            return False
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data['close'].values)
        current_price = data['close'].iloc[-1]
        return current_price <= lower[-1] and data['close'].iloc[-2] > lower[-2]
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        if len(data) < 21:
            return False
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data['close'].values)
        current_price = data['close'].iloc[-1]
        return current_price >= upper[-1] and data['close'].iloc[-2] < upper[-2]

class MovingAverageStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Moving Average Strategy")
        
    def should_buy(self, data: pd.DataFrame) -> bool:
        if len(data) < 21:
            return False
        ma = TechnicalIndicators.moving_average(data['close'].values)
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        return current_price > ma[-1] and prev_price <= ma[-2]
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        if len(data) < 21:
            return False
        ma = TechnicalIndicators.moving_average(data['close'].values)
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        return current_price < ma[-1] and prev_price >= ma[-2]

class ForexTrader:
    def __init__(self):
        self.api = MT5API()
        self.strategies = [
            RSIStrategy(),
            MACDStrategy(),
            BollingerBandsStrategy(),
            MovingAverageStrategy()
        ]
        self.active_trades = {}
        self.lot_size = 0.01
        
    def test_connection(self) -> bool:
        try:
            status = self.api.get_status()
            logger.info(f"MT5 Connection Status: {status.get('connected', False)}")
            return status.get('connected', False)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account_status(self) -> Dict:
        try:
            account = self.api.get_account_info()
            logger.info(f"Account Balance: {account.get('balance', 0)} {account.get('currency', 'USD')}")
            return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def backtest_strategy(self, strategy: TradingStrategy, symbol: str, data: pd.DataFrame) -> Dict:
        """Backtest a strategy on historical data"""
        balance = 10000  # Starting balance
        trades = []
        current_position = None
        
        for i in range(50, len(data)):  # Start after enough data for indicators
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            if current_position is None:
                # Look for entry signals
                if strategy.should_buy(current_data):
                    sl = strategy.calculate_stop_loss(current_price, True)
                    tp = strategy.calculate_take_profit(current_price, True)
                    current_position = {
                        'type': 'buy',
                        'entry_price': current_price,
                        'sl': sl,
                        'tp': tp,
                        'entry_time': i
                    }
                elif strategy.should_sell(current_data):
                    sl = strategy.calculate_stop_loss(current_price, False)
                    tp = strategy.calculate_take_profit(current_price, False)
                    current_position = {
                        'type': 'sell',
                        'entry_price': current_price,
                        'sl': sl,
                        'tp': tp,
                        'entry_time': i
                    }
            else:
                # Check exit conditions
                should_exit = False
                exit_reason = None
                
                if current_position['type'] == 'buy':
                    if current_price <= current_position['sl']:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price >= current_position['tp']:
                        should_exit = True
                        exit_reason = 'take_profit'
                    elif strategy.should_sell(current_data):
                        should_exit = True
                        exit_reason = 'signal'
                else:  # sell position
                    if current_price >= current_position['sl']:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price <= current_position['tp']:
                        should_exit = True
                        exit_reason = 'take_profit'
                    elif strategy.should_buy(current_data):
                        should_exit = True
                        exit_reason = 'signal'
                
                if should_exit:
                    if current_position['type'] == 'buy':
                        profit = (current_price - current_position['entry_price']) * 100000 * self.lot_size
                    else:
                        profit = (current_position['entry_price'] - current_price) * 100000 * self.lot_size
                    
                    balance += profit
                    trades.append({
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'type': current_position['type'],
                        'exit_reason': exit_reason,
                        'duration': i - current_position['entry_time']
                    })
                    current_position = None
        
        # Calculate performance metrics
        if trades:
            profits = [t['profit'] for t in trades]
            win_rate = len([p for p in profits if p > 0]) / len(profits)
            avg_profit = np.mean(profits)
            max_drawdown = min(0, min(profits))
            total_return = (balance - 10000) / 10000 * 100
        else:
            win_rate = 0
            avg_profit = 0
            max_drawdown = 0
            total_return = 0
        
        return {
            'strategy': strategy.name,
            'symbol': symbol,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_balance': balance,
            'trades': trades
        }
    
    def select_best_strategy(self, symbol: str) -> Optional[TradingStrategy]:
        """Test all strategies and select the best performing one"""
        try:
            # Get historical data for backtesting
            historical_data = self.api.get_historical_data(symbol, "M5", 500)
            if not historical_data:
                return None
                
            df = pd.DataFrame(historical_data)
            df['time'] = pd.to_datetime(df['time'])
            
            results = []
            for strategy in self.strategies:
                backtest_result = self.backtest_strategy(strategy, symbol, df)
                results.append(backtest_result)
                logger.info(f"Backtest {strategy.name}: Win Rate: {backtest_result['win_rate']:.2%}, "
                          f"Total Return: {backtest_result['total_return']:.2f}%, "
                          f"Trades: {backtest_result['total_trades']}")
            
            # Select strategy with best performance (combination of win rate and return)
            best_strategy = None
            best_score = -float('inf')
            
            for i, result in enumerate(results):
                if result['total_trades'] < 5:  # Need minimum trades for confidence
                    continue
                    
                # Score based on win rate, return, and number of trades
                score = (result['win_rate'] * 0.4 + 
                        (result['total_return'] / 100) * 0.5 + 
                        min(result['total_trades'] / 20, 1.0) * 0.1)
                
                if score > best_score and result['total_return'] > 0:
                    best_score = score
                    best_strategy = self.strategies[i]
            
            if best_strategy:
                logger.info(f"Selected best strategy: {best_strategy.name}")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error in strategy selection: {e}")
            return None
    
    def execute_trade(self, symbol: str, strategy: TradingStrategy, signal_type: str) -> bool:
        """Execute a trade based on strategy signal"""
        try:
            # Get current price for stop loss and take profit calculation
            recent_data = self.api.get_historical_data(symbol, "M1", 2)
            if not recent_data:
                return False
                
            current_price = recent_data[-1]['close']
            
            if signal_type == 'buy':
                order_type = 0  # BUY
                sl = strategy.calculate_stop_loss(current_price, True)
                tp = strategy.calculate_take_profit(current_price, True)
            else:
                order_type = 1  # SELL
                sl = strategy.calculate_stop_loss(current_price, False)
                tp = strategy.calculate_take_profit(current_price, False)
            
            # Place the order
            result = self.api.place_order(symbol, order_type, self.lot_size, sl, tp)
            
            if result.get('retcode') == 10009:  # Success
                logger.info(f"Trade executed: {signal_type.upper()} {symbol} at {current_price}, "
                          f"SL: {sl}, TP: {tp}")
                
                # Store trade information
                self.active_trades[result.get('order')] = {
                    'symbol': symbol,
                    'strategy': strategy.name,
                    'type': signal_type,
                    'entry_price': current_price,
                    'sl': sl,
                    'tp': tp,
                    'entry_time': datetime.now()
                }
                return True
            else:
                logger.error(f"Trade execution failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor open positions and close them after 5-30 minutes"""
        try:
            positions = self.api.get_positions()
            current_time = datetime.now()
            
            for position in positions:
                ticket = position.get('ticket')
                if ticket in self.active_trades:
                    trade_info = self.active_trades[ticket]
                    time_elapsed = (current_time - trade_info['entry_time']).total_seconds() / 60
                    
                    # Close position after 5-30 minutes based on profit
                    should_close = False
                    
                    if time_elapsed >= 30:  # Max hold time
                        should_close = True
                        reason = "max_time"
                    elif time_elapsed >= 5:  # Min hold time
                        # Check if position is profitable
                        current_profit = position.get('profit', 0)
                        if current_profit > 0:
                            should_close = True
                            reason = "profitable_exit"
                    
                    if should_close:
                        self.api.close_position(ticket)
                        logger.info(f"Position {ticket} closed after {time_elapsed:.1f} minutes ({reason})")
                        del self.active_trades[ticket]
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def run_trading_session(self):
        """Main trading loop"""
        logger.info("Starting automated forex trading session...")
        
        # Test connection
        if not self.test_connection():
            logger.error("Cannot connect to MT5 API. Exiting.")
            return
        
        # Get account info
        account = self.get_account_status()
        if not account:
            logger.error("Cannot get account information. Exiting.")
            return
        
        # Get tradable symbols
        try:
            symbols = self.api.get_tradable_symbols()
            if not symbols:
                logger.error("No tradable symbols found. Exiting.")
                return
            
            # Focus on major forex pairs
            major_pairs = [s for s in symbols if s in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']]
            if not major_pairs:
                major_pairs = symbols[:3]  # Use first 3 available
                
            logger.info(f"Trading symbols: {major_pairs}")
            
        except Exception as e:
            logger.error(f"Error getting tradable symbols: {e}")
            return
        
        # Main trading loop
        session_start = datetime.now()
        max_session_time = 120  # 2 hours max session
        
        while (datetime.now() - session_start).total_seconds() < max_session_time * 60:
            try:
                # Monitor existing positions
                self.monitor_positions()
                
                # Look for new trading opportunities
                for symbol in major_pairs:
                    if len(self.active_trades) >= 3:  # Max 3 concurrent trades
                        break
                    
                    # Skip if we already have an active trade for this symbol
                    if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                        continue
                    
                    # Select best strategy for this symbol
                    best_strategy = self.select_best_strategy(symbol)
                    if not best_strategy:
                        continue
                    
                    # Get recent data for signal generation
                    recent_data = self.api.get_historical_data(symbol, "M5", 100)
                    if not recent_data:
                        continue
                    
                    df = pd.DataFrame(recent_data)
                    df['time'] = pd.to_datetime(df['time'])
                    
                    # Check for trading signals
                    if best_strategy.should_buy(df):
                        logger.info(f"BUY signal for {symbol} using {best_strategy.name}")
                        self.execute_trade(symbol, best_strategy, 'buy')
                    elif best_strategy.should_sell(df):
                        logger.info(f"SELL signal for {symbol} using {best_strategy.name}")
                        self.execute_trade(symbol, best_strategy, 'sell')
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)
        
        logger.info("Trading session completed")

if __name__ == "__main__":
    trader = ForexTrader()
    trader.run_trading_session()