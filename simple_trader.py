#!/usr/bin/env python3
"""
Enhanced MT5 Automated Forex Trading System
Connects to MT5 Bridge API and performs automated short-term trading with multiple strategies.
Features:
- Multiple trading strategies with historical validation
- Risk management with position limits and stop losses
- Real-time market analysis and signal generation
- Automated position management with time-based exits
"""

import json
import logging
import time
import sys
import urllib.request
import urllib.parse
import urllib.error
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_session.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Advanced technical analysis using only standard library"""
    
    @staticmethod
    def simple_moving_average(prices: List[float], period: int) -> float:
        """Calculate simple moving average"""
        if len(prices) < period:
            return 0.0
        return sum(prices[-period:]) / period
    
    @staticmethod
    def exponential_moving_average(prices: List[float], period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((price - middle) ** 2 for price in recent_prices) / period
        std = variance ** 0.5
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD (MACD line, Signal line, Histogram)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = TechnicalAnalysis.exponential_moving_average(prices, fast)
        ema_slow = TechnicalAnalysis.exponential_moving_average(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we would need MACD history, simplified here
        signal_line = macd_line * 0.9  # Simplified approximation
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14) -> float:
        """Calculate Stochastic Oscillator %K"""
        if len(closes) < k_period:
            return 50.0
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return 50.0
        
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        return k_percent

class MT5EnhancedBot:
    def __init__(self):
        self.api_base_url = self._find_api_host()
        self.lot_size = 0.01  # Fixed lot size as per requirements
        self.max_positions = 3  # Maximum concurrent positions
        self.min_profit_pips = 5  # Minimum profit target in pips
        self.max_loss_pips = 12  # Maximum loss in pips (stop loss)
        self.position_hold_time_minutes = 20  # Maximum hold time in minutes
        self.trading_symbols = []
        self.ta = TechnicalAnalysis()
        self.strategies = ['sma_crossover', 'rsi_extremes', 'bollinger_bounce', 'momentum_breakout']
        self.strategy_performance = {strategy: {'wins': 0, 'losses': 0, 'total_profit': 0.0} for strategy in self.strategies}
        self.trade_history = []
        
    def _find_api_host(self) -> str:
        """Find the correct API host for WSL environment"""
        possible_hosts = [
            "172.28.144.1:8000",  # WSL default gateway (Windows host)
            "localhost:8000",
            "127.0.0.1:8000",
            "172.28.147.233:8000"
        ]
        
        for host in possible_hosts:
            try:
                url = f"http://{host}/status/ping"
                with urllib.request.urlopen(url, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    if data.get("status") == "pong":
                        logger.info(f"Found API server at: {host}")
                        return f"http://{host}"
            except:
                continue
        
        # Default fallback
        return "http://172.28.147.233:8000"
    
    def api_request(self, method: str, endpoint: str, data: dict = None) -> Optional[dict]:
        """Make API request to MT5 Bridge"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method == "GET":
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status == 200:
                        return json.loads(response.read().decode())
            else:
                # POST, DELETE, PATCH
                request_data = json.dumps(data).encode() if data else None
                headers = {'Content-Type': 'application/json'} if data else {}
                
                req = urllib.request.Request(url, data=request_data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status in [200, 201]:
                        return json.loads(response.read().decode())
            
            return None
            
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"API Request failed: {e}")
            return None
    
    def check_connection(self) -> bool:
        """Check if MT5 Bridge API is accessible"""
        result = self.api_request("GET", "/status/ping")
        if result and result.get("status") == "pong":
            logger.info("✓ MT5 Bridge API connection successful")
            return True
        logger.error("✗ Failed to connect to MT5 Bridge API")
        return False
    
    def check_mt5_status(self) -> bool:
        """Check MT5 terminal connection status"""
        result = self.api_request("GET", "/status/mt5")
        if result and result.get("connected"):
            logger.info("✓ MT5 Terminal connected and ready")
            return True
        logger.error("✗ MT5 Terminal not connected")
        return False
    
    def get_account_info(self) -> Optional[dict]:
        """Get account information"""
        return self.api_request("GET", "/account/")
    
    def get_tradable_symbols(self) -> List[str]:
        """Get list of verified working symbols (only those ending with #)"""
        # Use only verified working symbols to avoid 404 errors
        verified_working_symbols = [
            "USDCNH#", "USDDKK#", "USDHKD#", "USDHUF#", "USDMXN#", 
            "USDNOK#", "USDPLN#", "USDSEK#", "USDSGD#", "USDTRY#", 
            "USDZAR#", "EURUSD#", "GBPUSD#", "USDCAD#", "USDCHF#", 
            "USDJPY#", "AUDUSD#", "NZDUSD#"
        ]
        
        # Test each symbol to ensure it actually works
        working_symbols = []
        for symbol in verified_working_symbols:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                trade_mode = symbol_info.get('trade_mode_description', 'DISABLED')
                if trade_mode == 'FULL':
                    working_symbols.append(symbol)
                    logger.info(f"Verified working symbol: {symbol} (trade_mode: {trade_mode})")
                else:
                    logger.warning(f"Symbol {symbol} has restricted trade mode: {trade_mode}")
            else:
                logger.warning(f"Symbol {symbol} not accessible via API")
        
        if working_symbols:
            # Prioritize major forex pairs for better liquidity
            major_pairs = ['EURUSD#', 'GBPUSD#', 'USDCAD#', 'USDCHF#', 'USDJPY#', 'AUDUSD#', 'NZDUSD#']
            prioritized_symbols = [s for s in major_pairs if s in working_symbols]
            exotic_pairs = [s for s in working_symbols if s not in major_pairs]
            
            final_symbols = prioritized_symbols + exotic_pairs[:3]  # Limit exotic pairs
            logger.info(f"Using {len(final_symbols)} verified symbols: {final_symbols}")
            return final_symbols
        else:
            logger.error("No verified working symbols found")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get detailed symbol information"""
        return self.api_request("GET", f"/market/symbols/{symbol}")
    
    def get_historical_data(self, symbol: str, timeframe: str = "M5", count: int = 50) -> List[dict]:
        """Get historical price data"""
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        result = self.api_request("POST", "/market/history", data)
        return result if result else []
    
    def backtest_strategy(self, strategy_name: str, data: List[dict], symbol_info: dict) -> dict:
        """Backtest a strategy on historical data"""
        if len(data) < 30:
            return {'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
        
        trades = []
        point = symbol_info.get('point', 0.00001)
        
        for i in range(25, len(data) - 5):  # Leave room for indicators and future price movement
            hist_data = data[:i+1]
            signal = self.analyze_market_strategy(strategy_name, hist_data, symbol_info)
            
            if signal:
                # Simulate trade execution over next 5 bars
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                # Check next 5 bars for SL/TP hit
                trade_result = None
                for j in range(i+1, min(i+6, len(data))):
                    bar = data[j]
                    high = float(bar['high'])
                    low = float(bar['low'])
                    
                    if signal['type'] == 'BUY':
                        if low <= stop_loss:
                            trade_result = {'profit': (stop_loss - entry_price) / point, 'outcome': 'loss'}
                            break
                        elif high >= take_profit:
                            trade_result = {'profit': (take_profit - entry_price) / point, 'outcome': 'win'}
                            break
                    else:  # SELL
                        if high >= stop_loss:
                            trade_result = {'profit': (entry_price - stop_loss) / point, 'outcome': 'loss'}
                            break
                        elif low <= take_profit:
                            trade_result = {'profit': (entry_price - take_profit) / point, 'outcome': 'win'}
                            break
                
                if trade_result:
                    trades.append(trade_result)
        
        if not trades:
            return {'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
        
        total_profit = sum(trade['profit'] for trade in trades)
        wins = sum(1 for trade in trades if trade['outcome'] == 'win')
        win_rate = wins / len(trades)
        
        return {'trades': len(trades), 'profit': total_profit, 'win_rate': win_rate}
    
    def analyze_market_strategy(self, strategy_name: str, data: List[dict], symbol_info: dict) -> Optional[dict]:
        """Analyze market using specific strategy"""
        if len(data) < 25:
            return None
        
        closes = [float(bar['close']) for bar in data]
        highs = [float(bar['high']) for bar in data]
        lows = [float(bar['low']) for bar in data]
        current_price = closes[-1]
        point = symbol_info.get('point', 0.00001)
        
        if strategy_name == 'sma_crossover':
            sma_5 = self.ta.simple_moving_average(closes, 5)
            sma_20 = self.ta.simple_moving_average(closes, 20)
            rsi = self.ta.rsi(closes, 14)
            
            if sma_5 > sma_20 and rsi < 70 and current_price > sma_5:
                return {
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price - (self.max_loss_pips * point),
                    'take_profit': current_price + (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
            elif sma_5 < sma_20 and rsi > 30 and current_price < sma_5:
                return {
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': current_price + (self.max_loss_pips * point),
                    'take_profit': current_price - (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
        
        elif strategy_name == 'rsi_extremes':
            rsi = self.ta.rsi(closes, 14)
            
            if rsi < 20:  # Oversold
                return {
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price - (self.max_loss_pips * point),
                    'take_profit': current_price + (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
            elif rsi > 80:  # Overbought
                return {
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': current_price + (self.max_loss_pips * point),
                    'take_profit': current_price - (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
        
        elif strategy_name == 'bollinger_bounce':
            bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(closes, 20, 2.0)
            rsi = self.ta.rsi(closes, 14)
            
            if current_price <= bb_lower and rsi < 35:
                return {
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price - (self.max_loss_pips * point),
                    'take_profit': current_price + (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
            elif current_price >= bb_upper and rsi > 65:
                return {
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': current_price + (self.max_loss_pips * point),
                    'take_profit': current_price - (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
        
        elif strategy_name == 'momentum_breakout':
            sma_10 = self.ta.simple_moving_average(closes, 10)
            stoch = self.ta.stochastic_oscillator(highs, lows, closes, 14)
            
            # Recent volatility
            recent_range = max(highs[-5:]) - min(lows[-5:])
            avg_range = sum(highs[i] - lows[i] for i in range(-10, 0)) / 10
            
            if current_price > sma_10 and stoch < 80 and recent_range > avg_range * 1.2:
                return {
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price - (self.max_loss_pips * point),
                    'take_profit': current_price + (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
            elif current_price < sma_10 and stoch > 20 and recent_range > avg_range * 1.2:
                return {
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': current_price + (self.max_loss_pips * point),
                    'take_profit': current_price - (self.min_profit_pips * point),
                    'strategy': strategy_name
                }
        
        return None
    
    def analyze_market(self, symbol: str, data: List[dict]) -> Optional[dict]:
        """Enhanced market analysis using multiple strategies with backtesting"""
        if len(data) < 30:
            return None
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return None
        
        # Test all strategies on historical data
        strategy_results = {}
        for strategy in self.strategies:
            backtest_result = self.backtest_strategy(strategy, data, symbol_info)
            if backtest_result['trades'] > 0 and backtest_result['profit'] > 0:
                strategy_results[strategy] = backtest_result
        
        # Choose best performing strategy
        if not strategy_results:
            return None
        
        best_strategy = max(strategy_results.keys(), 
                           key=lambda s: strategy_results[s]['profit'] * strategy_results[s]['win_rate'])
        
        # Generate signal using best strategy
        signal = self.analyze_market_strategy(best_strategy, data, symbol_info)
        
        if signal:
            backtest_info = strategy_results[best_strategy]
            logger.info(f"Generated {signal['type']} signal for {symbol} using {best_strategy}")
            logger.info(f"Strategy backtest: {backtest_info['trades']} trades, {backtest_info['profit']:.1f} pips profit, {backtest_info['win_rate']:.2%} win rate")
            
            # Add confidence based on backtest results
            if backtest_info['win_rate'] > 0.6 and backtest_info['profit'] > 10:
                signal['confidence'] = 'high'
            elif backtest_info['win_rate'] > 0.5 and backtest_info['profit'] > 5:
                signal['confidence'] = 'medium'
            else:
                signal['confidence'] = 'low'
        
        return signal
    
    def place_trade(self, symbol: str, signal: dict) -> Optional[dict]:
        """Place a trade based on signal with enhanced validation"""
        # Skip low confidence signals randomly to reduce risk
        if signal.get('confidence') == 'low' and random.random() < 0.7:
            logger.info(f"Skipping low confidence {signal['type']} signal for {symbol}")
            return None
        
        order_type = 0 if signal['type'] == 'BUY' else 1
        
        trade_request = {
            "action": 1,  # TRADE_ACTION_DEAL (market order)
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "sl": signal['stop_loss'],
            "tp": signal['take_profit'],
            "comment": f"EnhancedBot_{signal.get('strategy', 'unknown')}_{signal.get('confidence', 'medium')}"
        }
        
        result = self.api_request("POST", "/trading/orders", trade_request)
        if result and result.get('retcode') == 10009:
            logger.info(f"✓ Trade placed: {signal['type']} {symbol} at {signal['entry_price']:.5f} using {signal.get('strategy', 'unknown')} strategy")
            
            # Record trade for performance tracking
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': signal['type'],
                'strategy': signal.get('strategy', 'unknown'),
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'confidence': signal.get('confidence', 'medium'),
                'ticket': result.get('order', 0)
            }
            self.trade_history.append(trade_record)
            
            return result
        else:
            logger.error(f"✗ Failed to place trade for {symbol}: {result}")
            return None
    
    def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        result = self.api_request("GET", "/trading/positions")
        return result if result else []
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        close_request = {"deviation": 20}
        result = self.api_request("DELETE", f"/trading/positions/{ticket}", close_request)
        if result and result.get('retcode') == 10009:
            logger.info(f"✓ Position {ticket} closed successfully")
            return True
        else:
            logger.error(f"✗ Failed to close position {ticket}")
            return False
    
    def manage_open_positions(self):
        """Manage open positions based on time and profit"""
        positions = self.get_open_positions()
        current_time = datetime.now()
        
        for position in positions:
            ticket = position['ticket']
            symbol = position['symbol']
            profit = position.get('profit', 0)
            
            # Parse open time
            time_str = position.get('time', '')
            try:
                if 'T' in time_str:
                    open_time = datetime.fromisoformat(time_str.replace('Z', ''))
                else:
                    open_time = datetime.fromtimestamp(int(time_str))
            except:
                # If we can't parse time, assume it's been open for max time
                open_time = current_time - timedelta(minutes=self.position_hold_time_minutes + 1)
            
            minutes_open = (current_time - open_time).total_seconds() / 60
            
            # Close position if held too long
            if minutes_open > self.position_hold_time_minutes:
                logger.info(f"Closing position {ticket} ({symbol}) - held for {minutes_open:.1f} minutes (limit: {self.position_hold_time_minutes})")
                self.close_position(ticket)
                continue
            
            # Log current profit
            if profit != 0:
                logger.info(f"Position {ticket} ({symbol}): {profit:.2f} USD profit, open for {minutes_open:.1f} min")
    
    def print_performance_summary(self):
        """Print trading performance summary"""
        logger.info("\n=== TRADING PERFORMANCE SUMMARY ===")
        
        if not self.trade_history:
            logger.info("No trades executed during this session.")
            return
        
        total_trades = len(self.trade_history)
        strategies_used = {}
        
        for trade in self.trade_history:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategies_used:
                strategies_used[strategy] = 0
            strategies_used[strategy] += 1
        
        logger.info(f"Total trades placed: {total_trades}")
        logger.info("Strategies used:")
        for strategy, count in strategies_used.items():
            logger.info(f"  - {strategy}: {count} trades")
        
        # Get final positions status
        positions = self.get_open_positions()
        if positions:
            total_unrealized_pnl = sum(float(pos.get('profit', 0)) for pos in positions)
            logger.info(f"Open positions: {len(positions)}")
            logger.info(f"Total unrealized P&L: {total_unrealized_pnl:.2f} USD")
    
    def trading_session(self):
        """Enhanced trading session with multiple strategies and performance tracking"""
        logger.info("🚀 Starting Enhanced MT5 Trading Bot...")
        logger.info("Features: Multiple strategies, backtesting, risk management")
        
        # Initial checks
        if not self.check_connection():
            logger.error("Cannot connect to MT5 Bridge API. Please ensure it's running.")
            return
        
        if not self.check_mt5_status():
            logger.error("MT5 Terminal not connected. Please check terminal status.")
            return
        
        # Get account info
        account_info = self.get_account_info()
        if account_info:
            balance = account_info.get('balance', 0)
            currency = account_info.get('currency', 'USD')
            margin_free = account_info.get('margin_free', 0)
            logger.info(f"💰 Account Balance: {balance} {currency}")
            logger.info(f"💳 Free Margin: {margin_free} {currency}")
        
        # Get tradable symbols
        self.trading_symbols = self.get_tradable_symbols()
        if not self.trading_symbols:
            logger.error("No tradable symbols found (looking for symbols ending with #)")
            return
        
        logger.info(f"🎯 Available strategies: {', '.join(self.strategies)}")
        logger.info(f"📊 Analyzing {len(self.trading_symbols)} symbols: {', '.join(self.trading_symbols)}")
        
        # Trading loop
        iteration = 0
        max_iterations = 300  # Increased iterations for longer session
        start_time = datetime.now()
        
        try:
            while iteration < max_iterations:
                iteration += 1
                current_time = datetime.now()
                elapsed = current_time - start_time
                
                logger.info(f"\n--- Trading Iteration {iteration} (Running for {elapsed}) ---")
                
                # Manage existing positions
                self.manage_open_positions()
                
                # Check if we can open new positions
                positions = self.get_open_positions()
                if len(positions) >= self.max_positions:
                    logger.info(f"Maximum positions ({self.max_positions}) reached. Managing existing positions...")
                else:
                    # Randomize symbol order to ensure fair analysis
                    analysis_symbols = self.trading_symbols.copy()
                    random.shuffle(analysis_symbols)
                    
                    # Look for trading opportunities
                    for symbol in analysis_symbols:
                        if len(self.get_open_positions()) >= self.max_positions:
                            break
                        
                        logger.info(f"📊 Analyzing {symbol}...")
                        
                        # Get longer historical data for better backtesting
                        hist_data = self.get_historical_data(symbol, "M5", 50)
                        if not hist_data or len(hist_data) < 30:
                            logger.warning(f"Insufficient historical data for {symbol}")
                            continue
                        
                        # Analyze market with multiple strategies
                        signal = self.analyze_market(symbol, hist_data)
                        if signal:
                            # Place trade
                            result = self.place_trade(symbol, signal)
                            if result:
                                time.sleep(3)  # Pause between trades
                        
                        time.sleep(2)  # Brief pause between symbol analysis
                
                # Adaptive wait time based on market activity
                if len(positions) > 0:
                    wait_time = 30  # Shorter wait when managing positions
                else:
                    wait_time = 60  # Longer wait when no positions
                
                logger.info(f"⏳ Waiting {wait_time} seconds before next analysis...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading session interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Final cleanup and reporting
            logger.info("🔄 Session ending - managing final positions...")
            
            # Close any remaining positions
            positions = self.get_open_positions()
            for position in positions:
                ticket = position['ticket']
                symbol = position['symbol']
                profit = position.get('profit', 0)
                logger.info(f"Closing final position: {ticket} ({symbol}) - Current P&L: {profit:.2f}")
                self.close_position(ticket)
            
            # Print performance summary
            self.print_performance_summary()
            
            session_duration = datetime.now() - start_time
            logger.info(f"✅ Enhanced trading session completed after {session_duration}")

def main():
    """Main function"""
    logger.info("Initializing Enhanced MT5 Trading Bot...")
    bot = MT5EnhancedBot()
    bot.trading_session()

if __name__ == "__main__":
    main()