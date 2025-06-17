#!/usr/bin/env python3
"""
Advanced Automated Forex Trading System with Multiple Strategies
- Implements multiple trading strategies with real-time validation
- Combines trend following, mean reversion, and breakout strategies
- Includes comprehensive risk management and position monitoring
- Designed for short-term trades (5-30 minutes hold time)
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
import urllib.error
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add custom SUCCESS log level
logging.addLevelName(25, "SUCCESS")
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)
logging.Logger.success = success

class MT5API:
    def __init__(self, host: str = "172.28.144.1", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to MT5 API"""
        url = f"{self.base_url}{endpoint}"
        
        if method == "GET":
            if data:
                query_string = urllib.parse.urlencode(data)
                url = f"{url}?{query_string}"
            req = urllib.request.Request(url)
        else:
            json_data = json.dumps(data).encode('utf-8') if data else None
            req = urllib.request.Request(url, data=json_data)
            req.add_header('Content-Type', 'application/json')
            if method == "DELETE":
                req.get_method = lambda: 'DELETE'
            elif method == "PATCH":
                req.get_method = lambda: 'PATCH'
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8')
            logger.error(f"HTTP Error {e.code}: {error_msg}")
            raise Exception(f"HTTP {e.code}: {error_msg}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_status(self) -> Dict:
        return self._make_request("GET", "/status/mt5")
    
    def get_account_info(self) -> Dict:
        return self._make_request("GET", "/account/")
        
    def get_tradable_symbols(self) -> List[str]:
        return self._make_request("GET", "/market/symbols/tradable")
        
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        return self._make_request("POST", "/market/history", data)
        
    def place_order(self, symbol: str, order_type: int, volume: float, sl: float = None, tp: float = None) -> Dict:
        order_data = {
            "action": 1,  # TRADE_ACTION_DEAL
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "comment": "Advanced automated trading"
        }
        if sl:
            order_data["sl"] = sl
        if tp:
            order_data["tp"] = tp
            
        return self._make_request("POST", "/trading/orders", order_data)
    
    def get_positions(self) -> List[Dict]:
        return self._make_request("GET", "/trading/positions")
    
    def close_position(self, ticket: int) -> Dict:
        return self._make_request("DELETE", f"/trading/positions/{ticket}", {"deviation": 20})

class TechnicalAnalysis:
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        
        result = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            result.append(avg)
        return result
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        result = []
        
        # Start with SMA for first value
        first_ema = sum(prices[:period]) / period
        result.append(first_ema)
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (result[-1] * (1 - multiplier))
            result.append(ema_value)
        
        return result
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
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
        
        rsi_values = []
        for i in range(period - 1, len(gains)):
            avg_gain = sum(gains[i - period + 1:i + 1]) / period
            avg_loss = sum(losses[i - period + 1:i + 1]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands: (upper, middle, lower)"""
        if len(prices) < period:
            return [], [], []
        
        sma_values = TechnicalAnalysis.sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(sma_values)):
            price_slice = prices[i:i + period]
            std = statistics.stdev(price_slice)
            
            upper_band.append(sma_values[i] + (std * std_dev))
            lower_band.append(sma_values[i] - (std * std_dev))
        
        return upper_band, sma_values, lower_band
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD: (macd_line, signal_line, histogram)"""
        if len(prices) < slow:
            return [], [], []
        
        ema_fast = TechnicalAnalysis.ema(prices, fast)
        ema_slow = TechnicalAnalysis.ema(prices, slow)
        
        # Align arrays - ema_slow is shorter
        align_offset = len(ema_fast) - len(ema_slow)
        if align_offset > 0:
            ema_fast = ema_fast[align_offset:]
        
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]
        signal_line = TechnicalAnalysis.ema(macd_line, signal)
        
        # Align histogram
        hist_offset = len(macd_line) - len(signal_line)
        if hist_offset > 0:
            macd_line = macd_line[hist_offset:]
        
        histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
        
        return macd_line, signal_line, histogram

class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.recent_signals = []
        self.performance_score = 0.0
        
    def analyze(self, prices: List[float], volumes: List[float] = None) -> Optional[str]:
        """Analyze price data and return BUY/SELL/HOLD signal"""
        raise NotImplementedError
    
    def get_confidence(self) -> float:
        """Return confidence level (0.0 to 1.0) for current signal"""
        return 0.5

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Trend Following")
        
    def analyze(self, prices: List[float], volumes: List[float] = None) -> Optional[str]:
        if len(prices) < 50:
            return None
        
        # Multiple timeframe trend analysis
        ema_fast = TechnicalAnalysis.ema(prices, 12)
        ema_slow = TechnicalAnalysis.ema(prices, 26)
        sma_long = TechnicalAnalysis.sma(prices, 50)
        
        if not ema_fast or not ema_slow or not sma_long:
            return None
        
        current_price = prices[-1]
        ema_fast_val = ema_fast[-1]
        ema_slow_val = ema_slow[-1]
        sma_long_val = sma_long[-1]
        
        # Strong trend conditions
        bullish_trend = (ema_fast_val > ema_slow_val and 
                        current_price > ema_fast_val and 
                        ema_slow_val > sma_long_val)
        
        bearish_trend = (ema_fast_val < ema_slow_val and 
                        current_price < ema_fast_val and 
                        ema_slow_val < sma_long_val)
        
        # Additional momentum check
        if len(prices) >= 5:
            price_momentum = (prices[-1] - prices[-5]) / prices[-5]
            if bullish_trend and price_momentum > 0.0001:  # 0.01% upward momentum
                self.confidence = 0.8
                return "BUY"
            elif bearish_trend and price_momentum < -0.0001:  # 0.01% downward momentum
                self.confidence = 0.8
                return "SELL"
        
        return None

class MeanReversionStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Mean Reversion")
        
    def analyze(self, prices: List[float], volumes: List[float] = None) -> Optional[str]:
        if len(prices) < 30:
            return None
        
        # Bollinger Bands for mean reversion
        upper, middle, lower = TechnicalAnalysis.bollinger_bands(prices, 20, 2.0)
        rsi_values = TechnicalAnalysis.rsi(prices, 14)
        
        if not upper or not rsi_values:
            return None
        
        current_price = prices[-1]
        current_rsi = rsi_values[-1]
        
        # Mean reversion conditions
        oversold_bb = current_price <= lower[-1]
        oversold_rsi = current_rsi <= 30
        
        overbought_bb = current_price >= upper[-1]
        overbought_rsi = current_rsi >= 70
        
        # Strong oversold - expect bounce up
        if oversold_bb and oversold_rsi:
            self.confidence = 0.85
            return "BUY"
        
        # Strong overbought - expect pullback
        if overbought_bb and overbought_rsi:
            self.confidence = 0.85
            return "SELL"
        
        # Weaker signals
        if oversold_bb or (oversold_rsi and current_rsi < 25):
            self.confidence = 0.6
            return "BUY"
        
        if overbought_bb or (overbought_rsi and current_rsi > 75):
            self.confidence = 0.6
            return "SELL"
        
        return None

class BreakoutStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Breakout")
        
    def analyze(self, prices: List[float], volumes: List[float] = None) -> Optional[str]:
        if len(prices) < 25:
            return None
        
        current_price = prices[-1]
        
        # Calculate support and resistance levels
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        longer_high = max(prices[-25:])
        longer_low = min(prices[-25:])
        
        # Price range analysis
        recent_range = recent_high - recent_low
        if recent_range == 0:
            return None
        
        # Breakout conditions
        range_threshold = recent_range * 0.1  # 10% of range
        
        # Upward breakout
        if (current_price >= recent_high and 
            current_price > longer_high and
            current_price - recent_high >= range_threshold):
            
            # Confirm with momentum
            if len(prices) >= 3:
                momentum = (prices[-1] - prices[-3]) / prices[-3]
                if momentum > 0.0005:  # 0.05% momentum
                    self.confidence = 0.75
                    return "BUY"
        
        # Downward breakout
        if (current_price <= recent_low and 
            current_price < longer_low and
            recent_low - current_price >= range_threshold):
            
            # Confirm with momentum
            if len(prices) >= 3:
                momentum = (prices[-1] - prices[-3]) / prices[-3]
                if momentum < -0.0005:  # -0.05% momentum
                    self.confidence = 0.75
                    return "SELL"
        
        return None

class MACDStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MACD")
        
    def analyze(self, prices: List[float], volumes: List[float] = None) -> Optional[str]:
        if len(prices) < 35:
            return None
        
        macd_line, signal_line, histogram = TechnicalAnalysis.macd(prices)
        
        if len(histogram) < 2:
            return None
        
        # MACD crossover signals
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        
        # Bullish crossover
        if (current_macd > current_signal and 
            prev_macd <= prev_signal and
            current_macd < 0):  # Below zero line for stronger signal
            self.confidence = 0.7
            return "BUY"
        
        # Bearish crossover
        if (current_macd < current_signal and 
            prev_macd >= prev_signal and
            current_macd > 0):  # Above zero line for stronger signal
            self.confidence = 0.7
            return "SELL"
        
        return None

class AdvancedForexTrader:
    def __init__(self):
        self.api = MT5API()
        self.strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(), 
            BreakoutStrategy(),
            MACDStrategy()
        ]
        self.active_trades = {}
        self.lot_size = 0.01
        self.max_concurrent_trades = 3
        self.trade_count = 0
        self.session_start = None
        
    def test_connection(self) -> bool:
        """Test connection to MT5 API"""
        try:
            status = self.api.get_status()
            is_connected = status.get('connected', False)
            logger.info(f"🔗 MT5 Connection Status: {is_connected}")
            return is_connected
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account_info()
            balance = account.get('balance', 0)
            currency = account.get('currency', 'USD')
            logger.info(f"💰 Account Balance: {balance} {currency}")
            return account
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, risk_percent: float = 1.0) -> float:
        """Calculate position size based on risk management"""
        try:
            account = self.api.get_account_info()
            balance = account.get('balance', 10000)
            
            # Calculate risk amount
            risk_amount = balance * (risk_percent / 100)
            
            # Calculate pip value and risk per lot
            if symbol == "USDJPY":
                pip_value = 0.01
                risk_per_lot = abs(entry_price - stop_loss) * 100000
            else:
                pip_value = 0.0001
                risk_per_lot = abs(entry_price - stop_loss) * 100000
            
            # Calculate lot size
            if risk_per_lot > 0:
                calculated_lots = risk_amount / risk_per_lot
                return min(max(calculated_lots, 0.01), 0.1)  # Min 0.01, max 0.1
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
        
        return self.lot_size
    
    def calculate_stop_loss(self, entry_price: float, is_buy: bool, symbol: str) -> float:
        """Calculate stop loss with dynamic pip calculation"""
        if symbol == "USDJPY":
            pip_value = 0.01
            stop_distance = 15 * pip_value  # 15 pips for JPY pairs
        else:
            pip_value = 0.0001
            stop_distance = 20 * pip_value  # 20 pips for major pairs
        
        if is_buy:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, is_buy: bool, symbol: str) -> float:
        """Calculate take profit with 2:1 risk-reward ratio"""
        if symbol == "USDJPY":
            pip_value = 0.01
            profit_distance = 30 * pip_value  # 30 pips for JPY pairs
        else:
            pip_value = 0.0001
            profit_distance = 40 * pip_value  # 40 pips for major pairs
        
        if is_buy:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def analyze_symbol(self, symbol: str) -> Tuple[Optional[str], float, str]:
        """Analyze symbol with all strategies and return consensus signal"""
        try:
            # Get comprehensive price data
            historical_data = self.api.get_historical_data(symbol, "M5", 100)
            if not historical_data or len(historical_data) < 50:
                return None, 0.0, "Insufficient data"
            
            prices = [candle['close'] for candle in historical_data]
            volumes = [candle.get('tick_volume', 0) for candle in historical_data]
            
            # Collect signals from all strategies
            signals = []
            strategy_details = []
            
            for strategy in self.strategies:
                signal = strategy.analyze(prices, volumes)
                if signal:
                    confidence = getattr(strategy, 'confidence', 0.5)
                    signals.append((signal, confidence, strategy.name))
                    strategy_details.append(f"{strategy.name}: {signal} ({confidence:.2f})")
            
            if not signals:
                return None, 0.0, "No signals generated"
            
            # Calculate consensus
            buy_weight = sum(conf for sig, conf, name in signals if sig == "BUY")
            sell_weight = sum(conf for sig, conf, name in signals if sig == "SELL")
            
            signal_details = " | ".join(strategy_details)
            
            # Require strong consensus
            if buy_weight > sell_weight and buy_weight >= 1.2:  # Minimum confidence threshold
                return "BUY", buy_weight, signal_details
            elif sell_weight > buy_weight and sell_weight >= 1.2:
                return "SELL", sell_weight, signal_details
            
            return None, max(buy_weight, sell_weight), signal_details
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None, 0.0, f"Analysis error: {e}"
    
    def execute_trade(self, symbol: str, signal: str, confidence: float) -> bool:
        """Execute a trade with comprehensive error handling"""
        try:
            # Get current price for order placement
            recent_data = self.api.get_historical_data(symbol, "M1", 3)
            if not recent_data:
                logger.error(f"❌ No current price data for {symbol}")
                return False
            
            current_price = recent_data[-1]['close']
            
            # Calculate trade parameters
            is_buy = signal == "BUY"
            order_type = 0 if is_buy else 1
            
            sl = self.calculate_stop_loss(current_price, is_buy, symbol)
            tp = self.calculate_take_profit(current_price, is_buy, symbol)
            
            # Dynamic lot size based on confidence and risk
            base_lots = self.calculate_position_size(symbol, current_price, sl, 0.5)  # 0.5% risk
            adjusted_lots = min(base_lots * (confidence / 1.0), 0.01)  # Max 0.01 as required
            
            logger.info(f"📊 {signal} Signal for {symbol}")
            logger.info(f"   Price: {current_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            logger.info(f"   Confidence: {confidence:.2f}, Lot Size: {adjusted_lots:.3f}")
            
            # Place order
            result = self.api.place_order(symbol, order_type, adjusted_lots, sl, tp)
            
            if result.get('retcode') == 10009:  # Success
                order_id = result.get('order', result.get('deal', f"manual_{self.trade_count}"))
                
                logger.success(f"✅ Trade Executed: {signal} {symbol} at {current_price:.5f}")
                logger.info(f"   Order ID: {order_id}, Confidence: {confidence:.2f}")
                
                # Store trade information
                self.active_trades[order_id] = {
                    'symbol': symbol,
                    'type': signal,
                    'entry_price': current_price,
                    'sl': sl,
                    'tp': tp,
                    'entry_time': datetime.now(),
                    'confidence': confidence,
                    'lot_size': adjusted_lots
                }
                
                self.trade_count += 1
                return True
            else:
                logger.error(f"❌ Trade execution failed for {symbol}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            return False
    
    def monitor_positions(self):
        """Advanced position monitoring with dynamic exit conditions"""
        try:
            positions = self.api.get_positions()
            current_time = datetime.now()
            
            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol', 'Unknown')
                current_profit = position.get('profit', 0)
                position_type = position.get('type', 0)  # 0=BUY, 1=SELL
                
                # Get trade information if available
                trade_info = self.active_trades.get(ticket, {})
                time_elapsed = 0
                confidence = trade_info.get('confidence', 0.5)
                
                if trade_info:
                    time_elapsed = (current_time - trade_info['entry_time']).total_seconds() / 60
                
                logger.info(f"📊 Position {ticket} ({symbol}): {time_elapsed:.1f}min, "
                          f"Profit: {current_profit:.2f}, Confidence: {confidence:.2f}")
                
                # Dynamic exit conditions based on time, profit, and confidence
                should_close = False
                reason = ""
                
                # Maximum hold time (30 minutes)
                if time_elapsed >= 30:
                    should_close = True
                    reason = "max_time_30min"
                
                # Minimum time + profit conditions
                elif time_elapsed >= 5:
                    if current_profit > 1.0:  # Good profit
                        should_close = True
                        reason = f"good_profit_{current_profit:.2f}"
                    elif time_elapsed >= 15 and current_profit > 0:  # Any profit after 15 min
                        should_close = True
                        reason = f"time_profit_{current_profit:.2f}"
                    elif time_elapsed >= 20 and current_profit > -2:  # Small loss acceptable after 20 min
                        should_close = True
                        reason = f"time_limit_small_loss_{current_profit:.2f}"
                
                # Emergency exit for large losses
                elif current_profit < -5.0:
                    should_close = True
                    reason = f"emergency_stop_loss_{current_profit:.2f}"
                
                if should_close:
                    try:
                        close_result = self.api.close_position(ticket)
                        logger.success(f"✅ Position {ticket} closed: {reason}")
                        logger.info(f"   Final profit: {current_profit:.2f}, Duration: {time_elapsed:.1f}min")
                        
                        # Remove from active trades
                        if ticket in self.active_trades:
                            del self.active_trades[ticket]
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to close position {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")
    
    def run_trading_session(self):
        """Main automated trading session"""
        logger.info("🚀 === Advanced Automated Forex Trading System ===")
        self.session_start = datetime.now()
        
        # System checks
        if not self.test_connection():
            logger.error("❌ Cannot connect to MT5 API. Exiting.")
            return
        
        account = self.get_account_info()
        if not account:
            logger.error("❌ Cannot get account information. Exiting.")
            return
        
        # Get trading symbols
        try:
            all_symbols = self.api.get_tradable_symbols()
            if not all_symbols:
                logger.error("❌ No tradable symbols found. Exiting.")
                return
            
            # Prioritize major forex pairs
            preferred_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
            trading_symbols = [s for s in preferred_symbols if s in all_symbols]
            
            if not trading_symbols:
                trading_symbols = all_symbols[:3]
                
            logger.info(f"🎯 Trading symbols: {trading_symbols}")
            
        except Exception as e:
            logger.error(f"❌ Error getting tradable symbols: {e}")
            return
        
        # Trading session parameters
        max_session_time = 60  # 60 minutes
        signal_check_interval = 180  # 3 minutes between signal checks
        last_signal_check = datetime.now() - timedelta(seconds=signal_check_interval)
        
        logger.info(f"⏰ Session duration: {max_session_time} minutes")
        logger.info(f"📊 Max concurrent trades: {self.max_concurrent_trades}")
        logger.info(f"💰 Base lot size: {self.lot_size}")
        
        # Main trading loop
        while (datetime.now() - self.session_start).total_seconds() < max_session_time * 60:
            try:
                current_time = datetime.now()
                session_elapsed = (current_time - self.session_start).total_seconds() / 60
                
                # Always monitor existing positions
                self.monitor_positions()
                
                # Check for new trading signals at intervals
                if (current_time - last_signal_check).total_seconds() >= signal_check_interval:
                    last_signal_check = current_time
                    
                    active_count = len([t for t in self.active_trades.values()])
                    logger.info(f"⏱️  Session: {session_elapsed:.1f}min | Active: {active_count}/{self.max_concurrent_trades} | Total: {self.trade_count}")
                    
                    if active_count >= self.max_concurrent_trades:
                        logger.info("🔄 Max trades active. Monitoring existing positions...")
                    else:
                        # Analyze each symbol for trading opportunities
                        for symbol in trading_symbols:
                            if active_count >= self.max_concurrent_trades:
                                break
                            
                            # Skip if already trading this symbol
                            if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                                continue
                            
                            # Analyze symbol
                            signal, confidence, details = self.analyze_symbol(symbol)
                            
                            if signal and confidence >= 1.2:  # Strong signal threshold
                                logger.info(f"🎯 {signal} signal for {symbol} (confidence: {confidence:.2f})")
                                logger.info(f"   Strategy details: {details}")
                                
                                if self.execute_trade(symbol, signal, confidence):
                                    active_count += 1
                                    time.sleep(3)  # Brief pause between trades
                            else:
                                logger.debug(f"📉 {symbol}: {signal or 'No signal'} (conf: {confidence:.2f}) - {details}")
                
                # Wait before next iteration
                time.sleep(45)  # Check every 45 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in main trading loop: {e}")
                time.sleep(60)
        
        # Session completion
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        logger.info(f"🏁 Trading session completed after {session_duration:.1f} minutes")
        
        # Final cleanup
        try:
            final_positions = self.api.get_positions()
            if final_positions:
                logger.info(f"🧹 Closing {len(final_positions)} remaining positions...")
                for position in final_positions:
                    ticket = position.get('ticket')
                    profit = position.get('profit', 0)
                    try:
                        self.api.close_position(ticket)
                        logger.success(f"✅ Final close - Position {ticket}, Profit: {profit:.2f}")
                    except Exception as e:
                        logger.error(f"❌ Failed to close position {ticket}: {e}")
            
            # Session summary
            logger.info("📊 === Trading Session Summary ===")
            logger.info(f"   Total trades executed: {self.trade_count}")
            logger.info(f"   Session duration: {session_duration:.1f} minutes")
            logger.info(f"   Strategies used: {len(self.strategies)}")
            
        except Exception as e:
            logger.error(f"❌ Error in final cleanup: {e}")

if __name__ == "__main__":
    trader = AdvancedForexTrader()
    trader.run_trading_session()