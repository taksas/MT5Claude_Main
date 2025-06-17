#!/usr/bin/env python3
"""
Direct Forex Trading System - bypasses backtesting and trades directly
Uses simple price action signals for immediate execution
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import urllib.request
import urllib.parse
import urllib.error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        try:
            with urllib.request.urlopen(req) as response:
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
            "action": "TRADE_ACTION_DEAL",
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "comment": "Direct automated trading"
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

class DirectTradingSignals:
    @staticmethod
    def get_trend_signal(prices: List[float]) -> str:
        """Get simple trend signal based on recent price action"""
        if len(prices) < 10:
            return "HOLD"
        
        # Look at last 5 vs previous 5 prices
        recent_avg = sum(prices[-5:]) / 5
        previous_avg = sum(prices[-10:-5]) / 5
        
        current_price = prices[-1]
        prev_price = prices[-2]
        
        # Strong uptrend signal
        if (recent_avg > previous_avg * 1.0001 and 
            current_price > prev_price and 
            current_price > recent_avg):
            return "BUY"
        
        # Strong downtrend signal  
        elif (recent_avg < previous_avg * 0.9999 and 
              current_price < prev_price and 
              current_price < recent_avg):
            return "SELL"
        
        return "HOLD"
    
    @staticmethod
    def get_breakout_signal(prices: List[float]) -> str:
        """Get breakout signal based on price movement"""
        if len(prices) < 20:
            return "HOLD"
        
        current_price = prices[-1]
        
        # Calculate recent high and low
        recent_high = max(prices[-10:])
        recent_low = min(prices[-10:])
        
        # Calculate longer-term range
        long_high = max(prices[-20:])
        long_low = min(prices[-20:])
        
        # Breakout above recent resistance
        if current_price >= recent_high and current_price > long_high * 1.0001:
            return "BUY"
        
        # Breakdown below recent support
        elif current_price <= recent_low and current_price < long_low * 0.9999:
            return "SELL"
        
        return "HOLD"

class DirectForexTrader:
    def __init__(self):
        self.api = MT5API()
        self.signals = DirectTradingSignals()
        self.active_trades = {}
        self.lot_size = 0.01
        self.max_trades = 2
        self.trade_count = 0
        
    def test_connection(self) -> bool:
        """Test connection to MT5 API"""
        try:
            status = self.api.get_status()
            logger.info(f"MT5 Connection Status: {status.get('connected', False)}")
            return status.get('connected', False)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account_info()
            logger.info(f"Account Balance: {account.get('balance', 0)} {account.get('currency', 'USD')}")
            return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def calculate_stop_loss(self, entry_price: float, is_buy: bool, symbol: str = "EURUSD") -> float:
        """Calculate stop loss with appropriate pip values"""
        if symbol in ["USDJPY"]:
            # For JPY pairs, use smaller pip value
            pip_value = 0.01
        else:
            # For major pairs like EURUSD
            pip_value = 0.0001
            
        stop_distance = 20 * pip_value  # 20 pips stop loss
        
        if is_buy:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, is_buy: bool, symbol: str = "EURUSD") -> float:
        """Calculate take profit with appropriate pip values"""
        if symbol in ["USDJPY"]:
            # For JPY pairs, use smaller pip value
            pip_value = 0.01
        else:
            # For major pairs like EURUSD
            pip_value = 0.0001
            
        profit_distance = 40 * pip_value  # 40 pips take profit (2:1 RR)
        
        if is_buy:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def execute_trade(self, symbol: str, signal_type: str) -> bool:
        """Execute a trade directly"""
        try:
            # Get current price data
            recent_data = self.api.get_historical_data(symbol, "M1", 3)
            if not recent_data:
                logger.error(f"No price data for {symbol}")
                return False
                
            current_price = recent_data[-1]['close']
            
            if signal_type == 'BUY':
                order_type = 0
                sl = self.calculate_stop_loss(current_price, True, symbol)
                tp = self.calculate_take_profit(current_price, True, symbol)
            else:  # SELL
                order_type = 1
                sl = self.calculate_stop_loss(current_price, False, symbol)
                tp = self.calculate_take_profit(current_price, False, symbol)
            
            logger.info(f"Attempting {signal_type} order for {symbol} at {current_price:.5f}")
            logger.info(f"Stop Loss: {sl:.5f}, Take Profit: {tp:.5f}")
            
            # Place order
            result = self.api.place_order(symbol, order_type, self.lot_size, sl, tp)
            
            if result.get('retcode') == 10009:  # Success
                order_id = result.get('order', result.get('deal'))
                logger.success(f"✓ Trade executed: {signal_type} {symbol} at {current_price:.5f}")
                logger.info(f"Order ID: {order_id}, SL: {sl:.5f}, TP: {tp:.5f}")
                
                # Store trade info
                if order_id:
                    self.active_trades[order_id] = {
                        'symbol': symbol,
                        'type': signal_type,
                        'entry_price': current_price,
                        'sl': sl,
                        'tp': tp,
                        'entry_time': datetime.now()
                    }
                    self.trade_count += 1
                return True
            else:
                logger.error(f"✗ Trade execution failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = self.api.get_positions()
            current_time = datetime.now()
            
            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol', 'Unknown')
                current_profit = position.get('profit', 0)
                
                time_elapsed = 0
                if ticket in self.active_trades:
                    time_elapsed = (current_time - self.active_trades[ticket]['entry_time']).total_seconds() / 60
                
                logger.info(f"Position {ticket} ({symbol}): {time_elapsed:.1f}min, Profit: {current_profit:.2f}")
                
                # Close conditions
                should_close = False
                reason = ""
                
                if time_elapsed >= 30:  # Max 30 minutes
                    should_close = True
                    reason = f"max_time_30min"
                elif time_elapsed >= 5 and current_profit > 0:  # Min 5 min + profit
                    should_close = True
                    reason = f"profitable_exit_after_{time_elapsed:.1f}min"
                elif time_elapsed >= 15 and current_profit > -5:  # 15 min + small loss acceptable
                    should_close = True
                    reason = f"time_limit_minor_loss"
                
                if should_close:
                    try:
                        self.api.close_position(ticket)
                        logger.info(f"✓ Position {ticket} closed: {reason}, Final profit: {current_profit:.2f}")
                        
                        if ticket in self.active_trades:
                            del self.active_trades[ticket]
                    except Exception as e:
                        logger.error(f"Failed to close position {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def analyze_symbol(self, symbol: str) -> Optional[str]:
        """Analyze symbol and return trading signal"""
        try:
            # Get recent price data
            historical_data = self.api.get_historical_data(symbol, "M5", 30)  # 30 x 5-min candles
            if not historical_data or len(historical_data) < 20:
                return None
            
            prices = [candle['close'] for candle in historical_data]
            
            # Get multiple signal types
            trend_signal = self.signals.get_trend_signal(prices)
            breakout_signal = self.signals.get_breakout_signal(prices)
            
            logger.info(f"{symbol} signals - Trend: {trend_signal}, Breakout: {breakout_signal}")
            
            # Combine signals for stronger confirmation
            if trend_signal == "BUY" and breakout_signal in ["BUY", "HOLD"]:
                return "BUY"
            elif trend_signal == "SELL" and breakout_signal in ["SELL", "HOLD"]:
                return "SELL"
            elif breakout_signal in ["BUY", "SELL"] and trend_signal == "HOLD":
                return breakout_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def run_direct_trading(self):
        """Main direct trading loop - no backtesting required"""
        logger.info("=== Starting Direct Automated Forex Trading ===")
        
        # Test connection
        if not self.test_connection():
            logger.error("❌ Cannot connect to MT5 API. Exiting.")
            return
        
        # Get account info
        account = self.get_account_info()
        if not account:
            logger.error("❌ Cannot get account information. Exiting.")
            return
        
        # Get tradable symbols
        try:
            all_symbols = self.api.get_tradable_symbols()
            if not all_symbols:
                logger.error("❌ No tradable symbols found. Exiting.")
                return
            
            # Prioritize major forex pairs
            preferred_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
            trading_symbols = [s for s in preferred_symbols if s in all_symbols]
            
            if not trading_symbols:
                trading_symbols = all_symbols[:3]  # Use first 3 available
                
            logger.info(f"🎯 Selected trading symbols: {trading_symbols}")
            
        except Exception as e:
            logger.error(f"❌ Error getting tradable symbols: {e}")
            return
        
        # Main trading session
        session_start = datetime.now()
        max_session_time = 45  # 45 minutes max session
        last_signal_check = datetime.now() - timedelta(minutes=3)
        
        logger.info(f"🚀 Trading session started - Max duration: {max_session_time} minutes")
        logger.info(f"📊 Max concurrent trades: {self.max_trades}, Lot size: {self.lot_size}")
        
        while (datetime.now() - session_start).total_seconds() < max_session_time * 60:
            try:
                current_time = datetime.now()
                session_elapsed = (current_time - session_start).total_seconds() / 60
                
                # Monitor existing positions every cycle
                self.monitor_positions()
                
                # Look for new signals every 2 minutes
                if (current_time - last_signal_check).total_seconds() >= 120:
                    last_signal_check = current_time
                    
                    active_count = len([t for t in self.active_trades.values()])
                    logger.info(f"⏱️  Session: {session_elapsed:.1f}min, Active trades: {active_count}/{self.max_trades}, Total trades: {self.trade_count}")
                    
                    if active_count >= self.max_trades:
                        logger.info(f"🔄 Max trades ({self.max_trades}) active. Monitoring only...")
                    else:
                        # Look for trading opportunities
                        for symbol in trading_symbols:
                            if active_count >= self.max_trades:
                                break
                            
                            # Skip if already trading this symbol
                            if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                                continue
                            
                            # Analyze for signals
                            signal = self.analyze_symbol(symbol)
                            if signal:
                                logger.info(f"📈 {signal} signal detected for {symbol}")
                                if self.execute_trade(symbol, signal):
                                    active_count += 1
                                    time.sleep(2)  # Brief pause between trades
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(60)
        
        logger.info("🏁 Trading session completed")
        
        # Final position cleanup
        try:
            final_positions = self.api.get_positions()
            if final_positions:
                logger.info(f"🧹 Closing {len(final_positions)} remaining positions...")
                for position in final_positions:
                    ticket = position.get('ticket')
                    profit = position.get('profit', 0)
                    self.api.close_position(ticket)
                    logger.info(f"✓ Final close - Position {ticket}, Profit: {profit:.2f}")
            
            logger.info(f"📊 Session Summary: Total trades executed: {self.trade_count}")
            
        except Exception as e:
            logger.error(f"❌ Error in final cleanup: {e}")

if __name__ == "__main__":
    # Custom logger level for success messages
    logging.addLevelName(25, "SUCCESS")
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)
    logging.Logger.success = success
    
    trader = DirectForexTrader()
    trader.run_direct_trading()