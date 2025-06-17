#!/usr/bin/env python3
"""
Final Automated Forex Trading System - Optimized for immediate execution
- Multiple strategy implementation with real-time signal generation
- Comprehensive risk management with 0.01 lot size maximum
- Short-term trades with 5-30 minute hold times
- Auto stop-loss and take-profit implementation
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
import urllib.error
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add SUCCESS level
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
            req = urllib.request.Request(url)
        else:
            json_data = json.dumps(data).encode('utf-8') if data else None
            req = urllib.request.Request(url, data=json_data)
            req.add_header('Content-Type', 'application/json')
            if method == "DELETE":
                req.get_method = lambda: 'DELETE'
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_status(self) -> Dict:
        return self._make_request("GET", "/status/mt5")
    
    def get_account_info(self) -> Dict:
        return self._make_request("GET", "/account/")
        
    def get_tradable_symbols(self) -> List[str]:
        return self._make_request("GET", "/market/symbols/tradable")
        
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        data = {"symbol": symbol, "timeframe": timeframe, "count": count}
        return self._make_request("POST", "/market/history", data)
        
    def place_order(self, symbol: str, order_type: int, volume: float, sl: float = None, tp: float = None) -> Dict:
        order_data = {
            "action": 1,  # TRADE_ACTION_DEAL  
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "comment": "Automated forex trading"
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

class TradingSignalGenerator:
    """Advanced signal generation with multiple strategy approaches"""
    
    @staticmethod
    def generate_trend_signal(prices: List[float]) -> Tuple[str, float]:
        """Generate trend-following signal"""
        if len(prices) < 10:
            return "HOLD", 0.0
        
        # Price momentum analysis
        current = prices[-1]
        recent_avg = sum(prices[-5:]) / 5
        older_avg = sum(prices[-10:-5]) / 5
        
        # Trend strength
        trend_strength = abs(recent_avg - older_avg) / older_avg
        
        if recent_avg > older_avg * 1.0002 and current > recent_avg:
            return "BUY", min(trend_strength * 100, 0.9)
        elif recent_avg < older_avg * 0.9998 and current < recent_avg:
            return "SELL", min(trend_strength * 100, 0.9)
        
        return "HOLD", 0.0
    
    @staticmethod
    def generate_breakout_signal(prices: List[float]) -> Tuple[str, float]:
        """Generate breakout signal"""
        if len(prices) < 15:
            return "HOLD", 0.0
        
        current = prices[-1]
        recent_high = max(prices[-10:])
        recent_low = min(prices[-10:])
        range_size = recent_high - recent_low
        
        if range_size == 0:
            return "HOLD", 0.0
        
        # Breakout detection
        if current >= recent_high and current > max(prices[-15:-10]):
            strength = (current - recent_high) / range_size
            return "BUY", min(strength * 2, 0.8)
        elif current <= recent_low and current < min(prices[-15:-10]):
            strength = (recent_low - current) / range_size
            return "SELL", min(strength * 2, 0.8)
        
        return "HOLD", 0.0
    
    @staticmethod
    def generate_mean_reversion_signal(prices: List[float]) -> Tuple[str, float]:
        """Generate mean reversion signal"""
        if len(prices) < 20:
            return "HOLD", 0.0
        
        current = prices[-1]
        mean = sum(prices[-20:]) / 20
        
        # Calculate standard deviation
        variance = sum((p - mean) ** 2 for p in prices[-20:]) / 20
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return "HOLD", 0.0
        
        # Z-score calculation
        z_score = (current - mean) / std_dev
        
        # Mean reversion signals
        if z_score < -1.5:  # Oversold
            return "BUY", min(abs(z_score) / 2, 0.9)
        elif z_score > 1.5:  # Overbought
            return "SELL", min(abs(z_score) / 2, 0.9)
        
        return "HOLD", 0.0

class AutomatedForexTrader:
    def __init__(self):
        self.api = MT5API()
        self.signal_generator = TradingSignalGenerator()
        self.active_trades = {}
        self.lot_size = 0.01
        self.max_trades = 2
        self.trade_count = 0
        self.total_profit = 0.0
        
    def test_connection(self) -> bool:
        """Test MT5 API connection"""
        try:
            status = self.api.get_status()
            connected = status.get('connected', False)
            logger.info(f"🔗 MT5 Status: {'Connected' if connected else 'Disconnected'}")
            return connected
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get trading account information"""
        try:
            account = self.api.get_account_info()
            balance = account.get('balance', 0)
            logger.info(f"💰 Account Balance: {balance} {account.get('currency', 'USD')}")
            return account
        except Exception as e:
            logger.error(f"❌ Account info error: {e}")
            return {}
    
    def calculate_stop_loss(self, entry_price: float, is_buy: bool, symbol: str) -> float:
        """Calculate stop loss based on symbol type"""
        if symbol == "USDJPY":
            pip_value = 0.01
            stop_pips = 20
        else:
            pip_value = 0.0001
            stop_pips = 25
        
        stop_distance = stop_pips * pip_value
        
        if is_buy:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, is_buy: bool, symbol: str) -> float:
        """Calculate take profit with 2:1 risk-reward ratio"""
        if symbol == "USDJPY":
            pip_value = 0.01
            profit_pips = 40
        else:
            pip_value = 0.0001
            profit_pips = 50
        
        profit_distance = profit_pips * pip_value
        
        if is_buy:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def analyze_trading_opportunity(self, symbol: str) -> Tuple[Optional[str], float]:
        """Comprehensive market analysis for trading opportunities"""
        try:
            # Get market data - try multiple timeframes if one fails
            historical_data = None
            for timeframe in ["M5", "M15", "M1"]:
                try:
                    historical_data = self.api.get_historical_data(symbol, timeframe, 50)
                    if historical_data and len(historical_data) >= 20:
                        break
                except:
                    continue
            
            if not historical_data or len(historical_data) < 15:
                logger.warning(f"📉 {symbol}: Insufficient market data")
                return None, 0.0
            
            prices = [candle['close'] for candle in historical_data]
            
            # Generate signals from multiple strategies
            trend_signal, trend_confidence = self.signal_generator.generate_trend_signal(prices)
            breakout_signal, breakout_confidence = self.signal_generator.generate_breakout_signal(prices)
            reversion_signal, reversion_confidence = self.signal_generator.generate_mean_reversion_signal(prices)
            
            signals = [
                (trend_signal, trend_confidence, "Trend"),
                (breakout_signal, breakout_confidence, "Breakout"),  
                (reversion_signal, reversion_confidence, "Mean Reversion")
            ]
            
            # Calculate consensus
            buy_score = sum(conf for sig, conf, name in signals if sig == "BUY")
            sell_score = sum(conf for sig, conf, name in signals if sig == "SELL")
            
            # Log signal details
            signal_details = " | ".join([f"{name}: {sig}({conf:.2f})" for sig, conf, name in signals if sig != "HOLD"])
            if signal_details:
                logger.info(f"📊 {symbol} signals: {signal_details}")
            
            # Decision making
            min_confidence = 0.6  # Minimum confidence threshold
            
            if buy_score > sell_score and buy_score >= min_confidence:
                return "BUY", buy_score
            elif sell_score > buy_score and sell_score >= min_confidence:
                return "SELL", sell_score
            
            return None, max(buy_score, sell_score)
            
        except Exception as e:
            logger.error(f"❌ Analysis error for {symbol}: {e}")
            return None, 0.0
    
    def execute_trade(self, symbol: str, signal: str, confidence: float) -> bool:
        """Execute trading order with comprehensive error handling"""
        try:
            # Get current market price
            recent_data = self.api.get_historical_data(symbol, "M1", 2)
            if not recent_data:
                # Generate mock current price for demonstration
                base_price = 1.1000 if symbol == "EURUSD" else 150.00
                current_price = base_price + random.uniform(-0.001, 0.001)
                logger.warning(f"⚠️  Using simulated price for {symbol}: {current_price:.5f}")
            else:
                current_price = recent_data[-1]['close']
            
            # Prepare trade parameters
            is_buy = signal == "BUY"
            order_type = 0 if is_buy else 1
            
            sl = self.calculate_stop_loss(current_price, is_buy, symbol)
            tp = self.calculate_take_profit(current_price, is_buy, symbol)
            
            logger.info(f"🎯 Executing {signal} order for {symbol}")
            logger.info(f"   Price: {current_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            logger.info(f"   Confidence: {confidence:.2f}, Lot Size: {self.lot_size}")
            
            # Place the order
            result = self.api.place_order(symbol, order_type, self.lot_size, sl, tp)
            
            if result.get('retcode') == 10009:  # Trade successful
                order_id = result.get('order', result.get('deal', f"trade_{self.trade_count}"))
                
                logger.success(f"✅ TRADE EXECUTED: {signal} {symbol} at {current_price:.5f}")
                logger.info(f"   Order ID: {order_id} | Risk-Reward: 1:2")
                
                # Store trade information
                self.active_trades[order_id] = {
                    'symbol': symbol,
                    'type': signal,
                    'entry_price': current_price,
                    'sl': sl,
                    'tp': tp,
                    'entry_time': datetime.now(),
                    'confidence': confidence,
                    'lot_size': self.lot_size
                }
                
                self.trade_count += 1
                return True
            else:
                logger.error(f"❌ Trade failed for {symbol}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Trade execution error for {symbol}: {e}")
            return False
    
    def monitor_active_positions(self):
        """Monitor and manage all active trading positions"""
        try:
            positions = self.api.get_positions()
            current_time = datetime.now()
            
            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol', 'Unknown')
                current_profit = position.get('profit', 0)
                
                # Get trade metadata
                trade_info = self.active_trades.get(ticket, {})
                time_elapsed = 0
                
                if trade_info:
                    time_elapsed = (current_time - trade_info['entry_time']).total_seconds() / 60
                
                logger.info(f"📊 Position {ticket} ({symbol}): {time_elapsed:.1f}min, P&L: {current_profit:.2f}")
                
                # Position management rules
                should_close = False
                close_reason = ""
                
                # Maximum hold time: 30 minutes
                if time_elapsed >= 30:
                    should_close = True
                    close_reason = "Maximum time limit (30min)"
                
                # Profitable exit after minimum time
                elif time_elapsed >= 5 and current_profit > 0.5:
                    should_close = True
                    close_reason = f"Profitable exit (+{current_profit:.2f})"
                
                # Time-based exit with small profit/loss
                elif time_elapsed >= 15 and current_profit > -1.0:
                    should_close = True
                    close_reason = f"Time limit with acceptable P&L ({current_profit:.2f})"
                
                # Emergency stop for large losses
                elif current_profit < -3.0:
                    should_close = True
                    close_reason = f"Emergency stop loss ({current_profit:.2f})"
                
                if should_close:
                    try:
                        self.api.close_position(ticket)
                        self.total_profit += current_profit
                        
                        logger.success(f"✅ Position {ticket} CLOSED: {close_reason}")
                        logger.info(f"   Duration: {time_elapsed:.1f}min | Final P&L: {current_profit:.2f}")
                        
                        # Remove from active trades
                        if ticket in self.active_trades:
                            del self.active_trades[ticket]
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to close position {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Position monitoring error: {e}")
    
    def run_automated_trading(self):
        """Main automated trading execution loop"""
        logger.info("🚀 === STARTING AUTOMATED FOREX TRADING SESSION ===")
        session_start = datetime.now()
        
        # Pre-flight checks
        if not self.test_connection():
            logger.error("❌ MT5 API connection failed. Cannot proceed.")
            return
        
        account = self.get_account_info()
        if not account:
            logger.error("❌ Account information unavailable. Cannot proceed.")
            return
        
        # Get available trading symbols
        try:
            symbols = self.api.get_tradable_symbols()
            if not symbols:
                logger.error("❌ No tradable symbols available.")
                return
            
            # Filter for major forex pairs
            major_pairs = [s for s in symbols if s in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']]
            trading_symbols = major_pairs if major_pairs else symbols[:2]
            
            logger.info(f"🎯 Selected trading instruments: {trading_symbols}")
            
        except Exception as e:
            logger.error(f"❌ Symbol retrieval error: {e}")
            return
        
        # Trading session configuration
        session_duration = 600  # 10 hours (600 minutes) autonomous operation
        signal_interval = 180  # Check signals every 3 minutes
        last_signal_time = datetime.now() - timedelta(seconds=signal_interval)
        
        logger.info(f"⏰ Session Parameters:")
        logger.info(f"   Duration: {session_duration/60:.1f} hours ({session_duration} minutes)")
        logger.info(f"   Signal check interval: {signal_interval/60:.1f} minutes") 
        logger.info(f"   Maximum concurrent trades: {self.max_trades}")
        logger.info(f"   Lot size per trade: {self.lot_size}")
        logger.info(f"🤖 AUTONOMOUS MODE: Running independently for 10 hours")
        
        # Main trading loop
        while (datetime.now() - session_start).total_seconds() < session_duration * 60:
            try:
                current_time = datetime.now()
                elapsed_minutes = (current_time - session_start).total_seconds() / 60
                
                # Always monitor existing positions
                self.monitor_active_positions()
                
                # Check for new trading opportunities at intervals
                if (current_time - last_signal_time).total_seconds() >= signal_interval:
                    last_signal_time = current_time
                    
                    active_count = len(self.active_trades)
                    
                    hours_elapsed = elapsed_minutes / 60
                    logger.info(f"⏱️  Session: {hours_elapsed:.1f}h ({elapsed_minutes:.0f}min) | Active: {active_count}/{self.max_trades} | Total: {self.trade_count} | P&L: {self.total_profit:.2f}")
                    
                    if active_count >= self.max_trades:
                        logger.info("🔄 Maximum trades active. Monitoring positions...")
                    else:
                        # Analyze each symbol for opportunities
                        for symbol in trading_symbols:
                            if active_count >= self.max_trades:
                                break
                            
                            # Skip if already trading this symbol
                            symbol_active = any(t['symbol'] == symbol for t in self.active_trades.values())
                            if symbol_active:
                                continue
                            
                            # Market analysis
                            signal, confidence = self.analyze_trading_opportunity(symbol)
                            
                            if signal and confidence >= 0.6:
                                logger.info(f"🎯 {signal} opportunity detected for {symbol} (confidence: {confidence:.2f})")
                                
                                if self.execute_trade(symbol, signal, confidence):
                                    active_count += 1
                                    time.sleep(2)  # Brief pause between trades
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                time.sleep(30)
        
        # Session completion and cleanup
        session_duration_actual = (datetime.now() - session_start).total_seconds() / 60
        logger.info(f"🏁 Trading session completed after {session_duration_actual:.1f} minutes")
        
        # Close any remaining positions
        try:
            remaining_positions = self.api.get_positions()
            if remaining_positions:
                logger.info(f"🧹 Closing {len(remaining_positions)} remaining positions...")
                for position in remaining_positions:
                    ticket = position.get('ticket')
                    profit = position.get('profit', 0)
                    try:
                        self.api.close_position(ticket)
                        self.total_profit += profit
                        logger.success(f"✅ Final close - Position {ticket}, P&L: {profit:.2f}")
                    except Exception as e:
                        logger.error(f"❌ Failed to close position {ticket}: {e}")
        except Exception as e:
            logger.error(f"❌ Final cleanup error: {e}")
        
        # Session summary
        logger.info("📊 === TRADING SESSION SUMMARY ===")
        logger.info(f"   Total trades executed: {self.trade_count}")
        logger.info(f"   Session duration: {session_duration_actual:.1f} minutes")
        logger.info(f"   Total profit/loss: {self.total_profit:.2f}")
        logger.info(f"   Trading symbols: {', '.join(trading_symbols)}")
        logger.info("✅ Automated trading session completed successfully!")

if __name__ == "__main__":
    trader = AutomatedForexTrader()
    trader.run_automated_trading()