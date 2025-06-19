#!/usr/bin/env python3
"""
Trading Engine - Core trading logic separated from visualization
"""

import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from urllib.parse import quote
import queue

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingEngine')

# Configuration
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "SYMBOLS": ["EURUSD#", "USDJPY#", "GBPUSD#"],
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.70,
    "MIN_STRATEGIES": 3,
    "MAX_SPREAD_PIPS": 2.5,
    "RISK_PER_TRADE": 0.01,
    "MAX_DAILY_LOSS": 0.03,
    "MAX_CONCURRENT": 2,
    "MIN_RR_RATIO": 1.5,
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY"
}

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Signal:
    type: SignalType
    confidence: float
    entry: float
    sl: float
    tp: float
    reason: str
    strategies: Dict[str, float] = None

@dataclass 
class Trade:
    ticket: int
    symbol: str
    type: str
    entry_price: float
    sl: float
    tp: float
    volume: float
    entry_time: datetime

class TradingEngine:
    def __init__(self, signal_queue: Optional[queue.Queue] = None):
        self.config = CONFIG
        self.api_base = CONFIG["API_BASE"]
        self.timezone = pytz.timezone(CONFIG["TIMEZONE"])
        
        # Communication queue for visualizer
        self.signal_queue = signal_queue
        
        # State
        self.active_trades = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_time = 0
        self.balance = None
        self.running = False
        
        # Market cache
        self.spread_cache = {}
        self.data_cache = {}
        
        # JPY account settings
        self.account_currency = CONFIG["ACCOUNT_CURRENCY"]
        
        # Strategy tracking
        self.last_signals = {}
        
    def start(self):
        """Initialize and start trading"""
        if not self._check_connection():
            logger.error("Cannot connect to API")
            return False
            
        self.balance = self._get_balance()
        if not self.balance:
            logger.error("Cannot get account balance")
            return False
            
        logger.info(f"Starting Trading Engine")
        logger.info(f"Balance: ¥{self.balance:,.0f}")
        logger.info(f"Symbols: {CONFIG['SYMBOLS']}")
        
        self.running = True
        self._run_loop()
        
    def _check_connection(self) -> bool:
        """Verify API connection"""
        try:
            resp = requests.get(f"{self.api_base}/status/mt5", timeout=5)
            return resp.status_code == 200
        except:
            return False
            
    def _get_balance(self) -> Optional[float]:
        """Get account balance"""
        try:
            resp = requests.get(f"{self.api_base}/account/", timeout=5)
            if resp.status_code == 200:
                return resp.json()['balance']
        except:
            pass
        return None
        
    def _calculate_position_size(self, symbol: str, sl_distance: float) -> float:
        """Calculate position size based on JPY account and risk"""
        # For JPY account, we need to consider pip values
        # EURUSD: 1 pip = ~150 JPY per 0.01 lot (depends on USDJPY rate)
        # USDJPY: 1 pip = 10 JPY per 0.01 lot
        # GBPUSD: 1 pip = ~180 JPY per 0.01 lot (depends on USDJPY rate)
        
        risk_amount = self.balance * CONFIG["RISK_PER_TRADE"]
        
        # Fixed position size for now - can be enhanced with dynamic calculation
        # This keeps risk consistent across all pairs
        return 0.01
        
    def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached market data"""
        # Check cache (30 second expiry)
        cache_key = f"{symbol}_data"
        if cache_key in self.data_cache:
            data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < 30:
                return data
                
        try:
            resp = requests.post(
                f"{self.api_base}/market/history",
                json={"symbol": symbol, "timeframe": CONFIG["TIMEFRAME"], "count": 60},
                timeout=5
            )
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json())
                self.data_cache[cache_key] = (df, time.time())
                return df
        except:
            pass
        return None
        
    def _check_spread(self, symbol: str) -> Tuple[bool, float]:
        """Check if spread is acceptable"""
        # Check cache
        if symbol in self.spread_cache:
            spread, timestamp = self.spread_cache[symbol]
            if time.time() - timestamp < 10:
                return spread <= CONFIG["MAX_SPREAD_PIPS"], spread
                
        try:
            resp = requests.get(f"{self.api_base}/market/symbols/{quote(symbol)}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                points = data.get('spread', 999)
                
                # Convert to pips
                spread = points * 0.01 if "JPY" in symbol else points * 0.1
                self.spread_cache[symbol] = (spread, time.time())
                
                return spread <= CONFIG["MAX_SPREAD_PIPS"], spread
        except:
            pass
        return False, 999
        
    def _analyze(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Analyze market and generate signal with strategy breakdown"""
        if len(df) < 30:
            return None
            
        # Price data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        current = close[-1]
        
        # Indicators
        sma_fast = pd.Series(close).rolling(10).mean().iloc[-1]
        sma_slow = pd.Series(close).rolling(30).mean().iloc[-1]
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - 100/(1 + rs)).iloc[-1]
        
        # Bollinger Bands
        bb_mean = pd.Series(close).rolling(20).mean().iloc[-1]
        bb_std = pd.Series(close).rolling(20).std().iloc[-1]
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        
        # ATR for volatility
        tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        atr = np.mean(tr[-14:])
        
        # Strategy signals with individual scoring
        strategies = {
            "Trend": 0,
            "RSI": 0,
            "Bollinger": 0,
            "Momentum": 0,
            "Volume": 0
        }
        
        buy_signals = 0
        sell_signals = 0
        reasons = []
        
        # 1. Trend
        if sma_fast > sma_slow and current > sma_fast:
            buy_signals += 1
            strategies["Trend"] = 1.0
            reasons.append("Uptrend")
        elif sma_fast < sma_slow and current < sma_fast:
            sell_signals += 1
            strategies["Trend"] = -1.0
            reasons.append("Downtrend")
            
        # 2. RSI
        if 30 < rsi < 40:
            buy_signals += 1
            strategies["RSI"] = (40 - rsi) / 10  # Stronger signal closer to 30
            reasons.append(f"RSI {rsi:.0f}")
        elif 60 < rsi < 70:
            sell_signals += 1
            strategies["RSI"] = -(rsi - 60) / 10  # Stronger signal closer to 70
            reasons.append(f"RSI {rsi:.0f}")
            
        # 3. Bollinger Bands
        if current <= bb_lower:
            buy_signals += 1
            strategies["Bollinger"] = min((bb_lower - current) / bb_std, 1.0)
            reasons.append("BB Low")
        elif current >= bb_upper:
            sell_signals += 1
            strategies["Bollinger"] = -min((current - bb_upper) / bb_std, 1.0)
            reasons.append("BB High")
            
        # 4. Momentum
        momentum = (current - close[-10]) / close[-10]
        if momentum > 0.001:
            buy_signals += 1
            strategies["Momentum"] = min(momentum * 100, 1.0)
            reasons.append("Momentum+")
        elif momentum < -0.001:
            sell_signals += 1
            strategies["Momentum"] = max(momentum * 100, -1.0)
            reasons.append("Momentum-")
            
        # 5. Volume
        vol_ratio = volume[-1] / np.mean(volume)
        if vol_ratio > 1.3:
            if close[-1] > close[-2]:
                buy_signals += 1
                strategies["Volume"] = min((vol_ratio - 1) / 2, 1.0)
                reasons.append("Volume+")
            else:
                sell_signals += 1
                strategies["Volume"] = -min((vol_ratio - 1) / 2, 1.0)
                reasons.append("Volume-")
                
        # Normalize strategy scores to 0-1 range for display
        display_strategies = {}
        for name, score in strategies.items():
            if score > 0:
                display_strategies[name] = score
            elif score < 0:
                display_strategies[name] = abs(score)
            else:
                display_strategies[name] = 0
                
        # Decision
        total = 5
        if buy_signals >= CONFIG["MIN_STRATEGIES"]:
            signal_type = SignalType.BUY
            confidence = buy_signals / total
        elif sell_signals >= CONFIG["MIN_STRATEGIES"]:
            signal_type = SignalType.SELL
            confidence = sell_signals / total
        else:
            # Send signal data to visualizer even if no trade
            if self.signal_queue:
                try:
                    self.signal_queue.put({
                        symbol: {
                            'type': 'NONE',
                            'confidence': max(buy_signals, sell_signals) / total,
                            'strategies': display_strategies,
                            'reasons': reasons[:2]
                        }
                    })
                except:
                    pass
            return None
            
        if confidence < CONFIG["MIN_CONFIDENCE"]:
            return None
            
        # Calculate SL/TP
        sl_distance = max(atr * 1.5, 0.0015 if "JPY" not in symbol else 0.15)
        tp_distance = sl_distance * CONFIG["MIN_RR_RATIO"]
        
        if signal_type == SignalType.BUY:
            sl = current - sl_distance
            tp = current + tp_distance
        else:
            sl = current + sl_distance
            tp = current - tp_distance
            
        signal = Signal(
            type=signal_type,
            confidence=confidence,
            entry=current,
            sl=round(sl, 5),
            tp=round(tp, 5),
            reason=" | ".join(reasons[:2]),
            strategies=display_strategies
        )
        
        # Send signal to visualizer
        if self.signal_queue:
            try:
                self.signal_queue.put({
                    symbol: {
                        'type': signal_type.value,
                        'confidence': confidence,
                        'strategies': display_strategies,
                        'reasons': reasons[:3]
                    }
                })
            except:
                pass
                
        return signal
        
    def _can_trade(self) -> bool:
        """Check if we can place new trade"""
        # Time check
        hour = datetime.now(self.timezone).hour
        if 3 <= hour < 7:
            return False
            
        # Daily loss check
        if self.daily_pnl <= -CONFIG["MAX_DAILY_LOSS"] * self.balance:
            return False
            
        # Concurrent trades
        if len(self.active_trades) >= CONFIG["MAX_CONCURRENT"]:
            return False
            
        # Time between trades (5 minutes)
        if time.time() - self.last_trade_time < 300:
            return False
            
        return True
        
    def _place_order(self, symbol: str, signal: Signal) -> bool:
        """Place order"""
        # Calculate position size
        sl_distance = abs(signal.entry - signal.sl)
        volume = self._calculate_position_size(symbol, sl_distance)
        
        order = {
            "action": 1,
            "symbol": symbol,
            "volume": volume,
            "type": 0 if signal.type == SignalType.BUY else 1,
            "sl": signal.sl,
            "tp": signal.tp,
            "comment": signal.reason[:20]
        }
        
        logger.info(f"Signal: {signal.type.value} {symbol} @ {signal.entry:.5f}")
        logger.info(f"SL: {signal.sl:.5f} TP: {signal.tp:.5f} Conf: {signal.confidence:.0%}")
        
        try:
            resp = requests.post(f"{self.api_base}/trading/orders", json=order, timeout=10)
            if resp.status_code == 201:
                result = resp.json()
                trade = Trade(
                    ticket=result.get('order'),
                    symbol=symbol,
                    type=signal.type.value,
                    entry_price=result.get('price', signal.entry),
                    sl=signal.sl,
                    tp=signal.tp,
                    volume=volume,
                    entry_time=datetime.now()
                )
                
                self.active_trades[trade.ticket] = trade
                self.daily_trades += 1
                self.last_trade_time = time.time()
                
                logger.info(f"✅ Trade opened: {trade.ticket}")
                return True
        except Exception as e:
            logger.error(f"Order failed: {e}")
            
        return False
        
    def _manage_positions(self):
        """Manage open positions"""
        try:
            resp = requests.get(f"{self.api_base}/trading/positions", timeout=5)
            if resp.status_code != 200:
                return
                
            positions = resp.json()
            open_tickets = {p['ticket'] for p in positions}
            
            # Check closed
            for ticket in list(self.active_trades.keys()):
                if ticket not in open_tickets:
                    logger.info(f"Trade {ticket} closed")
                    del self.active_trades[ticket]
                    
            # Manage open
            for pos in positions:
                ticket = pos['ticket']
                if ticket not in self.active_trades:
                    continue
                    
                trade = self.active_trades[ticket]
                profit = pos.get('profit', 0)
                duration = (datetime.now() - trade.entry_time).seconds
                
                # Take profit early (1500 JPY)
                if profit > 1500:
                    logger.info(f"Taking profit: {ticket} +¥{profit:,.0f}")
                    self._close_position(ticket)
                    
                # Time exit
                elif duration > 1800:  # 30 minutes
                    logger.info(f"Time exit: {ticket}")
                    self._close_position(ticket)
                    
                # Breakeven (500 JPY)
                elif profit > 500 and duration > 300:
                    self._move_breakeven(ticket, trade)
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")
            
    def _close_position(self, ticket: int):
        """Close position"""
        try:
            requests.delete(f"{self.api_base}/trading/positions/{ticket}", timeout=5)
        except:
            pass
            
    def _move_breakeven(self, ticket: int, trade: Trade):
        """Move to breakeven"""
        if hasattr(trade, 'be_moved'):
            return
            
        pip = 0.01 if "JPY" in trade.symbol else 0.0001
        new_sl = trade.entry_price + pip if trade.type == "BUY" else trade.entry_price - pip
        
        try:
            resp = requests.patch(
                f"{self.api_base}/trading/positions/{ticket}",
                json={"sl": new_sl, "tp": trade.tp},
                timeout=5
            )
            if resp.status_code == 200:
                trade.be_moved = True
                logger.info(f"Breakeven: {ticket}")
        except:
            pass
            
    def _run_loop(self):
        """Main trading loop"""
        cycle = 0
        last_hour = datetime.now().hour
        
        try:
            while self.running:
                cycle += 1
                
                # Hourly reset
                current_hour = datetime.now().hour
                if current_hour != last_hour and current_hour == 0:
                    self.daily_pnl = 0
                    self.daily_trades = 0
                    last_hour = current_hour
                    
                # Manage positions
                self._manage_positions()
                
                # Always analyze all symbols for visualization
                can_trade = self._can_trade()
                for symbol in CONFIG["SYMBOLS"]:
                    # Check spread
                    spread_ok, spread = self._check_spread(symbol)
                    
                    # Get data
                    df = self._get_market_data(symbol)
                    if df is None:
                        continue
                        
                    # Analyze (this will send signals to visualizer)
                    signal = self._analyze(symbol, df)
                    
                    # Only place orders if we can trade
                    if can_trade and signal and spread_ok:
                        if self._place_order(symbol, signal):
                            break
                                
                # Status
                if cycle % 40 == 0:
                    logger.info(f"Active: {len(self.active_trades)} Daily: {self.daily_trades}")
                    
                time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.running = False
            
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        logger.info("Stopping Trading Engine")

def main():
    """Run standalone engine"""
    engine = TradingEngine()
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()

if __name__ == "__main__":
    main()