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
    "SYMBOLS": [],  # Will be populated dynamically
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.60,  # Lowered as we now have quality filter
    "MIN_QUALITY": 0.60,     # Signal quality threshold
    "MIN_STRATEGIES": 5,     # Increased from 3 to 5 (out of 10 indicators)
    "MAX_SPREAD_PIPS": 2.5,
    "RISK_PER_TRADE": 0.01,
    "MAX_DAILY_LOSS": 0.03,
    "MAX_CONCURRENT": 2,
    "MIN_RR_RATIO": 1.5,
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY",
    "SYMBOL_FILTER": "FOREX",  # Filter for forex pairs only
    "MIN_VOLUME": 0.01  # Minimum tradable volume
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
        
        # Symbols list
        self.tradable_symbols = []
        
    def start(self):
        """Initialize and start trading"""
        if not self._check_connection():
            logger.error("Cannot connect to API")
            return False
            
        self.balance = self._get_balance()
        if not self.balance:
            logger.error("Cannot get account balance")
            return False
            
        # Discover tradable symbols
        self.tradable_symbols = self._discover_symbols()
        if not self.tradable_symbols:
            logger.error("No tradable symbols found")
            return False
            
        logger.info(f"Starting Trading Engine")
        logger.info(f"Balance: ¥{self.balance:,.0f}")
        logger.info(f"Trading {len(self.tradable_symbols)} symbols: {', '.join(self.tradable_symbols[:5])}...")
        
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
        
    def _discover_symbols(self) -> List[str]:
        """Discover all tradable forex symbols"""
        try:
            resp = requests.get(f"{self.api_base}/market/symbols", timeout=10)
            if resp.status_code == 200:
                all_symbols = resp.json()
                
                # Filter for forex pairs
                forex_symbols = []
                for symbol in all_symbols:
                    name = symbol.get('name', '')
                    
                    # Check if it's a forex pair (contains currency codes)
                    if any(curr in name for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']):
                        # Check if it's tradable
                        if symbol.get('trade_mode', 0) > 0:
                            # Check minimum volume
                            min_vol = symbol.get('volume_min', 0)
                            if min_vol <= CONFIG["MIN_VOLUME"]:
                                forex_symbols.append(name)
                                
                logger.info(f"Discovered {len(forex_symbols)} tradable forex pairs")
                return forex_symbols[:20]  # Limit to 20 symbols to avoid overload
                
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            
        # Fallback to default symbols
        return ["EURUSD#", "USDJPY#", "GBPUSD#", "USDCHF#", "AUDUSD#", "USDCAD#", "NZDUSD#", "EURJPY#", "GBPJPY#", "EURGBP#"]
        
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
        
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_5'] = pd.Series(close).rolling(5).mean().iloc[-1]
        indicators['sma_10'] = pd.Series(close).rolling(10).mean().iloc[-1]
        indicators['sma_20'] = pd.Series(close).rolling(20).mean().iloc[-1]
        indicators['sma_50'] = pd.Series(close).rolling(50).mean().iloc[-1] if len(close) >= 50 else indicators['sma_20']
        indicators['ema_12'] = pd.Series(close).ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = pd.Series(close).ewm(span=26).mean().iloc[-1]
        
        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        signal_line = pd.Series(close).ewm(span=12).mean().iloc[-1] - pd.Series(close).ewm(span=26).mean().iloc[-1]
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = macd_line - signal_line
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - 100/(1 + rs)).iloc[-1]
        
        # Stochastic
        low_14 = pd.Series(low).rolling(14).min().iloc[-1]
        high_14 = pd.Series(high).rolling(14).max().iloc[-1]
        indicators['stoch_k'] = 100 * (close[-1] - low_14) / (high_14 - low_14) if high_14 != low_14 else 50
        
        # ADX
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        indicators['atr'] = atr
        
        # Calculate +DI and -DI for ADX
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        indicators['adx'] = dx
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di
        
        # Bollinger Bands
        bb_mean = pd.Series(close).rolling(20).mean().iloc[-1]
        bb_std = pd.Series(close).rolling(20).std().iloc[-1]
        indicators['bb_upper'] = bb_mean + 2 * bb_std
        indicators['bb_lower'] = bb_mean - 2 * bb_std
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_mean
        
        # Volume indicators
        indicators['volume_sma'] = pd.Series(volume).rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Price action
        indicators['current'] = close[-1]
        indicators['prev_close'] = close[-2]
        indicators['high_low_range'] = high[-1] - low[-1]
        
        return indicators
        
    def _detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Detect support/resistance levels and market structure"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        structure = {}
        
        # Find recent swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(high) - 2):
            if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                swing_highs.append((i, high[i]))
            if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                swing_lows.append((i, low[i]))
        
        # Get recent support/resistance
        if swing_highs:
            structure['resistance'] = max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]
        else:
            structure['resistance'] = max(high[-20:])
            
        if swing_lows:
            structure['support'] = min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]
        else:
            structure['support'] = min(low[-20:])
            
        # Trend structure
        higher_highs = 0
        higher_lows = 0
        lower_highs = 0
        lower_lows = 0
        
        for i in range(1, min(len(swing_highs), 3)):
            if swing_highs[-(i+1)][1] < swing_highs[-i][1]:
                higher_highs += 1
            else:
                lower_highs += 1
                
        for i in range(1, min(len(swing_lows), 3)):
            if swing_lows[-(i+1)][1] < swing_lows[-i][1]:
                higher_lows += 1
            else:
                lower_lows += 1
                
        if higher_highs > lower_highs and higher_lows > lower_lows:
            structure['trend'] = 'uptrend'
        elif lower_highs > higher_highs and lower_lows > higher_lows:
            structure['trend'] = 'downtrend'
        else:
            structure['trend'] = 'ranging'
            
        return structure
        
    def _analyze(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Advanced market analysis with deep signal selection"""
        if len(df) < 50:
            return None
            
        # Calculate all indicators
        ind = self._calculate_indicators(df)
        structure = self._detect_market_structure(df)
        
        # Initialize advanced strategy scores
        strategies = {
            "Trend": 0,
            "RSI": 0,
            "MACD": 0,
            "Stochastic": 0,
            "Bollinger": 0,
            "ADX": 0,
            "Structure": 0,
            "Volume": 0,
            "Momentum": 0,
            "Divergence": 0
        }
        
        buy_score = 0
        sell_score = 0
        reasons = []
        signal_quality = 0
        
        # 1. Trend Analysis (Weight: 15%)
        trend_score = 0
        if ind['sma_5'] > ind['sma_10'] > ind['sma_20']:
            trend_score += 0.5
            if ind['current'] > ind['sma_5']:
                trend_score += 0.5
                reasons.append("Strong Uptrend")
        elif ind['sma_5'] < ind['sma_10'] < ind['sma_20']:
            trend_score -= 0.5
            if ind['current'] < ind['sma_5']:
                trend_score -= 0.5
                reasons.append("Strong Downtrend")
                
        strategies["Trend"] = abs(trend_score)
        if trend_score > 0:
            buy_score += trend_score * 15
        else:
            sell_score += abs(trend_score) * 15
            
        # 2. RSI Analysis (Weight: 12%)
        rsi_score = 0
        if ind['rsi'] < 30:
            rsi_score = 1.0
            reasons.append(f"RSI Oversold {ind['rsi']:.0f}")
        elif 30 <= ind['rsi'] < 40:
            rsi_score = (40 - ind['rsi']) / 10
        elif 60 < ind['rsi'] <= 70:
            rsi_score = -(ind['rsi'] - 60) / 10
        elif ind['rsi'] > 70:
            rsi_score = -1.0
            reasons.append(f"RSI Overbought {ind['rsi']:.0f}")
            
        strategies["RSI"] = abs(rsi_score)
        if rsi_score > 0:
            buy_score += rsi_score * 12
        else:
            sell_score += abs(rsi_score) * 12
            
        # 3. MACD Analysis (Weight: 13%)
        macd_score = 0
        if ind['macd'] > ind['macd_signal'] and ind['macd_histogram'] > 0:
            macd_score = min(ind['macd_histogram'] * 100, 1.0)
            if ind['macd'] > 0:
                reasons.append("MACD Bull Cross")
        elif ind['macd'] < ind['macd_signal'] and ind['macd_histogram'] < 0:
            macd_score = max(ind['macd_histogram'] * 100, -1.0)
            if ind['macd'] < 0:
                reasons.append("MACD Bear Cross")
                
        strategies["MACD"] = abs(macd_score)
        if macd_score > 0:
            buy_score += macd_score * 13
        else:
            sell_score += abs(macd_score) * 13
            
        # 4. Stochastic Analysis (Weight: 10%)
        stoch_score = 0
        if ind['stoch_k'] < 20:
            stoch_score = 1.0
            reasons.append("Stoch Oversold")
        elif 20 <= ind['stoch_k'] < 30:
            stoch_score = (30 - ind['stoch_k']) / 10
        elif 70 < ind['stoch_k'] <= 80:
            stoch_score = -(ind['stoch_k'] - 70) / 10
        elif ind['stoch_k'] > 80:
            stoch_score = -1.0
            reasons.append("Stoch Overbought")
            
        strategies["Stochastic"] = abs(stoch_score)
        if stoch_score > 0:
            buy_score += stoch_score * 10
        else:
            sell_score += abs(stoch_score) * 10
            
        # 5. Bollinger Bands Analysis (Weight: 10%)
        bb_score = 0
        bb_position = (ind['current'] - ind['bb_lower']) / (ind['bb_upper'] - ind['bb_lower'])
        
        if bb_position < 0:
            bb_score = 1.0
            reasons.append("Below BB Lower")
        elif bb_position < 0.2:
            bb_score = (0.2 - bb_position) / 0.2
        elif bb_position > 1:
            bb_score = -1.0
            reasons.append("Above BB Upper")
        elif bb_position > 0.8:
            bb_score = -(bb_position - 0.8) / 0.2
            
        strategies["Bollinger"] = abs(bb_score)
        if bb_score > 0:
            buy_score += bb_score * 10
        else:
            sell_score += abs(bb_score) * 10
            
        # 6. ADX Trend Strength (Weight: 10%)
        adx_score = 0
        if ind['adx'] > 25:  # Strong trend
            if ind['plus_di'] > ind['minus_di']:
                adx_score = min(ind['adx'] / 50, 1.0)
                reasons.append(f"ADX Strong Trend {ind['adx']:.0f}")
            else:
                adx_score = -min(ind['adx'] / 50, 1.0)
                reasons.append(f"ADX Strong Trend {ind['adx']:.0f}")
                
        strategies["ADX"] = abs(adx_score)
        if adx_score > 0:
            buy_score += adx_score * 10
        else:
            sell_score += abs(adx_score) * 10
            
        # 7. Market Structure (Weight: 12%)
        struct_score = 0
        if structure['trend'] == 'uptrend':
            struct_score = 0.5
            # Check if near support
            if abs(ind['current'] - structure['support']) / ind['current'] < 0.005:
                struct_score = 1.0
                reasons.append("At Support")
        elif structure['trend'] == 'downtrend':
            struct_score = -0.5
            # Check if near resistance
            if abs(ind['current'] - structure['resistance']) / ind['current'] < 0.005:
                struct_score = -1.0
                reasons.append("At Resistance")
                
        strategies["Structure"] = abs(struct_score)
        if struct_score > 0:
            buy_score += struct_score * 12
        else:
            sell_score += abs(struct_score) * 12
            
        # 8. Volume Analysis (Weight: 8%)
        vol_score = 0
        if ind['volume_ratio'] > 1.5:
            if ind['current'] > ind['prev_close']:
                vol_score = min((ind['volume_ratio'] - 1) / 2, 1.0)
                reasons.append("High Volume Buy")
            else:
                vol_score = -min((ind['volume_ratio'] - 1) / 2, 1.0)
                reasons.append("High Volume Sell")
                
        strategies["Volume"] = abs(vol_score)
        if vol_score > 0:
            buy_score += vol_score * 8
        else:
            sell_score += abs(vol_score) * 8
            
        # 9. Momentum (Weight: 5%)
        momentum = (ind['current'] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        mom_score = 0
        if momentum > 0.002:
            mom_score = min(momentum * 50, 1.0)
            reasons.append("Positive Momentum")
        elif momentum < -0.002:
            mom_score = max(momentum * 50, -1.0)
            reasons.append("Negative Momentum")
            
        strategies["Momentum"] = abs(mom_score)
        if mom_score > 0:
            buy_score += mom_score * 5
        else:
            sell_score += abs(mom_score) * 5
            
        # 10. Divergence Detection (Weight: 5%)
        # Check for RSI divergence
        div_score = 0
        if len(df) >= 20:
            recent_lows = pd.Series(df['low'].iloc[-20:]).rolling(3).min()
            recent_highs = pd.Series(df['high'].iloc[-20:]).rolling(3).max()
            
            # Bullish divergence: price makes lower low but RSI makes higher low
            if df['low'].iloc[-1] < df['low'].iloc[-10] and ind['rsi'] > 40:
                div_score = 0.8
                reasons.append("Bullish Divergence")
            # Bearish divergence: price makes higher high but RSI makes lower high
            elif df['high'].iloc[-1] > df['high'].iloc[-10] and ind['rsi'] < 60:
                div_score = -0.8
                reasons.append("Bearish Divergence")
                
        strategies["Divergence"] = abs(div_score)
        if div_score > 0:
            buy_score += div_score * 5
        else:
            sell_score += abs(div_score) * 5
            
        # Calculate final scores
        total_buy_score = buy_score / 100
        total_sell_score = sell_score / 100
        
        # Determine signal with quality assessment
        signal_type = None
        confidence = 0
        
        if total_buy_score > total_sell_score and total_buy_score >= 0.5:
            signal_type = SignalType.BUY
            confidence = total_buy_score
            signal_quality = self._assess_signal_quality(strategies, 'buy', ind, structure)
        elif total_sell_score > total_buy_score and total_sell_score >= 0.5:
            signal_type = SignalType.SELL
            confidence = total_sell_score
            signal_quality = self._assess_signal_quality(strategies, 'sell', ind, structure)
            
        # Normalize strategy scores for display
        display_strategies = {}
        for name, score in strategies.items():
            display_strategies[name] = abs(score)
            
        # Send to visualizer
        if self.signal_queue:
            try:
                self.signal_queue.put({
                    symbol: {
                        'type': signal_type.value if signal_type else 'NONE',
                        'confidence': confidence,
                        'strategies': display_strategies,
                        'reasons': reasons[:3],
                        'quality': signal_quality
                    }
                })
            except:
                pass
                
        # Only return signal if confidence meets threshold and quality is good
        if signal_type and confidence >= CONFIG["MIN_CONFIDENCE"] and signal_quality >= CONFIG["MIN_QUALITY"]:
            # Calculate dynamic SL/TP based on market conditions
            atr = ind['atr']
            
            # Adjust SL based on volatility and signal quality
            sl_multiplier = 1.5 - (signal_quality - 0.6) * 0.5  # Better signals get tighter stops
            sl_distance = max(atr * sl_multiplier, 0.0015 if "JPY" not in symbol else 0.15)
            
            # Dynamic TP based on market conditions
            if ind['adx'] > 30:  # Strong trend
                tp_multiplier = 2.0
            else:  # Ranging market
                tp_multiplier = 1.5
                
            tp_distance = sl_distance * tp_multiplier
            
            if signal_type == SignalType.BUY:
                sl = ind['current'] - sl_distance
                tp = ind['current'] + tp_distance
            else:
                sl = ind['current'] + sl_distance
                tp = ind['current'] - tp_distance
                
            return Signal(
                type=signal_type,
                confidence=confidence,
                entry=ind['current'],
                sl=round(sl, 5),
                tp=round(tp, 5),
                reason=" | ".join(reasons[:3]),
                strategies=display_strategies
            )
            
        return None
        
    def _assess_signal_quality(self, strategies: Dict, direction: str, indicators: Dict, structure: Dict) -> float:
        """Assess the quality of a trading signal (0-1)"""
        quality_score = 0
        
        # 1. Multiple confirmations (30%)
        confirmations = sum(1 for s in strategies.values() if s > 0.5)
        quality_score += (confirmations / len(strategies)) * 0.3
        
        # 2. Trend alignment (25%)
        if direction == 'buy' and structure['trend'] == 'uptrend':
            quality_score += 0.25
        elif direction == 'sell' and structure['trend'] == 'downtrend':
            quality_score += 0.25
        elif structure['trend'] == 'ranging':
            quality_score += 0.1  # Partial credit for ranging markets
            
        # 3. Not overbought/oversold (20%)
        if direction == 'buy' and indicators['rsi'] < 70 and indicators['stoch_k'] < 80:
            quality_score += 0.2
        elif direction == 'sell' and indicators['rsi'] > 30 and indicators['stoch_k'] > 20:
            quality_score += 0.2
            
        # 4. Volume confirmation (15%)
        if indicators['volume_ratio'] > 1.2:
            quality_score += 0.15
            
        # 5. ADX strength (10%)
        if indicators['adx'] > 25:
            quality_score += 0.1
            
        return min(quality_score, 1.0)
        
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
                for symbol in self.tradable_symbols:
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