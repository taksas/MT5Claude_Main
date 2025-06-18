#!/usr/bin/env python3
"""
Ultra-Short-Term Scalping Strategies for 1-10 Minute Trades
Optimized for rapid profit-taking with high win rates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from trading_strategies import TradingStrategy, SignalType, TradingSignal
import logging

logger = logging.getLogger(__name__)

class ScalpingStrategy(TradingStrategy):
    """Base class for ultra-short-term scalping strategies"""
    
    def calculate_scalping_levels(self, entry_price: float, signal_type: SignalType, 
                                 atr: float, pip_value: float = 0.0001) -> Tuple[float, float]:
        """Calculate tight stop loss and take profit for scalping"""
        # Very tight stops for 1-10 minute trades
        risk_ratio = 1.5  # Lower risk ratio for higher win rate
        stop_distance = atr * 0.8  # Tighter stop loss
        
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * risk_ratio)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * risk_ratio)
        
        return stop_loss, take_profit

class MicroMomentumStrategy(ScalpingStrategy):
    """Catches micro momentum moves on 1-minute timeframe"""
    
    def __init__(self):
        super().__init__("MicroMomentum")
        self.fast_period = 5
        self.slow_period = 8
        self.rsi_period = 7
        self.min_momentum = 0.0002  # Minimum price movement
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < max(self.slow_period, self.rsi_period) + 5:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate micro EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period).mean()
        
        # Calculate momentum
        df['momentum'] = df['close'] - df['close'].shift(3)
        
        # Calculate RSI for confirmation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        # Avoid division by zero in RSI calculation
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate micro ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=10).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        momentum = df['momentum'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        ema_diff = df['ema_fast'].iloc[-1] - df['ema_slow'].iloc[-1]
        
        # Volume spike detection
        volume_ma = df['tick_volume'].rolling(window=10).mean().iloc[-1]
        volume_spike = df['tick_volume'].iloc[-1] > volume_ma * 1.3
        
        # Bullish micro momentum
        if (momentum > self.min_momentum and ema_diff > 0 and 
            35 < rsi < 65 and volume_spike):  # RSI in neutral zone for momentum
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.85, 0.6 + abs(momentum) * 500 + (0.1 if volume_spike else 0))
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Micro momentum up {momentum:.5f}", df['time'].iloc[-1]
            )
        
        # Bearish micro momentum
        elif (momentum < -self.min_momentum and ema_diff < 0 and 
              35 < rsi < 65 and volume_spike):
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.85, 0.6 + abs(momentum) * 500 + (0.1 if volume_spike else 0))
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Micro momentum down {momentum:.5f}", df['time'].iloc[-1]
            )
        
        # Log why no signal was generated (debug)
        self.logger.debug(f"MicroMomentum no signal: momentum={momentum:.5f}, ema_diff={ema_diff:.5f}, "
                         f"rsi={rsi:.1f}, volume_spike={volume_spike}")
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, "No momentum", df['time'].iloc[-1])

class OrderFlowScalpingStrategy(ScalpingStrategy):
    """Analyzes order flow imbalances for quick scalps"""
    
    def __init__(self):
        super().__init__("OrderFlowScalping")
        self.volume_window = 5
        self.price_levels = 10
        self.imbalance_threshold = 1.5
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < 20:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Analyze bid/ask pressure through price and volume
        df['price_change'] = df['close'] - df['open']
        df['buy_pressure'] = df['tick_volume'] * (df['price_change'] > 0)
        df['sell_pressure'] = df['tick_volume'] * (df['price_change'] < 0)
        
        # Calculate recent order flow imbalance
        recent_buy = df['buy_pressure'].tail(self.volume_window).sum()
        recent_sell = df['sell_pressure'].tail(self.volume_window).sum()
        
        # Handle edge cases to avoid division by zero
        if recent_sell > 0 and recent_buy > 0:
            imbalance_ratio = recent_buy / recent_sell
        elif recent_buy > 0 and recent_sell == 0:
            imbalance_ratio = 2.0  # Strong buy pressure
        elif recent_sell > 0 and recent_buy == 0:
            imbalance_ratio = 0.5  # Strong sell pressure
        else:
            imbalance_ratio = 1.0  # Neutral
        
        # Price level analysis
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(self.price_levels).max()
        recent_low = df['low'].tail(self.price_levels).min()
        
        # Calculate price position safely
        price_range = recent_high - recent_low
        if price_range > 0:
            price_position = (current_price - recent_low) / price_range
        else:
            price_position = 0.5  # Default to middle if no range
        
        # ATR for stops
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=10).mean().iloc[-1]
        
        # Bullish order flow
        if (imbalance_ratio > self.imbalance_threshold and 
            price_position < 0.7 and  # Not at resistance
            df['close'].iloc[-1] > df['close'].iloc[-2]):  # Upward momentum
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.82, 0.55 + (imbalance_ratio - 1) * 0.2)
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Buy pressure {imbalance_ratio:.2f}x", df['time'].iloc[-1]
            )
        
        # Bearish order flow
        elif (imbalance_ratio < (1 / self.imbalance_threshold) and 
              price_position > 0.3 and  # Not at support
              df['close'].iloc[-1] < df['close'].iloc[-2]):  # Downward momentum
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.SELL, atr)
            sell_pressure_ratio = 1/imbalance_ratio if imbalance_ratio > 0 else 2.0
            confidence = min(0.82, 0.55 + (sell_pressure_ratio - 1) * 0.2)
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Sell pressure {sell_pressure_ratio:.2f}x", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"Order flow balanced", df['time'].iloc[-1])

class PriceActionScalpingStrategy(ScalpingStrategy):
    """Pure price action patterns for quick scalps"""
    
    def __init__(self):
        super().__init__("PriceActionScalping")
        self.min_wick_ratio = 0.6
        self.min_body_size = 0.0001
        self.pattern_lookback = 5
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate candlestick metrics
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Identify patterns
        current_idx = -1
        current_price = df['close'].iloc[current_idx]
        
        # Pin bar detection
        if df['range'].iloc[current_idx] > 0:
            upper_wick_ratio = df['upper_wick'].iloc[current_idx] / df['range'].iloc[current_idx]
            lower_wick_ratio = df['lower_wick'].iloc[current_idx] / df['range'].iloc[current_idx]
        else:
            upper_wick_ratio = 0
            lower_wick_ratio = 0
            
            # Bullish pin bar (long lower wick)
            if (lower_wick_ratio > self.min_wick_ratio and 
                df['close'].iloc[current_idx] > df['open'].iloc[current_idx] and
                df['low'].iloc[current_idx] < df['low'].iloc[current_idx-1]):
                
                # ATR for stops
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                atr = df['tr'].rolling(window=10).mean().iloc[-1]
                
                sl, tp = self.calculate_scalping_levels(current_price, SignalType.BUY, atr)
                confidence = min(0.80, 0.55 + lower_wick_ratio * 0.3)
                
                return TradingSignal(
                    SignalType.BUY, confidence, current_price, sl, tp,
                    f"Bullish pin bar", df['time'].iloc[-1]
                )
            
            # Bearish pin bar (long upper wick)
            elif (upper_wick_ratio > self.min_wick_ratio and 
                  df['close'].iloc[current_idx] < df['open'].iloc[current_idx] and
                  df['high'].iloc[current_idx] > df['high'].iloc[current_idx-1]):
                
                # ATR for stops
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                atr = df['tr'].rolling(window=10).mean().iloc[-1]
                
                sl, tp = self.calculate_scalping_levels(current_price, SignalType.SELL, atr)
                confidence = min(0.80, 0.55 + upper_wick_ratio * 0.3)
                
                return TradingSignal(
                    SignalType.SELL, confidence, current_price, sl, tp,
                    f"Bearish pin bar", df['time'].iloc[-1]
                )
        
        # Engulfing pattern
        if (df['body'].iloc[current_idx] > self.min_body_size and
            df['body'].iloc[current_idx-1] > 0):
            
            # Bullish engulfing
            if (df['close'].iloc[current_idx] > df['open'].iloc[current_idx] and
                df['close'].iloc[current_idx-1] < df['open'].iloc[current_idx-1] and
                df['open'].iloc[current_idx] < df['close'].iloc[current_idx-1] and
                df['close'].iloc[current_idx] > df['open'].iloc[current_idx-1]):
                
                # ATR for stops
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                atr = df['tr'].rolling(window=10).mean().iloc[-1]
                
                sl, tp = self.calculate_scalping_levels(current_price, SignalType.BUY, atr)
                confidence = 0.78
                
                return TradingSignal(
                    SignalType.BUY, confidence, current_price, sl, tp,
                    f"Bullish engulfing", df['time'].iloc[-1]
                )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           "No price action signal", df['time'].iloc[-1])

class VolatilityBreakoutScalpingStrategy(ScalpingStrategy):
    """Catches volatility expansion for quick profits"""
    
    def __init__(self):
        super().__init__("VolatilityBreakout")
        self.bb_period = 10
        self.bb_std = 1.5
        self.vol_expansion_threshold = 1.5
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < 20:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std)
        
        # Calculate volatility metrics
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_ma'] = df['bb_width'].rolling(window=20).mean()
        
        current_price = df['close'].iloc[-1]
        bb_width_ratio = df['bb_width'].iloc[-1] / df['bb_width_ma'].iloc[-1] if df['bb_width_ma'].iloc[-1] > 0 else 1
        
        # Detect squeeze and expansion
        in_squeeze = bb_width_ratio < 0.8  # Volatility contraction
        expanding = bb_width_ratio > self.vol_expansion_threshold
        
        # ATR for stops
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=10).mean().iloc[-1]
        
        # Bullish breakout
        if (current_price > df['bb_upper'].iloc[-1] and 
            df['close'].iloc[-2] <= df['bb_upper'].iloc[-2] and
            (expanding or in_squeeze)):
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.83, 0.60 + (bb_width_ratio - 1) * 0.15)
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Volatility breakout up", df['time'].iloc[-1]
            )
        
        # Bearish breakout
        elif (current_price < df['bb_lower'].iloc[-1] and 
              df['close'].iloc[-2] >= df['bb_lower'].iloc[-2] and
              (expanding or in_squeeze)):
            
            sl, tp = self.calculate_scalping_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.83, 0.60 + (bb_width_ratio - 1) * 0.15)
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Volatility breakout down", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"No volatility breakout", df['time'].iloc[-1])

class UltraScalpingEnsemble:
    """Ensemble of ultra-short-term scalping strategies"""
    
    def __init__(self):
        self.strategies = [
            MicroMomentumStrategy(),
            OrderFlowScalpingStrategy(),
            PriceActionScalpingStrategy(),
            VolatilityBreakoutScalpingStrategy()
        ]
        self.logger = logging.getLogger(__name__)
        self.min_ensemble_confidence = 0.75
        self.min_agreeing_strategies = 2
    
    def get_ensemble_signal(self, data: List[Dict[str, Any]]) -> Optional[TradingSignal]:
        """Get consensus signal from multiple scalping strategies"""
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(data)
                if signal.signal != SignalType.HOLD and signal.confidence >= 0.70:
                    signals.append(signal)
                    self.logger.debug(f"{strategy.name}: {signal.signal.value} "
                                    f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
            except Exception as e:
                self.logger.error(f"Error in {strategy.name}: {e}")
        
        if not signals:
            return None
        
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        # For scalping, we want strong agreement
        if len(buy_signals) >= self.min_agreeing_strategies and len(sell_signals) == 0:
            best_buy = max(buy_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            
            if avg_confidence >= self.min_ensemble_confidence:
                # Use tightest stop loss from all signals
                min_sl = max([s.stop_loss for s in buy_signals])
                # Use closest take profit for higher win rate
                min_tp = min([s.take_profit for s in buy_signals])
                
                return TradingSignal(
                    SignalType.BUY, avg_confidence, best_buy.entry_price,
                    min_sl, min_tp,
                    f"Scalp BUY ({len(buy_signals)} strategies)", best_buy.timestamp
                )
        
        elif len(sell_signals) >= self.min_agreeing_strategies and len(buy_signals) == 0:
            best_sell = max(sell_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            
            if avg_confidence >= self.min_ensemble_confidence:
                # Use tightest stop loss from all signals
                max_sl = min([s.stop_loss for s in sell_signals])
                # Use closest take profit for higher win rate
                max_tp = max([s.take_profit for s in sell_signals])
                
                return TradingSignal(
                    SignalType.SELL, avg_confidence, best_sell.entry_price,
                    max_sl, max_tp,
                    f"Scalp SELL ({len(sell_signals)} strategies)", best_sell.timestamp
                )
        
        return None