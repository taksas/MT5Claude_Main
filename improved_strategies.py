#!/usr/bin/env python3
"""
Improved Forex Trading Strategies for 2025
Based on current market research and backtesting optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from trading_strategies import TradingStrategy, SignalType, TradingSignal
import logging

class ImprovedMomentumStrategy(TradingStrategy):
    """
    Improved momentum strategy using EMA crossover with volume confirmation
    Optimized for 5-30 minute timeframes
    """
    def __init__(self):
        super().__init__("ImprovedMomentum")
        self.fast_ema = 12
        self.slow_ema = 26
        self.volume_threshold = 1.5
        self.min_confidence = 0.65
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < max(self.fast_ema, self.slow_ema) + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema).mean()
        
        # Calculate volume average
        df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
        
        # Calculate ATR for stop loss
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        prev_fast = df['ema_fast'].iloc[-2]
        prev_slow = df['ema_slow'].iloc[-2]
        curr_fast = df['ema_fast'].iloc[-1]
        curr_slow = df['ema_slow'].iloc[-1]
        
        # Volume confirmation
        volume_spike = df['tick_volume'].iloc[-1] > df['volume_ma'].iloc[-1] * self.volume_threshold
        
        # Trend strength
        price_above_fast = current_price > curr_fast
        price_above_slow = current_price > curr_slow
        ema_separation = abs(curr_fast - curr_slow) / current_price
        
        # Bullish crossover
        if (prev_fast <= prev_slow and curr_fast > curr_slow and 
            volume_spike and price_above_fast and ema_separation > 0.001):
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.9, self.min_confidence + ema_separation * 100 + (0.1 if volume_spike else 0))
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"EMA bullish crossover with volume", df['time'].iloc[-1]
            )
        
        # Bearish crossover
        elif (prev_fast >= prev_slow and curr_fast < curr_slow and 
              volume_spike and not price_above_slow and ema_separation > 0.001):
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.9, self.min_confidence + ema_separation * 100 + (0.1 if volume_spike else 0))
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"EMA bearish crossover with volume", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, "No EMA signal", df['time'].iloc[-1])

class VWAPScalpingStrategy(TradingStrategy):
    """
    VWAP-based scalping strategy optimized for short-term trades
    """
    def __init__(self):
        super().__init__("VWAPScalping")
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.min_confidence = 0.70
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < 50:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Price distance from VWAP
        vwap_distance = abs(current_price - current_vwap) / current_price
        
        # Bullish setup: Price above VWAP, RSI oversold recovery
        if (current_price > current_vwap and 
            prev_rsi <= self.rsi_oversold and current_rsi > self.rsi_oversold and
            vwap_distance < 0.002):  # Close to VWAP
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr * 0.8)  # Tighter stops for scalping
            confidence = min(0.9, self.min_confidence + (self.rsi_oversold - prev_rsi) / 20)
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"VWAP bullish scalp setup RSI:{current_rsi:.1f}", df['time'].iloc[-1]
            )
        
        # Bearish setup: Price below VWAP, RSI overbought decline
        elif (current_price < current_vwap and 
              prev_rsi >= self.rsi_overbought and current_rsi < self.rsi_overbought and
              vwap_distance < 0.002):
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr * 0.8)
            confidence = min(0.9, self.min_confidence + (prev_rsi - self.rsi_overbought) / 20)
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"VWAP bearish scalp setup RSI:{current_rsi:.1f}", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"VWAP hold RSI:{current_rsi:.1f}", df['time'].iloc[-1])

class KeltnerChannelStrategy(TradingStrategy):
    """
    Keltner Channel breakout strategy with RSI confirmation
    """
    def __init__(self):
        super().__init__("KeltnerChannel")
        self.period = 20
        self.multiplier = 2.0
        self.rsi_period = 14
        self.min_confidence = 0.68
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < max(self.period, self.rsi_period) + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate Keltner Channels
        df['ema'] = df['close'].ewm(span=self.period).mean()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.period).mean()
        df['upper_band'] = df['ema'] + (df['atr'] * self.multiplier)
        df['lower_band'] = df['ema'] - (df['atr'] * self.multiplier)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        upper_band = df['upper_band'].iloc[-1]
        lower_band = df['lower_band'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Bullish breakout above upper band with RSI momentum
        if (prev_price <= df['upper_band'].iloc[-2] and current_price > upper_band and
            current_rsi > 50 and current_rsi < 80):  # Avoid extreme overbought
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.9, self.min_confidence + (current_rsi - 50) / 100)
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Keltner upper breakout RSI:{current_rsi:.1f}", df['time'].iloc[-1]
            )
        
        # Bearish breakout below lower band with RSI momentum
        elif (prev_price >= df['lower_band'].iloc[-2] and current_price < lower_band and
              current_rsi < 50 and current_rsi > 20):  # Avoid extreme oversold
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.9, self.min_confidence + (50 - current_rsi) / 100)
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Keltner lower breakout RSI:{current_rsi:.1f}", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"Keltner no signal RSI:{current_rsi:.1f}", df['time'].iloc[-1])

class ALMAStochasticStrategy(TradingStrategy):
    """
    ALMA (Arnaud Legoux Moving Average) with Stochastic Oscillator
    """
    def __init__(self):
        super().__init__("ALMAStochastic")
        self.alma_period = 21
        self.alma_offset = 0.85
        self.alma_sigma = 6
        self.stoch_k = 21
        self.stoch_d = 3
        self.min_confidence = 0.65
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < self.alma_period + 20:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate ALMA
        df['alma'] = self._calculate_alma(df['close'])
        
        # Calculate Stochastic
        df['lowest_low'] = df['low'].rolling(window=self.stoch_k).min()
        df['highest_high'] = df['high'].rolling(window=self.stoch_k).max()
        df['stoch_k'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
        df['stoch_d'] = df['stoch_k'].rolling(window=self.stoch_d).mean()
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        current_alma = df['alma'].iloc[-1]
        prev_alma = df['alma'].iloc[-2]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        prev_stoch_k = df['stoch_k'].iloc[-2]
        
        # Trend determination
        alma_trending_up = current_alma > prev_alma
        price_above_alma = current_price > current_alma
        
        # Bullish setup: Price above ALMA, Stochastic oversold recovery
        if (price_above_alma and alma_trending_up and 
            prev_stoch_k <= 20 and stoch_k > 20 and stoch_k > stoch_d):
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.9, self.min_confidence + (stoch_k - 20) / 100)
            
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"ALMA bullish trend Stoch:{stoch_k:.1f}", df['time'].iloc[-1]
            )
        
        # Bearish setup: Price below ALMA, Stochastic overbought decline
        elif (not price_above_alma and not alma_trending_up and 
              prev_stoch_k >= 80 and stoch_k < 80 and stoch_k < stoch_d):
            
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.9, self.min_confidence + (80 - stoch_k) / 100)
            
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"ALMA bearish trend Stoch:{stoch_k:.1f}", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"ALMA hold Stoch:{stoch_k:.1f}", df['time'].iloc[-1])
    
    def _calculate_alma(self, prices: pd.Series) -> pd.Series:
        """Calculate ALMA (Arnaud Legoux Moving Average)"""
        period = self.alma_period
        offset = self.alma_offset
        sigma = self.alma_sigma
        
        m = offset * (period - 1)
        s = period / sigma
        
        weights = []
        for i in range(period):
            weights.append(np.exp(-((i - m) ** 2) / (2 * s * s)))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        alma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                alma_values.append(np.nan)
            else:
                window_prices = prices.iloc[i - period + 1:i + 1].values
                alma_value = np.sum(window_prices * weights)
                alma_values.append(alma_value)
        
        return pd.Series(alma_values, index=prices.index)

class ImprovedStrategyEnsemble:
    """
    Improved strategy ensemble with optimized strategies for 2025
    """
    def __init__(self):
        self.strategies = [
            ImprovedMomentumStrategy(),
            VWAPScalpingStrategy(),
            KeltnerChannelStrategy(),
            ALMAStochasticStrategy()
        ]
        self.logger = logging.getLogger(__name__)
        self.min_ensemble_confidence = 0.70
        self.min_agreeing_strategies = 2
    
    def get_ensemble_signal(self, data: List[Dict[str, Any]]) -> Optional[TradingSignal]:
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(data)
                if signal.signal != SignalType.HOLD and signal.confidence >= 0.65:
                    signals.append(signal)
                    self.logger.info(f"{strategy.name}: {signal.signal.value} "
                                   f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
            except Exception as e:
                self.logger.error(f"Error in {strategy.name}: {e}")
        
        if not signals:
            return None
        
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        # Require at least 2 strategies to agree with high confidence
        if len(buy_signals) >= self.min_agreeing_strategies and len(sell_signals) == 0:
            best_buy = max(buy_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            
            if avg_confidence >= self.min_ensemble_confidence:
                return TradingSignal(
                    SignalType.BUY, avg_confidence, best_buy.entry_price,
                    best_buy.stop_loss, best_buy.take_profit,
                    f"Enhanced BUY ({len(buy_signals)} strategies)", best_buy.timestamp
                )
        
        elif len(sell_signals) >= self.min_agreeing_strategies and len(buy_signals) == 0:
            best_sell = max(sell_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            
            if avg_confidence >= self.min_ensemble_confidence:
                return TradingSignal(
                    SignalType.SELL, avg_confidence, best_sell.entry_price,
                    best_sell.stop_loss, best_sell.take_profit,
                    f"Enhanced SELL ({len(sell_signals)} strategies)", best_sell.timestamp
                )
        
        return None