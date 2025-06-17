import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
# import talib  # Will implement manual calculations
import logging

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    signal: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: str

class TradingStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        pass
    
    def calculate_risk_levels(self, entry_price: float, signal_type: SignalType, 
                            atr: float, pip_value: float = 0.0001) -> Tuple[float, float]:
        risk_ratio = 2.0
        stop_distance = atr * 1.5
        
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * risk_ratio)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * risk_ratio)
        
        return stop_loss, take_profit

class MomentumBreakoutStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MomentumBreakout")
        self.lookback_period = 20
        self.volume_threshold = 1.2
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < self.lookback_period + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        df['high_max'] = df['high'].rolling(window=self.lookback_period).max()
        df['low_min'] = df['low'].rolling(window=self.lookback_period).min()
        df['volume_ma'] = df['tick_volume'].rolling(window=self.lookback_period).mean()
        
        current_price = df['close'].iloc[-1]
        high_breakout = current_price > df['high_max'].iloc[-2]
        low_breakout = current_price < df['low_min'].iloc[-2]
        volume_spike = df['tick_volume'].iloc[-1] > df['volume_ma'].iloc[-1] * self.volume_threshold
        
        
        # Calculate ATR manually
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        if high_breakout and volume_spike:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.9, 0.6 + (df['tick_volume'].iloc[-1] / df['volume_ma'].iloc[-1] - 1) * 0.3)
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Upward breakout with volume spike", df['time'].iloc[-1]
            )
        
        elif low_breakout and volume_spike:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.9, 0.6 + (df['tick_volume'].iloc[-1] / df['volume_ma'].iloc[-1] - 1) * 0.3)
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Downward breakout with volume spike", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, "No breakout signal", df['time'].iloc[-1])

class MACDStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MACD")
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < max(self.slow_period, self.signal_period) + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate MACD manually
        ema_fast = df['close'].ewm(span=self.fast_period).mean()
        ema_slow = df['close'].ewm(span=self.slow_period).mean()
        macd_line = (ema_fast - ema_slow).values
        macd_signal_series = pd.Series(macd_line).ewm(span=self.signal_period).mean()
        macd_signal = macd_signal_series.values
        
        current_price = df['close'].iloc[-1]
        current_macd = macd_line[-1]
        current_signal = macd_signal[-1]
        prev_macd = macd_line[-2]
        prev_signal = macd_signal[-2]
        
        
        # Calculate ATR manually
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
        bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal
        
        if bullish_crossover and current_macd < 0:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.85, 0.5 + abs(current_macd - current_signal) * 1000)
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"MACD bullish crossover below zero", df['time'].iloc[-1]
            )
        
        elif bearish_crossover and current_macd > 0:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.85, 0.5 + abs(current_macd - current_signal) * 1000)
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"MACD bearish crossover above zero", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, "No MACD signal", df['time'].iloc[-1])

class RSIStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("RSI")
        self.period = 14
        self.oversold_level = 30
        self.overbought_level = 70
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < self.period + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate RSI manually
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        rs = avg_gain / avg_loss
        rsi = (100 - (100 / (1 + rs))).values
        current_rsi = rsi[-1]
        prev_rsi = rsi[-2]
        current_price = df['close'].iloc[-1]
        
        
        # Calculate ATR manually
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        if prev_rsi <= self.oversold_level and current_rsi > self.oversold_level:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.8, (self.oversold_level - prev_rsi + 5) / 20)
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"RSI recovery from oversold ({current_rsi:.1f})", df['time'].iloc[-1]
            )
        
        elif prev_rsi >= self.overbought_level and current_rsi < self.overbought_level:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.8, (prev_rsi - self.overbought_level + 5) / 20)
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"RSI decline from overbought ({current_rsi:.1f})", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           f"RSI neutral ({current_rsi:.1f})", df['time'].iloc[-1])

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("BollingerBands")
        self.period = 20
        self.std_dev = 2
    
    def analyze(self, data: List[Dict[str, Any]]) -> TradingSignal:
        df = pd.DataFrame(data)
        
        if len(df) < self.period + 10:
            return TradingSignal(SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data", "")
        
        # Calculate Bollinger Bands manually
        middle_band = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        
        # Calculate ATR manually
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        touched_lower = prev_price <= lower_band.iloc[-2] and current_price > lower_band.iloc[-1]
        touched_upper = prev_price >= upper_band.iloc[-2] and current_price < upper_band.iloc[-1]
        
        bb_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / middle_band.iloc[-1]
        
        if touched_lower and bb_width > 0.02:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.BUY, atr)
            confidence = min(0.8, 0.5 + bb_width * 10)
            return TradingSignal(
                SignalType.BUY, confidence, current_price, sl, tp,
                f"Bounce from lower Bollinger Band", df['time'].iloc[-1]
            )
        
        elif touched_upper and bb_width > 0.02:
            sl, tp = self.calculate_risk_levels(current_price, SignalType.SELL, atr)
            confidence = min(0.8, 0.5 + bb_width * 10)
            return TradingSignal(
                SignalType.SELL, confidence, current_price, sl, tp,
                f"Rejection from upper Bollinger Band", df['time'].iloc[-1]
            )
        
        return TradingSignal(SignalType.HOLD, 0.0, current_price, 0.0, 0.0, 
                           "No Bollinger Band signal", df['time'].iloc[-1])

class StrategyEnsemble:
    def __init__(self):
        self.strategies = [
            MomentumBreakoutStrategy(),
            MACDStrategy(),
            RSIStrategy(),
            BollingerBandsStrategy()
        ]
        self.logger = logging.getLogger(__name__)
    
    def get_ensemble_signal(self, data: List[Dict[str, Any]]) -> Optional[TradingSignal]:
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(data)
                if signal.signal != SignalType.HOLD:
                    signals.append(signal)
                    self.logger.info(f"{strategy.name}: {signal.signal.value} "
                                   f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
            except Exception as e:
                self.logger.error(f"Error in {strategy.name}: {e}")
        
        if not signals:
            return None
        
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        if len(buy_signals) >= 2 and len(sell_signals) == 0:
            best_buy = max(buy_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            return TradingSignal(
                SignalType.BUY, avg_confidence, best_buy.entry_price,
                best_buy.stop_loss, best_buy.take_profit,
                f"Ensemble BUY ({len(buy_signals)} strategies agree)", best_buy.timestamp
            )
        
        elif len(sell_signals) >= 2 and len(buy_signals) == 0:
            best_sell = max(sell_signals, key=lambda x: x.confidence)
            avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            return TradingSignal(
                SignalType.SELL, avg_confidence, best_sell.entry_price,
                best_sell.stop_loss, best_sell.take_profit,
                f"Ensemble SELL ({len(sell_signals)} strategies agree)", best_sell.timestamp
            )
        
        return None