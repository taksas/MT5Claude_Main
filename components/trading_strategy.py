#!/usr/bin/env python3
"""
Ultra Trading Strategy Module - Clean Version
Contains core trading signal generation without bloated pseudo-science
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from collections import deque

from .trading_components import Signal, SignalType
from .indicators import QuantumUltraIntelligentIndicators, quantum_indicators
from .trading_components import SymbolUtils
from .trading_components import CONFIG, get_symbol_config

logger = logging.getLogger('TradingStrategy')

class TradingStrategy:
    """Clean trading strategy focused on actual trading logic"""
    
    def __init__(self):
        """Initialize trading strategy with essential components"""
        self.indicators = quantum_indicators  # Use global indicator instance
        self.symbol_utils = SymbolUtils()
        
        # Adjusted weights for more realistic evaluation
        self.category_weights = {
            'regime': 2.5,
            'price_action': 2.0,
            'patterns': 1.5,
            'mathematical': 1.8,
            'microstructure': 1.5,
            'sentiment': 1.0,
            'risk': 0.5,
            'composite': 2.0
        }
        
    def analyze_ultra(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Analyze market with intelligent indicators and generate trading signal"""
        try:
            # Step 1: Calculate all indicators from the indicator module
            analysis = self.indicators.calculate_ultra_indicators(df, current_price)
            if 'error' in analysis:
                logger.error(f"Indicator calculation failed for {symbol}: {analysis['error']}")
                return None

            # Step 2: Evaluate signal strength by interpreting the raw indicator data
            scores = {}
            scores['regime'] = self._evaluate_regime(analysis)
            scores['price_action'] = self._evaluate_price_action(analysis)
            scores['patterns'] = self._evaluate_patterns(analysis)
            scores['mathematical'] = self._evaluate_mathematical(analysis)
            scores['microstructure'] = self._evaluate_microstructure(analysis)
            scores['sentiment'] = self._evaluate_sentiment(analysis)
            scores['risk'] = self._evaluate_risk(analysis)
            scores['composite'] = analysis.get('composite_signal', 0.0)

            # Step 3: Calculate weighted final score
            weighted_score = sum(
                scores.get(category, 0.0) * weight 
                for category, weight in self.category_weights.items()
            )
            
            total_weight = sum(self.category_weights.values())
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Step 4: Determine signal direction and confidence
            # Lowered the initial threshold to allow more signals to be considered
            if abs(final_score) < CONFIG.get("MIN_CONFIDENCE", 0.3):
                return None  # No clear signal
                
            signal_type = SignalType.BUY if final_score > 0 else SignalType.SELL
            confidence = min(abs(final_score), 1.0)
            
            # Main confidence check
            if confidence < 0.80:
                return None
                
            # Step 5: Calculate Stop Loss and Take Profit
            symbol_base = symbol.rstrip('#')
            symbol_config = get_symbol_config(symbol_base)
            
            atr = analysis.get('traditional', {}).get('atr', 0)
            if atr <= 0:
                logger.warning(f"ATR for {symbol} is zero or invalid, falling back to percentage-based SL.")
                sl_distance = current_price * CONFIG["MIN_SL_DISTANCE_PERCENT"] * 2.5
            else:
                atr_multiplier = 1.5
                sl_distance = atr * atr_multiplier

            # Validate SL distance for safety
            min_sl_dist = current_price * CONFIG["MIN_SL_DISTANCE_PERCENT"]
            max_sl_dist = current_price * CONFIG["MAX_SL_DISTANCE_PERCENT"]
            
            if sl_distance < min_sl_dist:
                sl_distance = min_sl_dist
            elif sl_distance > max_sl_dist:
                sl_distance = max_sl_dist

            target_rr = symbol_config.get('target_rr_ratio', 2.0)
            tp_distance = sl_distance * target_rr
            
            if signal_type == SignalType.BUY:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                reason = f"BUY Signal: Score {final_score:.2f}"
            else: # SELL
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                reason = f"SELL Signal: Score {final_score:.2f}"
            
            # Quality score based on confidence and other factors
            quality = confidence * 0.8 + analysis.get('quantum_state', {}).get('coherence', 0.5) * 0.2
            
            return Signal(
                symbol=symbol,
                type=signal_type,
                confidence=confidence,
                entry=current_price,
                sl=sl,
                tp=tp,
                reason=reason,
                strategies={ # Pass the calculated scores for visualization
                    'final_score': final_score,
                    'confidence': confidence,
                    **scores
                },
                quality=quality
            )
            
        except Exception as e:
            logger.error(f"Critical error in analysis for {symbol}: {e}", exc_info=True)
            return None
    
    def force_trade_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Generate forced trade signal when no trades are happening"""
        try:
            # Use a simple momentum calculation for forcing a trade
            momentum = df['close'].pct_change(5).iloc[-1]
            if pd.isna(momentum) or abs(momentum) < 0.0005:
                return None # Not enough momentum
            
            signal_type = SignalType.BUY if momentum > 0 else SignalType.SELL
            
            symbol_base = symbol.rstrip('#')
            symbol_config = get_symbol_config(symbol_base)
            
            analysis = self.indicators.calculate_ultra_indicators(df, current_price)
            atr = analysis.get('traditional', {}).get('atr', 0)
            
            if atr > 0:
                sl_distance = atr * 2.0 # Wider SL for forced trades
            else:
                sl_distance = current_price * CONFIG["MIN_SL_DISTANCE_PERCENT"] * 3.0

            min_sl_dist = current_price * CONFIG["MIN_SL_DISTANCE_PERCENT"]
            max_sl_dist = current_price * CONFIG["MAX_SL_DISTANCE_PERCENT"]
            sl_distance = np.clip(sl_distance, min_sl_dist, max_sl_dist)

            min_rr = symbol_config.get('min_rr_ratio', 1.5)
            tp_distance = sl_distance * min_rr
            
            if signal_type == SignalType.BUY:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                reason = f"FORCED BUY: Momentum {momentum:.3%}"
            else: # SELL
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                reason = f"FORCED SELL: Momentum {momentum:.3%}"
            
            return Signal(
                symbol=symbol,
                type=signal_type,
                confidence=0.15,
                entry=current_price,
                sl=sl,
                tp=tp,
                reason=reason,
                strategies={'forced': True, 'momentum': momentum},
                quality=0.1
            )
            
        except Exception as e:
            logger.error(f"Error in forced signal for {symbol}: {e}")
            return None

    # --- REWRITTEN EVALUATION HELPER FUNCTIONS ---

    def _evaluate_regime(self, analysis: Dict[str, Any]) -> float:
        """Evaluate market regime."""
        regime = analysis.get('regime', {})
        score = 0
        trend = regime.get('trend')
        momentum = regime.get('momentum')
        
        if trend == 'bullish':
            score += 0.5
        elif trend == 'bearish':
            score -= 0.5

        if momentum == 'strong_up':
            score += 0.5
        elif momentum == 'weak_up':
            score += 0.2
        elif momentum == 'strong_down':
            score -= 0.5
        elif momentum == 'weak_down':
            score -= 0.2
            
        return np.clip(score, -1.0, 1.0)

    def _evaluate_price_action(self, analysis: Dict[str, Any]) -> float:
        """Evaluate price action signals from the 'price_action' key."""
        pa_signals = analysis.get('price_action', {})
        score = 0
        if pa_signals.get('engulfing_bull', 0) > 0:
            score += 1.0
        if pa_signals.get('engulfing_bear', 0) > 0:
            score -= 1.0
        return np.clip(score, -1.0, 1.0)

    def _evaluate_patterns(self, analysis: Dict[str, Any]) -> float:
        """Evaluate chart patterns from the 'patterns' key."""
        patterns = analysis.get('patterns', [])
        if not patterns:
            return 0.0
        
        score = 0
        for p in patterns:
            confidence = p.get('confidence', 0)
            if p.get('direction') == 'bullish':
                score += confidence
            elif p.get('direction') == 'bearish':
                score -= confidence
                
        return np.clip(score, -1.0, 1.0)

    def _evaluate_mathematical(self, analysis: Dict[str, Any]) -> float:
        """Evaluate mathematical indicators from the 'traditional' key."""
        indicators = analysis.get('traditional', {})
        score = 0.0
        
        ma_fast = indicators.get('ma_fast')
        ma_slow = indicators.get('ma_slow')
        rsi = indicators.get('rsi')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')

        if ma_fast is not None and ma_slow is not None:
            if ma_fast > ma_slow: score += 0.4
            elif ma_fast < ma_slow: score -= 0.4

        if rsi is not None:
            if rsi < 30: score += 0.3
            elif rsi > 70: score -= 0.3

        if macd is not None and macd_signal is not None:
            if macd > macd_signal: score += 0.3
            elif macd < macd_signal: score -= 0.3
            
        return np.clip(score, -1.0, 1.0)

    def _evaluate_microstructure(self, analysis: Dict[str, Any]) -> float:
        """Evaluate microstructure signals."""
        micro = analysis.get('microstructure', {})
        ofi = micro.get('order_flow_imbalance', 0)
        # OFI > 0 suggests buying pressure, OFI < 0 suggests selling pressure
        return np.clip(ofi * 2, -1.0, 1.0) # Amplify score slightly

    def _evaluate_sentiment(self, analysis: Dict[str, Any]) -> float:
        """Evaluate sentiment score."""
        # Sentiment score is already a float between -1 and 1
        return analysis.get('sentiment', 0.0)

    def _evaluate_risk(self, analysis: Dict[str, Any]) -> float:
        """Evaluate risk metrics. High risk should negatively impact the score."""
        risk = analysis.get('risk_metrics', {})
        risk_score = risk.get('overall_risk_score', 0.5)
        # Invert score: 1 (low risk) -> positive signal, 0 (high risk) -> negative signal
        return (0.5 - risk_score)