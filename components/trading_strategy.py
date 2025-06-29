#!/usr/bin/env python3
"""
Trading Strategy Module
Contains signal generation logic and strategy evaluation
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from .trading_models import Signal, SignalType
from .indicators import QuantumUltraIntelligentIndicators, quantum_indicators
from .symbol_utils import SymbolUtils
from .trading_config import CONFIG, get_symbol_config

logger = logging.getLogger('TradingStrategy')

class TradingStrategy:
    def __init__(self):
        self.indicators = quantum_indicators  # Use global quantum instance
        self.symbol_utils = SymbolUtils()
        
        # Category weights for signal generation
        self.category_weights = {
            'price_action': 2.0,
            'chart_patterns': 1.5,
            'mathematical': 1.2,
            'volatility': 1.0,
            'market_structure': 2.0,
            'momentum': 1.8,
            'volume': 1.3,
            'time_based': 0.8,
            'statistical': 1.0,
            'composite': 2.5
        }
        
    def analyze_ultra(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Analyze market with ultra-intelligent indicators for high-precision signals"""
        try:
            # Calculate ultra-intelligent indicators
            analysis = self.indicators.calculate_ultra_indicators(df, current_price)
            
            # Extract key components
            regime = analysis['regime']
            composite_signal = analysis['composite_signal']
            patterns = analysis.get('patterns', [])
            sentiment = analysis.get('sentiment', {})
            predictions = analysis.get('predictions', {})
            mtf = analysis.get('multi_timeframe', {})
            stats = analysis.get('statistics', {})
            confidence_score = analysis.get('confidence', 0.5)
            
            # Check if we have a valid signal from the neural network fusion
            if composite_signal['signal'] in ['strong_buy', 'buy', 'strong_sell', 'sell']:
                # Validate signal with additional checks
                
                # 1. Regime alignment check
                regime_aligned = False
                if composite_signal['signal'] in ['strong_buy', 'buy'] and regime['trend'] == 'bullish':
                    regime_aligned = True
                elif composite_signal['signal'] in ['strong_sell', 'sell'] and regime['trend'] == 'bearish':
                    regime_aligned = True
                
                # 2. Multi-timeframe confluence check
                mtf_aligned = mtf.get('aligned', False)
                
                # 3. Pattern confirmation
                pattern_confirmed = False
                if patterns:
                    bullish_patterns = [p for p in patterns if p['direction'] == 'bullish']
                    bearish_patterns = [p for p in patterns if p['direction'] == 'bearish']
                    
                    if composite_signal['signal'] in ['strong_buy', 'buy'] and len(bullish_patterns) > len(bearish_patterns):
                        pattern_confirmed = True
                    elif composite_signal['signal'] in ['strong_sell', 'sell'] and len(bearish_patterns) > len(bullish_patterns):
                        pattern_confirmed = True
                
                # 4. Market conditions check
                suitable_volatility = regime['volatility'] in ['medium', 'high']  # Avoid low volatility
                suitable_momentum = regime['momentum'] not in ['neutral']  # Avoid neutral momentum
                
                # Calculate final validation score
                validation_score = 0
                if regime_aligned: validation_score += 0.3
                if mtf_aligned: validation_score += 0.2
                if pattern_confirmed: validation_score += 0.2
                if suitable_volatility: validation_score += 0.15
                if suitable_momentum: validation_score += 0.15
                
                # Combine neural confidence with validation
                final_confidence = composite_signal['confidence'] * 0.7 + validation_score * 0.3
                
                # Enhanced reason building
                reasons = []
                if regime_aligned:
                    reasons.append(f"{regime['trend'].upper()} regime")
                if mtf_aligned:
                    reasons.append("MTF confluence")
                if pattern_confirmed:
                    top_pattern = patterns[0]['pattern_type'] if patterns else ""
                    reasons.append(f"{top_pattern} pattern")
                if predictions.get('ml_signal') == composite_signal['signal']:
                    reasons.append("ML prediction aligned")
                
                # Feature importance
                top_feature = max(composite_signal['feature_importance'].items(), key=lambda x: x[1])[0]
                reasons.append(f"{top_feature} dominant")
                
                # Determine signal type and check confidence
                signal_type = None
                if composite_signal['signal'] in ['strong_buy', 'buy']:
                    signal_type = SignalType.BUY
                elif composite_signal['signal'] in ['strong_sell', 'sell']:
                    signal_type = SignalType.SELL
                
                # Check minimum confidence
                if signal_type and final_confidence >= CONFIG["MIN_CONFIDENCE"]:
                    # Calculate stop loss and take profit
                    symbol_config = get_symbol_config(symbol)
                    instrument_type = self.symbol_utils.get_instrument_type(symbol)
                    
                    # Try to get ATR from traditional indicators
                    traditional = analysis.get('traditional', {})
                    atr = traditional.get('adaptive_atr', current_price * 0.001)
                    
                    # Check if patterns provide targets
                    pattern_sl = None
                    pattern_tp = None
                    if patterns:
                        relevant_patterns = [p for p in patterns if 
                                           (signal_type == SignalType.BUY and p['direction'] == 'bullish') or
                                           (signal_type == SignalType.SELL and p['direction'] == 'bearish')]
                        if relevant_patterns:
                            # Use the most confident pattern's targets
                            best_pattern = relevant_patterns[0]
                            pattern_sl = best_pattern['stop_loss']
                            pattern_tp = best_pattern['target_price']
                    
                    # Adjust for instrument type
                    if instrument_type == 'exotic':
                        sl_distance = atr * 2.5
                        tp_distance = atr * 5.0
                    elif instrument_type == 'metal':
                        sl_distance = atr * 2.0
                        tp_distance = atr * 4.0
                    elif instrument_type == 'crypto':
                        sl_distance = atr * 3.0
                        tp_distance = atr * 6.0
                    elif instrument_type == 'index':
                        sl_distance = atr * 1.5
                        tp_distance = atr * 3.0
                    else:  # Major pairs
                        sl_distance = atr * 2.0
                        tp_distance = atr * 3.0
                    
                    # Calculate SL/TP with pattern priority
                    if signal_type == SignalType.BUY:
                        sl = pattern_sl if pattern_sl else current_price - sl_distance
                        tp = pattern_tp if pattern_tp else current_price + tp_distance
                        reason = f"ULTRA BUY: {', '.join(reasons[:3])}"
                    else:
                        sl = pattern_sl if pattern_sl else current_price + sl_distance
                        tp = pattern_tp if pattern_tp else current_price - tp_distance
                        reason = f"ULTRA SELL: {', '.join(reasons[:3])}"
                    
                    # Calculate ultra quality score
                    quality = self._calculate_ultra_signal_quality(
                        final_confidence,
                        regime,
                        patterns,
                        composite_signal
                    )
                    
                    return Signal(
                        type=signal_type,
                        confidence=final_confidence,
                        entry=current_price,
                        sl=sl,
                        tp=tp,
                        reason=reason,
                        strategies={
                            'regime': regime,
                            'composite': composite_signal,
                            'patterns': len(patterns),
                            'mtf_aligned': mtf_aligned
                        },
                        quality=quality
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ultra analysis for {symbol}: {e}")
            return None
    
    
    def _evaluate_price_action(self, indicators: Dict[str, Any]) -> float:
        """Evaluate price action indicators"""
        score = 0
        
        # Bullish patterns
        if indicators.get('pin_bar_bull', 0) > 0:
            score += 2
        if indicators.get('engulfing_bull', 0) > 0:
            score += 2
        if indicators.get('hammer', 0) > 0:
            score += 1.5
        if indicators.get('three_white_soldiers', 0) > 0:
            score += 2
        
        # Bearish patterns
        if indicators.get('pin_bar_bear', 0) > 0:
            score -= 2
        if indicators.get('engulfing_bear', 0) > 0:
            score -= 2
        if indicators.get('hanging_man', 0) > 0:
            score -= 1.5
        if indicators.get('three_black_crows', 0) > 0:
            score -= 2
        
        # Neutral patterns
        if indicators.get('doji', 0) > 0:
            score *= 0.5  # Reduce confidence on doji
        
        # Momentum
        pa_momentum = indicators.get('pa_momentum', 0)
        if abs(pa_momentum) > 0.01:  # 1% move
            score += 1 if pa_momentum > 0 else -1
        
        return score
    
    def _evaluate_chart_patterns(self, indicators: Dict[str, Any]) -> float:
        """Evaluate chart pattern indicators"""
        score = 0
        
        if indicators.get('double_bottom', 0) > 0:
            score += 2
        if indicators.get('double_top', 0) > 0:
            score -= 2
        if indicators.get('falling_wedge', 0) > 0:
            score += 1.5
        if indicators.get('rising_wedge', 0) > 0:
            score -= 1.5
        if indicators.get('channel_lower', 0) > 0:
            score += 1
        if indicators.get('channel_upper', 0) > 0:
            score -= 1
        
        return score
    
    def _evaluate_mathematical(self, indicators: Dict[str, Any]) -> float:
        """Evaluate mathematical indicators"""
        score = 0
        
        # Fibonacci levels
        if indicators.get('fib_618', 0) > 0 or indicators.get('fib_500', 0) > 0:
            score += 1.5
        if indicators.get('fib_382', 0) > 0 or indicators.get('fib_236', 0) > 0:
            score -= 1.5
        
        # Pivot points
        if indicators.get('pivot_s1', 0) > 0:
            score += 1
        if indicators.get('pivot_r1', 0) > 0:
            score -= 1
        
        # Linear regression
        slope = indicators.get('lin_reg_slope', 0)
        if slope > 0:
            score += min(slope * 100, 2)
        else:
            score -= min(abs(slope) * 100, 2)
        
        return score
    
    def _evaluate_volatility(self, indicators: Dict[str, Any]) -> float:
        """Evaluate volatility indicators"""
        score = 0
        
        if indicators.get('bb_squeeze', 0) > 0:
            score += 1.5  # Breakout potential
        if indicators.get('keltner_lower', 0) > 0:
            score += 1
        if indicators.get('keltner_upper', 0) > 0:
            score -= 1
        if indicators.get('donchian_high', 0) > 0:
            score += 0.5  # Trend continuation
        if indicators.get('donchian_low', 0) > 0:
            score -= 0.5
        
        # Volatility ratio
        vol_ratio = indicators.get('vol_ratio', 1)
        if vol_ratio > 1.5:
            score += 1  # Increasing volatility
        
        return score
    
    def _evaluate_market_structure(self, indicators: Dict[str, Any]) -> float:
        """Evaluate market structure indicators"""
        score = 0
        
        if indicators.get('structure_break_up', 0) > 0:
            score += 3
        if indicators.get('structure_break_down', 0) > 0:
            score -= 3
        if indicators.get('higher_highs', 0) > 0.6:
            score += 2
        if indicators.get('lower_lows', 0) > 0.6:
            score -= 2
        if indicators.get('near_support', 0) > 0:
            score += 1
        if indicators.get('near_resistance', 0) > 0:
            score -= 1
        
        return score
    
    def _evaluate_momentum(self, indicators: Dict[str, Any]) -> float:
        """Evaluate momentum indicators"""
        score = 0
        
        # ROC
        for period in [5, 10, 20]:
            roc = indicators.get(f'roc_{period}', 0)
            if roc > 0.01:
                score += 0.5
            elif roc < -0.01:
                score -= 0.5
        
        # CCI
        cci = indicators.get('cci', 0)
        if cci < -100:
            score += 1  # Oversold
        elif cci > 100:
            score -= 1  # Overbought
        
        # Williams %R
        williams = indicators.get('williams_r', -50)
        if williams < -80:
            score += 1  # Oversold
        elif williams > -20:
            score -= 1  # Overbought
        
        return score
    
    def _evaluate_volume(self, indicators: Dict[str, Any]) -> float:
        """Evaluate volume indicators"""
        score = 0
        
        if indicators.get('volume_ratio', 0) > 1.5:
            score += 1  # High volume
        if indicators.get('obv_trend', 0) > 0:
            score += 1
        elif indicators.get('obv_trend', 0) < 0:
            score -= 1
        if indicators.get('chaikin_mf', 0) > 0.1:
            score += 1
        elif indicators.get('chaikin_mf', 0) < -0.1:
            score -= 1
        
        # MFI
        mfi = indicators.get('mfi', 50)
        if mfi < 20:
            score += 1  # Oversold
        elif mfi > 80:
            score -= 1  # Overbought
        
        return score
    
    def _evaluate_time_patterns(self, indicators: Dict[str, Any]) -> float:
        """Evaluate time-based patterns"""
        score = 0
        
        # Session overlap is usually high volatility
        if indicators.get('session_overlap', 0) > 0:
            score += 0.5
        
        # Avoid Monday/Friday
        if indicators.get('monday', 0) > 0 or indicators.get('friday', 0) > 0:
            score *= 0.8
        
        # Momentum
        if indicators.get('hourly_momentum', 0) > 0.005:
            score += 0.5
        elif indicators.get('hourly_momentum', 0) < -0.005:
            score -= 0.5
        
        return score
    
    def _evaluate_statistical(self, indicators: Dict[str, Any]) -> float:
        """Evaluate statistical indicators"""
        score = 0
        
        z_score = indicators.get('z_score', 0)
        if z_score < -2:
            score += 1.5  # Oversold
        elif z_score > 2:
            score -= 1.5  # Overbought
        
        # Mean reversion
        mean_rev = indicators.get('mean_reversion', 0)
        if mean_rev > 0.02:
            score += 1
        elif mean_rev < -0.02:
            score -= 1
        
        # Efficiency ratio - trend strength
        eff_ratio = indicators.get('efficiency_ratio', 0)
        if eff_ratio > 0.7:
            score *= 1.2  # Strong trend
        
        return score
    
    def _evaluate_composite(self, indicators: Dict[str, Any]) -> float:
        """Evaluate composite indicators"""
        score = 0
        
        # MACD
        if indicators.get('macd_histogram', 0) > 0:
            score += 1.5
        else:
            score -= 1.5
        
        # RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        
        # Stochastic
        if indicators.get('stoch_k', 50) < 20 and indicators.get('stoch_d', 50) < 20:
            score += 1.5
        elif indicators.get('stoch_k', 50) > 80 and indicators.get('stoch_d', 50) > 80:
            score -= 1.5
        
        # ADX - trend strength
        adx = indicators.get('adx', 0)
        if adx > 25:
            # Strong trend
            if indicators.get('plus_di', 0) > indicators.get('minus_di', 0):
                score += 2
            else:
                score -= 2
        
        # Ichimoku
        if indicators.get('above_cloud', 0) > 0:
            score += 2
        elif indicators.get('below_cloud', 0) > 0:
            score -= 2
        
        return score
    
    def _calculate_ultra_signal_quality(self, confidence: float, regime: Dict[str, Any],
                                      patterns: list, composite_signal: Dict[str, Any]) -> float:
        """Calculate ultra signal quality using intelligent metrics"""
        quality = confidence
        
        # Regime quality multiplier
        if regime['trend'] != 'ranging':
            quality *= 1.1
        if regime['volatility'] == 'medium':
            quality *= 1.05
        elif regime['volatility'] == 'high':
            quality *= 0.95
        
        # Pattern confirmation bonus
        if patterns:
            avg_pattern_conf = sum(p['confidence'] for p in patterns) / len(patterns)
            quality *= (0.9 + avg_pattern_conf * 0.2)
        
        # Composite signal strength
        signal_strength = composite_signal.get('strength', 0.5)
        if signal_strength > 0.8 or signal_strength < 0.2:
            quality *= 1.15  # Strong signals
        elif 0.3 < signal_strength < 0.7:
            quality *= 0.9   # Weak signals
        
        # Regime confidence boost
        quality *= (0.8 + regime['confidence'] * 0.2)
        
        return min(quality, 1.0)
    
    def _calculate_signal_quality(self, confidence: float, total_indicators: int, 
                                indicators: Dict[str, Any], instrument_type: str) -> float:
        """Calculate overall signal quality (legacy method for compatibility)"""
        quality = confidence
        
        # Adjust for number of confirming indicators
        if total_indicators >= 20:
            quality *= 1.2
        elif total_indicators >= 15:
            quality *= 1.1
        elif total_indicators < 10:
            quality *= 0.8
        
        # Adjust for instrument type
        if instrument_type == 'major':
            quality *= 1.1  # Major pairs are more reliable
        elif instrument_type == 'exotic':
            quality *= 0.9  # Exotic pairs are less reliable
        
        # Adjust for market conditions
        adx = indicators.get('adx', 0)
        if adx > 30:
            quality *= 1.1  # Strong trend
        elif adx < 20:
            quality *= 0.9  # Weak trend
        
        return min(quality, 1.0)
    
    def _get_pip_value(self, symbol: str, price: float) -> float:
        """Get pip value for a symbol"""
        if self.symbol_utils.is_jpy_pair(symbol):
            return 0.01
        elif self.symbol_utils.is_metal_pair(symbol):
            if 'XAU' in symbol:
                return 0.1
            else:
                return 0.01
        elif self.symbol_utils.is_crypto_pair(symbol):
            return 1.0  # Crypto typically quoted in whole units
        elif self.symbol_utils.is_index_pair(symbol):
            return 1.0  # Indices typically in points
        else:
            return 0.0001  # Standard forex