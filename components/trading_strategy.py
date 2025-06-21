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
from .indicators import TechnicalIndicators
from .symbol_utils import SymbolUtils
from .trading_config import CONFIG, get_symbol_config

logger = logging.getLogger('TradingStrategy')

class TradingStrategy:
    def __init__(self):
        self.indicators = TechnicalIndicators()
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
        """Analyze market with 100 indicators for ultra-precise signals"""
        try:
            # Calculate all indicators
            indicators = self.indicators.calculate_all_indicators(df, current_price)
            
            # Initialize scores
            buy_score = 0
            sell_score = 0
            positive_indicators = []
            negative_indicators = []
            
            # Price Action Analysis
            price_action_score = self._evaluate_price_action(indicators)
            if price_action_score > 0:
                buy_score += price_action_score * self.category_weights['price_action']
                positive_indicators.extend(['pin_bar_bull', 'engulfing_bull', 'hammer'])
            elif price_action_score < 0:
                sell_score += abs(price_action_score) * self.category_weights['price_action']
                negative_indicators.extend(['pin_bar_bear', 'engulfing_bear', 'hanging_man'])
            
            # Chart Patterns
            chart_score = self._evaluate_chart_patterns(indicators)
            if chart_score > 0:
                buy_score += chart_score * self.category_weights['chart_patterns']
                positive_indicators.extend(['double_bottom', 'falling_wedge'])
            elif chart_score < 0:
                sell_score += abs(chart_score) * self.category_weights['chart_patterns']
                negative_indicators.extend(['double_top', 'rising_wedge'])
            
            # Mathematical Indicators
            math_score = self._evaluate_mathematical(indicators)
            if math_score > 0:
                buy_score += math_score * self.category_weights['mathematical']
                positive_indicators.extend(['fib_support', 'pivot_support'])
            elif math_score < 0:
                sell_score += abs(math_score) * self.category_weights['mathematical']
                negative_indicators.extend(['fib_resistance', 'pivot_resistance'])
            
            # Volatility Analysis
            volatility_score = self._evaluate_volatility(indicators)
            if volatility_score > 0:
                buy_score += volatility_score * self.category_weights['volatility']
                positive_indicators.append('volatility_expansion')
            elif volatility_score < 0:
                sell_score += abs(volatility_score) * self.category_weights['volatility']
                negative_indicators.append('volatility_contraction')
            
            # Market Structure
            structure_score = self._evaluate_market_structure(indicators)
            if structure_score > 0:
                buy_score += structure_score * self.category_weights['market_structure']
                positive_indicators.extend(['structure_break_up', 'higher_highs'])
            elif structure_score < 0:
                sell_score += abs(structure_score) * self.category_weights['market_structure']
                negative_indicators.extend(['structure_break_down', 'lower_lows'])
            
            # Momentum Analysis
            momentum_score = self._evaluate_momentum(indicators)
            if momentum_score > 0:
                buy_score += momentum_score * self.category_weights['momentum']
                positive_indicators.extend(['positive_momentum', 'oversold_bounce'])
            elif momentum_score < 0:
                sell_score += abs(momentum_score) * self.category_weights['momentum']
                negative_indicators.extend(['negative_momentum', 'overbought_reversal'])
            
            # Volume Analysis
            volume_score = self._evaluate_volume(indicators)
            if volume_score > 0:
                buy_score += volume_score * self.category_weights['volume']
                positive_indicators.append('accumulation')
            elif volume_score < 0:
                sell_score += abs(volume_score) * self.category_weights['volume']
                negative_indicators.append('distribution')
            
            # Time-Based Patterns
            time_score = self._evaluate_time_patterns(indicators)
            if time_score > 0:
                buy_score += time_score * self.category_weights['time_based']
                positive_indicators.append('optimal_time')
            elif time_score < 0:
                sell_score += abs(time_score) * self.category_weights['time_based']
                negative_indicators.append('adverse_time')
            
            # Statistical Analysis
            stat_score = self._evaluate_statistical(indicators)
            if stat_score > 0:
                buy_score += stat_score * self.category_weights['statistical']
                positive_indicators.append('mean_reversion_up')
            elif stat_score < 0:
                sell_score += abs(stat_score) * self.category_weights['statistical']
                negative_indicators.append('mean_reversion_down')
            
            # Composite Indicators
            composite_score = self._evaluate_composite(indicators)
            if composite_score > 0:
                buy_score += composite_score * self.category_weights['composite']
                positive_indicators.extend(['macd_bull', 'rsi_oversold', 'ichimoku_bull'])
            elif composite_score < 0:
                sell_score += abs(composite_score) * self.category_weights['composite']
                negative_indicators.extend(['macd_bear', 'rsi_overbought', 'ichimoku_bear'])
            
            # Calculate total scores
            total_buy = buy_score
            total_sell = sell_score
            total_indicators = len(positive_indicators) + len(negative_indicators)
            
            # Determine signal
            signal_type = None
            confidence = 0
            
            if total_buy > total_sell and len(positive_indicators) >= CONFIG["MIN_INDICATORS"]:
                signal_type = SignalType.BUY
                confidence = min(total_buy / (total_buy + total_sell) if (total_buy + total_sell) > 0 else 0, 0.95)
            elif total_sell > total_buy and len(negative_indicators) >= CONFIG["MIN_INDICATORS"]:
                signal_type = SignalType.SELL
                confidence = min(total_sell / (total_buy + total_sell) if (total_buy + total_sell) > 0 else 0, 0.95)
            
            # Check minimum confidence
            if signal_type and confidence >= CONFIG["MIN_CONFIDENCE"]:
                # Calculate stop loss and take profit
                symbol_config = get_symbol_config(symbol)
                instrument_type = self.symbol_utils.get_instrument_type(symbol)
                
                atr = indicators.get('atr', current_price * 0.001)
                
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
                
                if signal_type == SignalType.BUY:
                    sl = current_price - sl_distance
                    tp = current_price + tp_distance
                    reason = f"BUY: {len(positive_indicators)} bullish indicators"
                else:
                    sl = current_price + sl_distance
                    tp = current_price - tp_distance
                    reason = f"SELL: {len(negative_indicators)} bearish indicators"
                
                # Calculate quality score
                quality = self._calculate_signal_quality(
                    confidence, 
                    total_indicators, 
                    indicators, 
                    instrument_type
                )
                
                return Signal(
                    type=signal_type,
                    confidence=confidence,
                    entry=current_price,
                    sl=sl,
                    tp=tp,
                    reason=reason,
                    strategies={
                        'buy_score': total_buy,
                        'sell_score': total_sell,
                        'indicators': total_indicators
                    },
                    quality=quality
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ultra analysis for {symbol}: {e}")
            return None
    
    def force_trade_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Generate forced trade signal when no trades are happening"""
        try:
            # Simple momentum-based forced signal
            momentum = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            if abs(momentum) < 0.0001:  # If price hasn't moved, skip
                return None
            
            signal_type = SignalType.BUY if momentum > 0 else SignalType.SELL
            
            # Minimal stop loss and take profit
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            if instrument_type == 'exotic':
                sl_pips = 100
                tp_pips = 150
            elif instrument_type == 'metal':
                sl_pips = 50
                tp_pips = 100
            elif instrument_type == 'crypto':
                sl_pips = 200
                tp_pips = 300
            else:
                sl_pips = 20
                tp_pips = 30
            
            pip_value = self._get_pip_value(symbol, current_price)
            
            if signal_type == SignalType.BUY:
                sl = current_price - (sl_pips * pip_value)
                tp = current_price + (tp_pips * pip_value)
                reason = f"FORCED BUY: Momentum {momentum:.2%}"
            else:
                sl = current_price + (sl_pips * pip_value)
                tp = current_price - (tp_pips * pip_value)
                reason = f"FORCED SELL: Momentum {momentum:.2%}"
            
            return Signal(
                type=signal_type,
                confidence=0.15,  # Low confidence for forced trades
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
    
    def _calculate_signal_quality(self, confidence: float, total_indicators: int, 
                                indicators: Dict[str, Any], instrument_type: str) -> float:
        """Calculate overall signal quality"""
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