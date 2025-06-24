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

from .trading_models import Signal, SignalType
from .indicators import QuantumUltraIntelligentIndicators, quantum_indicators
from .symbol_utils import SymbolUtils
from .trading_config import CONFIG, get_symbol_config

logger = logging.getLogger('TradingStrategy')

class TradingStrategy:
    """Clean trading strategy focused on actual trading logic"""
    
    def __init__(self):
        """Initialize trading strategy with essential components"""
        self.indicators = quantum_indicators  # Use global indicator instance
        self.symbol_utils = SymbolUtils()
        self.account_leverage = 1.0
        
        # Essential category weights for signal evaluation
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
        
        # Performance tracking for adaptive learning
        self.performance_history = deque(maxlen=1000)
        self.weight_performance = {k: deque(maxlen=100) for k in self.category_weights}
        self.meta_learning_rate = 0.01
        self.signal_success_rate = {}
        
    def set_account_leverage(self, leverage: float):
        """Set account leverage for T/P and S/L calculations"""
        self.account_leverage = leverage
        
    def analyze_ultra(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Analyze market with intelligent indicators and generate trading signal"""
        try:
            # Calculate indicators using the main indicator system
            analysis = self.indicators.calculate_ultra_indicators(df, current_price)
            
            # Extract essential components
            regime = analysis['regime']
            composite_signal = analysis['composite_signal']
            patterns = analysis.get('patterns', [])
            sentiment = analysis.get('sentiment', {})
            predictions = analysis.get('predictions', {})
            
            # Evaluate signal strength from different categories
            scores = {}
            
            # Price action evaluation
            scores['price_action'] = self._evaluate_price_action(analysis.get('price_action', {}))
            
            # Chart patterns evaluation
            scores['chart_patterns'] = self._evaluate_chart_patterns(patterns)
            
            # Mathematical indicators
            scores['mathematical'] = self._evaluate_mathematical(analysis.get('mathematical', {}))
            
            # Volatility assessment
            scores['volatility'] = self._evaluate_volatility(analysis.get('volatility', {}))
            
            # Market structure
            scores['market_structure'] = self._evaluate_market_structure(analysis.get('market_structure', {}))
            
            # Momentum indicators
            scores['momentum'] = self._evaluate_momentum(analysis.get('momentum', {}))
            
            # Volume analysis
            scores['volume'] = self._evaluate_volume(analysis.get('volume', {}))
            
            # Time-based patterns
            scores['time_based'] = self._evaluate_time_patterns(analysis.get('time_based', {}))
            
            # Statistical indicators
            scores['statistical'] = self._evaluate_statistical(analysis.get('statistical', {}))
            
            # Composite score
            scores['composite'] = composite_signal
            
            # Calculate weighted final score
            weighted_score = sum(
                scores.get(category, 0) * weight 
                for category, weight in self.category_weights.items()
            )
            
            total_weight = sum(self.category_weights.values())
            final_score = weighted_score / total_weight if total_weight > 0 else 0
            
            # Determine signal direction and confidence
            if abs(final_score) < 0.1:
                return None  # No clear signal
                
            signal_type = SignalType.BUY if final_score > 0 else SignalType.SELL
            confidence = min(abs(final_score), 1.0)
            
            # Only proceed with high confidence signals (70%+)
            if confidence < 0.70:
                return None
                
            # Calculate stop loss and take profit
            symbol_base = symbol.rstrip('#')
            symbol_config = get_symbol_config(symbol_base)
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            # Set stop loss based on instrument type and volatility
            if instrument_type == 'exotic':
                sl_percent = 0.020  # 2.0% for exotic pairs
            elif instrument_type == 'metal':
                sl_percent = 0.015  # 1.5% for metals
            elif instrument_type == 'crypto':
                sl_percent = 0.025  # 2.5% for crypto
            else:
                sl_percent = 0.012  # 1.2% for major pairs
                
            # Calculate actual stop loss and take profit levels
            sl_distance = current_price * sl_percent
            
            # Use symbol-specific risk-reward ratio
            target_rr = symbol_config.get('target_rr_ratio', 2.0)
            tp_distance = sl_distance * target_rr
            
            if signal_type == SignalType.BUY:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                reason = f"BUY Signal: {final_score:.2f} confidence"
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                reason = f"SELL Signal: {final_score:.2f} confidence"
                
            # Calculate quality score based on multiple factors
            quality = confidence * 0.7 + (1 - abs(final_score - confidence)) * 0.3
            
            return Signal(
                type=signal_type,
                confidence=confidence,
                entry=current_price,
                sl=sl,
                tp=tp,
                reason=reason,
                strategies={
                    'categories': scores,
                    'regime': regime,
                    'final_score': final_score,
                    'timestamp': datetime.now()
                },
                quality=quality
            )
            
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
            return None
    
    def force_trade_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Signal]:
        """Generate forced trade signal when no trades are happening"""
        try:
            # Simple momentum-based forced signal
            momentum = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            if abs(momentum) < 0.0001:  # If price hasn't moved, skip
                return None
            
            signal_type = SignalType.BUY if momentum > 0 else SignalType.SELL
            
            # Get symbol configuration
            symbol_base = symbol.rstrip('#')
            symbol_config = get_symbol_config(symbol_base)
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            # Use wider stop loss for forced trades
            if instrument_type == 'exotic':
                sl_percent = 0.025   # 2.5% for exotic pairs
            elif instrument_type == 'metal':
                sl_percent = 0.020   # 2.0% for metals
            elif instrument_type == 'crypto':
                sl_percent = 0.030   # 3.0% for crypto
            else:
                sl_percent = 0.015   # 1.5% for major pairs
            
            # Calculate stop loss and take profit
            spread_margin = 0.003  # 0.3% for spread safety
            sl_distance = current_price * (sl_percent + spread_margin)
            
            # Use minimum RR ratio for forced trades
            min_rr = symbol_config.get('min_rr_ratio', 1.5)
            tp_distance = sl_distance * min_rr
            
            # Ensure minimum profit after spread
            typical_spread = symbol_config.get('typical_spread', 2)
            if 'JPY' in symbol:
                pip_value = 0.01
                min_tp_distance = max((typical_spread * 5) * pip_value, current_price * 0.002)
            else:
                pip_value = 0.0001
                min_tp_distance = max((typical_spread * 5) * pip_value, current_price * 0.002)
            tp_distance = max(tp_distance, min_tp_distance)
            
            if signal_type == SignalType.BUY:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                reason = f"FORCED BUY: Momentum {momentum:.2%}"
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
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
            score *= 0.5  # Reduce confidence for indecision
        
        return max(-1, min(1, score / 5))  # Normalize to [-1, 1]
    
    def _evaluate_chart_patterns(self, patterns: list) -> float:
        """Evaluate chart patterns"""
        if not patterns:
            return 0
        
        score = 0
        for pattern in patterns:
            if pattern.get('type') in ['bullish_flag', 'cup_handle', 'ascending_triangle']:
                score += pattern.get('strength', 0)
            elif pattern.get('type') in ['bearish_flag', 'head_shoulders', 'descending_triangle']:
                score -= pattern.get('strength', 0)
        
        return max(-1, min(1, score))
    
    def _evaluate_mathematical(self, indicators: Dict[str, Any]) -> float:
        """Evaluate mathematical indicators"""
        score = 0
        
        # Moving averages
        if indicators.get('sma_signal', 0) > 0:
            score += 0.5
        elif indicators.get('sma_signal', 0) < 0:
            score -= 0.5
            
        # RSI
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            score -= 0.3  # Overbought
        elif rsi < 30:
            score += 0.3  # Oversold
        
        # MACD
        macd_signal = indicators.get('macd_signal', 0)
        score += macd_signal * 0.4
        
        return max(-1, min(1, score))
    
    def _evaluate_volatility(self, indicators: Dict[str, Any]) -> float:
        """Evaluate volatility indicators"""
        atr_signal = indicators.get('atr_signal', 0)
        bb_signal = indicators.get('bollinger_signal', 0)
        
        # Combine volatility signals
        score = (atr_signal + bb_signal) / 2
        return max(-1, min(1, score))
    
    def _evaluate_market_structure(self, indicators: Dict[str, Any]) -> float:
        """Evaluate market structure indicators"""
        structure_signal = indicators.get('structure_signal', 0)
        support_resistance = indicators.get('support_resistance_signal', 0)
        
        score = (structure_signal + support_resistance) / 2
        return max(-1, min(1, score))
    
    def _evaluate_momentum(self, indicators: Dict[str, Any]) -> float:
        """Evaluate momentum indicators"""
        momentum = indicators.get('momentum', 0)
        roc = indicators.get('rate_of_change', 0)
        
        score = (momentum + roc) / 2
        return max(-1, min(1, score))
    
    def _evaluate_volume(self, indicators: Dict[str, Any]) -> float:
        """Evaluate volume indicators"""
        volume_signal = indicators.get('volume_signal', 0)
        obv_signal = indicators.get('obv_signal', 0)
        
        score = (volume_signal + obv_signal) / 2
        return max(-1, min(1, score))
    
    def _evaluate_time_patterns(self, indicators: Dict[str, Any]) -> float:
        """Evaluate time-based patterns"""
        time_signal = indicators.get('time_signal', 0)
        seasonal = indicators.get('seasonal_signal', 0)
        
        score = (time_signal + seasonal) / 2 
        return max(-1, min(1, score))
    
    def _evaluate_statistical(self, indicators: Dict[str, Any]) -> float:
        """Evaluate statistical indicators"""
        stat_signal = indicators.get('statistical_signal', 0)
        correlation = indicators.get('correlation_signal', 0)
        
        score = (stat_signal + correlation) / 2
        return max(-1, min(1, score))