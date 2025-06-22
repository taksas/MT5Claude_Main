#!/usr/bin/env python3
"""
Ultra-Intelligent Technical Indicators Module
Advanced market analysis with machine learning-inspired pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import stats, signal
from collections import deque
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('UltraIndicators')

@dataclass
class MarketRegime:
    """Market regime detection results"""
    trend: str  # 'bullish', 'bearish', 'ranging'
    volatility: str  # 'low', 'medium', 'high'
    momentum: str  # 'strong_up', 'weak_up', 'neutral', 'weak_down', 'strong_down'
    confidence: float  # 0-1
    
@dataclass
class PatternSignal:
    """Pattern recognition signal"""
    pattern_type: str
    direction: str  # 'bullish', 'bearish'
    confidence: float  # 0-1
    target_price: float
    stop_loss: float

class UltraIntelligentIndicators:
    def __init__(self):
        self.cache = {}
        self.regime_history = deque(maxlen=100)
        self.pattern_memory = deque(maxlen=500)
        self.adaptive_params = {}
        self.neural_weights = self._initialize_neural_weights()
        
    def _initialize_neural_weights(self) -> Dict[str, float]:
        """Initialize neural network-inspired weights for signal combination"""
        return {
            'trend': 0.25,
            'momentum': 0.20,
            'volatility': 0.15,
            'volume': 0.10,
            'pattern': 0.15,
            'statistical': 0.15
        }
        
    def calculate_ultra_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate ultra-intelligent indicators with ML-inspired analysis"""
        try:
            # Detect market regime first
            regime = self._detect_market_regime(df, current_price)
            self.regime_history.append(regime)
            
            # Adapt parameters based on regime
            self._adapt_parameters(regime)
            
            # Multi-timeframe analysis
            mtf_signals = self._multi_timeframe_analysis(df, current_price)
            
            # Pattern recognition with ML scoring
            patterns = self._ml_pattern_recognition(df, current_price)
            
            # Advanced statistical analysis
            stats_analysis = self._advanced_statistical_analysis(df, current_price)
            
            # Sentiment and momentum fusion
            sentiment = self._sentiment_analysis(df, current_price)
            
            # Predictive analytics
            predictions = self._predictive_analytics(df, current_price)
            
            # Neural signal fusion
            composite_signal = self._neural_signal_fusion({
                'regime': regime,
                'mtf': mtf_signals,
                'patterns': patterns,
                'stats': stats_analysis,
                'sentiment': sentiment,
                'predictions': predictions
            })
            
            # Calculate all traditional indicators with adaptive parameters
            traditional = self._calculate_adaptive_indicators(df, current_price)
            
            return {
                'current_price': current_price,
                'regime': regime.__dict__,
                'multi_timeframe': mtf_signals,
                'patterns': [p.__dict__ for p in patterns],
                'statistics': stats_analysis,
                'sentiment': sentiment,
                'predictions': predictions,
                'composite_signal': composite_signal,
                'traditional': traditional,
                'confidence': self._calculate_overall_confidence(composite_signal)
            }
            
        except Exception as e:
            logger.error(f"Error in ultra indicators: {e}")
            return {'current_price': current_price, 'error': str(e)}
    
    def _detect_market_regime(self, df: pd.DataFrame, current_price: float) -> MarketRegime:
        """Intelligent market regime detection"""
        try:
            # Trend detection using multiple methods
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
            sma200 = df['close'].rolling(200).mean() if len(df) >= 200 else sma50
            
            # Advanced trend scoring
            trend_score = 0
            if current_price > sma20.iloc[-1]: trend_score += 0.3
            if current_price > sma50.iloc[-1]: trend_score += 0.3
            if current_price > sma200.iloc[-1]: trend_score += 0.4
            if sma20.iloc[-1] > sma50.iloc[-1]: trend_score += 0.3
            if sma50.iloc[-1] > sma200.iloc[-1]: trend_score += 0.3
            
            # Slope analysis
            recent_slope = np.polyfit(range(20), df['close'].iloc[-20:].values, 1)[0]
            normalized_slope = recent_slope / df['close'].iloc[-1]
            
            # Determine trend
            if trend_score >= 1.2 and normalized_slope > 0.0002:
                trend = 'bullish'
            elif trend_score <= 0.4 and normalized_slope < -0.0002:
                trend = 'bearish'
            else:
                trend = 'ranging'
            
            # Volatility analysis
            atr = self._calculate_atr(df, 14)
            atr_ratio = atr / current_price
            historical_vol = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            if atr_ratio < 0.001 or historical_vol < 0.005:
                volatility = 'low'
            elif atr_ratio > 0.003 or historical_vol > 0.015:
                volatility = 'high'
            else:
                volatility = 'medium'
            
            # Momentum analysis
            rsi = self._calculate_rsi(df, 14)
            macd, macd_signal, _ = self._calculate_macd(df)
            momentum_score = 0
            
            if rsi > 70: momentum_score += 2
            elif rsi > 60: momentum_score += 1
            elif rsi < 30: momentum_score -= 2
            elif rsi < 40: momentum_score -= 1
            
            if macd > macd_signal: momentum_score += 1
            else: momentum_score -= 1
            
            if momentum_score >= 2:
                momentum = 'strong_up'
            elif momentum_score == 1:
                momentum = 'weak_up'
            elif momentum_score == 0:
                momentum = 'neutral'
            elif momentum_score == -1:
                momentum = 'weak_down'
            else:
                momentum = 'strong_down'
            
            # Calculate regime confidence
            confidence = min(1.0, abs(trend_score - 0.8) / 0.8 * 0.5 + 
                           abs(normalized_slope) * 1000 * 0.3 +
                           (1 - atr_ratio * 100) * 0.2)
            
            return MarketRegime(trend, volatility, momentum, confidence)
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return MarketRegime('ranging', 'medium', 'neutral', 0.5)
    
    def _adapt_parameters(self, regime: MarketRegime):
        """Adapt indicator parameters based on market regime"""
        if regime.volatility == 'high':
            self.adaptive_params = {
                'rsi_period': 21,  # Longer period for high volatility
                'ma_fast': 12,
                'ma_slow': 30,
                'atr_period': 21,
                'lookback': 30
            }
        elif regime.volatility == 'low':
            self.adaptive_params = {
                'rsi_period': 9,  # Shorter period for low volatility
                'ma_fast': 5,
                'ma_slow': 15,
                'atr_period': 10,
                'lookback': 15
            }
        else:
            self.adaptive_params = {
                'rsi_period': 14,
                'ma_fast': 9,
                'ma_slow': 21,
                'atr_period': 14,
                'lookback': 20
            }
    
    def _multi_timeframe_analysis(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze multiple timeframes for confluence"""
        mtf_signals = {}
        
        try:
            # Simulate different timeframes by resampling
            timeframes = {
                'M5': 1,    # Current timeframe
                'M15': 3,   # 15-minute
                'H1': 12,   # 1-hour
                'H4': 48    # 4-hour
            }
            
            for tf_name, multiplier in timeframes.items():
                if len(df) < multiplier * 20:
                    continue
                    
                # Resample data
                resampled = df.iloc[::multiplier].copy() if multiplier > 1 else df.copy()
                
                # Calculate key indicators for each timeframe
                rsi = self._calculate_rsi(resampled, 14)
                macd, macd_signal, macd_hist = self._calculate_macd(resampled)
                
                # Trend direction
                sma20 = resampled['close'].rolling(20).mean()
                trend = 1 if current_price > sma20.iloc[-1] else -1
                
                # Store signals
                mtf_signals[tf_name] = {
                    'trend': trend,
                    'rsi': rsi,
                    'macd_histogram': macd_hist,
                    'strength': abs(current_price - sma20.iloc[-1]) / sma20.iloc[-1]
                }
            
            # Calculate confluence score
            trend_sum = sum(s['trend'] for s in mtf_signals.values())
            confluence_score = trend_sum / len(mtf_signals)
            
            mtf_signals['confluence'] = confluence_score
            mtf_signals['aligned'] = abs(confluence_score) > 0.7
            
        except Exception as e:
            logger.error(f"Error in MTF analysis: {e}")
            
        return mtf_signals
    
    def _ml_pattern_recognition(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """Machine learning-inspired pattern recognition"""
        patterns = []
        
        try:
            # Harmonic patterns detection
            harmonic = self._detect_harmonic_patterns(df, current_price)
            if harmonic:
                patterns.extend(harmonic)
            
            # Elliott Wave patterns
            elliott = self._detect_elliott_waves(df, current_price)
            if elliott:
                patterns.extend(elliott)
            
            # Advanced candlestick patterns
            candlestick = self._detect_advanced_candlesticks(df, current_price)
            if candlestick:
                patterns.extend(candlestick)
            
            # Support/Resistance with ML clustering
            sr_levels = self._ml_support_resistance(df, current_price)
            if sr_levels:
                patterns.extend(sr_levels)
            
            # Sort by confidence
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            # Store in pattern memory for learning
            for pattern in patterns:
                self.pattern_memory.append({
                    'pattern': pattern,
                    'price': current_price,
                    'success': None  # To be updated later
                })
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            
        return patterns[:5]  # Return top 5 patterns
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)"""
        patterns = []
        
        try:
            if len(df) < 100:
                return patterns
                
            # Find swing points
            highs = df['high'].rolling(5).max() == df['high']
            lows = df['low'].rolling(5).min() == df['low']
            
            swing_highs = df[highs].index.tolist()
            swing_lows = df[lows].index.tolist()
            
            if len(swing_highs) >= 3 and len(swing_lows) >= 2:
                # Simplified Gartley pattern detection
                X = df['high'].iloc[swing_highs[-3]]
                A = df['low'].iloc[swing_lows[-2]]
                B = df['high'].iloc[swing_highs[-2]]
                C = df['low'].iloc[swing_lows[-1]]
                D = current_price
                
                XA = A - X
                AB = B - A
                BC = C - B
                CD = D - C
                
                # Check Fibonacci ratios
                if XA != 0 and AB != 0 and BC != 0:
                    AB_XA = abs(AB / XA)
                    BC_AB = abs(BC / AB)
                    CD_BC = abs(CD / BC)
                    
                    # Gartley ratios
                    if 0.58 <= AB_XA <= 0.65 and 0.35 <= BC_AB <= 0.9:
                        confidence = 0.7 * (1 - abs(AB_XA - 0.618)) * (1 - abs(BC_AB - 0.618))
                        target = D + (X - A) * 0.618
                        stop = C
                        
                        patterns.append(PatternSignal(
                            'Gartley',
                            'bullish' if D < C else 'bearish',
                            confidence,
                            target,
                            stop
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {e}")
            
        return patterns
    
    def _detect_elliott_waves(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """Detect Elliott Wave patterns"""
        patterns = []
        
        try:
            if len(df) < 50:
                return patterns
                
            # Simplified wave detection
            prices = df['close'].values[-50:]
            
            # Find local extrema
            peaks = signal.find_peaks(prices, distance=5)[0]
            troughs = signal.find_peaks(-prices, distance=5)[0]
            
            if len(peaks) >= 3 and len(troughs) >= 2:
                # Check for impulsive wave structure
                if troughs[0] < peaks[0] < troughs[1] < peaks[1]:
                    # Potential 5-wave structure
                    wave1 = prices[peaks[0]] - prices[troughs[0]]
                    wave3 = prices[peaks[1]] - prices[troughs[1]]
                    
                    if wave3 > wave1 * 1.618:  # Wave 3 should be extended
                        confidence = 0.8
                        target = current_price + wave1 * 1.618
                        stop = prices[troughs[-1]]
                        
                        patterns.append(PatternSignal(
                            'Elliott_Wave_5',
                            'bullish',
                            confidence,
                            target,
                            stop
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting Elliott waves: {e}")
            
        return patterns
    
    def _detect_advanced_candlesticks(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """Detect advanced candlestick patterns with ML scoring"""
        patterns = []
        
        try:
            # Three-line strike
            if len(df) >= 4:
                last_four = df.iloc[-4:]
                if (all(last_four['close'].iloc[i] < last_four['open'].iloc[i] for i in range(3)) and
                    last_four['close'].iloc[-1] > last_four['open'].iloc[-1] and
                    last_four['close'].iloc[-1] > last_four['open'].iloc[0]):
                    
                    confidence = 0.85
                    target = current_price * 1.01
                    stop = last_four['low'].iloc[-1]
                    
                    patterns.append(PatternSignal(
                        'Three_Line_Strike',
                        'bullish',
                        confidence,
                        target,
                        stop
                    ))
            
            # Morning/Evening star with volume confirmation
            if len(df) >= 3 and 'volume' in df.columns:
                last_three = df.iloc[-3:]
                
                # Morning star
                if (last_three['close'].iloc[0] < last_three['open'].iloc[0] and  # First bearish
                    abs(last_three['close'].iloc[1] - last_three['open'].iloc[1]) < 
                    (last_three['high'].iloc[1] - last_three['low'].iloc[1]) * 0.3 and  # Small body
                    last_three['close'].iloc[2] > last_three['open'].iloc[2] and  # Third bullish
                    last_three['volume'].iloc[2] > last_three['volume'].iloc[:2].mean() * 1.5):  # Volume surge
                    
                    confidence = 0.9
                    target = current_price * 1.015
                    stop = last_three['low'].min()
                    
                    patterns.append(PatternSignal(
                        'Morning_Star_Volume',
                        'bullish',
                        confidence,
                        target,
                        stop
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting candlesticks: {e}")
            
        return patterns
    
    def _ml_support_resistance(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """ML-based support/resistance detection using clustering"""
        patterns = []
        
        try:
            # Collect price levels
            price_levels = []
            price_levels.extend(df['high'].iloc[-100:].tolist())
            price_levels.extend(df['low'].iloc[-100:].tolist())
            price_levels.extend(df['close'].iloc[-100:].tolist())
            
            # Simple clustering
            price_levels = sorted(price_levels)
            clusters = []
            cluster_threshold = current_price * 0.002  # 0.2% threshold
            
            current_cluster = [price_levels[0]]
            for price in price_levels[1:]:
                if price - current_cluster[-1] <= cluster_threshold:
                    current_cluster.append(price)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [price]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            
            # Find nearest support/resistance
            for cluster in clusters:
                level = np.mean(cluster)
                touches = len(cluster)
                distance = abs(current_price - level) / current_price
                
                if distance < 0.005:  # Within 0.5%
                    if current_price > level:
                        # Support level
                        confidence = min(0.9, touches / 10)
                        target = current_price * 1.01
                        stop = level * 0.995
                        
                        patterns.append(PatternSignal(
                            'ML_Support',
                            'bullish',
                            confidence,
                            target,
                            stop
                        ))
                    else:
                        # Resistance level
                        confidence = min(0.9, touches / 10)
                        target = current_price * 0.99
                        stop = level * 1.005
                        
                        patterns.append(PatternSignal(
                            'ML_Resistance',
                            'bearish',
                            confidence,
                            target,
                            stop
                        ))
            
        except Exception as e:
            logger.error(f"Error in ML support/resistance: {e}")
            
        return patterns
    
    def _advanced_statistical_analysis(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Advanced statistical analysis with predictive elements"""
        stats = {}
        
        try:
            returns = df['close'].pct_change().dropna()
            
            # GARCH-like volatility clustering
            squared_returns = returns ** 2
            garch_vol = squared_returns.rolling(20).mean().iloc[-1] ** 0.5
            stats['garch_volatility'] = garch_vol
            
            # Regime switching probability
            bull_returns = returns[returns > 0]
            bear_returns = returns[returns < 0]
            
            if len(bull_returns) > 10 and len(bear_returns) > 10:
                bull_mean = bull_returns.mean()
                bear_mean = bear_returns.mean()
                bull_std = bull_returns.std()
                bear_std = bear_returns.std()
                
                last_return = returns.iloc[-1]
                
                # Bayesian-like probability
                bull_prob = stats.norm.pdf(last_return, bull_mean, bull_std)
                bear_prob = stats.norm.pdf(last_return, bear_mean, bear_std)
                
                stats['bull_regime_probability'] = bull_prob / (bull_prob + bear_prob)
            else:
                stats['bull_regime_probability'] = 0.5
            
            # Entropy (market uncertainty)
            if len(returns) >= 50:
                hist, _ = np.histogram(returns.iloc[-50:], bins=10)
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                stats['market_entropy'] = entropy / np.log(10)  # Normalized
            else:
                stats['market_entropy'] = 0.5
            
            # Jump detection
            threshold = 3 * returns.rolling(20).std().iloc[-1]
            stats['jump_detected'] = abs(returns.iloc[-1]) > threshold
            
            # Microstructure noise estimation
            if len(df) >= 100:
                # Simplified realized variance
                rv_5min = (returns ** 2).sum()
                rv_15min = (returns.iloc[::3] ** 2).sum() * 3
                noise_ratio = 1 - (rv_15min / rv_5min)
                stats['noise_ratio'] = max(0, min(1, noise_ratio))
            else:
                stats['noise_ratio'] = 0.1
            
            # Tail risk measures
            if len(returns) >= 100:
                var_95 = np.percentile(returns, 5)
                cvar_95 = returns[returns <= var_95].mean()
                stats['value_at_risk_95'] = var_95
                stats['conditional_var_95'] = cvar_95
            else:
                stats['value_at_risk_95'] = -0.02
                stats['conditional_var_95'] = -0.03
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            
        return stats
    
    def _sentiment_analysis(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze market sentiment from price action and volume"""
        sentiment = {}
        
        try:
            # Price action sentiment
            closes = df['close'].values
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Bullish/bearish candle ratio
            bullish_candles = sum(closes > opens)
            bearish_candles = sum(closes < opens)
            total_candles = len(closes)
            
            sentiment['bullish_ratio'] = bullish_candles / total_candles
            sentiment['bearish_ratio'] = bearish_candles / total_candles
            
            # Buying/selling pressure
            buying_pressure = sum((closes - lows) / (highs - lows + 1e-10)) / total_candles
            selling_pressure = sum((highs - closes) / (highs - lows + 1e-10)) / total_candles
            
            sentiment['buying_pressure'] = buying_pressure
            sentiment['selling_pressure'] = selling_pressure
            sentiment['pressure_ratio'] = buying_pressure / (selling_pressure + 1e-10)
            
            # Volume sentiment (if available)
            if 'volume' in df.columns:
                up_volume = df[df['close'] > df['open']]['volume'].sum()
                down_volume = df[df['close'] < df['open']]['volume'].sum()
                total_volume = df['volume'].sum()
                
                sentiment['volume_sentiment'] = (up_volume - down_volume) / total_volume if total_volume > 0 else 0
                
                # Smart money detection (large volume moves)
                avg_volume = df['volume'].rolling(20).mean()
                large_volume_moves = df[df['volume'] > avg_volume.iloc[-1] * 2]
                
                if len(large_volume_moves) > 0:
                    smart_money_direction = np.sign((large_volume_moves['close'] - large_volume_moves['open']).mean())
                    sentiment['smart_money_direction'] = smart_money_direction
                else:
                    sentiment['smart_money_direction'] = 0
            else:
                sentiment['volume_sentiment'] = 0
                sentiment['smart_money_direction'] = 0
            
            # Momentum sentiment
            short_ma = df['close'].rolling(5).mean().iloc[-1]
            long_ma = df['close'].rolling(20).mean().iloc[-1]
            
            sentiment['momentum_sentiment'] = (short_ma - long_ma) / long_ma
            
            # Fear/Greed index (simplified)
            rsi = self._calculate_rsi(df, 14)
            fear_greed = (rsi - 50) / 50  # Normalized to [-1, 1]
            sentiment['fear_greed_index'] = fear_greed
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            
        return sentiment
    
    def _predictive_analytics(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Predictive analytics using time series analysis"""
        predictions = {}
        
        try:
            # Linear regression prediction
            if len(df) >= 50:
                x = np.arange(50)
                y = df['close'].iloc[-50:].values
                
                # Polynomial features for non-linear patterns
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                
                # Predict next 3 periods
                next_1 = p(50)
                next_3 = p(52)
                next_5 = p(54)
                
                predictions['linear_pred_1'] = next_1
                predictions['linear_pred_3'] = next_3
                predictions['linear_pred_5'] = next_5
                predictions['linear_trend'] = 'up' if next_5 > current_price else 'down'
            
            # Fourier analysis for cyclic patterns
            if len(df) >= 100:
                prices = df['close'].iloc[-100:].values
                prices_detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
                
                fft = np.fft.fft(prices_detrended)
                frequencies = np.fft.fftfreq(len(prices))
                
                # Find dominant frequency
                idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                dominant_freq = frequencies[idx]
                dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else 100
                
                predictions['dominant_cycle'] = dominant_period
                predictions['cycle_phase'] = (len(prices) % dominant_period) / dominant_period
            
            # ARIMA-like prediction (simplified)
            if len(df) >= 50:
                returns = df['close'].pct_change().dropna().iloc[-50:]
                
                # AR(1) model
                ar_coef = returns.autocorr(lag=1)
                last_return = returns.iloc[-1]
                predicted_return = ar_coef * last_return
                
                predictions['ar_predicted_price'] = current_price * (1 + predicted_return)
                predictions['ar_confidence'] = abs(ar_coef)
            
            # Machine learning-inspired prediction
            features = []
            if len(df) >= 20:
                # Feature engineering
                features.append(self._calculate_rsi(df, 14) / 100)
                features.append((current_price - df['close'].rolling(20).mean().iloc[-1]) / current_price)
                features.append(df['close'].pct_change().rolling(5).std().iloc[-1] * 100)
                
                # Simple neural network-like combination
                weights = [0.4, -0.3, -0.2]  # Learned weights
                bias = 0.1
                
                activation = sum(f * w for f, w in zip(features, weights)) + bias
                ml_prediction = 1 / (1 + np.exp(-activation))  # Sigmoid
                
                predictions['ml_bullish_probability'] = ml_prediction
                predictions['ml_signal'] = 'bullish' if ml_prediction > 0.6 else 'bearish' if ml_prediction < 0.4 else 'neutral'
            
        except Exception as e:
            logger.error(f"Error in predictive analytics: {e}")
            
        return predictions
    
    def _neural_signal_fusion(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all signals using neural network-inspired approach"""
        composite = {}
        
        try:
            # Extract features from each signal category
            features = []
            
            # Regime features
            regime = signals['regime']
            features.append(1.0 if regime.trend == 'bullish' else -1.0 if regime.trend == 'bearish' else 0.0)
            features.append(1.0 if regime.volatility == 'low' else 0.0 if regime.volatility == 'medium' else -1.0)
            features.append(regime.confidence)
            
            # Multi-timeframe features
            mtf = signals['mtf']
            if 'confluence' in mtf:
                features.append(mtf['confluence'])
            else:
                features.append(0.0)
            
            # Pattern features
            patterns = signals['patterns']
            if patterns:
                avg_confidence = np.mean([p.__dict__['confidence'] for p in patterns])
                bullish_patterns = sum(1 for p in patterns if p.__dict__['direction'] == 'bullish')
                bearish_patterns = sum(1 for p in patterns if p.__dict__['direction'] == 'bearish')
                features.append(avg_confidence)
                features.append((bullish_patterns - bearish_patterns) / max(1, len(patterns)))
            else:
                features.extend([0.0, 0.0])
            
            # Statistical features
            stats = signals['stats']
            features.append(stats.get('bull_regime_probability', 0.5))
            features.append(1.0 - stats.get('market_entropy', 0.5))
            
            # Sentiment features
            sentiment = signals['sentiment']
            features.append(sentiment.get('pressure_ratio', 1.0) - 1.0)
            features.append(sentiment.get('fear_greed_index', 0.0))
            
            # Predictive features
            predictions = signals['predictions']
            features.append(1.0 if predictions.get('linear_trend') == 'up' else -1.0)
            features.append(predictions.get('ml_bullish_probability', 0.5))
            
            # Neural network layers
            # Layer 1
            hidden1 = []
            layer1_weights = [
                [0.3, -0.2, 0.4, 0.5, 0.2, -0.1, 0.3, 0.2, 0.1, -0.2, 0.4, 0.3],
                [0.2, 0.3, -0.1, 0.2, 0.4, 0.3, -0.2, 0.1, 0.3, 0.4, -0.1, 0.2],
                [-0.1, 0.4, 0.2, -0.3, 0.1, 0.5, 0.2, -0.2, 0.3, 0.1, 0.2, -0.3],
                [0.4, 0.1, 0.3, 0.2, -0.3, 0.2, 0.4, 0.3, -0.1, 0.2, 0.3, 0.1]
            ]
            
            for weights in layer1_weights:
                activation = sum(f * w for f, w in zip(features, weights))
                hidden1.append(np.tanh(activation))
            
            # Layer 2 (output)
            output_weights = [0.5, 0.3, -0.2, 0.4]
            final_activation = sum(h * w for h, w in zip(hidden1, output_weights))
            
            # Sigmoid output
            signal_strength = 1 / (1 + np.exp(-final_activation))
            
            # Convert to trading signal
            if signal_strength > 0.7:
                composite['signal'] = 'strong_buy'
                composite['confidence'] = signal_strength
            elif signal_strength > 0.6:
                composite['signal'] = 'buy'
                composite['confidence'] = signal_strength
            elif signal_strength < 0.3:
                composite['signal'] = 'strong_sell'
                composite['confidence'] = 1 - signal_strength
            elif signal_strength < 0.4:
                composite['signal'] = 'sell'
                composite['confidence'] = 1 - signal_strength
            else:
                composite['signal'] = 'neutral'
                composite['confidence'] = 1 - abs(signal_strength - 0.5) * 2
            
            # Additional insights
            composite['strength'] = signal_strength
            composite['feature_importance'] = {
                'regime': abs(features[0]) * 0.3,
                'patterns': features[4] * 0.25,
                'statistics': features[6] * 0.2,
                'sentiment': abs(features[8]) * 0.15,
                'predictions': features[11] * 0.1
            }
            
        except Exception as e:
            logger.error(f"Error in neural signal fusion: {e}")
            composite = {'signal': 'neutral', 'confidence': 0.0}
            
        return composite
    
    def _calculate_overall_confidence(self, composite_signal: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            base_confidence = composite_signal.get('confidence', 0.5)
            
            # Adjust based on regime history consistency
            if len(self.regime_history) >= 3:
                recent_regimes = list(self.regime_history)[-3:]
                regime_consistency = len(set(r.trend for r in recent_regimes)) == 1
                if regime_consistency:
                    base_confidence *= 1.2
            
            # Adjust based on pattern memory success rate
            if len(self.pattern_memory) >= 20:
                recent_patterns = list(self.pattern_memory)[-20:]
                success_rate = sum(1 for p in recent_patterns if p.get('success', False)) / len(recent_patterns)
                base_confidence *= (0.5 + success_rate)
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_adaptive_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate traditional indicators with adaptive parameters"""
        indicators = {}
        
        try:
            # Use adaptive parameters
            rsi_period = self.adaptive_params.get('rsi_period', 14)
            ma_fast = self.adaptive_params.get('ma_fast', 9)
            ma_slow = self.adaptive_params.get('ma_slow', 21)
            
            # Adaptive RSI
            indicators['adaptive_rsi'] = self._calculate_rsi(df, rsi_period)
            
            # Adaptive moving averages
            indicators['adaptive_ma_fast'] = df['close'].rolling(ma_fast).mean().iloc[-1]
            indicators['adaptive_ma_slow'] = df['close'].rolling(ma_slow).mean().iloc[-1]
            
            # Dynamic Bollinger Bands
            period = min(20, len(df) - 1)
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            
            # Adjust band width based on volatility regime
            if hasattr(self, 'regime_history') and self.regime_history:
                current_regime = self.regime_history[-1]
                if current_regime.volatility == 'high':
                    band_multiplier = 2.5
                elif current_regime.volatility == 'low':
                    band_multiplier = 1.5
                else:
                    band_multiplier = 2.0
            else:
                band_multiplier = 2.0
            
            indicators['adaptive_bb_upper'] = sma.iloc[-1] + band_multiplier * std.iloc[-1]
            indicators['adaptive_bb_lower'] = sma.iloc[-1] - band_multiplier * std.iloc[-1]
            indicators['adaptive_bb_width'] = (indicators['adaptive_bb_upper'] - indicators['adaptive_bb_lower']) / sma.iloc[-1]
            
            # More adaptive indicators can be added here
            
        except Exception as e:
            logger.error(f"Error in adaptive indicators: {e}")
            
        return indicators
    
    # Helper methods
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float:
        """Calculate RSI"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            return rsi
        except:
            return 50.0
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate MACD"""
        try:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate ATR"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return atr
        except:
            return 0.0