#!/usr/bin/env python3
"""
Quantum-Inspired Technical Indicators Module
Advanced market analysis with a focus on robust, proven indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from scipy import stats, signal
from collections import deque
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('QuantumIndicators')

@dataclass
class MarketRegime:
    """Market regime detection results"""
    trend: str  # 'bullish', 'bearish', 'ranging'
    volatility: str  # 'low', 'medium', 'high', 'extreme'
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
    time_horizon: int  # Expected bars to target

@dataclass
class QuantumState:
    """Simplified Quantum market state representation"""
    amplitude: complex
    phase: float
    entanglement: float  # 0-1
    coherence: float  # 0-1
    measurement_basis: str  # 'price', 'momentum', 'volatility'

class QuantumUltraIntelligentIndicators:
    """
    An advanced trading analysis system focusing on:

    1. QUANTUM-INSPIRED MODELS:
       - Simplified market state representation using quantum concepts.
       - Coherence and entanglement to gauge market state clarity.

    2. ROBUST MARKET ANALYSIS:
       - Multi-faceted regime detection (trend, volatility, momentum).
       - Order flow and volume analysis for microstructure insights.

    3. PATTERN RECOGNITION:
       - Classic chart patterns (Harmonics, Elliott Waves).
       - Candlestick patterns with volume confirmation.

    4. COMPREHENSIVE SENTIMENT ANALYSIS:
       - Fear/greed index derived from price action and volatility.

    5. ADVANCED RISK METRICS:
       - VaR and CVaR for tail risk assessment.
       - Drawdown analysis and risk-adjusted return ratios (Sortino, Calmar).
       - Kelly Criterion for position sizing insights.

    6. ATTENTION-BASED SIGNAL FUSION:
       - Fusing diverse signals using an attention mechanism.
       - Adaptive weighting of signal categories.
    """
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.pattern_memory = deque(maxlen=200)
        self.adaptive_params = {}
        self.attention_weights = self._initialize_attention_mechanism()
        self.learning_rate = 0.001
        self.deep_memory = deque(maxlen=1000)

    def _initialize_attention_mechanism(self) -> Dict[str, float]:
        """Initialize attention weights for signal fusion"""
        weights = {
            'regime': 0.20,
            'patterns': 0.15,
            'statistics': 0.15,
            'microstructure': 0.10,
            'sentiment': 0.10,
            'risk': 0.10,
            'quantum': 0.20,
        }
        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def calculate_ultra_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate streamlined and robust technical indicators."""
        # --- Input Validation ---
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input 'df' must be a non-empty pandas DataFrame.")
        if len(df) < 50: # Increased minimum length for stability
            raise ValueError("DataFrame must contain at least 50 rows for stable analysis.")
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        if df[required_columns].isnull().values.any():
            # Fill NaNs, as this is a common issue.
            df.ffill(inplace=True)
            logger.warning("NaN values detected and forward-filled.")

        try:
            # 1. Quantum market state analysis (Simplified)
            quantum_state = self._initialize_quantum_state(df, current_price)
            
            # 2. Market Regime Detection
            regime = self._detect_market_regime(df, current_price, quantum_state)
            self.regime_history.append(regime)
            
            # 3. Parameter Adaptation
            self._adapt_parameters(regime, quantum_state)
            
            # 4. Core Indicator Calculations
            patterns = self._ml_pattern_recognition(df, current_price)
            stats_analysis = self._statistical_analysis(df, current_price)
            microstructure = self._analyze_microstructure(df)
            sentiment_score = self._advanced_sentiment_analysis(df, current_price) # Returns a float
            traditional = self._calculate_adaptive_indicators(df)

            # 5. Risk Analysis
            risk_metrics = self._calculate_advanced_risk_metrics(df, regime)

            # 6. Signal Fusion using Attention Mechanism
            all_signals = {
                'regime': regime,
                'patterns': patterns,
                'statistics': stats_analysis,
                'microstructure': microstructure,
                'sentiment': sentiment_score, # Pass the float score
                'risk': risk_metrics,
                'quantum': quantum_state,
            }
            composite_signal_score = self._quantum_neural_fusion(all_signals) # Returns a float

            # 7. Prepare output for trading_strategy integration
            price_action_signals = self._extract_price_action_signals(patterns)
            volume_signals = self._extract_volume_signals(df, microstructure)
            time_based_signals = self._extract_time_based_signals()

            # Construct final dictionary with corrected types
            return {
                'current_price': current_price,
                'regime': regime.__dict__,
                'quantum_state': {
                    'amplitude': abs(quantum_state.amplitude),
                    'phase': quantum_state.phase,
                    'entanglement': quantum_state.entanglement,
                    'coherence': quantum_state.coherence
                },
                'patterns': [p.__dict__ for p in patterns],
                'statistics': stats_analysis,
                'microstructure': microstructure,
                'sentiment': sentiment_score, # Store the float value
                'composite_signal': composite_signal_score, # Store the float value
                'traditional': traditional,
                'risk_metrics': risk_metrics,
                'price_action': price_action_signals,
                'mathematical': traditional,
                'volume': volume_signals,
                'time_based': time_based_signals,
            }

        except Exception as e:
            logger.error(f"Error in indicator calculation: {e}", exc_info=True)
            return {'current_price': current_price, 'error': str(e)}

    def _initialize_quantum_state(self, df: pd.DataFrame, current_price: float) -> QuantumState:
        """Initialize a simplified quantum-inspired representation of the market state."""
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 20:
                return QuantumState(complex(0,0), 0.0, 0.5, 0.5, 'price')

            volatility = returns.std()
            momentum = returns.rolling(10).mean().iloc[-1]
            
            k = momentum * 100
            omega = volatility * 100
            
            amplitude = complex(np.cos(k - omega), np.sin(k - omega)) * np.exp(-volatility * 10)
            phase = np.angle(amplitude)
            
            # Coherence from autocorrelation
            coherence = abs(returns.iloc[-50:].autocorr(lag=1)) if len(returns) >= 50 else 0.5

            # Entanglement from multi-timeframe volatility correlation
            vol_5 = returns.rolling(5).std()
            vol_20 = returns.rolling(20).std()
            entanglement = vol_5.corr(vol_20) if len(returns) >= 20 else 0.5
            
            return QuantumState(
                amplitude=amplitude,
                phase=phase,
                entanglement=float(np.nan_to_num(entanglement, nan=0.5)),
                coherence=float(np.nan_to_num(coherence, nan=0.5)),
                measurement_basis='price'
            )
        except Exception as e:
            logger.warning(f"Failed to initialize quantum state: {e}. Returning default.")
            return QuantumState(complex(1,0), 0, 0.5, 0.5, 'price')

    def _detect_market_regime(self, df: pd.DataFrame, current_price: float, quantum_state: QuantumState) -> MarketRegime:
        """Detects the current market regime based on trend, volatility, and momentum."""
        try:
            # Trend
            sma50 = df['close'].rolling(50).mean().iloc[-1]
            sma200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma50
            if current_price > sma50 > sma200:
                trend = 'bullish'
            elif current_price < sma50 < sma200:
                trend = 'bearish'
            else:
                trend = 'ranging'
            
            # Volatility
            atr = self._calculate_atr(df, 14)
            atr_ratio = atr / current_price if current_price > 0 else 0
            if atr_ratio > 0.005: volatility = 'extreme'
            elif atr_ratio > 0.003: volatility = 'high'
            elif atr_ratio < 0.001: volatility = 'low'
            else: volatility = 'medium'

            # Momentum
            rsi = self._calculate_rsi(df, 14)
            if rsi > 70: momentum = 'strong_up'
            elif rsi > 55: momentum = 'weak_up'
            elif rsi < 30: momentum = 'strong_down'
            elif rsi < 45: momentum = 'weak_down'
            else: momentum = 'neutral'
            
            # Confidence
            confidence = (quantum_state.coherence + (1-abs(atr_ratio*100-1))) / 2

            return MarketRegime(trend, volatility, momentum, confidence)

        except Exception as e:
            logger.warning(f"Failed to detect market regime: {e}. Returning default.")
            return MarketRegime('ranging', 'medium', 'neutral', 0.5)

    def _adapt_parameters(self, regime: MarketRegime, quantum_state: QuantumState):
        """Adapt indicator parameters based on market regime."""
        vol_map = {'low': 0.8, 'medium': 1.0, 'high': 1.2, 'extreme': 1.5}
        vol_factor = vol_map.get(regime.volatility, 1.0)
        
        quantum_factor = 1.0 + (quantum_state.coherence - 0.5) * -0.2 # Higher coherence -> shorter periods
        
        factor = vol_factor * quantum_factor

        self.adaptive_params = {
            'rsi_period': int(np.clip(14 * factor, 7, 28)),
            'atr_period': int(np.clip(14 * factor, 7, 28)),
            'ma_fast': int(np.clip(12 * factor, 5, 20)),
            'ma_slow': int(np.clip(26 * factor, 20, 50)),
        }

    def _ml_pattern_recognition(self, df: pd.DataFrame, current_price: float) -> List[PatternSignal]:
        """Recognize basic candlestick and chart patterns."""
        patterns = []
        if len(df) < 50: return patterns
        
        # Engulfing pattern
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last.close > prev.open and last.open < prev.close and prev.close < prev.open and last.close > last.open:
            patterns.append(PatternSignal('Engulfing_Bull', 'bullish', 0.7, current_price * 1.01, last.low, 10))
        if last.close < prev.open and last.open > prev.close and prev.close > prev.open and last.close < last.open:
            patterns.append(PatternSignal('Engulfing_Bear', 'bearish', 0.7, current_price * 0.99, last.high, 10))
            
        # Add more simplified patterns here if needed
        return patterns

    def _statistical_analysis(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Perform statistical analysis on price data."""
        stats_results = {}
        returns = df['close'].pct_change().dropna()
        if len(returns) < 20: return {}
        
        stats_results['volatility'] = returns.std()
        stats_results['skew'] = returns.skew()
        stats_results['kurtosis'] = returns.kurtosis()
        
        # Jump detection
        threshold = 3 * returns.rolling(20).std().iloc[-1]
        stats_results['jump_detected'] = abs(returns.iloc[-1]) > threshold

        return stats_results

    def _analyze_microstructure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure through volume."""
        microstructure = {}
        if 'volume' not in df.columns or df['volume'].isnull().all():
            return {'order_flow_imbalance': 0, 'volume_spike': False}

        # Order Flow Imbalance
        price_change = df['close'].diff()
        signed_volume = df['volume'] * np.sign(price_change)
        ofi = signed_volume.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, 1e-10)
        microstructure['order_flow_imbalance'] = ofi.iloc[-1] if not ofi.empty else 0
        
        # Volume Spike
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        microstructure['volume_spike'] = df['volume'].iloc[-1] > (vol_mean * 2)

        return microstructure

    def _advanced_sentiment_analysis(self, df: pd.DataFrame, current_price: float) -> float:
        """Analyze sentiment and return a single score."""
        returns = df['close'].pct_change().dropna()
        if len(returns) < 20: return 0.0

        # Fear/Greed Index from RSI and Volatility
        rsi = self._calculate_rsi(df, 14)
        vol = returns.rolling(20).std().iloc[-1]
        vol_percentile = stats.percentileofscore(returns.rolling(200).std().dropna(), vol) / 100 if len(returns) > 200 else 0.5
        
        rsi_score = (rsi - 50) / 50
        vol_score = 1 - (vol_percentile * 2) # High volatility -> more fear -> lower score
        
        return np.clip((rsi_score * 0.6 + vol_score * 0.4), -1, 1)

    def _quantum_neural_fusion(self, signals: Dict[str, Any]) -> float:
        """Fuse signals using an attention mechanism and return a single score."""
        try:
            features = self._extract_fusion_features(signals)
            
            final_score = 0.0
            total_weight = 0.0
            
            valid_features = {k: v for k, v in features.items() if v is not None}
            
            for category, feature_score in valid_features.items():
                weight = self.attention_weights.get(category, 0)
                final_score += feature_score * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score /= total_weight
            
            return np.clip(final_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error in signal fusion: {e}", exc_info=True)
            return 0.0

    def _extract_fusion_features(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """Extract a single score from each signal category."""
        features = {}

        # Regime
        regime = signals.get('regime')
        if regime:
            trend_score = {'bullish': 1, 'ranging': 0, 'bearish': -1}.get(regime.trend, 0)
            features['regime'] = trend_score * regime.confidence

        # Patterns
        patterns = signals.get('patterns', [])
        if patterns:
            bull_score = sum(p.confidence for p in patterns if p.direction == 'bullish')
            bear_score = sum(p.confidence for p in patterns if p.direction == 'bearish')
            features['patterns'] = bull_score - bear_score

        # Statistics
        stats_data = signals.get('statistics', {})
        features['statistics'] = -stats_data.get('skew', 0) * 0.1

        # Microstructure
        micro = signals.get('microstructure', {})
        features['microstructure'] = micro.get('order_flow_imbalance', 0)

        # Sentiment is now a float
        features['sentiment'] = signals.get('sentiment', 0.0)
        
        # Risk
        risk = signals.get('risk_metrics', {})
        features['risk'] = (0.5 - risk.get('overall_risk_score', 0.5)) 

        # Quantum
        qs = signals.get('quantum')
        if qs:
            features['quantum'] = np.tanh(np.real(qs.amplitude)) * qs.coherence

        # Ensure all features are float, default to 0.0 if None
        return {k: (v if isinstance(v, (int, float)) else 0.0) for k, v in features.items()}

    def _calculate_adaptive_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate traditional indicators with adaptive parameters."""
        indicators = {}
        params = self.adaptive_params
        
        # Use adaptive parameters or defaults
        rsi_period = params.get('rsi_period', 14)
        atr_period = params.get('atr_period', 14)
        ma_fast = params.get('ma_fast', 12)
        ma_slow = params.get('ma_slow', 26)

        indicators['rsi'] = self._calculate_rsi(df, rsi_period)
        indicators['atr'] = self._calculate_atr(df, atr_period)
        
        if len(df) >= ma_slow:
            indicators['ma_fast'] = df['close'].rolling(ma_fast).mean().iloc[-1]
            indicators['ma_slow'] = df['close'].rolling(ma_slow).mean().iloc[-1]
            macd, macd_signal, _ = self._calculate_macd(df, ma_fast, ma_slow)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
        
        return indicators

    def _calculate_advanced_risk_metrics(self, df: pd.DataFrame, regime: MarketRegime) -> Dict[str, Any]:
        """Calculate key risk metrics."""
        risk = {}
        returns = df['close'].pct_change().dropna()
        if len(returns) < 50: return {'overall_risk_score': 0.5}

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        risk['var_95'] = var_95
        risk['cvar_95'] = returns[returns <= var_95].mean()

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        risk['max_drawdown'] = drawdown.min()
        
        # Sortino Ratio
        annual_return = returns.mean() * 252
        downside_std = returns[returns < 0].std() * np.sqrt(252)
        risk['sortino_ratio'] = annual_return / downside_std if downside_std > 0 else 0

        # Overall Risk Score
        risk_score = (abs(var_95) * 10 + abs(risk.get('max_drawdown', 0)) * 2) / 2
        if regime.volatility in ['high', 'extreme']:
            risk_score *= 1.2
        risk['overall_risk_score'] = np.clip(risk_score, 0, 1)

        return risk

    # Helper methods for calculation
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float:
        if len(df) < period + 1: return 50.0
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        return np.nan_to_num(rsi, nan=50.0)

    def _calculate_macd(self, df: pd.DataFrame, fast: int, slow: int) -> Tuple[float, float, float]:
        if len(df) < slow: return 0.0, 0.0, 0.0
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1], macd_line.iloc[-1] - signal_line.iloc[-1]

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        if len(df) < period + 1: return 0.0
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return np.nan_to_num(atr, nan=0.0)

    # Helper methods for trading_strategy integration
    def _extract_price_action_signals(self, patterns: list) -> Dict[str, Any]:
        bullish = any(p.direction == 'bullish' for p in patterns)
        bearish = any(p.direction == 'bearish' for p in patterns)
        return {'engulfing_bull': 1 if bullish else 0, 'engulfing_bear': 1 if bearish else 0}

    def _extract_volume_signals(self, df: pd.DataFrame, microstructure: Dict[str, Any]) -> Dict[str, Any]:
        vol_signal = 0
        if 'volume' in df.columns and len(df) > 20:
            vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
            vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
            if vol_ma5 > vol_ma20 * 1.5: vol_signal = 1
            if vol_ma5 < vol_ma20 * 0.7: vol_signal = -1
        return {'volume_signal': vol_signal, 'obv_signal': microstructure.get('order_flow_imbalance', 0)}

    def _extract_time_based_signals(self) -> Dict[str, Any]:
        hour = pd.Timestamp.now(tz='UTC').hour
        # Simplified: London/NY overlap vs Asian session
        if 13 <= hour <= 16: return {'time_signal': 1} # High activity
        if 0 <= hour <= 7: return {'time_signal': -1} # Low activity
        return {'time_signal': 0}

# Create global instance for compatibility with other components
quantum_indicators = QuantumUltraIntelligentIndicators()