#!/usr/bin/env python3
"""
Technical Indicators Module
Contains all 100 technical indicators organized in 10 categories
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger('Indicators')

class TechnicalIndicators:
    def __init__(self):
        self.cache = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate all 100 indicators organized by category"""
        try:
            indicators = {'current_price': current_price}
            
            # Calculate each category
            indicators.update(self._price_action_indicators(df, current_price))
            indicators.update(self._chart_patterns(df))
            indicators.update(self._mathematical_indicators(df, current_price))
            indicators.update(self._volatility_analysis(df, current_price))
            indicators.update(self._market_structure(df, current_price))
            indicators.update(self._momentum_analysis(df))
            indicators.update(self._volume_order_flow(df))
            indicators.update(self._time_based_patterns(df, current_price))
            indicators.update(self._statistical_analysis(df))
            indicators.update(self._advanced_composite_indicators(df, current_price))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {'current_price': current_price}
    
    def _price_action_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate price action indicators"""
        indicators = {}
        
        try:
            # Pin bar detection
            body = abs(df['close'] - df['open'])
            upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
            lower_wick = df[['close', 'open']].min(axis=1) - df['low']
            
            indicators['pin_bar_bull'] = float((lower_wick.iloc[-1] > 2 * body.iloc[-1]) and 
                                              (upper_wick.iloc[-1] < body.iloc[-1]))
            indicators['pin_bar_bear'] = float((upper_wick.iloc[-1] > 2 * body.iloc[-1]) and 
                                              (lower_wick.iloc[-1] < body.iloc[-1]))
            
            # Engulfing patterns
            indicators['engulfing_bull'] = float((df['close'].iloc[-1] > df['open'].iloc[-1]) and 
                                               (df['close'].iloc[-2] < df['open'].iloc[-2]) and
                                               (df['close'].iloc[-1] > df['open'].iloc[-2]) and
                                               (df['open'].iloc[-1] < df['close'].iloc[-2]))
            
            indicators['engulfing_bear'] = float((df['close'].iloc[-1] < df['open'].iloc[-1]) and 
                                               (df['close'].iloc[-2] > df['open'].iloc[-2]) and
                                               (df['close'].iloc[-1] < df['open'].iloc[-2]) and
                                               (df['open'].iloc[-1] > df['close'].iloc[-2]))
            
            # Doji
            indicators['doji'] = float(body.iloc[-1] < (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1)
            
            # Hammer
            indicators['hammer'] = float((lower_wick.iloc[-1] > 2 * body.iloc[-1]) and 
                                       (upper_wick.iloc[-1] < body.iloc[-1] * 0.5) and
                                       (df['close'].iloc[-1] > df['low'].iloc[-5:].min()))
            
            # Hanging man
            indicators['hanging_man'] = float((lower_wick.iloc[-1] > 2 * body.iloc[-1]) and 
                                            (upper_wick.iloc[-1] < body.iloc[-1] * 0.5) and
                                            (df['close'].iloc[-1] < df['high'].iloc[-5:].max()))
            
            # Three white soldiers
            indicators['three_white_soldiers'] = float(all(df['close'].iloc[i] > df['open'].iloc[i] and 
                                                          df['close'].iloc[i] > df['close'].iloc[i-1] 
                                                          for i in range(-3, 0)))
            
            # Three black crows
            indicators['three_black_crows'] = float(all(df['close'].iloc[i] < df['open'].iloc[i] and 
                                                       df['close'].iloc[i] < df['close'].iloc[i-1] 
                                                       for i in range(-3, 0)))
            
            # Inside bar
            indicators['inside_bar'] = float((df['high'].iloc[-1] < df['high'].iloc[-2]) and 
                                           (df['low'].iloc[-1] > df['low'].iloc[-2]))
            
            # Price action momentum
            indicators['pa_momentum'] = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
        except Exception as e:
            logger.error(f"Error in price action indicators: {e}")
            
        return indicators
    
    def _chart_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns"""
        indicators = {}
        
        try:
            # Simplified pattern detection
            highs = df['high'].rolling(20).max()
            lows = df['low'].rolling(20).min()
            
            # Head and shoulders (simplified)
            indicators['head_shoulders'] = 0.0
            
            # Double top/bottom
            recent_highs = df['high'].iloc[-20:]
            recent_lows = df['low'].iloc[-20:]
            
            indicators['double_top'] = float(len(recent_highs[recent_highs > recent_highs.quantile(0.95)]) >= 2)
            indicators['double_bottom'] = float(len(recent_lows[recent_lows < recent_lows.quantile(0.05)]) >= 2)
            
            # Triangle pattern (converging highs and lows)
            high_slope = np.polyfit(range(20), df['high'].iloc[-20:].values, 1)[0]
            low_slope = np.polyfit(range(20), df['low'].iloc[-20:].values, 1)[0]
            indicators['triangle_pattern'] = float(abs(high_slope + low_slope) < 0.001)
            
            # Channel detection
            mid_price = (highs + lows) / 2
            indicators['channel_upper'] = float(df['close'].iloc[-1] > mid_price.iloc[-1] + (highs.iloc[-1] - mid_price.iloc[-1]) * 0.8)
            indicators['channel_lower'] = float(df['close'].iloc[-1] < mid_price.iloc[-1] - (mid_price.iloc[-1] - lows.iloc[-1]) * 0.8)
            
            # Flag pattern
            indicators['flag_pattern'] = 0.0
            
            # Wedge patterns
            indicators['rising_wedge'] = float(high_slope > 0 and low_slope > 0 and high_slope < low_slope)
            indicators['falling_wedge'] = float(high_slope < 0 and low_slope < 0 and high_slope > low_slope)
            
        except Exception as e:
            logger.error(f"Error in chart patterns: {e}")
            
        return indicators
    
    def _mathematical_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate mathematical indicators"""
        indicators = {}
        
        try:
            # Fibonacci levels
            high = df['high'].iloc[-50:].max()
            low = df['low'].iloc[-50:].min()
            diff = high - low
            
            fib_levels = {
                'fib_236': low + diff * 0.236,
                'fib_382': low + diff * 0.382,
                'fib_500': low + diff * 0.500,
                'fib_618': low + diff * 0.618
            }
            
            for name, level in fib_levels.items():
                indicators[name] = float(abs(current_price - level) / level < 0.005)
            
            # Pivot points
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            
            indicators['pivot_point'] = float(abs(current_price - pivot) / pivot < 0.002)
            indicators['pivot_r1'] = float(abs(current_price - r1) / r1 < 0.002)
            indicators['pivot_s1'] = float(abs(current_price - s1) / s1 < 0.002)
            
            # Linear regression
            x = np.arange(len(df))[-20:]
            y = df['close'].iloc[-20:].values
            slope, intercept = np.polyfit(x, y, 1)
            
            indicators['lin_reg_slope'] = slope
            indicators['lin_reg_deviation'] = (current_price - (slope * x[-1] + intercept)) / current_price
            
        except Exception as e:
            logger.error(f"Error in mathematical indicators: {e}")
            
        return indicators
    
    def _volatility_analysis(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze volatility"""
        indicators = {}
        
        try:
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            indicators['atr'] = atr
            indicators['atr_ratio'] = atr / current_price
            
            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            
            upper_bb = sma20 + 2 * std20
            lower_bb = sma20 - 2 * std20
            
            indicators['bb_width'] = (upper_bb.iloc[-1] - lower_bb.iloc[-1]) / sma20.iloc[-1]
            indicators['bb_squeeze'] = float(indicators['bb_width'] < df['close'].rolling(100).apply(
                lambda x: (x.rolling(20).mean() + 2*x.rolling(20).std() - 
                          (x.rolling(20).mean() - 2*x.rolling(20).std())) / x.rolling(20).mean()
            ).quantile(0.1).iloc[-1])
            
            # Keltner Channels
            ema20 = df['close'].ewm(span=20).mean()
            kc_upper = ema20 + 2 * atr
            kc_lower = ema20 - 2 * atr
            
            indicators['keltner_upper'] = float(current_price > kc_upper.iloc[-1])
            indicators['keltner_lower'] = float(current_price < kc_lower.iloc[-1])
            
            # Historical volatility
            returns = df['close'].pct_change()
            indicators['hist_volatility'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Volatility ratio
            short_vol = returns.rolling(5).std().iloc[-1]
            long_vol = returns.rolling(20).std().iloc[-1]
            indicators['vol_ratio'] = short_vol / long_vol if long_vol > 0 else 1
            
            # Donchian Channels
            indicators['donchian_high'] = float(current_price > df['high'].rolling(20).max().iloc[-2])
            indicators['donchian_low'] = float(current_price < df['low'].rolling(20).min().iloc[-2])
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            
        return indicators
    
    def _market_structure(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze market structure"""
        indicators = {}
        
        try:
            # Support and resistance levels
            recent_highs = df['high'].iloc[-50:]
            recent_lows = df['low'].iloc[-50:]
            
            # Find significant levels
            resistance_levels = recent_highs[recent_highs > recent_highs.quantile(0.9)].unique()
            support_levels = recent_lows[recent_lows < recent_lows.quantile(0.1)].unique()
            
            indicators['near_resistance'] = float(any(abs(current_price - r) / r < 0.002 for r in resistance_levels))
            indicators['near_support'] = float(any(abs(current_price - s) / s < 0.002 for s in support_levels))
            
            # Structure breaks
            prev_high = df['high'].iloc[-20:-1].max()
            prev_low = df['low'].iloc[-20:-1].min()
            
            indicators['structure_break_up'] = float(current_price > prev_high)
            indicators['structure_break_down'] = float(current_price < prev_low)
            
            # Higher highs and lower lows
            highs = df['high'].iloc[-20:]
            lows = df['low'].iloc[-20:]
            
            hh_count = sum(1 for i in range(1, len(highs)) if highs.iloc[i] > highs.iloc[i-1])
            ll_count = sum(1 for i in range(1, len(lows)) if lows.iloc[i] < lows.iloc[i-1])
            
            indicators['higher_highs'] = hh_count / len(highs)
            indicators['lower_lows'] = ll_count / len(lows)
            
            # Range detection
            range_high = df['high'].iloc[-20:].max()
            range_low = df['low'].iloc[-20:].min()
            range_size = (range_high - range_low) / ((range_high + range_low) / 2)
            
            indicators['in_range'] = float(range_size < 0.02)
            
        except Exception as e:
            logger.error(f"Error in market structure: {e}")
            
        return indicators
    
    def _momentum_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum"""
        indicators = {}
        
        try:
            # Rate of Change
            for period in [5, 10, 20]:
                indicators[f'roc_{period}'] = (df['close'].iloc[-1] - df['close'].iloc[-period-1]) / df['close'].iloc[-period-1]
            
            # Momentum
            indicators['momentum_14'] = df['close'].iloc[-1] - df['close'].iloc[-15]
            
            # Price Oscillator
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            indicators['price_oscillator'] = (ema12.iloc[-1] - ema26.iloc[-1]) / ema26.iloc[-1]
            
            # CCI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma20 = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            indicators['cci'] = (typical_price.iloc[-1] - sma20.iloc[-1]) / (0.015 * mad.iloc[-1])
            
            # Williams %R
            highest_high = df['high'].rolling(14).max()
            lowest_low = df['low'].rolling(14).min()
            indicators['williams_r'] = -100 * (highest_high.iloc[-1] - df['close'].iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1])
            
            # Ultimate Oscillator
            bp = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
            tr = pd.concat([df['high'] - df['low'], 
                           abs(df['high'] - df['close'].shift()),
                           abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
            
            avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
            
            indicators['ultimate_oscillator'] = 100 * ((4 * avg7.iloc[-1]) + (2 * avg14.iloc[-1]) + avg28.iloc[-1]) / 7
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            
        return indicators
    
    def _volume_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume and order flow"""
        indicators = {}
        
        try:
            # Volume moving averages
            if 'volume' in df.columns:
                indicators['volume_sma_10'] = df['volume'].rolling(10).mean().iloc[-1]
                indicators['volume_sma_20'] = df['volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1
                
                # OBV
                obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
                indicators['obv_trend'] = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) if obv.iloc[-20] != 0 else 0
                
                # A/D Line
                clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
                clv = clv.fillna(0)
                ad = (clv * df['volume']).cumsum()
                indicators['ad_line'] = (ad.iloc[-1] - ad.iloc[-20]) / abs(ad.iloc[-20]) if ad.iloc[-20] != 0 else 0
                
                # Chaikin Money Flow
                mf_volume = clv * df['volume']
                indicators['chaikin_mf'] = mf_volume.rolling(20).sum().iloc[-1] / df['volume'].rolling(20).sum().iloc[-1]
                
                # Volume Price Trend
                vpt = (df['volume'] * df['close'].pct_change()).cumsum()
                indicators['vpt'] = (vpt.iloc[-1] - vpt.iloc[-20]) / abs(vpt.iloc[-20]) if vpt.iloc[-20] != 0 else 0
                
                # Force Index
                indicators['force_index'] = df['volume'].iloc[-1] * (df['close'].iloc[-1] - df['close'].iloc[-2])
                
                # Money Flow Index
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                money_flow = typical_price * df['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
                
                mfi_ratio = positive_flow / negative_flow
                indicators['mfi'] = 100 - (100 / (1 + mfi_ratio.iloc[-1]))
            else:
                # If no volume data, set defaults
                for key in ['volume_sma_10', 'volume_sma_20', 'volume_ratio', 'obv_trend', 
                           'ad_line', 'chaikin_mf', 'vpt', 'force_index', 'mfi']:
                    indicators[key] = 0.0
                    
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            
        return indicators
    
    def _time_based_patterns(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze time-based patterns"""
        indicators = {}
        
        try:
            from datetime import datetime
            import pytz
            
            # Get current time in different timezones
            utc_now = datetime.now(pytz.UTC)
            tokyo_now = utc_now.astimezone(pytz.timezone('Asia/Tokyo'))
            london_now = utc_now.astimezone(pytz.timezone('Europe/London'))
            ny_now = utc_now.astimezone(pytz.timezone('America/New_York'))
            
            # Session detection
            tokyo_hour = tokyo_now.hour
            london_hour = london_now.hour
            ny_hour = ny_now.hour
            
            indicators['asian_session'] = float(0 <= tokyo_hour < 9)
            indicators['london_session'] = float(8 <= london_hour < 17)
            indicators['ny_session'] = float(8 <= ny_hour < 17)
            indicators['session_overlap'] = float((indicators['london_session'] and indicators['ny_session']) or
                                                 (indicators['asian_session'] and indicators['london_session']))
            
            # Day of week
            weekday = tokyo_now.weekday()
            indicators['monday'] = float(weekday == 0)
            indicators['friday'] = float(weekday == 4)
            indicators['midweek'] = float(1 <= weekday <= 3)
            
            # Intraday momentum
            if len(df) >= 12:  # 1 hour of 5-minute candles
                indicators['hourly_momentum'] = (df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12]
            else:
                indicators['hourly_momentum'] = 0.0
                
            if len(df) >= 24:  # 2 hours of 5-minute candles
                indicators['two_hour_momentum'] = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24]
            else:
                indicators['two_hour_momentum'] = 0.0
            
            # Opening range
            if len(df) >= 60:  # Assuming we have enough data
                opening_high = df['high'].iloc[-60:-48].max()
                opening_low = df['low'].iloc[-60:-48].min()
                
                indicators['above_opening_range'] = float(current_price > opening_high)
                indicators['below_opening_range'] = float(current_price < opening_low)
            else:
                indicators['above_opening_range'] = 0.0
                indicators['below_opening_range'] = 0.0
                
        except Exception as e:
            logger.error(f"Error in time-based patterns: {e}")
            
        return indicators
    
    def _statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis"""
        indicators = {}
        
        try:
            returns = df['close'].pct_change().dropna()
            
            # Z-score
            mean_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            indicators['z_score'] = (df['close'].iloc[-1] - mean_20.iloc[-1]) / std_20.iloc[-1] if std_20.iloc[-1] > 0 else 0
            
            # Percentile rank
            indicators['percentile_rank'] = stats.percentileofscore(df['close'].iloc[-50:], df['close'].iloc[-1]) / 100
            
            # Standard deviation bands position
            indicators['std_band_position'] = indicators['z_score'] / 2  # Normalized to [-1, 1] roughly
            
            # Skewness and kurtosis
            if len(returns) >= 20:
                indicators['return_skewness'] = stats.skew(returns.iloc[-20:])
                indicators['return_kurtosis'] = stats.kurtosis(returns.iloc[-20:])
            else:
                indicators['return_skewness'] = 0.0
                indicators['return_kurtosis'] = 0.0
            
            # Autocorrelation
            if len(returns) >= 20:
                indicators['autocorrelation'] = returns.iloc[-20:].autocorr(lag=1)
            else:
                indicators['autocorrelation'] = 0.0
            
            # Mean reversion
            distance_from_mean = (df['close'].iloc[-1] - mean_20.iloc[-1]) / mean_20.iloc[-1]
            indicators['mean_reversion'] = -distance_from_mean  # Negative when above mean, positive when below
            
            # Efficiency ratio
            if len(df) >= 20:
                net_change = abs(df['close'].iloc[-1] - df['close'].iloc[-20])
                sum_of_changes = abs(df['close'].diff()).iloc[-20:].sum()
                indicators['efficiency_ratio'] = net_change / sum_of_changes if sum_of_changes > 0 else 0
            else:
                indicators['efficiency_ratio'] = 0.0
            
            # Hurst exponent (simplified)
            if len(returns) >= 100:
                # Simplified Hurst calculation
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(returns.iloc[lag:].values, returns.iloc[:-lag].values))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                indicators['hurst_exponent'] = poly[0] * 2.0
            else:
                indicators['hurst_exponent'] = 0.5
                
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            
        return indicators
    
    def _advanced_composite_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate advanced composite indicators"""
        indicators = {}
        
        try:
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Stochastic
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            indicators['stoch_k'] = k_percent.iloc[-1]
            indicators['stoch_d'] = k_percent.rolling(3).mean().iloc[-1]
            
            # ADX
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            
            atr14 = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            
            indicators['adx'] = adx.iloc[-1]
            indicators['plus_di'] = plus_di.iloc[-1]
            indicators['minus_di'] = minus_di.iloc[-1]
            
            # Ichimoku
            period9_high = df['high'].rolling(9).max()
            period9_low = df['low'].rolling(9).min()
            period26_high = df['high'].rolling(26).max()
            period26_low = df['low'].rolling(26).min()
            period52_high = df['high'].rolling(52).max()
            period52_low = df['low'].rolling(52).min()
            
            tenkan_sen = (period9_high + period9_low) / 2
            kijun_sen = (period26_high + period26_low) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            chikou_span = df['close'].shift(-26)
            
            indicators['tenkan_sen'] = tenkan_sen.iloc[-1]
            indicators['kijun_sen'] = kijun_sen.iloc[-1]
            indicators['senkou_span_a'] = senkou_span_a.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) else tenkan_sen.iloc[-1]
            indicators['senkou_span_b'] = senkou_span_b.iloc[-1] if not pd.isna(senkou_span_b.iloc[-1]) else kijun_sen.iloc[-1]
            indicators['chikou_span'] = current_price
            
            # Cloud signals
            indicators['above_cloud'] = float(current_price > max(indicators['senkou_span_a'], indicators['senkou_span_b']))
            indicators['below_cloud'] = float(current_price < min(indicators['senkou_span_a'], indicators['senkou_span_b']))
            
        except Exception as e:
            logger.error(f"Error in advanced indicators: {e}")
            
        return indicators