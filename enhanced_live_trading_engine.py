#!/usr/bin/env python3
"""
Enhanced Live Trading Engine for MT5 Forex Trading
Implements advanced strategies, multi-timeframe analysis, and adaptive risk management
Optimized for short-term trading (5-30 minutes) with focus on high win rate
"""

import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from improved_strategies import ImprovedStrategyEnsemble
from trading_strategies import SignalType, TradingSignal
import threading
from collections import deque

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    LOW_VOLATILITY = "LOW_VOLATILITY"

@dataclass
class MarketAnalysis:
    condition: MarketCondition
    volatility: float
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    volume_profile: str
    timestamp: str

@dataclass
class EnhancedTrade:
    ticket: int
    symbol: str
    signal_type: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float = 0.01
    status: str = "OPEN"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    market_condition: Optional[MarketCondition] = None
    confidence: float = 0.0
    strategy_names: List[str] = field(default_factory=list)
    max_drawdown: float = 0.0
    trade_duration: Optional[int] = None

class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes for better signal confirmation"""
    
    def __init__(self, api_base: str):
        self.api_base = api_base
        self.timeframes = ["M1", "M5", "M15", "M30"]
        self.cache = {}
        self.cache_expiry = 30  # seconds
        
    def get_mtf_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get multi-timeframe analysis for a symbol"""
        analysis = {
            "trend_alignment": False,
            "momentum_score": 0.0,
            "volatility_profile": {},
            "key_levels": {}
        }
        
        trends = []
        volatilities = []
        
        for tf in self.timeframes:
            data = self._get_cached_data(symbol, tf)
            if data:
                trend = self._analyze_trend(data)
                volatility = self._calculate_volatility(data)
                trends.append(trend)
                volatilities.append(volatility)
        
        # Check trend alignment
        if trends:
            up_trends = sum(1 for t in trends if t == "UP")
            down_trends = sum(1 for t in trends if t == "DOWN")
            
            if up_trends >= 3:
                analysis["trend_alignment"] = True
                analysis["aligned_direction"] = "UP"
            elif down_trends >= 3:
                analysis["trend_alignment"] = True
                analysis["aligned_direction"] = "DOWN"
        
        # Calculate momentum score
        analysis["momentum_score"] = self._calculate_momentum_score(trends)
        
        # Volatility profile
        if volatilities:
            analysis["volatility_profile"] = {
                "current": volatilities[-1],
                "average": np.mean(volatilities),
                "trend": "increasing" if volatilities[-1] > np.mean(volatilities) else "decreasing"
            }
        
        return analysis
    
    def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        """Get market data with caching"""
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_expiry:
                return cached_data
        
        # Fetch new data
        url = f"{self.api_base}/market/history"
        data = {"symbol": symbol, "timeframe": timeframe, "count": 50}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                market_data = response.json()
                self.cache[cache_key] = (market_data, now)
                return market_data
        except Exception as e:
            logger.error(f"Failed to get {timeframe} data for {symbol}: {e}")
        
        return None
    
    def _analyze_trend(self, data: List[Dict]) -> str:
        """Analyze trend direction"""
        df = pd.DataFrame(data)
        if len(df) < 20:
            return "NEUTRAL"
        
        # Simple trend using EMA
        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=20).mean()
        
        if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-5] > ema_long.iloc[-5]:
            return "UP"
        elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-5] < ema_long.iloc[-5]:
            return "DOWN"
        else:
            return "NEUTRAL"
    
    def _calculate_volatility(self, data: List[Dict]) -> float:
        """Calculate volatility using ATR"""
        df = pd.DataFrame(data)
        if len(df) < 14:
            return 0.0
        
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        return atr / df['close'].iloc[-1]  # Normalized ATR
    
    def _calculate_momentum_score(self, trends: List[str]) -> float:
        """Calculate momentum score based on trend alignment"""
        if not trends:
            return 0.0
        
        score = 0.0
        weights = [0.1, 0.3, 0.4, 0.2]  # M1, M5, M15, M30 weights
        
        for i, trend in enumerate(trends[:len(weights)]):
            if trend == "UP":
                score += weights[i]
            elif trend == "DOWN":
                score -= weights[i]
        
        return score

class MarketConditionDetector:
    """Detects current market conditions for adaptive strategy selection"""
    
    def analyze_market_condition(self, data: List[Dict], mtf_analysis: Dict) -> MarketAnalysis:
        """Analyze current market conditions"""
        df = pd.DataFrame(data)
        
        if len(df) < 50:
            return MarketAnalysis(
                condition=MarketCondition.RANGING,
                volatility=0.0,
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                volume_profile="normal",
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate key metrics
        volatility = self._calculate_volatility_score(df)
        trend_strength = self._calculate_trend_strength(df)
        support_resistance = self._find_support_resistance(df)
        volume_profile = self._analyze_volume(df)
        
        # Determine market condition
        condition = self._determine_condition(volatility, trend_strength, mtf_analysis)
        
        return MarketAnalysis(
            condition=condition,
            volatility=volatility,
            trend_strength=trend_strength,
            support_levels=support_resistance['support'],
            resistance_levels=support_resistance['resistance'],
            volume_profile=volume_profile,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate normalized volatility score"""
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].std() * np.sqrt(252 * 24 * 12)  # Annualized for 5-min bars
        return min(volatility * 100, 1.0)  # Normalize to 0-1
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX concept"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate directional movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(abs(high[1:] - close[:-1]), 
                                 abs(low[1:] - close[:-1])))
        
        # Smooth the values
        period = 14
        if len(plus_dm) >= period:
            smoothed_plus = pd.Series(plus_dm).rolling(period).mean().iloc[-1]
            smoothed_minus = pd.Series(minus_dm).rolling(period).mean().iloc[-1]
            smoothed_tr = pd.Series(tr).rolling(period).mean().iloc[-1]
            
            if smoothed_tr > 0:
                plus_di = smoothed_plus / smoothed_tr
                minus_di = smoothed_minus / smoothed_tr
                
                if plus_di + minus_di > 0:
                    dx = abs(plus_di - minus_di) / (plus_di + minus_di)
                    return min(dx, 1.0)
        
        return 0.0
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find key support and resistance levels"""
        levels = {'support': [], 'resistance': []}
        
        # Use recent highs and lows
        recent_data = df.tail(20)
        current_price = df['close'].iloc[-1]
        
        # Find local minima and maxima
        for i in range(2, len(recent_data) - 2):
            # Local maximum (resistance)
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > current_price):
                levels['resistance'].append(recent_data['high'].iloc[i])
            
            # Local minimum (support)
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < current_price):
                levels['support'].append(recent_data['low'].iloc[i])
        
        # Sort and limit levels
        levels['support'] = sorted(levels['support'], reverse=True)[:3]
        levels['resistance'] = sorted(levels['resistance'])[:3]
        
        return levels
    
    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume profile"""
        recent_volume = df['tick_volume'].tail(10).mean()
        historical_volume = df['tick_volume'].mean()
        
        if recent_volume > historical_volume * 1.5:
            return "high"
        elif recent_volume < historical_volume * 0.5:
            return "low"
        else:
            return "normal"
    
    def _determine_condition(self, volatility: float, trend_strength: float, 
                           mtf_analysis: Dict) -> MarketCondition:
        """Determine overall market condition"""
        if volatility > 0.7:
            return MarketCondition.VOLATILE
        elif volatility < 0.3:
            return MarketCondition.LOW_VOLATILITY
        elif trend_strength > 0.6 and mtf_analysis.get("trend_alignment"):
            direction = mtf_analysis.get("aligned_direction", "")
            if direction == "UP":
                return MarketCondition.TRENDING_UP
            elif direction == "DOWN":
                return MarketCondition.TRENDING_DOWN
        
        return MarketCondition.RANGING

class RiskManager:
    """Enhanced risk management with dynamic position sizing and exposure control"""
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_daily_loss: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.daily_trades = deque()
        self.daily_pnl = 0.0
        self.max_concurrent_trades = 3
        self.correlation_threshold = 0.7
        
    def calculate_position_size(self, account_balance: float, stop_loss_pips: float, 
                              market_condition: MarketCondition, confidence: float) -> float:
        """Calculate dynamic position size based on risk and market conditions"""
        base_lot_size = 0.01
        
        # Adjust for market conditions
        condition_multipliers = {
            MarketCondition.TRENDING_UP: 1.2,
            MarketCondition.TRENDING_DOWN: 1.2,
            MarketCondition.RANGING: 0.8,
            MarketCondition.VOLATILE: 0.5,
            MarketCondition.LOW_VOLATILITY: 1.0
        }
        
        multiplier = condition_multipliers.get(market_condition, 1.0)
        
        # Adjust for confidence
        if confidence > 0.8:
            multiplier *= 1.1
        elif confidence < 0.6:
            multiplier *= 0.8
        
        # Calculate risk-based position size
        risk_amount = account_balance * self.max_risk_per_trade
        pip_value = 10  # Approximate for major pairs with 0.01 lot
        
        if stop_loss_pips > 0:
            calculated_lots = (risk_amount / (stop_loss_pips * pip_value)) * multiplier
            return min(max(base_lot_size, round(calculated_lots, 2)), 0.05)  # Cap at 0.05 lots
        
        return base_lot_size
    
    def check_trade_allowed(self, active_trades: Dict, account_info: Dict) -> Tuple[bool, str]:
        """Check if new trade is allowed based on risk rules"""
        # Check concurrent trades
        if len(active_trades) >= self.max_concurrent_trades:
            return False, "Maximum concurrent trades reached"
        
        # Check daily loss limit
        if self.daily_pnl < -account_info['balance'] * self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check margin
        if account_info['margin_free'] < account_info['balance'] * 0.2:
            return False, "Insufficient free margin"
        
        return True, "Trade allowed"
    
    def update_daily_stats(self, trade: EnhancedTrade):
        """Update daily trading statistics"""
        if trade.profit is not None:
            self.daily_pnl += trade.profit
            self.daily_trades.append(trade)
            
            # Remove trades older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            while self.daily_trades and datetime.fromisoformat(self.daily_trades[0].entry_time) < cutoff_time:
                old_trade = self.daily_trades.popleft()
                if old_trade.profit:
                    self.daily_pnl -= old_trade.profit

class EnhancedLiveTradingEngine:
    """Enhanced live trading engine with advanced features"""
    
    def __init__(self, api_base="http://172.28.144.1:8000"):
        self.api_base = api_base
        self.strategy_ensemble = ImprovedStrategyEnsemble()
        self.mtf_analyzer = MultiTimeframeAnalyzer(api_base)
        self.market_detector = MarketConditionDetector()
        self.risk_manager = RiskManager()
        
        # Trading parameters
        self.symbols = []  # Will be populated based on market scan
        self.active_trades = {}
        self.trade_history = []
        self.running = False
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # Market scan parameters
        self.min_spread_ratio = 0.002  # Max 0.2% spread
        self.min_volatility = 0.0005  # Min 0.05% volatility
        # Use actual tradable symbols from API
        self.preferred_pairs = ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#", 
                               "GBPJPY#", "USDCAD#", "USDCHF#", "EURGBP#", "NZDUSD#"]
        
    def scan_best_symbols(self) -> List[str]:
        """Scan and select best symbols for trading"""
        logger.info("🔍 Scanning market for best trading opportunities...")
        
        try:
            response = requests.get(f"{self.api_base}/market/symbols")
            if response.status_code != 200:
                logger.error("Failed to get symbols list")
                return ["USDJPY#"]  # Fallback
            
            all_symbols = response.json()
            tradeable_symbols = []
            
            for symbol in all_symbols:
                if "#" in symbol and symbol in self.preferred_pairs:
                    # Get symbol info
                    info_response = requests.get(f"{self.api_base}/market/symbols/{symbol}")
                    if info_response.status_code == 200:
                        symbol_info = info_response.json()
                        
                        # Check spread
                        if 'ask' in symbol_info and 'bid' in symbol_info:
                            spread = symbol_info['ask'] - symbol_info['bid']
                            spread_ratio = spread / symbol_info['bid']
                            
                            if spread_ratio <= self.min_spread_ratio:
                                # Check volatility
                                data = self.get_market_data(symbol, "M5", 50)
                                if data:
                                    df = pd.DataFrame(data)
                                    volatility = df['close'].pct_change().std()
                                    
                                    if volatility >= self.min_volatility:
                                        tradeable_symbols.append({
                                            'symbol': symbol,
                                            'spread_ratio': spread_ratio,
                                            'volatility': volatility,
                                            'score': volatility / spread_ratio  # Higher is better
                                        })
            
            # Sort by score and select top symbols
            tradeable_symbols.sort(key=lambda x: x['score'], reverse=True)
            selected = [s['symbol'] for s in tradeable_symbols[:5]]  # Top 5 symbols
            
            logger.info(f"✅ Selected symbols for trading: {selected}")
            return selected if selected else ["USDJPY#"]
            
        except Exception as e:
            logger.error(f"Error scanning symbols: {e}")
            return ["USDJPY#"]
    
    def check_api_connection(self):
        """Verify API connection and MT5 status"""
        try:
            response = requests.get(f"{self.api_base}/status/mt5")
            if response.status_code == 200:
                status = response.json()
                if status.get('connected') and status.get('trade_allowed'):
                    logger.info("✅ MT5 API connected and trading allowed")
                    return True
                else:
                    logger.error("❌ MT5 not ready for trading")
                    return False
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
    
    def get_account_info(self):
        """Get current account information"""
        try:
            response = requests.get(f"{self.api_base}/account/")
            if response.status_code == 200:
                account = response.json()
                logger.info(f"Account Balance: {account['balance']} {account['currency']}")
                logger.info(f"Free Margin: {account['margin_free']} {account['currency']}")
                logger.info(f"Current P/L: {account.get('profit', 0)} {account['currency']}")
                return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
        return None
    
    def get_market_data(self, symbol: str, timeframe: str = "M5", count: int = 100):
        """Get current market data for analysis"""
        url = f"{self.api_base}/market/history"
        data = {"symbol": symbol, "timeframe": timeframe, "count": count}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
        return None
    
    def analyze_and_trade(self, symbol: str):
        """Enhanced analysis with multi-timeframe and market condition detection"""
        # Check if we can place new trades
        account = self.get_account_info()
        if not account:
            return
        
        trade_allowed, reason = self.risk_manager.check_trade_allowed(self.active_trades, account)
        if not trade_allowed:
            logger.debug(f"Trade not allowed for {symbol}: {reason}")
            return
        
        # Get multi-timeframe analysis
        mtf_analysis = self.mtf_analyzer.get_mtf_analysis(symbol)
        
        # Get primary timeframe data
        market_data = self.get_market_data(symbol, "M5", 100)
        if not market_data:
            return
        
        # Detect market condition
        market_condition = self.market_detector.analyze_market_condition(market_data, mtf_analysis)
        
        # Skip if market is too volatile or low volatility
        if market_condition.condition in [MarketCondition.VOLATILE, MarketCondition.LOW_VOLATILITY]:
            logger.debug(f"Skipping {symbol} - unfavorable market condition: {market_condition.condition.value}")
            return
        
        try:
            # Get trading signal with market condition awareness
            signal = self.strategy_ensemble.get_ensemble_signal(market_data)
            
            if signal and signal.confidence >= 0.75:  # Higher threshold for live trading
                # Additional confirmation from MTF analysis
                if mtf_analysis['trend_alignment']:
                    aligned_dir = mtf_analysis.get('aligned_direction', '')
                    signal_dir = 'UP' if signal.signal == SignalType.BUY else 'DOWN'
                    
                    if aligned_dir != signal_dir:
                        logger.info(f"Signal rejected - MTF trend misalignment for {symbol}")
                        return
                
                # Calculate dynamic position size
                stop_loss_pips = abs(signal.entry_price - signal.stop_loss) / 0.0001
                position_size = self.risk_manager.calculate_position_size(
                    account['balance'], stop_loss_pips, 
                    market_condition.condition, signal.confidence
                )
                
                logger.info(f"🎯 Strong signal detected: {symbol} {signal.signal.value} "
                           f"(confidence: {signal.confidence:.2f}, position: {position_size} lots)")
                logger.info(f"Market condition: {market_condition.condition.value}, "
                           f"Volatility: {market_condition.volatility:.2f}")
                
                # Place the trade
                trade = self.place_order(signal, symbol, position_size, market_condition)
                if trade:
                    time.sleep(1)  # Brief pause between operations
                    
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
    
    def place_order(self, signal: TradingSignal, symbol: str, position_size: float, 
                    market_condition: MarketAnalysis) -> Optional[EnhancedTrade]:
        """Place a market order with enhanced parameters"""
        if signal.signal == SignalType.BUY:
            order_type = 0  # BUY
        else:
            order_type = 1  # SELL
        
        order_data = {
            "action": 1,  # DEAL (market order)
            "symbol": symbol,
            "volume": position_size,
            "type": order_type,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "comment": f"Enhanced AI - {market_condition.condition.value[:10]}"
        }
        
        try:
            response = requests.post(f"{self.api_base}/trading/orders", json=order_data)
            if response.status_code == 201:
                result = response.json()
                logger.info(f"✅ Order placed successfully: {result}")
                
                # Create enhanced trade record
                trade = EnhancedTrade(
                    ticket=result.get('order', 0),
                    symbol=symbol,
                    signal_type=signal.signal.value,
                    entry_time=datetime.now().isoformat(),
                    entry_price=result.get('price', signal.entry_price),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    volume=position_size,
                    market_condition=market_condition.condition,
                    confidence=signal.confidence,
                    strategy_names=signal.reason.split(',')[:1]
                )
                
                self.active_trades[trade.ticket] = trade
                self.performance_stats['total_trades'] += 1
                
                logger.info(f"📊 Trade opened: {symbol} {signal.signal.value} at {trade.entry_price} "
                           f"(SL: {trade.stop_loss}, TP: {trade.take_profit})")
                return trade
            else:
                logger.error(f"❌ Order failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
        return None
    
    def check_open_positions(self):
        """Check and update status of open positions with trailing stop"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                positions = response.json()
                current_tickets = {pos['ticket'] for pos in positions}
                
                # Check for closed trades
                for ticket, trade in list(self.active_trades.items()):
                    if ticket not in current_tickets:
                        # Trade was closed
                        trade.status = "CLOSED"
                        trade.exit_time = datetime.now().isoformat()
                        
                        # Calculate trade duration
                        entry_time = datetime.fromisoformat(trade.entry_time)
                        exit_time = datetime.fromisoformat(trade.exit_time)
                        trade.trade_duration = int((exit_time - entry_time).total_seconds() / 60)
                        
                        self.trade_history.append(trade)
                        del self.active_trades[ticket]
                        
                        # Update performance stats
                        if trade.profit and trade.profit > 0:
                            self.performance_stats['winning_trades'] += 1
                            self.performance_stats['consecutive_losses'] = 0
                        else:
                            self.performance_stats['consecutive_losses'] += 1
                        
                        if trade.profit:
                            self.performance_stats['total_profit'] += trade.profit
                            self.risk_manager.update_daily_stats(trade)
                        
                        logger.info(f"🔄 Trade {ticket} closed - P/L: {trade.profit:.2f} "
                                   f"Duration: {trade.trade_duration} minutes")
                
                # Update profit and implement trailing stop for open trades
                for position in positions:
                    ticket = position['ticket']
                    if ticket in self.active_trades:
                        trade = self.active_trades[ticket]
                        current_profit = position.get('profit', 0)
                        trade.profit = current_profit
                        
                        # Update max drawdown
                        if current_profit < 0:
                            trade.max_drawdown = min(trade.max_drawdown, current_profit)
                        
                        # Implement trailing stop for profitable trades
                        if current_profit > 20:  # If profit > $20, start trailing
                            current_price = position.get('price_current', trade.entry_price)
                            self._update_trailing_stop(trade, current_price, position)
                
                logger.debug(f"📈 Active trades: {len(self.active_trades)}")
                return positions
        except Exception as e:
            logger.error(f"Failed to check positions: {e}")
        return []
    
    def _update_trailing_stop(self, trade: EnhancedTrade, current_price: float, position: Dict):
        """Implement trailing stop logic"""
        pip_value = 0.0001 if "JPY" not in trade.symbol else 0.01
        
        if trade.signal_type == "BUY":
            # For buy trades, move stop loss up
            new_sl = current_price - (20 * pip_value)  # Trail by 20 pips
            if new_sl > trade.stop_loss:
                self._modify_position(position['ticket'], new_sl, trade.take_profit)
                trade.stop_loss = new_sl
                logger.info(f"📈 Trailing stop updated for {trade.symbol}: {new_sl}")
        else:
            # For sell trades, move stop loss down
            new_sl = current_price + (20 * pip_value)
            if new_sl < trade.stop_loss:
                self._modify_position(position['ticket'], new_sl, trade.take_profit)
                trade.stop_loss = new_sl
                logger.info(f"📈 Trailing stop updated for {trade.symbol}: {new_sl}")
    
    def _modify_position(self, ticket: int, sl: float, tp: float):
        """Modify an existing position"""
        modify_data = {
            "sl": sl,
            "tp": tp
        }
        
        try:
            response = requests.patch(f"{self.api_base}/trading/positions/{ticket}", json=modify_data)
            if response.status_code == 200:
                logger.debug(f"Position {ticket} modified successfully")
            else:
                logger.error(f"Failed to modify position {ticket}: {response.text}")
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
    
    def run_trading_session(self, duration_minutes=300):
        """Run an enhanced live trading session"""
        if not self.check_api_connection():
            logger.error("Cannot start trading - API connection failed")
            return
        
        account = self.get_account_info()
        if not account:
            logger.error("Cannot start trading - account info unavailable")
            return
        
        # Scan for best symbols
        self.symbols = self.scan_best_symbols()
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        logger.info(f"🚀 Starting enhanced live trading session for {duration_minutes} minutes")
        logger.info(f"Trading symbols: {self.symbols}")
        logger.info(f"Risk per trade: {self.risk_manager.max_risk_per_trade*100}%")
        logger.info(f"Max daily loss: {self.risk_manager.max_daily_loss*100}%")
        
        try:
            cycle_count = 0
            while self.running and datetime.now() < end_time:
                cycle_count += 1
                
                # Check existing positions
                self.check_open_positions()
                
                # Re-scan symbols every 30 cycles (approximately every 15 minutes)
                if cycle_count % 30 == 0:
                    self.symbols = self.scan_best_symbols()
                
                # Analyze each symbol
                for symbol in self.symbols:
                    self.analyze_and_trade(symbol)
                    time.sleep(0.5)  # Small delay between symbols
                
                # Wait before next analysis cycle
                time.sleep(30)  # 30-second intervals for better opportunity catching
                
                # Log status every 5 minutes
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    self._log_session_status(elapsed, duration_minutes)
                
                # Check for stop conditions
                if self.performance_stats['consecutive_losses'] >= 3:
                    logger.warning("⚠️ 3 consecutive losses - pausing trading for 30 minutes")
                    time.sleep(1800)  # 30 minute pause
                    self.performance_stats['consecutive_losses'] = 0
                    
        except KeyboardInterrupt:
            logger.info("🛑 Trading session interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Trading session error: {e}")
            self.running = False
        
        # Final status check
        self.check_open_positions()
        
        logger.info("📊 Enhanced trading session completed")
        self.print_session_summary()
    
    def _log_session_status(self, elapsed: float, total: float):
        """Log current session status"""
        win_rate = (self.performance_stats['winning_trades'] / 
                   max(self.performance_stats['total_trades'], 1)) * 100
        
        logger.info(f"⏰ Session progress: {elapsed:.1f}/{total} minutes | "
                   f"Trades: {self.performance_stats['total_trades']} | "
                   f"Win rate: {win_rate:.1f}% | "
                   f"P/L: ${self.performance_stats['total_profit']:.2f}")
    
    def print_session_summary(self):
        """Print enhanced trading session summary"""
        print("\n" + "="*70)
        print("ENHANCED LIVE TRADING SESSION SUMMARY")
        print("="*70)
        
        total_trades = len(self.trade_history) + len(self.active_trades)
        completed_trades = len(self.trade_history)
        
        if completed_trades > 0:
            profits = [trade.profit for trade in self.trade_history if trade.profit is not None]
            total_profit = sum(profits) if profits else 0
            winning_trades = len([p for p in profits if p > 0])
            win_rate = (winning_trades / completed_trades) * 100
            
            # Calculate average trade metrics
            avg_win = np.mean([p for p in profits if p > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([p for p in profits if p < 0]) if (completed_trades - winning_trades) > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum([p for p in profits if p > 0])
            gross_loss = abs(sum([p for p in profits if p < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Trade duration analysis
            durations = [t.trade_duration for t in self.trade_history if t.trade_duration]
            avg_duration = np.mean(durations) if durations else 0
            
            print(f"Total Trades: {total_trades}")
            print(f"Completed Trades: {completed_trades}")
            print(f"Active Trades: {len(self.active_trades)}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total P/L: ${total_profit:.2f}")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${abs(avg_loss):.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Average Trade Duration: {avg_duration:.1f} minutes")
            print(f"Best Trade: ${max(profits):.2f}" if profits else "Best Trade: N/A")
            print(f"Worst Trade: ${min(profits):.2f}" if profits else "Worst Trade: N/A")
            
            # Market condition analysis
            conditions = [t.market_condition for t in self.trade_history if t.market_condition]
            if conditions:
                print("\nPerformance by Market Condition:")
                for condition in set(conditions):
                    condition_trades = [t for t in self.trade_history if t.market_condition == condition]
                    condition_profit = sum([t.profit for t in condition_trades if t.profit])
                    print(f"  {condition.value}: {len(condition_trades)} trades, P/L: ${condition_profit:.2f}")
        else:
            print("No completed trades in this session")
        
        print("="*70)
    
    def stop_trading(self):
        """Gracefully stop trading"""
        self.running = False
        logger.info("🛑 Enhanced trading engine stopped")

def main():
    """Main execution function"""
    engine = EnhancedLiveTradingEngine()
    
    logger.info("🎯 Starting enhanced automated forex trading system")
    logger.info("Features: Multi-timeframe analysis, Market condition detection, Dynamic position sizing")
    logger.info("Risk Management: Trailing stops, Daily loss limits, Consecutive loss protection")
    
    try:
        # Run a 5-hour trading session
        engine.run_trading_session(duration_minutes=300)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    logger.info("🏁 Enhanced trading operation completed")

if __name__ == "__main__":
    main()