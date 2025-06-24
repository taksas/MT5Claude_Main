#!/usr/bin/env python3
"""
Trading Components Module
Merged file containing all trading system components under 10KB
"""

# Standard library imports
import logging
import time
import requests
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote

# =============================================================================
# TRADING MODELS
# =============================================================================

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Signal:
    symbol: str
    type: SignalType
    confidence: float
    entry: float
    sl: float
    tp: float
    reason: str
    strategies: Optional[Dict[str, float]] = None
    quality: float = 0.0

@dataclass 
class Trade:
    ticket: int
    symbol: str
    type: str
    entry_price: float
    sl: float
    tp: float
    volume: float
    entry_time: datetime

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Tradable symbols only - exact list from MT5 API
HIGH_PROFIT_SYMBOLS = {
    "CADJPY": {"priority": 1, "diversification_weight": 1.0},
    "CHFJPY": {"priority": 1, "diversification_weight": 1.0},
    "EURCAD": {"priority": 2, "diversification_weight": 1.2},
    "EURCHF": {"priority": 2, "diversification_weight": 1.1},
    "EURGBP": {"priority": 2, "diversification_weight": 1.1},
    "EURJPY": {"priority": 1, "diversification_weight": 1.0},
    "CADCHF": {"priority": 3, "diversification_weight": 1.3},
    "EURUSD": {"priority": 1, "diversification_weight": 1.0},
    "USDJPY": {"priority": 1, "diversification_weight": 1.0},
    "GBPCAD": {"priority": 2, "diversification_weight": 1.2},
    "GBPCHF": {"priority": 3, "diversification_weight": 1.3},
    "GBPJPY": {"priority": 1, "diversification_weight": 1.0},
    "GBPUSD": {"priority": 1, "diversification_weight": 1.0},
    "USDCAD": {"priority": 2, "diversification_weight": 1.1},
    "USDCHF": {"priority": 2, "diversification_weight": 1.1}
}

# Symbol-specific overrides (only when different from defaults)
SYMBOL_OVERRIDES = {
    "GBPJPY": {"typical_spread": 2, "target_rr_ratio": 3.0, "diversification_weight": 1.0},
    "GBPUSD": {"target_rr_ratio": 2.5, "diversification_weight": 1.0},
    "EURCAD": {"typical_spread": 3, "diversification_weight": 1.2},
    "GBPCAD": {"typical_spread": 3, "target_rr_ratio": 2.5, "diversification_weight": 1.2},
    "GBPCHF": {"typical_spread": 3, "target_rr_ratio": 2.2, "diversification_weight": 1.3},
    "CADCHF": {"typical_spread": 2, "diversification_weight": 1.3}
}

# ULTRA Aggressive Configuration - REVISED FOR SAFETY
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.30,
    "MAX_SPREAD_PIPS": 3.0,
    
    # --- 新しいリスク管理設定 ---
    # 口座全体で同時に負うことのできる最大リスクの割合 (例: 5%)
    "MAX_TOTAL_RISK": 0.05, 
    # 1トレードあたりの最大リスク (上記のMAX_TOTAL_RISKが優先される場合のフォールバック値)
    "RISK_PER_TRADE": 0.01, # 安全なデフォルト値 (1%) に変更
    # --- ここまで ---
    
    "MAX_CONCURRENT": 5,
    "MIN_RR_RATIO": 0.8,
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY",
    "MIN_VOLUME": 0.01,
    "POSITION_INTERVAL": 600,
    "FORCE_TRADE_INTERVAL": 120,
    "MIN_SL_DISTANCE_PERCENT": 0.0003,
    "MAX_SL_DISTANCE_PERCENT": 0.02,
    "AGGRESSIVE_MODE": True,
    "IGNORE_SPREAD": False,
    "SYMBOL_FILTER": "ALL",
    "MAX_SYMBOLS": 15,
    "DIVERSIFICATION_MODE": True,
    "PROACTIVE_POSITION_SEEKING": True
}

def get_symbol_config(symbol):
    """Get configuration for a specific symbol with defaults and diversification settings"""
    symbol_base = symbol.rstrip('#')
    
    # Default configuration
    config = {
        "typical_spread": 1,
        "min_rr_ratio": 1.5,
        "target_rr_ratio": 2.0,
        "max_rr_ratio": 3.0,
        "diversification_weight": 1.0,
        "priority": 2
    }
    
    # Apply symbol-specific overrides if they exist
    if symbol_base in SYMBOL_OVERRIDES:
        config.update(SYMBOL_OVERRIDES[symbol_base])
    
    # Apply diversification settings from HIGH_PROFIT_SYMBOLS
    if symbol_base in HIGH_PROFIT_SYMBOLS:
        config.update(HIGH_PROFIT_SYMBOLS[symbol_base])
        
    return config

# =============================================================================
# SYMBOL UTILITIES
# =============================================================================

class SymbolUtils:
    def __init__(self):
        # Only classify based on tradable symbols - all are forex majors/crosses
        # No exotic, metal, crypto, or index symbols in tradable list
        pass
    
    def get_instrument_type(self, symbol: str) -> str:
        """Get instrument type - all tradable symbols are forex majors/crosses"""
        # All tradable symbols are forex pairs, classify as major or cross
        symbol_clean = symbol.rstrip('#').upper()
        
        # Major USD pairs
        if symbol_clean in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'USDCHF']:
            return 'major'
        # All others are cross pairs
        else:
            return 'cross'
    
    def get_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol"""
        symbol_clean = symbol.rstrip('#').upper()
        if len(symbol_clean) >= 6:
            return symbol_clean[:3]
        return ""
    
    def get_quote_currency(self, symbol: str) -> str:
        """Extract quote currency from symbol"""
        symbol_clean = symbol.rstrip('#').upper()
        if len(symbol_clean) >= 6:
            return symbol_clean[3:6]
        return ""

# =============================================================================
# MT5 API CLIENT
# =============================================================================

logger = logging.getLogger('MT5APIClient')

class MT5APIClient:
    def __init__(self, api_base: str):
        self.api_base = api_base
        
    def check_connection(self) -> bool:
        """Check API connection"""
        try:
            response = requests.get(f"{self.api_base}/status/mt5", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_balance(self) -> Optional[float]:
        """Get account balance"""
        try:
            response = requests.get(f"{self.api_base}/account/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('balance', 0)
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get full account information"""
        try:
            response = requests.get(f"{self.api_base}/account/", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def discover_symbols(self) -> List[str]:
        """Discover all tradable symbols"""
        try:
            response = requests.get(f"{self.api_base}/market/symbols/tradable", timeout=10)
            if response.status_code == 200:
                # Response is a list of symbol names
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        try:
            # URL encode the symbol (# becomes %23)
            encoded_symbol = quote(symbol)
            response = requests.get(f"{self.api_base}/market/symbols/{encoded_symbol}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current price for a symbol"""
        try:
            # Get price from symbol info
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info and 'bid' in symbol_info and 'ask' in symbol_info:
                return {
                    'bid': symbol_info['bid'],
                    'ask': symbol_info['ask']
                }
            
            # Fallback: get latest candle
            response = requests.post(
                f"{self.api_base}/market/history",
                json={"symbol": symbol, "timeframe": "M1", "count": 1},
                timeout=5
            )
            if response.status_code == 200:
                candles = response.json()
                if candles and len(candles) > 0:
                    close_price = candles[0]['close']
                    return {
                        'bid': close_price,
                        'ask': close_price  # Using close as both bid/ask
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_market_history(self, symbol: str, timeframe: str, count: int = 100) -> Optional[Dict[str, Any]]:
        """Get historical market data"""
        try:
            response = requests.post(
                f"{self.api_base}/market/history",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": count
                },
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def place_order(self, order: Dict[str, Any]) -> Optional[int]:
        """Place trading order"""
        try:
            response = requests.post(
                f"{self.api_base}/trading/orders",
                json=order,
                timeout=10
            )
            if response.status_code in [200, 201]:
                data = response.json()
                # The API returns 'order' field, not 'ticket'
                return data.get('order') or data.get('ticket')
            else:
                logger.error(f"Order failed: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # API returns list directly
                if isinstance(data, list):
                    return data
                # Fallback if it's a dict with positions key
                elif isinstance(data, dict):
                    return data.get('positions', [])
                return []
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        try:
            # According to API docs, DELETE requires a JSON body with deviation parameter
            response = requests.delete(
                f"{self.api_base}/trading/positions/{ticket}",
                json={"deviation": 20},  # Allow 20 points deviation
                timeout=5
            )
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to close position {ticket}: {e}")
            return False
    
    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify position stop loss and take profit"""
        try:
            response = requests.patch(
                f"{self.api_base}/trading/positions/{ticket}",
                json={"sl": sl, "tp": tp},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to modify position {ticket}: {e}")
            return False

# =============================================================================
# MARKET DATA
# =============================================================================

logger = logging.getLogger('MarketData')

class MarketData:
    def __init__(self, api_client: MT5APIClient):
        self.api_client = api_client
        self.symbol_utils = SymbolUtils()
        
        # Cache management
        self.spread_cache = {}
        self.data_cache = {}
        self.cache_ttl = {
            'spread': 10,  # 10 seconds for spread cache
            'data': 30     # 30 seconds for market data cache
        }
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid price for a symbol"""
        try:
            price_data = self.api_client.get_current_price(symbol)
            if price_data:
                return price_data.get('bid')
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with caching"""
        cache_key = f"{symbol}_data_{count}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl['data']:
                return cached_data
        
        try:
            data = self.api_client.get_market_history(symbol, CONFIG["TIMEFRAME"], count)
            if data:
                # API returns list directly, not dict with candles
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
                
                # Cache the data
                self.data_cache[cache_key] = (df, time.time())
                return df
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def check_spread(self, symbol: str) -> Tuple[bool, float]:
        """Check if spread is acceptable for trading"""
        # Check cache
        if symbol in self.spread_cache:
            cached_spread, timestamp = self.spread_cache[symbol]
            if time.time() - timestamp < self.cache_ttl['spread']:
                max_spread = self._get_max_spread(symbol)
                return cached_spread <= max_spread, cached_spread
        
        try:
            symbol_info = self.api_client.get_symbol_info(symbol)
            if not symbol_info:
                return False, 999.0
            
            spread = symbol_info.get('spread', 999)
            
            # Convert spread to pips based on instrument type
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            if instrument_type == 'metal':
                # Metals are quoted differently
                spread_pips = spread / 10.0 if 'XAU' in symbol else spread
            elif instrument_type == 'crypto':
                # Crypto spreads are usually in points
                spread_pips = spread
            elif instrument_type == 'index':
                # Index spreads are in points
                spread_pips = spread
            else:
                # Forex - convert to pips
                digits = symbol_info.get('digits', 5)
                if digits == 3 or digits == 5:
                    spread_pips = spread / 10.0
                else:
                    spread_pips = spread
            
            # Cache the spread
            self.spread_cache[symbol] = (spread_pips, time.time())
            
            # Check against maximum allowed spread
            max_spread = self._get_max_spread(symbol)
            return spread_pips <= max_spread, spread_pips
            
        except Exception as e:
            logger.error(f"Error checking spread for {symbol}: {e}")
            return False, 999.0
    
    def _get_max_spread(self, symbol: str) -> float:
        """Get maximum allowed spread for a symbol"""
        # Check if symbol has specific configuration
        symbol_config = HIGH_PROFIT_SYMBOLS.get(symbol, {})
        if 'typical_spread' in symbol_config:
            # Use 1.5x typical spread as maximum
            return symbol_config['typical_spread'] * 1.5
        
        # Otherwise use instrument type defaults
        instrument_type = self.symbol_utils.get_instrument_type(symbol)
        
        if instrument_type == 'exotic':
            return CONFIG["MAX_SPREAD_EXOTIC"]
        elif instrument_type == 'metal':
            return CONFIG["MAX_SPREAD_METAL"]
        elif instrument_type == 'crypto':
            return CONFIG["MAX_SPREAD_CRYPTO"]
        elif instrument_type == 'index':
            return CONFIG["MAX_SPREAD_INDEX"]
        else:
            return CONFIG["MAX_SPREAD_PIPS"]
    
    def clear_cache(self):
        """Clear all cached data"""
        self.spread_cache.clear()
        self.data_cache.clear()
        logger.debug("Market data cache cleared")

# =============================================================================
# ORDER MANAGEMENT
# =============================================================================

logger = logging.getLogger('OrderManagement')

class OrderManagement:
    def __init__(self, api_client: MT5APIClient):
        self.api_client = api_client
        self.symbol_utils = SymbolUtils()
        
    def place_order(self, signal: Signal, symbol: str, volume: float) -> Optional[int]:
        """Place trading order"""
        try:
            # Prepare order according to API specification
            order = {
                "action": 1,  # TRADE_ACTION_DEAL (market order)
                "symbol": symbol,
                "volume": volume,
                "type": 0 if signal.type == SignalType.BUY else 1,  # ORDER_TYPE_BUY or ORDER_TYPE_SELL
                "sl": signal.sl,
                "tp": signal.tp,
                "comment": f"Ultra100_{signal.reason[:20]}",
                "deviation": 20,  # Allow 20 points deviation
                "magic": 100100  # Magic number for identification
            }
            
            logger.info(f"📊 Placing {signal.type.value} order for {symbol}")
            logger.info(f"   Volume: {volume}, Entry: {signal.entry}, SL: {signal.sl}, TP: {signal.tp}")
            
            ticket = self.api_client.place_order(order)
            
            if ticket:
                logger.info(f"✅ Order placed successfully! Ticket: {ticket}")
                return ticket
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def manage_positions(self, active_trades: Dict[str, Trade]) -> Dict[str, Any]:
        """Manage open positions"""
        results = {
            'closed': [],
            'modified': [],
            'errors': []
        }
        
        try:
            positions = self.api_client.get_positions()
            
            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol')
                
                if ticket not in active_trades:
                    continue
                
                trade = active_trades[ticket]
                current_price = position.get('price_current')
                profit = position.get('profit', 0)
                
                # Check if position should be managed
                if self._should_move_to_breakeven(trade, current_price, profit):
                    if self._move_breakeven(ticket, trade, current_price):
                        results['modified'].append(ticket)
                        logger.info(f"🎯 Moved position {ticket} to breakeven")
                
                # Check for early exit conditions
                if self._should_close_early(trade, current_price, profit):
                    if self.close_position(ticket):
                        results['closed'].append(ticket)
                        logger.info(f"🔒 Closed position {ticket} early")
                        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        try:
            success = self.api_client.close_position(ticket)
            if success:
                logger.info(f"✅ Position {ticket} closed successfully")
            else:
                logger.error(f"❌ Failed to close position {ticket}")
            return success
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def _move_breakeven(self, ticket: int, trade: Trade, current_price: float) -> bool:
        """Move stop loss to breakeven"""
        try:
            # Calculate new stop loss (entry + small buffer for costs)
            spread_buffer = abs(trade.entry_price * 0.0001)  # 1 pip buffer
            
            if trade.type == "BUY":
                new_sl = trade.entry_price + spread_buffer
                if new_sl >= current_price:  # Don't set SL above current price
                    return False
            else:
                new_sl = trade.entry_price - spread_buffer
                if new_sl <= current_price:  # Don't set SL below current price
                    return False
            
            # Only move if new SL is better than current
            if trade.type == "BUY" and new_sl <= trade.sl:
                return False
            elif trade.type == "SELL" and new_sl >= trade.sl:
                return False
            
            return self.api_client.modify_position(ticket, new_sl, trade.tp)
            
        except Exception as e:
            logger.error(f"Error moving to breakeven for {ticket}: {e}")
            return False
    
    def _should_move_to_breakeven(self, trade: Trade, current_price: float, 
                                 profit: float) -> bool:
        """Check if position should be moved to breakeven"""
        try:
            # Only move to BE if in profit
            if profit <= 0:
                return False
            
            # Check if price has moved enough
            if trade.type == "BUY":
                price_move = current_price - trade.entry_price
                target_move = (trade.tp - trade.entry_price) * 0.5  # 50% to TP
            else:
                price_move = trade.entry_price - current_price
                target_move = (trade.entry_price - trade.tp) * 0.5
            
            # Move to BE if reached 50% of target
            return price_move >= target_move and trade.sl != trade.entry_price
            
        except Exception as e:
            logger.error(f"Error checking breakeven condition: {e}")
            return False
    
    def _should_close_early(self, trade: Trade, current_price: float, 
                           profit: float) -> bool:
        """Check if position should be closed early"""
        try:
            # Close if position has been open too long (4 hours)
            time_open = (datetime.now() - trade.entry_time).total_seconds()
            if time_open > 14400:  # 4 hours
                return profit > 0  # Only close if in profit
            
            # Close if reversal detected
            if trade.type == "BUY":
                # If price drops significantly from high
                high_since_entry = max(current_price, trade.entry_price * 1.01)
                pullback = (high_since_entry - current_price) / high_since_entry
                if pullback > 0.02:  # 2% pullback (was 0.5%)
                    return True
            else:
                # If price rises significantly from low
                low_since_entry = min(current_price, trade.entry_price * 0.99)
                pullback = (current_price - low_since_entry) / low_since_entry
                if pullback > 0.02:  # 2% pullback (was 0.5%)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking early close condition: {e}")
            return False
    
    def get_position_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific position"""
        try:
            positions = self.api_client.get_positions()
            for position in positions:
                if position.get('ticket') == ticket:
                    return position
            return None
        except Exception as e:
            logger.error(f"Error getting position info for {ticket}: {e}")
            return None
    
    def close_all_positions(self) -> int:
        """Close all open positions"""
        closed_count = 0
        try:
            positions = self.api_client.get_positions()
            for position in positions:
                ticket = position.get('ticket')
                if self.close_position(ticket):
                    closed_count += 1
                    time.sleep(0.5)  # Small delay between closures
            
            logger.info(f"Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return closed_count