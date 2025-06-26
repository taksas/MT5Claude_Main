#!/usr/bin/env python3
"""
Market Data Module
Handles market data fetching, caching, and processing
"""

import logging
import time
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from .mt5_api_client import MT5APIClient
from .trading_config import HIGH_PROFIT_SYMBOLS, CONFIG, SymbolUtils

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