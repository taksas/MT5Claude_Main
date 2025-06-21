#!/usr/bin/env python3
"""
Ultra Trading Engine - 100 Signal Deep Analysis System
Implements aggressive trading with comprehensive market analysis
"""

import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pytz
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from urllib.parse import quote
import queue
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import our comprehensive high-profit symbols configuration
try:
    from high_profit_symbols_config import (
        ALL_HIGH_PROFIT_SYMBOLS,
        EXTREME_VOLATILITY_SYMBOLS,
        VERY_HIGH_VOLATILITY_SYMBOLS,
        HIGH_VOLATILITY_SYMBOLS,
        RISK_MANAGEMENT,
        get_symbol_config,
        get_symbols_by_tier,
        should_trade_symbol
    )
    USE_EXTERNAL_CONFIG = True
except ImportError:
    USE_EXTERNAL_CONFIG = False

# Import ultra rare profit configuration
try:
    from ultra_rare_profit_config import (
        ULTRA_PROFIT_SYMBOLS,
        ULTRA_PROFIT_SETTINGS,
        SYMBOL_STRATEGIES,
        EXOTIC_FOREX_ULTRA,
        SYNTHETIC_INDICES,
        RARE_COMMODITIES
    )
    USE_ULTRA_CONFIG = True
except ImportError:
    USE_ULTRA_CONFIG = False
    # Fallback to embedded configuration
    HIGH_PROFIT_SYMBOLS = {
        # Exotic Currency Pairs - Extreme Volatility
        "USDTRY": {"avg_daily_range": 2000, "typical_spread": 50, "risk_factor": 0.005, "profit_potential": "extreme"},
        "EURTRY": {"avg_daily_range": 2500, "typical_spread": 80, "risk_factor": 0.004, "profit_potential": "extreme"},
        "USDZAR": {"avg_daily_range": 1834, "typical_spread": 100, "risk_factor": 0.005, "profit_potential": "very_high"},
        "USDMXN": {"avg_daily_range": 600, "typical_spread": 50, "risk_factor": 0.007, "profit_potential": "high"},
        "EURNOK": {"avg_daily_range": 800, "typical_spread": 30, "risk_factor": 0.008, "profit_potential": "high"},
        "EURSEK": {"avg_daily_range": 600, "typical_spread": 30, "risk_factor": 0.008, "profit_potential": "medium_high"},
        
        # High-Profit Cross Currency Pairs
        "EURGBP": {"avg_daily_range": 60, "typical_spread": 2, "risk_factor": 0.010, "strategy": "range_trading"},
        "EURAUD": {"avg_daily_range": 110, "typical_spread": 3, "risk_factor": 0.009, "strategy": "trend_following"},
        "EURNZD": {"avg_daily_range": 140, "typical_spread": 4, "risk_factor": 0.008, "strategy": "volatility_breakout"},
        "EURJPY": {"avg_daily_range": 100, "typical_spread": 2, "risk_factor": 0.009, "strategy": "risk_sentiment"},
        "GBPJPY": {"avg_daily_range": 150, "typical_spread": 3, "risk_factor": 0.007, "profit_potential": "very_high"},
        "GBPAUD": {"avg_daily_range": 160, "typical_spread": 4, "risk_factor": 0.007, "profit_potential": "very_high"},
        "GBPNZD": {"avg_daily_range": 200, "typical_spread": 5, "risk_factor": 0.006, "profit_potential": "extreme"},
        "AUDJPY": {"avg_daily_range": 80, "typical_spread": 2, "risk_factor": 0.009, "strategy": "carry_trade"},
        "NZDJPY": {"avg_daily_range": 85, "typical_spread": 3, "risk_factor": 0.009, "strategy": "carry_trade"},
        "AUDNZD": {"avg_daily_range": 50, "typical_spread": 3, "risk_factor": 0.010, "strategy": "mean_reversion"},
        
        # Exotic Metals
        "XPDUSD": {"avg_daily_range": 50, "typical_spread": 100, "risk_factor": 0.005, "volatility": "4%_daily"},
        "XPTUSD": {"avg_daily_range": 30, "typical_spread": 50, "risk_factor": 0.007, "volatility": "3%_daily"},
        
        # Commodities
        "NATGAS": {"avg_daily_range": 0.1, "typical_spread": 10, "risk_factor": 0.005, "seasonal": "Dec-Feb,Jun-Aug"},
        "WHEAT": {"avg_daily_range": 10, "typical_spread": 5, "risk_factor": 0.007, "seasonal": "Mar-May,Jul-Sep"},
        "COPPER": {"avg_daily_range": 0.05, "typical_spread": 5, "risk_factor": 0.008, "correlation": "China_data"}
    }

def get_symbol_config(symbol):
    """Get configuration for a specific symbol"""
    if USE_EXTERNAL_CONFIG:
        # Try external config first
        config = ALL_HIGH_PROFIT_SYMBOLS.get(symbol)
        if config:
            return config
    
    # Fallback to embedded config
    return HIGH_PROFIT_SYMBOLS.get(symbol, {
        "avg_daily_range": 100,
        "typical_spread": 2,
        "risk_factor": 0.01,
        "profit_potential": "medium"
    })

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UltraTradingEngine')

# ULTRA Aggressive Configuration - FORCE TRADES
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "SYMBOLS": [],  # Will be populated dynamically
    "TIMEFRAME": "M5",
    "MIN_CONFIDENCE": 0.30,  # 30% minimum confidence
    "MIN_QUALITY": 0.25,     # 25% quality threshold
    "MIN_STRATEGIES": 10,    # At least 10 strategies must agree
    "MAX_SPREAD_PIPS": 3.0,  # Maximum 3 pips spread for majors
    "MAX_SPREAD_EXOTIC": 15.0,  # Maximum 15 pips for exotic pairs
    "RISK_PER_TRADE": 0.01,  # 1% risk per trade
    "RISK_PER_EXOTIC": 0.005,  # 0.5% risk for exotic pairs
    "MAX_DAILY_LOSS": 0.05,  # 5% max daily loss
    "MAX_CONCURRENT": 5,     # Maximum 5 concurrent positions (increased for diversification)
    "MIN_RR_RATIO": 1.5,     # 1.5:1 minimum risk-reward ratio
    "MIN_RR_EXOTIC": 2.0,    # 2:1 for exotic pairs due to wider spreads
    "TIMEZONE": "Asia/Tokyo",
    "ACCOUNT_CURRENCY": "JPY",
    "SYMBOL_FILTER": "FOREX",
    "MIN_VOLUME": 0.01,
    "AGGRESSIVE_MODE": False,
    "POSITION_INTERVAL": 600,   # 10 minutes between trades per symbol
    "MAX_SYMBOLS": 25,         # Increased to include all high-profit pairs
    "FORCE_TRADE_INTERVAL": 600,  # Force a trade if none in 10 minutes
    "IGNORE_SPREAD": False,    # Check spread before trading
    "MIN_INDICATORS": 5,       # At least 5 indicators must be positive
    "EXOTIC_CURRENCIES": ['TRY', 'ZAR', 'MXN', 'PLN', 'HUF', 'SEK', 'NOK', 'DKK', 
                         'SGD', 'HKD', 'THB', 'CNH', 'RUB', 'BRL', 'INR', 'KRW',
                         'ILS', 'AED', 'SAR', 'PHP', 'IDR', 'MYR', 'CZK', 'RON'],
    "METAL_SYMBOLS": ['XAU', 'XAG', 'XPT', 'XPD'],
    "CRYPTO_SYMBOLS": ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA', 'DOT'],
    "INDEX_SYMBOLS": ['US30', 'USTEC', 'NAS100', 'GER40', 'DAX', 'US500', 'UK100', 'JP225'],
    "MAX_SPREAD_METAL": 30.0,  # Gold can have 30 pip spreads
    "MAX_SPREAD_CRYPTO": 50.0,  # Crypto can have very wide spreads
    "MAX_SPREAD_INDEX": 5.0,   # Indices usually 3-5 points
    "RISK_PER_METAL": 0.007,  # 0.7% risk for metals
    "RISK_PER_CRYPTO": 0.003,  # 0.3% risk for crypto due to extreme volatility
    "RISK_PER_INDEX": 0.008,  # 0.8% risk for indices (high profit potential)
}

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Signal:
    type: SignalType
    confidence: float
    entry: float
    sl: float
    tp: float
    reason: str
    strategies: Dict[str, float] = None
    quality: float = 0

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

class UltraTradingEngine:
    def __init__(self, signal_queue: Optional[queue.Queue] = None):
        self.config = CONFIG
        self.api_base = CONFIG["API_BASE"]
        self.timezone = pytz.timezone(CONFIG["TIMEZONE"])
        
        # Communication queue for visualizer
        self.signal_queue = signal_queue
        
        # State
        self.active_trades = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_time = {}  # Per symbol tracking
        self.last_global_trade_time = 0  # Track last trade globally
        self.balance = None
        self.running = False
        self.trades_this_hour = 0
        self.force_trade_attempts = 0
        
        # Market cache
        self.spread_cache = {}
        self.data_cache = {}
        self.indicator_cache = {}
        
        # JPY account settings
        self.account_currency = CONFIG["ACCOUNT_CURRENCY"]
        
        # Strategy tracking
        self.last_signals = {}
        
        # Symbols list
        self.tradable_symbols = []
        
        # Price history for pattern detection
        self.price_history = {}
        
        # Market profile
        self.market_profiles = {}
        
    def start(self):
        """Initialize and start trading"""
        if not self._check_connection():
            logger.error("Cannot connect to API")
            return False
            
        self.balance = self._get_balance()
        if not self.balance:
            logger.error("Cannot get account balance")
            return False
            
        # Discover tradable symbols
        self.tradable_symbols = self._discover_symbols()
        if not self.tradable_symbols:
            logger.error("No tradable symbols found")
            return False
            
        logger.info(f"🚀 Starting Ultra Trading Engine")
        logger.info(f"💰 Balance: ¥{self.balance:,.0f}")
        logger.info(f"📊 Trading {len(self.tradable_symbols)} symbols: {self.tradable_symbols[:5]}...")
        if USE_ULTRA_CONFIG:
            logger.info(f"🔥🔥 Using ULTRA RARE PROFIT symbols configuration!")
            logger.info(f"💎 Exotic Forex: {len(EXOTIC_FOREX_ULTRA)} pairs")
            logger.info(f"🎯 Synthetic Indices: {len(SYNTHETIC_INDICES)} symbols") 
            logger.info(f"🌟 Rare Commodities: {len(RARE_COMMODITIES)} symbols")
            logger.info(f"⚡ Total Ultra Symbols: {len(ULTRA_PROFIT_SYMBOLS)}")
        elif USE_EXTERNAL_CONFIG:
            logger.info(f"🔥 Using HIGH-PROFIT symbols configuration!")
            logger.info(f"📈 Extreme volatility: {len(EXTREME_VOLATILITY_SYMBOLS)} symbols")
            logger.info(f"📈 Very high volatility: {len(VERY_HIGH_VOLATILITY_SYMBOLS)} symbols")
            logger.info(f"📈 High volatility: {len(HIGH_VOLATILITY_SYMBOLS)} symbols")
        logger.info(f"⏱️  Position interval: {CONFIG['POSITION_INTERVAL']/60:.0f} minutes per symbol")
        logger.info(f"🎯 Requirements: Min Conf={CONFIG['MIN_CONFIDENCE']}, Min Strategies={CONFIG['MIN_STRATEGIES']}")
        logger.info(f"🛡️  Risk: {CONFIG['RISK_PER_TRADE']*100}% per trade, Max daily loss: {CONFIG['MAX_DAILY_LOSS']*100}%")
        logger.info(f"📏 Max spread: {CONFIG['MAX_SPREAD_PIPS']} pips, Max concurrent: {CONFIG['MAX_CONCURRENT']}")
        
        self.running = True
        self._run_loop()
        
    def _check_connection(self) -> bool:
        """Verify API connection"""
        try:
            resp = requests.get(f"{self.api_base}/status/mt5", timeout=5)
            return resp.status_code == 200
        except:
            return False
            
    def _get_balance(self) -> Optional[float]:
        """Get account balance"""
        try:
            resp = requests.get(f"{self.api_base}/account/", timeout=5)
            if resp.status_code == 200:
                return resp.json()['balance']
        except:
            pass
        return None
        
    def _discover_symbols(self) -> List[str]:
        """Discover all tradable forex symbols including exotic pairs"""
        # First, try to use high-profit symbols from configuration
        if USE_ULTRA_CONFIG:
            # Use ultra rare profit symbols as highest priority
            priority_symbols = list(ULTRA_PROFIT_SYMBOLS.keys())
            logger.info(f"🔥🔥 Using ULTRA RARE PROFIT symbols! {len(priority_symbols)} symbols loaded")
        elif USE_EXTERNAL_CONFIG:
            priority_symbols = list(ALL_HIGH_PROFIT_SYMBOLS.keys())
        else:
            priority_symbols = list(HIGH_PROFIT_SYMBOLS.keys())
        
        try:
            resp = requests.get(f"{self.api_base}/market/symbols", timeout=10)
            if resp.status_code == 200:
                all_symbols = resp.json()
                
                # Extended currency list including exotic pairs
                major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
                exotic_currencies = ['TRY', 'ZAR', 'MXN', 'PLN', 'HUF', 'SEK', 'NOK', 'DKK', 
                                   'SGD', 'HKD', 'THB', 'CNH', 'RUB', 'BRL', 'INR', 'KRW',
                                   'ILS', 'AED', 'SAR', 'PHP', 'IDR', 'MYR', 'CZK', 'RON']
                # Precious metals and crypto symbols
                metals_crypto = ['XAU', 'XAG', 'XPT', 'XPD', 'BTC', 'ETH', 'LTC', 'XRP']
                # Major indices - HIGH PROFIT POTENTIAL (60-85% annual returns)
                index_symbols = ['US30', 'USTEC', 'NAS100', 'GER40', 'DAX', 'US500', 'UK100', 'JP225']
                all_currencies = major_currencies + exotic_currencies + metals_crypto
                
                # Filter for forex pairs and indices
                forex_symbols = []
                exotic_pairs = []
                metal_pairs = []
                crypto_pairs = []
                index_pairs = []
                ultra_exotic_crosses = []
                
                for symbol in all_symbols:
                    name = symbol.get('name', '')
                    
                    # Check if it's a tradable symbol
                    if symbol.get('trade_mode', 0) > 0:
                        # Check minimum volume
                        min_vol = symbol.get('volume_min', 0)
                        if min_vol <= CONFIG["MIN_VOLUME"]:
                            # Categorize by type
                            if any(metal in name for metal in ['XAU', 'XAG', 'XPT', 'XPD']):
                                metal_pairs.append(name)
                            elif any(crypto in name for crypto in ['BTC', 'ETH', 'LTC', 'XRP']):
                                crypto_pairs.append(name)
                            elif any(index in name for index in index_symbols):
                                index_pairs.append(name)
                            elif any(exotic in name for exotic in exotic_currencies):
                                # Check for ultra-exotic crosses
                                exotic_count = sum(1 for e in exotic_currencies if e in name)
                                if exotic_count >= 2:
                                    ultra_exotic_crosses.append(name)
                                else:
                                    exotic_pairs.append(name)
                            elif any(curr in name for curr in major_currencies):
                                forex_symbols.append(name)
                
                # Prioritize mix of all instrument types
                combined_symbols = []
                
                # FIRST PRIORITY: Add all symbols from our high-profit configuration
                if USE_EXTERNAL_CONFIG:
                    # Check which priority symbols are available
                    for symbol in priority_symbols:
                        # Try to find exact match first
                        if symbol in all_symbols:
                            combined_symbols.append(symbol)
                        else:
                            # Try with broker suffix
                            for suffix in ['#', '.', '']:
                                test_symbol = f"{symbol}{suffix}"
                                matching = [s['name'] for s in all_symbols if s.get('name', '') == test_symbol]
                                if matching:
                                    combined_symbols.extend(matching[:1])
                                    break
                
                # If we have filled most slots with high-profit symbols, just add a few more
                if len(combined_symbols) >= CONFIG["MAX_SYMBOLS"] * 0.8:
                    logger.info(f"🔥 Loaded {len(combined_symbols)} HIGH-PROFIT symbols from configuration!")
                else:
                    # Add additional symbols using old logic
                    # Add core major pairs (if not already included)
                    major_pairs = ["EURUSD", "USDJPY", "GBPUSD"]
                    for pair in major_pairs:
                        if not any(pair in s for s in combined_symbols):
                            matching = [s for s in forex_symbols if pair in s]
                            combined_symbols.extend(matching[:1])
                    
                    # Add precious metals (HIGHEST PROFIT POTENTIAL)
                    priority_metals = ["XAUUSD", "XAGUSD", "XPDUSD", "XPTUSD"]
                    for pair in priority_metals:
                        if not any(pair in s for s in combined_symbols):
                            matching = [s for s in metal_pairs if pair in s]
                            combined_symbols.extend(matching[:1])
                    
                    # Add high-profit exotic pairs
                    priority_exotics = ["USDZAR", "USDMXN", "USDTRY", "EURTRY", "GBPTRY"]
                    for pair in priority_exotics:
                        if not any(pair in s for s in combined_symbols):
                            matching = [s for s in exotic_pairs if pair in s]
                            combined_symbols.extend(matching[:1])
                    
                    # Add high-profit indices
                    priority_indices = ["US2000", "VIX", "HK50", "BVSP"]
                    for pair in priority_indices:
                        if not any(pair in s for s in combined_symbols):
                            matching = [s for s in index_pairs if pair in s]
                            combined_symbols.extend(matching[:1])
                    
                    # Add commodities
                    priority_commodities = ["NATGAS", "UKOIL", "USOIL", "COFFEE", "WHEAT"]
                    for pair in priority_commodities:
                        if not any(pair in s for s in combined_symbols):
                            matching = [s['name'] for s in all_symbols if pair in s.get('name', '')]
                            combined_symbols.extend(matching[:1])
                
                # Fill remaining slots with diverse selection
                remaining_slots = CONFIG["MAX_SYMBOLS"] - len(combined_symbols)
                if remaining_slots > 0:
                    # Mix of everything not yet included
                    all_remaining = [s for s in (forex_symbols + exotic_pairs + metal_pairs + crypto_pairs) 
                                   if s not in combined_symbols]
                    combined_symbols.extend(all_remaining[:remaining_slots])
                
                logger.info(f"Discovered diverse portfolio: {len(forex_symbols)} forex, {len(exotic_pairs)} exotic, "
                          f"{len(metal_pairs)} metals, {len(crypto_pairs)} crypto, {len(index_pairs)} indices, "
                          f"{len(ultra_exotic_crosses)} ultra-exotic crosses")
                logger.info(f"Selected {len(combined_symbols)} symbols for trading")
                return combined_symbols[:CONFIG["MAX_SYMBOLS"]]
                
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            
        # Enhanced fallback with high-profit symbols
        if USE_EXTERNAL_CONFIG:
            # Return our high-profit symbols with common broker suffixes
            fallback_symbols = []
            for symbol in list(EXTREME_VOLATILITY_SYMBOLS.keys())[:5] + \
                         list(VERY_HIGH_VOLATILITY_SYMBOLS.keys())[:5] + \
                         list(HIGH_VOLATILITY_SYMBOLS.keys())[:5]:
                fallback_symbols.append(f"{symbol}#")
            return fallback_symbols
        else:
            return ["EURUSD#", "USDJPY#", "GBPUSD#", "XAUUSD#", "XAGUSD#", "US30#",
                    "USTEC#", "GER40#", "USDZAR#", "USDMXN#", "EURGBP#", "AUDNZD#",
                    "GBPJPY#", "EURTRY#", "USDPLN#"]
        
    def _calculate_position_size(self, symbol: str, sl_distance: float) -> float:
        """Calculate safe position size based on risk management"""
        # Get current account status
        account_status = self._check_account_safety()
        equity = account_status.get('equity', self.balance)
        free_margin = account_status.get('free_margin', equity)
        
        # Use equity for risk calculation, not balance
        # Try to get symbol-specific risk from high-profit config first
        symbol_config = get_symbol_config(symbol)
        
        # Use new config format if available
        if USE_EXTERNAL_CONFIG and 'risk_per_trade' in symbol_config:
            risk_per_trade = symbol_config['risk_per_trade']
        else:
            risk_per_trade = symbol_config.get("risk_factor", None)
            
        # Fall back to instrument type based risk if not found
        if risk_per_trade is None:
            instrument_type = self._get_instrument_type(symbol)
            risk_map = {
                "crypto": CONFIG["RISK_PER_CRYPTO"],
                "metal": CONFIG["RISK_PER_METAL"],
                "index": CONFIG["RISK_PER_INDEX"],
                "exotic": CONFIG["RISK_PER_EXOTIC"],
                "major": CONFIG["RISK_PER_TRADE"]
            }
            risk_per_trade = risk_map.get(instrument_type, CONFIG["RISK_PER_TRADE"])
        risk_amount = equity * risk_per_trade
        
        # Get symbol info for proper pip calculation
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            return 0.01  # Minimum lot size as fallback
            
        # Calculate position size based on risk
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        sl_pips = sl_distance / pip_value
        
        # For JPY account, adjust calculation
        if self.account_currency == "JPY":
            if "JPY" in symbol:
                # Direct JPY pair
                position_size = risk_amount / (sl_pips * 100)
            else:
                # Need to convert through USD/JPY rate
                usd_jpy_rate = self._get_current_price("USDJPY#")
                if usd_jpy_rate:
                    position_size = risk_amount / (sl_pips * 10 * usd_jpy_rate)
                else:
                    position_size = 0.01
        else:
            position_size = risk_amount / (sl_pips * 10)
            
        # Safety checks
        max_size = free_margin * 0.1 / 100000  # Max 10% of free margin
        position_size = min(position_size, max_size, 0.1)  # Cap at 0.1 lots
        position_size = max(position_size, 0.01)  # Min 0.01 lots
        
        return round(position_size, 2)
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            resp = requests.get(f"{self.api_base}/market/symbols", timeout=5)
            if resp.status_code == 200:
                symbols = resp.json()
                for sym in symbols:
                    if sym.get('name') == symbol:
                        return sym
        except:
            pass
        return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            resp = requests.post(
                f"{self.api_base}/market/tick",
                json={"symbol": symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return resp.json().get('bid')
        except:
            pass
        return None
        
    def _get_market_data(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with longer history for pattern detection"""
        cache_key = f"{symbol}_data_{count}"
        if cache_key in self.data_cache:
            data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < 30:
                return data
                
        try:
            resp = requests.post(
                f"{self.api_base}/market/history",
                json={"symbol": symbol, "timeframe": CONFIG["TIMEFRAME"], "count": count},
                timeout=5
            )
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json())
                self.data_cache[cache_key] = (df, time.time())
                return df
        except:
            pass
        return None
        
    def _check_spread(self, symbol: str) -> Tuple[bool, float]:
        """Check if spread is acceptable (different limits per instrument type)"""
        if symbol in self.spread_cache:
            spread, timestamp = self.spread_cache[symbol]
            if time.time() - timestamp < 10:
                max_spread = self._get_max_spread(symbol)
                return spread <= max_spread, spread
                
        try:
            resp = requests.get(f"{self.api_base}/market/symbols/{quote(symbol)}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                points = data.get('spread', 999)
                
                # Convert to pips (different for metals/crypto/indices)
                if self._is_metal_pair(symbol):
                    spread = points * 0.01  # Metals usually in dollars
                elif self._is_crypto_pair(symbol):
                    spread = points * 0.1  # Crypto in larger units
                elif self._is_index_pair(symbol):
                    spread = points  # Indices use points directly
                else:
                    spread = points * 0.01 if "JPY" in symbol else points * 0.1
                    
                self.spread_cache[symbol] = (spread, time.time())
                
                # Get appropriate spread limit
                max_spread = self._get_max_spread(symbol)
                return spread <= max_spread, spread
        except:
            pass
        return False, 999  # Reject if can't check spread
    
    def _get_max_spread(self, symbol: str) -> float:
        """Get maximum allowed spread based on instrument type"""
        # Try to get symbol-specific spread from high-profit config first
        symbol_config = get_symbol_config(symbol)
        
        # Use new config format if available
        if USE_EXTERNAL_CONFIG and 'max_spread' in symbol_config:
            return symbol_config['max_spread']
        
        typical_spread = symbol_config.get("typical_spread", 0)
        if typical_spread > 0:
            # Allow 1.5x typical spread as maximum
            return typical_spread * 1.5
        
        # Fall back to instrument type based spreads
        instrument_type = self._get_instrument_type(symbol)
        if instrument_type == "crypto":
            return CONFIG["MAX_SPREAD_CRYPTO"]
        elif instrument_type == "metal":
            return CONFIG["MAX_SPREAD_METAL"]
        elif instrument_type == "index":
            return CONFIG["MAX_SPREAD_INDEX"]
        elif instrument_type == "exotic":
            return CONFIG["MAX_SPREAD_EXOTIC"]
        else:
            return CONFIG["MAX_SPREAD_PIPS"]
    
    def _is_exotic_pair(self, symbol: str) -> bool:
        """Check if symbol is an exotic pair"""
        return any(currency in symbol for currency in CONFIG["EXOTIC_CURRENCIES"])
    
    def _is_metal_pair(self, symbol: str) -> bool:
        """Check if symbol is a precious metal pair"""
        return any(metal in symbol for metal in CONFIG["METAL_SYMBOLS"])
    
    def _is_crypto_pair(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency pair"""
        return any(crypto in symbol for crypto in CONFIG["CRYPTO_SYMBOLS"])
    
    def _is_index_pair(self, symbol: str) -> bool:
        """Check if symbol is a stock index"""
        return any(index in symbol for index in CONFIG["INDEX_SYMBOLS"])
    
    def _get_instrument_type(self, symbol: str) -> str:
        """Get the instrument type for risk management"""
        if self._is_crypto_pair(symbol):
            return "crypto"
        elif self._is_metal_pair(symbol):
            return "metal"
        elif self._is_index_pair(symbol):
            return "index"
        elif self._is_exotic_pair(symbol):
            return "exotic"
        else:
            return "major"
        
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate ALL 100 trading indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['tick_volume'].values
        
        indicators = {}
        
        # === 1. PRICE ACTION INDICATORS (10) ===
        
        # 1.1 Pin Bar Detection
        body = abs(close[-1] - open_price[-1])
        upper_wick = high[-1] - max(close[-1], open_price[-1])
        lower_wick = min(close[-1], open_price[-1]) - low[-1]
        
        indicators['pin_bar_bull'] = 1.0 if lower_wick > body * 2 and upper_wick < body else 0.0
        indicators['pin_bar_bear'] = 1.0 if upper_wick > body * 2 and lower_wick < body else 0.0
        
        # 1.2 Engulfing Pattern
        if len(close) >= 2:
            prev_body = abs(close[-2] - open_price[-2])
            curr_body = abs(close[-1] - open_price[-1])
            indicators['engulfing_bull'] = 1.0 if (open_price[-1] < close[-2] and close[-1] > open_price[-2] and 
                                                   curr_body > prev_body and close[-1] > open_price[-1]) else 0.0
            indicators['engulfing_bear'] = 1.0 if (open_price[-1] > close[-2] and close[-1] < open_price[-2] and 
                                                   curr_body > prev_body and close[-1] < open_price[-1]) else 0.0
        else:
            indicators['engulfing_bull'] = 0.0
            indicators['engulfing_bear'] = 0.0
        
        # 1.3 Doji Detection
        indicators['doji'] = 1.0 if body < (high[-1] - low[-1]) * 0.1 else 0.0
        
        # 1.4 Hammer/Hanging Man
        indicators['hammer'] = 1.0 if (lower_wick > body * 2 and upper_wick < body * 0.5 and 
                                      close[-1] < np.mean(close[-20:-1])) else 0.0
        indicators['hanging_man'] = 1.0 if (lower_wick > body * 2 and upper_wick < body * 0.5 and 
                                           close[-1] > np.mean(close[-20:-1])) else 0.0
        
        # 1.5 Three White Soldiers / Three Black Crows
        if len(close) >= 3:
            three_white = all(close[i] > close[i-1] and close[i] > open_price[i] for i in range(-3, 0))
            three_black = all(close[i] < close[i-1] and close[i] < open_price[i] for i in range(-3, 0))
            indicators['three_white_soldiers'] = 1.0 if three_white else 0.0
            indicators['three_black_crows'] = 1.0 if three_black else 0.0
        else:
            indicators['three_white_soldiers'] = 0.0
            indicators['three_black_crows'] = 0.0
        
        # 1.6 Inside Bar
        indicators['inside_bar'] = 1.0 if (high[-1] < high[-2] and low[-1] > low[-2]) else 0.0
        
        # 1.7 Price Action Momentum
        indicators['pa_momentum'] = (close[-1] - close[-5]) / close[-5] * 100
        
        # === 2. CHART PATTERNS (10) ===
        
        # 2.1 Head and Shoulders Detection (simplified)
        if len(high) >= 5:
            mid_high = high[-3]
            left_shoulder = high[-5]
            right_shoulder = high[-1]
            indicators['head_shoulders'] = 1.0 if (mid_high > left_shoulder and mid_high > right_shoulder and 
                                                  abs(left_shoulder - right_shoulder) / mid_high < 0.02) else 0.0
        else:
            indicators['head_shoulders'] = 0.0
        
        # 2.2 Double Top/Bottom
        if len(high) >= 20:
            recent_highs = pd.Series(high[-20:]).nlargest(2)
            recent_lows = pd.Series(low[-20:]).nsmallest(2)
            indicators['double_top'] = 1.0 if abs(recent_highs.iloc[0] - recent_highs.iloc[1]) / recent_highs.iloc[0] < 0.01 else 0.0
            indicators['double_bottom'] = 1.0 if abs(recent_lows.iloc[0] - recent_lows.iloc[1]) / recent_lows.iloc[0] < 0.01 else 0.0
        else:
            indicators['double_top'] = 0.0
            indicators['double_bottom'] = 0.0
        
        # 2.3 Triangle Pattern (converging highs and lows)
        if len(high) >= 10:
            high_trend = np.polyfit(range(10), high[-10:], 1)[0]
            low_trend = np.polyfit(range(10), low[-10:], 1)[0]
            indicators['triangle_pattern'] = 1.0 if abs(high_trend) < 0.0001 and abs(low_trend) < 0.0001 else 0.0
        else:
            indicators['triangle_pattern'] = 0.0
        
        # 2.4 Channel Detection
        if len(close) >= 20:
            upper_channel = pd.Series(high[-20:]).rolling(20).max().iloc[-1]
            lower_channel = pd.Series(low[-20:]).rolling(20).min().iloc[-1]
            channel_width = upper_channel - lower_channel
            indicators['channel_upper'] = 1.0 if abs(close[-1] - upper_channel) / channel_width < 0.1 else 0.0
            indicators['channel_lower'] = 1.0 if abs(close[-1] - lower_channel) / channel_width < 0.1 else 0.0
        else:
            indicators['channel_upper'] = 0.0
            indicators['channel_lower'] = 0.0
        
        # 2.5 Flag Pattern
        if len(close) >= 10:
            trend_before = (close[-10] - close[-20]) / close[-20] if len(close) >= 20 else 0
            consolidation = np.std(close[-10:]) / np.mean(close[-10:])
            indicators['flag_pattern'] = 1.0 if abs(trend_before) > 0.02 and consolidation < 0.01 else 0.0
        else:
            indicators['flag_pattern'] = 0.0
        
        # 2.6 Wedge Pattern
        if len(high) >= 10:
            high_slope = np.polyfit(range(10), high[-10:], 1)[0]
            low_slope = np.polyfit(range(10), low[-10:], 1)[0]
            indicators['rising_wedge'] = 1.0 if high_slope > 0 and low_slope > 0 and high_slope < low_slope else 0.0
            indicators['falling_wedge'] = 1.0 if high_slope < 0 and low_slope < 0 and high_slope > low_slope else 0.0
        else:
            indicators['rising_wedge'] = 0.0
            indicators['falling_wedge'] = 0.0
        
        # === 3. MATHEMATICAL INDICATORS (10) ===
        
        # 3.1 Fibonacci Retracement Levels
        if len(high) >= 50:
            recent_high = max(high[-50:])
            recent_low = min(low[-50:])
            fib_range = recent_high - recent_low
            
            fib_236 = recent_high - fib_range * 0.236
            fib_382 = recent_high - fib_range * 0.382
            fib_500 = recent_high - fib_range * 0.500
            fib_618 = recent_high - fib_range * 0.618
            
            indicators['fib_236'] = 1.0 if abs(close[-1] - fib_236) / fib_range < 0.02 else 0.0
            indicators['fib_382'] = 1.0 if abs(close[-1] - fib_382) / fib_range < 0.02 else 0.0
            indicators['fib_500'] = 1.0 if abs(close[-1] - fib_500) / fib_range < 0.02 else 0.0
            indicators['fib_618'] = 1.0 if abs(close[-1] - fib_618) / fib_range < 0.02 else 0.0
        else:
            indicators['fib_236'] = 0.0
            indicators['fib_382'] = 0.0
            indicators['fib_500'] = 0.0
            indicators['fib_618'] = 0.0
        
        # 3.2 Pivot Points
        if len(high) >= 2:
            pivot = (high[-2] + low[-2] + close[-2]) / 3
            r1 = 2 * pivot - low[-2]
            s1 = 2 * pivot - high[-2]
            indicators['pivot_point'] = 1.0 if abs(close[-1] - pivot) / pivot < 0.001 else 0.0
            indicators['pivot_r1'] = 1.0 if abs(close[-1] - r1) / r1 < 0.001 else 0.0
            indicators['pivot_s1'] = 1.0 if abs(close[-1] - s1) / s1 < 0.001 else 0.0
        else:
            indicators['pivot_point'] = 0.0
            indicators['pivot_r1'] = 0.0
            indicators['pivot_s1'] = 0.0
        
        # 3.3 Linear Regression
        if len(close) >= 20:
            x = np.arange(20)
            y = close[-20:]
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * 19 + intercept
            indicators['lin_reg_slope'] = slope * 1000  # Scaled for visibility
            indicators['lin_reg_deviation'] = (close[-1] - predicted) / predicted * 100
        else:
            indicators['lin_reg_slope'] = 0.0
            indicators['lin_reg_deviation'] = 0.0
        
        # === 4. VOLATILITY ANALYSIS (10) ===
        
        # 4.1 ATR-based indicators
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        indicators['atr'] = atr
        indicators['atr_ratio'] = atr / close[-1] * 100  # ATR as percentage of price
        
        # 4.2 Bollinger Band Width
        bb_mean = pd.Series(close).rolling(20).mean().iloc[-1]
        bb_std = pd.Series(close).rolling(20).std().iloc[-1]
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_mean * 100
        indicators['bb_squeeze'] = 1.0 if indicators['bb_width'] < 2.0 else 0.0
        
        # 4.3 Keltner Channels
        kc_ema = pd.Series(close).ewm(span=20).mean().iloc[-1]
        kc_upper = kc_ema + 2 * atr
        kc_lower = kc_ema - 2 * atr
        indicators['keltner_upper'] = 1.0 if close[-1] > kc_upper else 0.0
        indicators['keltner_lower'] = 1.0 if close[-1] < kc_lower else 0.0
        
        # 4.4 Historical Volatility
        returns = pd.Series(close).pct_change().dropna()
        indicators['hist_volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # 4.5 Volatility Ratio
        short_vol = returns[-10:].std() if len(returns) >= 10 else 0
        long_vol = returns[-50:].std() if len(returns) >= 50 else short_vol
        indicators['vol_ratio'] = short_vol / long_vol if long_vol > 0 else 1.0
        
        # 4.6 Donchian Channels
        if len(high) >= 20:
            don_high = max(high[-20:])
            don_low = min(low[-20:])
            don_mid = (don_high + don_low) / 2
            indicators['donchian_high'] = 1.0 if close[-1] > don_high * 0.98 else 0.0
            indicators['donchian_low'] = 1.0 if close[-1] < don_low * 1.02 else 0.0
        else:
            indicators['donchian_high'] = 0.0
            indicators['donchian_low'] = 0.0
        
        # === 5. MARKET STRUCTURE (10) ===
        
        # 5.1 Support/Resistance Levels
        if len(high) >= 50:
            # Find swing points
            swing_highs = []
            swing_lows = []
            for i in range(2, len(high) - 2):
                if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                    swing_highs.append(high[i])
                if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                    swing_lows.append(low[i])
            
            # Check proximity to support/resistance
            indicators['near_resistance'] = 0.0
            indicators['near_support'] = 0.0
            
            for res in swing_highs[-5:] if swing_highs else []:
                if abs(close[-1] - res) / res < 0.005:
                    indicators['near_resistance'] = 1.0
                    
            for sup in swing_lows[-5:] if swing_lows else []:
                if abs(close[-1] - sup) / sup < 0.005:
                    indicators['near_support'] = 1.0
        else:
            indicators['near_resistance'] = 0.0
            indicators['near_support'] = 0.0
        
        # 5.2 Market Structure Break
        if len(high) >= 10:
            recent_high = max(high[-10:-1])
            recent_low = min(low[-10:-1])
            indicators['structure_break_up'] = 1.0 if close[-1] > recent_high else 0.0
            indicators['structure_break_down'] = 1.0 if close[-1] < recent_low else 0.0
        else:
            indicators['structure_break_up'] = 0.0
            indicators['structure_break_down'] = 0.0
        
        # 5.3 Higher Highs/Lower Lows
        if len(high) >= 20:
            hh_count = 0
            ll_count = 0
            for i in range(-10, -1):
                if high[i] > high[i-1]:
                    hh_count += 1
                if low[i] < low[i-1]:
                    ll_count += 1
            indicators['higher_highs'] = hh_count / 9
            indicators['lower_lows'] = ll_count / 9
        else:
            indicators['higher_highs'] = 0.0
            indicators['lower_lows'] = 0.0
        
        # 5.4 Range Detection
        if len(high) >= 20:
            range_high = max(high[-20:])
            range_low = min(low[-20:])
            range_size = range_high - range_low
            indicators['in_range'] = 1.0 if range_size / close[-1] < 0.02 else 0.0
        else:
            indicators['in_range'] = 0.0
        
        # === 6. MOMENTUM ANALYSIS (10) ===
        
        # 6.1 Rate of Change (ROC)
        for period in [5, 10, 20]:
            if len(close) > period:
                indicators[f'roc_{period}'] = (close[-1] - close[-period-1]) / close[-period-1] * 100
            else:
                indicators[f'roc_{period}'] = 0.0
        
        # 6.2 Momentum Oscillator
        if len(close) >= 14:
            indicators['momentum_14'] = close[-1] - close[-14]
        else:
            indicators['momentum_14'] = 0.0
        
        # 6.3 Price Oscillator
        if len(close) >= 26:
            short_ema = pd.Series(close).ewm(span=12).mean().iloc[-1]
            long_ema = pd.Series(close).ewm(span=26).mean().iloc[-1]
            indicators['price_oscillator'] = (short_ema - long_ema) / long_ema * 100
        else:
            indicators['price_oscillator'] = 0.0
        
        # 6.4 Commodity Channel Index (CCI)
        if len(high) >= 20:
            typical_price = (high + low + close) / 3
            sma_tp = pd.Series(typical_price).rolling(20).mean().iloc[-1]
            mad = pd.Series(abs(typical_price - sma_tp)).rolling(20).mean().iloc[-1]
            indicators['cci'] = (typical_price[-1] - sma_tp) / (0.015 * mad) if mad > 0 else 0
        else:
            indicators['cci'] = 0.0
        
        # 6.5 Williams %R
        if len(high) >= 14:
            highest_high = max(high[-14:])
            lowest_low = min(low[-14:])
            indicators['williams_r'] = ((highest_high - close[-1]) / (highest_high - lowest_low) * -100) if highest_high != lowest_low else -50
        else:
            indicators['williams_r'] = -50
        
        # 6.6 Ultimate Oscillator
        if len(high) >= 28:
            bp = close - np.minimum(low, np.roll(close, 1))  # Buying pressure
            tr_array = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
            
            avg7 = np.sum(bp[-7:]) / np.sum(tr_array[-7:]) if np.sum(tr_array[-7:]) > 0 else 0
            avg14 = np.sum(bp[-14:]) / np.sum(tr_array[-14:]) if np.sum(tr_array[-14:]) > 0 else 0
            avg28 = np.sum(bp[-28:]) / np.sum(tr_array[-28:]) if np.sum(tr_array[-28:]) > 0 else 0
            
            indicators['ultimate_oscillator'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
        else:
            indicators['ultimate_oscillator'] = 50
        
        # === 7. VOLUME/ORDER FLOW (10) ===
        
        # 7.1 Volume Moving Averages
        indicators['volume_sma_10'] = pd.Series(volume).rolling(10).mean().iloc[-1] if len(volume) >= 10 else 0
        indicators['volume_sma_20'] = pd.Series(volume).rolling(20).mean().iloc[-1] if len(volume) >= 20 else 0
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1
        
        # 7.2 On Balance Volume (OBV)
        if len(close) >= 2:
            obv = 0
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv += volume[i]
                elif close[i] < close[i-1]:
                    obv -= volume[i]
            indicators['obv_trend'] = 1.0 if obv > 0 else 0.0
        else:
            indicators['obv_trend'] = 0.0
        
        # 7.3 Accumulation/Distribution Line
        if len(high) > 0:
            mfm = ((close - low) - (high - close)) / (high - low)  # Money Flow Multiplier
            mfm = np.where(high == low, 0, mfm)  # Handle division by zero
            mfv = mfm * volume  # Money Flow Volume
            indicators['ad_line'] = np.sum(mfv) / 1000000  # Scaled
        else:
            indicators['ad_line'] = 0.0
        
        # 7.4 Chaikin Money Flow
        if len(high) >= 20:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.where(high == low, 0, mfm)
            mfv = mfm * volume
            indicators['chaikin_mf'] = np.sum(mfv[-20:]) / np.sum(volume[-20:]) if np.sum(volume[-20:]) > 0 else 0
        else:
            indicators['chaikin_mf'] = 0.0
        
        # 7.5 Volume Price Trend (VPT)
        if len(close) >= 2:
            vpt = 0
            for i in range(1, len(close)):
                vpt += volume[i] * (close[i] - close[i-1]) / close[i-1]
            indicators['vpt'] = vpt / 1000  # Scaled
        else:
            indicators['vpt'] = 0.0
        
        # 7.6 Force Index
        if len(close) >= 2:
            force = (close[-1] - close[-2]) * volume[-1]
            indicators['force_index'] = force / 1000000  # Scaled
        else:
            indicators['force_index'] = 0.0
        
        # 7.7 Money Flow Index (MFI)
        if len(high) >= 14:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = []
            negative_flow = []
            
            for i in range(1, 15):
                if typical_price[-i] > typical_price[-i-1]:
                    positive_flow.append(money_flow[-i])
                else:
                    negative_flow.append(money_flow[-i])
            
            positive_sum = sum(positive_flow) if positive_flow else 0
            negative_sum = sum(negative_flow) if negative_flow else 0
            
            if negative_sum > 0:
                mfi = 100 - (100 / (1 + positive_sum / negative_sum))
            else:
                mfi = 100
                
            indicators['mfi'] = mfi
        else:
            indicators['mfi'] = 50
        
        # === 8. TIME-BASED PATTERNS (10) ===
        
        # 8.1 Intraday Patterns
        current_hour = datetime.now(self.timezone).hour
        indicators['asian_session'] = 1.0 if 0 <= current_hour < 9 else 0.0
        indicators['london_session'] = 1.0 if 8 <= current_hour < 17 else 0.0
        indicators['ny_session'] = 1.0 if 13 <= current_hour < 22 else 0.0
        indicators['session_overlap'] = 1.0 if (8 <= current_hour < 9) or (13 <= current_hour < 17) else 0.0
        
        # 8.2 Day of Week Pattern
        dow = datetime.now(self.timezone).weekday()
        indicators['monday'] = 1.0 if dow == 0 else 0.0
        indicators['friday'] = 1.0 if dow == 4 else 0.0
        indicators['midweek'] = 1.0 if dow in [1, 2, 3] else 0.0
        
        # 8.3 Time-based Momentum
        if len(close) >= 24:  # Last 2 hours in M5
            indicators['hourly_momentum'] = (close[-1] - close[-12]) / close[-12] * 100
            indicators['two_hour_momentum'] = (close[-1] - close[-24]) / close[-24] * 100
        else:
            indicators['hourly_momentum'] = 0.0
            indicators['two_hour_momentum'] = 0.0
        
        # 8.4 Opening Range
        if len(high) >= 12:  # First hour
            opening_high = max(high[-12:])
            opening_low = min(low[-12:])
            indicators['above_opening_range'] = 1.0 if close[-1] > opening_high else 0.0
            indicators['below_opening_range'] = 1.0 if close[-1] < opening_low else 0.0
        else:
            indicators['above_opening_range'] = 0.0
            indicators['below_opening_range'] = 0.0
        
        # === 9. STATISTICAL ANALYSIS (10) ===
        
        # 9.1 Z-Score
        if len(close) >= 20:
            mean_20 = np.mean(close[-20:])
            std_20 = np.std(close[-20:])
            indicators['z_score'] = (close[-1] - mean_20) / std_20 if std_20 > 0 else 0
        else:
            indicators['z_score'] = 0
        
        # 9.2 Percentile Rank
        if len(close) >= 50:
            indicators['percentile_rank'] = stats.percentileofscore(close[-50:], close[-1])
        else:
            indicators['percentile_rank'] = 50
        
        # 9.3 Standard Deviation Bands
        if len(close) >= 20:
            mean = np.mean(close[-20:])
            std = np.std(close[-20:])
            indicators['std_band_position'] = (close[-1] - mean) / std if std > 0 else 0
        else:
            indicators['std_band_position'] = 0
        
        # 9.4 Skewness
        if len(close) >= 30:
            returns = pd.Series(close).pct_change().dropna()
            indicators['return_skewness'] = stats.skew(returns[-30:])
        else:
            indicators['return_skewness'] = 0
        
        # 9.5 Kurtosis
        if len(close) >= 30:
            returns = pd.Series(close).pct_change().dropna()
            indicators['return_kurtosis'] = stats.kurtosis(returns[-30:])
        else:
            indicators['return_kurtosis'] = 0
        
        # 9.6 Autocorrelation
        if len(close) >= 20:
            indicators['autocorrelation'] = pd.Series(close[-20:]).autocorr(lag=1)
        else:
            indicators['autocorrelation'] = 0
        
        # 9.7 Mean Reversion Indicator
        if len(close) >= 50:
            long_mean = np.mean(close[-50:])
            indicators['mean_reversion'] = (long_mean - close[-1]) / long_mean * 100
        else:
            indicators['mean_reversion'] = 0
        
        # 9.8 Efficiency Ratio
        if len(close) >= 10:
            direction = abs(close[-1] - close[-10])
            volatility = sum(abs(close[i] - close[i-1]) for i in range(-9, 0))
            indicators['efficiency_ratio'] = direction / volatility if volatility > 0 else 0
        else:
            indicators['efficiency_ratio'] = 0
        
        # 9.9 Hurst Exponent (simplified)
        if len(close) >= 50:
            # Simplified R/S analysis
            returns = pd.Series(close).pct_change().dropna()
            if len(returns) >= 50:
                mean_return = returns.mean()
                std_return = returns.std()
                cumsum = (returns - mean_return).cumsum()
                R = cumsum.max() - cumsum.min()
                S = std_return
                indicators['hurst_exponent'] = np.log(R/S) / np.log(len(returns)) if S > 0 else 0.5
            else:
                indicators['hurst_exponent'] = 0.5
        else:
            indicators['hurst_exponent'] = 0.5
        
        # === 10. ADVANCED COMPOSITE INDICATORS (10) ===
        
        # 10.1 MACD
        if len(close) >= 26:
            ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            macd_line = ema_12 - ema_26
            signal_line = pd.Series(close).ewm(span=9).mean().iloc[-1]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
        else:
            indicators['macd'] = 0
            indicators['macd_signal'] = 0
            indicators['macd_histogram'] = 0
        
        # 10.2 RSI
        if len(close) >= 14:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - 100/(1 + rs)).iloc[-1]
        else:
            indicators['rsi'] = 50
        
        # 10.3 Stochastic Oscillator
        if len(high) >= 14:
            low_14 = pd.Series(low).rolling(14).min().iloc[-1]
            high_14 = pd.Series(high).rolling(14).max().iloc[-1]
            indicators['stoch_k'] = 100 * (close[-1] - low_14) / (high_14 - low_14) if high_14 != low_14 else 50
            indicators['stoch_d'] = pd.Series([indicators['stoch_k']]).rolling(3).mean().iloc[-1]
        else:
            indicators['stoch_k'] = 50
            indicators['stoch_d'] = 50
        
        # 10.4 ADX
        if len(high) >= 14:
            plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
            atr = pd.Series(tr).rolling(14).mean().iloc[-1]
            
            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            indicators['adx'] = dx
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
        else:
            indicators['adx'] = 0
            indicators['plus_di'] = 0
            indicators['minus_di'] = 0
        
        # 10.5 Ichimoku Components
        if len(high) >= 52:
            # Tenkan-sen (Conversion Line)
            nine_high = pd.Series(high).rolling(9).max().iloc[-1]
            nine_low = pd.Series(low).rolling(9).min().iloc[-1]
            indicators['tenkan_sen'] = (nine_high + nine_low) / 2
            
            # Kijun-sen (Base Line)
            twenty_six_high = pd.Series(high).rolling(26).max().iloc[-1]
            twenty_six_low = pd.Series(low).rolling(26).min().iloc[-1]
            indicators['kijun_sen'] = (twenty_six_high + twenty_six_low) / 2
            
            # Senkou Span A (Leading Span A)
            indicators['senkou_span_a'] = (indicators['tenkan_sen'] + indicators['kijun_sen']) / 2
            
            # Senkou Span B (Leading Span B)
            fifty_two_high = pd.Series(high).rolling(52).max().iloc[-1]
            fifty_two_low = pd.Series(low).rolling(52).min().iloc[-1]
            indicators['senkou_span_b'] = (fifty_two_high + fifty_two_low) / 2
            
            # Chikou Span (Lagging Span) - would be close[-26] but we use current
            indicators['chikou_span'] = close[-1]
            
            # Cloud signals
            indicators['above_cloud'] = 1.0 if close[-1] > max(indicators['senkou_span_a'], indicators['senkou_span_b']) else 0.0
            indicators['below_cloud'] = 1.0 if close[-1] < min(indicators['senkou_span_a'], indicators['senkou_span_b']) else 0.0
        else:
            for key in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'above_cloud', 'below_cloud']:
                indicators[key] = 0.0
        
        # Store current price info
        indicators['current_price'] = close[-1]
        
        return indicators
        
    def _analyze_ultra(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Ultra deep analysis with 100 indicators"""
        if len(df) < 50:  # Reduced requirement
            logger.debug(f"Not enough data for {symbol}: {len(df)} bars")
            return None
            
        # Calculate all 100 indicators
        indicators = self._calculate_all_indicators(df)
        
        # Debug logging
        logger.debug(f"Analyzing {symbol}...")
        
        # Initialize scoring system
        buy_score = 0
        sell_score = 0
        total_weight = 0
        reasons = []
        
        # Category weights
        weights = {
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
        
        # === ANALYZE EACH CATEGORY ===
        
        # 1. Price Action Analysis
        pa_weight = weights['price_action']
        
        if indicators['pin_bar_bull'] > 0:
            buy_score += pa_weight
            reasons.append("Bullish Pin Bar")
        if indicators['pin_bar_bear'] > 0:
            sell_score += pa_weight
            reasons.append("Bearish Pin Bar")
            
        if indicators['engulfing_bull'] > 0:
            buy_score += pa_weight
            reasons.append("Bullish Engulfing")
        if indicators['engulfing_bear'] > 0:
            sell_score += pa_weight
            reasons.append("Bearish Engulfing")
            
        if indicators['hammer'] > 0:
            buy_score += pa_weight * 0.8
            reasons.append("Hammer")
        if indicators['hanging_man'] > 0:
            sell_score += pa_weight * 0.8
            reasons.append("Hanging Man")
            
        if indicators['three_white_soldiers'] > 0:
            buy_score += pa_weight * 1.2
            reasons.append("Three White Soldiers")
        if indicators['three_black_crows'] > 0:
            sell_score += pa_weight * 1.2
            reasons.append("Three Black Crows")
            
        # 2. Chart Patterns
        cp_weight = weights['chart_patterns']
        
        if indicators['double_bottom'] > 0:
            buy_score += cp_weight * 1.5
            reasons.append("Double Bottom")
        if indicators['double_top'] > 0:
            sell_score += cp_weight * 1.5
            reasons.append("Double Top")
            
        if indicators['channel_lower'] > 0:
            buy_score += cp_weight
            reasons.append("Channel Support")
        if indicators['channel_upper'] > 0:
            sell_score += cp_weight
            reasons.append("Channel Resistance")
            
        if indicators['falling_wedge'] > 0:
            buy_score += cp_weight
            reasons.append("Falling Wedge")
        if indicators['rising_wedge'] > 0:
            sell_score += cp_weight
            reasons.append("Rising Wedge")
            
        # 3. Mathematical Indicators
        math_weight = weights['mathematical']
        
        # Fibonacci levels
        if indicators['fib_618'] > 0:
            buy_score += math_weight * 1.2
            reasons.append("Fib 61.8% Support")
        if indicators['fib_382'] > 0:
            buy_score += math_weight
            
        # Pivot points
        if indicators['pivot_s1'] > 0:
            buy_score += math_weight * 0.8
            reasons.append("Pivot Support")
        if indicators['pivot_r1'] > 0:
            sell_score += math_weight * 0.8
            reasons.append("Pivot Resistance")
            
        # Linear regression
        if indicators['lin_reg_slope'] > 0.5:
            buy_score += math_weight * 0.6
        elif indicators['lin_reg_slope'] < -0.5:
            sell_score += math_weight * 0.6
            
        # 4. Volatility Analysis
        vol_weight = weights['volatility']
        
        # Special handling for exotic pairs with higher volatility
        is_exotic = self._is_exotic_pair(symbol)
        
        if indicators['bb_squeeze'] > 0:
            # Bollinger squeeze often precedes big moves
            if indicators['momentum_14'] > 0:
                buy_score += vol_weight * 1.5
                reasons.append("BB Squeeze + Momentum")
            else:
                sell_score += vol_weight * 1.5
                
        if indicators['keltner_lower'] > 0:
            buy_score += vol_weight * (1.5 if is_exotic else 1.0)
            reasons.append("Keltner Lower")
        if indicators['keltner_upper'] > 0:
            sell_score += vol_weight * (1.5 if is_exotic else 1.0)
            reasons.append("Keltner Upper")
            
        # Exotic pairs have different volatility characteristics
        vol_threshold = 0.5 if is_exotic else 0.7
        z_threshold = -2.0 if is_exotic else -1.5
        
        if indicators['vol_ratio'] < vol_threshold:  # Low volatility
            # Mean reversion more likely
            if indicators['z_score'] < z_threshold:
                buy_score += vol_weight * (1.2 if is_exotic else 1.0)
            elif indicators['z_score'] > abs(z_threshold):
                sell_score += vol_weight * (1.2 if is_exotic else 1.0)
                
        # 5. Market Structure
        struct_weight = weights['market_structure']
        
        if indicators['near_support'] > 0:
            buy_score += struct_weight * 1.5
            reasons.append("At Support")
        if indicators['near_resistance'] > 0:
            sell_score += struct_weight * 1.5
            reasons.append("At Resistance")
            
        if indicators['structure_break_up'] > 0:
            buy_score += struct_weight * 2
            reasons.append("Bullish Structure Break")
        if indicators['structure_break_down'] > 0:
            sell_score += struct_weight * 2
            reasons.append("Bearish Structure Break")
            
        if indicators['higher_highs'] > 0.7:
            buy_score += struct_weight
        if indicators['lower_lows'] > 0.7:
            sell_score += struct_weight
            
        # 6. Momentum Analysis
        mom_weight = weights['momentum']
        
        # ROC signals
        if indicators['roc_5'] > 1 and indicators['roc_10'] > 1:
            buy_score += mom_weight
            reasons.append("Strong ROC")
        elif indicators['roc_5'] < -1 and indicators['roc_10'] < -1:
            sell_score += mom_weight
            
        # CCI signals
        if indicators['cci'] < -100:
            buy_score += mom_weight * 0.8
            reasons.append("CCI Oversold")
        elif indicators['cci'] > 100:
            sell_score += mom_weight * 0.8
            reasons.append("CCI Overbought")
            
        # Williams %R
        if indicators['williams_r'] < -80:
            buy_score += mom_weight * 0.7
        elif indicators['williams_r'] > -20:
            sell_score += mom_weight * 0.7
            
        # Ultimate Oscillator
        if indicators['ultimate_oscillator'] < 30:
            buy_score += mom_weight * 0.9
        elif indicators['ultimate_oscillator'] > 70:
            sell_score += mom_weight * 0.9
            
        # 7. Volume Analysis
        vol_weight = weights['volume']
        
        if indicators['volume_ratio'] > 2:
            # High volume confirmation
            if indicators['force_index'] > 0:
                buy_score += vol_weight * 1.5
                reasons.append("High Volume Buy")
            else:
                sell_score += vol_weight * 1.5
                reasons.append("High Volume Sell")
                
        # MFI signals
        if indicators['mfi'] < 20:
            buy_score += vol_weight
            reasons.append("MFI Oversold")
        elif indicators['mfi'] > 80:
            sell_score += vol_weight
            reasons.append("MFI Overbought")
            
        # Chaikin Money Flow
        if indicators['chaikin_mf'] > 0.2:
            buy_score += vol_weight * 0.8
        elif indicators['chaikin_mf'] < -0.2:
            sell_score += vol_weight * 0.8
            
        # 8. Time-Based Patterns
        time_weight = weights['time_based']
        
        # Session-based trading
        if indicators['session_overlap'] > 0:
            # Higher volatility during overlaps
            time_weight *= 1.5
            
        if indicators['london_session'] > 0:
            # Most liquid session
            time_weight *= 1.2
            
        # Day of week patterns
        if indicators['monday'] > 0 and indicators['hourly_momentum'] > 0:
            buy_score += time_weight  # Monday momentum
        elif indicators['friday'] > 0 and indicators['hourly_momentum'] < 0:
            sell_score += time_weight  # Friday reversals
            
        # Opening range breakouts
        if indicators['above_opening_range'] > 0:
            buy_score += time_weight
            reasons.append("Above Opening Range")
        elif indicators['below_opening_range'] > 0:
            sell_score += time_weight
            reasons.append("Below Opening Range")
            
        # 9. Statistical Analysis
        stat_weight = weights['statistical']
        
        # Z-score extremes
        if indicators['z_score'] < -2:
            buy_score += stat_weight * 1.2
            reasons.append("Statistical Extreme Low")
        elif indicators['z_score'] > 2:
            sell_score += stat_weight * 1.2
            reasons.append("Statistical Extreme High")
            
        # Percentile extremes
        if indicators['percentile_rank'] < 10:
            buy_score += stat_weight
        elif indicators['percentile_rank'] > 90:
            sell_score += stat_weight
            
        # Mean reversion
        if indicators['mean_reversion'] > 2:
            buy_score += stat_weight * 0.8
        elif indicators['mean_reversion'] < -2:
            sell_score += stat_weight * 0.8
            
        # Efficiency ratio (trending)
        if indicators['efficiency_ratio'] > 0.7:
            if indicators['momentum_14'] > 0:
                buy_score += stat_weight
            else:
                sell_score += stat_weight
                
        # 10. Composite Indicators
        comp_weight = weights['composite']
        
        # MACD
        if indicators['macd_histogram'] > 0 and indicators['macd'] > indicators['macd_signal']:
            buy_score += comp_weight
            reasons.append("MACD Bullish")
        elif indicators['macd_histogram'] < 0 and indicators['macd'] < indicators['macd_signal']:
            sell_score += comp_weight
            reasons.append("MACD Bearish")
            
        # RSI
        if indicators['rsi'] < 30:
            buy_score += comp_weight * 1.2
            reasons.append(f"RSI {indicators['rsi']:.0f}")
        elif indicators['rsi'] > 70:
            sell_score += comp_weight * 1.2
            reasons.append(f"RSI {indicators['rsi']:.0f}")
            
        # Stochastic
        if indicators['stoch_k'] < 20 and indicators['stoch_d'] < 20:
            buy_score += comp_weight * 0.9
            reasons.append("Stoch Oversold")
        elif indicators['stoch_k'] > 80 and indicators['stoch_d'] > 80:
            sell_score += comp_weight * 0.9
            reasons.append("Stoch Overbought")
            
        # ADX trend strength
        if indicators['adx'] > 25:
            if indicators['plus_di'] > indicators['minus_di']:
                buy_score += comp_weight * 0.8
                reasons.append(f"Strong Trend ADX {indicators['adx']:.0f}")
            else:
                sell_score += comp_weight * 0.8
                reasons.append(f"Strong Trend ADX {indicators['adx']:.0f}")
                
        # Ichimoku
        if indicators['above_cloud'] > 0:
            buy_score += comp_weight * 1.5
            reasons.append("Above Ichimoku Cloud")
        elif indicators['below_cloud'] > 0:
            sell_score += comp_weight * 1.5
            reasons.append("Below Ichimoku Cloud")
            
        # === CALCULATE FINAL SCORES ===
        
        total_possible = sum(weights.values()) * 10  # Approximate max score
        buy_confidence = buy_score / total_possible
        sell_confidence = sell_score / total_possible
        
        # ULTRA AGGRESSIVE - Determine signal with minimal requirements
        signal_type = None
        confidence = 0
        
        # Much lower thresholds
        min_conf = CONFIG.get("MIN_CONFIDENCE", 0.10)
        
        if buy_score > sell_score and buy_confidence >= min_conf:
            signal_type = SignalType.BUY
            confidence = buy_confidence
        elif sell_score > buy_score and sell_confidence >= min_conf:
            signal_type = SignalType.SELL
            confidence = sell_confidence
        elif buy_score > 2:  # Even with low confidence, if we have some signals
            signal_type = SignalType.BUY
            confidence = max(buy_confidence, 0.15)
        elif sell_score > 2:
            signal_type = SignalType.SELL
            confidence = max(sell_confidence, 0.15)
            
        # ULTRA SIMPLIFIED - Count ANY positive indicator as active
        active_strategies = 0
        strategy_scores = {}
        
        # Just count everything that's positive
        for key, value in indicators.items():
            if isinstance(value, (int, float)) and value > 0:
                active_strategies += 1
                
        # Create simple strategy scores for display
        strategy_scores['Active'] = active_strategies / 100
        strategy_scores['Confidence'] = confidence
        
        # FORCE more strategies to be "active" if needed
        if active_strategies < CONFIG.get("MIN_INDICATORS", 1):
            active_strategies = CONFIG.get("MIN_INDICATORS", 1)
            logger.warning(f"Forcing active strategies to minimum: {active_strategies}")
        
        # Send to visualizer
        if self.signal_queue:
            try:
                self.signal_queue.put({
                    symbol: {
                        'type': signal_type.value if signal_type else 'NONE',
                        'confidence': confidence,
                        'strategies': strategy_scores,
                        'reasons': reasons[:5],
                        'quality': confidence  # Simplified quality = confidence
                    }
                })
            except:
                pass
                
        # Log analysis results
        logger.debug(f"{symbol} - Buy: {buy_score:.2f}, Sell: {sell_score:.2f}, Active: {active_strategies}")
        
        # ULTRA AGGRESSIVE - Generate signal with minimal criteria
        if signal_type and active_strategies >= CONFIG.get("MIN_INDICATORS", 1):
            logger.info(f"🎯 SIGNAL FOUND for {symbol}: {signal_type.value} conf={confidence:.1%}")
            
            # Calculate dynamic SL/TP
            atr = indicators['atr']
            
            # Adjust for exotic pairs - wider stops due to higher volatility
            is_exotic = self._is_exotic_pair(symbol)
            
            if is_exotic:
                sl_multiplier = 2.0  # Wider stops for exotic pairs
                tp_multiplier = CONFIG["MIN_RR_EXOTIC"]  # Higher RR for exotic pairs
                min_sl = 0.0020 if "JPY" not in symbol else 0.20  # Larger minimum SL
            else:
                sl_multiplier = 1.0 if CONFIG["AGGRESSIVE_MODE"] else 1.5
                tp_multiplier = CONFIG["MIN_RR_RATIO"]
                min_sl = 0.0010 if "JPY" not in symbol else 0.10
            
            sl_distance = max(atr * sl_multiplier, min_sl)
            tp_distance = sl_distance * tp_multiplier
            
            current_price = indicators['current_price']
            
            if signal_type == SignalType.BUY:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                
            return Signal(
                type=signal_type,
                confidence=confidence,
                entry=current_price,
                sl=round(sl, 5),
                tp=round(tp, 5),
                reason=" | ".join(reasons[:3]),
                strategies=strategy_scores,
                quality=confidence
            )
            
        return None
        
    def _force_trade_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """FORCE a trade signal when no trades happening - ULTRA AGGRESSIVE"""
        if len(df) < 10:
            return None
            
        close = df['close'].values
        current = close[-1]
        
        # Simple momentum check
        momentum = (close[-1] - close[-5]) / close[-5]
        
        # Force a trade based on ANY movement
        if momentum > 0:
            signal_type = SignalType.BUY
            reason = "FORCED: Positive momentum"
        else:
            signal_type = SignalType.SELL
            reason = "FORCED: Negative momentum"
            
        # Minimal SL/TP
        sl_distance = 0.0010 if "JPY" not in symbol else 0.10
        tp_distance = sl_distance * 0.5  # Very small TP for quick profits
        
        if signal_type == SignalType.BUY:
            sl = current - sl_distance
            tp = current + tp_distance
        else:
            sl = current + sl_distance
            tp = current - tp_distance
            
        logger.warning(f"⚠️ FORCING TRADE on {symbol} - No trades in {(time.time() - self.last_global_trade_time)/60:.1f} minutes")
        
        return Signal(
            type=signal_type,
            confidence=0.15,  # Low confidence but enough to trade
            entry=current,
            sl=round(sl, 5),
            tp=round(tp, 5),
            reason=reason,
            strategies={'FORCED': 1.0},
            quality=0.1
        )
        
    def _check_account_safety(self) -> Dict[str, Any]:
        """Check account safety metrics"""
        try:
            resp = requests.get(f"{self.api_base}/account/", timeout=5)
            if resp.status_code == 200:
                account = resp.json()
                
                balance = account.get('balance', 0)
                equity = account.get('equity', balance)
                margin = account.get('margin', 0)
                free_margin = account.get('margin_free', equity - margin)
                margin_level = (equity / margin * 100) if margin > 0 else float('inf')
                
                return {
                    'balance': balance,
                    'equity': equity,
                    'margin': margin,
                    'free_margin': free_margin,
                    'margin_level': margin_level,
                    'profit': account.get('profit', 0),
                    'is_safe': margin_level > 200 and free_margin > balance * 0.3
                }
        except Exception as e:
            logger.error(f"Failed to check account safety: {e}")
            return {'is_safe': False, 'margin_level': 0}
    
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can safely trade this symbol"""
        # Check account safety first
        account_status = self._check_account_safety()
        if not account_status.get('is_safe', False):
            logger.warning(f"❌ Account not safe! Margin Level: {account_status.get('margin_level', 0):.1f}%")
            return False
        
        # Check symbol cooldown
        if symbol in self.last_trade_time:
            time_since_last = time.time() - self.last_trade_time[symbol]
            if time_since_last < CONFIG["POSITION_INTERVAL"]:
                return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.balance * CONFIG["MAX_DAILY_LOSS"]:
            logger.warning(f"Daily loss limit reached: ¥{self.daily_pnl:,.0f}")
            return False
        
        # Check concurrent positions
        if len(self.active_trades) >= CONFIG["MAX_CONCURRENT"]:
            return False
            
        return True
        
    def _should_force_trade(self) -> bool:
        """Check if we should force a trade due to inactivity"""
        time_since_last = time.time() - self.last_global_trade_time
        return time_since_last > CONFIG.get("FORCE_TRADE_INTERVAL", 600)
        
    def _place_order(self, symbol: str, signal: Signal) -> bool:
        """Place order with proper risk management"""
        # Pre-trade validation
        account_status = self._check_account_safety()
        if not account_status.get('is_safe', False):
            logger.error(f"Cannot place order - account not safe! Margin Level: {account_status.get('margin_level', 0):.1f}%")
            return False
            
        volume = self._calculate_position_size(symbol, abs(signal.entry - signal.sl))
        
        # Estimate required margin
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info:
            contract_size = symbol_info.get('trade_contract_size', 100000)
            leverage = 100  # Typical forex leverage
            required_margin = (volume * contract_size * signal.entry) / leverage
            
            if required_margin > account_status.get('free_margin', 0) * 0.5:
                logger.warning(f"Insufficient margin for {symbol}. Required: ¥{required_margin:,.0f}, Available: ¥{account_status.get('free_margin', 0):,.0f}")
                return False
        
        order = {
            "action": 1,
            "symbol": symbol,
            "volume": volume,
            "type": 0 if signal.type == SignalType.BUY else 1,
            "sl": signal.sl,
            "tp": signal.tp,
            "comment": f"Safe: {signal.reason[:10]}"
        }
        
        logger.info(f"🎯 SIGNAL: {signal.type.value} {symbol} @ {signal.entry:.5f}")
        logger.info(f"   SL: {signal.sl:.5f} TP: {signal.tp:.5f} Conf: {signal.confidence:.1%}")
        logger.info(f"   Volume: {volume} Margin Level: {account_status.get('margin_level', float('inf')):.1f}%")
        
        try:
            resp = requests.post(f"{self.api_base}/trading/orders", json=order, timeout=10)
            if resp.status_code == 201:
                result = resp.json()
                trade = Trade(
                    ticket=result.get('order'),
                    symbol=symbol,
                    type=signal.type.value,
                    entry_price=result.get('price', signal.entry),
                    sl=signal.sl,
                    tp=signal.tp,
                    volume=volume,
                    entry_time=datetime.now()
                )
                
                self.active_trades[trade.ticket] = trade
                self.daily_trades += 1
                self.last_trade_time[symbol] = time.time()
                self.last_global_trade_time = time.time()  # Track global trade time
                self.trades_this_hour += 1
                
                logger.info(f"✅ Trade opened: {trade.ticket} | Total today: {self.daily_trades}")
                return True
        except Exception as e:
            logger.error(f"Order failed: {e}")
            
        return False
        
    def _manage_positions(self):
        """Aggressive position management"""
        try:
            resp = requests.get(f"{self.api_base}/trading/positions", timeout=5)
            if resp.status_code != 200:
                return
                
            positions = resp.json()
            open_tickets = {p['ticket'] for p in positions}
            
            # Check closed
            for ticket in list(self.active_trades.keys()):
                if ticket not in open_tickets:
                    logger.info(f"Trade {ticket} closed")
                    del self.active_trades[ticket]
                    
            # Manage open positions aggressively
            for pos in positions:
                ticket = pos['ticket']
                if ticket not in self.active_trades:
                    continue
                    
                trade = self.active_trades[ticket]
                profit = pos.get('profit', 0)
                duration = (datetime.now() - trade.entry_time).seconds
                
                # ULTRA AGGRESSIVE exits
                # Take ANY profit quickly
                if profit > 100:  # Just 100 JPY
                    logger.info(f"Quick profit: {ticket} +¥{profit:,.0f}")
                    self._close_position(ticket)
                    
                # Very quick time-based exit
                elif duration > 300:  # 5 minutes
                    if profit > 0:
                        logger.info(f"Time exit (profit): {ticket} +¥{profit:,.0f}")
                        self._close_position(ticket)
                    elif duration > 600:  # 10 minutes for losses
                        logger.info(f"Time exit (timeout): {ticket}")
                        self._close_position(ticket)
                        
                # Quick breakeven
                elif profit > 50 and duration > 60:  # 50 JPY after 1 minute
                    self._move_breakeven(ticket, trade)
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")
            
    def _close_position(self, ticket: int):
        """Close position"""
        try:
            requests.delete(f"{self.api_base}/trading/positions/{ticket}", timeout=5)
        except:
            pass
            
    def _move_breakeven(self, ticket: int, trade: Trade):
        """Move to breakeven"""
        if hasattr(trade, 'be_moved'):
            return
            
        pip = 0.01 if "JPY" in trade.symbol else 0.0001
        new_sl = trade.entry_price + pip if trade.type == "BUY" else trade.entry_price - pip
        
        try:
            resp = requests.patch(
                f"{self.api_base}/trading/positions/{ticket}",
                json={"sl": new_sl, "tp": trade.tp},
                timeout=5
            )
            if resp.status_code == 200:
                trade.be_moved = True
                logger.info(f"Breakeven: {ticket}")
        except:
            pass
            
    def _run_loop(self):
        """Main trading loop - ULTRA AGGRESSIVE WITH FORCE TRADING"""
        cycle = 0
        last_hour = datetime.now().hour
        symbols_traded = set()
        
        # Initialize last trade time
        self.last_global_trade_time = time.time()
        
        try:
            while self.running:
                cycle += 1
                
                # Hourly reset
                current_hour = datetime.now().hour
                if current_hour != last_hour:
                    if current_hour == 0:
                        self.daily_pnl = 0
                        self.daily_trades = 0
                        symbols_traded.clear()
                    self.trades_this_hour = 0
                    last_hour = current_hour
                    
                # Manage positions
                self._manage_positions()
                
                # CHECK IF WE NEED TO FORCE A TRADE
                force_trade = self._should_force_trade()
                if force_trade:
                    logger.warning(f"⚠️ FORCE TRADE MODE - No trades in {(time.time()-self.last_global_trade_time)/60:.1f} minutes!")
                    self.force_trade_attempts += 1
                
                # Aggressive trading - try to trade all symbols
                traded_this_cycle = False
                signal_found = False
                
                # Shuffle symbols for variety
                import random
                shuffled_symbols = self.tradable_symbols.copy()
                random.shuffle(shuffled_symbols)
                
                for symbol in shuffled_symbols:
                    # In force mode, trade ANY symbol
                    if not force_trade and not self._can_trade_symbol(symbol):
                        continue
                        
                    # Check spread
                    spread_ok, spread = self._check_spread(symbol)
                    if not spread_ok and not force_trade:
                        if cycle % 50 == 0:  # Log occasionally
                            logger.debug(f"Spread too wide for {symbol}: {spread:.1f} pips")
                        continue
                    
                    # Get data
                    df = self._get_market_data(symbol, count=100)
                    if df is None:
                        continue
                        
                    # Try normal analysis first
                    signal = self._analyze_ultra(symbol, df)
                    
                    # If no signal and force mode, create one
                    if not signal and force_trade:
                        signal = self._force_trade_signal(symbol, df)
                        
                    if signal:
                        signal_found = True
                        # Always try to place order in ULTRA mode
                        if self._place_order(symbol, signal):
                            traded_this_cycle = True
                            symbols_traded.add(symbol)
                            force_trade = False  # Reset force mode
                            
                            # Don't break - keep looking for more trades!
                            if len(self.active_trades) >= CONFIG["MAX_CONCURRENT"]:
                                break
                                
                # If still no trades after checking all symbols, lower thresholds even more
                if force_trade and not traded_this_cycle:
                    logger.error(f"🚨 CRITICAL: No trades found even in force mode! Attempts: {self.force_trade_attempts}")
                    # Take the first available symbol and force it
                    for symbol in self.tradable_symbols[:5]:  # Try first 5 symbols
                        df = self._get_market_data(symbol, count=20)
                        if df is not None:
                            signal = self._force_trade_signal(symbol, df)
                            if signal and self._place_order(symbol, signal):
                                logger.warning(f"🔥 EMERGENCY TRADE PLACED on {symbol}")
                                break
                                
                # Status update
                if cycle % 30 == 0:  # Every 5 minutes
                    account_status = self._check_account_safety()
                    time_since_trade = (time.time() - self.last_global_trade_time) / 60
                    logger.info(f"=== TRADING STATUS ===")
                    logger.info(f"📊 Active: {len(self.active_trades)}/{CONFIG['MAX_CONCURRENT']} | Daily trades: {self.daily_trades}")
                    logger.info(f"💰 Equity: ¥{account_status.get('equity', 0):,.0f} | Margin Level: {account_status.get('margin_level', float('inf')):.1f}%")
                    logger.info(f"⏱️  Last trade: {time_since_trade:.1f} min ago")
                    logger.info(f"📈 Daily P&L: ¥{self.daily_pnl:,.0f} ({self.daily_pnl/self.balance*100:.2f}%)")
                    
                    if time_since_trade > 5:
                        logger.warning(f"⚠️ WARNING: No trades in {time_since_trade:.1f} minutes!")
                    
                time.sleep(10)  # Check every 10 seconds for more aggressive scanning
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.running = False
            
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        logger.info("Stopping Ultra Trading Engine")

def main():
    """Run ultra trading engine"""
    engine = UltraTradingEngine()
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()

if __name__ == "__main__":
    main()