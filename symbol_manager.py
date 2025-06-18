#!/usr/bin/env python3
"""
Symbol Manager - Dynamically fetch and rank tradable symbols
"""

import requests
import logging
from typing import List, Dict, Tuple
import time

logger = logging.getLogger(__name__)

class SymbolManager:
    """Manages symbol selection and ranking"""
    
    def __init__(self, api_base="http://172.28.144.1:8000"):
        self.api_base = api_base
        self.all_symbols = []
        self.symbol_info_cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.last_update = 0
        
        # Ranking criteria
        self.major_pairs = ["EURUSD#", "USDJPY#", "GBPUSD#", "AUDUSD#", "USDCAD#", "USDCHF#", "NZDUSD#"]
        self.high_volume_pairs = ["EURJPY#", "GBPJPY#", "EURGBP#", "AUDJPY#", "CADJPY#", "EURAUD#"]
        
    def fetch_tradable_symbols(self) -> List[str]:
        """Fetch all tradable symbols from API"""
        try:
            response = requests.get(f"{self.api_base}/market/symbols/tradable")
            if response.status_code == 200:
                self.all_symbols = response.json()
                self.last_update = time.time()
                logger.info(f"Fetched {len(self.all_symbols)} tradable symbols")
                return self.all_symbols
            else:
                logger.error(f"Failed to fetch symbols: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed info for a symbol"""
        # Check cache
        if symbol in self.symbol_info_cache:
            info, timestamp = self.symbol_info_cache[symbol]
            if time.time() - timestamp < self.cache_expiry:
                return info
        
        try:
            response = requests.get(f"{self.api_base}/market/symbols/{symbol}")
            if response.status_code == 200:
                info = response.json()
                self.symbol_info_cache[symbol] = (info, time.time())
                return info
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return {}
    
    def calculate_spread_cost(self, symbol_info: Dict) -> float:
        """Calculate spread cost in percentage"""
        if 'bid' in symbol_info and 'ask' in symbol_info and symbol_info['bid'] > 0:
            spread = symbol_info['ask'] - symbol_info['bid']
            return (spread / symbol_info['bid']) * 100
        return float('inf')
    
    def rank_symbols_for_scalping(self, max_symbols: int = 5) -> List[str]:
        """Rank symbols for scalping based on spread and liquidity"""
        if time.time() - self.last_update > self.cache_expiry:
            self.fetch_tradable_symbols()
        
        if not self.all_symbols:
            # Fallback to known good symbols
            return ["EURUSD#", "USDJPY#", "GBPUSD#", "EURJPY#", "AUDUSD#"][:max_symbols]
        
        symbol_scores = []
        
        for symbol in self.all_symbols:
            info = self.get_symbol_info(symbol)
            if not info or 'bid' not in info:
                continue
            
            # Calculate score based on multiple factors
            spread_cost = self.calculate_spread_cost(info)
            
            # Skip if spread is too wide (> 0.05% for scalping)
            if spread_cost > 0.05:
                continue
            
            score = 0
            
            # Prefer major pairs (highest liquidity)
            if symbol in self.major_pairs:
                score += 100
            elif symbol in self.high_volume_pairs:
                score += 50
            
            # Prefer tighter spreads
            score += (0.05 - spread_cost) * 1000
            
            # JPY pairs often have good volatility for scalping
            if "JPY#" in symbol:
                score += 10
            
            symbol_scores.append((symbol, score, spread_cost))
        
        # Sort by score (descending)
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log the top symbols
        logger.info("Top symbols for scalping:")
        for i, (symbol, score, spread) in enumerate(symbol_scores[:max_symbols]):
            logger.info(f"  {i+1}. {symbol} - Score: {score:.1f}, Spread: {spread:.4f}%")
        
        return [s[0] for s in symbol_scores[:max_symbols]]
    
    def get_best_symbol_now(self, exclude_symbols: List[str] = None) -> str:
        """Get the single best symbol to trade right now"""
        ranked = self.rank_symbols_for_scalping(10)
        
        if exclude_symbols:
            for symbol in ranked:
                if symbol not in exclude_symbols:
                    return symbol
        
        return ranked[0] if ranked else "EURUSD#"

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = SymbolManager()
    symbols = manager.fetch_tradable_symbols()
    
    print(f"\nAll tradable symbols ({len(symbols)}):")
    print(symbols)
    
    print("\nBest symbols for scalping:")
    best = manager.rank_symbols_for_scalping(5)
    for i, symbol in enumerate(best, 1):
        print(f"{i}. {symbol}")
    
    print(f"\nBest single symbol now: {manager.get_best_symbol_now()}")