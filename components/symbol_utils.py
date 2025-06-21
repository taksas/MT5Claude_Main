#!/usr/bin/env python3
"""
Symbol Utilities Module
Handles symbol classification and identification
"""

from typing import List
from .trading_config import CONFIG

class SymbolUtils:
    def __init__(self):
        self.exotic_currencies = CONFIG["EXOTIC_CURRENCIES"]
        self.metal_symbols = CONFIG["METAL_SYMBOLS"]
        self.crypto_symbols = CONFIG["CRYPTO_SYMBOLS"]
        self.index_symbols = CONFIG["INDEX_SYMBOLS"]
    
    def is_exotic_pair(self, symbol: str) -> bool:
        """Check if symbol is an exotic currency pair"""
        return any(exotic in symbol for exotic in self.exotic_currencies)
    
    def is_metal_pair(self, symbol: str) -> bool:
        """Check if symbol is a metal pair"""
        return any(metal in symbol for metal in self.metal_symbols)
    
    def is_crypto_pair(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency pair"""
        return any(crypto in symbol for crypto in self.crypto_symbols)
    
    def is_index_pair(self, symbol: str) -> bool:
        """Check if symbol is an index"""
        return any(index in symbol for index in self.index_symbols)
    
    def get_instrument_type(self, symbol: str) -> str:
        """Get the instrument type of a symbol"""
        if self.is_crypto_pair(symbol):
            return 'crypto'
        elif self.is_metal_pair(symbol):
            return 'metal'
        elif self.is_index_pair(symbol):
            return 'index'
        elif self.is_exotic_pair(symbol):
            return 'exotic'
        else:
            return 'major'
    
    def filter_symbols(self, symbols: List[str], include_types: List[str] = None) -> List[str]:
        """Filter symbols by instrument type"""
        if include_types is None:
            include_types = ['major', 'exotic', 'metal', 'crypto', 'index']
        
        filtered = []
        for symbol in symbols:
            instrument_type = self.get_instrument_type(symbol)
            if instrument_type in include_types:
                filtered.append(symbol)
        
        return filtered
    
    def get_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol"""
        # Most forex pairs are 6 characters (EURUSD)
        # Some might be longer (EURUSD.m, EURUSD_i, etc.)
        clean_symbol = symbol.split('.')[0].split('_')[0]
        
        if len(clean_symbol) >= 6:
            return clean_symbol[:3]
        return ""
    
    def get_quote_currency(self, symbol: str) -> str:
        """Extract quote currency from symbol"""
        clean_symbol = symbol.split('.')[0].split('_')[0]
        
        if len(clean_symbol) >= 6:
            return clean_symbol[3:6]
        return ""
    
    def is_jpy_pair(self, symbol: str) -> bool:
        """Check if symbol involves JPY"""
        return 'JPY' in symbol.upper()
    
    def get_pip_value(self, symbol: str, digits: int) -> float:
        """Get pip value for a symbol based on digits"""
        if self.is_jpy_pair(symbol) and not self.is_metal_pair(symbol):
            # JPY pairs typically have 3 digits, pip is 0.01
            return 0.01
        elif digits == 3 or digits == 5:
            # 5 digit broker, pip is 0.0001
            return 0.0001
        elif digits == 2 or digits == 4:
            # 4 digit broker, pip is 0.01 or 0.0001
            return 0.01 if digits == 2 else 0.0001
        else:
            # Default
            return 0.0001