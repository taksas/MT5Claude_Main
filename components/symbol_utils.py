#!/usr/bin/env python3
"""
Symbol Utilities Module - Simplified
Handles essential symbol classification
"""

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