#!/usr/bin/env python3
"""
MT5 API Client Module
Handles all HTTP communication with the MT5 Bridge API
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote

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