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
            response = requests.get(f"{self.api_base}/market/symbols", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('symbols', [])
            return []
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        try:
            response = requests.get(f"{self.api_base}/market/symbols/{quote(symbol)}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current price for a symbol"""
        try:
            response = requests.post(
                f"{self.api_base}/market/tick",
                json={"symbol": symbol},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'bid': data.get('bid', 0),
                    'ask': data.get('ask', 0)
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
            if response.status_code == 200:
                data = response.json()
                return data.get('ticket')
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
                return data.get('positions', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        try:
            response = requests.delete(
                f"{self.api_base}/trading/positions/{ticket}",
                timeout=5
            )
            return response.status_code == 200
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