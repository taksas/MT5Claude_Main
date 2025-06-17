import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from urllib.parse import quote

class MT5Client:
    def __init__(self, host: str = "172.28.144.1", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.available_symbols = []
    
    def ping(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/status/ping", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "pong"
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False
    
    def check_mt5_status(self) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/status/mt5", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"MT5 status check failed: {e}")
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/account/", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_tradable_symbols(self) -> List[str]:
        try:
            response = self.session.get(f"{self.base_url}/market/symbols/tradable", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get tradable symbols: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        try:
            encoded_symbol = quote(symbol, safe='')
            response = self.session.get(f"{self.base_url}/market/symbols/{encoded_symbol}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 100) -> List[Dict[str, Any]]:
        try:
            data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "count": count
            }
            response = self.session.post(f"{self.base_url}/market/history", 
                                       json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    def place_order(self, action: int, symbol: str, volume: float, order_type: int, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "") -> Dict[str, Any]:
        try:
            order_data = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "comment": comment
            }
            
            if price is not None:
                order_data["price"] = price
            if sl is not None:
                order_data["sl"] = sl
            if tp is not None:
                order_data["tp"] = tp
            
            response = self.session.post(f"{self.base_url}/trading/orders", 
                                       json=order_data, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            response = self.session.get(f"{self.base_url}/trading/positions", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> Dict[str, Any]:
        try:
            data = {}
            if volume is not None:
                data["volume"] = volume
            data["deviation"] = 20
            
            response = self.session.delete(f"{self.base_url}/trading/positions/{ticket}", 
                                         json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to close position {ticket}: {e}")
            raise
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict[str, Any]:
        try:
            data = {}
            if sl is not None:
                data["sl"] = sl
            if tp is not None:
                data["tp"] = tp
                
            response = self.session.patch(f"{self.base_url}/trading/positions/{ticket}", 
                                        json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to modify position {ticket}: {e}")
            raise

    def discover_available_symbols(self):
        """Discover all available trading symbols"""
        try:
            all_symbols = self.get_tradable_symbols()
            self.available_symbols = all_symbols
            self.logger.info(f"Discovered {len(all_symbols)} available symbols")
            return all_symbols
        except Exception as e:
            self.logger.error(f"Failed to discover symbols: {e}")
            return []
    
    def get_forex_symbols(self):
        """Get forex symbols from available symbols"""
        if not self.available_symbols:
            self.discover_available_symbols()
        
        # Common forex patterns
        forex_patterns = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        forex_symbols = []
        
        for symbol in self.available_symbols:
            # Check if symbol contains forex pairs
            if any(pattern in symbol for pattern in forex_patterns):
                forex_symbols.append(symbol)
        
        return forex_symbols

def test_connection():
    client = MT5Client()
    
    print("Testing MT5 API connection...")
    
    if not client.ping():
        print("❌ Server is not responding")
        return False
    print("✅ Server ping successful")
    
    try:
        mt5_status = client.check_mt5_status()
        if mt5_status.get("connected"):
            print("✅ MT5 connection successful")
            print(f"   Trade allowed: {mt5_status.get('trade_allowed')}")
            print(f"   DLLs allowed: {mt5_status.get('dlls_allowed')}")
        else:
            print("❌ MT5 not connected")
            return False
    except Exception as e:
        print(f"❌ MT5 status check failed: {e}")
        return False
    
    try:
        account_info = client.get_account_info()
        print("✅ Account info retrieved")
        print(f"   Balance: {account_info.get('balance', 0):.2f}")
        print(f"   Free margin: {account_info.get('margin_free', 0):.2f}")
        print(f"   Currency: {account_info.get('currency', 'N/A')}")
    except Exception as e:
        print(f"❌ Account info failed: {e}")
        return False
    
    try:
        all_symbols = client.discover_available_symbols()
        print(f"✅ Found {len(all_symbols)} total available symbols")
        
        # Show forex symbols specifically
        forex_symbols = client.get_forex_symbols()
        print(f"✅ Found {len(forex_symbols)} forex symbols")
        
        if forex_symbols:
            print("   Available forex pairs:")
            for i, symbol in enumerate(sorted(forex_symbols)[:10]):  # Show first 10
                print(f"     {symbol}")
            if len(forex_symbols) > 10:
                print(f"     ... and {len(forex_symbols) - 10} more")
        
        # Check for hash symbols specifically
        hash_symbols = [s for s in all_symbols if s.endswith('#')]
        if hash_symbols:
            print(f"   Hash symbols available: {hash_symbols[:5]}{'...' if len(hash_symbols) > 5 else ''}")
        else:
            print("   No hash symbols found - will use standard forex symbols")
            
    except Exception as e:
        print(f"❌ Symbol discovery failed: {e}")
        return False
    
    print("✅ All connectivity tests passed!")
    return True

if __name__ == "__main__":
    test_connection()