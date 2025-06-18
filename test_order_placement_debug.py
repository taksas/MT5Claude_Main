#!/usr/bin/env python3
"""Test script to debug order placement with the MT5 API"""

import requests
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE = "http://172.28.144.1:8000"

def test_api_connection():
    """Test basic API connectivity"""
    try:
        response = requests.get(f"{API_BASE}/status/mt5")
        if response.status_code == 200:
            status = response.json()
            logger.info(f"✅ API Status: {json.dumps(status, indent=2)}")
            return status.get('connected') and status.get('trade_allowed')
        else:
            logger.error(f"❌ API connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        return False

def get_symbol_info(symbol):
    """Get detailed symbol information"""
    try:
        response = requests.get(f"{API_BASE}/market/symbols/{symbol}")
        if response.status_code == 200:
            info = response.json()
            logger.info(f"📊 Symbol {symbol} info:")
            logger.info(f"   - Filling modes: {info.get('filling_mode', 'Not specified')}")
            logger.info(f"   - Trade mode: {info.get('trade_mode', 'Not specified')}")
            logger.info(f"   - Bid: {info.get('bid', 'N/A')}")
            logger.info(f"   - Ask: {info.get('ask', 'N/A')}")
            return info
        else:
            logger.error(f"Failed to get symbol info: {response.status_code}")
    except Exception as e:
        logger.error(f"Error getting symbol info: {e}")
    return None

def test_order_minimal(symbol="EURUSD#"):
    """Test order with minimal parameters (let API handle defaults)"""
    logger.info(f"\n🧪 Testing minimal order for {symbol}")
    
    # Get current price
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return False
    
    current_price = symbol_info.get('ask', 0)
    
    # Calculate simple SL/TP
    pip_value = 0.0001 if "JPY" not in symbol else 0.01
    sl = round(current_price - (10 * pip_value), 5)
    tp = round(current_price + (15 * pip_value), 5)
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": sl,
        "tp": tp,
        "comment": "Test minimal order"
    }
    
    logger.info(f"📤 Sending order: {json.dumps(order_data, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
        logger.info(f"📥 Response status: {response.status_code}")
        logger.info(f"📥 Response body: {response.text}")
        
        if response.status_code == 201:
            result = response.json()
            logger.info(f"✅ Order successful! Ticket: {result.get('order')}")
            return result.get('order')
        else:
            logger.error(f"❌ Order failed with status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Request error: {e}")
        return None

def test_order_with_filling(symbol="EURUSD#", filling_mode=None):
    """Test order with specific filling mode"""
    logger.info(f"\n🧪 Testing order with filling mode {filling_mode} for {symbol}")
    
    # Get current price
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return False
    
    current_price = symbol_info.get('ask', 0)
    
    # Calculate simple SL/TP
    pip_value = 0.0001 if "JPY" not in symbol else 0.01
    sl = round(current_price - (10 * pip_value), 5)
    tp = round(current_price + (15 * pip_value), 5)
    
    order_data = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": 0.01,
        "type": 0,  # BUY
        "sl": sl,
        "tp": tp,
        "comment": f"Test filling {filling_mode}"
    }
    
    if filling_mode is not None:
        order_data["type_filling"] = filling_mode
    
    logger.info(f"📤 Sending order: {json.dumps(order_data, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE}/trading/orders", json=order_data)
        logger.info(f"📥 Response status: {response.status_code}")
        logger.info(f"📥 Response body: {response.text}")
        
        if response.status_code == 201:
            result = response.json()
            logger.info(f"✅ Order successful! Ticket: {result.get('order')}")
            return result.get('order')
        else:
            logger.error(f"❌ Order failed with status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Request error: {e}")
        return None

def close_position(ticket):
    """Close a position by ticket"""
    if not ticket:
        return
        
    logger.info(f"\n🔄 Closing position {ticket}")
    try:
        response = requests.delete(f"{API_BASE}/trading/positions/{ticket}")
        if response.status_code == 200:
            logger.info(f"✅ Position {ticket} closed successfully")
        else:
            logger.error(f"❌ Failed to close position: {response.text}")
    except Exception as e:
        logger.error(f"❌ Error closing position: {e}")

def main():
    """Run order placement tests"""
    logger.info("="*60)
    logger.info("MT5 ORDER PLACEMENT DEBUG TEST")
    logger.info("="*60)
    
    # Check API connection
    if not test_api_connection():
        logger.error("Cannot proceed - API not ready")
        return
    
    test_symbol = "EURUSD#"  # Change this to test different symbols
    
    # Test 1: Minimal order (let API auto-determine filling)
    ticket1 = test_order_minimal(test_symbol)
    if ticket1:
        input("\nPress Enter to close this position and continue...")
        close_position(ticket1)
    
    # Test 2: With explicit filling modes
    # MT5 standard filling modes: FOK=0, IOC=1, RETURN=2
    for mode in [0, 1, 2]:
        ticket = test_order_with_filling(test_symbol, mode)
        if ticket:
            input(f"\nPress Enter to close position (filling mode {mode})...")
            close_position(ticket)
    
    logger.info("\n✅ Test completed!")

if __name__ == "__main__":
    main()