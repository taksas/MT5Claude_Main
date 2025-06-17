#!/usr/bin/env python3
"""
MT5 Automated Forex Trading System
Connects to MT5 Bridge API and performs automated short-term trading with strict risk management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_session.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MT5TradingBot:
    def __init__(self, api_base_url: str = "http://172.28.144.1:8000"):
        self.api_base_url = api_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.lot_size = 0.01  # Fixed lot size as per requirements
        self.max_positions = 3  # Maximum concurrent positions
        self.min_profit_pips = 5  # Minimum profit target in pips
        self.max_loss_pips = 10  # Maximum loss in pips (stop loss)
        self.position_hold_time = timedelta(minutes=30)  # Maximum hold time
        self.active_positions = {}
        # Only use symbols that actually work with the API
        self.verified_working_symbols = [
            "USDCNH#", "USDDKK#", "USDHKD#", "USDHUF#", "USDMXN#", 
            "USDNOK#", "USDPLN#", "USDSEK#", "USDSGD#", "USDTRY#", 
            "USDZAR#", "EURUSD#", "GBPUSD#", "USDCAD#", "USDCHF#", 
            "USDJPY#", "AUDUSD#", "NZDUSD#"
        ]
        self.trading_symbols = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def api_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API request to MT5 Bridge"""
        url = f"{self.api_base_url}{endpoint}"
        try:
            async with self.session.request(method, url, json=data) as response:
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"API Request failed: {e}")
            return None
    
    async def check_connection(self) -> bool:
        """Check if MT5 Bridge API is accessible"""
        result = await self.api_request("GET", "/status/ping")
        if result and result.get("status") == "pong":
            logger.info("✓ MT5 Bridge API connection successful")
            return True
        logger.error("✗ Failed to connect to MT5 Bridge API")
        return False
    
    async def check_mt5_status(self) -> bool:
        """Check MT5 terminal connection status"""
        result = await self.api_request("GET", "/status/mt5")
        if result and result.get("connected"):
            logger.info("✓ MT5 Terminal connected and ready")
            return True
        logger.error("✗ MT5 Terminal not connected")
        return False
    
    async def get_account_info(self) -> dict:
        """Get account information"""
        return await self.api_request("GET", "/account/")
    
    async def get_tradable_symbols(self) -> List[str]:
        """Get list of verified working symbols"""
        # Use pre-verified symbols that actually work with the API
        logger.info(f"Using {len(self.verified_working_symbols)} verified working symbols: {self.verified_working_symbols}")
        return self.verified_working_symbols
    
    async def get_symbol_info(self, symbol: str) -> dict:
        """Get detailed symbol information"""
        return await self.api_request("GET", f"/market/symbols/{symbol}")
    
    async def get_historical_data(self, symbol: str, timeframe: str = "M5", count: int = 100) -> List[dict]:
        """Get historical price data"""
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        return await self.api_request("POST", "/market/history", data)
    
    def calculate_technical_indicators(self, data: List[dict]) -> dict:
        """Calculate basic technical indicators"""
        if len(data) < 20:
            return {}
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Current values
        latest = df.iloc[-1]
        return {
            'current_price': latest['close'],
            'sma_5': latest['sma_5'],
            'sma_20': latest['sma_20'],
            'rsi': latest['rsi'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle']
        }
    
    def generate_trading_signal(self, symbol: str, indicators: dict, symbol_info: dict) -> Optional[dict]:
        """Generate trading signal based on technical analysis"""
        if not indicators or any(pd.isna(v) for v in indicators.values() if isinstance(v, (int, float))):
            return None
        
        current_price = indicators['current_price']
        sma_5 = indicators['sma_5']
        sma_20 = indicators['sma_20']
        rsi = indicators['rsi']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        point = symbol_info.get('point', 0.00001)
        
        signal = None
        
        # Strategy 1: SMA Crossover with RSI confirmation
        if sma_5 > sma_20 and rsi < 70 and current_price > sma_5:
            # Bullish signal
            stop_loss = current_price - (self.max_loss_pips * point)
            take_profit = current_price + (self.min_profit_pips * point)
            signal = {
                'type': 'BUY',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'SMA_Crossover_RSI'
            }
        elif sma_5 < sma_20 and rsi > 30 and current_price < sma_5:
            # Bearish signal
            stop_loss = current_price + (self.max_loss_pips * point)
            take_profit = current_price - (self.min_profit_pips * point)
            signal = {
                'type': 'SELL',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'SMA_Crossover_RSI'
            }
        
        # Strategy 2: Bollinger Bands Mean Reversion
        elif current_price <= bb_lower and rsi < 30:
            # Oversold, potential bounce
            stop_loss = current_price - (self.max_loss_pips * point)
            take_profit = current_price + (self.min_profit_pips * point)
            signal = {
                'type': 'BUY',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'BB_Mean_Reversion'
            }
        elif current_price >= bb_upper and rsi > 70:
            # Overbought, potential reversal
            stop_loss = current_price + (self.max_loss_pips * point)
            take_profit = current_price - (self.min_profit_pips * point)
            signal = {
                'type': 'SELL',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'BB_Mean_Reversion'
            }
        
        if signal:
            logger.info(f"Generated {signal['type']} signal for {symbol} using {signal['strategy']}")
        
        return signal
    
    async def place_trade(self, symbol: str, signal: dict) -> Optional[dict]:
        """Place a trade based on signal"""
        order_type = 0 if signal['type'] == 'BUY' else 1
        
        trade_request = {
            "action": 1,  # TRADE_ACTION_DEAL
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "sl": signal['stop_loss'],
            "tp": signal['take_profit'],
            "comment": f"Auto_{signal['strategy']}"
        }
        
        result = await self.api_request("POST", "/trading/orders", trade_request)
        if result and result.get('retcode') == 10009:
            logger.info(f"✓ Trade placed: {signal['type']} {symbol} at {signal['entry_price']}")
            return result
        else:
            logger.error(f"✗ Failed to place trade for {symbol}")
            return None
    
    async def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        return await self.api_request("GET", "/trading/positions") or []
    
    async def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        close_request = {"deviation": 20}
        result = await self.api_request("DELETE", f"/trading/positions/{ticket}", close_request)
        if result and result.get('retcode') == 10009:
            logger.info(f"✓ Position {ticket} closed successfully")
            return True
        else:
            logger.error(f"✗ Failed to close position {ticket}")
            return False
    
    async def manage_positions(self):
        """Manage open positions - check time limits and profit/loss"""
        positions = await self.get_open_positions()
        current_time = datetime.now()
        
        for position in positions:
            ticket = position['ticket']
            symbol = position['symbol']
            open_time = datetime.fromisoformat(position.get('time', '').replace('Z', '+00:00'))
            
            # Check if position has been held too long
            if current_time - open_time > self.position_hold_time:
                logger.info(f"Closing position {ticket} due to time limit")
                await self.close_position(ticket)
                continue
            
            # Check current profit/loss
            current_profit = position.get('profit', 0)
            if current_profit != 0:
                logger.info(f"Position {ticket} current P&L: {current_profit}")
    
    async def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting automated trading session...")
        
        # Initial setup
        if not await self.check_connection():
            return
        
        if not await self.check_mt5_status():
            return
        
        account_info = await self.get_account_info()
        if account_info:
            logger.info(f"Account Balance: {account_info.get('balance')} {account_info.get('currency')}")
        
        self.trading_symbols = await self.get_tradable_symbols()
        if not self.trading_symbols:
            logger.error("No tradable symbols found")
            return
        
        # Main trading loop
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"--- Trading Iteration {iteration} ---")
                
                # Manage existing positions
                await self.manage_positions()
                
                # Check current position count
                positions = await self.get_open_positions()
                current_positions = len(positions)
                
                if current_positions >= self.max_positions:
                    logger.info(f"Maximum positions ({self.max_positions}) reached, skipping new trades")
                else:
                    # Look for new trading opportunities
                    for symbol in self.trading_symbols:
                        if current_positions >= self.max_positions:
                            break
                        
                        # Get symbol info and historical data
                        symbol_info = await self.get_symbol_info(symbol)
                        if not symbol_info:
                            continue
                        
                        historical_data = await self.get_historical_data(symbol, "M5", 50)
                        if not historical_data:
                            continue
                        
                        # Calculate indicators and generate signal
                        indicators = self.calculate_technical_indicators(historical_data)
                        signal = self.generate_trading_signal(symbol, indicators, symbol_info)
                        
                        if signal:
                            # Place trade
                            result = await self.place_trade(symbol, signal)
                            if result:
                                current_positions += 1
                                # Wait between trades
                                await asyncio.sleep(2)
                
                # Wait before next iteration
                logger.info("Waiting 30 seconds before next analysis...")
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

async def main():
    """Main function to start the trading bot"""
    # Try different host IPs for WSL environment
    possible_hosts = [
        "172.28.144.1",  # Current WSL host IP
        "localhost",
        "127.0.0.1"
    ]
    
    bot = None
    for host in possible_hosts:
        try:
            api_url = f"http://{host}:8000"
            logger.info(f"Trying to connect to API at {api_url}")
            
            async with MT5TradingBot(api_url) as trading_bot:
                if await trading_bot.check_connection():
                    bot = trading_bot
                    break
        except Exception as e:
            logger.warning(f"Failed to connect to {host}: {e}")
            continue
    
    if bot is None:
        logger.error("Could not connect to MT5 Bridge API. Please ensure the bridge server is running.")
        return
    
    # Start trading
    await bot.trading_loop()

if __name__ == "__main__":
    asyncio.run(main())