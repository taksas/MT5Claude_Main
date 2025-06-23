#!/usr/bin/env python3
"""
Order Management Module
Handles order execution, position management, and trade modifications
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from .mt5_api_client import MT5APIClient
from .trading_models import Trade, Signal, SignalType
from .symbol_utils import SymbolUtils

logger = logging.getLogger('OrderManagement')

class OrderManagement:
    def __init__(self, api_client: MT5APIClient):
        self.api_client = api_client
        self.symbol_utils = SymbolUtils()
        
    def place_order(self, signal: Signal, symbol: str, volume: float) -> Optional[int]:
        """Place trading order"""
        try:
            # Prepare order according to API specification
            order = {
                "action": 1,  # TRADE_ACTION_DEAL (market order)
                "symbol": symbol,
                "volume": volume,
                "type": 0 if signal.type == SignalType.BUY else 1,  # ORDER_TYPE_BUY or ORDER_TYPE_SELL
                "sl": signal.sl,
                "tp": signal.tp,
                "comment": f"Ultra100_{signal.reason[:20]}",
                "deviation": 20,  # Allow 20 points deviation
                "magic": 100100  # Magic number for identification
            }
            
            logger.info(f"📊 Placing {signal.type.value} order for {symbol}")
            logger.info(f"   Volume: {volume}, Entry: {signal.entry}, SL: {signal.sl}, TP: {signal.tp}")
            
            ticket = self.api_client.place_order(order)
            
            if ticket:
                logger.info(f"✅ Order placed successfully! Ticket: {ticket}")
                return ticket
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def manage_positions(self, active_trades: Dict[str, Trade]) -> Dict[str, Any]:
        """Manage open positions"""
        results = {
            'closed': [],
            'modified': [],
            'errors': []
        }
        
        try:
            positions = self.api_client.get_positions()
            
            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol')
                
                if ticket not in active_trades:
                    continue
                
                trade = active_trades[ticket]
                current_price = position.get('price_current')
                profit = position.get('profit', 0)
                
                # Check if position should be managed
                if self._should_move_to_breakeven(trade, current_price, profit):
                    if self._move_breakeven(ticket, trade, current_price):
                        results['modified'].append(ticket)
                        logger.info(f"🎯 Moved position {ticket} to breakeven")
                
                # Check for early exit conditions
                if self._should_close_early(trade, current_price, profit):
                    if self.close_position(ticket):
                        results['closed'].append(ticket)
                        logger.info(f"🔒 Closed position {ticket} early")
                        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        try:
            success = self.api_client.close_position(ticket)
            if success:
                logger.info(f"✅ Position {ticket} closed successfully")
            else:
                logger.error(f"❌ Failed to close position {ticket}")
            return success
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def _move_breakeven(self, ticket: int, trade: Trade, current_price: float) -> bool:
        """Move stop loss to breakeven"""
        try:
            # Calculate new stop loss (entry + small buffer for costs)
            spread_buffer = abs(trade.entry_price * 0.0001)  # 1 pip buffer
            
            if trade.type == "BUY":
                new_sl = trade.entry_price + spread_buffer
                if new_sl >= current_price:  # Don't set SL above current price
                    return False
            else:
                new_sl = trade.entry_price - spread_buffer
                if new_sl <= current_price:  # Don't set SL below current price
                    return False
            
            # Only move if new SL is better than current
            if trade.type == "BUY" and new_sl <= trade.sl:
                return False
            elif trade.type == "SELL" and new_sl >= trade.sl:
                return False
            
            return self.api_client.modify_position(ticket, new_sl, trade.tp)
            
        except Exception as e:
            logger.error(f"Error moving to breakeven for {ticket}: {e}")
            return False
    
    def _should_move_to_breakeven(self, trade: Trade, current_price: float, 
                                 profit: float) -> bool:
        """Check if position should be moved to breakeven"""
        try:
            # Only move to BE if in profit
            if profit <= 0:
                return False
            
            # Check if price has moved enough
            if trade.type == "BUY":
                price_move = current_price - trade.entry_price
                target_move = (trade.tp - trade.entry_price) * 0.5  # 50% to TP
            else:
                price_move = trade.entry_price - current_price
                target_move = (trade.entry_price - trade.tp) * 0.5
            
            # Move to BE if reached 50% of target
            return price_move >= target_move and trade.sl != trade.entry_price
            
        except Exception as e:
            logger.error(f"Error checking breakeven condition: {e}")
            return False
    
    def _should_close_early(self, trade: Trade, current_price: float, 
                           profit: float) -> bool:
        """Check if position should be closed early"""
        try:
            # Close if position has been open too long (4 hours)
            time_open = (datetime.now() - trade.entry_time).total_seconds()
            if time_open > 14400:  # 4 hours
                return profit > 0  # Only close if in profit
            
            # Close if reversal detected
            if trade.type == "BUY":
                # If price drops significantly from high
                high_since_entry = max(current_price, trade.entry_price * 1.01)
                pullback = (high_since_entry - current_price) / high_since_entry
                if pullback > 0.005:  # 0.5% pullback
                    return True
            else:
                # If price rises significantly from low
                low_since_entry = min(current_price, trade.entry_price * 0.99)
                pullback = (current_price - low_since_entry) / low_since_entry
                if pullback > 0.005:  # 0.5% pullback
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking early close condition: {e}")
            return False
    
    def get_position_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific position"""
        try:
            positions = self.api_client.get_positions()
            for position in positions:
                if position.get('ticket') == ticket:
                    return position
            return None
        except Exception as e:
            logger.error(f"Error getting position info for {ticket}: {e}")
            return None
    
    def close_all_positions(self) -> int:
        """Close all open positions"""
        closed_count = 0
        try:
            positions = self.api_client.get_positions()
            for position in positions:
                ticket = position.get('ticket')
                if self.close_position(ticket):
                    closed_count += 1
                    time.sleep(0.5)  # Small delay between closures
            
            logger.info(f"Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return closed_count