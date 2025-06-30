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
from .trading_config import Trade, Signal, SignalType, SymbolUtils

logger = logging.getLogger('OrderManagement')

class OrderManagement:
    def __init__(self, api_client: MT5APIClient):
        self.api_client = api_client
        self.symbol_utils = SymbolUtils()
        
    def place_order(self, signal: Signal, symbol: str, volume: float) -> Optional[int]:
        """Place trading order"""
        try:
            # Get symbol info for validation
            symbol_info = self.api_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Cannot get symbol info for {symbol}")
                return None
            
            # Get current price
            price_info = self.api_client.get_current_price(symbol)
            if not price_info:
                logger.error(f"Cannot get current price for {symbol}")
                return None
            
            current_price = price_info['ask'] if signal.type == SignalType.BUY else price_info['bid']
            
            # Validate and adjust stops
            sl, tp = self._validate_and_adjust_stops(
                symbol, symbol_info, signal.type, current_price, signal.sl, signal.tp
            )
            
            # Prepare order according to API specification
            order = {
                "action": 1,  # TRADE_ACTION_DEAL (market order)
                "symbol": symbol,
                "volume": volume,
                "type": 0 if signal.type == SignalType.BUY else 1,  # ORDER_TYPE_BUY or ORDER_TYPE_SELL
                "comment": f"Ultra100_{signal.reason[:20]}",
                "deviation": 20,  # Allow 20 points deviation
                "magic": 100100  # Magic number for identification
            }
            
            # Only add sl/tp if they are valid (some brokers don't accept 0)
            if sl > 0:
                order["sl"] = sl
            if tp > 0:
                order["tp"] = tp
            
            logger.info(f"📊 Placing {signal.type.value} order for {symbol}")
            logger.info(f"   Volume: {volume}, Entry: {current_price}")
            logger.info(f"   Validated SL: {sl} (original: {signal.sl})")
            logger.info(f"   Validated TP: {tp} (original: {signal.tp})")
            
            ticket = self.api_client.place_order(order)
            
            if ticket:
                logger.info(f"✅ Order placed successfully! Ticket: {ticket}")
                return ticket
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise
    
    def _validate_and_adjust_stops(self, symbol: str, symbol_info: Dict[str, Any], 
                                   signal_type: SignalType, current_price: float, 
                                   sl: float, tp: float) -> tuple:
        """Validate and adjust stop loss and take profit according to MT5 requirements"""
        try:
            # Extract symbol properties
            digits = symbol_info.get('digits', 5)
            point = symbol_info.get('point', 0.00001)
            stop_level_points = symbol_info.get('stoplevel', 0)
            stop_level = stop_level_points * point
            
            # Get current bid/ask for more accurate validation
            price_info = self.api_client.get_current_price(symbol)
            bid = price_info.get('bid', current_price)
            ask = price_info.get('ask', current_price)
            
            # Log detailed symbol info for debugging
            logger.info(f"Symbol {symbol} validation info:")
            logger.info(f"  - Digits: {digits}, Point: {point}")
            logger.info(f"  - Stop level points: {stop_level_points}, Stop level: {stop_level}")
            logger.info(f"  - Current bid: {bid}, ask: {ask}")
            logger.info(f"  - Original SL: {sl}, TP: {tp}")
            
            # Round stops to symbol precision
            sl = round(sl, digits)
            tp = round(tp, digits)
            
            # Add minimum buffer based on spread and broker requirements
            spread = ask - bid
            min_buffer = max(stop_level, spread * 2, 20 * point)  # At least 20 points or 2x spread
            
            # Validate and adjust stops with proper distance
            if signal_type == SignalType.BUY:
                # For BUY orders: use ASK for entry, so stops are relative to ASK
                reference_price = ask
                
                # SL must be below bid - min_buffer
                max_sl = bid - min_buffer
                if sl >= max_sl:
                    sl = round(max_sl - 10 * point, digits)
                    logger.warning(f"Adjusted BUY SL from {sl} to {sl} (below bid {bid} - buffer {min_buffer})")
                
                # TP must be above ask + min_buffer
                min_tp = ask + min_buffer
                if tp <= min_tp:
                    tp = round(min_tp + 10 * point, digits)
                    logger.warning(f"Adjusted BUY TP from {tp} to {tp} (above ask {ask} + buffer {min_buffer})")
            else:
                # For SELL orders: use BID for entry, so stops are relative to BID
                reference_price = bid
                
                # SL must be above ask + min_buffer
                min_sl = ask + min_buffer
                if sl <= min_sl:
                    sl = round(min_sl + 10 * point, digits)
                    logger.warning(f"Adjusted SELL SL from {sl} to {sl} (above ask {ask} + buffer {min_buffer})")
                
                # TP must be below bid - min_buffer
                max_tp = bid - min_buffer
                if tp >= max_tp:
                    tp = round(max_tp - 10 * point, digits)
                    logger.warning(f"Adjusted SELL TP from {tp} to {tp} (below bid {bid} - buffer {min_buffer})")
            
            # Additional validation: ensure stops are not too far (some brokers limit this)
            max_distance = current_price * 0.1  # 10% max distance
            
            if signal_type == SignalType.BUY:
                if current_price - sl > max_distance:
                    sl = round(current_price - max_distance, digits)
                    logger.warning(f"Adjusted SL to {sl} due to max distance limit")
                if tp - current_price > max_distance:
                    tp = round(current_price + max_distance, digits)
                    logger.warning(f"Adjusted TP to {tp} due to max distance limit")
            else:
                if sl - current_price > max_distance:
                    sl = round(current_price + max_distance, digits)
                    logger.warning(f"Adjusted SL to {sl} due to max distance limit")
                if current_price - tp > max_distance:
                    tp = round(current_price - max_distance, digits)
                    logger.warning(f"Adjusted TP to {tp} due to max distance limit")
            
            # Final validation - ensure stops are valid numbers and not zero
            if sl <= 0 or tp <= 0:
                logger.error(f"Invalid stops after validation: SL={sl}, TP={tp}")
                # Use conservative defaults based on ATR
                atr_estimate = current_price * 0.001  # 0.1% as fallback
                if signal_type == SignalType.BUY:
                    sl = round(bid - atr_estimate * 2, digits)
                    tp = round(ask + atr_estimate * 3, digits)
                else:
                    sl = round(ask + atr_estimate * 2, digits)
                    tp = round(bid - atr_estimate * 3, digits)
                logger.warning(f"Using fallback stops: SL={sl}, TP={tp}")
            
            # Log final validated stops
            logger.info(f"Final validated stops for {symbol}:")
            logger.info(f"  - SL: {sl} (distance from entry: {abs(reference_price - sl)/point:.0f} points)")
            logger.info(f"  - TP: {tp} (distance from entry: {abs(tp - reference_price)/point:.0f} points)")
            logger.info(f"  - Min required distance: {min_buffer/point:.0f} points")
            
            return sl, tp
            
        except Exception as e:
            logger.error(f"Error validating stops: {e}")
            # Return conservative stops if validation fails completely
            if signal_type == SignalType.BUY:
                return round(current_price * 0.995, 5), round(current_price * 1.01, 5)
            else:
                return round(current_price * 1.005, 5), round(current_price * 0.99, 5)
    
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
                    logger.debug(f"Position {ticket} not in active trades, skipping")
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
            raise
    
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
            raise
    
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
            raise
    
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
            raise
    
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
            raise
    
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