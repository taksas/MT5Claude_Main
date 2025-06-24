#!/usr/bin/env python3
"""
Risk Management Module
Handles position sizing, risk calculations, and account safety checks
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from .symbol_utils import SymbolUtils
from .trading_config import CONFIG, get_symbol_config

logger = logging.getLogger('RiskManagement')

class RiskManagement:
    def __init__(self):
        self.symbol_utils = SymbolUtils()
        self.account_currency = CONFIG["ACCOUNT_CURRENCY"]
        
    def calculate_position_size(self, symbol: str, sl_distance: float, current_price: float, 
                               balance: float, symbol_info: Dict[str, Any], 
                               account_info: Dict[str, Any] = None) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get account leverage for margin safety
            leverage = 100  # Default conservative leverage
            if account_info:
                leverage = account_info.get('leverage', 100)
            
            # Get instrument type and risk percentage
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            risk_percentage = self._get_risk_percentage(symbol, instrument_type)
            
            # Calculate risk amount in account currency
            risk_amount = balance * risk_percentage
            
            # Get symbol specifications
            contract_size = symbol_info.get('trade_contract_size', 100000)
            min_volume = symbol_info.get('volume_min', CONFIG["MIN_VOLUME"])
            # Limit maximum volume to prevent excessive position sizes
            max_volume = min(symbol_info.get('volume_max', 1.0), 1.0)  # Max 1.0 lot
            volume_step = symbol_info.get('volume_step', 0.01)
            
            # Calculate pip value
            pip_value = self._calculate_pip_value(
                symbol, current_price, contract_size, symbol_info
            )
            
            # Calculate position size
            if pip_value > 0 and sl_distance > 0:
                # Convert SL distance to pips
                digits = symbol_info.get('digits', 5)
                
                if 'JPY' in symbol.upper():
                    sl_pips = sl_distance * 100  # JPY pairs
                elif digits == 5 or digits == 3:
                    sl_pips = sl_distance * 10000  # 5-digit broker
                elif digits == 4 or digits == 2:
                    sl_pips = sl_distance * 100  # 4-digit broker
                else:
                    sl_pips = sl_distance * 10000  # Default
                
                # Position size = Risk Amount / (SL in pips × Pip Value)
                position_size = risk_amount / (sl_pips * pip_value)
                
                # Round to volume step
                position_size = round(position_size / volume_step) * volume_step
                
                # Apply limits
                position_size = max(min_volume, min(position_size, max_volume))
                
                # Additional safety for exotic/volatile instruments
                if instrument_type in ['exotic', 'crypto']:
                    position_size = min(position_size, 0.05)  # Max 0.05 lots
                elif instrument_type == 'metal':
                    position_size = min(position_size, 0.1)   # Max 0.1 lots
                else:
                    position_size = min(position_size, 0.2)   # Max 0.2 lots for majors
                
                # Get free margin from account info if available
                free_margin = balance  # Default to balance
                if account_info and 'margin_free' in account_info:
                    free_margin = account_info.get('margin_free', balance)
                
                # Ultra-conservative fallback for low balance/margin situations
                if free_margin < 1000:  # Less than 1000 units of currency free
                    position_size = min_volume  # Use minimum possible
                    logger.warning(f"Very low free margin ({free_margin:.2f}), using minimum volume: {min_volume}")
                
                # Critical margin safety check based on leverage
                # Calculate max position size to keep margin level safe
                sl_percent = sl_distance / current_price
                
                # With high leverage, we need to limit position size dramatically
                # Target: Keep margin level above 150% after potential S/L hit
                if leverage >= 500:
                    max_margin_usage = 0.10  # Use only 10% of available margin
                elif leverage >= 200:
                    max_margin_usage = 0.20  # Use only 20% of available margin
                elif leverage >= 100:
                    max_margin_usage = 0.30  # Use only 30% of available margin
                else:
                    max_margin_usage = 0.50  # Use only 50% of available margin
                
                # Calculate maximum safe position value
                # Available margin for new positions = Free Margin × max_margin_usage
                # Free margin already calculated above
                
                # Maximum position value we can open = free_margin × leverage × usage_percent
                max_position_value = free_margin * leverage * max_margin_usage
                actual_position_value = position_size * contract_size * current_price
                
                if actual_position_value > max_position_value:
                    # Scale down position size for safety
                    original_size = position_size
                    position_size = (max_position_value / actual_position_value) * position_size
                    position_size = round(position_size / volume_step) * volume_step
                    position_size = max(min_volume, position_size)
                    
                    logger.warning(f"Position size reduced for margin safety: {original_size} → {position_size} lots")
                
                # CRITICAL: Final absolute monetary risk validation
                # Calculate actual monetary risk in account currency
                sl_pips_final = sl_pips  # Already calculated above
                actual_pip_value = pip_value * position_size  # Pip value for this position size
                total_monetary_risk = sl_pips_final * actual_pip_value
                
                # First check: if even minimum volume is too risky, reject the trade
                min_risk = sl_pips_final * pip_value * min_volume
                max_acceptable_risk = balance * 0.05  # Never risk more than 5% of account per trade
                
                if min_risk > max_acceptable_risk:
                    logger.critical(f"TRADE REJECTED - MINIMUM RISK TOO HIGH for {symbol}:")
                    logger.critical(f"  Minimum position: {min_volume} lots")
                    logger.critical(f"  Minimum risk: {min_risk:.2f} {self.account_currency}")
                    logger.critical(f"  Maximum acceptable risk: {max_acceptable_risk:.2f} {self.account_currency}")
                    logger.critical(f"  Stop-loss distance too wide: {sl_pips_final:.1f} pips")
                    logger.critical(f"  Consider tighter stop-loss or skip this trade")
                    return 0.0  # Return 0 to prevent trade
                
                # Safety check: ensure risk doesn't exceed account balance
                if total_monetary_risk > max_acceptable_risk:
                    # Scale down position size to keep risk acceptable
                    original_position_size = position_size
                    safe_position_size = (max_acceptable_risk / total_monetary_risk) * position_size
                    position_size = round(safe_position_size / volume_step) * volume_step
                    position_size = max(min_volume, position_size)
                    
                    # Recalculate actual risk with new position size
                    actual_pip_value = pip_value * position_size
                    total_monetary_risk = sl_pips_final * actual_pip_value
                    
                    logger.critical(f"RISK PROTECTION ACTIVATED for {symbol}:")
                    logger.critical(f"  Original position: {original_position_size} lots")
                    logger.critical(f"  Original risk: {sl_pips_final * pip_value * original_position_size:.2f} {self.account_currency}")
                    logger.critical(f"  Reduced to: {position_size} lots")
                    logger.critical(f"  Final risk: {total_monetary_risk:.2f} {self.account_currency}")
                    logger.critical(f"  Risk vs balance: {(total_monetary_risk/balance)*100:.1f}%")
                
                # Log final calculation details
                margin_required = actual_position_value / leverage
                logger.info(f"Position sizing for {symbol}:")
                logger.info(f"  Leverage: {leverage}:1")
                logger.info(f"  Balance: {balance:.2f}")
                logger.info(f"  Free margin: {free_margin:.2f}")
                logger.info(f"  Risk %: {risk_percentage*100:.1f}%")
                logger.info(f"  Risk amount: {risk_amount:.2f}")
                logger.info(f"  S/L distance: {sl_distance:.5f} ({(sl_distance/current_price)*100:.2f}%)")
                logger.info(f"  S/L pips: {sl_pips_final:.1f}")
                logger.info(f"  Pip value: {pip_value:.2f} per lot")
                logger.info(f"  Position size: {position_size} lots")
                logger.info(f"  Position value: {actual_position_value:.2f}")
                logger.info(f"  Margin required: {margin_required:.2f}")
                logger.info(f"  Max margin usage: {max_margin_usage*100:.0f}%")
                logger.info(f"  TOTAL MONETARY RISK: {total_monetary_risk:.2f} {self.account_currency}")
                logger.info(f"  Risk vs balance: {(total_monetary_risk/balance)*100:.1f}%")
                
                return position_size
            else:
                return min_volume
                
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return CONFIG["MIN_VOLUME"]
    
    def check_account_safety(self, balance: float, equity: float, margin: float, 
                           daily_pnl: float, margin_level: float = None) -> Tuple[bool, str]:
        """Check if account is safe to trade"""
        try:
            # Balance check removed - trade with any balance
            
            # Check daily loss limit - only check if P&L is negative
            if daily_pnl < 0 and balance > 0:
                daily_loss_pct = abs(daily_pnl) / balance
                if daily_loss_pct > CONFIG["MAX_DAILY_LOSS"]:
                    return False, f"Daily loss limit exceeded: {daily_loss_pct:.1%} (Loss: {daily_pnl:.2f})"
            
            # Check margin level
            if margin > 0:
                current_margin_level = (equity / margin) * 100
                if current_margin_level < 150:  # Minimum 150% margin level for safety
                    return False, f"Margin level too low: {current_margin_level:.0f}%"
            elif margin_level is not None:
                # Use provided margin level
                if margin_level < 150:
                    return False, f"Margin level too low: {margin_level:.0f}%"
            
            # Check drawdown
            if equity < balance * 0.95:  # More than 5% drawdown
                drawdown = (balance - equity) / balance
                if drawdown > 0.10:  # 10% max drawdown
                    return False, f"Drawdown too high: {drawdown:.1%}"
            
            return True, "Account safe"
            
        except Exception as e:
            logger.error(f"Error checking account safety: {e}")
            return False, "Safety check error"
    
    def can_trade_symbol(self, symbol: str, active_trades: Dict[str, Any], 
                        last_trade_time: Dict[str, float], 
                        pending_orders: Dict[str, Any] = None,
                        signal_type: str = None) -> Tuple[bool, str]:
        """Check if we can trade this symbol"""
        try:
            # Check maximum concurrent positions
            if len(active_trades) >= CONFIG["MAX_CONCURRENT"]:
                return False, "Max concurrent positions reached"
            
            # Check if symbol has pending order
            if pending_orders and symbol in pending_orders:
                return False, "Pending order already exists"
            
            # Check if symbol already has position
            # active_trades is Dict[ticket, Trade], so check symbol in trade objects
            for ticket, trade in active_trades.items():
                if trade.symbol == symbol:
                    # If we have a position, check if it's the opposite direction
                    if signal_type and trade.type != signal_type:
                        return False, f"Opposite position already open ({trade.type})"
                    else:
                        return False, "Position already open"
            
            # Check position interval
            if symbol in last_trade_time:
                time_since_last = time.time() - last_trade_time[symbol]
                if time_since_last < CONFIG["POSITION_INTERVAL"]:
                    remaining = CONFIG["POSITION_INTERVAL"] - time_since_last
                    return False, f"Too soon, wait {remaining:.0f}s"
            
            # Check instrument-specific limits
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            # Count positions by type
            type_counts = {}
            for ticket, trade in active_trades.items():
                sym = trade.symbol
                t = self.symbol_utils.get_instrument_type(sym)
                type_counts[t] = type_counts.get(t, 0) + 1
            
            # Limit positions per instrument type
            max_per_type = {
                'exotic': 2,
                'crypto': 1,
                'metal': 2,
                'index': 2,
                'major': 3
            }
            
            current_count = type_counts.get(instrument_type, 0)
            max_allowed = max_per_type.get(instrument_type, 3)
            
            if current_count >= max_allowed:
                return False, f"Max {instrument_type} positions reached"
            
            return True, "Can trade"
            
        except Exception as e:
            logger.error(f"Error checking trade permission for {symbol}: {e}")
            return False, "Permission check error"
    
    def _get_risk_percentage(self, symbol: str, instrument_type: str) -> float:
        """Get risk percentage for symbol"""
        # Check symbol-specific configuration
        symbol_config = get_symbol_config(symbol)
        if 'risk_factor' in symbol_config:
            return symbol_config['risk_factor']
        
        # Use instrument type defaults
        risk_map = {
            'exotic': CONFIG["RISK_PER_EXOTIC"],
            'crypto': CONFIG["RISK_PER_CRYPTO"],
            'metal': CONFIG["RISK_PER_METAL"],
            'index': CONFIG["RISK_PER_INDEX"],
            'major': CONFIG["RISK_PER_TRADE"]
        }
        
        return risk_map.get(instrument_type, CONFIG["RISK_PER_TRADE"])
    
    def check_margin_safety_before_trade(self, symbol: str, position_size: float, 
                                       current_price: float, sl_distance: float,
                                       account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if opening this position would be safe for margin requirements"""
        try:
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            margin = account_info.get('margin', 0)
            margin_free = account_info.get('margin_free', equity)
            leverage = account_info.get('leverage', 100)
            
            # Calculate margin required for this position
            contract_size = 100000  # Standard lot
            position_value = position_size * contract_size * current_price
            margin_required = position_value / leverage
            
            # Check if we have enough free margin
            if margin_required > margin_free:
                return False, f"Insufficient free margin: {margin_free:.2f} < {margin_required:.2f}"
            
            # Calculate new margin level after opening position
            new_margin = margin + margin_required
            new_margin_level = (equity / new_margin) * 100 if new_margin > 0 else float('inf')
            
            # Ensure margin level stays above 150%
            if new_margin_level < 150:
                return False, f"Opening position would drop margin level to {new_margin_level:.0f}%"
            
            # Calculate worst-case margin level if S/L is hit
            sl_percent = sl_distance / current_price
            potential_loss = position_value * sl_percent
            worst_case_equity = equity - potential_loss
            worst_case_margin_level = (worst_case_equity / new_margin) * 100 if new_margin > 0 else 0
            
            # Ensure we stay above margin call level (50%) even if S/L is hit
            if worst_case_margin_level < 75:  # Safety buffer above 50% margin call
                return False, f"S/L hit would drop margin level to {worst_case_margin_level:.0f}%"
            
            return True, f"Safe to trade: Margin level {new_margin_level:.0f}% → {worst_case_margin_level:.0f}% (worst case)"
            
        except Exception as e:
            logger.error(f"Error checking margin safety: {e}")
            return False, "Margin safety check failed"
    
    def _calculate_pip_value(self, symbol: str, current_price: float, 
                           contract_size: float, symbol_info: Dict[str, Any]) -> float:
        """Calculate pip value in account currency"""
        try:
            quote_currency = self.symbol_utils.get_quote_currency(symbol)
            base_currency = self.symbol_utils.get_base_currency(symbol)
            digits = symbol_info.get('digits', 5)
            
            # Determine pip size
            if 'JPY' in symbol.upper():
                pip_size = 0.01
            elif digits == 5 or digits == 3:
                pip_size = 0.0001
            elif digits == 4 or digits == 2:
                pip_size = 0.01 if digits == 2 else 0.0001
            else:
                pip_size = 0.0001
            
            # Calculate pip value
            if quote_currency == self.account_currency:
                # Direct calculation
                pip_value = pip_size * contract_size
            elif base_currency == self.account_currency:
                # Inverse calculation
                pip_value = (pip_size * contract_size) / current_price
            else:
                # Cross calculation - calculate proper pip values for JPY account
                if self.account_currency == 'JPY':
                    # For cross-currency pairs with JPY account, we need to convert through USD or estimate
                    if 'USD' in symbol:
                        # For pairs like GBPUSD, EURUSD with JPY account
                        # Pip value = pip_size * contract_size * USD/JPY rate
                        # Using approximate USD/JPY rate of 150 (conservative estimate)
                        usd_jpy_rate = 150.0
                        pip_value = pip_size * contract_size * usd_jpy_rate
                    elif 'GBP' in symbol and symbol.startswith('GBP'):
                        # For GBP pairs with JPY account (e.g., GBPCHF, GBPCAD)
                        # Using approximate GBP/JPY rate of 180
                        gbp_jpy_rate = 180.0
                        pip_value = pip_size * contract_size * gbp_jpy_rate
                    elif 'EUR' in symbol and symbol.startswith('EUR'):
                        # For EUR pairs with JPY account (e.g., EURCHF, EURCAD)
                        # Using approximate EUR/JPY rate of 160
                        eur_jpy_rate = 160.0
                        pip_value = pip_size * contract_size * eur_jpy_rate
                    else:
                        # Conservative fallback for other pairs
                        # Use 100 JPY per pip per full lot as safe estimate
                        pip_value = pip_size * contract_size * 100
                else:
                    # For non-JPY account currencies, use standard USD calculations
                    pip_value = pip_size * contract_size  # Standard calculation
            
            # Scale pip value based on actual lot size (pip value is per 0.01 lot)
            # No need to adjust here as we'll calculate based on position size
            
            return pip_value
            
        except Exception as e:
            logger.error(f"Error calculating pip value for {symbol}: {e}")
            return 10  # Default pip value
    
    def calculate_risk_reward_ratio(self, entry: float, sl: float, tp: float, 
                                   signal_type: str) -> float:
        """Calculate risk-reward ratio"""
        try:
            if signal_type == "BUY":
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp
            
            if risk > 0:
                return reward / risk
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating RR ratio: {e}")
            return 0
    
    def validate_absolute_risk(self, symbol: str, position_size: float, sl_distance: float,
                             current_price: float, balance: float, symbol_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that the absolute monetary risk is acceptable for the account"""
        try:
            # Calculate pip value
            contract_size = symbol_info.get('trade_contract_size', 100000)
            pip_value = self._calculate_pip_value(symbol, current_price, contract_size, symbol_info)
            
            # Calculate pip distance
            digits = symbol_info.get('digits', 5)
            if 'JPY' in symbol.upper():
                sl_pips = sl_distance * 100
            elif digits == 5 or digits == 3:
                sl_pips = sl_distance * 10000
            elif digits == 4 or digits == 2:
                sl_pips = sl_distance * 100
            else:
                sl_pips = sl_distance * 10000
            
            # Calculate total monetary risk
            actual_pip_value = pip_value * position_size
            total_monetary_risk = sl_pips * actual_pip_value
            
            # Check absolute risk limits
            max_risk_per_trade = balance * 0.05  # Never risk more than 5% per trade
            max_catastrophic_risk = balance * 0.20  # Never risk more than 20% (catastrophic protection)
            
            if total_monetary_risk > max_catastrophic_risk:
                return False, f"CATASTROPHIC RISK: {total_monetary_risk:.2f} > {max_catastrophic_risk:.2f} (20% of balance)"
            
            if total_monetary_risk > max_risk_per_trade:
                return False, f"Excessive risk: {total_monetary_risk:.2f} > {max_risk_per_trade:.2f} (5% limit)"
            
            # Check if risk is reasonable compared to balance
            risk_percentage = (total_monetary_risk / balance) * 100
            if risk_percentage > 3.0:  # More than 3% is concerning
                logger.warning(f"High risk warning: {risk_percentage:.1f}% of account balance at risk")
            
            return True, f"Risk acceptable: {total_monetary_risk:.2f} ({risk_percentage:.1f}% of balance)"
            
        except Exception as e:
            logger.error(f"Error validating absolute risk: {e}")
            return False, "Risk validation error"
    
    def get_available_symbols_for_trading(self, active_trades: Dict[str, Any], 
                                        all_symbols: List[str]) -> List[str]:
        """Get list of symbols available for new positions (diversification-focused)"""
        try:
            # Get symbols that currently have positions
            occupied_symbols = set()
            for ticket, trade in active_trades.items():
                occupied_symbols.add(trade.symbol)
            
            # Return symbols that don't have positions
            available_symbols = [symbol for symbol in all_symbols if symbol not in occupied_symbols]
            
            logger.info(f"Position diversification: {len(occupied_symbols)} symbols occupied, {len(available_symbols)} available")
            return available_symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return all_symbols
    
    def should_seek_new_positions(self, account_info: Dict[str, Any], active_trades: Dict[str, Any],
                                 max_positions: int = None) -> Tuple[bool, str]:
        """Check if we should proactively seek new positions based on margin levels"""
        try:
            if max_positions is None:
                max_positions = CONFIG["MAX_CONCURRENT"]
            
            current_positions = len(active_trades)
            
            # Don't seek more positions if at maximum
            if current_positions >= max_positions:
                return False, f"At maximum positions ({current_positions}/{max_positions})"
            
            # Check margin availability for new positions
            margin_free = account_info.get('margin_free', 0)
            margin_level = account_info.get('margin_level', 0)
            balance = account_info.get('balance', 0)
            
            # Don't seek positions if margin is low
            if margin_level > 0 and margin_level < 300:  # Below 300% margin level
                return False, f"Margin level too low for new positions: {margin_level:.0f}%"
            
            # Don't seek positions if free margin is very low
            if margin_free < balance * 0.2:  # Less than 20% of balance as free margin
                return False, f"Insufficient free margin: {margin_free:.2f} ({(margin_free/balance)*100:.1f}% of balance)"
            
            # Proactively seek positions if we have capacity and good margin
            available_slots = max_positions - current_positions
            return True, f"Should seek {available_slots} more positions (margin level: {margin_level:.0f}%)"
            
        except Exception as e:
            logger.error(f"Error checking if should seek new positions: {e}")
            return False, "Error in position seeking check"

    def validate_trade_parameters(self, symbol: str, signal_type: str, 
                                entry: float, sl: float, tp: float) -> Tuple[bool, str]:
        """Validate trade parameters"""
        try:
            # Check basic validity
            if entry <= 0 or sl <= 0 or tp <= 0:
                return False, "Invalid price levels"
            
            # Check stop loss placement
            if signal_type == "BUY":
                if sl >= entry:
                    return False, "Invalid SL for BUY"
                if tp <= entry:
                    return False, "Invalid TP for BUY"
                sl_distance = entry - sl
            else:
                if sl <= entry:
                    return False, "Invalid SL for SELL"
                if tp >= entry:
                    return False, "Invalid TP for SELL"
                sl_distance = sl - entry
            
            # Check minimum and maximum stop loss distance
            sl_distance_percent = sl_distance / entry
            
            if sl_distance_percent < CONFIG["MIN_SL_DISTANCE_PERCENT"]:
                return False, f"SL too close: {sl_distance_percent*100:.3f}% < {CONFIG['MIN_SL_DISTANCE_PERCENT']*100:.1f}%"
            
            if sl_distance_percent > CONFIG["MAX_SL_DISTANCE_PERCENT"]:
                return False, f"SL too far: {sl_distance_percent*100:.3f}% > {CONFIG['MAX_SL_DISTANCE_PERCENT']*100:.1f}%"
            
            # Check risk-reward ratio
            rr_ratio = self.calculate_risk_reward_ratio(entry, sl, tp, signal_type)
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            
            min_rr = CONFIG["MIN_RR_EXOTIC"] if instrument_type == 'exotic' else CONFIG["MIN_RR_RATIO"]
            
            if rr_ratio < min_rr:
                return False, f"RR ratio too low: {rr_ratio:.2f}"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return False, "Validation error"