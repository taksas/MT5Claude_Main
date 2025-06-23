#!/usr/bin/env python3
"""
Risk Management Module
Handles position sizing, risk calculations, and account safety checks
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from .symbol_utils import SymbolUtils
from .trading_config import CONFIG, get_symbol_config

logger = logging.getLogger('RiskManagement')

class RiskManagement:
    def __init__(self):
        self.symbol_utils = SymbolUtils()
        self.account_currency = CONFIG["ACCOUNT_CURRENCY"]
        
    def calculate_position_size(self, symbol: str, sl_distance: float, current_price: float, 
                               balance: float, symbol_info: Dict[str, Any]) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get instrument type and risk percentage
            instrument_type = self.symbol_utils.get_instrument_type(symbol)
            risk_percentage = self._get_risk_percentage(symbol, instrument_type)
            
            # Calculate risk amount in account currency
            risk_amount = balance * risk_percentage
            
            # Get symbol specifications
            contract_size = symbol_info.get('trade_contract_size', 100000)
            min_volume = symbol_info.get('volume_min', CONFIG["MIN_VOLUME"])
            max_volume = symbol_info.get('volume_max', 100)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            # Calculate pip value
            pip_value = self._calculate_pip_value(
                symbol, current_price, contract_size, symbol_info
            )
            
            # Calculate position size
            if pip_value > 0 and sl_distance > 0:
                # Convert SL distance to pips
                digits = symbol_info.get('digits', 5)
                
                if self.symbol_utils.is_jpy_pair(symbol) and not self.symbol_utils.is_metal_pair(symbol):
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
                    position_size = min(position_size, min_volume * 5)
                elif instrument_type == 'metal':
                    position_size = min(position_size, min_volume * 10)
                
                return position_size
            else:
                return min_volume
                
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return CONFIG["MIN_VOLUME"]
    
    def check_account_safety(self, balance: float, equity: float, margin: float, 
                           daily_pnl: float) -> Tuple[bool, str]:
        """Check if account is safe to trade"""
        try:
            # Balance check removed - trade with any balance
            
            # Check daily loss limit
            daily_loss_pct = abs(daily_pnl / balance) if balance > 0 else 0
            if daily_loss_pct > CONFIG["MAX_DAILY_LOSS"]:
                return False, f"Daily loss limit exceeded: {daily_loss_pct:.1%}"
            
            # Check margin level
            if margin > 0:
                margin_level = (equity / margin) * 100
                if margin_level < 200:  # Minimum 200% margin level
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
                        last_trade_time: Dict[str, float]) -> Tuple[bool, str]:
        """Check if we can trade this symbol"""
        try:
            # Check maximum concurrent positions
            if len(active_trades) >= CONFIG["MAX_CONCURRENT"]:
                return False, "Max concurrent positions reached"
            
            # Check if symbol already has position
            # active_trades is Dict[ticket, Trade], so check symbol in trade objects
            for ticket, trade in active_trades.items():
                if trade.symbol == symbol:
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
    
    def _calculate_pip_value(self, symbol: str, current_price: float, 
                           contract_size: float, symbol_info: Dict[str, Any]) -> float:
        """Calculate pip value in account currency"""
        try:
            quote_currency = self.symbol_utils.get_quote_currency(symbol)
            base_currency = self.symbol_utils.get_base_currency(symbol)
            digits = symbol_info.get('digits', 5)
            
            # Determine pip size
            if self.symbol_utils.is_jpy_pair(symbol) and not self.symbol_utils.is_metal_pair(symbol):
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
                # Cross calculation - need conversion rate
                # For simplicity, assume standard pip values
                if self.account_currency == 'JPY':
                    if 'JPY' in symbol:
                        pip_value = 1000  # ¥1000 per pip for XXX/JPY pairs
                    else:
                        pip_value = 100  # ¥100 per pip for other pairs
                else:
                    pip_value = 10  # $10 per pip standard
            
            # Adjust for mini/micro lots
            min_volume = symbol_info.get('volume_min', 0.01)
            if min_volume < 1:
                pip_value *= min_volume
            
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
            else:
                if sl <= entry:
                    return False, "Invalid SL for SELL"
                if tp >= entry:
                    return False, "Invalid TP for SELL"
            
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