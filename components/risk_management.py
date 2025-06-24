#!/usr/bin/env python3
"""
Risk Management Module
Handles position sizing, risk calculations, and account safety checks
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import numpy as np

from .trading_components import SymbolUtils, CONFIG, get_symbol_config, Signal

logger = logging.getLogger('RiskManagement')

class RiskManagement:
    def __init__(self):
        self.symbol_utils = SymbolUtils()
        self.account_currency = CONFIG["ACCOUNT_CURRENCY"]
        self.api_client = None

    def adjust_trade_parameters_for_risk(self, signal: Signal, balance: float, 
                                         symbol_info: Dict[str, Any]) -> Tuple[Signal, float]:
        """
        Calculates volume and adjusts SL/TP to ensure the trade fits within the account's risk budget.
        Returns an adjusted Signal object and the calculated volume.
        """
        min_volume = symbol_info.get('volume_min', CONFIG["MIN_VOLUME"])
        
        # Get risk percentage using the new dynamic logic
        risk_percentage = self._get_risk_percentage(signal.symbol, self.symbol_utils.get_instrument_type(signal.symbol))
        logger.info(f"Using dynamic risk per trade: {risk_percentage:.2%}")

        max_monetary_risk = balance * risk_percentage

        pip_value_per_lot = self._calculate_pip_value(signal.symbol, signal.entry, symbol_info.get('trade_contract_size', 100000), symbol_info)
        pip_value_for_min_volume = pip_value_per_lot * min_volume
        
        if pip_value_for_min_volume <= 0:
            logger.error(f"Pip value for min volume is zero or negative for {signal.symbol}. Cannot trade.")
            return signal, 0.0

        max_sl_pips = max_monetary_risk / pip_value_for_min_volume if pip_value_for_min_volume > 0 else 0

        original_sl_distance = abs(signal.entry - signal.sl)
        original_sl_pips = self._convert_distance_to_pips(original_sl_distance, signal.symbol, symbol_info)
        
        final_sl_distance = original_sl_distance
        final_volume = min_volume

        if original_sl_pips > max_sl_pips and max_sl_pips > 0:
            logger.warning(f"Strategy SL ({original_sl_pips:.1f} pips) for {signal.symbol} exceeds risk budget. Overriding to fit.")
            
            final_sl_pips = max_sl_pips
            final_sl_distance = self._convert_pips_to_distance(final_sl_pips, signal.symbol, symbol_info)
            final_volume = min_volume
            
            original_tp_distance = abs(signal.tp - signal.entry)
            original_rr_ratio = original_tp_distance / original_sl_distance if original_sl_distance > 0 else 0
            
            if original_rr_ratio > 0:
                new_tp_distance = final_sl_distance * original_rr_ratio
                if signal.type.value == "BUY":
                    signal.tp = signal.entry + new_tp_distance
                else: # SELL
                    signal.tp = signal.entry - new_tp_distance

            if signal.type.value == "BUY":
                signal.sl = signal.entry - final_sl_distance
            else: # SELL
                signal.sl = signal.entry + final_sl_distance
            
            logger.info(f"Adjusted SL to {final_sl_pips:.1f} pips and updated TP to maintain RR. Volume set to {final_volume} lots.")

        else:
            # The proposed SL is within budget, calculate volume normally
            volume = self.calculate_position_size(signal.symbol, original_sl_distance, signal.entry, balance, symbol_info)
            final_volume = volume
        
        if final_volume < min_volume:
            logger.error(f"Final calculated volume {final_volume} is below minimum {min_volume}. Rejecting trade.")
            return signal, 0.0
            
        # --- FIXED LOT OVERRIDE REMOVED ---
        # The 0.01 lot override has been removed from here.

        return signal, final_volume

    def calculate_position_size(self, symbol: str, sl_distance: float, current_price: float, 
                               balance: float, symbol_info: Dict[str, Any], 
                               account_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculates position size based on risk."""
        try:
            risk_percentage = self._get_risk_percentage(symbol, self.symbol_utils.get_instrument_type(symbol))
            risk_amount = balance * risk_percentage
            
            contract_size = symbol_info.get('trade_contract_size', 100000)
            min_volume = symbol_info.get('volume_min', CONFIG["MIN_VOLUME"])
            max_volume = symbol_info.get('volume_max', 10.0) # Increased max volume limit
            volume_step = symbol_info.get('volume_step', 0.01)

            pip_value_per_lot = self._calculate_pip_value(symbol, current_price, contract_size, symbol_info)
            sl_pips = self._convert_distance_to_pips(sl_distance, symbol, symbol_info)

            if pip_value_per_lot <= 0 or sl_pips <= 0:
                logger.warning(f"Cannot calculate position size for {symbol} due to invalid pip value or SL pips.")
                return 0.0

            position_size = risk_amount / (sl_pips * pip_value_per_lot)
            
            position_size = round(position_size / volume_step) * volume_step
            position_size = np.clip(position_size, min_volume, max_volume)
            
            return position_size

        except Exception as e:
            logger.error(f"Error in calculate_position_size for {symbol}: {e}")
            return 0.0

    def _get_risk_percentage(self, symbol: str, instrument_type: str) -> float:
        """
        Get risk percentage for a single trade.
        This now divides the total risk budget by the number of max concurrent trades.
        """
        max_total_risk = CONFIG.get("MAX_TOTAL_RISK", 0.05)
        max_concurrent = CONFIG.get("MAX_CONCURRENT", 1)

        if max_concurrent > 0:
            # Divide the total risk budget across the max number of allowed trades
            risk_per_trade = max_total_risk / max_concurrent
            return risk_per_trade
        else:
            # Fallback to a single trade risk if config is invalid
            return CONFIG.get("RISK_PER_TRADE", 0.01)

    # --- OTHER FUNCTIONS REMAIN UNCHANGED ---
    # (The rest of the file is the same as your original)
    def _convert_distance_to_pips(self, distance: float, symbol: str, symbol_info: Dict[str, Any]) -> float:
        digits = symbol_info.get('digits', 5)
        point = symbol_info.get('point', 0.00001)
        if 'JPY' in symbol.upper():
            return distance / (point * 100)
        return distance / (point * 10)

    def _convert_pips_to_distance(self, pips: float, symbol: str, symbol_info: Dict[str, Any]) -> float:
        digits = symbol_info.get('digits', 5)
        point = symbol_info.get('point', 0.00001)
        if 'JPY' in symbol.upper():
            return pips * (point * 100)
        return pips * (point * 10)

    def check_account_safety(self, balance: float, equity: float, margin: float, 
                           daily_pnl: float = None, margin_level: float = None) -> Tuple[bool, str]:
        try:
            if margin > 0:
                current_margin_level = (equity / margin) * 100
                if current_margin_level < 150:
                    return False, f"Margin level too low: {current_margin_level:.0f}%"
            elif margin_level is not None:
                if margin_level < 150:
                    return False, f"Margin level too low: {margin_level:.0f}%"
            
            if equity < balance * 0.95:
                drawdown = (balance - equity) / balance
                if drawdown > 0.10:
                    return False, f"Drawdown too high: {drawdown:.1%}"
            
            return True, "Account safe"
            
        except Exception as e:
            logger.error(f"Error checking account safety: {e}")
            return False, "Safety check error"
    
    def can_trade_symbol(self, symbol: str, active_trades: Dict[str, Any], 
                        last_trade_time: Dict[str, float], 
                        pending_orders: Optional[Dict[str, Any]] = None,
                        signal_type: Optional[str] = None) -> Tuple[bool, str]:
        try:
            if len(active_trades) >= CONFIG["MAX_CONCURRENT"]:
                return False, "Max concurrent positions reached"
            
            if pending_orders and symbol in pending_orders:
                return False, "Pending order already exists"
            
            for ticket, trade in active_trades.items():
                if trade.symbol == symbol:
                    return False, "Position already open"
            
            if symbol in last_trade_time:
                time_since_last = time.time() - last_trade_time[symbol]
                if time_since_last < CONFIG["POSITION_INTERVAL"]:
                    remaining = CONFIG["POSITION_INTERVAL"] - time_since_last
                    return False, f"Too soon, wait {remaining:.0f}s"
            
            return True, "Can trade"
            
        except Exception as e:
            logger.error(f"Error checking trade permission for {symbol}: {e}")
            return False, "Permission check error"
    
    def check_margin_safety_before_trade(self, symbol: str, position_size: float, 
                                       current_price: float, sl_distance: float,
                                       account_info: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            equity = account_info.get('equity', 0)
            margin = account_info.get('margin', 0)
            margin_free = account_info.get('margin_free', 0)
            leverage = account_info.get('leverage', 100)
            
            if not self.api_client:
                return False, "API client not available in RiskManagement"
            
            symbol_info = self.api_client.get_symbol_info(symbol)
            if not symbol_info: return False, "Could not get symbol info for margin check"
            contract_size = symbol_info.get('trade_contract_size', 100000)

            position_value = position_size * contract_size
            
            margin_required = position_value / leverage
            
            if margin_required > margin_free:
                return False, f"Insufficient free margin: {margin_free:.2f} < {margin_required:.2f}"
            
            new_margin = margin + margin_required
            new_margin_level = (equity / new_margin) * 100 if new_margin > 0 else float('inf')
            
            if new_margin_level < 150:
                return False, f"Opening position would drop margin level to {new_margin_level:.0f}%"
            
            pip_value_per_lot = self._calculate_pip_value(symbol, current_price, contract_size, symbol_info)
            pips_to_sl = self._convert_distance_to_pips(sl_distance, symbol, symbol_info)
            potential_loss = pips_to_sl * pip_value_per_lot * position_size
            worst_case_equity = equity - potential_loss
            
            if new_margin > 0 and worst_case_equity <= new_margin:
                return False, f"S/L hit would cause margin call. Worst equity {worst_case_equity:.2f} vs new margin {new_margin:.2f}"

            return True, "Margin safety OK"
            
        except Exception as e:
            logger.error(f"Error checking margin safety: {e}")
            return False, "Margin safety check failed"
    
    def _calculate_pip_value(self, symbol: str, current_price: float, 
                           contract_size: float, symbol_info: Dict[str, Any]) -> float:
        try:
            quote_currency = self.symbol_utils.get_quote_currency(symbol)
            point = symbol_info.get('point', 0.00001)

            if 'JPY' in symbol:
                pip_size = point * 100
            else:
                pip_size = point * 10

            if quote_currency == self.account_currency:
                return pip_size * contract_size
            
            elif self.symbol_utils.get_base_currency(symbol) == self.account_currency:
                return pip_size
            
            else:
                if not self.api_client:
                     logger.warning("API client not available for cross-rate lookup. Using estimations.")
                     cross_rate = 1.0
                else:
                    cross_pair = f"{quote_currency}{self.account_currency}"
                    cross_info = self.api_client.get_current_price(cross_pair)
                    if cross_info and 'bid' in cross_info:
                        cross_rate = cross_info['bid']
                    else:
                        logger.warning(f"Could not get live cross rate for {cross_pair}. Using 1.0 as fallback.")
                        cross_rate = 1.0

                return pip_size * contract_size * cross_rate

        except Exception as e:
            logger.error(f"Error calculating pip value for {symbol}: {e}")
            return 0.0

    def validate_trade_parameters(self, symbol: str, signal_type: str, 
                                entry: float, sl: float, tp: float) -> Tuple[bool, str]:
        try:
            if entry <= 0 or sl <= 0 or tp <= 0:
                return False, "Invalid price levels"
            
            if signal_type == "BUY":
                if sl >= entry: return False, "Invalid SL for BUY"
                if tp <= entry: return False, "Invalid TP for BUY"
            else: # SELL
                if sl <= entry: return False, "Invalid SL for SELL"
                if tp >= entry: return False, "Invalid TP for SELL"
            
            sl_distance = abs(entry - sl)
            sl_distance_percent = sl_distance / entry
            safety_margin = 1e-9
            
            if sl_distance_percent < (CONFIG["MIN_SL_DISTANCE_PERCENT"] - safety_margin):
                return False, f"SL too close: {sl_distance_percent*100:.3f}% < {CONFIG['MIN_SL_DISTANCE_PERCENT']*100:.1f}%"
            
            if sl_distance_percent > (CONFIG["MAX_SL_DISTANCE_PERCENT"] + safety_margin):
                return False, f"SL too far: {sl_distance_percent*100:.3f}% > {CONFIG['MAX_SL_DISTANCE_PERCENT']*100:.1f}%"
            
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            min_rr = get_symbol_config(symbol).get('min_rr_ratio', CONFIG["MIN_RR_RATIO"])
            
            if rr_ratio < min_rr:
                return False, f"RR ratio {rr_ratio:.2f} is below minimum required {min_rr:.2f}"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return False, "Validation error"

    def get_available_symbols_for_trading(self, active_trades: Dict[str, Any], 
                                        all_symbols: List[str]) -> List[str]:
        try:
            occupied_symbols = {trade.symbol for trade in active_trades.values()}
            
            available_symbols = [symbol for symbol in all_symbols if symbol not in occupied_symbols]
            
            return available_symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return all_symbols

    def should_seek_new_positions(self, account_info: Dict[str, Any], active_trades: Dict[str, Any],
                                 max_positions: int = None) -> Tuple[bool, str]:
        try:
            if max_positions is None:
                max_positions = CONFIG["MAX_CONCURRENT"]
            
            current_positions = len(active_trades)
            
            if current_positions >= max_positions:
                return False, f"At maximum positions ({current_positions}/{max_positions})"
            
            margin_level = account_info.get('margin_level', 0)
            
            if margin_level > 300:
                available_slots = max_positions - current_positions
                return True, f"Should seek {available_slots} more positions (margin level: {margin_level:.0f}%)"
            else:
                return False, f"Margin level too low for new positions: {margin_level:.0f}%"
            
        except Exception as e:
            logger.error(f"Error checking if should seek new positions: {e}")
            return False, "Error in position seeking check"