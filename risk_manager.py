import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from trading_strategies import TradingSignal, SignalType

@dataclass
class RiskLimits:
    max_daily_loss: float = -500.0
    max_daily_trades: int = 10
    max_concurrent_trades: int = 3
    max_risk_per_trade: float = 0.02
    max_total_risk: float = 0.06
    min_time_between_trades: int = 5
    forbidden_hours_utc: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.forbidden_hours_utc is None:
            self.forbidden_hours_utc = [(22, 23), (23, 1)]

@dataclass
class TradeStats:
    trades_today: int = 0
    daily_pnl: float = 0.0
    active_trades: int = 0
    last_trade_time: Optional[datetime] = None
    consecutive_losses: int = 0
    total_risk_exposure: float = 0.0

class RiskManager:
    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.trade_stats = TradeStats()
        self.logger = logging.getLogger(__name__)
        self.daily_reset_time = None
        
    def reset_daily_stats(self):
        now = datetime.utcnow()
        if (self.daily_reset_time is None or 
            now.date() > self.daily_reset_time.date()):
            self.trade_stats.trades_today = 0
            self.trade_stats.daily_pnl = 0.0
            self.trade_stats.consecutive_losses = 0
            self.daily_reset_time = now
            self.logger.info("Daily trading stats reset")
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        self.reset_daily_stats()
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        
        # Check forbidden trading hours
        for start_hour, end_hour in self.risk_limits.forbidden_hours_utc:
            if start_hour <= end_hour:
                if start_hour <= current_hour < end_hour:
                    return False, f"Trading forbidden during hours {start_hour}-{end_hour} UTC"
            else:
                if current_hour >= start_hour or current_hour < end_hour:
                    return False, f"Trading forbidden during hours {start_hour}-{end_hour} UTC"
        
        # Check daily loss limit
        if self.trade_stats.daily_pnl <= self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: {self.trade_stats.daily_pnl:.2f}"
        
        # Check daily trade count
        if self.trade_stats.trades_today >= self.risk_limits.max_daily_trades:
            return False, f"Daily trade limit exceeded: {self.trade_stats.trades_today}"
        
        # Check concurrent trades
        if self.trade_stats.active_trades >= self.risk_limits.max_concurrent_trades:
            return False, f"Max concurrent trades exceeded: {self.trade_stats.active_trades}"
        
        # Check time between trades
        if (self.trade_stats.last_trade_time and 
            (current_time - self.trade_stats.last_trade_time).total_seconds() < 
            self.risk_limits.min_time_between_trades * 60):
            return False, "Minimum time between trades not met"
        
        # Check total risk exposure
        if self.trade_stats.total_risk_exposure >= self.risk_limits.max_total_risk:
            return False, f"Total risk exposure limit exceeded: {self.trade_stats.total_risk_exposure:.2f}"
        
        # Check consecutive losses (halt after 3)
        if self.trade_stats.consecutive_losses >= 3:
            return False, "Too many consecutive losses - trading halted"
        
        return True, "Trading allowed"
    
    def validate_trade_signal(self, signal: TradingSignal, account_balance: float, 
                            current_positions: List[Dict]) -> Tuple[bool, str, float]:
        # Check if trading is allowed
        allowed, reason = self.is_trading_allowed()
        if not allowed:
            return False, reason, 0.0
        
        # Check signal confidence
        if signal.confidence < 0.6:
            return False, f"Signal confidence too low: {signal.confidence:.2f}", 0.0
        
        # Calculate position size
        position_size = self.calculate_position_size(
            signal.entry_price, signal.stop_loss, account_balance
        )
        
        if position_size < 0.01:
            return False, "Position size too small", 0.0
        
        # Check for conflicting positions
        same_symbol_positions = [p for p in current_positions 
                               if p.get('symbol', '').startswith(signal.entry_price)]
        
        if same_symbol_positions:
            existing_direction = same_symbol_positions[0].get('type')
            signal_direction = 0 if signal.signal == SignalType.BUY else 1
            
            if existing_direction != signal_direction:
                return False, "Conflicting position exists for this symbol", 0.0
        
        return True, "Trade approved", position_size
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              account_balance: float, pip_value: float = 0.0001) -> float:
        if entry_price == 0 or stop_loss == 0:
            return 0.01
            
        risk_amount = account_balance * self.risk_limits.max_risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        pips_at_risk = price_diff / pip_value
        
        if pips_at_risk == 0:
            return 0.01
        
        # Position size calculation for forex
        position_size = risk_amount / (pips_at_risk * pip_value * 100000)
        
        # Round to 0.01 lot increments and cap at 1.0 lot
        return max(0.01, min(1.0, round(position_size, 2)))
    
    def update_trade_opened(self, position_size: float, risk_amount: float):
        self.trade_stats.trades_today += 1
        self.trade_stats.active_trades += 1
        self.trade_stats.last_trade_time = datetime.utcnow()
        self.trade_stats.total_risk_exposure += risk_amount
        
        self.logger.info(f"Trade opened - Active: {self.trade_stats.active_trades}, "
                        f"Today: {self.trade_stats.trades_today}, "
                        f"Risk: {self.trade_stats.total_risk_exposure:.2f}")
    
    def update_trade_closed(self, pnl: float, risk_amount: float):
        self.trade_stats.active_trades = max(0, self.trade_stats.active_trades - 1)
        self.trade_stats.daily_pnl += pnl
        self.trade_stats.total_risk_exposure = max(0, self.trade_stats.total_risk_exposure - risk_amount)
        
        if pnl < 0:
            self.trade_stats.consecutive_losses += 1
        else:
            self.trade_stats.consecutive_losses = 0
        
        self.logger.info(f"Trade closed - PnL: {pnl:.2f}, "
                        f"Daily PnL: {self.trade_stats.daily_pnl:.2f}, "
                        f"Active: {self.trade_stats.active_trades}")
    
    def get_emergency_stop_reasons(self) -> List[str]:
        reasons = []
        
        if self.trade_stats.daily_pnl <= self.risk_limits.max_daily_loss * 0.8:
            reasons.append("Approaching daily loss limit")
        
        if self.trade_stats.consecutive_losses >= 2:
            reasons.append("Multiple consecutive losses")
        
        if self.trade_stats.total_risk_exposure >= self.risk_limits.max_total_risk * 0.8:
            reasons.append("High risk exposure")
        
        return reasons
    
    def should_emergency_stop(self) -> bool:
        return len(self.get_emergency_stop_reasons()) > 0
    
    def get_status_report(self) -> Dict:
        self.reset_daily_stats()
        emergency_reasons = self.get_emergency_stop_reasons()
        
        return {
            "trades_today": self.trade_stats.trades_today,
            "daily_pnl": self.trade_stats.daily_pnl,
            "active_trades": self.trade_stats.active_trades,
            "consecutive_losses": self.trade_stats.consecutive_losses,
            "total_risk_exposure": self.trade_stats.total_risk_exposure,
            "emergency_stop": len(emergency_reasons) > 0,
            "emergency_reasons": emergency_reasons,
            "trading_allowed": self.is_trading_allowed()[0]
        }