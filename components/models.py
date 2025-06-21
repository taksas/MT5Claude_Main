"""
Trading models and data structures
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    type: SignalType
    confidence: float
    entry: float
    sl: float
    tp: float
    reason: str
    strategies: Optional[Dict[str, float]] = None
    quality: float = 0


@dataclass 
class Trade:
    ticket: int
    symbol: str
    type: str
    entry_price: float
    sl: float
    tp: float
    volume: float
    entry_time: datetime