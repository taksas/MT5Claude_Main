# High-Profit Trading Symbols Configuration

## Symbol Categories & Expected Returns

### 1. Stock Indices - HIGHEST RETURNS (60-85% Annual)
| Symbol | Daily Range | Best Hours (GMT) | Risk | Target Pips | Notes |
|--------|-------------|------------------|------|-------------|--------|
| US30 | 200-660 pts | 14:30-21:00 | 0.8% | 150-200 | Tuesday most volatile |
| USTEC/NAS100 | 100-316 pts | 14:30-21:00 | 0.8% | 150+ (1.5x ATR) | Thursday peaks |
| GER40/DAX | 80-120 pts | 07:00-09:00, 14:30-16:00 | 0.8% | 80-120 | EUR correlation |

### 2. Precious Metals - STABLE HIGH RETURNS (45-60% Annual)
| Symbol | Daily Range | Best Hours | Risk | Target Pips | Spread Limit |
|--------|-------------|------------|------|-------------|--------------|
| XAUUSD | 150-250 pips | London-NY overlap | 0.7% | 75-150 | 50 pips |
| XAGUSD | 300-500 pips | London-NY overlap | 0.7% | 150-300 | 100 pips |

### 3. Exotic Forex - EXTREME VOLATILITY (200-500 pip ranges)
| Symbol | Daily Range | Best Hours | Risk | Target Pips | Spread Limit |
|--------|-------------|------------|------|-------------|--------------|
| USDZAR | 200-300 pips | 12:00-16:00 GMT | 0.5% | 150-200 | 150 pips |
| USDMXN | 150-250 pips | 14:00-21:00 GMT | 0.5% | 100-150 | 30 pips |
| USDTRY | 500+ pips | 07:00-17:00 GMT | 0.5% | 200-300 | 50 pips |

### 4. Cross-Currency Mean Reversion (25-40% Annual)
| Symbol | Strategy | Risk | Target Pips | Win Rate |
|--------|----------|------|-------------|----------|
| EURGBP | Range trading | 1.0% | 30-50 | 55-60% |
| AUDNZD | Mean reversion | 1.0% | 40-60 | 55-60% |
| GBPJPY | Volatility breakout | 0.5% | 100-150 | 50-55% |

## Optimized Trading Parameters

### Entry Requirements by Instrument Type
```python
INSTRUMENT_REQUIREMENTS = {
    "index": {
        "min_confidence": 0.60,  # Higher for indices
        "min_strategies": 15,    # More confirmation needed
        "avoid_news": True,      # Very news sensitive
        "time_filter": True      # Strict session timing
    },
    "metal": {
        "min_confidence": 0.50,
        "min_strategies": 12,
        "seasonal_filter": True,  # Best in certain months
        "correlation_check": True # Check USD strength
    },
    "exotic": {
        "min_confidence": 0.70,  # Need high confidence
        "min_strategies": 20,    # Many confirmations
        "spread_filter": True,   # Critical for exotics
        "volatility_filter": True # Avoid extreme moves
    }
}
```

### Risk Management Rules
1. **Position Sizing**:
   - Indices: 0.8% risk (high profit potential)
   - Metals: 0.7% risk (stable volatility)
   - Exotic Pairs: 0.5% risk (extreme volatility)
   - Cross Pairs: 1.0% risk (mean reversion)

2. **Maximum Concurrent Positions**:
   - Total: 5 positions max
   - Per category: 2 max
   - Correlation check required

3. **Stop Loss Guidelines**:
   - Indices: 100-150 points
   - Gold: 100-150 pips
   - Silver: 200-300 pips
   - Exotic: 150-250 pips
   - Cross: 50-80 pips

## Trading Schedule Optimization

### Best Trading Windows
- **Asian Session**: Avoid (low liquidity)
- **European Open** (07:00-09:00 GMT): DAX, EUR crosses
- **London Session** (08:00-12:00 GMT): Metals, GBP pairs
- **NY Overlap** (12:00-16:00 GMT): HIGHEST VOLUME - All instruments
- **US Session** (14:30-21:00 GMT): Indices, USD pairs

### Avoid Trading
- First/last 30 minutes of sessions
- Major news events (NFP, FOMC, ECB)
- Low liquidity holidays
- Month/quarter end (rebalancing)

## Expected Performance

### Realistic Monthly Returns by Strategy
- Conservative (single instrument): 3-5%
- Moderate (diversified): 5-8%
- Aggressive (all instruments): 8-12%

### Key Success Factors
1. **Strict Risk Management** - Never exceed position limits
2. **Quality Over Quantity** - Wait for high-confidence setups
3. **Instrument Specialization** - Master 2-3 instruments first
4. **Time Zone Discipline** - Trade only optimal hours
5. **Correlation Awareness** - Avoid overexposure

## Implementation Priority
1. Start with Gold (XAUUSD) - most liquid, clear patterns
2. Add US30 index - high returns, good liquidity  
3. Include USDZAR - best exotic for algorithms
4. Expand to EURGBP - stable mean reversion
5. Scale to full portfolio once profitable

Remember: These are REALISTIC targets based on extensive research. Claims of 20-30% monthly returns are unsustainable and lead to account destruction.