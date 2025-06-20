# High-Profit Commodity CFDs for MT5 Automated Trading

## Executive Summary
This analysis identifies commodity CFDs available on MetaTrader 5 with high profit potential for automated trading. The commodities are categorized by type, volatility, and trading characteristics.

## Top High-Profit Commodities by Category

### 1. Precious Metals (Beyond XAUUSD)
- **XAGUSD (Silver)**: 3.5% daily volatility, $0.80 average daily range
- **XPDUSD (Palladium)**: 4.0% daily volatility, $50 average daily range - EXTREME volatility
- **XPTUSD (Platinum)**: 3.0% daily volatility, $30 average daily range

### 2. Energy Commodities
- **USOIL (WTI Crude)**: 3.2% daily volatility, $2.50 average daily range
- **UKOIL (Brent Crude)**: 3.3% daily volatility, $2.80 average daily range
- **NATGAS (Natural Gas)**: 5.0% daily volatility - HIGHEST VOLATILITY, $0.15 average daily range

### 3. Agricultural Commodities
- **COFFEE**: 3.0% daily volatility, weather-dependent, high profit potential
- **SUGAR**: 2.8% daily volatility, correlated with oil/ethanol
- **WHEAT**: 2.5% daily volatility, seasonal patterns, $15 average daily range
- **SOYBEAN**: 2.3% daily volatility, China-demand driven, $25 average daily range
- **COCOA**: 2.5% daily volatility, West Africa weather-dependent, $50 average daily range
- **CORN**: 2.2% daily volatility, ethanol demand correlation

### 4. Industrial Metals
- **NICKEL**: 3.5% daily volatility, $300 average daily range - VERY HIGH profit potential
- **COPPER**: 2.0% daily volatility, China PMI correlation, $0.05 average daily range
- **ZINC**: 2.2% daily volatility, construction demand, $50 average daily range
- **ALUMINUM**: 1.8% daily volatility, energy price correlation
- **LEAD**: 2.0% daily volatility, battery demand driven

## Extreme Volatility Commodities (Best for Scalping)
1. **NATGAS**: 5.0% daily volatility - Seasonal (Dec-Feb, Jun-Aug)
2. **XPDUSD**: 4.0% daily volatility - Industrial demand driven
3. **NICKEL**: 3.5% daily volatility - EV battery demand
4. **XAGUSD**: 3.5% daily volatility - Safe haven + industrial use

## Recommended Trading Approach by Volatility

### Extreme Volatility (>4% daily)
- **Symbols**: NATGAS, XPDUSD
- **Strategy**: Scalping with tight stops
- **Risk**: 0.3% per trade
- **TP Multiplier**: 2.5x

### Very High Volatility (3-4% daily)
- **Symbols**: XAGUSD, USOIL, UKOIL, NICKEL, COFFEE
- **Strategy**: Trend following with volatility filters
- **Risk**: 0.5% per trade
- **TP Multiplier**: 2.0x

### High Volatility (2-3% daily)
- **Symbols**: XPTUSD, WHEAT, COCOA, COPPER, SUGAR, SOYBEAN
- **Strategy**: Breakout trading
- **Risk**: 0.7% per trade
- **TP Multiplier**: 1.5x

## Key Trading Sessions for Commodities

### London Session (08:00-17:00 GMT)
- Best for: Metals, Energy, Agricultural
- High liquidity for European commodities

### New York Session (09:30-16:00 EST)
- Best for: Energy (USOIL), Agricultural (Chicago markets)
- Overlap with London provides maximum liquidity

### Shanghai Session (09:00-15:00 CST)
- Best for: Industrial metals
- China demand influences pricing

### Chicago Session (08:30-13:15 CST)
- Best for: Agricultural commodities
- CBOT trading hours

## Seasonal Trading Opportunities

### Winter (Dec-Feb)
- **NATGAS**: Heating demand spike
- **COFFEE**: Brazil frost risk

### Spring (Mar-May)
- **WHEAT**: Planting season volatility
- **CORN**: Pre-planting price movements

### Summer (Jun-Aug)
- **NATGAS**: Cooling demand
- **Agricultural**: Weather market volatility

### Fall (Sep-Nov)
- **Agricultural**: Harvest pressure
- **Metals**: Construction season strength

## Risk Management Guidelines

### Position Sizing by Commodity Type
- **Extreme Volatility** (NATGAS, XPDUSD): 0.3% risk per trade
- **Precious Metals**: 0.5-0.7% risk per trade
- **Energy**: 0.5-0.7% risk per trade
- **Agricultural**: 0.7-0.8% risk per trade
- **Industrial Metals**: 0.6-0.8% risk per trade

### Maximum Spread Limits
- **Precious Metals**: 30-100 pips (XPDUSD highest)
- **Energy**: 3-10 pips
- **Agricultural**: 3-10 pips
- **Industrial Metals**: 5-20 pips

## Integration with Current Trading System

To add these commodities to your ultra_trading_engine.py:

1. Import the commodity configuration:
```python
from commodity_cfds_config import COMMODITY_CFDS, get_high_profit_commodities
```

2. Add commodity symbols to the HIGH_PROFIT_SYMBOLS dictionary
3. Adjust spread limits based on commodity type
4. Implement seasonal filters for agricultural commodities
5. Add news event calendars for commodity-specific events

## Top 10 Commodities for Automated Trading (Ranked by Profit Potential)

1. **NATGAS** - 5.0% volatility, extreme seasonal moves
2. **XPDUSD** - 4.0% volatility, industrial demand spikes
3. **NICKEL** - 3.5% volatility, EV battery boom
4. **XAGUSD** - 3.5% volatility, dual safe-haven/industrial
5. **UKOIL** - 3.3% volatility, geopolitical sensitivity
6. **USOIL** - 3.2% volatility, global benchmark
7. **COFFEE** - 3.0% volatility, weather shocks
8. **XPTUSD** - 3.0% volatility, auto industry cycles
9. **SUGAR** - 2.8% volatility, ethanol correlation
10. **COCOA** - 2.5% volatility, supply constraints

## Implementation Notes

- Start with 2-3 commodities from different categories for diversification
- Monitor correlation between commodities (e.g., USOIL vs UKOIL)
- Adjust lot sizes based on account currency and commodity contract specifications
- Use tighter stops during news events (EIA inventory, USDA reports)
- Consider time-of-day filters for each commodity's primary session