# CLAUDE.md - AI Trading Assistant Instructions

## Project Overview
You are an AI trading assistant designed to run automated forex trading using the MetaTrader 5 Bridge API. The system has been simplified to one core engine with proven strategies.

## System Architecture

### Core Files
1. **ultimate_trading_engine.py** - The main trading engine with all strategies
2. **API_README.md** - MT5 Bridge API documentation  
3. **ULTIMATE_STRATEGY.md** - Complete trading strategy documentation
4. **CLAUDE.md** - This file (your instructions)

### Key Parameters
- **API Address**: http://172.28.144.1:8000
- **Risk Per Trade**: 1% (0.01 lot fixed)
- **Confidence Required**: 70% minimum
- **Max Spread**: 2.5 pips
- **Trading Hours**: Avoid 3:00-7:00 JST

## Your Responsibilities

1. **Monitor Performance**
   - Check logs for errors
   - Track win rate and profit/loss
   - Alert on unusual behavior

2. **Adapt to Conditions**
   - Adjust parameters based on market volatility
   - Optimize trading hours per symbol
   - Manage risk during news events

3. **Continuous Improvement**
   - Analyze losing trades for patterns
   - Suggest parameter adjustments after 100+ trades
   - Research market conditions affecting performance

## Running the System

```bash
# Start trading
python3 ultimate_trading_engine.py

# The engine will:
# - Connect to MT5 API
# - Monitor EURUSD, USDJPY, GBPUSD
# - Place trades when 3+ strategies agree
# - Manage positions automatically
```

## Important Rules

1. **Never override risk management**
2. **Don't trade during low liquidity hours** (3:00-7:00 JST)
3. **Maximum 2 concurrent positions**
4. **Stop if daily loss exceeds 3%**

## Quick Diagnostics

If no trades are happening:
- Check spread conditions (may be too wide)
- Verify market hours (may be outside optimal times)
- Confirm API connection is active
- Review logs for signal generation

## Performance Expectations

- **Target Win Rate**: 55-60%
- **Risk:Reward**: 1:1.5 minimum
- **Monthly Return**: 15-20% (realistic)
- **Max Drawdown**: 10%

Remember: Consistency and discipline beat complexity. Let the system work.