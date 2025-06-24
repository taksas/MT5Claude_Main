# CLAUDE.md - AI Trading Assistant Instructions

## Project Overview
You are an AI trading assistant designed to run automated forex trading using the MetaTrader 5 Bridge API. The system has been simplified to one core engine with proven strategies.

## System Architecture

### Core Files
1. **main.py** - Entry point for engine and visualizer
2. **API_README.md** - MT5 Bridge API documentation  
3. **CLAUDE.md** - This file (your instructions)
4. **components** - this is a file for engine, visualizer, model, config and any other trading system

### Key Parameters
- **API Address**: http://172.28.144.1:8000
- **Risk Per Trade**: 1% (0.01 lot fixed)
- **Confidence Required**: 70% minimum
- **Max Spread**: 2.5 pips
- **Symbols**: all tradable symbols are with "#", for example"JPYUSD#", not "JPYUSD".

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
python3 main.py

# The engine will:
# - Connect to MT5 API
# - Manage positions automatically

# The visualizer will:
# - Show engine metrics for users REALTIME
```

# Important Notice
If you do test, need to use "USDJPY#" symbol.

## Quick Diagnostics

If no trades are happening:
- Check spread conditions (may be too wide)
- Verify market hours (may be outside optimal times)
- Confirm API connection is active
- Review logs for signal generation