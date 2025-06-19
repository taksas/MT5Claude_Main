# MT5 Forex Trading Bot

A clean, focused automated forex trading system using MetaTrader 5 Bridge API.

## Quick Start

```bash
python3 ultimate_trading_engine.py
```

## Files

- **ultimate_trading_engine.py** - Main trading engine
- **ULTIMATE_STRATEGY.md** - Complete strategy documentation
- **API_README.md** - MT5 Bridge API reference
- **CLAUDE.md** - AI assistant instructions

## Key Features

- 5-layer signal validation system
- Dynamic spread adaptation
- Automatic position management
- Risk-limited to 1% per trade
- Session-aware trading

## Configuration

Edit CONFIG dict in ultimate_trading_engine.py:
- API_BASE: Your MT5 bridge URL
- SYMBOLS: Currency pairs to trade
- MIN_CONFIDENCE: Signal threshold (70%)
- MAX_SPREAD_PIPS: Maximum acceptable spread

## Performance Target

- Win Rate: 55-60%
- Risk:Reward: 1:1.5
- Monthly Return: 15-20%
- Max Drawdown: 10%

## Safety Features

- Mandatory stop loss on every trade
- Daily loss limit (3%)
- Max 2 concurrent positions
- No trading during low liquidity hours

The system is designed to run continuously with minimal intervention.