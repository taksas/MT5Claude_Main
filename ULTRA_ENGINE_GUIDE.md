# Ultra Trading Engine - 100 Signal System

## Overview
The Ultra Trading Engine implements an aggressive trading strategy using 100 different technical indicators across 10 categories. It aims to have positions in every symbol approximately every 10 minutes.

## Configuration (Aggressive Mode)
- **Minimum Confidence**: 35% (lowered from 60%)
- **Minimum Quality**: 30% (lowered from 60%) 
- **Minimum Strategies**: 15 out of 100 indicators must confirm
- **Max Concurrent Positions**: 10
- **Risk Per Trade**: 2% (doubled)
- **Position Interval**: 600 seconds (10 minutes target)
- **Max Daily Loss**: 10%

## 100 Indicators Across 10 Categories

### 1. PRICE ACTION (10 indicators) - Weight: 2.0
- Pin Bar Detection (Bullish/Bearish)
- Engulfing Pattern (Bullish/Bearish)
- Doji Detection
- Hammer/Hanging Man
- Three White Soldiers/Three Black Crows
- Inside Bar
- Price Action Momentum

### 2. CHART PATTERNS (10 indicators) - Weight: 1.5
- Head and Shoulders
- Double Top/Bottom
- Triangle Pattern
- Channel Detection (Upper/Lower)
- Flag Pattern
- Wedge Pattern (Rising/Falling)

### 3. MATHEMATICAL INDICATORS (10 indicators) - Weight: 1.2
- Fibonacci Retracement Levels (23.6%, 38.2%, 50%, 61.8%)
- Pivot Points (Standard, R1, S1)
- Linear Regression (Slope & Deviation)

### 4. VOLATILITY ANALYSIS (10 indicators) - Weight: 1.0
- ATR and ATR Ratio
- Bollinger Band Width & Squeeze
- Keltner Channels (Upper/Lower)
- Historical Volatility
- Volatility Ratio
- Donchian Channels

### 5. MARKET STRUCTURE (10 indicators) - Weight: 2.0
- Support/Resistance Level Detection
- Market Structure Break (Up/Down)
- Higher Highs/Lower Lows Count
- Range Detection
- Swing Point Analysis

### 6. MOMENTUM ANALYSIS (10 indicators) - Weight: 1.8
- Rate of Change (5, 10, 20 periods)
- Momentum Oscillator
- Price Oscillator
- Commodity Channel Index (CCI)
- Williams %R
- Ultimate Oscillator

### 7. VOLUME/ORDER FLOW (10 indicators) - Weight: 1.3
- Volume Moving Averages & Ratio
- On Balance Volume (OBV)
- Accumulation/Distribution Line
- Chaikin Money Flow
- Volume Price Trend (VPT)
- Force Index
- Money Flow Index (MFI)

### 8. TIME-BASED PATTERNS (10 indicators) - Weight: 0.8
- Asian/London/NY Session Detection
- Session Overlap
- Day of Week Patterns
- Hourly/Two-Hour Momentum
- Opening Range Breakout

### 9. STATISTICAL ANALYSIS (10 indicators) - Weight: 1.0
- Z-Score
- Percentile Rank
- Standard Deviation Bands
- Skewness & Kurtosis
- Autocorrelation
- Mean Reversion Indicator
- Efficiency Ratio
- Hurst Exponent

### 10. ADVANCED COMPOSITE (10 indicators) - Weight: 2.5
- MACD (Line, Signal, Histogram)
- RSI
- Stochastic (K, D)
- ADX (+DI, -DI)
- Ichimoku Cloud Components

## Deep Analysis Process

### Signal Generation
1. **Calculate all 100 indicators** for each symbol
2. **Weight each category** based on importance
3. **Score buy/sell signals** independently
4. **Count active strategies** (must have 15+ confirming)
5. **Generate signal** if confidence > 35%

### Position Management (Aggressive)
- **Quick Profit**: Close at 500 JPY profit
- **Time Exit**: 
  - Profitable positions: 15 minutes
  - Losing positions: 30 minutes
- **Breakeven**: Move SL to BE at 200 JPY profit after 3 minutes

### Risk Management
- **Position Size**: 0.02 lots (doubled for aggressive mode)
- **Stop Loss**: 1.0x ATR (tighter stops)
- **Take Profit**: 1.0x ATR (1:1 RR for more trades)
- **Max Daily Loss**: 10% of account

## Trading Logic

### Symbol Discovery
- Automatically discovers all tradable forex pairs
- Filters for proper trade modes and minimum volumes
- Trades up to 30 symbols simultaneously

### Aggressive Features
1. **Per-Symbol Cooldown**: 10-minute minimum between trades per symbol
2. **Continuous Scanning**: Checks all symbols every 15 seconds
3. **Wider Spread Tolerance**: Up to 3.5 pips
4. **Session Awareness**: Increases activity during overlaps
5. **Statistical Extremes**: Trades mean reversion aggressively

## Example Signal Analysis

```
Symbol: EURUSD#
Buy Score: 42.5 / 100 (42.5% confidence)
Active Strategies: 18 / 100

Contributing Factors:
- Bullish Pin Bar (PA: 2.0)
- At Fibonacci 61.8% Support (Math: 1.2)
- Bollinger Squeeze + Momentum (Vol: 1.5)
- At Support Level (Structure: 3.0)
- RSI Oversold 28 (Composite: 3.0)
- London Session Active (Time: 0.96)
- Z-Score -2.1 (Statistical: 1.2)
```

## Performance Expectations

### Aggressive Mode Targets
- **Trades Per Day**: 50-100+
- **Win Rate**: 45-55% (lower but more volume)
- **Average Trade Duration**: 15-30 minutes
- **Daily Volatility**: Higher (up to 10% swings)

### Benefits
1. **High Trade Frequency**: Multiple opportunities per hour
2. **Market Coverage**: Trades all major pairs
3. **Quick Profits**: Fast turnover of capital
4. **Diversification**: Risk spread across many positions

### Risks
1. **Higher Drawdowns**: Up to 10% daily possible
2. **Increased Costs**: More spreads and commissions
3. **Overtrading**: May enter suboptimal trades
4. **Psychological Pressure**: Requires discipline

## Monitoring

The visualizer will show:
- All active signals across 30+ symbols
- Real-time indicator calculations
- Position status and P&L
- Daily statistics

## Usage

```bash
# Start the ultra trading engine
python3 main.py

# Or run standalone
python3 ultra_trading_engine.py
```

The system will aggressively seek trading opportunities across all available forex pairs, using deep multi-angle analysis to identify high-probability setups while maintaining acceptable risk levels.