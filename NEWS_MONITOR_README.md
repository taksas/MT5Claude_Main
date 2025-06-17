# News Monitor Module Documentation

## Overview
The News Monitor module provides real-time monitoring of forex news and economic events to help avoid trading during high-impact news releases that can cause unpredictable market volatility.

## Features

### 1. Multi-Source News Aggregation
- Fetches economic calendar events from multiple sources
- Monitors high-impact economic releases (NFP, Interest Rates, GDP, etc.)
- Tracks news sentiment for major currencies

### 2. Risk Assessment
- Categorizes events by impact level (high/medium/low)
- Identifies which currency pairs are affected
- Provides time-based risk windows

### 3. Trading Recommendations
- Symbol-specific safety checks
- Market sentiment analysis
- Integration with trading strategies

## Usage

### Basic Usage

```python
from news_monitor import NewsMonitor

# Create monitor instance
monitor = NewsMonitor()

# Check if safe to trade a symbol
is_safe, risk_level, events = monitor.check_symbol_risk('EURUSD')

# Get trading recommendation
recommendation = monitor.get_trading_recommendation('EURUSD')
```

### Integration with Trading Engine

```python
from news_aware_trader import NewsAwareTradingEngine

# Create news-aware trading engine
engine = NewsAwareTradingEngine(
    api_url="http://172.28.144.1:8000",
    risk_per_trade=0.01,
    news_risk_tolerance='low'  # 'low', 'medium', or 'high'
)
```

## Risk Tolerance Levels

- **Low**: Avoids trading during any medium or high impact events
- **Medium**: Only avoids high impact events
- **High**: Only avoids if multiple high impact events coincide

## Key Methods

### `check_symbol_risk(symbol, time_window_minutes=30)`
Checks if it's safe to trade a specific symbol within the time window.

### `get_trading_recommendation(symbol)`
Provides comprehensive trading recommendation based on news and sentiment.

### `should_avoid_trading(symbol, risk_tolerance='low')`
Simple boolean check for whether to avoid trading.

### `get_market_sentiment()`
Returns sentiment analysis for all major currencies.

## Event Categories Monitored

### High Impact Events
- Central Bank Interest Rate Decisions
- Non-Farm Payrolls (NFP)
- GDP Releases
- Inflation Data (CPI)
- FOMC Minutes
- Central Bank Press Conferences

### Medium Impact Events
- Retail Sales
- Manufacturing PMI
- Consumer Confidence
- Trade Balance
- Unemployment Claims

## Implementation Notes

1. **Data Sources**: Currently uses mock data for testing. In production, implement proper web scraping or use economic calendar APIs.

2. **Update Frequency**: News cache updates every 5 minutes by default.

3. **Time Windows**: Checks for events within 30 minutes by default (configurable).

4. **Currency Detection**: Automatically identifies affected currencies from event descriptions.

## Testing

Run the test script to verify functionality:

```bash
python test_news_monitor.py
```

## Production Considerations

1. **API Integration**: Consider using professional economic calendar APIs:
   - ForexFactory API
   - Investing.com API
   - Economic calendar services

2. **Rate Limiting**: Implement proper rate limiting for web scraping

3. **Error Handling**: Robust error handling for network failures

4. **Caching**: Efficient caching to minimize API calls

5. **Real-time Updates**: Consider WebSocket connections for real-time news

## Example Output

```
Checking EURUSD:
  Safe to trade: False
  Risk level: high
  Recommendation: avoid
  Reason: High impact news event within 30 minutes
  Upcoming events:
    - US Non-Farm Payrolls (high impact)
    - ECB Interest Rate Decision (high impact)
```