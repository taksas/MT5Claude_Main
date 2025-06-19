# Ultimate Trading System - Component Guide

## Overview
The trading system is now separated into two main components:
1. **Trading Engine** - Handles all trading logic and order execution
2. **Visualizer** - Provides real-time monitoring dashboard

## Components

### 1. Trading Engine (trading_engine.py)
- Core trading logic
- Strategy analysis and signal generation
- Order placement and position management
- Sends strategy confidence data to visualizer

### 2. Visualizer (visualizer.py)
- Real-time dashboard showing:
  - Account balance and equity in JPY
  - Current profit/loss
  - Open positions with details
  - Closed positions from today
  - Strategy confidence levels for each symbol
- Updates every second
- Clear terminal-based display

### 3. Main Coordinator (main.py)
- Manages both components
- Handles inter-process communication
- Auto-restarts components if they crash
- Graceful shutdown handling

## Running the System

### Option 1: Run Everything Together
```bash
python3 main.py
```
This starts both the trading engine and visualizer in separate processes.

### Option 2: Run Components Separately

Terminal 1 - Trading Engine:
```bash
python3 trading_engine.py
```

Terminal 2 - Visualizer:
```bash
python3 visualizer.py
```

## Display Information

The visualizer shows:

### Account Status
- Balance: Total account balance in JPY
- Equity: Current equity including floating P&L
- Floating: Current unrealized profit/loss
- Daily P&L: Total profit/loss for today

### Open Positions
- Ticket number
- Symbol
- Direction (BUY/SELL)
- Volume
- Entry price
- Current price
- P&L in JPY
- Time held

### Closed Today
- Recent closed positions
- Exit details and final P&L

### Strategy Signals
- Current signal type (BUY/SELL/NONE)
- Overall confidence percentage
- Individual strategy scores:
  - Trend
  - RSI
  - Bollinger Bands
  - Momentum
  - Volume

## Key Features

1. **Multi-Process Architecture**
   - Trading and visualization run independently
   - System continues trading even if visualizer crashes
   - Clean separation of concerns

2. **Real-time Updates**
   - Dashboard refreshes every second
   - Shows live strategy analysis
   - Tracks all positions and P&L

3. **JPY Account Support**
   - All values displayed in JPY
   - Proper formatting with ¥ symbol
   - Adjusted thresholds for JPY accounts

## Troubleshooting

- If visualizer shows "No active signals", the trading engine is analyzing but hasn't found trade opportunities
- If no positions open, check spread conditions and market hours
- Use Ctrl+C to cleanly shutdown the system