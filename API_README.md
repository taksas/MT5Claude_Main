# MetaTrader 5 Bridge API - Complete Documentation

## Overview
This is a REST API bridge server that allows external AI trading agents to control MetaTrader 5 (MT5) terminal. The system enables complete trading automation through HTTP endpoints.

## System Architecture
- **API Server**: FastAPI-based REST server running on Windows
- **MT5 Integration**: Direct connection to MT5 terminal via MetaTrader5 Python package
- **Network**: Accessible from WSL/Docker/external systems via HTTP

## Quick Start

### 1. Prerequisites
- Windows 10/11 with MT5 Terminal installed
- Python 3.8+
- MT5 account with algo trading enabled
- Network access between client and server

### 2. Installation
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-dotenv pandas pytz MetaTrader5
```

### 3. Configuration
Create `.env` file:
```
MT5_LOGIN=12345678
MT5_PASSWORD="your_password"
MT5_SERVER="YourBroker-Server"
MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
```

### 4. MT5 Terminal Setup
1. Open MT5 → Tools → Options → Expert Advisors
2. Enable "Allow automated trading"
3. Enable "Allow DLL imports"
4. Click the AutoTrading button (should be green)

### 5. Start Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Status Endpoints

#### GET /status/ping
Health check endpoint.
```json
Response: {"status": "pong"}
```

#### GET /status/mt5
MT5 connection status and terminal info.
```json
{
  "connected": true,
  "trade_allowed": true,
  "dlls_allowed": true,
  "build": 3802,
  "company": "MetaQuotes Software Corp."
}
```

### Account Information

#### GET /account/
Get complete account details.
```json
{
  "login": 12345678,
  "balance": 10000.0,
  "equity": 10050.0,
  "margin": 100.0,
  "margin_free": 9950.0,
  "margin_level": 10050.0,
  "leverage": 100,
  "currency": "USD"
}
```

### Market Data

#### GET /market/symbols
List all available symbols with details.

#### GET /market/symbols/tradable
Get only tradable symbols (TRADE_MODE_FULL).
```json
["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"]
```

#### GET /market/symbols/{symbol}
Get specific symbol details.
```json
{
  "name": "EURUSD",
  "point": 0.00001,
  "digits": 5,
  "trade_mode": "FULL",
  "volume_min": 0.01,
  "volume_max": 100.0,
  "volume_step": 0.01
}
```

#### POST /market/history
Get historical OHLC data.
```json
Request:
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "count": 100
}

Response:
[{
  "time": "2024-01-01T00:00:00",
  "open": 1.10234,
  "high": 1.10345,
  "low": 1.10123,
  "close": 1.10234,
  "tick_volume": 1234,
  "spread": 2
}]
```

### Trading Operations

#### POST /trading/orders
Place new order (market or pending).

**Market Order:**
```json
{
  "action": 1,        // TRADE_ACTION_DEAL
  "symbol": "EURUSD",
  "volume": 0.01,
  "type": 0,          // ORDER_TYPE_BUY
  "sl": 1.10000,
  "tp": 1.11000,
  "comment": "AI trade"
}
```

**Pending Order:**
```json
{
  "action": 5,        // TRADE_ACTION_PENDING
  "symbol": "EURUSD",
  "volume": 0.01,
  "type": 2,          // ORDER_TYPE_BUY_LIMIT
  "price": 1.10100,
  "sl": 1.10000,
  "tp": 1.10300
}
```

#### GET /trading/positions
Get all open positions.
```json
[{
  "ticket": 12345678,
  "symbol": "EURUSD",
  "type": "BUY",
  "volume": 0.01,
  "price_open": 1.10234,
  "price_current": 1.10250,
  "profit": 1.60
}]
```

#### DELETE /trading/positions/{ticket}
Close position (full or partial).
```json
Request:
{
  "volume": 0.01,    // Optional for partial close
  "deviation": 20
}
```

#### PATCH /trading/positions/{ticket}
Modify position SL/TP.
```json
{
  "sl": 1.10100,
  "tp": 1.10400
}
```

#### GET /trading/orders
Get pending orders.

#### DELETE /trading/orders/{ticket}
Cancel pending order.

## Order Type Constants

### Action Types
- 1: TRADE_ACTION_DEAL (Market order)
- 5: TRADE_ACTION_PENDING (Pending order)
- 6: TRADE_ACTION_MODIFY (Modify order)
- 7: TRADE_ACTION_REMOVE (Cancel order)

### Order Types
- 0: ORDER_TYPE_BUY
- 1: ORDER_TYPE_SELL
- 2: ORDER_TYPE_BUY_LIMIT
- 3: ORDER_TYPE_SELL_LIMIT
- 4: ORDER_TYPE_BUY_STOP
- 5: ORDER_TYPE_SELL_STOP

### Timeframes
- M1: 1 minute
- M5: 5 minutes
- M15: 15 minutes
- M30: 30 minutes
- H1: 1 hour
- H4: 4 hours
- D1: 1 day
- W1: 1 week
- MN1: 1 month

## Error Handling

### HTTP Status Codes
- **200**: Success
- **201**: Created (order placed)
- **400**: Bad request (invalid parameters)
- **404**: Not found (symbol/ticket)
- **422**: Unprocessable (trade logic error)
- **503**: MT5 not connected

### Common MT5 Return Codes
- 10009: TRADE_RETCODE_DONE (Success)
- 10004: TRADE_RETCODE_REQUOTE
- 10006: TRADE_RETCODE_REJECT
- 10016: TRADE_RETCODE_INVALID_STOPS
- 10019: TRADE_RETCODE_NO_MONEY

## Network Setup for WSL

### Finding Host IP
```bash
# In Windows CMD:
ipconfig
# Look for "Ethernet adapter vEthernet (WSL)"
# Use that IPv4 address
```

### Accessing from WSL
```python
API_BASE = "http://172.28.144.1:8000"  # Replace with your host IP
```

### Firewall Configuration
If connection fails:
1. Windows Defender Firewall → Advanced Settings
2. Inbound Rules → New Rule
3. Port → TCP → 8000
4. Allow connection → Apply to all

## Usage Examples

### Python Client Example
```python
import requests

class MT5Client:
    def __init__(self, base_url="http://172.28.144.1:8000"):
        self.base_url = base_url
    
    def get_account(self):
        return requests.get(f"{self.base_url}/account/").json()
    
    def place_market_order(self, symbol, volume, order_type):
        data = {
            "action": 1,
            "symbol": symbol,
            "volume": volume,
            "type": 0 if order_type == "BUY" else 1
        }
        return requests.post(f"{self.base_url}/trading/orders", json=data)
    
    def get_positions(self):
        return requests.get(f"{self.base_url}/trading/positions").json()
```

### Complete Trading Workflow
```python
# 1. Check account
account = client.get_account()
if account["margin_free"] < 100:
    print("Insufficient margin")
    exit()

# 2. Get symbol info
symbol_info = requests.get(f"{base_url}/market/symbols/EURUSD").json()

# 3. Get historical data for analysis
history = requests.post(f"{base_url}/market/history", 
                       json={"symbol": "EURUSD", "timeframe": "H1", "count": 100})

# 4. Place order based on analysis
order = client.place_market_order("EURUSD", 0.01, "BUY")

# 5. Monitor position
positions = client.get_positions()

# 6. Modify SL/TP
if positions:
    ticket = positions[0]["ticket"]
    requests.patch(f"{base_url}/trading/positions/{ticket}", 
                  json={"sl": 1.10000, "tp": 1.11000})

# 7. Close position
requests.delete(f"{base_url}/trading/positions/{ticket}")
```

## Best Practices

1. **Connection Management**: Always check `/status/mt5` before trading
2. **Error Handling**: Implement retry logic for network failures
3. **Risk Management**: Validate positions don't exceed risk limits
4. **Logging**: Log all trading operations for audit
5. **Testing**: Use demo account first

## Troubleshooting

### MT5 Not Connected
- Check MT5 terminal is running
- Verify login credentials in .env
- Ensure algo trading is enabled
- Check if MT5 path is correct

### Network Issues
- Verify firewall settings
- Check host IP hasn't changed
- Test with curl: `curl http://172.28.144.1:8000/status/ping`

### Trading Errors
- Check account has sufficient margin
- Verify symbol is tradable
- Ensure market is open
- Check for valid SL/TP levels

## Security Notes

1. This API has no authentication - use only on trusted networks
2. Consider adding API key authentication for production
3. Use HTTPS with proper certificates for internet exposure
4. Implement rate limiting to prevent abuse
5. Log all operations for security audit

## Performance Tips

1. Use connection pooling in clients
2. Batch operations when possible
3. Cache symbol information
4. Implement websocket for real-time data
5. Use async operations for better throughput