#!/usr/bin/env python3
"""
Test system with low balance
"""

from components.risk_management import RiskManagement
from components.mt5_api_client import MT5APIClient
from components.trading_config import CONFIG

# Initialize components
risk_manager = RiskManagement()
api_client = MT5APIClient(CONFIG["API_BASE"])

# Get current account info
account_info = api_client.get_account_info()
if account_info:
    balance = account_info['balance']
    equity = account_info['equity']
    margin = account_info.get('margin', 0)
    
    print(f"Current Balance: ¥{balance:,.0f}")
    print(f"Current Equity: ¥{equity:,.0f}")
    
    # Test account safety check
    safe, reason = risk_manager.check_account_safety(balance, equity, margin, 0)
    
    if safe:
        print("✅ Account is SAFE to trade - No minimum balance restriction!")
    else:
        print(f"❌ Account not safe: {reason}")
    
    # Test position sizing
    test_symbol = "EURUSD"
    sl_distance = 0.001  # 10 pips
    current_price = 1.0850
    
    # Mock symbol info
    symbol_info = {
        'trade_contract_size': 100000,
        'volume_min': 0.01,
        'volume_max': 100,
        'volume_step': 0.01,
        'digits': 5
    }
    
    # Mock account info with leverage
    account_info = {
        'balance': balance,
        'equity': balance,
        'leverage': 100  # Typical forex leverage
    }
    
    position_size = risk_manager.calculate_position_size(
        test_symbol, sl_distance, current_price, balance, symbol_info, account_info
    )
    
    print(f"\nPosition size calculation for {test_symbol}:")
    print(f"  Balance: ¥{balance:,.0f}")
    print(f"  Risk per trade: {CONFIG['RISK_PER_TRADE']*100}%")
    print(f"  Calculated volume: {position_size} lots")
else:
    print("Could not connect to account")