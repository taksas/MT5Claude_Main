#!/usr/bin/env python3
"""
Script to print MT5 filling mode constants for verification
Run this on the Windows side where MT5 is installed
"""

try:
    import MetaTrader5 as mt5
    
    print("MT5 Filling Mode Constants:")
    print(f"ORDER_FILLING_FOK = {mt5.ORDER_FILLING_FOK}")
    print(f"ORDER_FILLING_IOC = {mt5.ORDER_FILLING_IOC}")
    print(f"ORDER_FILLING_RETURN = {mt5.ORDER_FILLING_RETURN}")
    
    # Also check if there are any other filling constants
    for attr in dir(mt5):
        if "FILLING" in attr and "ORDER" in attr:
            print(f"{attr} = {getattr(mt5, attr)}")
            
except ImportError:
    print("MetaTrader5 module not available. This script must be run where MT5 is installed.")