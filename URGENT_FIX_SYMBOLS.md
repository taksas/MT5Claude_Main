# URGENT: Fix Symbol Visibility in MT5

## The Problem
All trading symbols (EURUSD#, USDJPY#, etc.) are NOT visible in MT5 Market Watch.
This is why you're getting "Order check failed: Done (retcode: 0)" errors.

## Solution - Do This NOW:

1. **Open MT5 Terminal on Windows**

2. **Open Market Watch**:
   - Press `Ctrl+M` or go to View → Market Watch

3. **Show All Symbols**:
   - Right-click in the Market Watch window
   - Select "Show All" 
   - This should display all available symbols

4. **Find Your Trading Symbols**:
   - Look for symbols ending with `#` (like EURUSD#, USDJPY#)
   - They might be under categories like:
     - Forex
     - Forex Majors
     - Forex-C
     - Or a special XM Trading category

5. **If "Show All" doesn't work**:
   - Right-click in Market Watch
   - Select "Symbols" 
   - Navigate through the tree to find your symbols
   - Double-click or check the box next to each symbol you want to trade

6. **Verify Symbols are Visible**:
   - The symbols should now appear in the Market Watch list
   - They should show bid/ask prices

## After Adding Symbols:

1. **Restart the API server** to ensure it picks up the changes
2. Run the trading engine again

## Alternative: Use Standard Symbols

If you can't find symbols with `#`, try using standard symbols without the hash:
- EURUSD (instead of EURUSD#)
- USDJPY (instead of USDJPY#)
- GBPUSD (instead of GBPUSD#)

## Test After Fix:

Run this command to verify symbols are now visible:
```bash
python3 diagnose_order_issue.py
```

The symbols should show "Visible: True" and orders should work.