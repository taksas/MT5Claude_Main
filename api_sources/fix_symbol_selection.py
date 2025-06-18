#!/usr/bin/env python3
"""
Add this function to mt5_handler.py to fix symbol selection
"""

def ensure_symbol_visible(self, symbol: str) -> bool:
    """Ensure symbol is visible in Market Watch"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Symbol {symbol} not found")
        return False
        
    if not symbol_info.visible:
        logging.info(f"Symbol {symbol} is not visible, attempting to select it...")
        
        # Try to select the symbol
        if mt5.symbol_select(symbol, True):
            # Wait for the selection to take effect
            import time
            time.sleep(0.5)
            
            # Verify it's now visible
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and symbol_info.visible:
                logging.info(f"✅ Successfully made {symbol} visible")
                return True
            else:
                logging.error(f"❌ Failed to make {symbol} visible - still not visible after selection")
                return False
        else:
            logging.error(f"❌ Failed to select symbol {symbol}")
            return False
    
    return True

# Updated place_order method - add this check before order placement:
"""
# Ensure symbol is visible
if not self.ensure_symbol_visible(symbol):
    raise TradeExecutionError(f"Cannot trade {symbol} - unable to add to Market Watch. Please add it manually in MT5.")
"""