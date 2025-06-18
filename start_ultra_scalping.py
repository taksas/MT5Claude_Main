#!/usr/bin/env python3
"""
Quick start script for ultra-short-term scalping
Automatically starts trading with optimal settings
"""

import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scalping_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("ULTRA-SHORT-TERM FOREX SCALPING SYSTEM")
    logger.info("="*70)
    logger.info("Target: 1-10 minute trades with high win rate")
    logger.info("Starting automated trading...")
    
    try:
        # Import and run the ultra-scalping engine
        from ultra_scalping_engine import UltraScalpingEngine
        
        engine = UltraScalpingEngine()
        
        # Run for 5 hours (300 minutes)
        engine.run_scalping_session(duration_minutes=300)
        
    except ImportError:
        # Fallback to enhanced live trading engine with scalping mode
        logger.info("Using enhanced live trading engine in scalping mode...")
        from live_trading_engine import LiveTradingEngine
        
        engine = LiveTradingEngine(use_scalping=True)
        engine.run_trading_session(duration_minutes=300)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Trading stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    
    logger.info("="*70)
    logger.info("Trading session completed. Check logs for details.")
    logger.info("="*70)

if __name__ == "__main__":
    main()