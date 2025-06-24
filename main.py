#!/usr/bin/env python3
"""
Main Ultra Trading System - All-in-One Launcher
Runs trading engine with optional visualizer
"""

import sys
import time
import logging
import argparse
import signal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Main')

class UltraTradingSystem:
    def __init__(self, mode='engine', visualize=False):
        self.mode = mode
        self.visualize = visualize
        self.engine = None
        self.visualizer = None
        self.shutdown = False
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.shutdown = True
        self.stop()
        
    def start(self):
        """Start the trading system"""
        print("=" * 70)
        print("   ULTRA TRADING SYSTEM - HIGH-PROFIT CONFIGURATION")
        print("=" * 70)
        print(f"\nMode: {self.mode.upper()}")
        print(f"Visualizer: {'ENABLED' if self.visualize else 'DISABLED'}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nHigh-Profit Symbols Active:")
        
        print("\nPress Ctrl+C to shutdown\n")
        print("=" * 70)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            if self.mode == 'engine' or self.mode == 'both':
                # Import and run engine
                logger.info("Starting Ultra Trading Engine...")
                from components.engine_core import UltraTradingEngine
                self.engine = UltraTradingEngine()
                
                if self.visualize and self.mode == 'engine':
                    # Run engine with visualizer in same process
                    logger.info("Starting Visualizer...")
                    from components.visualizer import TradingVisualizer
                    self.visualizer = TradingVisualizer()
                    
                    # Start visualizer in thread
                    import threading
                    viz_thread = threading.Thread(target=self.visualizer.start)
                    viz_thread.daemon = True
                    viz_thread.start()
                
                # Start engine (blocks until shutdown)
                self.engine.start()
                
            elif self.mode == 'visualizer':
                # Run visualizer only
                logger.info("Starting Visualizer Only Mode...")
                from components.visualizer import TradingVisualizer
                self.visualizer = TradingVisualizer()
                self.visualizer.start()
                
                
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping Ultra Trading System...")
        
        if self.engine:
            self.engine.stop()
                
        if self.visualizer:
            self.visualizer.stop()
                
        logger.info("System stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ultra Trading System')
    parser.add_argument('--mode', choices=['engine', 'visualizer', 'both'], 
                       default='engine', help='Run mode')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Enable visualizer with engine')
    
    args = parser.parse_args()
    
    system = UltraTradingSystem(mode=args.mode, visualize=args.visualize)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown initiated...")
    finally:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()