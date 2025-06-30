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
        
        print("🔥 HIGH-PROFIT SYMBOLS ACTIVE:")
        print("- Exotic Currencies: USDTRY, USDZAR, USDMXN")
        print("- Cross Pairs: GBPJPY, GBPNZD, EURAUD")
        print("- Metals: XAUUSD, XPDUSD, XPTUSD")
        print("- Indices: US30, RUSSELL2K, MDAX")
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
                
                # Create signal queue for engine-visualizer communication
                signal_queue = None
                if self.visualize and self.mode == 'engine':
                    import queue
                    signal_queue = queue.Queue()
                
                self.engine = UltraTradingEngine(signal_queue=signal_queue)
                
                if self.visualize and self.mode == 'engine':
                    # Run engine with visualizer in same process
                    logger.info("Starting Visualizer...")
                    from components.visualizer import TradingVisualizer
                    self.visualizer = TradingVisualizer(data_queue=signal_queue)
                    
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
                
            elif self.mode == 'both':
                # Run both in separate processes
                from multiprocessing import Process, Queue
                
                signal_queue = Queue()
                
                # Engine process
                def run_engine(queue):
                    from components.engine_core import UltraTradingEngine
                    engine = UltraTradingEngine(signal_queue=queue)
                    engine.start()
                
                # Visualizer process
                def run_visualizer(queue):
                    from components.visualizer import TradingVisualizer
                    viz = TradingVisualizer(signal_queue=queue)
                    viz.start()
                
                engine_proc = Process(target=run_engine, args=(signal_queue,))
                viz_proc = Process(target=run_visualizer, args=(signal_queue,))
                
                engine_proc.start()
                time.sleep(2)  # Let engine initialize
                viz_proc.start()
                
                # Wait for processes
                engine_proc.join()
                viz_proc.join()
                
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
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
                
        if self.visualizer:
            try:
                self.visualizer.stop()
            except Exception as e:
                logger.error(f"Error stopping visualizer: {e}")
                
        logger.info("System stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ultra Trading System')
    parser.add_argument('--mode', choices=['engine', 'visualizer', 'both'], 
                       default='engine', help='Run mode')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Enable visualizer with engine')
    parser.add_argument('--aggressive', '-a', action='store_true',
                       help='Enable aggressive trading mode')
    
    args = parser.parse_args()
    
    # Quick launch shortcuts
    if len(sys.argv) == 1:
        # No arguments - run engine only
        system = UltraTradingSystem(mode='engine', visualize=False)
    else:
        system = UltraTradingSystem(mode=args.mode, visualize=args.visualize)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown initiated...")
    finally:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()