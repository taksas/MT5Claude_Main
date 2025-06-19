#!/usr/bin/env python3
"""
Main coordinator - Runs trading engine and visualizer together
"""

import sys
import time
import logging
import threading
import queue
from multiprocessing import Process, Queue
import signal

from trading_engine import TradingEngine
from visualizer import TradingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Main')

class TradingSystem:
    def __init__(self):
        # Create queue for inter-process communication
        self.signal_queue = Queue()
        
        # Processes
        self.engine_process = None
        self.visualizer_process = None
        
        # Shutdown flag
        self.shutdown = False
        
    def run_engine(self, signal_queue):
        """Run trading engine in separate process"""
        engine = TradingEngine(signal_queue)
        try:
            engine.start()
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop()
            
    def run_visualizer(self, signal_queue):
        """Run visualizer in separate process"""
        visualizer = TradingVisualizer(signal_queue)
        try:
            visualizer.start()
        except KeyboardInterrupt:
            pass
        finally:
            visualizer.stop()
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.shutdown = True
        self.stop()
        
    def start(self):
        """Start the trading system"""
        logger.info("Starting Ultimate Trading System")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start trading engine process
            logger.info("Starting Trading Engine...")
            self.engine_process = Process(
                target=self.run_engine,
                args=(self.signal_queue,),
                name="TradingEngine"
            )
            self.engine_process.start()
            
            # Give engine time to initialize
            time.sleep(2)
            
            # Start visualizer process
            logger.info("Starting Visualizer...")
            self.visualizer_process = Process(
                target=self.run_visualizer,
                args=(self.signal_queue,),
                name="Visualizer"
            )
            self.visualizer_process.start()
            
            # Monitor processes
            while not self.shutdown:
                # Check if processes are alive
                if not self.engine_process.is_alive():
                    logger.error("Trading Engine crashed, restarting...")
                    self.restart_engine()
                    
                if not self.visualizer_process.is_alive():
                    logger.error("Visualizer crashed, restarting...")
                    self.restart_visualizer()
                    
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.stop()
            
    def restart_engine(self):
        """Restart trading engine"""
        if self.engine_process:
            self.engine_process.terminate()
            self.engine_process.join(timeout=5)
            
        self.engine_process = Process(
            target=self.run_engine,
            args=(self.signal_queue,),
            name="TradingEngine"
        )
        self.engine_process.start()
        
    def restart_visualizer(self):
        """Restart visualizer"""
        if self.visualizer_process:
            self.visualizer_process.terminate()
            self.visualizer_process.join(timeout=5)
            
        self.visualizer_process = Process(
            target=self.run_visualizer,
            args=(self.signal_queue,),
            name="Visualizer"
        )
        self.visualizer_process.start()
        
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping Ultimate Trading System...")
        
        # Terminate processes
        if self.engine_process and self.engine_process.is_alive():
            logger.info("Stopping Trading Engine...")
            self.engine_process.terminate()
            self.engine_process.join(timeout=10)
            
        if self.visualizer_process and self.visualizer_process.is_alive():
            logger.info("Stopping Visualizer...")
            self.visualizer_process.terminate()
            self.visualizer_process.join(timeout=10)
            
        logger.info("System stopped")

def main():
    """Main entry point"""
    print("=" * 60)
    print("   ULTIMATE TRADING SYSTEM - Multi-Process Edition")
    print("=" * 60)
    print("\nStarting system components...")
    print("Press Ctrl+C to shutdown\n")
    
    system = TradingSystem()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown initiated...")
    finally:
        system.stop()
        print("\nGoodbye!")

if __name__ == "__main__":
    main()