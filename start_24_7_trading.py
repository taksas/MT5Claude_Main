#!/usr/bin/env python3
"""
24/7 Continuous Trading Bot Launcher
Runs the trading bot continuously with automatic restarts and monitoring
"""

import subprocess
import time
import logging
from datetime import datetime
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'24_7_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingBotManager:
    def __init__(self):
        self.process = None
        self.running = True
        self.restart_count = 0
        self.start_time = datetime.now()
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)
        
    def start_bot(self):
        """Start the trading bot process"""
        try:
            logger.info(f"Starting trading bot (attempt #{self.restart_count + 1})")
            
            # Start the trading engine in 24/7 mode
            self.process = subprocess.Popen(
                ['python3', 'live_trading_engine.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.restart_count += 1
            
            # Monitor output
            while self.process.poll() is None and self.running:
                line = self.process.stdout.readline()
                if line:
                    print(line.strip())
                    
                    # Check for critical errors
                    if "Fatal error" in line or "API connection failed" in line:
                        logger.warning("Critical error detected, will restart soon...")
                        
            return_code = self.process.wait()
            logger.info(f"Bot process ended with code: {return_code}")
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            
    def run(self):
        """Main loop to keep the bot running 24/7"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("="*60)
        logger.info("24/7 FOREX TRADING BOT MANAGER")
        logger.info("="*60)
        logger.info("Features:")
        logger.info("- Automatic restart on crashes")
        logger.info("- Avoids trading during 3 AM - 8 AM (wide spreads)")
        logger.info("- Continuous operation with monitoring")
        logger.info("- Press Ctrl+C to stop")
        logger.info("="*60)
        
        while self.running:
            try:
                self.start_bot()
                
                if self.running:
                    # Calculate uptime
                    uptime = datetime.now() - self.start_time
                    hours = uptime.total_seconds() / 3600
                    
                    logger.info(f"Bot stopped after {hours:.1f} hours")
                    logger.info(f"Restart count: {self.restart_count}")
                    logger.info("Waiting 60 seconds before restart...")
                    
                    # Wait before restart
                    for i in range(60):
                        if not self.running:
                            break
                        time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if self.running:
                    logger.info("Waiting 5 minutes before restart...")
                    time.sleep(300)
                    
        logger.info("Bot manager stopped")
        logger.info(f"Total uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours")
        logger.info(f"Total restarts: {self.restart_count}")

def main():
    manager = TradingBotManager()
    manager.run()

if __name__ == "__main__":
    main()