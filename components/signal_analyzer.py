#!/usr/bin/env python3
"""
Signal Analyzer Module
Independent component that continuously analyzes symbols and generates signals
Runs separately from the trading engine
"""

import logging
import time
import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from .trading_config import CONFIG, HIGH_PROFIT_SYMBOLS
from .mt5_api_client import MT5APIClient
from .market_data import MarketData
from .trading_strategy import TradingStrategy
from .symbol_utils import SymbolUtils

logger = logging.getLogger('SignalAnalyzer')

class SignalAnalyzer:
    """Independent signal analyzer that runs continuously"""
    
    def __init__(self, signal_queue: Optional[queue.Queue] = None):
        # Initialize components
        self.api_client = MT5APIClient(CONFIG["API_BASE"])
        self.market_data = MarketData(self.api_client)
        self.symbol_utils = SymbolUtils()
        self.strategy = TradingStrategy()
        
        # Communication queue
        self.signal_queue = signal_queue
        
        # State management
        self.running = False
        self.tradable_symbols = []
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds for real-time updates
        
        logger.info("Signal Analyzer initialized")
    
    def start(self):
        """Start the signal analyzer"""
        logger.info("Starting Signal Analyzer...")
        
        # Check API connection
        if not self.api_client.check_connection():
            logger.error("Cannot connect to API")
            return False
        
        # Discover symbols
        self.tradable_symbols = self._discover_symbols()
        if not self.tradable_symbols:
            logger.error("No tradable symbols found")
            return False
        
        logger.info(f"Analyzing {len(self.tradable_symbols)} symbols")
        
        # Send initial symbol list
        if self.signal_queue:
            self.signal_queue.put({
                'symbol_list': self.tradable_symbols
            })
        
        self.running = True
        
        # Start analysis loop
        self._run_analysis_loop()
        
        return True
    
    def stop(self):
        """Stop the signal analyzer"""
        self.running = False
        logger.info("Signal Analyzer stopped")
    
    def _discover_symbols(self) -> List[str]:
        """Discover tradable symbols"""
        try:
            all_symbols = self.api_client.discover_symbols()
            
            # Only use HIGH_PROFIT_SYMBOLS
            permitted_symbols = [s + "#" for s in HIGH_PROFIT_SYMBOLS.keys()]
            
            final_symbols = []
            for symbol in permitted_symbols:
                if symbol in all_symbols:
                    final_symbols.append(symbol)
                else:
                    logger.warning(f"Symbol {symbol} not available")
            
            return final_symbols
            
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []
    
    def _run_analysis_loop(self):
        """Main analysis loop - runs independently"""
        logger.info("Starting analysis loop...")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Analyze all symbols in parallel
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for symbol in self.tradable_symbols:
                        future = executor.submit(self._analyze_symbol, symbol)
                        futures.append((symbol, future))
                    
                    # Collect results and send batch update
                    batch_update = {}
                    for symbol, future in futures:
                        try:
                            result = future.result(timeout=2)
                            if result:
                                batch_update[symbol] = result
                        except Exception as e:
                            logger.debug(f"Error analyzing {symbol}: {e}")
                            # Send error status
                            batch_update[symbol] = {
                                'type': 'ERROR',
                                'confidence': 0.0,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'ANALYSIS_ERROR',
                                'error': str(e)
                            }
                    
                    # Send batch update
                    if self.signal_queue and batch_update:
                        self.signal_queue.put(batch_update)
                
                # Calculate sleep time to maintain consistent interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(1)
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol and return signal data"""
        try:
            # Check spread
            spread_ok, spread = self.market_data.check_spread(symbol)
            
            # Get market data
            df = self.market_data.get_market_data(symbol)
            if df is None or len(df) < 50:
                return {
                    'type': 'MONITORING',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'INSUFFICIENT_DATA'
                }
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price:
                return {
                    'type': 'MONITORING',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'NO_PRICE'
                }
            
            # Generate signal using strategy
            signal = self.strategy.analyze_ultra(symbol, df, current_price)
            
            # Get latest metrics from strategy
            metrics = self.strategy.last_metrics.get(symbol, {})
            
            # Prepare signal data for visualizer
            if signal:
                return {
                    'type': signal.type.value,
                    'confidence': signal.confidence,
                    'entry': signal.entry,
                    'sl': signal.sl,
                    'tp': signal.tp,
                    'reason': signal.reason,
                    'quality': signal.quality,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'SIGNAL_GENERATED',
                    'metrics': metrics,
                    'spread': spread,
                    'spread_ok': spread_ok
                }
            else:
                # No signal but still send monitoring update
                confidence = metrics.get('confidence', 0.0)
                return {
                    'type': 'MONITORING',
                    'confidence': confidence,
                    'entry': current_price.get('ask', 0) if isinstance(current_price, dict) else 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'ANALYZING',
                    'metrics': metrics,
                    'spread': spread,
                    'spread_ok': spread_ok
                }
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'type': 'ERROR',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'status': 'EXCEPTION',
                'error': str(e)
            }

def run_signal_analyzer(signal_queue: queue.Queue):
    """Run signal analyzer in a separate thread or process"""
    analyzer = SignalAnalyzer(signal_queue)
    try:
        analyzer.start()
    except KeyboardInterrupt:
        logger.info("Signal Analyzer interrupted")
    finally:
        analyzer.stop()

if __name__ == "__main__":
    # Test standalone
    logging.basicConfig(level=logging.INFO)
    test_queue = queue.Queue()
    analyzer = SignalAnalyzer(test_queue)
    
    # Run in thread
    analyzer_thread = threading.Thread(target=analyzer.start)
    analyzer_thread.daemon = True
    analyzer_thread.start()
    
    # Print signals
    try:
        while True:
            if not test_queue.empty():
                signal = test_queue.get()
                print(f"Signal: {signal}")
            time.sleep(1)
    except KeyboardInterrupt:
        analyzer.stop()