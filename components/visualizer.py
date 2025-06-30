#!/usr/bin/env python3
"""
Trading Visualizer - Real-time monitoring dashboard
Shows account balance, profit, positions, and strategy confidence
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import queue
from dataclasses import dataclass, field
import logging
import collections
from enum import Enum

# 'Visualizer'という名前でロガーを取得するのは、ログメッセージのソースを識別するために引き続き有用です。
logger = logging.getLogger('Visualizer')

# Configuration
CONFIG = {
    "API_BASE": "http://172.28.144.1:8000",
    "REFRESH_RATE": 1,  # seconds
    "ACCOUNT_CURRENCY": "JPY",
    "LOG_MAX_LINES": 8, # ログ表示領域の最大行数
    "LOG_LEVEL_FILTER": ["ERROR", "WARNING", "INFO"] # 表示するログレベル
}

@dataclass
class DisplayData:
    """Data structure for display"""
    balance: float = 0
    equity: float = 0
    profit: float = 0
    positions: List[Dict] = None
    strategy_signals: Dict[str, Dict] = None
    last_update: datetime = None

class SignalDirection(Enum):
    """Signal direction enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class CurrencyPairData:
    """Data structure for currency pair analysis"""
    symbol: str
    is_monitoring: bool = False
    confidence: float = 0.0
    direction: SignalDirection = SignalDirection.NEUTRAL
    last_signal_time: datetime = field(default_factory=datetime.now)
    signal_count: int = 0
    avg_confidence: float = 0.0
    last_price: float = 0.0
    analysis_status: str = "IDLE"  # IDLE, ANALYZING, TRADING

class PairMonitoringAgent:
    """Agent responsible for tracking currency pair monitoring status"""
    
    def __init__(self):
        self.monitored_pairs: Dict[str, CurrencyPairData] = {}
        self.active_symbols: List[str] = []
        
    def update_active_symbols(self, symbols: List[str]):
        """Update list of actively monitored symbols"""
        self.active_symbols = symbols
        
        # Add new symbols
        for symbol in symbols:
            if symbol not in self.monitored_pairs:
                self.monitored_pairs[symbol] = CurrencyPairData(
                    symbol=symbol,
                    is_monitoring=True,
                    analysis_status="ANALYZING"
                )
        
        # Mark symbols as not monitoring if they're no longer active
        for symbol in self.monitored_pairs:
            if symbol in symbols:
                self.monitored_pairs[symbol].is_monitoring = True
                if self.monitored_pairs[symbol].analysis_status == "IDLE":
                    self.monitored_pairs[symbol].analysis_status = "ANALYZING"
            else:
                self.monitored_pairs[symbol].is_monitoring = False
                self.monitored_pairs[symbol].analysis_status = "IDLE"
    
    def get_monitored_pairs(self) -> Dict[str, CurrencyPairData]:
        """Get all monitored currency pairs"""
        return self.monitored_pairs

class ConfidenceAgent:
    """Agent responsible for confidence analysis and calculation"""
    
    def __init__(self, max_history: int = 10):
        self.confidence_history: Dict[str, collections.deque] = {}
        self.max_history = max_history
        
    def update_confidence(self, symbol: str, confidence: float, timestamp: datetime = None):
        """Update confidence for a symbol"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.confidence_history:
            self.confidence_history[symbol] = collections.deque(maxlen=self.max_history)
        
        self.confidence_history[symbol].append((confidence, timestamp))
    
    def get_current_confidence(self, symbol: str) -> float:
        """Get current confidence for a symbol"""
        if symbol not in self.confidence_history or not self.confidence_history[symbol]:
            return 0.0
        return self.confidence_history[symbol][-1][0]
    
    def get_average_confidence(self, symbol: str) -> float:
        """Get average confidence over history"""
        if symbol not in self.confidence_history or not self.confidence_history[symbol]:
            return 0.0
        
        confidences = [conf for conf, _ in self.confidence_history[symbol]]
        return sum(confidences) / len(confidences)
    
    def get_confidence_trend(self, symbol: str) -> str:
        """Get confidence trend (RISING, FALLING, STABLE)"""
        if symbol not in self.confidence_history or len(self.confidence_history[symbol]) < 2:
            return "STABLE"
        
        recent = list(self.confidence_history[symbol])[-2:]
        if recent[1][0] > recent[0][0] + 0.05:
            return "RISING"
        elif recent[1][0] < recent[0][0] - 0.05:
            return "FALLING"
        else:
            return "STABLE"

class DirectionalAgent:
    """Agent responsible for buy/sell recommendation analysis"""
    
    def __init__(self, max_signals: int = 5):
        self.signal_history: Dict[str, collections.deque] = {}
        self.max_signals = max_signals
        
    def update_signal(self, symbol: str, direction: SignalDirection, confidence: float, timestamp: datetime = None):
        """Update signal for a symbol"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.signal_history:
            self.signal_history[symbol] = collections.deque(maxlen=self.max_signals)
        
        self.signal_history[symbol].append((direction, confidence, timestamp))
    
    def get_current_direction(self, symbol: str) -> SignalDirection:
        """Get current directional bias"""
        if symbol not in self.signal_history or not self.signal_history[symbol]:
            return SignalDirection.NEUTRAL
        return self.signal_history[symbol][-1][0]
    
    def get_directional_strength(self, symbol: str) -> float:
        """Get strength of directional bias (0-1)"""
        if symbol not in self.signal_history or not self.signal_history[symbol]:
            return 0.0
        
        recent_signals = list(self.signal_history[symbol])
        if not recent_signals:
            return 0.0
        
        # Calculate consensus strength
        buy_count = sum(1 for sig, _, _ in recent_signals if sig == SignalDirection.BUY)
        sell_count = sum(1 for sig, _, _ in recent_signals if sig == SignalDirection.SELL)
        neutral_count = sum(1 for sig, _, _ in recent_signals if sig == SignalDirection.NEUTRAL)
        
        total = len(recent_signals)
        if total == 0:
            return 0.0
        
        max_consensus = max(buy_count, sell_count, neutral_count)
        return max_consensus / total
    
    def get_signal_consistency(self, symbol: str) -> str:
        """Get signal consistency (CONSISTENT, MIXED, VOLATILE)"""
        if symbol not in self.signal_history or len(self.signal_history[symbol]) < 2:
            return "MIXED"
        
        recent_signals = list(self.signal_history[symbol])
        directions = [sig for sig, _, _ in recent_signals]
        
        unique_directions = len(set(directions))
        if unique_directions == 1:
            return "CONSISTENT"
        elif unique_directions == 2:
            return "MIXED"
        else:
            return "VOLATILE"

class VisualizationAgent:
    """Agent responsible for formatting and managing visual display"""
    
    def __init__(self):
        self.display_mode = "COMPACT"  # COMPACT, DETAILED
        self.max_pairs_display = 8
        
    def format_pair_display(self, pair_data: CurrencyPairData, confidence: float, 
                          direction: SignalDirection, strength: float) -> Dict[str, Any]:
        """Format currency pair data for display"""
        
        # Determine display color based on direction and confidence
        if direction == SignalDirection.BUY:
            color = "🟢" if confidence > 0.6 else "🟡"
            direction_text = "BUY"
        elif direction == SignalDirection.SELL:
            color = "🔴" if confidence > 0.6 else "🟠" 
            direction_text = "SELL"
        else:
            color = "⚪"
            direction_text = "WAIT"
        
        # Format confidence bar
        conf_bar = self._create_confidence_bar(confidence)
        
        # Format symbol name (remove # suffix)
        clean_symbol = pair_data.symbol.replace('#', '')
        
        return {
            'symbol': clean_symbol,
            'color': color,
            'direction': direction_text,
            'confidence': confidence,
            'confidence_bar': conf_bar,
            'strength': strength,
            'status': pair_data.analysis_status,
            'monitoring': pair_data.is_monitoring,
            'last_update': pair_data.last_signal_time
        }
    
    def _create_confidence_bar(self, confidence: float, width: int = 10) -> str:
        """Create visual confidence bar"""
        filled = int(confidence * width)
        empty = width - filled
        return "█" * filled + "░" * empty
    
    def format_compact_display(self, formatted_pairs: List[Dict[str, Any]]) -> str:
        """Format pairs for compact display"""
        if not formatted_pairs:
            return "No currency pairs being monitored"
        
        lines = []
        lines.append("📊 CURRENCY PAIR ANALYSIS:")
        
        # Sort by confidence (highest first)
        sorted_pairs = sorted(formatted_pairs, key=lambda x: x['confidence'], reverse=True)
        
        for pair in sorted_pairs[:self.max_pairs_display]:
            symbol = pair['symbol']
            color = pair['color']
            direction = pair['direction']
            confidence = pair['confidence']
            conf_bar = pair['confidence_bar']
            
            line = f"{color} {symbol}: {direction} {confidence:.1%} {conf_bar}"
            
            # Add error indicator if present
            if 'error' in pair:
                line += f" ⚠️"
            elif pair.get('status') == 'ERROR':
                line += f" ❌"
            
            lines.append(line)
        
        if len(sorted_pairs) > self.max_pairs_display:
            lines.append(f"... (+{len(sorted_pairs) - self.max_pairs_display} more pairs)")
        
        return "\n".join(lines)

# ログをdequeに保存するためのカスタムハンドラ
class FilteredDequeHandler(logging.Handler):
    """A logging handler that stores filtered messages in a deque."""
    def __init__(self, deque_instance: collections.deque, level_filter: List[str]):
        super().__init__()
        self.deque = deque_instance
        self.level_filter = set(level_filter)
        self.last_log_time = time.time()
        self.log_count = 0
        self.max_logs_per_second = 10

    def emit(self, record: logging.LogRecord):
        """Format and append the log record to the deque with filtering."""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_log_time < 1:
            self.log_count += 1
            if self.log_count > self.max_logs_per_second:
                return
        else:
            self.log_count = 0
            self.last_log_time = current_time
            
        # Level filtering
        if record.levelname in self.level_filter:
            # Skip repetitive API connection errors
            log_entry = self.format(record)
            if "API connection failed" in log_entry and len(self.deque) > 0:
                if "API connection failed" in list(self.deque)[-1]:
                    return
            self.deque.append(log_entry)


class TradingVisualizer:
    def __init__(self, data_queue: Optional[queue.Queue] = None):
        self.api_base = CONFIG["API_BASE"]
        self.data_queue = data_queue
        self.running = False
        self.display_data = DisplayData()

        self.log_messages = collections.deque(maxlen=CONFIG["LOG_MAX_LINES"] * 2)  # Buffer extra for filtering
        self._configure_logging()

        self.display_data.strategy_signals = {}
        self.cleanup_counter = 0  # For periodic cleanup

        # Initialize multi-agent system for currency pair analysis
        self.pair_monitor = PairMonitoringAgent()
        self.confidence_agent = ConfidenceAgent()
        self.directional_agent = DirectionalAgent()
        self.visualization_agent = VisualizationAgent()
        
        # Track active symbols from engine
        self.last_symbol_update = datetime.now()
        self.symbol_update_interval = timedelta(minutes=5)  # Only update symbols every 5 minutes
        
        # Initialize with HIGH_PROFIT_SYMBOLS as fallback
        self._initialize_fallback_symbols()

        if self.data_queue:
            logger.info("Visualizer: Signal queue connected successfully")
            logger.info("Visualizer: Multi-agent currency pair analysis system initialized")
        else:
            logger.warning("Visualizer: No signal queue provided - signals will not be displayed")

    def _initialize_fallback_symbols(self):
        """Initialize with HIGH_PROFIT_SYMBOLS as fallback"""
        try:
            from .trading_config import HIGH_PROFIT_SYMBOLS
            
            # Convert HIGH_PROFIT_SYMBOLS to symbol format with #
            fallback_symbols = [symbol + "#" for symbol in HIGH_PROFIT_SYMBOLS.keys()]
            
            if fallback_symbols:
                self.pair_monitor.update_active_symbols(fallback_symbols)
                logger.info(f"Visualizer: Initialized with {len(fallback_symbols)} fallback symbols")
            
        except Exception as e:
            logger.error(f"Error initializing fallback symbols: {e}")

    def _configure_logging(self):
        """Configure selective logging for visualizer display."""
        # Create filtered handler for display only
        deque_handler = FilteredDequeHandler(self.log_messages, CONFIG["LOG_LEVEL_FILTER"])
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        deque_handler.setFormatter(formatter)
        
        # Add handler to specific loggers only (avoid root logger and Visualizer to prevent loops)
        important_loggers = ['UltraTradingEngine', 'OrderManagement', 'MT5APIClient']
        for logger_name in important_loggers:
            target_logger = logging.getLogger(logger_name)
            # Check if our handler is already added to avoid duplicates
            handler_exists = any(isinstance(h, FilteredDequeHandler) for h in target_logger.handlers)
            if not handler_exists:
                target_logger.addHandler(deque_handler)
            target_logger.propagate = False  # Prevent duplicate logs

    def clear_screen(self):
        """Clear terminal screen"""
        # Use ANSI escape codes for more reliable clearing
        print('\033[2J\033[H', end='', flush=True)

    def format_currency(self, amount: float) -> str:
        """Format currency for display"""
        return f"¥{amount:,.0f}"

    def get_account_info(self) -> Optional[Dict]:
        """Fetch account information"""
        try:
            resp = requests.get(f"{self.api_base}/account/", timeout=2)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection failed for account info: {e}")
        return None

    def get_positions(self) -> List[Dict]:
        """Fetch current positions"""
        try:
            resp = requests.get(f"{self.api_base}/trading/positions", timeout=2)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection failed for positions: {e}")
        return []

    def update_display_data(self):
        """Update all display data"""
        account = self.get_account_info()
        if account:
            self.display_data.balance = account.get('balance', 0)
            self.display_data.equity = account.get('equity', 0)
            self.display_data.profit = account.get('profit', 0)

        self.display_data.positions = self.get_positions()

        if self.data_queue:
            # Process all available queue items for real-time updates
            processed_count = 0
            max_process_per_cycle = 20  # Increased to handle more updates
            
            while not self.data_queue.empty() and processed_count < max_process_per_cycle:
                try:
                    signals = self.data_queue.get_nowait()
                    if '_daily_pnl_update' not in signals:
                        # Check if this is symbol list update
                        if 'symbol_list' in signals:
                            symbol_list = signals['symbol_list']
                            self.pair_monitor.update_active_symbols(symbol_list)
                            self.last_symbol_update = datetime.now()
                            logger.info(f"Updated active symbols: {len(symbol_list)} symbols")
                            continue
                        
                        # Process batch updates (multiple symbols at once)
                        for symbol, data in signals.items():
                            if isinstance(data, dict):
                                # Update agents with signal data
                                signal_type = data.get('type', 'NONE')
                                confidence = data.get('confidence', 0)
                                timestamp_str = data.get('timestamp', datetime.now().isoformat())
                                status = data.get('status', 'UNKNOWN')
                                error = data.get('error', None)
                                
                                # Parse timestamp
                                try:
                                    if isinstance(timestamp_str, str):
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    else:
                                        timestamp = datetime.now()
                                except:
                                    timestamp = datetime.now()
                                
                                # Always update agents even for MONITORING status
                                self.confidence_agent.update_confidence(symbol, confidence, timestamp)
                                
                                # Update directional agent
                                if signal_type == 'BUY':
                                    direction = SignalDirection.BUY
                                elif signal_type == 'SELL':
                                    direction = SignalDirection.SELL
                                else:
                                    direction = SignalDirection.NEUTRAL
                                
                                self.directional_agent.update_signal(symbol, direction, confidence, timestamp)
                                
                                # Update monitoring status for the symbol
                                if symbol in self.pair_monitor.monitored_pairs:
                                    pair_data = self.pair_monitor.monitored_pairs[symbol]
                                    pair_data.last_signal_time = timestamp
                                    pair_data.confidence = confidence
                                    pair_data.direction = direction
                                    
                                    # Update analysis status based on signal
                                    if status == 'ACTIVE' or signal_type == 'MONITORING':
                                        pair_data.analysis_status = 'ANALYZING'
                                    elif status in ['EXECUTION_FAILED', 'ORDER_FAILED', 'ANALYSIS_ERROR']:
                                        pair_data.analysis_status = 'ERROR'
                                    elif signal_type in ['BUY', 'SELL']:
                                        pair_data.analysis_status = 'TRADING'
                                
                                # Keep legacy signal storage for compatibility
                                filtered_data = {
                                    'type': signal_type,
                                    'confidence': confidence,
                                    'timestamp': timestamp_str,
                                    'status': status
                                }
                                
                                # Add error indicator if present
                                if error:
                                    filtered_data['error'] = error  # Full error message
                                    # Log errors for debugging
                                    if status in ['EXECUTION_FAILED', 'ORDER_FAILED']:
                                        logger.warning(f"Order failed for {symbol}: {error}")
                                
                                self.display_data.strategy_signals[symbol] = filtered_data
                    processed_count += 1
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing signal queue: {e}")
                    break
            
            # Clean old strategy signals to prevent memory buildup
            if len(self.display_data.strategy_signals) > 10:
                # Keep only the 10 most recent signals
                sorted_signals = sorted(
                    self.display_data.strategy_signals.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                self.display_data.strategy_signals = dict(sorted_signals[:10])

        # Check if we need to reinitialize (only if no pairs are monitored)
        if not self.pair_monitor.monitored_pairs:
            # Re-initialize if no pairs are being monitored
            self._initialize_fallback_symbols()
            self.last_symbol_update = datetime.now()

        self.display_data.last_update = datetime.now()
        
        # Periodic cleanup every 30 cycles to prevent memory leaks
        self.cleanup_counter += 1
        if self.cleanup_counter >= 30:
            self._periodic_cleanup()
            self.cleanup_counter = 0
    
    def _periodic_cleanup(self):
        """Perform periodic memory cleanup"""
        # Clear excessive queue data if any
        if self.data_queue and self.data_queue.qsize() > 50:
            # Drain excessive items from queue
            drained = 0
            while not self.data_queue.empty() and drained < 30:
                try:
                    self.data_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            logger.warning(f"Drained {drained} excessive items from signal queue")
        
        # Clear old strategy signals
        if len(self.display_data.strategy_signals) > 5:
            self.display_data.strategy_signals.clear()
            logger.info("Cleared old strategy signals for memory management")

    def display_header(self):
        # """Display header information"""
        # print("                ULTRA TRADING MONITOR")
        None
        
    def display_account(self):
        """Display account information in compact format"""
        if self.display_data.balance > 0:
            print(f"Balance: {self.format_currency(self.display_data.balance)} | "
                  f"Equity: {self.format_currency(self.display_data.equity)} | "
                  f"Floating: {self.format_currency(self.display_data.profit)} "
                  f"({self.display_data.profit/self.display_data.balance*100:+.1f}%)")
        else:
            print("Balance: N/A | Equity: N/A | Floating: N/A")


    def display_positions(self):
        """Display open positions in compact format"""
        if self.display_data.positions:
            print(f"📈 POSITIONS ({len(self.display_data.positions)}): ", end="")
            total_profit = sum(pos.get('profit', 0) for pos in self.display_data.positions)

            for i, pos in enumerate(self.display_data.positions[:4]):
                symbol = pos['symbol'].replace('#', '')
                type_icon = "🟢" if pos['type'] == 0 else "🔴"
                profit = pos['profit']
                profit_str = f"+{profit:.0f}" if profit > 0 else f"{profit:.0f}"
                print(f"{type_icon}{symbol}:{profit_str}¥", end=" ")

            if len(self.display_data.positions) > 4:
                print(f"(+{len(self.display_data.positions)-4} more)", end=" ")

            print(f"| Total: {self.format_currency(total_profit)}")

    def display_strategy_signals(self):
        """Display currency pair analysis using multi-agent system"""
        try:
            # Get monitored pairs from the monitoring agent
            monitored_pairs = self.pair_monitor.get_monitored_pairs()
            
            if not monitored_pairs:
                print("📊 No currency pairs being monitored")
                return
            
            # Format pairs for display
            formatted_pairs = []
            for symbol, pair_data in monitored_pairs.items():
                if pair_data.is_monitoring:
                    # Get data from agents
                    confidence = self.confidence_agent.get_current_confidence(symbol)
                    direction = self.directional_agent.get_current_direction(symbol)
                    strength = self.directional_agent.get_directional_strength(symbol)
                    
                    # Update pair data with latest info
                    pair_data.confidence = confidence
                    pair_data.direction = direction
                    
                    # Check for any errors in signal data
                    signal_data = self.display_data.strategy_signals.get(symbol, {})
                    if signal_data.get('status') in ['EXECUTION_FAILED', 'ORDER_FAILED']:
                        pair_data.analysis_status = 'ERROR'
                    elif signal_data.get('status') == 'ANALYSIS_ERROR':
                        pair_data.analysis_status = 'ANALYSIS_ERROR'
                    
                    # Format for display
                    formatted_pair = self.visualization_agent.format_pair_display(
                        pair_data, confidence, direction, strength
                    )
                    
                    # Add error info if present
                    if 'error' in signal_data:
                        formatted_pair['error'] = signal_data['error']
                    
                    formatted_pairs.append(formatted_pair)
            
            # Display formatted pairs
            if formatted_pairs:
                display_text = self.visualization_agent.format_compact_display(formatted_pairs)
                print(display_text)
            else:
                print("📊 No active currency pair signals")
                
        except Exception as e:
            logger.error(f"Error displaying strategy signals: {e}")
            print("📊 Error displaying currency pair analysis")

    def display_logs(self):
        """Display the latest log messages efficiently."""
        if not self.log_messages:
            print("No system logs.")
        else:
            # Convert to list once and display
            recent_logs = list(self.log_messages)
            for msg in recent_logs[-CONFIG["LOG_MAX_LINES"]:]:  # Show only most recent
                # Display full message without truncation
                print(msg)


    def display_footer(self):
        """Display footer information"""
        print("\n" + "=" * 60)
        update_time = self.display_data.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.display_data.last_update else "N/A"
        print(f"Last Update: {update_time} | Press Ctrl+C to exit")

    def run_display_loop(self):
        """Main display loop"""
        while self.running:
            try:
                self.update_display_data()
                self.clear_screen()
                self.display_header()
                self.display_account()
                self.display_positions()
                self.display_strategy_signals()
                self.display_logs()
                self.display_footer()
                # Clear old logs periodically to prevent memory buildup
                if len(self.log_messages) > CONFIG["LOG_MAX_LINES"] * 1.5:
                    # Keep only recent logs
                    recent_logs = list(self.log_messages)[-CONFIG["LOG_MAX_LINES"]:]
                    self.log_messages.clear()
                    self.log_messages.extend(recent_logs)
                    
                time.sleep(CONFIG["REFRESH_RATE"])

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"An error occurred in the display loop: {e}", exc_info=False)
                print(f"Visualizer Error: {e}")
                time.sleep(1)

    def start(self):
        """Start the visualizer"""
        self.running = True
        logger.info("Starting Trading Visualizer")

        if not self.get_account_info():
            logger.error("Could not connect to API. Please check API server and network.")
            self.running = False
            self.clear_screen()
            self.display_header()
            self.display_logs()
            print("\n" + "="*50)
            print("Visualizer stopped due to connection error.")
            return

        self.run_display_loop()

    def stop(self):
        """Stop the visualizer"""
        if self.running:
            self.running = False
            logger.info("Stopping Trading Visualizer")
            print("\nTrading Visualizer stopped.")

def main():
    """Run standalone visualizer"""
    visualizer = TradingVisualizer()
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nShutting down by user request...")
    finally:
        visualizer.stop()

if __name__ == "__main__":
    main()