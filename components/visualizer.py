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
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue
from dataclasses import dataclass
import logging
import collections

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

        if self.data_queue:
            logger.info("Visualizer: Signal queue connected successfully")
        else:
            logger.warning("Visualizer: No signal queue provided - signals will not be displayed")

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
        os.system('cls' if os.name == 'nt' else 'clear')

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
            # Process limited number of queue items to prevent memory buildup
            processed_count = 0
            max_process_per_cycle = 5
            
            while not self.data_queue.empty() and processed_count < max_process_per_cycle:
                try:
                    signals = self.data_queue.get_nowait()
                    if '_daily_pnl_update' not in signals:
                        for symbol, data in signals.items():
                            # Keep only essential data to reduce memory usage
                            if isinstance(data, dict):
                                filtered_data = {
                                    'type': data.get('type', 'NONE'),
                                    'confidence': data.get('confidence', 0),
                                    'timestamp': data.get('timestamp', datetime.now().isoformat())
                                }
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
        None

    def display_logs(self):
        """Display the latest log messages efficiently."""
        if not self.log_messages:
            print("No system logs.")
        else:
            # Convert to list once and display
            recent_logs = list(self.log_messages)
            for msg in recent_logs[-CONFIG["LOG_MAX_LINES"]:]:  # Show only most recent
                # Truncate long messages
                if len(msg) > 80:
                    msg = msg[:77] + "..."
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