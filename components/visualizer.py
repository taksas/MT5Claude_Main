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
    "LOG_MAX_LINES": 20 # ログ表示領域の最大行数
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
class DequeHandler(logging.Handler):
    """A logging handler that stores messages in a deque."""
    def __init__(self, deque_instance: collections.deque):
        super().__init__()
        self.deque = deque_instance

    def emit(self, record: logging.LogRecord):
        """Format and append the log record to the deque."""
        log_entry = self.format(record)
        self.deque.append(log_entry)


class TradingVisualizer:
    def __init__(self, data_queue: Optional[queue.Queue] = None):
        self.api_base = CONFIG["API_BASE"]
        self.data_queue = data_queue
        self.running = False
        self.display_data = DisplayData()

        self.log_messages = collections.deque(maxlen=CONFIG["LOG_MAX_LINES"])
        self._configure_logging()

        self.display_data.strategy_signals = {}

        if self.data_queue:
            logger.info("Visualizer: Signal queue connected successfully")
        else:
            logger.warning("Visualizer: No signal queue provided - signals will not be displayed")

    def _configure_logging(self):
        """Configure the root logger to send all logs to the DequeHandler."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)

        deque_handler = DequeHandler(self.log_messages)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%H:%M:%S')
        deque_handler.setFormatter(formatter)
        root_logger.addHandler(deque_handler)

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
            while not self.data_queue.empty():
                try:
                    signals = self.data_queue.get_nowait()
                    if '_daily_pnl_update' not in signals:
                        for symbol, data in signals.items():
                            self.display_data.strategy_signals[symbol] = data
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing signal queue: {e}")
                    break

        self.display_data.last_update = datetime.now()

    def display_header(self):
        """Display header information"""
        print("                ULTRA TRADING MONITOR")
    def display_account(self):
        """Display account information in compact format"""
        print("\n💰 ACCOUNT: ", end="")
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
            print(f"\n📈 POSITIONS ({len(self.display_data.positions)}): ", end="")
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
        """Display strategy confidence levels in a compact, single-line format."""
        print("\n🎯 STRATEGY SIGNALS (COMPACT VIEW)")
        print("-" * 60)

        if not self.display_data.strategy_signals:
            print("Waiting for signals...")
            return

        # Sort symbols by confidence for prioritized display
        sorted_symbols = sorted(
            self.display_data.strategy_signals.items(),
            key=lambda item: item[1].get('confidence', 0),
            reverse=True
        )

        for symbol, data in sorted_symbols:
            try:
                signal_type = data.get('type', 'NONE')
                confidence = data.get('confidence', 0)
                quality = data.get('quality', 0)
                strategies = data.get('strategies', {})
                reason = data.get('reasons', [""])[0]

                if signal_type == "BUY":
                    signal_icon, signal_color = "🟢", "\033[92m"
                elif signal_type == "SELL":
                    signal_icon, signal_color = "🔴", "\033[91m"
                else:
                    signal_icon, signal_color = "⚪", "\033[90m"

                # --- Compact Strategy Display Logic ---
                strategy_str = ""
                if strategies:
                    # Filter for strategies with a meaningful score
                    contributing_strats = {
                        k: v for k, v in strategies.items()
                        if k not in ['final_score', 'confidence', 'quality'] and abs(v) > 0.1
                    }
                    if contributing_strats:
                        # Sort by absolute score value, descending
                        sorted_strats = sorted(
                            contributing_strats.items(),
                            key=lambda item: abs(item[1]),
                            reverse=True
                        )
                        # Format the top 3 contributing strategies
                        strat_parts = [f"{k[:4]}:{v:.1f}" for k, v in sorted_strats[:3]]
                        strategy_str = f"| Strats: {', '.join(strat_parts)}"
                
                elif "No clear signal" not in reason:
                     strategy_str = f"| {reason}"


                # Print the single, compact line for each symbol
                print(f"{signal_icon} {signal_color}{symbol:<8}\033[0m {signal_color}{signal_type:<4}\033[0m "
                      f"C:{confidence:<5.1%} Q:{quality:<5.1%} {strategy_str}")

            except Exception as e:
                print(f"   ❌ Error displaying {symbol}: {str(e)[:40]}...")

        print("-" * 60)
        total_symbols = len(self.display_data.strategy_signals)
        buy_count = sum(1 for s in self.display_data.strategy_signals.values() if s.get('type') == 'BUY')
        sell_count = sum(1 for s in self.display_data.strategy_signals.values() if s.get('type') == 'SELL')
        high_conf = sum(1 for s in self.display_data.strategy_signals.values() if s.get('confidence', 0) >= 0.7)
        print(f"📊 Summary: {total_symbols} symbols | "
              f"🟢 {buy_count} BUY | 🔴 {sell_count} SELL | "
              f"⭐ {high_conf} High Conf (>70%)")

    def display_logs(self):
        """Display the latest log messages."""
        print("\n" + "--- SYSTEM LOGS " + "-" * 45)
        if not self.log_messages:
            print("No new log messages.")
        else:
            for msg in list(self.log_messages):
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