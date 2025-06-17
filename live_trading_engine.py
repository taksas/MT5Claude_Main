import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from trading_strategies import StrategyEnsemble, SignalType
from improved_strategies import ImprovedStrategyEnsemble
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LiveTrade:
    ticket: int
    symbol: str
    signal_type: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float = 0.01
    status: str = "OPEN"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None

class LiveTradingEngine:
    def __init__(self, api_base="http://172.28.144.1:8000", use_improved_strategies=True):
        self.api_base = api_base
        # Use improved strategies for better performance
        self.strategy_ensemble = ImprovedStrategyEnsemble() if use_improved_strategies else StrategyEnsemble()
        self.symbols = ["USDJPY#", "EURUSD#", "GBPUSD#"]  # Expanded symbol list for more opportunities
        self.active_trades = {}
        self.max_concurrent_trades = 2  # Allow up to 2 concurrent trades for diversification
        self.trade_history = []
        self.running = False
        self.min_time_between_trades = 60  # Minimum seconds between trades on same symbol
        self.last_trade_time = {}
        
    def check_api_connection(self):
        """Verify API connection and MT5 status"""
        try:
            response = requests.get(f"{self.api_base}/status/mt5")
            if response.status_code == 200:
                status = response.json()
                if status.get('connected') and status.get('trade_allowed'):
                    logger.info("✅ MT5 API connected and trading allowed")
                    return True
                else:
                    logger.error("❌ MT5 not ready for trading")
                    return False
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
        
    def get_account_info(self):
        """Get current account information"""
        try:
            response = requests.get(f"{self.api_base}/account/")
            if response.status_code == 200:
                account = response.json()
                logger.info(f"Account Balance: {account['balance']} {account['currency']}")
                logger.info(f"Free Margin: {account['margin_free']} {account['currency']}")
                return account
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
        return None
        
    def get_market_data(self, symbol, timeframe="M5", count=100):
        """Get current market data for analysis"""
        url = f"{self.api_base}/market/history"
        data = {"symbol": symbol, "timeframe": timeframe, "count": count}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
        return None
        
    def place_order(self, signal, symbol):
        """Place a market order based on trading signal"""
        if signal.signal == SignalType.BUY:
            order_type = 0  # BUY
        else:
            order_type = 1  # SELL
            
        order_data = {
            "action": 1,  # DEAL (market order)
            "symbol": symbol,
            "volume": 0.01,  # Fixed lot size
            "type": order_type,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "comment": f"AI Trade - {signal.reason[:50]}"
        }
        
        try:
            response = requests.post(f"{self.api_base}/trading/orders", json=order_data)
            if response.status_code == 201:
                result = response.json()
                logger.info(f"✅ Order placed successfully: {result}")
                
                # Create trade record
                trade = LiveTrade(
                    ticket=result.get('order', 0),
                    symbol=symbol,
                    signal_type=signal.signal.value,
                    entry_time=datetime.now().isoformat(),
                    entry_price=result.get('price', signal.entry_price),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                self.active_trades[trade.ticket] = trade
                logger.info(f"📊 Trade opened: {symbol} {signal.signal.value} at {trade.entry_price}")
                return trade
            else:
                logger.error(f"❌ Order failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
        return None
        
    def check_open_positions(self):
        """Check and update status of open positions"""
        try:
            response = requests.get(f"{self.api_base}/trading/positions")
            if response.status_code == 200:
                positions = response.json()
                current_tickets = {pos['ticket'] for pos in positions}
                
                # Check for closed trades
                for ticket, trade in list(self.active_trades.items()):
                    if ticket not in current_tickets:
                        # Trade was closed
                        trade.status = "CLOSED"
                        trade.exit_time = datetime.now().isoformat()
                        self.trade_history.append(trade)
                        del self.active_trades[ticket]
                        logger.info(f"🔄 Trade {ticket} was closed")
                
                # Update profit for open trades
                for position in positions:
                    ticket = position['ticket']
                    if ticket in self.active_trades:
                        self.active_trades[ticket].profit = position.get('profit', 0)
                        
                logger.info(f"📈 Active trades: {len(self.active_trades)}")
                return positions
        except Exception as e:
            logger.error(f"Failed to check positions: {e}")
        return []
        
    def analyze_and_trade(self, symbol):
        """Analyze market and place trades if signal is strong"""
        if len(self.active_trades) >= self.max_concurrent_trades:
            logger.info("Max concurrent trades reached, skipping new analysis")
            return
        
        # Check if we traded this symbol recently
        if symbol in self.last_trade_time:
            time_since_last_trade = time.time() - self.last_trade_time[symbol]
            if time_since_last_trade < self.min_time_between_trades:
                logger.debug(f"Too soon to trade {symbol} again, waiting {self.min_time_between_trades - time_since_last_trade:.0f}s")
                return
        
        # Check if we already have an open position in this symbol
        symbol_positions = [t for t in self.active_trades.values() if t.symbol == symbol]
        if symbol_positions:
            logger.debug(f"Already have an open position in {symbol}, skipping")
            return
            
        # Get market data
        market_data = self.get_market_data(symbol, "M5", 100)
        if not market_data:
            return
        
        # Also get M1 data for better entry timing
        market_data_m1 = self.get_market_data(symbol, "M1", 20)
            
        try:
            # Get trading signal
            signal = self.strategy_ensemble.get_ensemble_signal(market_data)
            
            if signal and signal.confidence >= 0.72:  # Slightly higher threshold for improved strategies
                # Confirm with M1 timeframe
                if market_data_m1:
                    m1_signal = self.strategy_ensemble.get_ensemble_signal(market_data_m1)
                    if m1_signal and m1_signal.signal != signal.signal:
                        logger.info(f"Signal conflict between M5 and M1 for {symbol}, skipping")
                        return
                
                logger.info(f"🎯 Strong signal detected: {symbol} {signal.signal.value} "
                           f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
                
                # Place the trade
                trade = self.place_order(signal, symbol)
                if trade:
                    self.last_trade_time[symbol] = time.time()
                    time.sleep(2)  # Brief pause between operations
                    
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
            
    def run_trading_session(self, duration_minutes=300):
        """Run a live trading session for specified duration"""
        if not self.check_api_connection():
            logger.error("Cannot start trading - API connection failed")
            return
            
        account = self.get_account_info()
        if not account:
            logger.error("Cannot start trading - account info unavailable")
            return
            
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        logger.info(f"🚀 Starting live trading session for {duration_minutes} minutes")
        logger.info(f"Trading symbols: {self.symbols}")
        
        try:
            while self.running and datetime.now() < end_time:
                # Check existing positions
                self.check_open_positions()
                
                # Analyze each symbol
                for symbol in self.symbols:
                    self.analyze_and_trade(symbol)
                    time.sleep(1)  # Small delay between symbols
                
                # Wait before next analysis cycle
                time.sleep(15)  # 15-second intervals for more frequent checks
                
                # Log status every 5 minutes
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if int(elapsed) % 5 == 0:
                    logger.info(f"⏰ Session running: {elapsed:.1f}/{duration_minutes} minutes")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Trading session interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Trading session error: {e}")
            self.running = False
            
        # Final status check
        final_positions = self.check_open_positions()
        
        logger.info("📊 Trading session completed")
        logger.info(f"Active trades: {len(self.active_trades)}")
        logger.info(f"Completed trades: {len(self.trade_history)}")
        
        # Summary
        self.print_session_summary()
        
    def print_session_summary(self):
        """Print trading session summary"""
        print("\n" + "="*60)
        print("LIVE TRADING SESSION SUMMARY")
        print("="*60)
        
        total_trades = len(self.trade_history) + len(self.active_trades)
        completed_trades = len(self.trade_history)
        
        if completed_trades > 0:
            profits = [trade.profit for trade in self.trade_history if trade.profit is not None]
            total_profit = sum(profits) if profits else 0
            winning_trades = len([p for p in profits if p > 0])
            win_rate = (winning_trades / completed_trades) * 100
            
            print(f"Total Trades: {total_trades}")
            print(f"Completed Trades: {completed_trades}")
            print(f"Active Trades: {len(self.active_trades)}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total Profit: {total_profit:.2f} JPY")
        else:
            print("No completed trades in this session")
            
        print("="*60)
        
    def stop_trading(self):
        """Gracefully stop trading"""
        self.running = False
        logger.info("🛑 Trading engine stopped")

def main():
    """Main execution function"""
    engine = LiveTradingEngine()
    
    logger.info("🎯 Starting automated forex trading operation")
    logger.info("Strategy: USDJPY# short-term momentum + technical indicators")
    logger.info("Risk Management: 0.01 lot size, SL/TP on every trade")
    
    try:
        # Run a 30-minute trading session
        engine.run_trading_session(duration_minutes=300)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    logger.info("🏁 Trading operation completed")

if __name__ == "__main__":
    main()