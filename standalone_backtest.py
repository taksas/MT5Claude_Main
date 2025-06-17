#!/usr/bin/env python3
"""
Standalone Backtesting System for Forex Trading Strategies
Generates synthetic market data and tests trading strategies
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List
from trading_strategies import StrategyEnsemble, SignalType
from backtesting import Backtester
from risk_manager import RiskManager

def generate_realistic_forex_data(symbol: str, bars: int = 1000, timeframe_minutes: int = 5) -> List[Dict]:
    """Generate realistic forex OHLC data with proper volatility patterns"""
    
    # Base prices for different currency pairs
    base_prices = {
        'EURUSD': 1.1000,
        'GBPUSD': 1.3000, 
        'USDJPY': 150.00,
        'USDCAD': 1.3500,
        'AUDUSD': 0.6700,
        'NZDUSD': 0.6200,
        'USDCHF': 0.9000
    }
    
    # Determine base price
    base_price = base_prices.get(symbol.replace('#', ''), 1.1000)
    
    # Generate time series
    start_time = datetime.utcnow() - timedelta(minutes=bars * timeframe_minutes)
    
    data = []
    current_price = base_price
    
    for i in range(bars):
        timestamp = start_time + timedelta(minutes=i * timeframe_minutes)
        
        # Add realistic price movement with volatility clustering
        volatility = 0.001 + 0.0005 * np.sin(i / 50)  # Variable volatility
        price_change = np.random.normal(0, volatility)
        
        # Add slight trend component
        trend = 0.00005 * np.sin(i / 100)
        current_price += price_change + trend
        
        # Generate OHLC from current price
        spread = abs(np.random.normal(0, volatility * 0.5))
        
        open_price = current_price
        high_price = current_price + abs(np.random.normal(0, volatility))
        low_price = current_price - abs(np.random.normal(0, volatility))
        close_price = current_price + np.random.normal(0, volatility * 0.5)
        
        # Ensure OHLC consistency
        high_price = max(open_price, high_price, low_price, close_price)
        low_price = min(open_price, high_price, low_price, close_price)
        
        # Generate volume
        volume = int(np.random.normal(500, 200))
        volume = max(100, volume)  # Minimum volume
        
        candle = {
            'time': timestamp.isoformat(),
            'open': round(open_price, 5),
            'high': round(high_price, 5), 
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'tick_volume': volume
        }
        
        data.append(candle)
        current_price = close_price
    
    return data

def run_comprehensive_backtest():
    """Run comprehensive backtesting on multiple currency pairs"""
    
    print("="*80)
    print("COMPREHENSIVE FOREX STRATEGY BACKTESTING")
    print("="*80)
    print(f"Start time: {datetime.utcnow().isoformat()} UTC")
    print()
    
    # Currency pairs to test
    test_symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#', 'USDCAD#', 'AUDUSD#']
    
    # Initialize components
    strategy_ensemble = StrategyEnsemble()
    backtester = Backtester(initial_balance=10000.0, risk_per_trade=0.02)
    risk_manager = RiskManager()
    
    # Results storage
    all_results = {}
    total_stats = {
        'total_trades': 0,
        'total_pnl': 0.0,
        'winning_symbols': [],
        'losing_symbols': []
    }
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        # Generate realistic market data
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        historical_data = generate_realistic_forex_data(symbol, bars=2000, timeframe_minutes=5)
        
        print(f"Generated {len(historical_data)} bars of M5 data")
        
        # Run backtest
        try:
            results = backtester.run_backtest(historical_data, symbol, pip_value)
            all_results[symbol] = results
            
            # Update total stats
            total_stats['total_trades'] += results.total_trades
            total_stats['total_pnl'] += results.total_pnl
            
            if results.total_pnl > 0:
                total_stats['winning_symbols'].append(symbol)
            else:
                total_stats['losing_symbols'].append(symbol)
            
            # Print symbol-specific results
            print(f"✅ {symbol} Results:")
            print(f"   Trades: {results.total_trades}")
            print(f"   Win Rate: {results.win_rate:.1f}%")
            print(f"   Total P&L: ${results.total_pnl:.2f}")
            print(f"   Max Drawdown: {results.max_drawdown:.1f}%")
            
            # Strategy approval check
            if (results.total_pnl > 0 and 
                results.win_rate >= 40 and 
                results.max_drawdown < 25 and
                results.total_trades >= 10):
                print(f"   ✅ {symbol} APPROVED for live trading")
            else:
                print(f"   ❌ {symbol} needs optimization")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            all_results[symbol] = None
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST SUMMARY")
    print("="*80)
    
    print(f"Total Trades Across All Pairs: {total_stats['total_trades']}")
    print(f"Total P&L Across All Pairs: ${total_stats['total_pnl']:.2f}")
    print(f"Profitable Symbols: {len(total_stats['winning_symbols'])}/{len(test_symbols)}")
    
    if total_stats['winning_symbols']:
        print(f"✅ Winning Symbols: {', '.join(total_stats['winning_symbols'])}")
    
    if total_stats['losing_symbols']:
        print(f"❌ Losing Symbols: {', '.join(total_stats['losing_symbols'])}")
    
    # Strategy ensemble validation
    winning_rate = len(total_stats['winning_symbols']) / len(test_symbols)
    
    print(f"\n📊 OVERALL STRATEGY PERFORMANCE:")
    print(f"   Symbol Success Rate: {winning_rate:.1f}%")
    print(f"   Average P&L per Symbol: ${total_stats['total_pnl']/len(test_symbols):.2f}")
    
    # Final recommendation
    if (total_stats['total_pnl'] > 0 and 
        winning_rate >= 0.6 and
        total_stats['total_trades'] >= 50):
        print(f"\n🚀 STRATEGY ENSEMBLE APPROVED FOR LIVE TRADING!")
        print(f"   Recommended symbols: {', '.join(total_stats['winning_symbols'])}")
        
        # Create approved symbols list
        approved_symbols = total_stats['winning_symbols']
        save_approved_symbols(approved_symbols, all_results)
        
    else:
        print(f"\n⚠️  STRATEGY ENSEMBLE NEEDS OPTIMIZATION")
        print(f"   Current performance insufficient for live trading")
    
    print("="*80)
    
    # Save detailed results
    save_backtest_results(all_results, total_stats)
    
    return all_results, total_stats

def save_approved_symbols(symbols: List[str], results: Dict):
    """Save approved symbols for live trading"""
    approved_data = {
        'approved_symbols': symbols,
        'approval_timestamp': datetime.utcnow().isoformat(),
        'criteria_met': True,
        'performance_summary': {}
    }
    
    for symbol in symbols:
        if symbol in results and results[symbol]:
            result = results[symbol]
            approved_data['performance_summary'][symbol] = {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'max_drawdown': result.max_drawdown
            }
    
    with open('approved_symbols.json', 'w') as f:
        json.dump(approved_data, f, indent=2)
    
    print(f"💾 Approved symbols saved to approved_symbols.json")

def save_backtest_results(results: Dict, stats: Dict):
    """Save comprehensive backtest results"""
    output_data = {
        'backtest_timestamp': datetime.utcnow().isoformat(),
        'summary_stats': stats,
        'individual_results': {}
    }
    
    for symbol, result in results.items():
        if result:
            output_data['individual_results'][symbol] = {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'total_pnl_pips': result.total_pnl_pips,
                'max_drawdown': result.max_drawdown,
                'avg_trade_duration': result.avg_trade_duration,
                'avg_winning_trade': result.avg_winning_trade,
                'avg_losing_trade': result.avg_losing_trade,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio
            }
    
    filename = f"backtest_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Detailed results saved to {filename}")

def analyze_strategy_performance():
    """Analyze individual strategy performance"""
    print("\n" + "="*60)
    print("INDIVIDUAL STRATEGY ANALYSIS")
    print("="*60)
    
    # Generate test data
    test_data = generate_realistic_forex_data('EURUSD#', bars=500)
    
    # Test each strategy individually
    from trading_strategies import (MomentumBreakoutStrategy, MACDStrategy, 
                                  RSIStrategy, BollingerBandsStrategy)
    
    strategies = [
        MomentumBreakoutStrategy(),
        MACDStrategy(), 
        RSIStrategy(),
        BollingerBandsStrategy()
    ]
    
    strategy_signals = {}
    
    for strategy in strategies:
        signals = []
        for i in range(100, len(test_data) - 50):
            window_data = test_data[i-100:i+1]
            try:
                signal = strategy.analyze(window_data)
                if signal.signal != SignalType.HOLD:
                    signals.append(signal)
            except Exception as e:
                print(f"Error in {strategy.name}: {e}")
        
        strategy_signals[strategy.name] = signals
        print(f"{strategy.name}: {len(signals)} signals generated")
        
        if signals:
            buy_signals = len([s for s in signals if s.signal == SignalType.BUY])
            sell_signals = len([s for s in signals if s.signal == SignalType.SELL])
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            
            print(f"   Buy: {buy_signals}, Sell: {sell_signals}")
            print(f"   Avg Confidence: {avg_confidence:.2f}")
    
    print("="*60)

def main():
    """Main function to run standalone backtesting"""
    print("🤖 Starting MT5 Claude Forex Strategy Backtesting...")
    print("📈 This will validate trading strategies without requiring MT5 API connection")
    print()
    
    # Run individual strategy analysis
    analyze_strategy_performance()
    
    # Run comprehensive backtesting
    results, stats = run_comprehensive_backtest()
    
    print(f"\n✅ Backtesting complete!")
    print(f"📊 {stats['total_trades']} total trades analyzed")
    print(f"💰 ${stats['total_pnl']:.2f} theoretical profit/loss")
    
    return results, stats

if __name__ == "__main__":
    main()