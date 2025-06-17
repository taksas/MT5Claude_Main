#!/usr/bin/env python3
"""
Enhanced Backtesting System with Improved 2025 Forex Strategies
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List
from improved_strategies import ImprovedStrategyEnsemble
from backtesting import Backtester
from risk_manager import RiskManager

def generate_enhanced_forex_data(symbol: str, bars: int = 2000, timeframe_minutes: int = 5) -> List[Dict]:
    """Generate more realistic forex data with proper market microstructure"""
    
    base_prices = {
        'EURUSD': 1.0850,  # Current approximate levels
        'GBPUSD': 1.2650, 
        'USDJPY': 149.50,
        'USDCAD': 1.3580,
        'AUDUSD': 0.6720,
        'NZDUSD': 0.6180,
        'USDCHF': 0.8950
    }
    
    base_price = base_prices.get(symbol.replace('#', ''), 1.0850)
    start_time = datetime.utcnow() - timedelta(minutes=bars * timeframe_minutes)
    
    data = []
    current_price = base_price
    
    # Enhanced volatility patterns based on trading sessions
    for i in range(bars):
        timestamp = start_time + timedelta(minutes=i * timeframe_minutes)
        hour = timestamp.hour
        
        # Session-based volatility (higher during London/NY overlap)
        if 13 <= hour <= 17:  # London-NY overlap
            session_volatility = 1.5
        elif 8 <= hour <= 17:  # London session
            session_volatility = 1.2
        elif 0 <= hour <= 9:  # Asian session
            session_volatility = 0.8
        else:  # Off-hours
            session_volatility = 0.6
        
        # News impact simulation (random news events)
        news_impact = 1.0
        if random.random() < 0.05:  # 5% chance of news
            news_impact = random.uniform(0.7, 1.8)
        
        # Base volatility with clustering
        base_volatility = 0.0008 if 'JPY' in symbol else 0.0004
        volatility = base_volatility * session_volatility * news_impact
        
        # Add volatility clustering (GARCH effect)
        if i > 10:
            recent_moves = [abs(data[j]['close'] - data[j-1]['close']) for j in range(max(0, i-10), i)]
            if recent_moves:
                avg_recent_vol = np.mean(recent_moves)
                volatility *= (1 + avg_recent_vol / base_price * 1000)
        
        # Generate price movement with slight momentum
        momentum = 0.0001 * np.sin(i / 50) if random.random() < 0.3 else 0
        price_change = np.random.normal(momentum, volatility)
        current_price += price_change
        
        # Generate realistic OHLC
        intra_volatility = volatility * 0.7
        open_price = current_price
        
        # High and low with realistic ranges
        high_offset = abs(np.random.normal(0, intra_volatility))
        low_offset = abs(np.random.normal(0, intra_volatility))
        
        high_price = current_price + high_offset
        low_price = current_price - low_offset
        
        # Close price with some drift
        close_drift = np.random.normal(0, intra_volatility * 0.5)
        close_price = current_price + close_drift
        
        # Ensure OHLC consistency
        high_price = max(open_price, high_price, low_price, close_price)
        low_price = min(open_price, high_price, low_price, close_price)
        
        # Generate realistic volume based on volatility
        base_volume = 800 if 'JPY' in symbol else 600
        volume_variance = session_volatility * news_impact
        volume = int(np.random.normal(base_volume * volume_variance, base_volume * 0.3))
        volume = max(100, volume)
        
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

def run_enhanced_backtest():
    """Run enhanced backtesting with improved strategies"""
    
    print("="*90)
    print("ENHANCED FOREX STRATEGY BACKTESTING - 2025 OPTIMIZED STRATEGIES")
    print("="*90)
    print(f"Start time: {datetime.utcnow().isoformat()} UTC")
    print()
    
    # Focus on most liquid pairs during optimal trading hours
    test_symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#', 'USDCAD#', 'AUDUSD#']
    
    # Initialize enhanced components
    enhanced_ensemble = ImprovedStrategyEnsemble()
    enhanced_backtester = Backtester(initial_balance=10000.0, risk_per_trade=0.015)  # Slightly lower risk
    risk_manager = RiskManager()
    
    all_results = {}
    total_stats = {
        'total_trades': 0,
        'total_pnl': 0.0,
        'winning_symbols': [],
        'losing_symbols': [],
        'approved_symbols': []
    }
    
    for symbol in test_symbols:
        print(f"\n🔍 Enhanced Testing {symbol}...")
        
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        # Generate more realistic market data
        historical_data = generate_enhanced_forex_data(symbol, bars=3000, timeframe_minutes=5)
        print(f"Generated {len(historical_data)} bars of enhanced M5 data")
        
        try:
            # Run backtest with enhanced ensemble
            results = enhanced_backtester.run_backtest(historical_data, symbol, pip_value)
            
            # Override ensemble in backtester for this test
            original_ensemble = enhanced_backtester.strategy_ensemble
            enhanced_backtester.strategy_ensemble = enhanced_ensemble
            
            # Run enhanced backtest
            enhanced_results = enhanced_backtester.run_backtest(historical_data, symbol, pip_value)
            
            # Restore original ensemble
            enhanced_backtester.strategy_ensemble = original_ensemble
            
            all_results[symbol] = enhanced_results
            
            # Update total stats
            total_stats['total_trades'] += enhanced_results.total_trades
            total_stats['total_pnl'] += enhanced_results.total_pnl
            
            if enhanced_results.total_pnl > 0:
                total_stats['winning_symbols'].append(symbol)
            else:
                total_stats['losing_symbols'].append(symbol)
            
            # Enhanced approval criteria
            is_approved = (
                enhanced_results.total_pnl > 100 and  # Minimum $100 profit
                enhanced_results.win_rate >= 45 and   # At least 45% win rate
                enhanced_results.max_drawdown < 20 and  # Less than 20% drawdown
                enhanced_results.total_trades >= 15 and  # Minimum trade sample
                enhanced_results.profit_factor > 1.2   # Good profit factor
            )
            
            if is_approved:
                total_stats['approved_symbols'].append(symbol)
            
            # Print enhanced results
            print(f"✅ {symbol} Enhanced Results:")
            print(f"   Trades: {enhanced_results.total_trades}")
            print(f"   Win Rate: {enhanced_results.win_rate:.1f}%")
            print(f"   Total P&L: ${enhanced_results.total_pnl:.2f}")
            print(f"   Max Drawdown: {enhanced_results.max_drawdown:.1f}%")
            print(f"   Profit Factor: {enhanced_results.profit_factor:.2f}")
            print(f"   Sharpe Ratio: {enhanced_results.sharpe_ratio:.2f}")
            
            if is_approved:
                print(f"   🚀 {symbol} APPROVED for live trading!")
            else:
                print(f"   ❌ {symbol} needs further optimization")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            all_results[symbol] = None
    
    # Print comprehensive enhanced summary
    print("\n" + "="*90)
    print("ENHANCED BACKTEST COMPREHENSIVE SUMMARY")
    print("="*90)
    
    print(f"Total Trades Across All Pairs: {total_stats['total_trades']}")
    print(f"Total P&L Across All Pairs: ${total_stats['total_pnl']:.2f}")
    print(f"Profitable Symbols: {len(total_stats['winning_symbols'])}/{len(test_symbols)}")
    print(f"APPROVED Symbols: {len(total_stats['approved_symbols'])}/{len(test_symbols)}")
    
    if total_stats['winning_symbols']:
        print(f"✅ Winning Symbols: {', '.join(total_stats['winning_symbols'])}")
    
    if total_stats['approved_symbols']:
        print(f"🚀 APPROVED Symbols: {', '.join(total_stats['approved_symbols'])}")
    
    if total_stats['losing_symbols']:
        print(f"❌ Losing Symbols: {', '.join(total_stats['losing_symbols'])}")
    
    # Enhanced performance metrics
    approval_rate = len(total_stats['approved_symbols']) / len(test_symbols)
    winning_rate = len(total_stats['winning_symbols']) / len(test_symbols)
    
    print(f"\n📊 ENHANCED STRATEGY PERFORMANCE:")
    print(f"   Symbol Approval Rate: {approval_rate:.1%}")
    print(f"   Symbol Winning Rate: {winning_rate:.1%}")
    print(f"   Average P&L per Symbol: ${total_stats['total_pnl']/len(test_symbols):.2f}")
    
    # Enhanced final recommendation
    if (total_stats['total_pnl'] > 200 and 
        approval_rate >= 0.4 and
        len(total_stats['approved_symbols']) >= 2 and
        total_stats['total_trades'] >= 75):
        
        print(f"\n🎯 ENHANCED STRATEGY ENSEMBLE APPROVED FOR LIVE TRADING!")
        print(f"   🚀 Ready for deployment with approved symbols")
        print(f"   💰 Expected performance: ${total_stats['total_pnl']:.2f} across test period")
        print(f"   ⚡ Recommended symbols: {', '.join(total_stats['approved_symbols'])}")
        
        save_approved_trading_config(total_stats['approved_symbols'], all_results)
        
    else:
        print(f"\n⚠️  ENHANCED STRATEGY ENSEMBLE NEEDS FURTHER OPTIMIZATION")
        print(f"   Current performance: {approval_rate:.1%} approval rate")
        print(f"   Target: 40%+ approval rate with 2+ symbols")
    
    print("="*90)
    
    # Save enhanced results
    save_enhanced_results(all_results, total_stats)
    
    return all_results, total_stats

def save_approved_trading_config(symbols: List[str], results: Dict):
    """Save approved trading configuration for live deployment"""
    config = {
        'deployment_ready': True,
        'approved_symbols': symbols,
        'approval_timestamp': datetime.utcnow().isoformat(),
        'strategy_type': 'enhanced_2025_ensemble',
        'risk_management': {
            'max_risk_per_trade': 0.015,
            'max_concurrent_trades': 3,
            'max_daily_trades': 12,
            'required_confidence': 0.70
        },
        'performance_summary': {},
        'trading_sessions': {
            'london_session': {'start': 8, 'end': 17, 'priority': 'high'},
            'ny_overlap': {'start': 13, 'end': 17, 'priority': 'highest'},
            'asian_session': {'start': 0, 'end': 9, 'priority': 'medium'}
        }
    }
    
    for symbol in symbols:
        if symbol in results and results[symbol]:
            result = results[symbol]
            config['performance_summary'][symbol] = {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'max_drawdown': result.max_drawdown,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio
            }
    
    with open('live_trading_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"🎯 Live trading configuration saved to live_trading_config.json")

def save_enhanced_results(results: Dict, stats: Dict):
    """Save enhanced backtest results"""
    output_data = {
        'backtest_type': 'enhanced_2025_strategies',
        'backtest_timestamp': datetime.utcnow().isoformat(),
        'summary_stats': stats,
        'individual_results': {},
        'strategy_improvements': {
            'improved_momentum': 'EMA crossover with volume confirmation',
            'vwap_scalping': 'VWAP-RSI scalping for short-term trades',
            'keltner_channel': 'Breakout strategy with RSI confirmation',
            'alma_stochastic': 'ALMA trend with stochastic momentum'
        }
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
    
    filename = f"enhanced_backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Enhanced results saved to {filename}")

def main():
    """Main function for enhanced backtesting"""
    print("🚀 Starting Enhanced MT5 Claude Forex Strategy Backtesting...")
    print("📈 Testing 2025-optimized trading strategies")
    print("⚡ Enhanced with VWAP, Keltner Channels, ALMA, and improved momentum")
    print()
    
    results, stats = run_enhanced_backtest()
    
    print(f"\n✅ Enhanced backtesting complete!")
    print(f"📊 {stats['total_trades']} total trades analyzed")
    print(f"💰 ${stats['total_pnl']:.2f} theoretical profit/loss")
    print(f"🎯 {len(stats['approved_symbols'])} symbols approved for live trading")
    
    return results, stats

if __name__ == "__main__":
    main()