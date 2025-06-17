"""
Test script for the news monitor module
"""

import asyncio
import logging
from datetime import datetime
from news_monitor import NewsMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_news_monitor():
    """Test the news monitor functionality"""
    
    print("=" * 60)
    print("NEWS MONITOR TEST")
    print("=" * 60)
    
    # Create monitor
    monitor = NewsMonitor()
    
    # Test 1: Get all events
    print("\n1. Fetching all events...")
    events = monitor.get_all_events()
    print(f"   Found {len(events)} events")
    
    if events:
        print("\n   Recent events:")
        for event in events[:3]:
            print(f"   - {event.get('title', 'No title')}")
            print(f"     Impact: {event.get('impact', 'unknown')}")
            print(f"     Time: {event.get('time', 'unknown')}")
            print()
    
    # Test 2: Check symbol risk
    symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#']
    
    print("\n2. Checking symbol risks...")
    for symbol in symbols:
        is_safe, risk_level, relevant_events = monitor.check_symbol_risk(symbol.replace('#', ''))
        
        print(f"\n   {symbol}:")
        print(f"   - Safe to trade: {is_safe}")
        print(f"   - Risk level: {risk_level}")
        print(f"   - Relevant events: {len(relevant_events)}")
        
        if relevant_events:
            for event in relevant_events[:2]:
                print(f"     * {event.get('title', 'No title')} ({event.get('impact', 'unknown')})")
    
    # Test 3: Get market sentiment
    print("\n3. Market sentiment analysis...")
    sentiment = monitor.get_market_sentiment()
    
    for currency, sent in sentiment.items():
        print(f"   {currency}: {sent}")
    
    # Test 4: Get trading recommendations
    print("\n4. Trading recommendations...")
    
    for symbol in symbols:
        rec = monitor.get_trading_recommendation(symbol.replace('#', ''))
        
        print(f"\n   {symbol}:")
        print(f"   - Recommendation: {rec['recommendation']}")
        print(f"   - Reason: {rec['reason']}")
        print(f"   - Base sentiment: {rec['base_currency_sentiment']}")
        print(f"   - Quote sentiment: {rec['quote_currency_sentiment']}")
    
    # Test 5: Risk tolerance tests
    print("\n5. Risk tolerance tests...")
    
    risk_tolerances = ['low', 'medium', 'high']
    test_symbol = 'EURUSD'
    
    for tolerance in risk_tolerances:
        should_avoid = monitor.should_avoid_trading(test_symbol, tolerance)
        print(f"   {tolerance} tolerance - Avoid {test_symbol}: {should_avoid}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_integration_example():
    """Show how to integrate with trading logic"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)
    
    monitor = NewsMonitor()
    
    # Simulated trading decision
    symbol = 'EURUSD'
    
    print(f"\nEvaluating trade for {symbol}...")
    
    # 1. Check if safe to trade
    is_safe, risk_level, events = monitor.check_symbol_risk(symbol)
    
    if not is_safe:
        print(f"⚠️  AVOID TRADE - High impact news detected!")
        for event in events:
            print(f"   - {event['title']} at {event['time']}")
        return
    
    # 2. Get recommendation
    rec = monitor.get_trading_recommendation(symbol)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Risk Level: {risk_level}")
    print(f"   Recommendation: {rec['recommendation']}")
    print(f"   Reason: {rec['reason']}")
    
    # 3. Make decision
    if rec['recommendation'] in ['buy', 'sell']:
        print(f"\n✅ EXECUTE {rec['recommendation'].upper()} trade")
        print(f"   Note: Use proper risk management")
    elif rec['recommendation'] == 'caution':
        print(f"\n⚠️  PROCEED WITH CAUTION")
        print(f"   Consider reducing position size")
    else:
        print(f"\n❌ NO TRADE - Wait for better conditions")


if __name__ == "__main__":
    # Run tests
    test_news_monitor()
    test_integration_example()