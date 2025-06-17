"""
Forex News Monitor Module
Monitors economic events and news to assess trading risks
"""

import requests
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import re
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class NewsMonitor:
    """
    Monitors forex news and economic events to assess trading risks
    """
    
    def __init__(self):
        self.high_impact_keywords = [
            'interest rate', 'gdp', 'inflation', 'cpi', 'employment', 'nfp',
            'non-farm payroll', 'fomc', 'ecb', 'boe', 'boj', 'central bank',
            'monetary policy', 'trade balance', 'retail sales', 'pmi',
            'manufacturing', 'consumer confidence', 'unemployment'
        ]
        
        self.currency_map = {
            'USD': ['dollar', 'usd', 'united states', 'us ', 'america', 'fed', 'fomc'],
            'EUR': ['euro', 'eur', 'european', 'ecb', 'eurozone'],
            'GBP': ['pound', 'gbp', 'british', 'uk ', 'england', 'boe'],
            'JPY': ['yen', 'jpy', 'japan', 'boj'],
            'AUD': ['aussie', 'aud', 'australia', 'rba'],
            'NZD': ['kiwi', 'nzd', 'new zealand', 'rbnz'],
            'CAD': ['loonie', 'cad', 'canada', 'boc'],
            'CHF': ['franc', 'chf', 'swiss', 'snb']
        }
        
        self.cached_events = []
        self.last_update = None
        self.update_interval = 300  # Update every 5 minutes
        
    def fetch_forex_factory_events(self) -> List[Dict]:
        """
        Fetch economic calendar from Forex Factory
        Note: This is a simplified scraper - in production, use proper APIs
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Get today's date for the calendar
            today = datetime.now().strftime('%Y-%m-%d')
            
            # This is a placeholder - Forex Factory requires more complex scraping
            # In production, use proper economic calendar APIs
            logger.info("Checking for economic events...")
            
            # For now, return mock data for testing
            # In production, implement proper web scraping or use APIs
            return self._get_mock_events()
            
        except Exception as e:
            logger.error(f"Error fetching Forex Factory events: {e}")
            return []
    
    def fetch_fxstreet_news(self) -> List[Dict]:
        """
        Fetch news from FXStreet API (if available) or scrape
        """
        try:
            # FXStreet has RSS feeds that can be parsed
            rss_url = "https://www.fxstreet.com/rss/news"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(rss_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Parse RSS feed
                from xml.etree import ElementTree
                root = ElementTree.fromstring(response.content)
                
                news_items = []
                for item in root.findall('.//item')[:10]:  # Get latest 10 items
                    title = item.find('title').text if item.find('title') is not None else ''
                    description = item.find('description').text if item.find('description') is not None else ''
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                    
                    news_items.append({
                        'title': title,
                        'description': description,
                        'time': self._parse_rss_date(pub_date),
                        'source': 'FXStreet'
                    })
                
                return news_items
            
        except Exception as e:
            logger.error(f"Error fetching FXStreet news: {e}")
        
        return []
    
    def fetch_investing_com_data(self) -> List[Dict]:
        """
        Fetch economic calendar data (simplified version)
        In production, use proper APIs or more robust scraping
        """
        try:
            # This is a placeholder for the concept
            # Investing.com requires more complex handling
            logger.info("Checking Investing.com economic calendar...")
            
            # Return empty for now - implement proper scraping if needed
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Investing.com data: {e}")
            return []
    
    def get_all_events(self, force_update: bool = False) -> List[Dict]:
        """
        Get all events from various sources
        """
        now = datetime.now()
        
        # Check if we need to update
        if (not force_update and self.last_update and 
            (now - self.last_update).seconds < self.update_interval and
            self.cached_events):
            return self.cached_events
        
        all_events = []
        
        # Fetch from multiple sources
        all_events.extend(self.fetch_forex_factory_events())
        all_events.extend(self.fetch_fxstreet_news())
        all_events.extend(self.fetch_investing_com_data())
        
        # Sort by time
        all_events.sort(key=lambda x: x.get('time', now), reverse=True)
        
        self.cached_events = all_events
        self.last_update = now
        
        return all_events
    
    def assess_event_impact(self, event: Dict) -> str:
        """
        Assess the impact level of an event
        Returns: 'high', 'medium', 'low'
        """
        title = event.get('title', '').lower()
        description = event.get('description', '').lower()
        impact = event.get('impact', 'medium')
        
        # Check for high impact keywords
        for keyword in self.high_impact_keywords:
            if keyword in title or keyword in description:
                return 'high'
        
        # If impact is already specified
        if impact.lower() in ['high', 'red']:
            return 'high'
        elif impact.lower() in ['medium', 'orange']:
            return 'medium'
        
        return 'low'
    
    def get_affected_currencies(self, event: Dict) -> List[str]:
        """
        Determine which currencies are affected by an event
        """
        affected = []
        title = event.get('title', '').lower()
        description = event.get('description', '').lower()
        currency = event.get('currency', '')
        
        # Check explicit currency
        if currency:
            affected.append(currency.upper())
        
        # Check for currency mentions
        for curr, keywords in self.currency_map.items():
            for keyword in keywords:
                if keyword in title or keyword in description:
                    if curr not in affected:
                        affected.append(curr)
        
        return affected
    
    def check_symbol_risk(self, symbol: str, time_window_minutes: int = 30) -> Tuple[bool, str, List[Dict]]:
        """
        Check if it's safe to trade a specific symbol
        Returns: (is_safe, risk_level, relevant_events)
        """
        # Extract currencies from symbol
        base_currency = symbol[:3].upper()
        quote_currency = symbol[3:6].upper()
        
        now = datetime.now()
        future_time = now + timedelta(minutes=time_window_minutes)
        
        # Get all events
        events = self.get_all_events()
        
        relevant_events = []
        highest_risk = 'low'
        
        for event in events:
            event_time = event.get('time', now)
            
            # Check if event is within time window
            if now <= event_time <= future_time:
                affected_currencies = self.get_affected_currencies(event)
                
                # Check if our symbol is affected
                if base_currency in affected_currencies or quote_currency in affected_currencies:
                    impact = self.assess_event_impact(event)
                    relevant_events.append({
                        **event,
                        'impact': impact,
                        'affected_currencies': affected_currencies
                    })
                    
                    # Update highest risk
                    if impact == 'high':
                        highest_risk = 'high'
                    elif impact == 'medium' and highest_risk != 'high':
                        highest_risk = 'medium'
        
        # Determine if safe to trade
        is_safe = highest_risk != 'high'
        
        return is_safe, highest_risk, relevant_events
    
    def get_market_sentiment(self) -> Dict[str, str]:
        """
        Get overall market sentiment based on recent news
        """
        events = self.get_all_events()
        
        sentiment_scores = {}
        
        for currency in self.currency_map.keys():
            positive_count = 0
            negative_count = 0
            
            for event in events[:20]:  # Check last 20 events
                if currency in self.get_affected_currencies(event):
                    title = event.get('title', '').lower()
                    description = event.get('description', '').lower()
                    
                    # Simple sentiment analysis
                    positive_words = ['rise', 'gain', 'up', 'positive', 'growth', 'increase', 'strong']
                    negative_words = ['fall', 'drop', 'down', 'negative', 'decline', 'decrease', 'weak']
                    
                    for word in positive_words:
                        if word in title or word in description:
                            positive_count += 1
                            break
                    
                    for word in negative_words:
                        if word in title or word in description:
                            negative_count += 1
                            break
            
            # Determine sentiment
            if positive_count > negative_count:
                sentiment_scores[currency] = 'bullish'
            elif negative_count > positive_count:
                sentiment_scores[currency] = 'bearish'
            else:
                sentiment_scores[currency] = 'neutral'
        
        return sentiment_scores
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """
        Parse RSS date format
        """
        try:
            # RSS date format: 'Mon, 17 Jun 2024 12:00:00 GMT'
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
        except:
            return datetime.now()
    
    def _get_mock_events(self) -> List[Dict]:
        """
        Get mock events for testing
        In production, replace with real data
        """
        now = datetime.now()
        
        return [
            {
                'title': 'US Non-Farm Payrolls',
                'description': 'US employment data release',
                'time': now + timedelta(hours=2),
                'currency': 'USD',
                'impact': 'high',
                'forecast': '200K',
                'previous': '175K'
            },
            {
                'title': 'ECB Interest Rate Decision',
                'description': 'European Central Bank rate decision',
                'time': now + timedelta(hours=4),
                'currency': 'EUR',
                'impact': 'high',
                'forecast': '4.25%',
                'previous': '4.25%'
            },
            {
                'title': 'UK Retail Sales',
                'description': 'UK retail sales data',
                'time': now + timedelta(hours=1),
                'currency': 'GBP',
                'impact': 'medium',
                'forecast': '0.3%',
                'previous': '-0.2%'
            }
        ]
    
    def should_avoid_trading(self, symbol: str, risk_tolerance: str = 'low') -> bool:
        """
        Simple method to check if trading should be avoided
        risk_tolerance: 'low', 'medium', 'high'
        """
        is_safe, risk_level, events = self.check_symbol_risk(symbol)
        
        if risk_tolerance == 'low':
            # Avoid trading if any medium or high impact events
            return risk_level in ['medium', 'high']
        elif risk_tolerance == 'medium':
            # Only avoid high impact events
            return risk_level == 'high'
        else:  # high tolerance
            # Only avoid if multiple high impact events
            high_impact_count = sum(1 for e in events if e.get('impact') == 'high')
            return high_impact_count > 1
    
    def get_trading_recommendation(self, symbol: str) -> Dict:
        """
        Get a comprehensive trading recommendation based on news
        """
        is_safe, risk_level, events = self.check_symbol_risk(symbol)
        sentiment = self.get_market_sentiment()
        
        base_currency = symbol[:3].upper()
        quote_currency = symbol[3:6].upper()
        
        recommendation = {
            'symbol': symbol,
            'is_safe': is_safe,
            'risk_level': risk_level,
            'upcoming_events': events,
            'base_currency_sentiment': sentiment.get(base_currency, 'neutral'),
            'quote_currency_sentiment': sentiment.get(quote_currency, 'neutral'),
            'recommendation': 'wait',
            'reason': ''
        }
        
        # Determine recommendation
        if not is_safe:
            recommendation['recommendation'] = 'avoid'
            recommendation['reason'] = f'High impact news event within 30 minutes'
        elif risk_level == 'medium':
            recommendation['recommendation'] = 'caution'
            recommendation['reason'] = 'Medium impact events detected - trade with reduced size'
        else:
            base_sent = sentiment.get(base_currency, 'neutral')
            quote_sent = sentiment.get(quote_currency, 'neutral')
            
            if base_sent == 'bullish' and quote_sent == 'bearish':
                recommendation['recommendation'] = 'buy'
                recommendation['reason'] = f'{base_currency} bullish, {quote_currency} bearish'
            elif base_sent == 'bearish' and quote_sent == 'bullish':
                recommendation['recommendation'] = 'sell'
                recommendation['reason'] = f'{base_currency} bearish, {quote_currency} bullish'
            else:
                recommendation['recommendation'] = 'neutral'
                recommendation['reason'] = 'No clear sentiment direction'
        
        return recommendation


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create news monitor
    monitor = NewsMonitor()
    
    # Test symbols
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    print("Forex News Monitor Test")
    print("=" * 50)
    
    # Check each symbol
    for symbol in test_symbols:
        print(f"\nChecking {symbol}:")
        
        # Get recommendation
        rec = monitor.get_trading_recommendation(symbol)
        
        print(f"  Safe to trade: {rec['is_safe']}")
        print(f"  Risk level: {rec['risk_level']}")
        print(f"  Recommendation: {rec['recommendation']}")
        print(f"  Reason: {rec['reason']}")
        
        if rec['upcoming_events']:
            print(f"  Upcoming events:")
            for event in rec['upcoming_events']:
                print(f"    - {event['title']} ({event['impact']} impact)")
    
    # Get market sentiment
    print("\nMarket Sentiment:")
    sentiment = monitor.get_market_sentiment()
    for currency, sent in sentiment.items():
        print(f"  {currency}: {sent}")