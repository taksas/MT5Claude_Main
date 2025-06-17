"""
News-Aware Trading System
Integrates news monitoring with live trading to avoid high-impact events
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from news_monitor import NewsMonitor
from enhanced_live_trading_engine import EnhancedLiveTradingEngine

logger = logging.getLogger(__name__)

class NewsAwareTradingEngine(EnhancedLiveTradingEngine):
    """
    Enhanced trading engine that incorporates news monitoring
    """
    
    def __init__(self, api_url: str = "http://172.28.144.1:8000", 
                 risk_per_trade: float = 0.01,
                 max_daily_loss: float = 0.05,
                 news_risk_tolerance: str = 'low'):
        super().__init__(api_url, risk_per_trade, max_daily_loss)
        self.news_monitor = NewsMonitor()
        self.news_risk_tolerance = news_risk_tolerance
        self.news_check_interval = 300  # Check news every 5 minutes
        self.last_news_check = None
        
    async def should_trade_symbol(self, symbol: str) -> tuple[bool, str]:
        """
        Check if we should trade a symbol based on technical and news analysis
        """
        # First check technical conditions
        should_trade_tech, reason_tech = await super().should_trade_symbol(symbol)
        
        if not should_trade_tech:
            return False, reason_tech
        
        # Check news conditions
        try:
            # Get news recommendation
            news_rec = self.news_monitor.get_trading_recommendation(symbol)
            
            # Check if we should avoid trading
            if self.news_monitor.should_avoid_trading(symbol, self.news_risk_tolerance):
                reason = f"News Risk: {news_rec['reason']}"
                logger.warning(f"Avoiding {symbol} due to news: {reason}")
                return False, reason
            
            # If news suggests caution, we might still trade but with adjusted parameters
            if news_rec['risk_level'] == 'medium':
                logger.info(f"Caution for {symbol}: {news_rec['reason']}")
                # You could adjust position size or other parameters here
            
            # Log news sentiment
            logger.info(f"News sentiment for {symbol}: "
                       f"{news_rec['base_currency_sentiment']}/{news_rec['quote_currency_sentiment']}")
            
            return True, f"Technical OK, News: {news_rec['recommendation']}"
            
        except Exception as e:
            logger.error(f"Error checking news for {symbol}: {e}")
            # If news check fails, use risk tolerance to decide
            if self.news_risk_tolerance == 'low':
                return False, "News check failed - avoiding trade"
            else:
                return True, "Technical OK, news check failed but proceeding"
    
    async def pre_trade_checks(self, symbol: str) -> Dict:
        """
        Perform comprehensive pre-trade checks including news
        """
        checks = await super().pre_trade_checks(symbol)
        
        # Add news checks
        try:
            news_rec = self.news_monitor.get_trading_recommendation(symbol)
            
            checks['news_safe'] = news_rec['is_safe']
            checks['news_risk_level'] = news_rec['risk_level']
            checks['upcoming_events'] = len(news_rec['upcoming_events'])
            checks['news_recommendation'] = news_rec['recommendation']
            
            # Overall pass/fail including news
            if not checks['news_safe'] and self.news_risk_tolerance == 'low':
                checks['all_checks_passed'] = False
                checks['failure_reasons'].append(f"High impact news event")
                
        except Exception as e:
            logger.error(f"Error in news pre-trade checks: {e}")
            checks['news_safe'] = False
            checks['news_risk_level'] = 'unknown'
            
        return checks
    
    async def analyze_market_conditions(self):
        """
        Analyze overall market conditions including news
        """
        conditions = await super().analyze_market_conditions()
        
        # Add news sentiment
        try:
            sentiment = self.news_monitor.get_market_sentiment()
            conditions['news_sentiment'] = sentiment
            
            # Check for any high-impact events in next hour
            high_impact_count = 0
            events = self.news_monitor.get_all_events()
            
            for event in events:
                if self.news_monitor.assess_event_impact(event) == 'high':
                    event_time = event.get('time', datetime.now())
                    if (event_time - datetime.now()).total_seconds() < 3600:
                        high_impact_count += 1
            
            conditions['upcoming_high_impact_events'] = high_impact_count
            
            # Adjust market condition based on news
            if high_impact_count > 2:
                conditions['overall_condition'] = 'high_risk'
                conditions['trade_recommendation'] = 'avoid'
                
        except Exception as e:
            logger.error(f"Error analyzing news conditions: {e}")
            
        return conditions
    
    async def periodic_news_update(self):
        """
        Periodically update news data
        """
        while self.is_running:
            try:
                # Force update news cache
                logger.info("Updating news data...")
                events = self.news_monitor.get_all_events(force_update=True)
                logger.info(f"Found {len(events)} news events")
                
                # Log any high impact events in next 2 hours
                for event in events:
                    if self.news_monitor.assess_event_impact(event) == 'high':
                        event_time = event.get('time', datetime.now())
                        time_until = (event_time - datetime.now()).total_seconds() / 60
                        
                        if 0 < time_until < 120:
                            logger.warning(f"High impact event in {time_until:.0f} minutes: "
                                         f"{event.get('title', 'Unknown')}")
                
                # Check all active positions for news risk
                if self.positions:
                    for symbol, position in self.positions.items():
                        if not self.news_monitor.should_avoid_trading(symbol, 'medium'):
                            continue
                            
                        # Consider closing position before high impact news
                        logger.warning(f"High impact news coming for {symbol} - "
                                     f"consider closing position")
                        
                        # You could implement auto-close logic here
                        # await self.close_position(symbol, "News risk")
                
            except Exception as e:
                logger.error(f"Error in news update: {e}")
            
            await asyncio.sleep(self.news_check_interval)
    
    async def start(self):
        """
        Start the news-aware trading engine
        """
        logger.info("Starting News-Aware Trading Engine...")
        
        # Start news monitoring task
        asyncio.create_task(self.periodic_news_update())
        
        # Start the main trading engine
        await super().start()
    
    def get_position_size_adjustment(self, symbol: str) -> float:
        """
        Adjust position size based on news risk
        """
        base_adjustment = super().get_position_size_adjustment(symbol)
        
        try:
            news_rec = self.news_monitor.get_trading_recommendation(symbol)
            
            # Reduce position size for medium risk news
            if news_rec['risk_level'] == 'medium':
                news_adjustment = 0.5  # 50% reduction
            else:
                news_adjustment = 1.0
                
            return base_adjustment * news_adjustment
            
        except Exception as e:
            logger.error(f"Error getting news position adjustment: {e}")
            return base_adjustment * 0.5  # Be conservative if news check fails


async def main():
    """
    Run the news-aware trading system
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'news_aware_trading_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and configure the trading engine
    engine = NewsAwareTradingEngine(
        api_url="http://172.28.144.1:8000",
        risk_per_trade=0.01,
        max_daily_loss=0.05,
        news_risk_tolerance='low'  # 'low', 'medium', or 'high'
    )
    
    # Set trading parameters
    engine.symbols = ['EURUSD#', 'GBPUSD#', 'USDJPY#', 'AUDUSD#']
    engine.lot_size = 0.01
    engine.stop_loss_pips = 20
    engine.take_profit_pips = 30
    engine.max_positions = 2
    engine.correlation_threshold = 0.7
    
    # Add strategies
    from improved_strategies import (
        AdvancedTrendFollowing,
        MeanReversionStrategy,
        BreakoutStrategy,
        MultiTimeframeStrategy
    )
    
    engine.strategies = [
        AdvancedTrendFollowing(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        MultiTimeframeStrategy()
    ]
    
    # Run the engine
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Shutting down news-aware trading engine...")
        await engine.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())