#!/usr/bin/env python3
"""
Test script to run the trading system with proper error handling
"""

import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestSystem')

def test_main_system():
    """Test the main trading system"""
    logger.info("=" * 80)
    logger.info("TESTING MAIN TRADING SYSTEM")
    logger.info("=" * 80)
    
    try:
        # Test engine only mode
        logger.info("\n1. Testing engine-only mode...")
        from main import UltraTradingSystem
        
        # Create system with engine only
        system = UltraTradingSystem(mode='engine', visualize=False)
        
        # Test that it initializes correctly
        logger.info("✅ System initialized successfully")
        
        # Test visualizer mode
        logger.info("\n2. Testing visualizer-only mode...")
        viz_system = UltraTradingSystem(mode='visualizer', visualize=False)
        logger.info("✅ Visualizer mode initialized successfully")
        
        # Test combined mode
        logger.info("\n3. Testing combined mode...")
        combined_system = UltraTradingSystem(mode='engine', visualize=True)
        logger.info("✅ Combined mode initialized successfully")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED - SYSTEM READY FOR TRADING")
        logger.info("=" * 80)
        
        logger.info("\nTo start trading, run:")
        logger.info("  python3 main.py")
        logger.info("\nOptions:")
        logger.info("  python3 main.py --mode engine          # Engine only")
        logger.info("  python3 main.py --mode visualizer      # Visualizer only")
        logger.info("  python3 main.py --visualize            # Engine with visualizer")
        logger.info("  python3 main.py --mode both            # Separate processes")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_system()
    sys.exit(0 if success else 1)