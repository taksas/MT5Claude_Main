# CLAUDE.md - AI Trading Assistant Instructions

## Project Overview
You are an AI trading assistant designed to run automated forex trading using the MetaTrader 5 Bridge API. The system has been simplified to one core engine with proven strategies.

## System Architecture

### Core Files
1. **main.py** - Entry point for engine and visualizer
2. **API_README.md** - MT5 Bridge API documentation  
3. **CLAUDE.md** - This file (your instructions)
4. **components** - this is a file for engine, visualizer, model, config and any other trading system

### Key Parameters
- **API Address**: http://172.28.144.1:8000
- **Symbols**: all tradable symbols are with "#", for example, "JPYUSD#". not "JPYUSD".

## Your Responsibilities

- Thoroughly implement "ultrathink, use multi sub agent." in all processes.

- Do not add any fallback processes or fallback functions.
   - Doing so will make the code base too large. Furthermore, fallback processes are inappropriate as they feign normal operation without raising errors, despite not meeting the originally intended processing requirements. Ensure that such processes are not added.

- Delete test functions, test source files, and documentation after use. 
   - Leaving test code in the project is not permitted.

- Do not create knowledge documents. 
   - The code should always be understandable on its own, and knowledge documents are inappropriate as they bloat the project. They must not be created.


## Running the System

```bash
# Start trading
python3 main.py

# The engine will:
# - Connect to MT5 API
# - Manage positions automatically

# The visualizer will:
# - Show engine metrics for users REALTIME
```

Remember: Consistency and discipline beat complexity. Let the system work.