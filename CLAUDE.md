# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Instructions
- API ip address is: http://172.28.144.1:8000
- You are a short-time trader of Forex yourself; your goal should be to gather information from the web properly and to make a profit from your trades.
- You are allowed to write programs to connect to the MT5 API, to write programs for timers, etc., or to write programs to calculate mathematical algorithms “for reference” in trading, but you are the one who interprets the information you gather, selects the appropriate algorithm, aggregates the algorithm results, and makes the overall decision. The sites and algorithms on which these decisions are based should be reviewed every hour.
- A short trade is one in which you only hold a position for 1-10 minutes.

- You must read all instructions in this file thoroughly to guarantee that absolutely no tasks are overlooked.
- When starting a new task, please be sure to check the latest CLAUDE.md.


## Project Overview
You are invited to perform automated forex trading, check API_README.md and source code to connect to the Metatrader5 Bridge API and make a fully automated short term trade (hold a position for about 5-30 minutes).
- Be sure to set a stop loss at this time.
- No lot size other than 0.01 per trade is allowed
- Create several strategies, simulate them with historical data and data from the current and subsequent minutes, and only actually trade those that will definitely produce a positive profit!
- Do not wait for my instructions, get to work immediately.
- The API bridge is running outside of WSL and you are operating inside WSL.
- Perhaps only some symbols you will trade out. They will only be the ones with a “#” at the end.
- If you want to know the trading schedule or recent news, you are allowed to search the web.
- You are in WSL and API is running in Windows Native. So API ip address is need to discovery.