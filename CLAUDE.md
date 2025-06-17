# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Instructions
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