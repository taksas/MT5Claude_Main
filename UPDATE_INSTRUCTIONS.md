# Update Instructions

Apply the following tasks to the entire existing trading engine.
**At this time, do not change any of the trading logic or signals that govern the trading strategy.**
Break down each task into smaller sub-tasks and apply the corrections with `ultrathink`.

---

## Task 1: Only Tradable Symbols

You are to trade only the following symbols.

Other symbols are either not tradable or unsuitable for trading due to excessively wide spreads.

```
CADJPY#, CHFJPY#, EURCAD#, EURCHF#, EURGBP#, EURJPY#, CADCHF#, EURUSD#, USDJPY#, GBPCAD#, GBPCHF#, GBPJPY#, GBPUSD#, USDCAD#, USDCHF#
```

---

## Task 2: Fix Stop Loss Rate Problem

You are setting the stop loss too low.
The brokerage account on which this trading system operates has a mandatory liquidation (margin call) at a 20% margin maintenance level.
You are frequently setting the stop loss at a level even lower than this, which means there is a substantial risk of losing almost all of your capital in a single trade.

> This is unacceptable.

Set the stop loss to incur a smaller loss, and accordingly, only take positions when you have sufficient confidence.

---

## Task 3: Fix Too Large Trading Lot

You are almost always using nearly all of your capital to trade with the maximum lot size.
This carries a high risk of incurring a large loss in a single trade and also makes it difficult to hold positions in multiple symbols simultaneously.

> This is unacceptable.

Please improve this.
