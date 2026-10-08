# QuoteLab project entry

Use these bullets after reviewing the code, reproducing a run, and being able to
derive its accounting identity. Do not describe this as a real-market backtest or
an optimal-control solver. The local project is ready to publish to a repository;
no repository or hosted demo URL has been created by this delivery.

**QuoteLab | Inventory-Aware Market-Making Research**  
Python, NumPy, pandas, Monte Carlo simulation, market microstructure

- Built an event-driven market-making simulator with informed order flow, jump
  shocks, inventory limits, transaction costs, and exact P&L attribution; validated
  accounting and causal execution with 33 automated tests.
- Evaluated four quoting policies over 12,580 simulated sessions using shared
  market tapes, six regimes, held-out seeds, and paired bootstrap intervals;
  isolated inventory and volatility effects through feature ablations.
- Demonstrated how quote aggressiveness and refresh speed change adverse selection,
  fill volume, and drawdown; compared development-tuned baselines and documented
  regimes where adaptive quoting underperformed.

For a quant application, place this project first. CacheForge is a strong second
project for quant development because it supplies systems and concurrency evidence.
Keep FilingLens as the third if space allows. Replace an existing general web-app
project to keep the resume to one page.

