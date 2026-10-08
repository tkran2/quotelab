# Own the project in an interview

Read the memo, rerun one scenario, and change one assumption before presenting
this as work you can explain. The tests and data provide specific claims to defend.

## A 45-second explanation

"I studied when market makers should quote more aggressively. I built a synthetic
event-driven market with ordinary flow, informed traders, jumps, and transaction
costs. I used identical exogenous market tapes to compare fixed spreads, inventory
skew, volatility widening, and their combination. The main finding was that
inventory control and adverse-selection control solve different problems: skew
reduced exposure but did not stop toxic fills. I also compared development-selected
widths and showed a failure case where slower refresh made the combined policy lose."

Use your own wording. Be clear about which parts you understand and which you
would need to investigate further.

## Questions you should be able to answer

**1. What does a market maker do?**

It stands ready to buy at a bid and sell at an ask. A completed round trip can
earn spread, but holding inventory exposes it to price moves and informed traders
can choose to trade only when its price is bad.

**2. Why does a long position move the reservation price down?**

With positive inventory, a price fall hurts wealth. Lowering the center makes the
ask more attractive and the bid less attractive, encouraging sells and discouraging
additional buys. In this model the shift is `-gamma * q * variance * horizon`.

**3. How does an informed buyer hurt you?**

Suppose the quote reference is $100 and your ask is $100.04. The efficient price
jumps to $100.20 before you refresh. An informed buyer pays your stale ask. You
sell a share worth $100.20 for $100.04, losing $0.16 before costs.
The decomposition is +$0.04 quote edge and -$0.20 fill repricing.

**4. Derive the P&L identity rather than memorizing it.**

Before a step wealth is `x + q*m_old`. After a price move and fills it is
`x - sum(dq*p + fee) + (q + sum(dq))*m_new`. Subtract old wealth. You get
`q*(m_new-m_old) + sum(dq*(m_new-p)) - fees`. Add and subtract the quote
reference inside each fill to split quote edge from fill repricing. Paid close
removes the residual inventory and subtracts liquidation costs.

**5. Are you counting adverse selection twice?**

No. Quote edge plus fill repricing equals execution edge. Markouts and informed
execution P&L are diagnostics, not additional deductions from total profit.
The entire component sum is checked against independently updated cash and inventory.

**6. Why not use the same RNG seed inside every strategy?**

Strategies can consume different numbers of random draws. Then later prices or
orders diverge even if the original seed is the same. QuoteLab generates immutable
tapes first and makes no policy-dependent random draws in the engine.

**7. How do you prevent look-ahead?**

The quote function receives only observed mid, inventory, variance from past
returns, and remaining time. The engine moves the efficient price afterward.
The true regime sigma and trader labels are never passed to the policy. A test
changes only future tape prices and confirms that earlier decisions do not change.

**8. Is the adaptive policy always better?**

No. At the reference width it improves several outcomes, but it also quotes wider.
After selecting one width per policy on development seeds, mean P&L is lower than
the tuned fixed policy in some regimes and several paired intervals include zero.
Its strongest advantage in these experiments is controlling inventory and drawdown.

**9. What does the volatility-only ablation tell you?**

It changes width while keeping the quote center neutral. Comparing it with fixed
helps isolate width effects; comparing it with the combined policy isolates the
additional inventory skew. In jump and burst regimes it sometimes earns more
mean P&L than the combined policy while taking substantially more inventory risk.

**10. Why not claim a large annualized Sharpe ratio?**

These are synthetic 30-minute sessions without realistic capital allocation,
carry, or an empirically validated return distribution. Annualization would add
an unsupported economic interpretation. Report dollar P&L, drawdown, inventory,
loss frequency, and paired uncertainty instead.

**11. What is the unit of statistical independence?**

A complete seed-level simulated session. Trades within the same path share price
moves and inventory and are correlated. For a strategy contrast, compute one
P&L difference per identical tape and bootstrap those differences.

**12. Why are seed holdouts useful, and what do they fail to prove?**

They test whether width selection exploits lucky simulated paths. They cannot
validate the generative market assumptions or replace a chronological real-data
holdout. Forty paths also cannot precisely estimate rare-event tail risk.

**13. When can tightening quotes increase profit?**

For ordinary flow with exponential depth, expected fill intensity declines with
distance from mid. Ignoring ticks, inventory caps, jumps, and closing costs,
profit rate is proportional to `(delta - fee)*exp(-k*delta)`. Differentiate:
the optimum is `delta = fee + 1/k`. Tightening from an overly wide quote toward
that optimum increases profit; tightening past it sacrifices too much edge.
Informed flow adds a conditional execution loss and changes this balance.

**14. Why are informed execution losses guaranteed negative here?**

The simulated informed trader knows the current efficient price and fills only
when its own edge exceeds a threshold. That definition makes the dealer's
execution edge negative. This is an explicit oracle stress assumption, not an
empirical discovery about the distribution of trader profits.

**15. What fails when refresh slows?**

Exposed quotes accumulate more price changes. The policy's inventory correction
also becomes stale between refreshes. Wider adaptive quotes help, but the toxicity
stress still becomes unprofitable at eight-step refresh. This model captures
exposure age, not actual exchange cancel/replace mechanics.

**16. What are your biggest limitations?**

No competing makers, no queue position, no price impact, oracle informed traders,
independent ordinary-order depth, and instant replenishment at old quotes. The
model can overstate profitable ordinary fills. The most useful extension would
test whether the result survives conservative queue and cancellation assumptions.

## Three hands-on exercises

1. Run a single toxic session. Find an informed fill in its ledger and calculate
   quote edge, repricing, and execution edge by hand. Match the result to the CSV.
2. Set gamma to zero and run the dashboard. Explain why the combined policy then
   behaves like volatility widening and inventory-only becomes fixed quoting.
3. Run with fees $0 and $0.01 on the same tape. Prove the P&L difference equals
   `$0.01 * (passive fills + shares liquidated)` because fees do not change decisions.

## A concrete extension that would be yours

Implement finite displayed quote size and a replenishment delay. Predict how this
changes the toxic/benign P&L tradeoff before running the code. Use new development
and evaluation seeds, keep the old benchmark, and write a one-page update covering
what changed, what failed, and what the revised evidence supports.

