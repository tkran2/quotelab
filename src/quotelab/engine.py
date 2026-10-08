"""Self-financing fills, inventory caps, markouts, and terminal liquidation.

    net P&L = quote edge + fill repricing + pre-fill inventory P&L
              - execution fees - liquidation costs

Quote edge is signed edge relative to the quote's reference midpoint. It includes
inventory skew and tick rounding; it must not be mislabelled as pure half-spread.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import MarketConfig, StrategyConfig
from .market import EventTape, generate_tape
from .strategy import make_quote


@dataclass
class SimulationResult:
    metrics: dict[str, Any]
    path: pd.DataFrame
    trades: pd.DataFrame
    market: MarketConfig
    strategy: StrategyConfig

    def to_payload(self) -> dict[str, Any]:
        # JSON has no NaN; sparse/no-fill cases use null, not invented zero markouts.
        def records(df):
            return df.astype(object).where(pd.notnull(df), None).to_dict("records")
        return {"metrics": {k: v if not isinstance(v, float) or np.isfinite(v) else None
                            for k, v in self.metrics.items()},
                "path": records(self.path), "trades": records(self.trades),
                "market": asdict(self.market), "strategy": asdict(self.strategy)}


TRADE_COLUMNS = ["step", "event_id", "dealer_side", "price", "reference_mid",
                 "execution_mid", "informed", "inventory_after", "fee", "quote_edge",
                 "fill_repricing", "execution_edge", "markout_1s", "markout_5s", "markout_20s"]


def simulate(market: MarketConfig, strategy: StrategyConfig, seed: int = 42,
             tape: EventTape | None = None, record: bool = True) -> SimulationResult:
    tape = generate_tape(market, seed) if tape is None else tape
    if len(tape.mid) != market.steps + 1:
        raise ValueError("market length and tape length disagree")
    n, dt = market.steps, market.dt
    mids, offsets = tape.mid, tape.offsets
    inv_limit, fee = strategy.inventory_limit, market.maker_fee
    q, cash = 0, 0.0
    quote_edge = repricing = inventory_pnl = fees = 0.0
    noise_edge = informed_edge = 0.0
    fills = info_fills = cap_blocks = eligible = 0
    peak = drawdown = max_reconciliation = 0.0
    sum_abs_q = cap_steps = peak_q = 0
    terminal_q = 0
    variance = strategy.initial_sigma ** 2
    markout_sum = np.zeros((2, 3))
    markout_count = np.zeros((2, 3), dtype=int)
    rows, trade_rows = [], []
    quote = None
    if record:
        rows.append((0.0, mids[0], 0, 0.0, 0.0, np.nan, np.nan,
                     np.sqrt(variance), 0.0, 0.0, 0.0, 0.0, 0.0))
    for t in range(n):
        if t:
            observed_return = mids[t] - mids[t - 1]
            variance = ((1 - strategy.ewma_alpha) * variance
                        + strategy.ewma_alpha * observed_return ** 2 / dt)
        if t % strategy.refresh_steps == 0:
            quote = make_quote(strategy, float(mids[t]), q, variance, (n - t) * dt, dt)
        # The efficient price moves after quotes are submitted. Only informed flow
        # knows this new value when deciding whether to trade against those quotes.
        new_mid = float(mids[t + 1])
        inventory_pnl += q * (new_mid - mids[t])
        marked_wealth = cash + q * new_mid
        peak = max(peak, marked_wealth)
        drawdown = max(drawdown, peak - marked_wealth)
        for j in range(int(offsets[t]), int(offsets[t + 1])):
            side = -int(tape.taker_side[j])  # +1 = dealer buys; -1 = dealer sells
            price = quote.bid if side == 1 else quote.ask
            is_info = bool(tape.informed[j])
            if is_info:
                willing = side * (new_mid - price) < -market.informed_edge
            elif side == 1:
                willing = price >= mids[t] - tape.depth[j]
            else:
                willing = price <= mids[t] + tape.depth[j]
            if not willing:
                continue
            eligible += 1
            if abs(q + side) > inv_limit:
                cap_blocks += 1
                continue
            # Resting one-share quotes auto-refill at their unchanged price until
            # the next scheduled refresh, subject to a per-fill inventory check.
            cash -= side * price + fee
            q += side
            marked_wealth = cash + q * new_mid
            peak = max(peak, marked_wealth)
            drawdown = max(drawdown, peak - marked_wealth)
            edge = side * (quote.reference_mid - price)
            reprice = side * (new_mid - quote.reference_mid)
            exec_edge = side * (new_mid - price)
            quote_edge += edge
            repricing += reprice
            fees += fee
            fills += 1
            info_fills += is_info
            peak_q = max(peak_q, abs(q))
            if is_info:
                informed_edge += exec_edge - fee
            else:
                noise_edge += exec_edge - fee
            markouts = []
            for h_index, seconds in enumerate((1, 5, 20)):
                horizon = max(1, int(round(seconds / dt)))
                index = t + 1 + horizon
                if index <= n:
                    value = side * (mids[index] - price)
                    markout_sum[int(is_info), h_index] += value
                    markout_count[int(is_info), h_index] += 1
                else:
                    value = np.nan  # exclude truncated horizons rather than clipping to close
                markouts.append(value)
            if record:
                trade_rows.append((t, j, side, price, quote.reference_mid, new_mid, is_info,
                                   q, fee, edge, reprice, exec_edge, *markouts))
        wealth = cash + q * new_mid
        component_value = quote_edge + repricing + inventory_pnl - fees
        max_reconciliation = max(max_reconciliation, abs(wealth - component_value))
        peak = max(peak, wealth)
        drawdown = max(drawdown, peak - wealth)
        sum_abs_q += abs(q)
        cap_steps += abs(q) == inv_limit
        if record:
            rows.append(((t + 1) * dt, new_mid, q, cash, wealth, quote.bid, quote.ask,
                         np.sqrt(variance), quote_edge, repricing, inventory_pnl, fees, 0.0))
    terminal_q = q
    liquidation_unit_cost = market.liquidation_half_spread + market.liquidation_slippage + fee
    liquidation_cost = abs(q) * liquidation_unit_cost
    liquidation_price = float(mids[-1]) - np.sign(q) * (
        market.liquidation_half_spread + market.liquidation_slippage)
    cash += q * liquidation_price - abs(q) * fee
    q = 0
    drawdown = max(drawdown, peak - cash)
    max_reconciliation = max(max_reconciliation, abs(
        cash - (quote_edge + repricing + inventory_pnl - fees - liquidation_cost)))
    if max_reconciliation > 1e-7:
        raise ArithmeticError(f"P&L identity failed: {max_reconciliation:.3g}")
    if record:
        # Same terminal timestamp, extra row: actual inventory is zero after close.
        rows.append((n * dt, mids[-1], 0, cash, cash, np.nan, np.nan, np.sqrt(variance),
                     quote_edge, repricing, inventory_pnl, fees, liquidation_cost))
    metrics = {
        "scenario": market.name, "seed": int(tape.seed), "policy": strategy.policy,
        "half_spread": strategy.half_spread, "refresh_steps": strategy.refresh_steps,
        "net_pnl": float(cash), "quote_edge": float(quote_edge),
        "fill_repricing": float(repricing), "inventory_pnl": float(inventory_pnl),
        "execution_fees": float(fees), "liquidation_cost": float(liquidation_cost),
        "max_drawdown": float(drawdown), "avg_abs_inventory": sum_abs_q / n,
        "peak_abs_inventory": peak_q, "terminal_inventory_before_close": terminal_q,
        "terminal_inventory": 0, "fills": fills, "informed_fills": info_fills,
        "informed_fill_fraction": info_fills / fills if fills else np.nan,
        "noise_execution_pnl": float(noise_edge), "informed_execution_pnl": float(informed_edge),
        "potential_orders": len(tape.taker_side), "eligible_orders": eligible,
        "inventory_blocked_orders": cap_blocks, "cap_time_fraction": cap_steps / n,
        "pnl_per_fill": cash / fills if fills else np.nan,
        "accounting_residual": float(max_reconciliation),
    }
    for k, seconds in enumerate((1, 5, 20)):
        for label, kind in [("noise", 0), ("informed", 1)]:
            c = int(markout_count[kind, k])
            metrics[f"{label}_markout_{seconds}s"] = markout_sum[kind, k] / c if c else np.nan
            metrics[f"{label}_markout_{seconds}s_count"] = c
    path = pd.DataFrame(rows, columns=["time", "mid", "inventory", "cash", "wealth", "bid",
                        "ask", "estimated_sigma", "quote_edge", "fill_repricing", "inventory_pnl",
                        "execution_fees", "liquidation_cost"])
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    return SimulationResult(metrics, path, trades, market, strategy)

