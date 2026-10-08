"""Causal quoting; this module cannot inspect a future price or a trader label."""

from dataclasses import dataclass
import math

from .config import StrategyConfig


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    reference_mid: float
    reservation_price: float
    nominal_half_spread: float


def make_quote(config: StrategyConfig, observed_mid: float, inventory: int,
               variance: float, remaining_seconds: float, dt: float = 1.0) -> Quote:
    """A-S-inspired inventory center plus an explicit volatility buffer.

    This heuristic is NOT the exact optimal control solution in the paper.
    A passive constraint and ticks make the implementation explicit and testable.
    """
    horizon = min(config.inventory_horizon, remaining_seconds)
    center, half = observed_mid, config.half_spread
    if config.policy in {"inventory", "adaptive"}:
        center -= config.risk_aversion * inventory * variance * horizon
    if config.policy in {"volatility", "adaptive"}:
        half += 0.5 * config.risk_aversion * variance * horizon
        half += config.volatility_buffer * math.sqrt(variance * config.refresh_steps * dt)
    # Do not deliberately cross the most recently observed public midpoint.
    tick = config.tick
    bid = math.floor((min(center - half, observed_mid - tick) + 1e-10) / tick) * tick
    ask = math.ceil((max(center + half, observed_mid + tick) - 1e-10) / tick) * tick
    bid = max(tick, bid)
    if bid >= ask:
        raise ValueError("invalid quote; increase starting price or reduce risk parameters")
    return Quote(bid, ask, observed_mid, center, half)

