"""All prices and fees are dollars per share; time is measured in seconds."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MarketConfig:
    name: str = "benign"
    steps: int = 1800
    dt: float = 1.0
    start_price: float = 100.0
    sigma: float = 0.02  # dollars / sqrt(second)
    noise_rate_per_side: float = 1.2  # potential orders / second / side
    informed_rate: float = 0.03  # total potential informed orders / second
    depth_mean: float = 0.10  # exponential ordinary-order willingness-to-pay depth
    jump_rate: float = 0.0005  # Poisson jump intensity / second
    jump_sigma: float = 0.20
    informed_edge: float = 0.005  # informed trader's minimum edge
    maker_fee: float = 0.003
    liquidation_half_spread: float = 0.01
    liquidation_slippage: float = 0.005
    burst_sigma: float | None = None
    forced_shock: float = 0.0  # symmetric random-sign jump at session midpoint

    def __post_init__(self):
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        for name, value in asdict(self).items():
            if name in {"name", "steps"} or value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt <= 0 or self.start_price <= 0 or self.depth_mean <= 0:
            raise ValueError("dt, start_price, and depth_mean must be positive")


@dataclass(frozen=True)
class StrategyConfig:
    policy: str = "adaptive"
    half_spread: float = 0.04
    risk_aversion: float = 0.12  # inverse dollars
    inventory_horizon: float = 40.0  # seconds, capped by remaining session time
    volatility_buffer: float = 0.8
    ewma_alpha: float = 0.04
    initial_sigma: float = 0.025  # common prior; never initialized to future regime sigma
    inventory_limit: int = 30
    refresh_steps: int = 1  # quotes remain exposed to at least one new price innovation
    tick: float = 0.01

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        for name in ["half_spread", "risk_aversion", "inventory_horizon",
                     "volatility_buffer", "initial_sigma", "tick"]:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.tick <= 0 or self.half_spread <= 0:
            raise ValueError("tick and half_spread must be positive")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not isinstance(self.inventory_limit, int) or self.inventory_limit < 1:
            raise ValueError("inventory_limit must be a positive integer")
        if not isinstance(self.refresh_steps, int) or self.refresh_steps < 1:
            raise ValueError("refresh_steps must be a positive integer")


POLICIES = ("fixed", "inventory", "volatility", "adaptive")
POLICY_LABELS = {
    "fixed": "Fixed spread",
    "inventory": "Inventory skew",
    "volatility": "Volatility widening",
    "adaptive": "Inventory + volatility",
}
SPREAD_GRID = (0.01, 0.025, 0.04, 0.06, 0.09, 0.14)


def scenarios(steps: int = 1800) -> dict[str, MarketConfig]:
    """Prespecified synthetic stress regimes, deliberately not fitted to real data."""
    return {
        "benign": MarketConfig(steps=steps),
        "toxic": MarketConfig(
            name="toxic", steps=steps, sigma=0.09, noise_rate_per_side=0.4,
            informed_rate=1.8, jump_rate=0.004, jump_sigma=0.45,
        ),
        "volatility_burst": MarketConfig(
            name="volatility_burst", steps=steps, sigma=0.025, burst_sigma=0.11,
            noise_rate_per_side=1.0, informed_rate=0.5,
            jump_rate=0.001, jump_sigma=0.3,
        ),
        "jump_shock": MarketConfig(
            name="jump_shock", steps=steps, sigma=0.03, noise_rate_per_side=0.8,
            informed_rate=0.6, jump_rate=0.003, jump_sigma=0.65, forced_shock=1.5,
        ),
        "thin_market": MarketConfig(
            name="thin_market", steps=steps, sigma=0.05, noise_rate_per_side=0.15,
            informed_rate=0.5, jump_rate=0.002, jump_sigma=0.4,
        ),
        "high_cost": MarketConfig(
            name="high_cost", steps=steps, maker_fee=0.025,
            liquidation_half_spread=0.02, liquidation_slippage=0.01,
        ),
    }

