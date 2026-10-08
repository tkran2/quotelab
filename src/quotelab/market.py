"""Generate an immutable exogenous tape BEFORE a strategy is evaluated.

Named independent random streams make policy comparisons use common random
numbers. No random draws are made by the execution engine after a policy acts.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from .config import MarketConfig


@dataclass(frozen=True)
class EventTape:
    mid: np.ndarray
    offsets: np.ndarray  # event range for step t: [offsets[t], offsets[t+1])
    taker_side: np.ndarray  # +1 = taker buys, -1 = taker sells
    informed: np.ndarray
    depth: np.ndarray
    sigma_path: np.ndarray  # diagnostic only, never supplied to quoting policy
    seed: int

    def __post_init__(self):
        n = len(self.mid) - 1
        if n < 1 or len(self.offsets) != n + 1 or len(self.sigma_path) != n:
            raise ValueError("inconsistent tape lengths")
        count = len(self.taker_side)
        if len(self.informed) != count or len(self.depth) != count:
            raise ValueError("inconsistent event arrays")
        if self.offsets[0] != 0 or self.offsets[-1] != count:
            raise ValueError("offset boundaries do not match event count")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be nondecreasing")
        if not np.isin(self.taker_side, [-1, 1]).all():
            raise ValueError("taker_side must be +1 or -1")
        if not np.isfinite(self.mid).all() or np.any(self.mid <= 0):
            raise ValueError("arithmetic price model produced a nonpositive/invalid price")
        if not np.isfinite(self.depth).all() or np.any(self.depth < 0):
            raise ValueError("depth must be finite and nonnegative")
        for array in [self.mid, self.offsets, self.taker_side, self.informed,
                      self.depth, self.sigma_path]:
            array.flags.writeable = False

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in [self.mid, self.offsets, self.taker_side, self.informed, self.depth]:
            h.update(a.tobytes())
        return h.hexdigest()


def generate_tape(config: MarketConfig, seed: int) -> EventTape:
    price_rng, noise_rng, informed_rng, ordering_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    n = config.steps
    sigma = np.full(n, config.sigma)
    if config.burst_sigma is not None:
        sigma[n // 3: 2 * n // 3] = config.burst_sigma
    increments = sigma * np.sqrt(config.dt) * price_rng.standard_normal(n)
    jump_counts = price_rng.poisson(config.jump_rate * config.dt, n)
    # Conditional on K independent normal jumps, their sum is N(0, K * jump_sigma^2).
    increments += config.jump_sigma * np.sqrt(jump_counts) * price_rng.standard_normal(n)
    if config.forced_shock:
        increments[n // 2] += config.forced_shock * price_rng.choice([-1, 1])
    mid = np.r_[config.start_price, config.start_price + np.cumsum(increments)]

    noise_counts = noise_rng.poisson(2 * config.noise_rate_per_side * config.dt, n)
    info_counts = informed_rng.poisson(config.informed_rate * config.dt, n)
    noise_steps = np.repeat(np.arange(n), noise_counts)
    info_steps = np.repeat(np.arange(n), info_counts)
    count_noise, count_info = len(noise_steps), len(info_steps)
    event_steps = np.r_[noise_steps, info_steps]
    side = np.r_[noise_rng.choice([-1, 1], count_noise),
                 informed_rng.choice([-1, 1], count_info)].astype(np.int8)
    informed = np.r_[np.zeros(count_noise, dtype=bool), np.ones(count_info, dtype=bool)]
    depth = np.r_[noise_rng.exponential(config.depth_mean, count_noise), np.zeros(count_info)]
    # Independent within-step arrival priority. Both sides compete for the same risk budget.
    order = np.lexsort((ordering_rng.random(len(event_steps)), event_steps))
    offsets = np.r_[0, np.cumsum(noise_counts + info_counts)].astype(np.int64)
    return EventTape(mid, offsets, side[order], informed[order], depth[order], sigma, seed)

