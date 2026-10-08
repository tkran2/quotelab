"""Seed-level uncertainty; fills on the same path are NOT independent trials."""

import numpy as np
import pandas as pd


def bootstrap_mean_ci(values, seed: int = 20261007, draws: int = 3000):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return (float(a[0]), float(a[0])) if len(a) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        lo, hi = bootstrap_mean_ci(group.net_pnl)
        row.update(n_seeds=len(group), mean_pnl=group.net_pnl.mean(), pnl_ci_low=lo,
                   pnl_ci_high=hi, std_pnl=group.net_pnl.std(ddof=1),
                   loss_rate=(group.net_pnl < 0).mean(),
                   pnl_p05=group.net_pnl.quantile(0.05),
                   mean_max_drawdown=group.max_drawdown.mean(),
                   mean_abs_inventory=group.avg_abs_inventory.mean(),
                   mean_fills=group.fills.mean(),
                   mean_informed_fraction=group.informed_fill_fraction.mean())
        for column in ["quote_edge", "fill_repricing", "inventory_pnl", "execution_fees",
                       "liquidation_cost", "noise_execution_pnl", "informed_execution_pnl",
                       "cap_time_fraction", "accounting_residual"]:
            row[f"mean_{column}"] = group[column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(frame: pd.DataFrame, candidate: str, baseline: str = "fixed"):
    """Pair by scenario and tape seed before bootstrapping candidate - baseline."""
    rows = []
    for scenario, group in frame.groupby("scenario", sort=True):
        table = group.pivot(index="seed", columns="policy", values="net_pnl")
        pairs = table[[candidate, baseline]].dropna()
        difference = pairs[candidate] - pairs[baseline]
        low, high = bootstrap_mean_ci(difference)
        rows.append({"scenario": scenario, "candidate": candidate, "baseline": baseline,
                     "n_pairs": len(pairs), "mean_delta_pnl": difference.mean(),
                     "delta_ci_low": low, "delta_ci_high": high,
                     "win_fraction": (difference > 0).mean()})
    return pd.DataFrame(rows)

