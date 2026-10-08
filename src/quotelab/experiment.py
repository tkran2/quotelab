"""Prespecified sweeps with disjoint development and evaluation seeds."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd

from .config import POLICIES, SPREAD_GRID, MarketConfig, StrategyConfig, scenarios
from .engine import simulate
from .market import generate_tape
from .metrics import paired_comparison, summarize


def _batch(job):
    market, seed, spreads, refreshes, split, experiment = job
    tape = generate_tape(market, seed)
    rows = []
    for refresh in refreshes:
        for spread in spreads:
            for policy in POLICIES:
                strategy = StrategyConfig(policy=policy, half_spread=spread, refresh_steps=refresh)
                result = simulate(market, strategy, tape=tape, record=False)
                rows.append({**result.metrics, "split": split, "experiment": experiment})
    return rows


def run_research(output: Path, steps: int = 1800, train_seeds: int = 30,
                 test_seeds: int = 40, sensitivity_seeds: int = 25, workers: int = 1):
    if min(train_seeds, test_seeds, sensitivity_seeds) < 2:
        raise ValueError("research requires at least two seeds per split")
    if max(train_seeds, test_seeds, sensitivity_seeds) > 10000:
        raise ValueError("seed counts must not overlap the reserved split ranges")
    output.mkdir(parents=True, exist_ok=True)
    regime = scenarios(steps)
    train = list(range(10000, 10000 + train_seeds))
    test = list(range(20000, 20000 + test_seeds))
    stress = list(range(30000, 30000 + sensitivity_seeds))
    jobs = []
    for market in regime.values():
        for split, seeds in [("development", train), ("held_out", test)]:
            jobs.extend((market, s, SPREAD_GRID, (1,), split, "main") for s in seeds)
    # Hold total potential arrival intensity fixed while replacing noise with informed flow.
    for fraction in (0.0, 0.15, 0.4, 0.7):
        market = MarketConfig(
            name=f"toxicity_{fraction:.2f}", steps=steps, sigma=0.06,
            noise_rate_per_side=1.2 * (1 - fraction), informed_rate=2.4 * fraction,
            jump_rate=0.003, jump_sigma=0.35,
        )
        jobs.extend((market, s, (0.01, 0.04, 0.09, 0.14), (1,), "held_out", "toxicity")
                    for s in stress)
    for name in ("toxic", "volatility_burst", "jump_shock"):
        jobs.extend((regime[name], s, (0.04,), (1, 3, 8), "held_out", "refresh") for s in stress)

    start = time.perf_counter()
    rows = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        batches = executor.map(_batch, jobs) if executor else map(_batch, jobs)
        for i, batch in enumerate(batches, 1):
            rows.extend(batch)
            if i % 40 == 0 or i == len(jobs):
                print(f"Completed {i}/{len(jobs)} market tapes; {len(rows):,} sessions", flush=True)
    finally:
        if executor:
            executor.shutdown()
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "runs.csv", index=False)
    main = raw[raw.experiment == "main"]
    summary = summarize(main, ["scenario", "split", "policy", "half_spread"])
    summary.to_csv(output / "spread_summary.csv", index=False)
    reference = main[(main.split == "held_out") & (main.half_spread == 0.04)]
    summarize(reference, ["scenario", "policy"]).to_csv(output / "reference_summary.csv", index=False)
    paired = pd.concat([paired_comparison(reference, p) for p in POLICIES if p != "fixed"])
    paired.to_csv(output / "paired_deltas.csv", index=False)
    for experiment in ("toxicity", "refresh"):
        keys = ["scenario", "policy", "half_spread", "refresh_steps"]
        summarize(raw[raw.experiment == experiment], keys).to_csv(
            output / f"{experiment}_summary.csv", index=False)

    # Lock ONE width per policy using development seeds, equal scenario weights.
    # The same inventory parameters are prespecified and never searched on held-out seeds.
    dev = summary[summary.split == "development"].copy()
    dev["objective"] = dev.mean_pnl - 0.25 * dev.mean_max_drawdown
    objective = dev.groupby(["policy", "half_spread"], as_index=False).objective.mean()
    selected = objective.loc[objective.groupby("policy").objective.idxmax()].copy()
    selected.to_csv(output / "selected_parameters.csv", index=False)
    chosen = main[main.split == "held_out"].merge(selected[["policy", "half_spread"]],
                                                    on=["policy", "half_spread"])
    summarize(chosen, ["scenario", "policy", "half_spread"]).to_csv(
        output / "selected_heldout_summary.csv", index=False)
    pd.concat([paired_comparison(chosen, p) for p in POLICIES if p != "fixed"]).to_csv(
        output / "selected_paired_deltas.csv", index=False)

    demos = {}
    for name, market in regime.items():
        tape = generate_tape(market, 20000)
        demos[name] = {}
        for policy in POLICIES:
            result = simulate(market, StrategyConfig(policy=policy), tape=tape)
            demos[name][policy] = result.to_payload()
            result.path.to_csv(output / f"path_{name}_{policy}.csv", index=False)
            result.trades.to_csv(output / f"trades_{name}_{policy}.csv", index=False)
    (output / "demo_paths.json").write_text(json.dumps(demos, allow_nan=False), encoding="utf-8")
    manifest = {
        "project_version": "1.0.0", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "steps": steps,
        "train_seeds": train, "test_seeds": test, "sensitivity_seeds": stress,
        "spread_grid": SPREAD_GRID, "policy_defaults": asdict(StrategyConfig()),
        "scenarios": {name: asdict(m) for name, m in regime.items()},
        "n_sessions": len(raw), "n_main_sessions": len(main), "n_market_tapes": len(jobs),
        "bootstrap_draws": 3000, "bootstrap_seed": 20261007,
        "selection_objective": "equal-scenario mean P&L minus 0.25 * mean maximum drawdown",
        "maximum_accounting_residual": float(raw.accounting_residual.max()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(raw):,} sessions in {manifest['elapsed_seconds']:.1f}s to {output}", flush=True)
    return manifest

