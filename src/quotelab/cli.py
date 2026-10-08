"""CLI for individual simulations, the prespecified research suite, and dashboard."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import POLICIES, StrategyConfig, scenarios
from .engine import simulate


def main():
    parser = argparse.ArgumentParser(prog="quotelab")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("simulate", help="run one auditable session")
    one.add_argument("--scenario", choices=list(scenarios()), default="toxic")
    one.add_argument("--policy", choices=POLICIES, default="adaptive")
    one.add_argument("--seed", type=int, default=20000)
    one.add_argument("--steps", type=int, default=1800)
    one.add_argument("--half-spread", type=float, default=0.04)
    one.add_argument("--refresh", type=int, default=1)
    one.add_argument("--fee", type=float)
    one.add_argument("--output", type=Path, default=Path("results/single_run"))
    research = sub.add_parser("research", help="run all experiments with disjoint seeds")
    research.add_argument("--output", type=Path, default=Path("results"))
    research.add_argument("--steps", type=int, default=1800)
    research.add_argument("--train-seeds", type=int, default=30)
    research.add_argument("--test-seeds", type=int, default=40)
    research.add_argument("--sensitivity-seeds", type=int, default=25)
    research.add_argument("--workers", type=int, default=1)
    dashboard = sub.add_parser("dashboard", help="serve an interactive local research dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.command == "simulate":
        market = scenarios(args.steps)[args.scenario]
        if args.fee is not None:
            market = replace(market, maker_fee=args.fee)
        strategy = StrategyConfig(policy=args.policy, half_spread=args.half_spread,
                                  refresh_steps=args.refresh)
        result = simulate(market, strategy, args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        result.path.to_csv(args.output / "path.csv", index=False)
        result.trades.to_csv(args.output / "trades.csv", index=False)
        (args.output / "run.json").write_text(json.dumps(result.to_payload(), allow_nan=False))
        print(json.dumps(result.to_payload()["metrics"], indent=2))
    elif args.command == "research":
        from .experiment import run_research
        run_research(args.output, args.steps, args.train_seeds, args.test_seeds,
                     args.sensitivity_seeds, args.workers)
    else:
        from .server import serve
        serve(args.port, args.results)

