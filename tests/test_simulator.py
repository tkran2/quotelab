"""Hand-calculated economics, statistical checks, and leakage/inventory invariants."""

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from quotelab.config import POLICIES, MarketConfig, StrategyConfig, scenarios
from quotelab.engine import simulate
from quotelab.market import EventTape, generate_tape
from quotelab.metrics import bootstrap_mean_ci, paired_comparison
from quotelab.strategy import make_quote


def tape(mid, events):
    """Events are lists per step: (taker side, informed, noise depth)."""
    flat = [e for row in events for e in row]
    offsets = np.r_[0, np.cumsum([len(row) for row in events])].astype(np.int64)
    return EventTape(np.array(mid, dtype=float), offsets,
                     np.array([e[0] for e in flat], dtype=np.int8),
                     np.array([e[1] for e in flat], dtype=bool),
                     np.array([e[2] for e in flat], dtype=float),
                     np.zeros(len(mid) - 1), 0)


def free_market(steps):
    return MarketConfig(steps=steps, maker_fee=0, liquidation_half_spread=0,
                        liquidation_slippage=0)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.policy = StrategyConfig(policy="fixed", half_spread=0.04)

    def test_one_buy_hand_calculation(self):
        result = simulate(free_market(1), self.policy, tape=tape([100, 100], [[(-1, False, .2)]]))
        self.assertAlmostEqual(result.metrics["net_pnl"], .04)
        self.assertEqual(result.trades.dealer_side.tolist(), [1])
        self.assertEqual(result.metrics["terminal_inventory"], 0)

    def test_one_sell_hand_calculation(self):
        result = simulate(free_market(1), self.policy, tape=tape([100, 100], [[(1, False, .2)]]))
        self.assertAlmostEqual(result.metrics["net_pnl"], .04)
        self.assertEqual(result.trades.dealer_side.tolist(), [-1])

    def test_round_trip_spread_less_fees(self):
        m = replace(free_market(1), maker_fee=.003)
        r = simulate(m, self.policy, tape=tape([100, 100], [[(-1, False, .2), (1, False, .2)]]))
        self.assertAlmostEqual(r.metrics["net_pnl"], .08 - .006)
        self.assertAlmostEqual(r.metrics["liquidation_cost"], 0)

    def test_informed_buyer_lifts_stale_ask(self):
        r = simulate(free_market(1), self.policy, tape=tape([100, 100.2], [[(1, True, 0)]]))
        self.assertAlmostEqual(r.metrics["net_pnl"], -.16)
        self.assertAlmostEqual(r.metrics["quote_edge"], .04)
        self.assertAlmostEqual(r.metrics["fill_repricing"], -.20)
        self.assertEqual(r.metrics["informed_fills"], 1)

    def test_informed_seller_hits_stale_bid(self):
        r = simulate(free_market(1), self.policy, tape=tape([100, 99.8], [[(-1, True, 0)]]))
        self.assertAlmostEqual(r.metrics["net_pnl"], -.16)

    def test_informed_trader_skips_unprofitable_quote(self):
        r = simulate(free_market(1), self.policy,
                     tape=tape([100, 100], [[(1, True, 0), (-1, True, 0)]]))
        self.assertEqual(r.metrics["fills"], 0)
        self.assertEqual(r.metrics["net_pnl"], 0)

    def test_existing_inventory_marks_before_new_fills(self):
        r = simulate(free_market(2), self.policy,
                     tape=tape([100, 100, 101], [[(-1, False, .2)], []]))
        self.assertAlmostEqual(r.metrics["inventory_pnl"], 1)
        self.assertAlmostEqual(r.metrics["net_pnl"], 1.04)

    def test_terminal_liquidation_is_not_free(self):
        m = MarketConfig(steps=1, maker_fee=.003,
                         liquidation_half_spread=.01, liquidation_slippage=.005)
        r = simulate(m, self.policy, tape=tape([100, 100], [[(-1, False, .2)]]))
        self.assertAlmostEqual(r.metrics["liquidation_cost"], .018)
        self.assertAlmostEqual(r.metrics["net_pnl"], .04 - .003 - .018)
        self.assertEqual(r.path.inventory.iloc[-1], 0)
        self.assertAlmostEqual(r.path.cash.iloc[-1], r.path.wealth.iloc[-1])

    def test_fee_increase_has_exact_turnover_effect(self):
        m = free_market(50)
        shared = generate_tape(m, 123)
        r1 = simulate(m, self.policy, tape=shared)
        r2 = simulate(replace(m, maker_fee=.01), self.policy, tape=shared)
        turns = r1.metrics["fills"] + abs(r1.metrics["terminal_inventory_before_close"])
        self.assertEqual(r1.metrics["fills"], r2.metrics["fills"])
        self.assertAlmostEqual(r1.metrics["net_pnl"] - r2.metrics["net_pnl"], .01 * turns)

    def test_markouts_use_future_price_after_execution(self):
        m = free_market(3)
        r = simulate(m, self.policy,
                     tape=tape([100, 100, 100.2, 99.7], [[(-1, False, .2)], [], []]))
        self.assertAlmostEqual(r.trades.markout_1s.iloc[0], .24)
        self.assertTrue(np.isnan(r.trades.markout_5s.iloc[0]))
        self.assertEqual(r.metrics["noise_markout_5s_count"], 0)

    def test_drawdown_includes_intra_step_adverse_fill(self):
        r = simulate(free_market(1), self.policy,
                     tape=tape([100, 100.2], [[(1, True, 0), (-1, False, .2)]]))
        self.assertGreaterEqual(r.metrics["max_drawdown"], .16 - 1e-9)

    def test_wealth_reconciles_at_every_recorded_time(self):
        for p in POLICIES:
            for seed in range(5):
                r = simulate(scenarios(200)["jump_shock"], StrategyConfig(policy=p), seed)
                expected = (r.path.quote_edge + r.path.fill_repricing + r.path.inventory_pnl
                            - r.path.execution_fees - r.path.liquidation_cost)
                np.testing.assert_allclose(r.path.wealth, expected, atol=1e-8)
                self.assertLess(r.metrics["accounting_residual"], 1e-8)


class RiskAndCausalityTests(unittest.TestCase):
    def test_cap_checked_for_every_fill_including_same_step(self):
        events = [[(-1, False, .5)] * 100 + [(1, False, .5)] * 100]
        r = simulate(free_market(1), StrategyConfig(policy="fixed", inventory_limit=3),
                     tape=tape([100, 100], events))
        self.assertLessEqual(r.trades.inventory_after.abs().max(), 3)
        self.assertGreater(r.metrics["inventory_blocked_orders"], 0)

    def test_inventory_hedging_direction(self):
        s = StrategyConfig(policy="inventory")
        neutral = make_quote(s, 100, 0, .000625, 40)
        long = make_quote(s, 100, 10, .000625, 40)
        short = make_quote(s, 100, -10, .000625, 40)
        self.assertLess(long.reservation_price, neutral.reservation_price)
        self.assertLess(long.ask, neutral.ask)
        self.assertGreater(short.bid, neutral.bid)

    def test_adaptive_volatility_widens_quotes(self):
        s = StrategyConfig()
        quiet = make_quote(s, 100, 0, .0001, 40)
        volatile = make_quote(s, 100, 0, .01, 40)
        self.assertGreater(volatile.ask - volatile.bid, quiet.ask - quiet.bid)

    def test_volatility_only_ablation_keeps_neutral_center(self):
        s = StrategyConfig(policy="volatility")
        q = make_quote(s, 100, 20, .01, 40)
        self.assertEqual(q.reservation_price, 100)
        self.assertGreater(q.ask - q.bid, .08)

    def test_ticks_and_passivity(self):
        for q in (-30, 0, 30):
            quote = make_quote(StrategyConfig(), 100.003, q, .1, 40)
            self.assertLess(quote.bid, 100.003)
            self.assertGreater(quote.ask, 100.003)
            self.assertAlmostEqual(quote.bid / .01, round(quote.bid / .01))
            self.assertAlmostEqual(quote.ask / .01, round(quote.ask / .01))

    def test_refresh_interval_really_keeps_stale_quotes(self):
        shared = tape([100, 100.1, 100.2, 100.3, 100.4], [[], [], [], []])
        r = simulate(free_market(4), StrategyConfig(policy="fixed", refresh_steps=3), tape=shared)
        self.assertEqual(r.path.ask.iloc[1], r.path.ask.iloc[2])
        self.assertEqual(r.path.ask.iloc[2], r.path.ask.iloc[3])
        self.assertNotEqual(r.path.ask.iloc[3], r.path.ask.iloc[4])

    def test_future_changes_cannot_change_prefix_decisions(self):
        m = scenarios(300)["volatility_burst"]
        a = generate_tape(m, 95)
        changed_mid = a.mid.copy()
        changed_mid[151:] += np.linspace(.1, 20, len(changed_mid) - 151)
        b = EventTape(changed_mid, a.offsets.copy(), a.taker_side.copy(), a.informed.copy(),
                      a.depth.copy(), a.sigma_path.copy(), 95)
        for policy in POLICIES:
            r1 = simulate(m, StrategyConfig(policy=policy), tape=a)
            r2 = simulate(m, StrategyConfig(policy=policy), tape=b)
            pd.testing.assert_frame_equal(r1.path.iloc[:151], r2.path.iloc[:151])

    def test_same_seed_same_tape(self):
        m = MarketConfig(steps=100)
        self.assertEqual(generate_tape(m, 8).fingerprint(), generate_tape(m, 8).fingerprint())
        self.assertNotEqual(generate_tape(m, 8).fingerprint(), generate_tape(m, 9).fingerprint())

    def test_policy_evaluation_does_not_mutate_tape(self):
        m = MarketConfig(steps=100)
        shared = generate_tape(m, 8)
        before = shared.fingerprint()
        for policy in POLICIES:
            simulate(m, StrategyConfig(policy=policy), tape=shared)
        self.assertEqual(before, shared.fingerprint())
        with self.assertRaises(ValueError):
            shared.mid[0] = 99

    def test_common_prior_does_not_use_true_regime_sigma(self):
        a = make_quote(StrategyConfig(), 100, 0, .025**2, 40)
        r = simulate(scenarios(5)["toxic"], StrategyConfig(), 8)
        self.assertEqual(r.path.ask.iloc[1], a.ask)

    def test_summary_and_recorded_modes_agree(self):
        m = scenarios(100)["toxic"]
        tape_ = generate_tape(m, 7)
        a = simulate(m, StrategyConfig(), tape=tape_, record=True)
        b = simulate(m, StrategyConfig(), tape=tape_, record=False)
        for k in a.metrics:
            if isinstance(a.metrics[k], float):
                if np.isnan(a.metrics[k]):
                    self.assertTrue(np.isnan(b.metrics[k]))
                else:
                    self.assertAlmostEqual(a.metrics[k], b.metrics[k])
            else:
                self.assertEqual(a.metrics[k], b.metrics[k])


class StatisticalAndInterfaceTests(unittest.TestCase):
    def test_static_noise_fill_rate_matches_exponential_model(self):
        m = MarketConfig(steps=8000, sigma=0, informed_rate=0, jump_rate=0,
                         noise_rate_per_side=1, maker_fee=0)
        s = StrategyConfig(policy="fixed", half_spread=.04, inventory_limit=100000)
        r = simulate(m, s, 73, record=False)
        expected = 2 * m.steps * np.exp(-.04 / m.depth_mean)
        self.assertLess(abs(r.metrics["fills"] - expected), 6 * np.sqrt(expected))

    def test_no_fill_json_uses_null_not_nan(self):
        m = MarketConfig(steps=10, noise_rate_per_side=0, informed_rate=0)
        payload = simulate(m, StrategyConfig()).to_payload()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["metrics"]["informed_fill_fraction"])

    def test_empty_markout_interval_is_missing(self):
        self.assertTrue(np.isnan(bootstrap_mean_ci([])[0]))

    def test_bootstrap_constant_effect(self):
        lo, hi = bootstrap_mean_ci([3] * 20)
        self.assertEqual((lo, hi), (3, 3))

    def test_paired_bootstrap_removes_shared_market_noise(self):
        rows = []
        for seed, noise in enumerate([-1000, -30, 0, 100, 700]):
            rows.extend([{"scenario": "s", "seed": seed, "policy": "fixed", "net_pnl": noise},
                         {"scenario": "s", "seed": seed, "policy": "adaptive", "net_pnl": noise + 7}])
        r = paired_comparison(pd.DataFrame(rows), "adaptive").iloc[0]
        self.assertEqual(r.mean_delta_pnl, 7)
        self.assertEqual(r.delta_ci_low, 7)
        self.assertEqual(r.delta_ci_high, 7)

    def test_invalid_configs_rejected(self):
        for kwargs in [{"half_spread": 0}, {"refresh_steps": 0}, {"inventory_limit": 0},
                       {"ewma_alpha": 1.5}, {"risk_aversion": float("nan")}, {"policy": "oracle"}]:
            with self.assertRaises(ValueError):
                StrategyConfig(**kwargs)
        with self.assertRaises(ValueError):
            MarketConfig(dt=0)

    def test_experiment_smoke_and_seed_separation(self):
        from quotelab.experiment import run_research
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = run_research(path, steps=25, train_seeds=2,
                                    test_seeds=2, sensitivity_seeds=2)
            self.assertFalse(set(manifest["train_seeds"]) & set(manifest["test_seeds"]))
            raw = pd.read_csv(path / "runs.csv")
            self.assertEqual(len(raw), manifest["n_sessions"])
            self.assertLess(raw.accounting_residual.max(), 1e-8)
            self.assertTrue((path / "paired_deltas.csv").exists())
            selected = pd.read_csv(path / "selected_parameters.csv")
            self.assertEqual(len(selected), 4)

    def test_dashboard_request_compares_identical_price_paths(self):
        from quotelab.server import simulate_request
        result = simulate_request({"scenario": "toxic", "steps": 50, "seed": 1234})
        self.assertEqual(set(result["results"]), set(POLICIES))
        paths = [x["path"] for x in result["results"].values()]
        for path in paths:
            self.assertEqual([x["mid"] for x in path], [x["mid"] for x in paths[0]])
        self.assertEqual(len(result["tape_sha256"]), 64)

    def test_dashboard_request_validation(self):
        from quotelab.server import simulate_request
        for payload in [[], {"steps": 999999}, {"seed": True}, {"maker_fee": float("nan")},
                        {"scenario": "made_up"}, {"refresh_steps": 1.5}]:
            with self.assertRaises(ValueError):
                simulate_request(payload)

    def test_offline_dashboard_can_embed_empty_results(self):
        from quotelab.server import dashboard_html
        with tempfile.TemporaryDirectory() as directory:
            content = dashboard_html(Path(directory))
            self.assertIn('"live":false', content)
            self.assertNotIn('__QUOTELAB_DATA__', content)


if __name__ == "__main__":
    unittest.main()

