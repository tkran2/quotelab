"""A loopback-only dashboard and bounded simulation endpoint, no cloud services."""

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json

import pandas as pd

from .config import POLICIES, StrategyConfig, scenarios
from .engine import simulate
from .market import generate_tape


def json_records(path):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.astype(object).where(pd.notnull(df), None).to_dict("records")


def research_payload(results: Path):
    manifest = results / "manifest.json"
    demos = results / "demo_paths.json"
    data = {
        "manifest": json.loads(manifest.read_text()) if manifest.exists() else {},
        "demos": json.loads(demos.read_text()) if demos.exists() else {},
        "spread": json_records(results / "spread_summary.csv"),
        "reference": json_records(results / "reference_summary.csv"),
        "paired": json_records(results / "paired_deltas.csv"),
        "selected": json_records(results / "selected_parameters.csv"),
        "selected_heldout": json_records(results / "selected_heldout_summary.csv"),
        "toxicity": json_records(results / "toxicity_summary.csv"),
        "refresh": json_records(results / "refresh_summary.csv"),
    }
    # Trade-level audit files remain in results/. Only paths are needed by this view.
    for policies in data["demos"].values():
        for result in policies.values():
            result.pop("trades", None)
    return data


def dashboard_html(results: Path, live: bool = False):
    template = (Path(__file__).parent / "web" / "dashboard.html").read_text(encoding="utf-8")
    payload = research_payload(results)
    payload["live"] = live
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("__QUOTELAB_DATA__", encoded)


def simulate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    name = payload.get("scenario", "toxic")
    if name not in scenarios():
        raise ValueError("unknown market scenario")
    def integer(key, default, low, high):
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        return value
    def number(key, default, low, high):
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not low <= value <= high:
            raise ValueError(f"{key} must be a number from {low} to {high}")
        return float(value)
    steps = integer("steps", 1800, 20, 10000)
    seed = integer("seed", 20000, 0, 2**32 - 1)
    refresh = integer("refresh_steps", 1, 1, 50)
    limit = integer("inventory_limit", 30, 1, 200)
    half = number("half_spread", .04, .005, 2)
    gamma = number("risk_aversion", .12, 0, 2)
    buffer = number("volatility_buffer", .8, 0, 5)
    fee = number("maker_fee", scenarios()[name].maker_fee, 0, .25)
    market = replace(scenarios(steps)[name], maker_fee=fee)
    tape = generate_tape(market, seed)
    results = {}
    for policy in POLICIES:
        config = StrategyConfig(policy=policy, half_spread=half, risk_aversion=gamma,
                                volatility_buffer=buffer, inventory_limit=limit, refresh_steps=refresh)
        result = simulate(market, config, tape=tape).to_payload()
        result.pop("trades", None)
        results[policy] = result
    return {"scenario": name, "seed": seed, "tape_sha256": tape.fingerprint(), "results": results}


def serve(port: int, results: Path):
    content = dashboard_html(results, live=True).encode()
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, content, "text/html")
            elif self.path == "/health":
                self.respond(200, b'{"status":"ok"}')
            else:
                self.respond(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/api/simulate":
                self.respond(404, b'{"error":"not found"}')
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.respond(415, b'{"error":"send application/json"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                result = simulate_request(payload)
                self.respond(200, json.dumps(result, allow_nan=False).encode())
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, json.dumps({"error": str(error)}).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"QuoteLab dashboard: http://127.0.0.1:{port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

