"""Generate publication-style figures, an evidence-backed memo, and offline explorer.

Run from the project root after `python -m quotelab research`.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from quotelab.config import POLICIES, POLICY_LABELS
from quotelab.metrics import bootstrap_mean_ci
from quotelab.server import dashboard_html


RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
COLORS = {"fixed": "#477ad1", "inventory": "#c58a2d", "volatility": "#8b6bb4", "adaptive": "#008c78"}
NAMES = {"benign": "Benign", "toxic": "Toxic", "volatility_burst": "Vol. burst",
         "jump_shock": "Jump shock", "thin_market": "Thin", "high_cost": "High costs"}
ORDER = list(NAMES)


def setup_fonts():
    base = Path("/usr/share/fonts/truetype/dejavu")
    if base.exists():
        for name, filename in [("QL", "DejaVuSans.ttf"), ("QL-Bold", "DejaVuSans-Bold.ttf")]:
            pdfmetrics.registerFont(TTFont(name, str(base / filename)))
        pdfmetrics.registerFontFamily("QL", normal="QL", bold="QL-Bold", italic="QL", boldItalic="QL-Bold")
        return "QL", "QL-Bold"
    return "Helvetica", "Helvetica-Bold"


def figures(spread, reference, refresh, toxicity):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#465b70", "text.color": "#142a3d",
                         "axes.edgecolor": "#d5dee7", "xtick.color": "#647789",
                         "ytick.color": "#647789", "grid.color": "#e6ebf0",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2), constrained_layout=True)
    for row, scenario in enumerate(("benign", "toxic")):
        g = spread[(spread.scenario == scenario) & (spread.split == "held_out")]
        for policy in POLICIES:
            p = g[g.policy == policy].sort_values("half_spread")
            x = p.half_spread.to_numpy()
            axes[row, 0].plot(x, p.mean_pnl, "o-", color=COLORS[policy], ms=3, label=POLICY_LABELS[policy])
            axes[row, 0].fill_between(x, p.pnl_ci_low, p.pnl_ci_high, color=COLORS[policy], alpha=.10)
            axes[row, 1].plot(x, p.mean_max_drawdown, "o-", color=COLORS[policy], ms=3)
        axes[row, 0].set_title(f"{NAMES[scenario]} flow: spread / profit", loc="left", fontweight="bold")
        axes[row, 1].set_title(f"{NAMES[scenario]} flow: spread / drawdown", loc="left", fontweight="bold")
        for ax in axes[row]:
            ax.grid(axis="y", alpha=.7)
            ax.set_xlabel("Base half-spread ($ / share)")
        axes[row, 0].axhline(0, color="#b0bfca", ls="--", lw=.8)
        axes[row, 0].set_ylabel("Mean net P&L ($)")
        axes[row, 1].set_ylabel("Mean max drawdown ($)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(FIGURES / "frontiers.png", dpi=190)
    fig.savefig(FIGURES / "frontiers.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    fields = ["quote_edge", "fill_repricing", "inventory_pnl", "execution_fees", "liquidation_cost", "pnl"]
    labels = ["Quote\nedge", "Fill\nrepricing", "Inventory", "Fees", "Close", "Net"]
    for ax, scenario in zip(axes, ("benign", "toxic")):
        for i, policy in enumerate(("fixed", "adaptive")):
            r = reference[(reference.scenario == scenario) & (reference.policy == policy)].iloc[0]
            values = [r[f"mean_{f}"] for f in fields]
            values[3], values[4] = -values[3], -values[4]
            ax.bar(np.arange(6) + (i - .5) * .34, values, width=.32,
                   label=POLICY_LABELS[policy], color=COLORS[policy])
        ax.set_xticks(range(6), labels, fontsize=8)
        ax.set_ylabel("Mean dollars / session")
        ax.set_title(f"{NAMES[scenario]}: accounting components", loc="left", fontweight="bold")
        ax.axhline(0, color="#aebdca", lw=.8)
        ax.grid(axis="y", alpha=.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(FIGURES / "attribution.png", dpi=190)
    fig.savefig(FIGURES / "attribution.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), constrained_layout=True)
    for policy in POLICIES:
        p = refresh[(refresh.scenario == "toxic") & (refresh.policy == policy)].sort_values("refresh_steps")
        axes[0].plot(p.refresh_steps, p.mean_pnl, "o-", color=COLORS[policy], ms=3, label=POLICY_LABELS[policy])
        axes[0].fill_between(p.refresh_steps, p.pnl_ci_low, p.pnl_ci_high, color=COLORS[policy], alpha=.10)
        p = toxicity[(toxicity.policy == policy) & (toxicity.half_spread == .04)].copy()
        p["fraction"] = p.scenario.str.split("_").str[-1].astype(float)
        p = p.sort_values("fraction")
        axes[1].plot(p.fraction, p.mean_pnl, "o-", color=COLORS[policy], ms=3)
        axes[1].fill_between(p.fraction, p.pnl_ci_low, p.pnl_ci_high, color=COLORS[policy], alpha=.10)
    for ax in axes:
        ax.grid(axis="y", alpha=.6)
        ax.axhline(0, color="#aebdca", ls="--", lw=.8)
        ax.set_ylabel("Mean net P&L ($)")
    axes[0].set_title("Toxic flow: refresh stress", loc="left", fontweight="bold")
    axes[0].set_xlabel("Quote refresh interval (steps)")
    axes[0].set_xticks([1, 3, 8])
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_title("Replace ordinary with informed flow", loc="left", fontweight="bold")
    axes[1].set_xlabel("Informed share of potential arrivals")
    fig.savefig(FIGURES / "stress.png", dpi=190)
    fig.savefig(FIGURES / "stress.svg")
    plt.close(fig)


def make_memo(manifest, raw, spread, reference, paired, selected, tuned, tuned_pairs, refresh):
    font, bold = setup_fonts()
    navy, green, muted = colors.HexColor("#142a3d"), colors.HexColor("#008c78"), colors.HexColor("#647789")
    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=27, leading=32, textColor=navy, spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=20, leading=25, textColor=navy, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=11.5, leading=16, textColor=navy, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=font, fontSize=9, leading=13.7, textColor=navy, spaceAfter=8),
        "small": ParagraphStyle("small", fontName=font, fontSize=7.8, leading=11.4, textColor=muted, spaceAfter=7),
        "eyebrow": ParagraphStyle("eyebrow", fontName=bold, fontSize=8, leading=11, textColor=green, spaceAfter=9),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=8, leading=11, textColor=navy),
        "head": ParagraphStyle("head", fontName=bold, fontSize=7.5, leading=10, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=13, textColor=navy),
    }
    story = []
    def p(text, style="body"):
        story.append(Paragraph(text.replace("P&L", "P&amp;L"), styles[style]))
    def heading(title, subtitle):
        p("QUOTELAB / RESEARCH NOTE", "eyebrow")
        p(title, "h1")
        p(subtitle, "small")
    def tab(headers, rows, widths):
        data = [[Paragraph(str(x).replace("P&L", "P&amp;L"), styles["head"]) for x in headers]]
        data += [[Paragraph(str(x), styles["cell"]) for x in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), .5, navy)]))
        story.extend([t, Spacer(1, 8)])
    def figure(name, ratio, caption):
        story.append(Image(str(FIGURES / name), width=524, height=524 * ratio))
        p(caption, "small")
    def value(frame, scenario, policy, column):
        return float(frame[(frame.scenario == scenario) & (frame.policy == policy)].iloc[0][column])
    dollar = lambda x: f"{'-' if x < 0 else ''}${abs(x):,.2f}"

    p("QUANTITATIVE RESEARCH / OCTOBER 2026", "eyebrow")
    p("When does aggressive<br/>quoting pay?", "title")
    p("Inventory risk, informed flow, and the cost of stale prices", "h2")
    p(f"An event-driven synthetic study prepared for Thomas Krane. {manifest['n_sessions']:,} "
      f"sessions across {manifest['n_market_tapes']} exogenous market tapes; four policy ablations.")
    p("The useful distinction is between reducing inventory exposure and selecting better "
      "trades. Inventory skew substantially reduced risk, but at tight widths it did not "
      "remove toxic execution losses. Volatility widening changed the filled-order mix and "
      "the edge earned per trade. A tuned static baseline explains part of the apparent "
      "profit advantage of the combined policy.")
    p("Reference experiment: mean net P&amp;L", "h2")
    rows = [[NAMES[s]] + [dollar(value(reference, s, policy, "mean_pnl")) for policy in POLICIES]
            for s in ORDER]
    tab(["Regime", "Fixed", "Inventory", "Volatility", "Combined"], rows, [112, 103, 103, 103, 103])
    p(f"{len(manifest['test_seeds'])} held-out seeds per regime. Base half-spread $0.04 for each "
      f"policy; actual widths differ when the volatility feature is active. {manifest['steps'] * manifest['scenarios']['benign']['dt'] / 60:.0f}-minute "
      "sessions, one-share passive fills, $100 starting price, 30-share cap. Paid close is included.", "small")
    p("Three conclusions to defend", "h2")
    p("<b>1. Aggressiveness is conditional.</b> Tightening from an overly wide quote can "
      "increase ordinary-flow profit. Tightening further sacrifices edge and increases "
      "the share of toxic fills; fees can turn high turnover into a loss.")
    p("<b>2. Risk control is a separate result.</b> In toxic flow, inventory-only quoting "
      f"reduced average absolute inventory from {value(reference,'toxic','fixed','mean_abs_inventory'):.2f} "
      f"to {value(reference,'toxic','inventory','mean_abs_inventory'):.2f} shares, while mean net P&amp;L "
      f"remained {dollar(value(reference,'toxic','inventory','mean_pnl'))}.")
    p("<b>3. Adaptation has limits.</b> Development-selected fixed spreads narrowed the "
      "profit gap. Slower refresh and unanticipated shocks remain failure cases. These "
      "results measure behavior within the specified simulation, not a validated trading edge.")
    p("The delivered artifacts include source code, all run-level metrics, illustrative fill "
      "ledgers, a local dashboard, an offline explorer, and an interview guide.", "small")

    story.append(PageBreak())
    heading("A causal market model", "All monetary quantities are dollars per share. The delivered study uses one-second steps.")
    p("Information and execution order", "h2")
    tab(["Within step t", "What happens"], [
        ["1 / Observe", "The policy sees public mid[t], its own inventory, and prior returns. EWMA variance uses only price changes already observed."],
        ["2 / Quote", "At scheduled refreshes, bid/ask prices are rounded outward to ticks. Quotes are constrained to remain passive relative to the observed midpoint."],
        ["3 / Move and fill", "The efficient price moves to mid[t+1]. Ordinary flow uses mid[t] plus random depth; informed flow sees the new value. Independently ordered one-share events test the standing quotes."],
        ["4 / Mark and close", "Inventory is capped at each fill. Wealth is marked at the efficient price. At the session end, residual shares are liquidated with spread, slippage, and fees."],
    ], [111, 413])
    p("Quotes and feature ablations", "h2")
    p("Center r = observed mid - gamma * inventory * EWMA variance * horizon.<br/>"
      "Width h = base half-spread + 0.5 * gamma * variance * horizon<br/>"
      "+ volatility buffer * sqrt(variance * refresh exposure).", "code")
    tab(["Policy", "Center shifts with inventory", "Width changes with variance"], [
        ["Fixed", "No", "No"], ["Inventory", "Yes", "No"],
        ["Volatility", "No", "Yes"], ["Combined", "Yes", "Yes"],
    ], [112, 206, 206])
    p("Gamma = 0.12; risk horizon = min(40 seconds, time remaining); EWMA alpha = 0.04; "
      "common initial sigma = 0.025; volatility buffer = 0.8; tick = $0.01. The inventory "
      "center is motivated by Avellaneda-Stoikov. This implementation adds causal "
      "variance estimates and an explicit buffer; it is a heuristic rather than the "
      "paper's optimal-control solution.", "small")
    p("Market assumptions", "h2")
    p("The efficient price is an arithmetic diffusion plus compound-Poisson normal jumps. "
      "Six regimes vary volatility, jump size, ordinary/informed arrival rates, and fees. "
      "The burst regime changes volatility in the middle third; the shock regime includes "
      "a random-sign $1.50 jump at the midpoint. Ordinary willingness-to-pay depth is "
      "exponential with mean $0.10. Informed trades require at least $0.005 edge.")
    p("An immutable tape is created before policy evaluation from separate random streams "
      "for prices, ordinary flow, informed flow, and arrival priority. Strategy actions "
      "never consume random numbers or influence later market events.", "small")

    story.append(PageBreak())
    heading("Every dollar has an explanation", "Independent cash/inventory updates must reconcile with the component sum.")
    p("Let dq be the dealer's signed one-share fill, p its price, m_ref the midpoint "
      "used to form the quote, and m_exec the efficient price at execution. The step "
      "begins with inventory q_before.")
    p("Quote edge      = dq * (m_ref - p)<br/>Fill repricing  = dq * (m_exec - m_ref)<br/>"
      "Inventory P&amp;L = q_before * (m_exec - m_previous)<br/>"
      "Net P&amp;L       = sum of those components - fees - paid close", "code")
    p("Quote edge includes inventory skew and tick rounding. Fill repricing can be "
      "positive or negative. Summing quote edge and repricing gives execution edge; "
      "markouts are diagnostics and are never deducted again.")
    p("A stale-ask example", "h2")
    p("Quote reference $100; ask $100.04; new efficient price $100.20. An informed "
      "buyer lifts the ask, so the dealer sells one share. Quote edge is +$0.04 and "
      "repricing is -$0.20: execution P&amp;L is -$0.16 before fees and closing costs. "
      "Selling above an old midpoint does not imply a profitable trade.")
    figure("attribution.png", 3.6 / 8.6, "Figure 1. Mean component sums at base half-spread $0.04. Costs are shown as negative contributions; net includes paid close.")
    p("Execution quality and verification", "h2")
    p("Ledgers record trader type, signed quantity, quote reference, execution midpoint, "
      "inventory after fill, fee, and 1/5/20-second markouts. Horizons extending past "
      "session close are excluded. Informed execution losses are negative by the "
      "oracle trader definition; their size and frequency are the useful stress diagnostics.")
    p(f"Maximum reconciliation residual across all {manifest['n_sessions']:,} sessions: "
      f"{manifest['maximum_accounting_residual']:.2e} dollars. The 33 tests include hand-computed "
      "fills, paid liquidation, exact fee effects, per-fill caps, future-price prefix "
      "invariance, shared tapes, and a statistical exponential fill-rate check.", "small")

    story.append(PageBreak())
    heading("Aggressiveness has a frontier", "Confidence bands describe uncertainty across held-out synthetic sessions.")
    figure("frontiers.png", 5.2 / 8.6, "Figure 2. Tight quotes can increase turnover while decreasing P&L. Inventory skew and volatility widening have different effects on drawdown.")
    p("Illustrative paired width contrasts", "h2")
    main = raw[(raw.experiment == "main") & (raw.split == "held_out") & (raw.policy == "fixed")]
    contrasts = []
    for name, tight, wide in [("benign", .09, .14), ("toxic", .04, .09), ("high_cost", .01, .04)]:
        t = main[main.scenario == name].pivot(index="seed", columns="half_spread", values="net_pnl")
        d = t[tight] - t[wide]
        low, high = bootstrap_mean_ci(d)
        contrasts.append([NAMES[name], f"${wide:.2f} to ${tight:.2f}", dollar(d.mean()), f"[{low:.2f}, {high:.2f}]"])
    tab(["Regime", "Tighten base width", "P&L change", "Paired 95% interval"], contrasts, [94, 141, 116, 173])
    p("These width contrasts are exploratory examples from the fixed grid, not a claim "
      "that one optimum transfers between regimes. All use the same seeds within each contrast.", "small")
    p("A simple expected-value explanation", "h2")
    p("With only ordinary exponential-depth flow, fresh symmetric quotes, no caps, and "
      "no inventory risk, expected profit rate is proportional to "
      "(distance - fee) * exp(-k * distance). Its stationary maximum is at "
      "distance = fee + 1/k. Tightening helps when moving toward that maximum "
      "from above. Informed execution losses and stale-reference exposure alter "
      "the calculation, so maximizing fill count is an insufficient objective.")

    story.append(PageBreak())
    heading("A fairer baseline changes the claim", "One width per policy is selected on development seeds and then frozen.")
    p(f"Development uses {len(manifest['train_seeds'])} seeds per regime; evaluation uses "
      f"{len(manifest['test_seeds'])} disjoint seeds. Selection assigns equal weight to "
      "all six regimes and maximizes mean P&amp;L - 0.25 * mean maximum drawdown. "
      "The score weight is a stated research choice, not an economic optimum.")
    width_text = "; ".join(f"{POLICY_LABELS[r.policy]} ${r.half_spread:.3f}" for r in selected.itertuples())
    p("<b>Frozen base widths:</b> " + width_text + ".")
    rows = [[NAMES[s]] + [dollar(value(tuned, s, policy, "mean_pnl")) for policy in POLICIES] for s in ORDER]
    tab(["Regime", "Fixed", "Inventory", "Volatility", "Combined"], rows, [112, 103, 103, 103, 103])
    p("Selected-width held-out mean P&amp;L. Selected widths can differ across policies; "
      "the inventory parameters and model assumptions remain unchanged.", "small")
    p("What the feature ablations support", "h2")
    p("At the reference width, volatility-only quoting explains much of the profit "
      "improvement but retains substantially more inventory risk than the combined "
      "policy. Inventory-only quoting removes much of that risk without reliably "
      "increasing mean profit. After width selection, the combined policy has lower "
      "mean P&amp;L than fixed quoting in some regimes; an adaptive policy is not a "
      "universal profit improvement.")
    rows = []
    for s in ("benign", "toxic", "volatility_burst"):
        r = tuned_pairs[(tuned_pairs.scenario == s) & (tuned_pairs.candidate == "adaptive")].iloc[0]
        rows.append([NAMES[s], dollar(r.mean_delta_pnl), f"[{r.delta_ci_low:.2f}, {r.delta_ci_high:.2f}]",
                     f"{value(tuned,s,'fixed','mean_max_drawdown'):.2f} / {value(tuned,s,'adaptive','mean_max_drawdown'):.2f}"])
    tab(["Regime", "Combined - fixed", "Paired 95% interval", "Mean drawdown: fixed / combined"], rows, [92, 116, 143, 173])
    p("Intervals and inference", "h2")
    p("Each interval resamples entire seed-level sessions 3,000 times. For policy "
      "effects, paired differences are computed first. Inference does not count "
      "correlated fills as independent trials. Intervals measure Monte Carlo "
      "uncertainty within this model; exploratory comparisons have no multiplicity "
      "correction. Forty evaluation seeds are insufficient for precise rare-tail estimates.")

    story.append(PageBreak())
    heading("Failure cases and the next test", "A model earns trust by making its assumptions and failures easy to inspect.")
    figure("stress.png", 3.3 / 8.6, "Figure 3. Additional sensitivity seeds: 25 per setting. Left: toxic-flow refresh stress. Right: informed share changes at a fixed total potential order rate of 2.4/second, base half-spread $0.04.")
    p("A measurable failure", "h2")
    r = refresh[(refresh.scenario == "toxic") & (refresh.policy == "adaptive")].set_index("refresh_steps")
    p(f"In toxic flow, the combined policy's mean P&amp;L changes from {dollar(r.loc[1,'mean_pnl'])} "
      f"at one-step refresh to {dollar(r.loc[8,'mean_pnl'])} at eight-step refresh. "
      "Its variance estimate reacts after a jump; it cannot preempt a fresh price "
      "innovation. Quotes and inventory correction both become stale between refreshes.")
    p("Limitations that matter", "h2")
    p("Instant refills, independent ordinary depth, and oracle informed traders simplify "
      "execution. There are no competitors, queues, impact, funding requirements, or "
      "network cancel dynamics. Seed holdout checks simulation selection; it does not "
      "establish market generalization. Forty evaluation paths do not identify rare-tail risk.")
    p("The next experiment", "h2")
    p("Add finite displayed size, price-time-priority queues, and delayed cancellation "
      "or replenishment. Predict the direction of the fill-mix change before running. "
      "Then test whether the conclusion survives with fresh development/evaluation "
      "seeds. Real-data replay should use conservative queue assumptions and a "
      "chronological calibration/validation split.")
    p("Reproduce / inspect", "h2")
    p("python -m quotelab research --workers 4<br/>python scripts/build_report.py<br/>"
      "python -m quotelab dashboard", "code")
    p("Audit metrics, ledgers, paths, and seeds in results/. Open the offline explorer "
      "in a browser; use the local app to rerun settings.", "small")
    p('<b>Primary references:</b><br/><link href="https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf" color="#008c78">Avellaneda &amp; Stoikov (2008). High-frequency trading in a limit order book.</link><br/>'
      '<link href="https://www.janestreet.com/probability-markets/" color="#008c78">Jane Street. Probability &amp; Markets guide.</link>', "small")

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(green)
        canvas.rect(44, 757, 24, 3, fill=1, stroke=0)
        canvas.setFont(font, 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(44, 28, "QUOTELAB  /  SYNTHETIC MARKET-MAKING RESEARCH")
        canvas.drawRightString(568, 28, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(REPORTS / "QuoteLab_Research_Memo.pdf"), pagesize=(612, 792),
                            leftMargin=44, rightMargin=44, topMargin=48, bottomMargin=46,
                            title="QuoteLab: When does aggressive quoting pay?",
                            author="Prepared for Thomas Krane")
    doc.build(story, onFirstPage=frame, onLaterPages=frame)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    frames = {name: pd.read_csv(RESULTS / (name + ".csv")) for name in [
        "runs", "spread_summary", "reference_summary", "paired_deltas", "selected_parameters",
        "selected_heldout_summary", "selected_paired_deltas", "refresh_summary", "toxicity_summary"]}
    figures(frames["spread_summary"], frames["reference_summary"], frames["refresh_summary"], frames["toxicity_summary"])
    make_memo(manifest, frames["runs"], frames["spread_summary"], frames["reference_summary"],
              frames["paired_deltas"], frames["selected_parameters"], frames["selected_heldout_summary"],
              frames["selected_paired_deltas"], frames["refresh_summary"])
    (REPORTS / "QuoteLab_Dashboard.html").write_text(dashboard_html(RESULTS), encoding="utf-8")
    print(f"Built memo, figures, and offline explorer in {REPORTS}")


if __name__ == "__main__":
    main()

