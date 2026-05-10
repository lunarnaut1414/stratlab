"""Print a regime fingerprint of the IS window.

Usage::

    python -m stratlab.arena.regime_check
    python -m stratlab.arena.regime_check --window oos

Reports year-by-year benchmark return, top-2-year concentration, VIX-regime
breakdown, and a warning if the window is structurally biased toward one
regime. Intended as a Step-0 pre-flight check before kicking off a round —
helps the orchestrator (and the user) calibrate expectations for IS Calmar
inflation BEFORE spending compute on Phase 1.

Mechanism: load SPY (benchmark) + ^VIX from the cache for the chosen window;
group SPY returns by calendar year; report distribution + concentration. No
network access — operates purely on cached bars.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from stratlab.arena import config


def _load_close_series(symbol: str, start: str, end: str) -> pd.Series:
    from stratlab.data.provider import load_bars

    bars = load_bars(symbol, start=start, end=end)
    if bars.empty:
        raise RuntimeError(
            f"no cached bars for {symbol} in {start}..{end} — run "
            f"`python -m stratlab.refresh` first"
        )
    return bars["close"]


def fingerprint(window: str = "is") -> dict:
    """Return a dict of regime statistics for the chosen window."""
    if window == "is":
        start, end = config.is_window_str()
    elif window == "oos":
        start, end = config.oos_window_str()
    else:
        raise ValueError(f"window must be 'is' or 'oos', got {window!r}")

    spy = _load_close_series(config.BENCHMARK_TICKER, start, end)
    spy_returns = spy.pct_change().dropna()
    spy_log = np.log1p(spy_returns)
    yearly_log = spy_log.groupby(spy_log.index.year).sum()
    yearly_pct = (np.exp(yearly_log) - 1.0).rename("year_return")

    total_log = float(yearly_log.sum())
    if total_log > 0 and len(yearly_log) >= 2:
        top2_share = float(yearly_log.nlargest(2).sum() / total_log)
    else:
        top2_share = 0.0

    spy_eq = (1 + spy_returns).cumprod()
    spy_dd = (spy_eq - spy_eq.cummax()) / spy_eq.cummax()
    max_dd = float(spy_dd.min())

    try:
        vix = _load_close_series("^VIX", start, end)
        vix_aligned = vix.reindex(spy_returns.index).ffill()
        vix_stats = {
            "mean": float(vix_aligned.mean()),
            "median": float(vix_aligned.median()),
            "pct_below_18": float((vix_aligned < 18).mean()),
            "pct_above_25": float((vix_aligned > 25).mean()),
            "pct_above_30": float((vix_aligned > 30).mean()),
        }
    except Exception:
        vix_stats = None

    warnings: list[str] = []
    if top2_share > 0.50:
        warnings.append(
            f"⚠ Top-2-year concentration is {top2_share:.0%} — Calmar metrics on "
            f"this window will overstate strategy quality. Treat headline IS "
            f"Calmar as ~{1 - top2_share:.0%} of its face value when judging "
            f"OOS prospects."
        )
    if vix_stats and vix_stats["pct_below_18"] > 0.55:
        warnings.append(
            f"⚠ VIX < 18 on {vix_stats['pct_below_18']:.0%} of days — "
            f"strategies that gate on a 'calm regime' will be active most of "
            f"the window in IS but dormant in higher-vol OOS years."
        )
    if vix_stats and vix_stats["pct_above_25"] < 0.05:
        warnings.append(
            f"⚠ VIX > 25 on only {vix_stats['pct_above_25']:.0%} of days — "
            f"defensive limbs of regime-switching strategies will fire rarely "
            f"in IS, and the strategies' tail-risk behavior is essentially "
            f"untested."
        )

    return {
        "window": window,
        "start": start,
        "end": end,
        "yearly_returns": yearly_pct.to_dict(),
        "total_return": float(np.exp(total_log) - 1.0),
        "top2_year_share": top2_share,
        "max_drawdown": max_dd,
        "vix": vix_stats,
        "warnings": warnings,
    }


def _format_report(fp: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Regime fingerprint — {fp['window'].upper()} window")
    lines.append(f"  range: {fp['start']} .. {fp['end']}")
    lines.append("")
    lines.append("## Year-by-year benchmark return")
    for year, ret in fp["yearly_returns"].items():
        lines.append(f"  {year}: {ret:>+7.1%}")
    lines.append(f"  total: {fp['total_return']:>+7.1%}")
    lines.append(f"  top-2-year share of total log-return: {fp['top2_year_share']:.0%}")
    lines.append(f"  max benchmark drawdown: {fp['max_drawdown']:.1%}")
    lines.append("")
    if fp["vix"] is not None:
        v = fp["vix"]
        lines.append("## VIX regime")
        lines.append(f"  mean / median: {v['mean']:.1f} / {v['median']:.1f}")
        lines.append(f"  pct days VIX < 18: {v['pct_below_18']:.0%}")
        lines.append(f"  pct days VIX > 25: {v['pct_above_25']:.0%}")
        lines.append(f"  pct days VIX > 30: {v['pct_above_30']:.0%}")
        lines.append("")
    if fp["warnings"]:
        lines.append("## Warnings")
        for w in fp["warnings"]:
            lines.append(f"  {w}")
    else:
        lines.append("## Warnings")
        lines.append("  (none — window appears reasonably balanced)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--window", choices=("is", "oos"), default="is",
        help="Which window to fingerprint. Default: is.",
    )
    args = parser.parse_args(argv)
    try:
        fp = fingerprint(args.window)
    except Exception as exc:
        sys.stderr.write(f"[regime_check] {exc}\n")
        return 1
    print(_format_report(fp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
