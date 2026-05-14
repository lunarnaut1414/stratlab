"""Print a regime fingerprint of the IS window, or evaluate a signal expression.

Usage::

    python -m stratlab.arena.regime_check
    python -m stratlab.arena.regime_check --window oos
    python -m stratlab.arena.regime_check --signal "VIX<20"
    python -m stratlab.arena.regime_check --signal "TLT_21d > IEF_21d"
    python -m stratlab.arena.regime_check --signal "JNK > JNK_30d_MA"

Default mode (no --signal) reports year-by-year benchmark return, top-2-year
concentration, VIX-regime breakdown, and warnings. Used as a Step-0 pre-flight
check before a round.

--signal mode evaluates a simple boolean expression over the chosen window
and prints the percentage of trading days the expression is True, with a
year-by-year breakdown. Used by generators to pre-validate signal frequency
BEFORE designing a strategy around it — saves wasted submission attempts on
gates that fire too rarely (or too often) in the IS bull window. Asked for
across multiple rounds (gen_7/gen_8 wishlist, 7+ requests).

Signal grammar (ATOM OP ATOM, no AND/OR yet):
  ATOM := <TICKER>                   # close price at each bar (e.g. VIX, SPY)
       | <TICKER>_<N>d              # N-day return (e.g. TLT_21d)
       | <TICKER>_<N>d_MA           # N-day simple moving average of close
       | <NUMBER>                    # numeric literal
  OP   := < | > | <= | >= | == | !=
  Special: ticker name may include ``^`` prefix (^VIX, ^TNX); use ``VIX``
  as a convenience alias that resolves to ``^VIX``.

Mechanism: load required tickers from the cache for the chosen window; no
network access — operates purely on cached bars.
"""
from __future__ import annotations

import argparse
import re
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


# --- Signal expression evaluator ----------------------------------------

# ATOM_RE matches either NUMBER or TICKER[_<N>d[_MA]].
# TICKER allows alphanumerics, ^, ., =, /, - (covers ^VIX, ES=F, BRK-B).
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_ATOM_RE = re.compile(
    r"^(?P<ticker>[\^A-Za-z0-9.=/\-]+?)(?:_(?P<n>\d+)d(?:_(?P<kind>MA|RET))?)?$"
)
_COMPARISON_RE = re.compile(r"\s*(?P<op><=|>=|==|!=|<|>)\s*")

# Convenience aliases for common signal-only indices that agents write without
# the ``^`` prefix. The cache stores them as ``^VIX``, etc.
_TICKER_ALIASES = {
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "MOVE": "^MOVE",
    "SKEW": "^SKEW",
    "OVX": "^OVX",
    "GVZ": "^GVZ",
    "TNX": "^TNX",
    "IRX": "^IRX",
    "FVX": "^FVX",
    "TYX": "^TYX",
}


def _resolve_ticker(t: str) -> str:
    """Map a bare alias like 'VIX' to its cached symbol '^VIX'."""
    return _TICKER_ALIASES.get(t, t)


def _parse_atom(atom: str) -> tuple[str, float | None, str | None, str | None]:
    """Parse one atom. Returns (kind, number, ticker, transform):
      - ("literal", value, None, None)         for a numeric literal
      - ("series", None, ticker, transform)    where transform ∈ {None, "CLOSE", "RET_<N>", "MA_<N>"}
    """
    atom = atom.strip()
    if _NUMBER_RE.match(atom):
        return ("literal", float(atom), None, None)
    m = _ATOM_RE.match(atom)
    if not m:
        raise ValueError(f"cannot parse atom: {atom!r}")
    ticker = _resolve_ticker(m.group("ticker"))
    n = m.group("n")
    kind = m.group("kind")
    if n is None:
        return ("series", None, ticker, "CLOSE")
    n_int = int(n)
    if kind == "MA":
        return ("series", None, ticker, f"MA_{n_int}")
    # default for "TICKER_<N>d" is N-day return (TLT_21d means 21-day return)
    return ("series", None, ticker, f"RET_{n_int}")


def _series_for_atom(
    atom: tuple[str, float | None, str | None, str | None],
    start: str,
    end: str,
    cache: dict[str, pd.Series],
) -> pd.Series | float:
    """Compute the per-bar value for an atom over the window."""
    kind, lit, ticker, transform = atom
    if kind == "literal":
        return float(lit)
    if ticker not in cache:
        cache[ticker] = _load_close_series(ticker, start, end)
    close = cache[ticker]
    if transform == "CLOSE":
        return close
    if transform.startswith("RET_"):
        n = int(transform.split("_", 1)[1])
        return close.pct_change(n)
    if transform.startswith("MA_"):
        n = int(transform.split("_", 1)[1])
        return close.rolling(n, min_periods=n).mean()
    raise ValueError(f"unknown transform: {transform!r}")


def _compare(lhs, rhs, op: str) -> pd.Series:
    """Apply a comparison operator. Either side may be a scalar or Series.
    NaN values propagate (treated as False after dropna)."""
    if op == "<":
        return lhs < rhs
    if op == ">":
        return lhs > rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">=":
        return lhs >= rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    raise ValueError(f"unknown op: {op!r}")


def evaluate_signal(expression: str, window: str = "is") -> dict:
    """Evaluate a single-comparison boolean expression over the window and
    return {percent_true, by_year, n_total, n_valid, expression, start, end}.
    """
    m = _COMPARISON_RE.search(expression)
    if not m:
        raise ValueError(
            f"expression must contain one of < > <= >= == != ; got {expression!r}"
        )
    op = m.group("op")
    lhs_str = expression[: m.start()].strip()
    rhs_str = expression[m.end():].strip()
    if not lhs_str or not rhs_str:
        raise ValueError(f"expression must be ATOM OP ATOM; got {expression!r}")

    lhs_atom = _parse_atom(lhs_str)
    rhs_atom = _parse_atom(rhs_str)

    if window == "is":
        start, end = config.is_window_str()
    elif window == "oos":
        start, end = config.oos_window_str()
    else:
        raise ValueError(f"window must be 'is' or 'oos', got {window!r}")

    cache: dict[str, pd.Series] = {}
    lhs_val = _series_for_atom(lhs_atom, start, end, cache)
    rhs_val = _series_for_atom(rhs_atom, start, end, cache)

    # Align indexes if both are Series; ffill is appropriate because index
    # levels and prices are persistent state, not events.
    if isinstance(lhs_val, pd.Series) and isinstance(rhs_val, pd.Series):
        idx = lhs_val.index.union(rhs_val.index).sort_values()
        lhs_val = lhs_val.reindex(idx).ffill()
        rhs_val = rhs_val.reindex(idx).ffill()
    elif isinstance(lhs_val, pd.Series):
        idx = lhs_val.index
    elif isinstance(rhs_val, pd.Series):
        idx = rhs_val.index
    else:
        raise ValueError("both sides are literals — there's nothing to evaluate over time")

    # Restrict to window
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if isinstance(lhs_val, pd.Series):
        lhs_val = lhs_val.loc[(lhs_val.index >= start_ts) & (lhs_val.index <= end_ts)]
    if isinstance(rhs_val, pd.Series):
        rhs_val = rhs_val.loc[(rhs_val.index >= start_ts) & (rhs_val.index <= end_ts)]

    truth = _compare(lhs_val, rhs_val, op)
    if isinstance(truth, bool):
        # both literals — already validated above, but defensive
        raise ValueError("expression produced a constant truth value")

    # Drop NaN rows (where lookback wasn't yet satisfied) before counting.
    truth = truth.dropna()
    n_total = int(len(truth))
    if n_total == 0:
        return {
            "expression": expression,
            "window": window,
            "start": start,
            "end": end,
            "n_total": 0,
            "percent_true": float("nan"),
            "by_year": {},
        }
    percent = float(truth.mean())
    by_year: dict[int, dict] = {}
    for yr, g in truth.groupby(truth.index.year):
        by_year[int(yr)] = {
            "n_days": int(len(g)),
            "n_true": int(g.sum()),
            "percent": float(g.mean()),
        }

    return {
        "expression": expression,
        "window": window,
        "start": start,
        "end": end,
        "n_total": n_total,
        "percent_true": percent,
        "by_year": by_year,
    }


def _format_signal_report(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Signal frequency — {result['window'].upper()} window")
    lines.append(f"  expression : {result['expression']!r}")
    lines.append(f"  range      : {result['start']} .. {result['end']}")
    if result["n_total"] == 0:
        lines.append("  (no overlap between expression's lookbacks and window)")
        return "\n".join(lines)
    lines.append(f"  pct true   : {result['percent_true']:.1%}  ({result['n_total']} bars)")
    lines.append("")
    lines.append("## Year-by-year")
    lines.append(f"  {'year':>4}  {'n_days':>6}  {'n_true':>6}  {'pct_true':>8}")
    for yr, stats in result["by_year"].items():
        lines.append(
            f"  {yr:>4}  {stats['n_days']:>6}  {stats['n_true']:>6}  {stats['percent']:>7.1%}"
        )
    # Heuristic warnings
    pct = result["percent_true"]
    if pct < 0.10:
        lines.append("")
        lines.append("⚠ Signal fires <10% of days — defensive/risk-on branch will rarely engage in IS")
    elif pct > 0.90:
        lines.append("")
        lines.append("⚠ Signal fires >90% of days — gate adds little information, strategy ≈ unconditional")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--window", choices=("is", "oos"), default="is",
        help="Which window to evaluate over. Default: is.",
    )
    parser.add_argument(
        "--signal", default=None,
        help="A boolean expression like 'VIX<20' or 'TLT_21d>IEF_21d'. "
             "When given, prints the percentage of days the signal is True "
             "(year-by-year) instead of the default regime fingerprint.",
    )
    args = parser.parse_args(argv)

    if args.signal is not None:
        try:
            result = evaluate_signal(args.signal, args.window)
        except Exception as exc:
            sys.stderr.write(f"[regime_check] {exc}\n")
            return 1
        print(_format_signal_report(result))
        return 0

    try:
        fp = fingerprint(args.window)
    except Exception as exc:
        sys.stderr.write(f"[regime_check] {exc}\n")
        return 1
    print(_format_report(fp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
