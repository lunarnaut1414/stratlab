"""Print a regime fingerprint of the IS window, or evaluate a signal expression.

Usage::

    python -m stratlab.arena.regime_check
    python -m stratlab.arena.regime_check --window oos
    python -m stratlab.arena.regime_check --signal "VIX<20"
    python -m stratlab.arena.regime_check --signal "TLT_21d > IEF_21d"
    python -m stratlab.arena.regime_check --signal "JNK > JNK_30d_MA"
    python -m stratlab.arena.regime_check --signal "TYX - TNX > 0.5"
    python -m stratlab.arena.regime_check --signal "VIX_pct252 < 0.3"
    python -m stratlab.arena.regime_check --signal "JNK_20d > LQD_20d AND RSP_20d > SPY_20d"

Default mode (no --signal) reports year-by-year benchmark return, top-2-year
concentration, VIX-regime breakdown, and warnings.

--signal mode evaluates a boolean expression over the chosen window and
prints the percentage of trading days the expression is True, with year-by-year
breakdown. Used to pre-validate signal frequency BEFORE designing a strategy
around it. Asked for across 4 rounds (10+ requests).

Signal grammar:

  EXPR    := COMPARISON [(AND|OR) COMPARISON]*    # composite via AND/OR (no parens)
  COMPARISON := TERM CMP TERM                      # CMP: < > <= >= == !=
  TERM    := ATOM | ATOM (+|-) ATOM                # arithmetic spread of two atoms
  ATOM    := <TICKER>                              # close price at each bar
           | <TICKER>_<N>d                         # N-day return (e.g. TLT_21d)
           | <TICKER>_<N>d_MA                      # N-day simple moving average
           | <TICKER>_pct<N>                       # close's percentile rank in N-day window
           | <TICKER>_<M>d_pct<N>                  # M-day return's percentile rank in N-day window
           | <NUMBER>                              # numeric literal

  Special: ticker name may include ``^`` prefix (^VIX, ^TNX); bare ``VIX``
  resolves to ``^VIX`` (same for VVIX/MOVE/SKEW/OVX/GVZ/TNX/IRX/FVX/TYX).
  AND/OR are case-insensitive. Whitespace around operators is optional.

Mechanism: load required tickers from cache; no network access.
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

# Tickers contain alphanumerics + ^, =, /, but NO underscores (those are
# transform separators). BRK-B/JPM-PA share-class hyphens are still allowed.
# Note: bare hyphens here are ambiguous with subtraction in arithmetic terms;
# the arithmetic splitter handles spaces/known suffixes to disambiguate.
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_ATOM_RE = re.compile(
    r"^(?P<ticker>[\^A-Za-z0-9.=/\-]+?)"
    r"(?:_(?P<m>\d+)d(?:_(?P<kind>MA|RET))?)?"          # optional N-day return / MA
    r"(?:_pct(?P<pct>\d+))?$"                            # optional percentile suffix
)
_COMPARISON_RE = re.compile(r"\s*(?P<op><=|>=|==|!=|<|>)\s*")
# Top-level AND/OR splitter (no parentheses support). Word-boundary so we
# don't match "AND" inside ticker names (NASDAQ has none, but defensive).
_LOGIC_SPLIT_RE = re.compile(r"\s+(AND|OR)\s+", re.IGNORECASE)
# Arithmetic infix detector — only splits on +/- when surrounded by whitespace,
# so ticker hyphens like BRK-B don't get split.
_ARITH_SPLIT_RE = re.compile(r"\s+([+\-])\s+")

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
      - ("series", None, ticker, transform)    where transform encodes the
        compound: "CLOSE", "RET_<N>", "MA_<N>", "PCT_<N>",
        "RET_<M>_PCT_<N>", "MA_<M>_PCT_<N>".
    """
    atom = atom.strip()
    if _NUMBER_RE.match(atom):
        return ("literal", float(atom), None, None)
    m = _ATOM_RE.match(atom)
    if not m:
        raise ValueError(f"cannot parse atom: {atom!r}")
    ticker = _resolve_ticker(m.group("ticker"))
    n_inner = m.group("m")
    kind = m.group("kind")
    pct_window = m.group("pct")

    # Base transform (close, return, MA)
    if n_inner is None:
        base = "CLOSE"
    else:
        n_int = int(n_inner)
        if kind == "MA":
            base = f"MA_{n_int}"
        else:
            base = f"RET_{n_int}"

    # Apply percentile wrapper if present
    if pct_window is None:
        return ("series", None, ticker, base)
    pct_n = int(pct_window)
    if base == "CLOSE":
        return ("series", None, ticker, f"PCT_{pct_n}")
    # e.g. "RET_21_PCT_252" or "MA_30_PCT_252"
    return ("series", None, ticker, f"{base}_PCT_{pct_n}")


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Per-bar percentile rank of s within its trailing ``window`` values.
    Value in [0, 1]: 0 = bottom of distribution, 1 = top. NaN until window
    is satisfied. Implemented via rolling.rank(pct=True).
    """
    return s.rolling(window, min_periods=window).rank(pct=True)


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
    # Decompose transform into optional base + optional percentile wrapper.
    parts = transform.split("_PCT_") if "_PCT_" in transform else None
    if parts:
        base, pct_str = parts[0], parts[1]
        pct_n = int(pct_str)
    elif transform.startswith("PCT_"):
        base, pct_n = "CLOSE", int(transform.split("_", 1)[1])
    else:
        base, pct_n = transform, None

    if base == "CLOSE":
        underlying = close
    elif base.startswith("RET_"):
        n = int(base.split("_", 1)[1])
        underlying = close.pct_change(n)
    elif base.startswith("MA_"):
        n = int(base.split("_", 1)[1])
        underlying = close.rolling(n, min_periods=n).mean()
    else:
        raise ValueError(f"unknown base transform: {base!r}")

    if pct_n is None:
        return underlying
    return _rolling_pct_rank(underlying, pct_n)


def _parse_term(term_str: str, start: str, end: str, cache: dict) -> pd.Series | float:
    """Parse a TERM = ATOM | ATOM (+|-) ATOM. Returns the computed value/series."""
    term_str = term_str.strip()
    # Detect arithmetic with whitespace-padded +/- (so BRK-B isn't split)
    m = _ARITH_SPLIT_RE.search(term_str)
    if not m:
        atom = _parse_atom(term_str)
        return _series_for_atom(atom, start, end, cache)
    op = m.group(1)
    lhs_str = term_str[: m.start()].strip()
    rhs_str = term_str[m.end():].strip()
    lhs_atom = _parse_atom(lhs_str)
    rhs_atom = _parse_atom(rhs_str)
    lhs_val = _series_for_atom(lhs_atom, start, end, cache)
    rhs_val = _series_for_atom(rhs_atom, start, end, cache)
    if op == "+":
        return lhs_val + rhs_val
    return lhs_val - rhs_val


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


def _evaluate_comparison(
    comp_str: str,
    start: str,
    end: str,
    cache: dict,
) -> pd.Series:
    """Evaluate a single COMPARISON (TERM CMP TERM) and return a boolean Series."""
    m = _COMPARISON_RE.search(comp_str)
    if not m:
        raise ValueError(
            f"sub-expression missing comparison operator: {comp_str!r}"
        )
    op = m.group("op")
    lhs_str = comp_str[: m.start()].strip()
    rhs_str = comp_str[m.end():].strip()
    if not lhs_str or not rhs_str:
        raise ValueError(f"comparison must be TERM OP TERM; got {comp_str!r}")
    lhs_val = _parse_term(lhs_str, start, end, cache)
    rhs_val = _parse_term(rhs_str, start, end, cache)

    # Align indexes if both are Series; ffill since prices are persistent state.
    if isinstance(lhs_val, pd.Series) and isinstance(rhs_val, pd.Series):
        idx = lhs_val.index.union(rhs_val.index).sort_values()
        lhs_val = lhs_val.reindex(idx).ffill()
        rhs_val = rhs_val.reindex(idx).ffill()
    elif not isinstance(lhs_val, pd.Series) and not isinstance(rhs_val, pd.Series):
        raise ValueError(
            f"both sides of comparison are literals: {comp_str!r}"
        )

    # Restrict to window
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if isinstance(lhs_val, pd.Series):
        lhs_val = lhs_val.loc[(lhs_val.index >= start_ts) & (lhs_val.index <= end_ts)]
    if isinstance(rhs_val, pd.Series):
        rhs_val = rhs_val.loc[(rhs_val.index >= start_ts) & (rhs_val.index <= end_ts)]

    return _compare(lhs_val, rhs_val, op)


def evaluate_signal(expression: str, window: str = "is") -> dict:
    """Evaluate a (possibly AND/OR composite) boolean expression over the
    window. Returns {percent_true, by_year, n_total, expression, start, end}.
    """
    if window == "is":
        start, end = config.is_window_str()
    elif window == "oos":
        start, end = config.oos_window_str()
    else:
        raise ValueError(f"window must be 'is' or 'oos', got {window!r}")

    cache: dict[str, pd.Series] = {}

    # Split top-level expression by AND/OR. Left-associative evaluation.
    parts = _LOGIC_SPLIT_RE.split(expression.strip())
    # parts is [comp, op, comp, op, ...] with op uppercase
    truth: pd.Series | None = None
    for i, segment in enumerate(parts):
        seg = segment.strip()
        if i == 0:
            truth = _evaluate_comparison(seg, start, end, cache)
            continue
        # Odd indices are AND/OR; even (>=2) are comparisons.
        if i % 2 == 1:
            logic_op = seg.upper()
            continue
        rhs_truth = _evaluate_comparison(seg, start, end, cache)
        # Align indexes (both series of booleans by here).
        idx = truth.index.union(rhs_truth.index).sort_values()
        lhs = truth.reindex(idx)
        rhs = rhs_truth.reindex(idx)
        if logic_op == "AND":
            truth = lhs & rhs
        elif logic_op == "OR":
            truth = lhs | rhs
        else:
            raise ValueError(f"unknown logic op: {logic_op!r}")

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
