"""Submit a strategy to the arena.

Usage::

    python -m stratlab.arena.submit <strategy_module_path> \\
        --gen <N> --agent-id <name> [--parent-id <id>] [--notes <text>]

The strategy module must export at least::

    STRATEGY    : an instantiated Strategy (subclass of stratlab.strategies.base.Strategy)
    NAME        : short slug, e.g. "rsi_meanrev_v1"
    HYPOTHESIS  : one-sentence rationale (free text)

Optional::

    UNIVERSE    : "sp500" (default), "sp500+hedge", a list of tickers, or a
                  zero-arg callable returning a list of tickers
    PARENT_ID   : strategy_id this is a mutation of (for lineage); CLI flag
                  takes precedence

Submission flow:

1. Import the module. Fail fast if required exports are missing.
2. Resolve the universe; load OHLCV for the IS window only.
3. Run a backtest over the IS window.
4. Enforce min-trade gate (config.MIN_TRADES_IS).
5. Compute IS daily-return correlation to current top-5 by IS Calmar; reject
   if absolute correlation exceeds config.CORR_REJECT_THRESHOLD.
6. Generate a tearsheet, append a row to the leaderboard CSV, and append
   the strategy's daily returns column to the parquet returns matrix.

On rejection, nothing is appended to the leaderboard and the rejection
reason is logged to ``tmp/arena/dead_ends.md`` plus stderr.

Exit codes:

- 0 : accepted
- 2 : trade-count gate failed
- 3 : correlation rejection
- 4 : data / module / runtime error
- 5 : Calmar floor (config.MIN_CALMAR_IS) not met
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stratlab.analytics.metrics import compute_subperiod_metrics
from stratlab.analytics.tearsheet import tearsheet
from stratlab.arena import config
from stratlab.arena.config import ensure_dirs, is_window_str
from stratlab.arena.leaderboard import (
    append_returns,
    append_row,
    loss_mode_corr_to,
    max_corr_to,
    read_leaderboard,
    read_returns_matrix,
    top_k_by,
)
from stratlab.engine.backtest import Backtest
from stratlab.strategies.base import Strategy

_DEFAULT_COMMISSION_PCT: float = 0.001
_DEFAULT_SLIPPAGE_PCT: float = 0.0005

_REQUIRED_EXPORTS = ("STRATEGY", "NAME", "HYPOTHESIS")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def load_strategy_module(path: Path) -> Any:
    """Import a strategy module from a filesystem path.

    Uses ``spec_from_file_location`` and does NOT add the module to
    ``sys.modules`` — each call yields a fresh module with fresh state,
    so backtests run on a clean strategy instance.
    """
    if not path.exists():
        raise FileNotFoundError(f"strategy module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_arena_strategy_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [s for s in _REQUIRED_EXPORTS if not hasattr(module, s)]
    if missing:
        raise ValueError(
            f"strategy module {path} missing required export(s): {missing}"
        )
    if not isinstance(module.STRATEGY, Strategy):
        raise TypeError(
            f"{path}: STRATEGY must be an instance of Strategy, "
            f"got {type(module.STRATEGY).__name__}"
        )
    return module


def resolve_universe(spec: Any) -> list[str]:
    """Translate a UNIVERSE declaration into a list of tickers.

    Accepts:
        - callable() → list[str]
        - list[str] | tuple[str, ...]
        - "sp500"          → current S&P 500 constituents
        - "sp500+hedge"    → SP500 + SPY (trend signal) + SH (inverse hedge)
        - "popular_etfs"   → curated broad/sector/factor/asset-class ETF list
        - "inverse_etfs"   → inverse ETFs (SH, SDS, PSQ, ...)
        - "leveraged_etfs" → broad-market leveraged ETFs (TQQQ, UPRO, SOXL, ...)
        - "single_stock_leveraged_etfs" → TSLL, NVDL, AAPB, ...
    """
    from stratlab.data.universe import (
        inverse_etfs,
        leveraged_etfs,
        popular_etfs,
        single_stock_leveraged_etfs,
        sp500_tickers,
    )

    if callable(spec):
        return [str(t) for t in spec()]
    if isinstance(spec, (list, tuple)):
        return [str(t) for t in spec]
    if isinstance(spec, str):
        if spec == "sp500":
            return sp500_tickers()
        if spec == "sp500+hedge":
            return sp500_tickers() + ["SPY", "SH"]
        if spec == "popular_etfs":
            return popular_etfs()
        if spec == "inverse_etfs":
            return inverse_etfs()
        if spec == "leveraged_etfs":
            return leveraged_etfs()
        if spec == "single_stock_leveraged_etfs":
            return single_stock_leveraged_etfs()
        raise ValueError(f"unknown UNIVERSE string: {spec!r}")
    raise TypeError(
        f"UNIVERSE must be str, list, tuple, or callable; got {type(spec).__name__}"
    )


def _slug(name: str) -> str:
    """Lowercase alphanumeric + underscores; collapses runs and trims edges."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


def _params_json(strategy: Strategy) -> str:
    """JSON-encode strategy.params, falling back to repr() for non-JSON values."""
    try:
        return json.dumps(strategy.params, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(
            {k: repr(v) for k, v in strategy.params.items()}, sort_keys=True
        )


def _load_benchmark_returns(start: str, end: str) -> pd.Series:
    """Load benchmark daily returns over the IS window. Used for loss-mode
    correlation. Returns an empty Series if the benchmark cache is missing."""
    try:
        from stratlab.data.provider import load_bars

        bars = load_bars(config.BENCHMARK_TICKER, start=start, end=end)
        if bars.empty:
            return pd.Series(dtype=float)
        return bars["close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


def _record_dead_end(strategy_id: str, reason: str) -> None:
    """Append a one-line entry to dead_ends.md so failures are durable."""
    config.DEAD_ENDS.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with config.DEAD_ENDS.open("a") as f:
        f.write(f"- {stamp} — `{strategy_id}` — {reason}\n")


def _fail(
    strategy_id: str,
    reason: str,
    exit_code: int,
    intent_id: str = "",
) -> None:
    """Print rejection to stderr, record to dead_ends.md, mark the intent
    (if any) as abandoned, exit."""
    sys.stderr.write(f"[submit] REJECTED {strategy_id}: {reason}\n")
    _record_dead_end(strategy_id, reason)
    if intent_id:
        from stratlab.arena.intents import mark_status
        try:
            mark_status(intent_id, "abandoned", notes=reason)
        except Exception as exc:
            sys.stderr.write(f"[submit] (warning: could not mark intent: {exc})\n")
    sys.exit(exit_code)


def submit(
    strategy_path: Path,
    gen: int,
    agent_id: str,
    parent_id: str = "",
    notes: str = "",
    intent_id: str = "",
    commission_pct: float = _DEFAULT_COMMISSION_PCT,
    slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
) -> str:
    """Validate + run + record a strategy submission. Returns the
    assigned ``strategy_id`` on success; raises / sys.exits on failure.

    When ``commission_pct`` or ``slippage_pct`` differ from the harness
    defaults, the run is treated as a **cost-stress probe**: metrics are
    printed but NOT written to the leaderboard or returns matrix, so
    leaderboard rows remain comparable under one fixed cost assumption.
    """
    ensure_dirs()
    is_cost_stress = (
        commission_pct != _DEFAULT_COMMISSION_PCT
        or slippage_pct != _DEFAULT_SLIPPAGE_PCT
    )

    module = load_strategy_module(strategy_path)
    universe_spec = getattr(module, "UNIVERSE", "sp500")
    tickers = resolve_universe(universe_spec)

    is_start, is_end = is_window_str()
    # Pre-filter universe to tickers whose cached data overlaps the IS window
    # at all. Excludes tickers with NO data in [is_start, is_end] (eliminates
    # yfinance "Failed to get ticker" noise for post-2018 IPOs) but KEEPS
    # tickers that IPO'd mid-window — the engine's per-bar NaN filter handles
    # the pre-IPO portion. Survivorship bias from index-departures (Lehman,
    # GE pre-2018, etc.) is not addressed here since those names aren't in
    # `tickers` at all.
    from stratlab.data.inception import filter_universe_by_window_overlap
    pre_count = len(tickers)
    tickers = filter_universe_by_window_overlap(tickers, start=is_start, end=is_end)
    if pre_count != len(tickers):
        sys.stderr.write(
            f"[submit] universe filtered by IS overlap: "
            f"{len(tickers)}/{pre_count} tickers have cached data in {is_start}..{is_end}\n"
        )

    from stratlab.data.universe import load_universe
    is_data = load_universe(tickers, start=is_start, end=is_end)
    if not is_data:
        raise RuntimeError(
            f"no IS data loaded for universe {universe_spec!r} "
            f"between {is_start} and {is_end} — run "
            f"`python -m stratlab.refresh` and retry."
        )

    strategy_id = f"gen{gen}_{_slug(module.NAME)}"

    bt = Backtest(
        data=is_data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )
    result = bt.run()

    n_trades = int(result.metrics.get("n_trades", 0))

    if is_cost_stress:
        # Don't run the gates (no leaderboard mutation); just print raw metrics
        # at the requested cost level so the user can compare against the
        # baseline submission's metrics on the leaderboard.
        cal = float(result.metrics.get("calmar", 0.0))
        sh = float(result.metrics.get("sharpe", 0.0))
        cagr = float(result.metrics.get("cagr", 0.0))
        mdd = float(result.metrics.get("max_drawdown", 0.0))
        print(
            f"[submit:cost-stress] {strategy_id} commission={commission_pct:.4f} "
            f"slippage={slippage_pct:.4f}"
        )
        print(f"  Calmar  : {cal:>+7.3f}")
        print(f"  Sharpe  : {sh:>+7.3f}")
        print(f"  CAGR    : {cagr:>7.1%}")
        print(f"  MaxDD   : {mdd:>7.1%}")
        print(f"  n_trades: {n_trades:>7d}")
        print("  (probe run — leaderboard NOT updated)")
        return strategy_id

    if n_trades < config.MIN_TRADES_IS:
        _fail(
            strategy_id,
            f"n_trades={n_trades} below MIN_TRADES_IS={config.MIN_TRADES_IS}",
            exit_code=2,
            intent_id=intent_id,
        )

    is_calmar = float(result.metrics.get("calmar", 0.0))
    if is_calmar < config.MIN_CALMAR_IS:
        _fail(
            strategy_id,
            f"IS Calmar {is_calmar:.3f} below MIN_CALMAR_IS={config.MIN_CALMAR_IS} "
            f"— write a postmortem to dead_ends.md, don't submit weak strategies",
            exit_code=5,
            intent_id=intent_id,
        )

    leaderboard = read_leaderboard()
    top5 = top_k_by(
        leaderboard,
        metric="is_calmar",
        k=config.TOP_K_FOR_CORR_CHECK,
        require_n_trades=config.MIN_TRADES_IS,
    )
    returns_matrix = read_returns_matrix()
    max_corr, twin_id = max_corr_to(
        result.returns,
        returns_matrix,
        top5["strategy_id"].dropna().tolist(),
    )
    if abs(max_corr) > config.CORR_REJECT_THRESHOLD:
        _fail(
            strategy_id,
            f"|corr| {max_corr:+.3f} to {twin_id!r} exceeds "
            f"{config.CORR_REJECT_THRESHOLD} — try an uncorrelated angle",
            exit_code=3,
            intent_id=intent_id,
        )

    fig = tearsheet(
        result,
        benchmark=config.BENCHMARK_TICKER,
        title=f"{strategy_id} — IS {is_start}..{is_end}",
    )
    tearsheet_path = config.TEARSHEETS_DIR / f"{strategy_id}.html"
    fig.write_html(str(tearsheet_path))

    equity_curve_path = config.EQUITY_CURVES_DIR / f"{strategy_id}.csv"
    result.equity_curve.to_frame(name="equity").to_csv(equity_curve_path)

    subperiod = compute_subperiod_metrics(result.equity_curve, result.returns)

    benchmark_returns = _load_benchmark_returns(is_start, is_end)
    loss_corr, loss_twin_id = loss_mode_corr_to(
        result.returns,
        returns_matrix,
        top5["strategy_id"].dropna().tolist(),
        benchmark_returns,
    )

    row = {
        "strategy_id": strategy_id,
        "generation": gen,
        "parent_id": parent_id or getattr(module, "PARENT_ID", "") or "",
        "agent_id": agent_id,
        "name": module.NAME,
        "path": str(strategy_path),
        "hypothesis": module.HYPOTHESIS,
        "params_json": _params_json(module.STRATEGY),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "is_sharpe": float(result.metrics.get("sharpe", 0.0)),
        "is_calmar": float(result.metrics.get("calmar", 0.0)),
        "is_sortino": float(result.metrics.get("sortino", 0.0)),
        "is_cagr": float(result.metrics.get("cagr", 0.0)),
        "is_max_dd": float(result.metrics.get("max_drawdown", 0.0)),
        "is_annual_vol": float(result.metrics.get("annual_volatility", 0.0)),
        "is_win_rate": float(result.metrics.get("win_rate", 0.0)),
        "is_n_trades": n_trades,
        "is_turnover": float(result.metrics.get("turnover_annualized", 0.0)),
        **subperiod,
        "corr_to_top5": round(float(max_corr), 4),
        "loss_mode_corr_to_top5": round(float(loss_corr), 4),
        "tearsheet_path": str(tearsheet_path),
        "equity_curve_path": str(equity_curve_path),
        "notes": notes,
    }
    append_row(row)
    append_returns(strategy_id, result.returns)

    if intent_id:
        from stratlab.arena.intents import mark_status
        try:
            mark_status(intent_id, "submitted", strategy_id=strategy_id)
        except Exception as exc:
            sys.stderr.write(f"[submit] (warning: could not mark intent: {exc})\n")

    print(f"[submit] {strategy_id} ACCEPTED")
    print(f"  IS Calmar       : {row['is_calmar']:>7.2f}")
    print(f"  IS Calmar (h1)  : {row['is_calmar_h1']:>7.2f}")
    print(f"  IS Calmar (h2)  : {row['is_calmar_h2']:>7.2f}")
    print(f"  IS PnL top-2y % : {row['is_pnl_top2y_pct']:>7.1%}")
    print(f"  IS Sharpe       : {row['is_sharpe']:>7.2f}")
    print(f"  IS Sortino      : {row['is_sortino']:>7.2f}")
    print(f"  IS CAGR         : {row['is_cagr']:>7.1%}")
    print(f"  IS Max DD       : {row['is_max_dd']:>7.1%}")
    print(f"  IS n_trades     : {n_trades:>7d}")
    print(f"  corr_to_top5    : {max_corr:>+7.3f}")
    print(f"  loss_mode_corr  : {loss_corr:>+7.3f}")
    print(f"  tearsheet       : {tearsheet_path}")
    print(f"  equity_curve    : {equity_curve_path}")
    return strategy_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "strategy_path",
        type=Path,
        help="Path to the strategy module (must export STRATEGY, NAME, HYPOTHESIS).",
    )
    parser.add_argument(
        "--gen", type=int, required=True,
        help="Generation number (0 = seeds, 1+ = agent-produced).",
    )
    parser.add_argument(
        "--agent-id", default="manual",
        help='Agent / human identifier — e.g. "sonnet-4-6", "human", "claude-code".',
    )
    parser.add_argument(
        "--parent-id", default="",
        help="strategy_id of the predecessor this mutates (lineage tracking).",
    )
    parser.add_argument(
        "--notes", default="",
        help="Free-text note carried into the leaderboard row.",
    )
    parser.add_argument(
        "--intent-id", default="",
        help="Pre-committed intent_id from `stratlab.arena.intents commit`. "
             "On accept/reject, the intent is auto-marked submitted/abandoned.",
    )
    parser.add_argument(
        "--commission-pct", type=float, default=_DEFAULT_COMMISSION_PCT,
        help=f"Commission per fill (one-way, fraction of notional). "
             f"Default {_DEFAULT_COMMISSION_PCT}. Non-default values run as a "
             f"COST-STRESS PROBE — metrics print only, leaderboard is NOT updated.",
    )
    parser.add_argument(
        "--slippage-pct", type=float, default=_DEFAULT_SLIPPAGE_PCT,
        help=f"Slippage per market fill (fraction of notional). "
             f"Default {_DEFAULT_SLIPPAGE_PCT}. Non-default values run as a "
             f"COST-STRESS PROBE — metrics print only, leaderboard is NOT updated.",
    )
    args = parser.parse_args(argv)

    try:
        submit(
            args.strategy_path,
            gen=args.gen,
            agent_id=args.agent_id,
            parent_id=args.parent_id,
            notes=args.notes,
            intent_id=args.intent_id,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
        )
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"[submit] ERROR: {exc}\n")
        traceback.print_exc()
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
