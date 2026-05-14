"""Fast sub-sample IS Calmar estimate without writing to the leaderboard.

Usage::

    python -m stratlab.arena.is_calmar_estimate <strategy_module_path>
    python -m stratlab.arena.is_calmar_estimate <path> --start 2010-01-01 --end 2014-12-31

Runs the same backtest pipeline as submit.py but over a shortened IS sub-window
(default: first half, 2010-01-01 to 2014-12-31). Prints estimated IS Calmar,
Sharpe, n_trades, max_dd; DOES NOT write to leaderboard / returns matrix /
tearsheets / dead_ends / intent registry. Use this for fast iteration on
threshold/asset choices BEFORE committing to a full submit.

Roughly 2x faster than a full IS run (half the bars to simulate). Estimated
Calmar over a sub-window is an APPROXIMATION of full IS Calmar — strategies
with strong sub-period asymmetry (h1 vs h2 split) will be misestimated. Use
this for "is it in the right ballpark" not "will it pass the 0.5 gate".

Asked for in gen_8 by sonnet-4 and opus-2 (the "pre-submission Calmar
estimate" wishlist item).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

from stratlab.arena import config
from stratlab.arena.submit import load_strategy_module, resolve_universe
from stratlab.data.inception import filter_universe_by_window_overlap
from stratlab.data.universe import load_universe
from stratlab.engine.backtest import Backtest


# Default sub-sample: first half of IS. Roughly half the bars to simulate,
# while preserving enough history for typical 200d/252d lookbacks to warm up.
_DEFAULT_START = "2010-01-01"
_DEFAULT_END = "2014-12-31"


def estimate(
    strategy_path: Path,
    start: str = _DEFAULT_START,
    end: str = _DEFAULT_END,
) -> dict:
    """Run a sub-window backtest and return a metrics dict. No side effects."""
    module = load_strategy_module(strategy_path)
    universe_spec = getattr(module, "UNIVERSE", "sp500")
    tickers = resolve_universe(universe_spec)
    pre_count = len(tickers)
    tickers = filter_universe_by_window_overlap(tickers, start=start, end=end)
    if pre_count != len(tickers):
        sys.stderr.write(
            f"[is_calmar_estimate] universe filtered by window overlap: "
            f"{len(tickers)}/{pre_count} tickers have cached data in {start}..{end}\n"
        )

    data = load_universe(tickers, start=start, end=end)
    if not data:
        raise RuntimeError(
            f"no data loaded for universe {universe_spec!r} in {start}..{end} — "
            f"run `python -m stratlab.refresh` first"
        )

    bt = Backtest(
        data=data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
    )
    result = bt.run()
    return {
        "calmar": float(result.metrics.get("calmar", 0.0)),
        "sharpe": float(result.metrics.get("sharpe", 0.0)),
        "sortino": float(result.metrics.get("sortino", 0.0)),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
        "annual_volatility": float(result.metrics.get("annual_volatility", 0.0)),
        "n_trades": int(result.metrics.get("n_trades", 0)),
        "start": start,
        "end": end,
        "name": getattr(module, "NAME", strategy_path.stem),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "strategy_path", type=Path,
        help="Path to the strategy module (must export STRATEGY, NAME, HYPOTHESIS).",
    )
    parser.add_argument(
        "--start", default=_DEFAULT_START,
        help=f"Sub-window start (YYYY-MM-DD). Default {_DEFAULT_START}.",
    )
    parser.add_argument(
        "--end", default=_DEFAULT_END,
        help=f"Sub-window end (YYYY-MM-DD). Default {_DEFAULT_END}.",
    )
    args = parser.parse_args(argv)

    try:
        m = estimate(args.strategy_path, start=args.start, end=args.end)
    except Exception as exc:
        sys.stderr.write(f"[is_calmar_estimate] ERROR: {exc}\n")
        traceback.print_exc()
        return 1

    full_is_start = config.IS_START.isoformat()
    full_is_end = config.IS_END.isoformat()
    print(f"[is_calmar_estimate] {m['name']} on {m['start']}..{m['end']}")
    print(f"  (full IS window for comparison: {full_is_start}..{full_is_end})")
    print(f"  estimated Calmar : {m['calmar']:>+7.3f}")
    print(f"  estimated Sharpe : {m['sharpe']:>+7.3f}")
    print(f"  estimated Sortino: {m['sortino']:>+7.3f}")
    print(f"  estimated CAGR   : {m['cagr']:>7.1%}")
    print(f"  estimated MaxDD  : {m['max_drawdown']:>7.1%}")
    print(f"  n_trades         : {m['n_trades']:>7d}")
    print(f"  (sub-sample probe — leaderboard NOT updated)")
    if m["calmar"] < 0.3:
        print("  ⚠ Calmar far below 0.5 floor — re-design before full submit")
    elif m["calmar"] < 0.5:
        print("  ⚠ Calmar near 0.5 floor — full IS might just miss the gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
