"""Out-of-sample evaluation runner — the only entry point allowed to read
OOS data.

Reads the leaderboard, picks the top-K entries by IS Calmar that haven't
yet been OOS-evaluated, runs each over the frozen OOS window, and writes
the resulting metrics into the leaderboard's ``oos_*`` columns.

Keeping OOS access concentrated in one module makes leakage auditable —
the only way IS-side code (submit.py, generators) can see OOS metrics is
by reading the leaderboard CSV *after* this script has run, and even
then the columns labeled ``oos_*`` are off-limits to generator prompts
by convention.

Usage::

    python -m stratlab.arena.promote [--top K] [--strategy-id ID]

By default, evaluates the top-K (config.TOP_K_PROMOTE) by IS Calmar that
have ``oos_evaluated_at`` empty. Use ``--strategy-id`` to force-evaluate
a specific entry (useful for re-runs after a bug fix).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from stratlab.analytics.tearsheet import tearsheet_combined
from stratlab.arena import config
from stratlab.arena.config import ensure_dirs, is_window_str, oos_window_str
from stratlab.arena.leaderboard import (
    read_leaderboard,
    top_k_by,
    update_oos,
)
from stratlab.arena.submit import load_strategy_module, resolve_universe
from stratlab.engine.backtest import Backtest


def _select_for_promotion(top_k: int, strategy_id: str | None) -> list[dict]:
    """Pick which leaderboard rows to OOS-evaluate.

    If ``strategy_id`` is given, force-evaluate that one. Otherwise take
    the top-K by IS Calmar that haven't been OOS-evaluated yet.
    """
    df = read_leaderboard()
    if df.empty:
        return []

    if strategy_id:
        match = df[df["strategy_id"] == strategy_id]
        if match.empty:
            raise ValueError(f"strategy_id {strategy_id!r} not in leaderboard")
        return match.to_dict(orient="records")

    pending = df[df["oos_evaluated_at"].isna() | (df["oos_evaluated_at"] == "")]
    if pending.empty:
        return []
    selected = top_k_by(
        pending, metric="is_calmar", k=top_k, require_n_trades=config.MIN_TRADES_IS,
    )
    return selected.to_dict(orient="records")


def _run_window(module, universe_spec, start: str, end: str):
    """Backtest ``module.STRATEGY`` over ``[start, end]``. Used for both IS
    and OOS runs at promote time so the combined tearsheet has equity curves
    for both windows."""
    from stratlab.data.inception import filter_universe_by_window_overlap
    from stratlab.data.universe import load_universe

    tickers = resolve_universe(universe_spec)
    tickers = filter_universe_by_window_overlap(tickers, start=start, end=end)
    data = load_universe(tickers, start=start, end=end)
    if not data:
        raise RuntimeError(f"no data loaded for {start}..{end}")

    bt = Backtest(
        data=data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
    )
    return bt.run()


def evaluate_oos(row: dict) -> dict:
    """Run one strategy over the OOS window AND re-run it over IS so the
    saved tearsheet covers the full lifetime with the IS/OOS boundary
    visible. Returns a dict of OOS metrics suitable for ``update_oos``.

    Re-running IS at promote time costs ~2× per strategy but keeps
    ``submit.py`` untouched and avoids persisting equity-curve state. The
    OOS metrics returned to the leaderboard come solely from the OOS
    backtest — IS metrics from this re-run are not written back (they
    already exist on the leaderboard from submit time)."""
    strategy_path = Path(row["path"])
    module = load_strategy_module(strategy_path)
    universe_spec = getattr(module, "UNIVERSE", "sp500")

    is_start, is_end = is_window_str()
    oos_start, oos_end = oos_window_str()

    is_result = _run_window(module, universe_spec, is_start, is_end)
    oos_result = _run_window(module, universe_spec, oos_start, oos_end)

    fig = tearsheet_combined(
        is_result, oos_result,
        benchmark=config.BENCHMARK_TICKER,
        title=f"{row['strategy_id']} — IS {is_start}..{is_end}  ·  OOS {oos_start}..{oos_end}",
    )
    combined_tearsheet_path = config.TEARSHEETS_DIR / f"{row['strategy_id']}_oos.html"
    fig.write_html(str(combined_tearsheet_path))

    return {
        "oos_sharpe": float(oos_result.metrics.get("sharpe", 0.0)),
        "oos_calmar": float(oos_result.metrics.get("calmar", 0.0)),
        "oos_max_dd": float(oos_result.metrics.get("max_drawdown", 0.0)),
        "oos_cagr": float(oos_result.metrics.get("cagr", 0.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--top", type=int, default=config.TOP_K_PROMOTE,
        help=f"Promote the top-K (by IS Calmar) un-evaluated entries. "
             f"Default {config.TOP_K_PROMOTE}.",
    )
    parser.add_argument(
        "--strategy-id", default=None,
        help="Force-evaluate a specific strategy_id (re-runs even if already OOS-evaluated).",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    selected = _select_for_promotion(args.top, args.strategy_id)
    if not selected:
        print("[promote] nothing to evaluate (leaderboard empty or all rows already scored).")
        return 0

    print(f"[promote] evaluating {len(selected)} strategy/strategies on OOS window")
    failures = 0
    for row in selected:
        sid = row["strategy_id"]
        try:
            print(f"  [{sid}] starting OOS evaluation...")
            oos_metrics = evaluate_oos(row)
            update_oos(sid, oos_metrics)
            print(
                f"  [{sid}] OOS Calmar={oos_metrics['oos_calmar']:+.2f} "
                f"Sharpe={oos_metrics['oos_sharpe']:+.2f} "
                f"MaxDD={oos_metrics['oos_max_dd']:.1%} "
                f"CAGR={oos_metrics['oos_cagr']:.1%}"
            )
        except Exception as exc:
            failures += 1
            sys.stderr.write(f"  [{sid}] FAILED: {exc}\n")
            traceback.print_exc()

    print(f"[promote] done — {len(selected) - failures} ok, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
