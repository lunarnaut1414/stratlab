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

import pandas as pd

from stratlab.analytics.metrics import compute_metrics, compute_period_returns
from stratlab.analytics.tearsheet import tearsheet_combined
from stratlab.arena import config
from stratlab.arena.config import ensure_dirs, is_window_str, oos_window_str
from stratlab.arena.leaderboard import (
    read_leaderboard,
    top_k_by,
    update_oos,
)
from stratlab.arena.submit import load_strategy_module, resolve_universe
from stratlab.engine.backtest import Backtest, BacktestResult


# Warmup buffer for OOS: many strategies need a 252-day lookback (52-week
# high, beta, long SMA). Without it, the first ~year of OOS shows no trades
# while the strategy waits for enough history. 500 calendar days covers any
# reasonable lookback (252 trading days ≈ 365 calendar days, plus headroom).
_OOS_WARMUP_CALENDAR_DAYS = 500


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


def _run_window(module, universe_spec, start: str, end: str, *, warmup_days: int = 0):
    """Backtest ``module.STRATEGY`` over ``[start, end]``.

    When ``warmup_days > 0``, data is pre-loaded from ``start - warmup_days``
    so the strategy has lookback history available on day 1 of ``[start,
    end]``. The result is then sliced back to the requested window so
    metrics, equity curve, fills, and trades all reflect only the
    user-visible interval. Without this, strategies with a 252-day lookback
    (52-week high, beta, long SMA) sit silent for the first ~year of the
    backtest while history accumulates.
    """
    from stratlab.data.inception import filter_universe_by_window_overlap
    from stratlab.data.universe import load_universe

    if warmup_days > 0:
        load_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).date().isoformat()
    else:
        load_start = start

    tickers = resolve_universe(universe_spec)
    tickers = filter_universe_by_window_overlap(tickers, start=load_start, end=end)
    data = load_universe(tickers, start=load_start, end=end)
    if not data:
        raise RuntimeError(f"no data loaded for {load_start}..{end}")

    bt = Backtest(
        data=data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
    )
    result = bt.run()
    if warmup_days > 0:
        result = _slice_result(result, start, end, config.DEFAULT_INITIAL_CASH)
    return result


def _slice_result(
    result: BacktestResult,
    start: str,
    end: str,
    initial_cash: float,
) -> BacktestResult:
    """Trim a BacktestResult to ``[start, end]`` and rebase metrics.

    Equity curve gets multiplicatively rescaled so the first in-window bar
    starts at ``initial_cash`` (otherwise CAGR would be computed off the
    warmup-end equity, hiding any warmup-period drift). Fills and trades
    are filtered by their timestamps so the tearsheet shows only in-window
    activity. Metrics are recomputed on the sliced series.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    eq = result.equity_curve
    eq_idx = eq.index >= start_ts
    eq_idx &= eq.index <= end_ts
    eq_sliced = eq[eq_idx]
    if len(eq_sliced) < 2:
        # Not enough in-window data — return unchanged rather than crash.
        return result
    eq_rebased = eq_sliced * (initial_cash / float(eq_sliced.iloc[0]))

    rets_sliced = eq_rebased.pct_change().fillna(0.0)

    fills_sliced = [f for f in result.fills if start_ts <= f.timestamp <= end_ts]
    trades_sliced = [
        t for t in result.trades
        if start_ts <= t.entry_time <= end_ts and start_ts <= t.exit_time <= end_ts
    ]

    metrics = compute_metrics(eq_rebased, rets_sliced)
    metrics["n_trades"] = len(fills_sliced)
    # Carry over engine-side counters that compute_metrics doesn't know about,
    # falling back to the original result where the slice doesn't change them.
    for key in ("dropped_orders", "borrow_cost", "turnover_annualized"):
        if key in result.metrics:
            metrics[key] = result.metrics[key]

    return BacktestResult(
        equity_curve=eq_rebased,
        returns=rets_sliced,
        fills=fills_sliced,
        trades=trades_sliced,
        metrics=metrics,
    )


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
    oos_result = _run_window(
        module, universe_spec, oos_start, oos_end,
        warmup_days=_OOS_WARMUP_CALENDAR_DAYS,
    )

    fig = tearsheet_combined(
        is_result, oos_result,
        benchmark=config.BENCHMARK_TICKER,
        title=f"{row['strategy_id']} — IS {is_start}..{is_end}  ·  OOS {oos_start}..{oos_end}",
    )
    combined_tearsheet_path = config.TEARSHEETS_DIR / f"{row['strategy_id']}_oos.html"
    fig.write_html(str(combined_tearsheet_path))

    oos_curve_path = config.EQUITY_CURVES_DIR / f"{row['strategy_id']}_oos.csv"
    oos_result.equity_curve.to_frame(name="equity").to_csv(oos_curve_path)

    stitched = _stitch_equity(is_result.equity_curve, oos_result.equity_curve)
    period_returns = compute_period_returns(stitched)

    out: dict = {
        "oos_sharpe": float(oos_result.metrics.get("sharpe", 0.0)),
        "oos_calmar": float(oos_result.metrics.get("calmar", 0.0)),
        "oos_max_dd": float(oos_result.metrics.get("max_drawdown", 0.0)),
        "oos_cagr": float(oos_result.metrics.get("cagr", 0.0)),
        "equity_curve_oos_path": str(oos_curve_path),
    }
    out.update(period_returns)
    return out


def _stitch_equity(is_eq: pd.Series, oos_eq: pd.Series) -> pd.Series:
    """Concatenate IS and OOS equity curves into a continuous lifetime track.

    OOS gets multiplicatively rescaled so its first bar continues from where
    IS ended — investors see one unbroken curve from inception to today,
    matching how a fund track record would be presented.
    """
    if is_eq.empty:
        return oos_eq
    if oos_eq.empty:
        return is_eq
    is_end_val = float(is_eq.iloc[-1])
    oos_start_val = float(oos_eq.iloc[0])
    if oos_start_val <= 0:
        return is_eq
    rescaled_oos = oos_eq * (is_end_val / oos_start_val)
    # Keep IS bars; drop any OOS bars that fall on or before IS's last date.
    oos_after = rescaled_oos[rescaled_oos.index > is_eq.index[-1]]
    return pd.concat([is_eq, oos_after])


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
