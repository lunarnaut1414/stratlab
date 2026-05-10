"""Re-run a leaderboard strategy on the IS window and dump its full
trade-by-trade log to CSV.

Lets you inspect what a strategy actually *did* — entry/exit prices,
holding period, per-trade PnL — instead of just aggregate metrics.

Usage::

    python -m stratlab.arena.dump_trades <strategy_id> [--oos] [--out PATH]

Default output: ``tmp/arena/trades/<strategy_id>.csv`` (IS) or
``<strategy_id>_oos.csv`` (with ``--oos``). One row per round-trip trade.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stratlab.arena import config
from stratlab.arena.leaderboard import read_leaderboard
from stratlab.arena.submit import load_strategy_module, resolve_universe
from stratlab.engine.backtest import Backtest


def dump_trades(
    strategy_id: str,
    *,
    use_oos: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Re-run a strategy and write its trade log to CSV. Returns the
    output path."""
    df = read_leaderboard()
    row = df[df["strategy_id"] == strategy_id]
    if row.empty:
        raise ValueError(f"strategy_id {strategy_id!r} not in leaderboard")
    row = row.iloc[0]

    module = load_strategy_module(Path(row["path"]))
    tickers = resolve_universe(getattr(module, "UNIVERSE", "sp500"))

    if use_oos:
        start, end = config.oos_window_str()
    else:
        start, end = config.is_window_str()

    from stratlab.data.universe import load_universe
    data = load_universe(tickers, start=start, end=end)
    if not data:
        raise RuntimeError(
            f"no data for {strategy_id} {start}..{end} — refresh and retry"
        )

    bt = Backtest(
        data=data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
    )
    result = bt.run()

    rows = []
    for t in result.trades:
        rows.append({
            "symbol": t.symbol,
            "side": t.side,
            "entry_time": t.entry_time.date().isoformat(),
            "exit_time": t.exit_time.date().isoformat(),
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4),
            "size": t.size,
            "gross_pnl": round(t.gross_pnl, 2),
            "return_pct": round(t.return_pct, 4),
            "holding_days": (t.exit_time - t.entry_time).days,
        })
    trades_df = pd.DataFrame(rows)

    out_path = out_path or (
        config.ARENA_DIR / "trades"
        / f"{strategy_id}{'_oos' if use_oos else ''}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_path, index=False)

    n = len(trades_df)
    if n:
        wins = (trades_df["gross_pnl"] > 0).sum()
        total_pnl = trades_df["gross_pnl"].sum()
        print(f"[dump_trades] {strategy_id} ({'OOS' if use_oos else 'IS'})")
        print(f"  trades:    {n}")
        print(f"  win rate:  {wins/n:.1%} ({wins}/{n})")
        print(f"  total pnl: ${total_pnl:,.2f}")
        print(f"  written to: {out_path}")
    else:
        print(f"[dump_trades] {strategy_id}: no closed round-trip trades")

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("strategy_id", help="strategy_id from the leaderboard")
    parser.add_argument(
        "--oos", action="store_true",
        help="Re-run on the OOS window instead of IS.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: tmp/arena/trades/<strategy_id>[_oos].csv)",
    )
    args = parser.parse_args(argv)

    try:
        dump_trades(args.strategy_id, use_oos=args.oos, out_path=args.out)
    except Exception as exc:
        sys.stderr.write(f"[dump_trades] FAILED: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
