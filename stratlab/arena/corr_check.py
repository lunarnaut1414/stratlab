"""Predict a candidate's IS-return correlation to top-5 BEFORE submitting.

Usage::

    python -m stratlab.arena.corr_check <strategy_module_path>
    python -m stratlab.arena.corr_check <path> --top 5 --show-all

Runs the candidate's full IS backtest (same cost as submit) but does NOT:

  - write to the leaderboard
  - write to the returns matrix
  - generate a tearsheet
  - append to dead_ends.md
  - consume / mark an intent

The output is: max |corr| to current top-K leaderboard rows (default K=5),
the twin strategy_id that drives that max, plus the loss-mode correlation
(stress-day corr) for context. Use this to iterate on a candidate's signal
mix until corr falls below the 0.85 rejection threshold without burning
intent slots or polluting dead_ends.md.

Asked for across multiple gen_8 agents (sonnet-4, sonnet-6, sonnet-9,
sonnet-10, opus-3) — second-highest-recurrence wishlist item.

Limit: this is NOT faster than submit (the backtest itself is the bulk of
the cost). Use stratlab.arena.is_calmar_estimate first to filter weak
candidates before paying the full-IS backtest cost.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from stratlab.arena import config
from stratlab.arena.config import is_window_str
from stratlab.arena.leaderboard import (
    loss_mode_corr_to,
    max_corr_to,
    read_leaderboard,
    read_returns_matrix,
    top_k_by,
)
from stratlab.arena.submit import (
    _load_benchmark_returns,
    load_strategy_module,
    resolve_universe,
)
from stratlab.data.inception import filter_universe_by_window_overlap
from stratlab.data.universe import load_universe
from stratlab.engine.backtest import Backtest


def corr_check(
    strategy_path: Path,
    top_k: int = 5,
    show_all: bool = False,
) -> dict:
    """Run the candidate's IS backtest and report max-corr to top-K.

    Returns a dict with keys: max_corr, twin_id, loss_mode_corr, loss_twin_id,
    is_calmar (informational), n_trades, per_target_corrs (if show_all),
    plus a `passes_corr_filter` bool.
    """
    module = load_strategy_module(strategy_path)
    universe_spec = getattr(module, "UNIVERSE", "sp500")
    tickers = resolve_universe(universe_spec)

    is_start, is_end = is_window_str()
    pre_count = len(tickers)
    tickers = filter_universe_by_window_overlap(tickers, start=is_start, end=is_end)
    if pre_count != len(tickers):
        sys.stderr.write(
            f"[corr_check] universe filtered: {len(tickers)}/{pre_count} "
            f"tickers cached in {is_start}..{is_end}\n"
        )

    data = load_universe(tickers, start=is_start, end=is_end)
    if not data:
        raise RuntimeError(f"no IS data loaded for {universe_spec!r}")

    bt = Backtest(
        data=data,
        strategy=module.STRATEGY,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        allow_short=False,
    )
    result = bt.run()

    leaderboard = read_leaderboard()
    top_df = top_k_by(
        leaderboard,
        metric="is_calmar",
        k=top_k,
        require_n_trades=config.MIN_TRADES_IS,
    )
    target_ids = top_df["strategy_id"].dropna().tolist()
    returns_matrix = read_returns_matrix()

    max_corr, twin_id = max_corr_to(result.returns, returns_matrix, target_ids)
    benchmark_returns = _load_benchmark_returns(is_start, is_end)
    loss_corr, loss_twin_id = loss_mode_corr_to(
        result.returns, returns_matrix, target_ids, benchmark_returns,
    )

    out = {
        "max_corr": float(max_corr),
        "twin_id": twin_id,
        "loss_mode_corr": float(loss_corr),
        "loss_twin_id": loss_twin_id,
        "is_calmar": float(result.metrics.get("calmar", 0.0)),
        "is_sharpe": float(result.metrics.get("sharpe", 0.0)),
        "n_trades": int(result.metrics.get("n_trades", 0)),
        "target_ids": target_ids,
        "passes_corr_filter": abs(max_corr) <= config.CORR_REJECT_THRESHOLD,
    }
    if show_all:
        per_target: dict[str, float] = {}
        for tid in target_ids:
            if tid not in returns_matrix.columns:
                continue
            common = result.returns.index.intersection(returns_matrix.index)
            if len(common) < 30:
                continue
            a = result.returns.loc[common]
            b = returns_matrix[tid].loc[common]
            if a.std() == 0 or b.std() == 0:
                continue
            per_target[tid] = float(a.corr(b))
        out["per_target_corrs"] = per_target
    return out


def _format(result: dict, strategy_path: Path) -> str:
    lines = [f"[corr_check] {strategy_path.name}"]
    lines.append(f"  IS Calmar (info) : {result['is_calmar']:>+7.3f}")
    lines.append(f"  IS Sharpe (info) : {result['is_sharpe']:>+7.3f}")
    lines.append(f"  n_trades         : {result['n_trades']:>7d}")
    lines.append("")
    lines.append(f"  max |corr| to top-{len(result['target_ids'])} : "
                 f"{result['max_corr']:>+7.3f}  ({result['twin_id'] or '—'})")
    lines.append(f"  loss-mode corr    : {result['loss_mode_corr']:>+7.3f}  "
                 f"({result['loss_twin_id'] or '—'})")
    threshold = config.CORR_REJECT_THRESHOLD
    if result["passes_corr_filter"]:
        lines.append(f"  ✓ passes filter (|corr| ≤ {threshold})")
    else:
        lines.append(f"  ✗ WOULD BE REJECTED (|corr| > {threshold} vs "
                     f"{result['twin_id']!r}) — try an uncorrelated angle")
    if "per_target_corrs" in result:
        lines.append("")
        lines.append("## Per-target correlations")
        for tid, c in sorted(result["per_target_corrs"].items(), key=lambda x: -abs(x[1])):
            lines.append(f"  {c:>+7.3f}  {tid}")
    lines.append("")
    lines.append("  (dry-run — leaderboard NOT updated, no intent consumed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "strategy_path", type=Path,
        help="Path to the strategy module (must export STRATEGY, NAME, HYPOTHESIS).",
    )
    parser.add_argument(
        "--top", type=int, default=config.TOP_K_FOR_CORR_CHECK,
        help=f"How many top leaderboard rows to compare against. "
             f"Default {config.TOP_K_FOR_CORR_CHECK} (matches submit.py).",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Also print per-target correlations sorted by |corr|.",
    )
    args = parser.parse_args(argv)

    try:
        result = corr_check(args.strategy_path, top_k=args.top, show_all=args.show_all)
    except Exception as exc:
        sys.stderr.write(f"[corr_check] ERROR: {exc}\n")
        traceback.print_exc()
        return 1
    print(_format(result, args.strategy_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
