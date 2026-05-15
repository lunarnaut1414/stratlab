"""Predict a candidate's IS-return correlation to top-K BEFORE submitting.

Usage::

    python -m stratlab.arena.corr_check <strategy_module_path>
    python -m stratlab.arena.corr_check <path> --top 10
    python -m stratlab.arena.corr_check <path> --exclude-gen 10
    python -m stratlab.arena.corr_check <path> --show-all

Runs the candidate's full IS backtest (same cost as submit) but does NOT:

  - write to the leaderboard
  - write to the returns matrix
  - generate a tearsheet
  - append to dead_ends.md
  - consume / mark an intent

Default output: IS Calmar (with 0.5 floor check), max-corr, **top-3 blockers
with their generation**, loss-mode correlation. Use this to iterate on a
candidate's signal mix until corr falls below the 0.85 rejection threshold
without burning intent slots or polluting dead_ends.md.

The --exclude-gen flag is useful when many concurrent same-round submissions
destabilize the top-5; you can iterate against the stable cross-round
baseline by excluding the current generation.

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
    exclude_gen: int | None = None,
    show_all: bool = False,
) -> dict:
    """Run the candidate's IS backtest and report max-corr to top-K.

    Returns a dict with keys: max_corr, twin_id, twin_gen, loss_mode_corr,
    loss_twin_id, is_calmar, is_sharpe, n_trades, top3_blockers (list of
    {strategy_id, generation, corr}), per_target_corrs (if show_all),
    passes_corr_filter, passes_calmar_floor.
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
    if exclude_gen is not None:
        leaderboard = leaderboard[
            leaderboard["generation"].astype(str) != str(exclude_gen)
        ]
    top_df = top_k_by(
        leaderboard,
        metric="is_calmar",
        k=top_k,
        require_n_trades=config.MIN_TRADES_IS,
    )
    target_ids = top_df["strategy_id"].dropna().tolist()
    # Build id -> generation map for output annotation.
    gen_by_id: dict[str, int] = {}
    for _, row in top_df.iterrows():
        sid = row.get("strategy_id")
        gen = row.get("generation")
        if isinstance(sid, str) and gen is not None:
            try:
                gen_by_id[sid] = int(gen)
            except (TypeError, ValueError):
                pass

    returns_matrix = read_returns_matrix()
    max_corr, twin_id = max_corr_to(result.returns, returns_matrix, target_ids)
    benchmark_returns = _load_benchmark_returns(is_start, is_end)
    loss_corr, loss_twin_id = loss_mode_corr_to(
        result.returns, returns_matrix, target_ids, benchmark_returns,
    )

    # Always compute per-target corrs so we can show top-3 blockers.
    per_target: dict[str, float] = {}
    common = result.returns.index.intersection(returns_matrix.index)
    if len(common) >= 30:
        a = result.returns.loc[common]
        for tid in target_ids:
            if tid not in returns_matrix.columns:
                continue
            b = returns_matrix[tid].loc[common]
            if a.std() == 0 or b.std() == 0:
                continue
            per_target[tid] = float(a.corr(b))

    top3 = sorted(per_target.items(), key=lambda x: -abs(x[1]))[:3]
    top3_blockers = [
        {"strategy_id": sid, "generation": gen_by_id.get(sid), "corr": c}
        for sid, c in top3
    ]

    is_calmar = float(result.metrics.get("calmar", 0.0))
    out = {
        "max_corr": float(max_corr),
        "twin_id": twin_id,
        "twin_gen": gen_by_id.get(twin_id),
        "loss_mode_corr": float(loss_corr),
        "loss_twin_id": loss_twin_id,
        "loss_twin_gen": gen_by_id.get(loss_twin_id),
        "is_calmar": is_calmar,
        "is_sharpe": float(result.metrics.get("sharpe", 0.0)),
        "n_trades": int(result.metrics.get("n_trades", 0)),
        "target_ids": target_ids,
        "top3_blockers": top3_blockers,
        "passes_corr_filter": abs(max_corr) <= config.CORR_REJECT_THRESHOLD,
        "passes_calmar_floor": is_calmar >= config.MIN_CALMAR_IS,
        "exclude_gen": exclude_gen,
    }
    if show_all:
        out["per_target_corrs"] = per_target
    return out


def _fmt_gen(gen: int | None) -> str:
    return f"gen_{gen}" if gen is not None else "?"


def _format(result: dict, strategy_path: Path) -> str:
    lines = [f"[corr_check] {strategy_path.name}"]
    cal_marker = "✓" if result["passes_calmar_floor"] else "✗ BELOW FLOOR"
    lines.append(
        f"  IS Calmar  : {result['is_calmar']:>+7.3f}  "
        f"({cal_marker} {config.MIN_CALMAR_IS} floor)"
    )
    lines.append(f"  IS Sharpe  : {result['is_sharpe']:>+7.3f}")
    lines.append(f"  n_trades   : {result['n_trades']:>7d}")
    lines.append("")
    n_targets = len(result["target_ids"])
    exclude_note = (
        f" (excluding gen_{result['exclude_gen']})"
        if result["exclude_gen"] is not None else ""
    )
    lines.append(f"  max |corr| to top-{n_targets}{exclude_note}: "
                 f"{result['max_corr']:>+7.3f}  "
                 f"({result['twin_id'] or '—'} | {_fmt_gen(result['twin_gen'])})")
    lines.append(f"  loss-mode corr  : {result['loss_mode_corr']:>+7.3f}  "
                 f"({result['loss_twin_id'] or '—'} | {_fmt_gen(result['loss_twin_gen'])})")

    threshold = config.CORR_REJECT_THRESHOLD
    if result["passes_corr_filter"]:
        lines.append(f"  ✓ passes corr filter (|corr| ≤ {threshold})")
    else:
        lines.append(
            f"  ✗ WOULD BE REJECTED (|corr| > {threshold} vs "
            f"{result['twin_id']!r} [{_fmt_gen(result['twin_gen'])}])"
            f" — try an uncorrelated angle"
        )

    if result["top3_blockers"]:
        lines.append("")
        lines.append("## Top-3 blockers (most-correlated targets)")
        for b in result["top3_blockers"]:
            lines.append(
                f"  {b['corr']:>+7.3f}  {b['strategy_id']}  ({_fmt_gen(b['generation'])})"
            )

    if "per_target_corrs" in result:
        lines.append("")
        lines.append("## All target correlations")
        for tid, c in sorted(result["per_target_corrs"].items(), key=lambda x: -abs(x[1])):
            lines.append(f"  {c:>+7.3f}  {tid}  ({_fmt_gen(result.get('exclude_gen') and -1)})")

    lines.append("")
    if not result["passes_calmar_floor"] and result["passes_corr_filter"]:
        lines.append("  ⚠ corr OK but IS Calmar below 0.5 floor — submit would exit 5")
    elif result["passes_calmar_floor"] and not result["passes_corr_filter"]:
        lines.append("  ⚠ Calmar OK but corr rejection imminent — mutate signal mix")

    lines.append("  (dry-run — leaderboard NOT updated, no intent consumed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "strategy_path", type=Path,
        help="Path to the strategy module (must export STRATEGY, NAME, HYPOTHESIS).",
    )
    parser.add_argument(
        "--top", "--top-k", type=int, default=config.TOP_K_FOR_CORR_CHECK,
        dest="top",
        help=f"How many top leaderboard rows to compare against. "
             f"Default {config.TOP_K_FOR_CORR_CHECK} (matches submit.py).",
    )
    parser.add_argument(
        "--exclude-gen", type=int, default=None,
        help="Exclude strategies from this generation when picking corr targets. "
             "Useful for iterating against the stable cross-round baseline "
             "without being destabilized by concurrent same-round submissions.",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Also print every target correlation (not just top-3).",
    )
    args = parser.parse_args(argv)

    try:
        result = corr_check(
            args.strategy_path,
            top_k=args.top,
            exclude_gen=args.exclude_gen,
            show_all=args.show_all,
        )
    except Exception as exc:
        sys.stderr.write(f"[corr_check] ERROR: {exc}\n")
        traceback.print_exc()
        return 1
    print(_format(result, args.strategy_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
