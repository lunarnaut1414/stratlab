"""Dump a strategy's daily equity curve to stdout (or copy a path).

Usage::

    python -m stratlab.arena.dump_equity_curve <strategy_id>
    python -m stratlab.arena.dump_equity_curve <strategy_id> --path

The equity curve is persisted at submit time under
``tmp/arena/equity_curves/<strategy_id>.csv``. This CLI is a thin reader so
ad-hoc analysis (sub-period drawdown, regime decomposition, custom plotting)
doesn't have to re-run the backtest or scrape it from the tearsheet HTML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stratlab.arena import config
from stratlab.arena.leaderboard import read_leaderboard


def _resolve_path(strategy_id: str) -> Path:
    df = read_leaderboard()
    match = df[df["strategy_id"] == strategy_id]
    if match.empty:
        raise ValueError(f"strategy_id {strategy_id!r} not in leaderboard")
    candidate = match.iloc[0].get("equity_curve_path", "")
    if isinstance(candidate, str) and candidate:
        path = Path(candidate)
        if path.exists():
            return path
    fallback = config.EQUITY_CURVES_DIR / f"{strategy_id}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"no equity curve on disk for {strategy_id} — leaderboard row points to "
        f"{candidate!r}, fallback {fallback} does not exist. Older submissions "
        f"predate equity-curve persistence; re-submit to regenerate."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("strategy_id", help="Leaderboard strategy_id, e.g. gen5_vix_gated_sp500_momentum")
    parser.add_argument(
        "--path", action="store_true",
        help="Print only the path to the CSV (don't dump contents).",
    )
    args = parser.parse_args(argv)

    try:
        path = _resolve_path(args.strategy_id)
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"[dump_equity_curve] {exc}\n")
        return 1

    if args.path:
        print(path)
        return 0

    with path.open() as f:
        sys.stdout.write(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
