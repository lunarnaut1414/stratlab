"""Pairwise IS-return correlation matrix for the leaderboard.

Usage::

    python -m stratlab.arena.corr_dump                       # top-12 by IS Calmar
    python -m stratlab.arena.corr_dump --top 20
    python -m stratlab.arena.corr_dump --strategies gen5_a gen6_b gen7_c
    python -m stratlab.arena.corr_dump --top 10 --csv corr.csv

Reads ``tmp/arena/returns_matrix.csv`` and emits a Pearson-correlation matrix
of IS daily returns. Used by the ensemble role (opus-3) to verify pairwise
<0.3 corr before building a multi-component ensemble. Asked for in gen_6,
gen_7, gen_8 wishlists — three rounds running.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stratlab.arena.leaderboard import read_leaderboard, read_returns_matrix, top_k_by


def pairwise_corr(
    strategy_ids: list[str],
    returns_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the Pearson correlation matrix of daily returns for the given
    strategy_ids. Order is preserved. IDs not in the returns matrix are
    silently dropped."""
    rm = returns_matrix if returns_matrix is not None else read_returns_matrix()
    if rm.empty:
        return pd.DataFrame()
    cols = [s for s in strategy_ids if s in rm.columns]
    if not cols:
        return pd.DataFrame()
    return rm[cols].corr(method="pearson")


def _resolve_strategy_ids(
    top: int,
    explicit: list[str] | None,
) -> list[str]:
    if explicit:
        return list(explicit)
    df = read_leaderboard()
    if df.empty:
        return []
    top_df = top_k_by(df, metric="is_calmar", k=top)
    return top_df["strategy_id"].dropna().tolist()


def _format_matrix(corr: pd.DataFrame, decimals: int = 2) -> str:
    """Pretty-print as a fixed-width text table. Truncates long ids to keep
    the layout readable in a terminal."""
    if corr.empty:
        return "(no overlapping strategies in returns matrix)"
    ids = list(corr.columns)
    short = {i: (i if len(i) <= 32 else i[:29] + "...") for i in ids}
    n = len(ids)
    col_w = 7  # fixed-width numeric columns: "+0.XX" plus padding
    lines: list[str] = []
    header = " " * 38 + "".join(f"{j:>{col_w}}" for j in range(1, n + 1))
    lines.append(header)
    for i, sid in enumerate(ids, 1):
        row_vals = "".join(
            f"{corr.loc[sid, c]:>+{col_w}.{decimals}f}" for c in ids
        )
        lines.append(f"{i:>3}. {short[sid]:<32}{row_vals}")
    lines.append("")
    lines.append("Legend (row number → strategy_id):")
    for i, sid in enumerate(ids, 1):
        lines.append(f"  {i:>3}. {sid}")
    return "\n".join(lines)


def _summary_stats(corr: pd.DataFrame) -> str:
    """Print summary stats over the off-diagonal entries (the actual pairs)."""
    if corr.empty or len(corr) < 2:
        return ""
    import numpy as np
    arr = corr.values.copy()
    mask = ~np.eye(arr.shape[0], dtype=bool)
    off = arr[mask]
    if off.size == 0:
        return ""
    lines = [
        "",
        "## Off-diagonal summary",
        f"  pairs analyzed : {off.size // 2}",
        f"  mean |corr|    : {abs(off).mean():.3f}",
        f"  median |corr|  : {float(pd.Series(abs(off)).median()):.3f}",
        f"  max |corr|     : {abs(off).max():.3f}",
        f"  min corr       : {off.min():+.3f}",
        f"  max corr       : {off.max():+.3f}",
        f"  pairs |corr|<0.3 : {int((abs(off) < 0.3).sum()) // 2}",
        f"  pairs |corr|>0.7 : {int((abs(off) > 0.7).sum()) // 2}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--top", type=int, default=12,
        help="Compute pairwise corrs over the top-K leaderboard rows by IS "
             "Calmar (default 12). Ignored if --strategies is given.",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Explicit list of strategy_ids to correlate. Overrides --top.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Write the matrix to this path instead of printing. Pretty output "
             "still prints to stderr summary.",
    )
    parser.add_argument(
        "--decimals", type=int, default=2,
        help="Decimals to print in the pretty matrix (default 2).",
    )
    args = parser.parse_args(argv)

    strategy_ids = _resolve_strategy_ids(args.top, args.strategies)
    if not strategy_ids:
        sys.stderr.write("[corr_dump] no strategies to correlate\n")
        return 1

    rm = read_returns_matrix()
    if rm.empty:
        sys.stderr.write("[corr_dump] returns_matrix.csv is empty or missing\n")
        return 1

    corr = pairwise_corr(strategy_ids, rm)
    if corr.empty:
        sys.stderr.write(
            "[corr_dump] none of the requested strategies are in returns_matrix.csv\n"
        )
        return 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(args.csv)
        sys.stderr.write(f"[corr_dump] wrote matrix to {args.csv}\n")
        sys.stderr.write(_summary_stats(corr) + "\n")
        return 0

    print(_format_matrix(corr, decimals=args.decimals))
    print(_summary_stats(corr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
