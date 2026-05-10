"""Remove strategies from the arena leaderboard, returns matrix, and tearsheets.

Used to retract submissions that shouldn't have made it onto the leaderboard
(curator-flagged look-ahead leaks, sub-bailout junk that got through, etc.).
Idempotent — running twice on the same ID is safe.

Usage::

    python -m stratlab.arena.cleanup <strategy_id> [<strategy_id> ...] [--delete-source]

Without ``--delete-source``, the strategy module file (``stratlab/strategies/
arena/gen_*/<slug>.py``) is preserved; only the leaderboard / returns matrix /
tearsheets are cleared. With ``--delete-source``, the module file is also
deleted — use this for strategies with confirmed look-ahead bugs that
shouldn't ever be re-submitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stratlab.arena import config
from stratlab.arena.leaderboard import read_leaderboard, read_returns_matrix


def remove_strategy(
    strategy_id: str,
    *,
    delete_source: bool = False,
    leaderboard_path: Path | None = None,
    returns_path: Path | None = None,
    tearsheets_dir: Path | None = None,
) -> dict:
    """Remove a single strategy from all arena artifacts. Returns a dict
    summarizing what was removed."""
    leaderboard_path = leaderboard_path or config.LEADERBOARD
    returns_path = returns_path or config.RETURNS_MATRIX
    tearsheets_dir = tearsheets_dir or config.TEARSHEETS_DIR

    summary = {"strategy_id": strategy_id, "actions": []}

    # 1. Drop leaderboard row(s) — capture path for source deletion later.
    df = read_leaderboard(leaderboard_path)
    mask = df["strategy_id"] == strategy_id
    source_paths: list[str] = []
    if mask.any():
        source_paths = [p for p in df.loc[mask, "path"].dropna().tolist() if p]
        df = df[~mask]
        df.to_csv(leaderboard_path, index=False)
        summary["actions"].append(f"removed {int(mask.sum())} leaderboard row(s)")

    # 2. Drop returns column.
    rm = read_returns_matrix(returns_path)
    if not rm.empty and strategy_id in rm.columns:
        rm = rm.drop(columns=[strategy_id])
        if rm.shape[1] > 0:
            rm.to_csv(returns_path)
        else:
            returns_path.unlink(missing_ok=True)
        summary["actions"].append("removed returns column")

    # 3. Delete tearsheets (IS + OOS).
    for suffix in ("", "_oos"):
        ts = tearsheets_dir / f"{strategy_id}{suffix}.html"
        if ts.exists():
            ts.unlink()
            summary["actions"].append(f"deleted {ts.name}")

    # 4. Optionally delete the strategy source module.
    if delete_source:
        for sp in source_paths:
            p = Path(sp)
            if p.exists():
                p.unlink()
                summary["actions"].append(f"deleted source {p.name}")

    if not summary["actions"]:
        summary["actions"].append("not found — no changes")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "strategy_ids", nargs="+",
        help="One or more strategy_ids to remove.",
    )
    parser.add_argument(
        "--delete-source", action="store_true",
        help="Also delete the strategy module .py file (use for confirmed bugs).",
    )
    args = parser.parse_args(argv)

    failures = 0
    for sid in args.strategy_ids:
        try:
            summary = remove_strategy(sid, delete_source=args.delete_source)
            print(f"[cleanup] {sid}: {'; '.join(summary['actions'])}")
        except Exception as exc:
            failures += 1
            sys.stderr.write(f"[cleanup] {sid} FAILED: {exc}\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
