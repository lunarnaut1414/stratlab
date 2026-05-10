"""Bootstrap the arena leaderboard with gen-0 seed strategies.

Submits the three existing strategy templates as ``gen0`` entries:

- ``rsi_momentum``         — single-asset RSI mean-reversion on SPY
- ``low_vol_factor``       — cross-sectional low-volatility long basket on S&P 500
- ``momentum_plus_inverse``— long top-K momentum + tactical SH hedge

These are validated, well-understood strategies — they give agents a
known-good reference population to mutate from in gen 1+.

Usage::

    python -m stratlab.arena.seed [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stratlab.arena.config import STRATEGIES_DIR, ensure_dirs
from stratlab.arena.submit import submit


SEED_FILES = [
    "gen_0/rsi_momentum.py",
    "gen_0/low_vol_factor.py",
    "gen_0/momentum_plus_inverse.py",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without running backtests.",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    paths = [STRATEGIES_DIR / p for p in SEED_FILES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.stderr.write(f"[seed] missing seed files: {missing}\n")
        sys.stderr.write(
            "[seed] These should ship with the package — check the install.\n"
        )
        return 1

    if args.dry_run:
        print("[seed] would submit:")
        for p in paths:
            print(f"  - {p}")
        return 0

    failures = 0
    for p in paths:
        try:
            print(f"\n[seed] submitting {p.name}...")
            submit(p, gen=0, agent_id="seed", parent_id="", notes="gen-0 seed")
        except SystemExit as exc:
            # submit() exits 2/3 on rejection; treat as failure but keep going
            if exc.code != 0:
                failures += 1
        except Exception as exc:
            failures += 1
            sys.stderr.write(f"[seed] {p.name} FAILED: {exc}\n")

    print(f"\n[seed] done — {len(paths) - failures}/{len(paths)} seeds accepted")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
