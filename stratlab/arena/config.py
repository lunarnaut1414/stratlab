"""Static rules for the strategies arena.

Single source of truth for the calendar split, gates, thresholds, and
filesystem paths. Both ``submit.py`` and ``promote.py`` import from
here so the protocol can't drift between entry points.

Calendar split (immutable for the lifetime of an arena run):

- IS window:  2010-01-01 to 2018-12-31  (~9 years)
  Used for ranking-during-search. Visible to generators.
- OOS window: 2019-01-01 to 2024-12-31  (~6 years)
  Frozen. Read ONLY by ``promote.py``. Generators never see OOS metrics.

Changing these mid-run invalidates previously-submitted strategies'
metrics — agents will be optimizing against shifted goalposts. If you
need to rerun on a different split, archive the existing
``tmp/arena/`` directory first and start clean.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# --- Calendar split (do not modify mid-run) -------------------------------
IS_START: date = date(2010, 1, 1)
IS_END: date = date(2018, 12, 31)
OOS_START: date = date(2019, 1, 1)
OOS_END: date = date(2026, 5, 1)
# OOS_END extended from 2024-12-31 to 2026-05-01 to use the most recent data
# available. IS-OOS split is preserved (IS unchanged at 2010-2018).


# --- Gates and thresholds -------------------------------------------------
MIN_TRADES_IS: int = 50
"""Minimum closed trades over the IS window. Strategies that fire fewer
than this are rejected — too few samples to distinguish skill from luck.
"""

MIN_CALMAR_IS: float = 0.5
"""Minimum IS Calmar ratio. Strategies below this floor are rejected outright
— the leaderboard isn't a graveyard of explored ideas, it's a curated
working set. Round-1 pilot showed agents will submit Calmar-0.1 strategies
when the rule is advisory; harness enforcement removes the temptation.
Postmortems for sub-floor attempts go to ``dead_ends.md``.
"""

CORR_REJECT_THRESHOLD: float = 0.85
"""Reject submissions whose IS daily-return correlation with any current
top-5 entry exceeds this. 0.85 is conventional in factor research — above
that, two strategies are usually doing the same thing in disguise.
"""

TOP_K_FOR_CORR_CHECK: int = 5
"""How many leaderboard entries (by IS Calmar) the corr filter compares
against. Set to 5 so generators have headroom to mutate established ideas
without colliding with the absolute leaders.
"""

TOP_K_PROMOTE: int = 10
"""How many top entries (by IS Calmar) ``promote.py`` evaluates on OOS by
default. Override with ``--top K`` on the CLI.
"""


# --- Backtest defaults ----------------------------------------------------
BENCHMARK_TICKER: str = "SPY"
DEFAULT_INITIAL_CASH: float = 100_000.0


# --- Filesystem paths -----------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

ARENA_DIR: Path = _REPO_ROOT / "tmp" / "arena"
LEADERBOARD: Path = ARENA_DIR / "leaderboard.csv"
RETURNS_MATRIX: Path = ARENA_DIR / "returns_matrix.csv"
TEARSHEETS_DIR: Path = ARENA_DIR / "tearsheets"
EQUITY_CURVES_DIR: Path = ARENA_DIR / "equity_curves"
DEAD_ENDS: Path = ARENA_DIR / "dead_ends.md"

STRATEGIES_DIR: Path = _REPO_ROOT / "stratlab" / "strategies" / "arena"


def is_window_str() -> tuple[str, str]:
    """ISO-format ``(start, end)`` strings for the IS window."""
    return IS_START.isoformat(), IS_END.isoformat()


def oos_window_str() -> tuple[str, str]:
    """ISO-format ``(start, end)`` strings for the OOS window."""
    return OOS_START.isoformat(), OOS_END.isoformat()


def ensure_dirs() -> None:
    """Create the arena and tearsheets directories on demand."""
    ARENA_DIR.mkdir(parents=True, exist_ok=True)
    TEARSHEETS_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
