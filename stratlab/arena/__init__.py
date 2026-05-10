"""Strategies arena: agents generate, submit, and compete.

Two-tier architecture:

- **Generators** (low-cost LLM, e.g. Sonnet) write one strategy each, blind
  to the leaderboard and to other generators' submissions. Their submissions
  flow through :mod:`stratlab.arena.submit`, which validates against the
  in-sample window only and rejects high-correlation duplicates.

- **Curator** (heavyweight LLM, e.g. Opus) periodically reads the full
  leaderboard, runs OOS evaluation via :mod:`stratlab.arena.promote`, picks
  promotions, proposes ensembles, and writes a memo for the human running
  the arena.

The split is enforced by file conventions: generators only ever read
in-sample columns and never see ``tmp/arena/round_*/``; the curator reads
everything. Out-of-sample metrics live behind the :mod:`promote` entry
point so generators can't accidentally use them.

See ``docs/arena/PILOT.md`` for the manual run procedure and
``docs/arena/{generator,curator}_prompt.md`` for the agent prompts.
"""
from stratlab.arena.config import (
    ARENA_DIR,
    CORR_REJECT_THRESHOLD,
    IS_END,
    IS_START,
    LEADERBOARD,
    MIN_TRADES_IS,
    OOS_END,
    OOS_START,
    TOP_K_PROMOTE,
    ensure_dirs,
)
from stratlab.arena.leaderboard import (
    COLUMNS,
    GENERATOR_VISIBLE_COLUMNS,
    append_returns,
    append_row,
    max_corr_to,
    read_leaderboard,
    read_returns_matrix,
    top_k_by,
    update_oos,
)

__all__ = [
    "ARENA_DIR",
    "CORR_REJECT_THRESHOLD",
    "IS_END",
    "IS_START",
    "LEADERBOARD",
    "MIN_TRADES_IS",
    "OOS_END",
    "OOS_START",
    "TOP_K_PROMOTE",
    "COLUMNS",
    "GENERATOR_VISIBLE_COLUMNS",
    "append_returns",
    "append_row",
    "ensure_dirs",
    "max_corr_to",
    "read_leaderboard",
    "read_returns_matrix",
    "top_k_by",
    "update_oos",
]
