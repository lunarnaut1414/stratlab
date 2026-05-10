"""Leaderboard CSV + returns-matrix I/O for the strategies arena.

The leaderboard is a flat append-only CSV (written by submit.py, updated
in place by promote.py). The returns matrix is a parquet keyed by date
with one column per submitted strategy — used by submit.py for
correlation-based duplicate rejection and available to the curator for
post-hoc ensemble analysis.

Schema is hard-coded — agents shouldn't be able to drop or rename columns.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from stratlab.arena import config


COLUMNS: list[str] = [
    # Identity / lineage
    "strategy_id",       # gen{N}_{slug}
    "generation",        # int
    "parent_id",         # str (empty for seeds / unparented)
    "agent_id",          # which agent / human authored it
    "name",              # human-readable, from module's NAME
    "path",              # path to the strategy module
    "hypothesis",        # one-sentence rationale (from module's HYPOTHESIS)
    "params_json",       # JSON-encoded strategy.params
    "created_at",        # ISO timestamp
    # In-sample metrics — visible to generators
    "is_sharpe",
    "is_calmar",
    "is_sortino",
    "is_cagr",
    "is_max_dd",
    "is_annual_vol",
    "is_win_rate",
    "is_n_trades",
    "is_turnover",
    # Sub-period stability / regime concentration (populated at submit time)
    "is_calmar_h1",        # Calmar on first half of IS
    "is_calmar_h2",        # Calmar on second half of IS
    "is_calmar_min",       # min(h1, h2) — stability rank
    "is_pnl_top2y_pct",    # fraction of total log-PnL from best 2 calendar years
    # Out-of-sample metrics — populated only by promote.py
    "oos_sharpe",
    "oos_calmar",
    "oos_max_dd",
    "oos_cagr",
    "oos_evaluated_at",
    # Diversity / reporting
    "corr_to_top5",          # max abs corr to any current top-5 at submit time
    "loss_mode_corr_to_top5",  # max abs corr on bottom-decile SPY days
    "tearsheet_path",
    "equity_curve_path",     # path to per-strategy daily equity CSV
    "notes",
]

GENERATOR_VISIBLE_COLUMNS: list[str] = [
    "strategy_id", "generation", "parent_id", "name", "hypothesis",
    "is_sharpe", "is_calmar", "is_sortino", "is_cagr",
    "is_max_dd", "is_annual_vol", "is_win_rate", "is_n_trades",
    "is_turnover", "is_calmar_h1", "is_calmar_h2", "is_calmar_min",
    "is_pnl_top2y_pct", "corr_to_top5", "loss_mode_corr_to_top5",
    "created_at",
]
"""Columns generators are allowed to see when constructing prompts.
Notably absent: every ``oos_*`` column."""


def read_leaderboard(path: Path | None = None) -> pd.DataFrame:
    """Read the leaderboard CSV, returning an empty schema-shaped frame
    if the file doesn't exist yet."""
    path = path or config.LEADERBOARD
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS]


def append_row(row: dict, path: Path | None = None) -> None:
    """Append a single row to the leaderboard CSV, enforcing schema order."""
    path = path or config.LEADERBOARD
    df = read_leaderboard(path)
    full_row = {c: row.get(c, pd.NA) for c in COLUMNS}
    new_row_df = pd.DataFrame([full_row])
    df_new = new_row_df if df.empty else pd.concat([df, new_row_df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(path, index=False)


def update_oos(
    strategy_id: str,
    oos_metrics: dict,
    path: Path | None = None,
) -> None:
    """Update the OOS columns for one strategy in place. Stamps
    ``oos_evaluated_at`` with the current time."""
    path = path or config.LEADERBOARD
    df = read_leaderboard(path)
    mask = df["strategy_id"] == strategy_id
    if not mask.any():
        raise ValueError(f"strategy_id {strategy_id!r} not in leaderboard")
    # Cast string-typed cols to object so timestamp/text writes don't fight a
    # NaN-inferred float64 dtype.
    df["oos_evaluated_at"] = df["oos_evaluated_at"].astype(object)
    for col, val in oos_metrics.items():
        if col not in COLUMNS:
            raise ValueError(f"unknown leaderboard column {col!r}")
        df.loc[mask, col] = val
    df.loc[mask, "oos_evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    df.to_csv(path, index=False)


def top_k_by(
    df: pd.DataFrame,
    metric: str = "is_calmar",
    k: int = 5,
    *,
    require_n_trades: int | None = None,
) -> pd.DataFrame:
    """Top-k rows by a numeric metric. Optionally pre-filter on n_trades."""
    if df.empty:
        return df
    out = df.copy()
    if require_n_trades is not None:
        n_trades = out["is_n_trades"].fillna(0).astype(float)
        out = out[n_trades >= require_n_trades]
    return out.sort_values(metric, ascending=False).head(k)


# --- Returns matrix (parquet) --------------------------------------------

def read_returns_matrix(path: Path | None = None) -> pd.DataFrame:
    """Read the daily-returns matrix. Empty frame if file is missing."""
    path = path or config.RETURNS_MATRIX
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def append_returns(
    strategy_id: str,
    returns: pd.Series,
    path: Path | None = None,
) -> None:
    """Add (or replace) one strategy's daily-returns column.

    Date index is the union of existing and new — gaps fill with 0.0
    so cross-strategy correlation has a defined value on every day.
    """
    path = path or config.RETURNS_MATRIX
    existing = read_returns_matrix(path)
    s = returns.copy()
    s.name = strategy_id
    if existing.empty:
        merged = s.to_frame()
    else:
        all_idx = existing.index.union(s.index).sort_values()
        existing = existing.reindex(all_idx).fillna(0.0)
        existing[strategy_id] = s.reindex(all_idx).fillna(0.0)
        merged = existing
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path)


def loss_mode_corr_to(
    new_returns: pd.Series,
    existing: pd.DataFrame,
    target_ids: Iterable[str],
    benchmark_returns: pd.Series,
    *,
    quantile: float = 0.10,
    min_overlap: int = 20,
) -> tuple[float, str]:
    """Maximum absolute correlation against ``target_ids`` computed only on
    days when the benchmark is in the bottom ``quantile`` of returns.

    Two strategies can have low overall daily-return correlation while sharing
    an identical loss-mode (e.g., both rely on TLT-as-defensive — daily corr
    ≈0.57, but they crash together when bond-equity correlation flips). This
    metric isolates the stress-day correlation that ``max_corr_to`` averages
    away. Lower min_overlap because conditional samples are sparse by design.
    """
    if benchmark_returns is None or benchmark_returns.empty:
        return 0.0, ""
    valid_ids = [t for t in target_ids if t in existing.columns]
    if not valid_ids:
        return 0.0, ""

    cutoff = float(benchmark_returns.quantile(quantile))
    stress_days = benchmark_returns.index[benchmark_returns <= cutoff]
    common_idx = (
        new_returns.index
        .intersection(existing.index)
        .intersection(stress_days)
    )
    if len(common_idx) < min_overlap:
        return 0.0, ""

    new_aligned = new_returns.loc[common_idx]
    if new_aligned.std() == 0:
        return 0.0, ""

    best_id, best_corr = "", 0.0
    for col in valid_ids:
        col_aligned = existing[col].loc[common_idx]
        if col_aligned.std() == 0:
            continue
        c = float(new_aligned.corr(col_aligned))
        if abs(c) > abs(best_corr):
            best_corr = c
            best_id = col
    return best_corr, best_id


def max_corr_to(
    new_returns: pd.Series,
    existing: pd.DataFrame,
    target_ids: Iterable[str],
    *,
    min_overlap: int = 30,
) -> tuple[float, str]:
    """Maximum absolute Pearson correlation between ``new_returns`` and
    each of ``target_ids`` in ``existing``. Returns ``(corr, id_of_max)``;
    ``(0.0, "")`` if there's no comparison set or insufficient overlap.
    """
    valid_ids = [t for t in target_ids if t in existing.columns]
    if not valid_ids:
        return 0.0, ""
    common_idx = new_returns.index.intersection(existing.index)
    if len(common_idx) < min_overlap:
        return 0.0, ""
    new_aligned = new_returns.loc[common_idx]
    if new_aligned.std() == 0:
        return 0.0, ""
    best_id, best_corr = "", 0.0
    for col in valid_ids:
        col_aligned = existing[col].loc[common_idx]
        if col_aligned.std() == 0:
            continue
        c = float(new_aligned.corr(col_aligned))
        if abs(c) > abs(best_corr):
            best_corr = c
            best_id = col
    return best_corr, best_id
