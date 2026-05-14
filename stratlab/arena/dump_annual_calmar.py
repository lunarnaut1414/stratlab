"""Per-calendar-year return and Calmar for a strategy's equity curve.

Usage::

    python -m stratlab.arena.dump_annual_calmar <strategy_id>
    python -m stratlab.arena.dump_annual_calmar <strategy_id> --oos
    python -m stratlab.arena.dump_annual_calmar <strategy_id> --csv out.csv

Reads ``tmp/arena/equity_curves/<strategy_id>.csv`` (or the OOS variant) and
computes per-year total return, max drawdown within the year, and Calmar
(annual return / |max DD in year|). Useful for diagnosing WHICH years carry
or break a strategy — e.g. catching a -10% 2018 year hidden by strong
2012-2013-2017 returns. Asked for by sonnet-9 (gen_8), opus-4 (gen_6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stratlab.arena import config
from stratlab.arena.leaderboard import read_leaderboard


def _resolve_equity_path(strategy_id: str, oos: bool) -> Path:
    """Find the equity curve CSV — prefer the leaderboard's recorded path."""
    df = read_leaderboard()
    match = df[df["strategy_id"] == strategy_id]
    column = "equity_curve_oos_path" if oos else "equity_curve_path"
    if not match.empty:
        candidate = match.iloc[0].get(column, "")
        if isinstance(candidate, str) and candidate:
            p = Path(candidate)
            if p.exists():
                return p
    suffix = "_oos" if oos else ""
    fallback = config.EQUITY_CURVES_DIR / f"{strategy_id}{suffix}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"no {'OOS' if oos else 'IS'} equity curve for {strategy_id}; "
        f"checked leaderboard '{column}' and {fallback}"
    )


def annual_metrics(equity: pd.Series) -> pd.DataFrame:
    """Compute per-calendar-year {return, max_dd, calmar, end_value} from a
    daily equity series indexed by date.

    Calmar within a year = annual_return / |max DD within that year|.
    If max DD is 0 (monotonic year), Calmar is reported as inf.
    """
    if equity.empty:
        return pd.DataFrame()
    eq = equity.sort_index()
    years = eq.index.year.unique()
    rows: list[dict] = []
    for y in sorted(years):
        slc = eq[eq.index.year == y]
        if len(slc) < 2:
            continue
        start_val = float(slc.iloc[0])
        end_val = float(slc.iloc[-1])
        ann_ret = end_val / start_val - 1.0 if start_val != 0 else 0.0
        running_peak = slc.cummax()
        dd = slc / running_peak - 1.0
        max_dd = float(dd.min())
        if max_dd == 0.0:
            calmar = float("inf") if ann_ret > 0 else 0.0
        else:
            calmar = ann_ret / abs(max_dd)
        rows.append({
            "year": int(y),
            "return": ann_ret,
            "max_dd": max_dd,
            "calmar": calmar,
            "end_value": end_val,
            "n_bars": int(len(slc)),
        })
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame, strategy_id: str, window: str) -> str:
    if df.empty:
        return f"(no data in {window} equity curve for {strategy_id})"
    lines = [f"# {strategy_id} — {window} per-year metrics", ""]
    lines.append(f"  {'year':>4}  {'return':>8}  {'max_dd':>8}  {'calmar':>8}  {'end_value':>10}  {'n_bars':>7}")
    for _, r in df.iterrows():
        cal = r["calmar"]
        cal_s = f"{cal:>+8.2f}" if np.isfinite(cal) else f"{'+inf':>8}"
        lines.append(
            f"  {int(r['year']):>4}  {r['return']:>+7.1%}  {r['max_dd']:>+7.1%}  "
            f"{cal_s}  {r['end_value']:>10,.0f}  {int(r['n_bars']):>7d}"
        )
    # Summary
    rets = df["return"]
    cals_finite = df.loc[np.isfinite(df["calmar"]), "calmar"]
    lines.append("")
    lines.append("## Summary")
    lines.append(f"  positive years : {int((rets > 0).sum())} / {len(rets)}")
    lines.append(f"  worst year     : {rets.min():>+7.1%}")
    lines.append(f"  best year      : {rets.max():>+7.1%}")
    lines.append(f"  worst calmar   : {(cals_finite.min() if len(cals_finite) else float('nan')):>+8.2f}")
    lines.append(f"  median calmar  : {(cals_finite.median() if len(cals_finite) else float('nan')):>+8.2f}")
    lines.append(f"  worst max_dd   : {df['max_dd'].min():>+7.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("strategy_id")
    parser.add_argument(
        "--oos", action="store_true",
        help="Use the OOS equity curve instead of IS.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Write per-year metrics to this CSV instead of pretty-printing.",
    )
    args = parser.parse_args(argv)

    try:
        path = _resolve_equity_path(args.strategy_id, oos=args.oos)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[dump_annual_calmar] {exc}\n")
        return 1

    eq_df = pd.read_csv(path, index_col=0, parse_dates=True)
    eq = eq_df["equity"] if "equity" in eq_df.columns else eq_df.iloc[:, 0]
    metrics = annual_metrics(eq)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.csv, index=False)
        sys.stderr.write(f"[dump_annual_calmar] wrote {args.csv}\n")
        return 0

    window = "OOS" if args.oos else "IS"
    print(_format_table(metrics, args.strategy_id, window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
