"""Out-of-sample evaluation helpers.

The two functions here turn the playground from "did it work?" into
"did it work *and* beat the benchmark *and* generalize across regimes?".

- :func:`walk_forward` runs the strategy once over the full data range,
  then slices the resulting equity curve into ``window_years``-long
  windows and reports return-based metrics per window. Strategies that
  look great in one regime but blow up in another get caught here.

- :func:`compare_to_benchmark` contrasts a backtest result against
  buy-and-hold of a benchmark (default SPY) over the same date range.
  Tells the agent whether the strategy actually beats just holding the
  index.

Both functions operate on results from the existing ``Backtest`` engine
— no separate retraining or re-fitting. Stateful strategies retain their
trajectory across window boundaries (which is the realistic behavior).
"""
from __future__ import annotations

import pandas as pd

from stratlab.analytics.metrics import compute_metrics
from stratlab.engine.backtest import Backtest, BacktestResult
from stratlab.strategies.base import Strategy

_METRIC_KEYS = (
    "cagr", "sharpe", "sortino", "max_drawdown",
    "annual_volatility", "calmar", "win_rate",
)


def walk_forward(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    window_years: float = 1.0,
    initial_cash: float = 100_000.0,
    **bt_kwargs,
) -> pd.DataFrame:
    """Evaluate a strategy across non-overlapping rolling windows.

    Returns one row per window with start/end dates and return-based
    metrics (cagr, sharpe, sortino, max_drawdown, annual_volatility,
    win_rate). Trade-level stats stay on the overall ``BacktestResult``.

    Strategies see the full price history naturally — only the *scoring*
    is windowed. So a strategy that needs a 200-bar SMA warmup is fine;
    its early bars just won't contribute much equity change.
    """
    if not data:
        raise ValueError("walk_forward: data is empty")

    bt = Backtest(
        data=data, strategy=strategy, initial_cash=initial_cash, **bt_kwargs,
    )
    result = bt.run()

    window_bars = int(round(window_years * 252))
    if window_bars < 20:
        raise ValueError(f"window_years={window_years} is too small (<20 bars)")

    eq = result.equity_curve
    rets = result.returns
    rows = []
    for start_idx in range(0, len(eq) - window_bars + 1, window_bars):
        end_idx = start_idx + window_bars
        win_eq = eq.iloc[start_idx:end_idx]
        win_ret = rets.iloc[start_idx:end_idx]
        m = compute_metrics(win_eq, win_ret)
        rows.append({
            "start": win_eq.index[0].date().isoformat(),
            "end": win_eq.index[-1].date().isoformat(),
            **{k: m[k] for k in _METRIC_KEYS},
        })

    if not rows:
        raise ValueError(
            f"Equity curve ({len(eq)} bars) shorter than one window "
            f"of {window_bars} bars — reduce window_years or extend data."
        )
    return pd.DataFrame(rows)


def compare_to_benchmark(
    result: BacktestResult,
    benchmark: str | pd.Series = "SPY",
) -> pd.DataFrame:
    """Compare a backtest result to buy-and-hold of a benchmark.

    ``benchmark`` is either a ticker string (auto-loaded from the local
    cache for the result's date range) or a price ``Series``.

    Returns a DataFrame indexed by metric name with columns
    ``strategy``, ``benchmark``, and ``alpha`` (= strategy − benchmark).
    """
    if isinstance(benchmark, str):
        from stratlab.data.provider import load_bars
        idx = result.equity_curve.index
        bench_df = load_bars(
            benchmark,
            start=idx[0].date().isoformat(),
            end=idx[-1].date().isoformat(),
        )
        if bench_df.empty:
            raise ValueError(
                f"Benchmark '{benchmark}' has no cached data — run "
                f"`python -m stratlab.refresh --tickers {benchmark}` first."
            )
        bench_close = bench_df["close"]
    else:
        bench_close = benchmark

    bench_close = bench_close.reindex(result.equity_curve.index).ffill().dropna()
    if len(bench_close) < 2:
        raise ValueError("Benchmark has no overlap with result's date range")

    bench_eq = bench_close / bench_close.iloc[0]
    bench_ret = bench_eq.pct_change().fillna(0.0)
    bench_metrics = compute_metrics(bench_eq, bench_ret)

    rows = []
    for k in _METRIC_KEYS:
        s = float(result.metrics.get(k, float("nan")))
        b = float(bench_metrics.get(k, float("nan")))
        rows.append({"metric": k, "strategy": s, "benchmark": b, "alpha": s - b})
    return pd.DataFrame(rows).set_index("metric")
