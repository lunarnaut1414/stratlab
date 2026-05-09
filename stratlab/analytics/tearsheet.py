"""Multi-panel performance tearsheet for a backtest result.

Renders the standard set of panels you'd want when reviewing a strategy:

- Equity curve vs benchmark (normalized to 1.0)
- Underwater drawdown
- Monthly returns heatmap (year x month)
- Rolling 6-month Sharpe
- Round-trip trade scatter (return % vs holding days)

Built on plotly (already a project dep). Returns a ``go.Figure`` so the
caller can ``.show()`` interactively, ``.write_html(path)`` to save a
shareable file, or ``.write_image(path)`` for a PNG (requires kaleido).

The headline metrics (CAGR / Sharpe / MaxDD / Calmar) appear in the
figure title, so a saved tearsheet is self-describing without needing
the metrics dict alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.backtest import BacktestResult


def tearsheet(
    result: BacktestResult,
    benchmark: str | pd.Series | None = "SPY",
    title: str = "Strategy Tearsheet",
):
    """Build a multi-panel tearsheet for a ``BacktestResult``.

    ``benchmark`` is either a ticker string (auto-loaded from the local
    cache for the result's date range), a price ``Series``, or ``None``
    to suppress the benchmark overlay.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = result.equity_curve
    rets = result.returns
    if len(eq) < 2:
        raise ValueError("BacktestResult equity_curve has fewer than 2 points")

    bench_eq = _resolve_benchmark(benchmark, eq.index)

    m = result.metrics
    headline = (
        f"{title} &nbsp; — &nbsp; "
        f"CAGR <b>{m.get('cagr', 0):.1%}</b> &nbsp;|&nbsp; "
        f"Sharpe <b>{m.get('sharpe', 0):.2f}</b> &nbsp;|&nbsp; "
        f"MaxDD <b>{m.get('max_drawdown', 0):.1%}</b> &nbsp;|&nbsp; "
        f"Calmar <b>{m.get('calmar', 0):.2f}</b>"
    )

    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.40, 0.15, 0.25, 0.20],
        vertical_spacing=0.07, horizontal_spacing=0.10,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "Equity Curve (normalized to 1.0)",
            "Underwater Drawdown",
            "Monthly Returns",
            "Rolling 6-month Sharpe",
            "Round-trip Trades — return % vs holding days",
        ],
    )

    _add_equity_panel(fig, eq, bench_eq, row=1, col=1)
    _add_drawdown_panel(fig, eq, row=2, col=1)
    _add_monthly_heatmap(fig, rets, row=3, col=1)
    _add_rolling_sharpe(fig, rets, row=3, col=2)
    _add_trade_scatter(fig, result.trades, row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=1150,
        showlegend=True,
        title=dict(text=headline, x=0.01, xanchor="left", font=dict(size=14)),
        margin=dict(t=80, l=60, r=40, b=40),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return fig


def _resolve_benchmark(benchmark, idx: pd.DatetimeIndex) -> pd.Series | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, str):
        try:
            from stratlab.data.provider import load_bars
            df = load_bars(
                benchmark,
                start=idx[0].date().isoformat(),
                end=idx[-1].date().isoformat(),
            )
            if df.empty:
                return None
            close = df["close"]
        except Exception:
            return None
    else:
        close = benchmark
    close = close.reindex(idx).ffill().dropna()
    if len(close) < 2:
        return None
    return close / close.iloc[0]


def _add_equity_panel(fig, eq: pd.Series, bench_eq, row: int, col: int) -> None:
    import plotly.graph_objects as go

    norm = eq / eq.iloc[0]
    fig.add_trace(
        go.Scatter(x=norm.index, y=norm.values, name="Strategy",
                   line=dict(color="#2962FF", width=2)),
        row=row, col=col,
    )
    if bench_eq is not None:
        fig.add_trace(
            go.Scatter(x=bench_eq.index, y=bench_eq.values, name="Benchmark",
                       line=dict(color="#9E9E9E", width=1.5, dash="dot")),
            row=row, col=col,
        )


def _add_drawdown_panel(fig, eq: pd.Series, row: int, col: int) -> None:
    import plotly.graph_objects as go

    dd = (eq - eq.cummax()) / eq.cummax()
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                   fill="tozeroy", line=dict(color="#FF6D00"), showlegend=False),
        row=row, col=col,
    )


def _add_monthly_heatmap(fig, rets: pd.Series, row: int, col: int) -> None:
    import plotly.graph_objects as go

    if len(rets) == 0:
        return
    monthly = (1 + rets).resample("ME").prod() - 1.0
    pivot = monthly.to_frame("ret")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.month
    grid = pivot.pivot(index="year", columns="month", values="ret")
    grid = grid.reindex(columns=range(1, 13))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig.add_trace(
        go.Heatmap(
            z=grid.values * 100,
            x=month_labels,
            y=grid.index.astype(str),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="%", thickness=10, x=0.46, len=0.22, y=0.39),
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
            showscale=True,
        ),
        row=row, col=col,
    )


def _add_rolling_sharpe(fig, rets: pd.Series, row: int, col: int) -> None:
    import plotly.graph_objects as go

    if len(rets) < 130:
        return
    window = 126  # ~6 months
    rolling = rets.rolling(window).apply(
        lambda r: np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0,
        raw=False,
    )
    fig.add_trace(
        go.Scatter(x=rolling.index, y=rolling.values, name="Rolling Sharpe",
                   line=dict(color="#00C853"), showlegend=False),
        row=row, col=col,
    )
    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"),
                  row=row, col=col)


def _add_trade_scatter(fig, trades, row: int, col: int) -> None:
    import plotly.graph_objects as go

    if not trades:
        fig.add_annotation(
            text="(no closed round-trip trades)",
            xref=f"x{_axis_id(row, col)}", yref=f"y{_axis_id(row, col)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#888"), row=row, col=col,
        )
        return

    holding = np.array([
        max((t.exit_time - t.entry_time).days, 0) for t in trades
    ])
    rets = np.array([t.return_pct * 100 for t in trades])
    sides = np.array([t.side for t in trades])

    longs = sides == "long"
    shorts = sides == "short"
    if longs.any():
        fig.add_trace(
            go.Scatter(
                x=holding[longs], y=rets[longs], mode="markers",
                name="Long", marker=dict(color="#2962FF", size=7, opacity=0.7),
            ),
            row=row, col=col,
        )
    if shorts.any():
        fig.add_trace(
            go.Scatter(
                x=holding[shorts], y=rets[shorts], mode="markers",
                name="Short", marker=dict(color="#FF6D00", size=7, opacity=0.7),
            ),
            row=row, col=col,
        )
    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"),
                  row=row, col=col)
    fig.update_xaxes(title_text="Holding days", row=row, col=col)
    fig.update_yaxes(title_text="Return %", row=row, col=col)


def _axis_id(row: int, col: int) -> str:
    """Plotly subplot axis IDs are 1-based and serialized row-by-row."""
    n = (row - 1) * 2 + col
    return "" if n == 1 else str(n)
