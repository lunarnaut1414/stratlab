from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stratlab.engine.backtest import BacktestResult


def plot_equity(result: BacktestResult, title: str = "Equity Curve") -> None:
    """Plot the equity curve and drawdown using plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = result.equity_curve
    dd = (eq - eq.cummax()) / eq.cummax()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=[title, "Drawdown"],
    )

    fig.add_trace(
        go.Scatter(x=eq.index, y=eq.values, name="Equity", line=dict(color="#2962FF")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#FF6D00"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark", height=600, showlegend=False,
        yaxis_title="Portfolio Value ($)", yaxis2_title="Drawdown %",
        yaxis2_tickformat=".1%",
    )
    fig.show()
