from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute standard performance metrics from an equity curve."""
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    n_years = len(equity) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

    excess = returns - risk_free_rate / periods_per_year
    sharpe = (
        float(np.sqrt(periods_per_year) * excess.mean() / excess.std())
        if excess.std() > 0
        else 0.0
    )

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min())

    downside = returns[returns < 0]
    sortino = (
        float(np.sqrt(periods_per_year) * excess.mean() / downside.std())
        if len(downside) > 0 and downside.std() > 0
        else 0.0
    )

    annual_vol = float(returns.std() * np.sqrt(periods_per_year))

    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "annual_volatility": round(annual_vol, 4),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": 0,
    }
