"""SP500 Sharpe-Ranked Momentum with Credit Gate — gen_8 sonnet-9

Hypothesis: Rank SP500 stocks by 63-day Sharpe ratio (daily return / daily
realized volatility). Hold top-20 equal-weight when JNK above 20d SMA
(credit risk-on) AND SPY above 200d SMA (trend filter). Rotate to TLT
when either gate fails. Monthly rebalance (21 bars).

Rationale: Sharpe-ranked selection rewards stocks with HIGH return AND LOW
vol — different from raw-return momentum which rewards high-return stocks
regardless of volatility. In the IS window (2010-2018, mostly bull),
low-vol high-return stocks compound better and suffer smaller drawdowns
than pure high-momentum picks. The dual JNK+SPY gate avoids equity
exposure during credit stress or bear markets.

Differentiation: Prior Sharpe-rank attempts (gen6_ic_fc7e3df0) failed at
corr gate (0.892 vs sp500_52wk_high_breakout). This version uses a MONTHLY
rebalance (21 bars) vs biweekly, and credit gate (JNK) instead of pure SPY
SMA. Also uses full 63d Sharpe window vs the prior 63d drawdown proxy.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

SHARPE_WINDOW = 63        # ~3 months of trading days
TREND_WINDOW = 200        # SPY 200d SMA trend gate
JNK_MA_WINDOW = 20        # JNK credit gate
TOP_K = 20
REBALANCE_DAYS = 21       # Monthly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["^VIX", "JNK", "TLT", "SPY"]


UNIVERSE = _universe


class Sp500SharpeRankedCreditGated(Strategy):
    """Top-20 SP500 stocks by 63d Sharpe ratio, dual JNK+SPY gate."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = TREND_WINDOW + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        # --- Credit gate: JNK above 20d SMA ---
        jnk_hist = ctx.history("JNK")
        credit_ok = False
        if len(jnk_hist) >= JNK_MA_WINDOW + 1:
            jnk_close = jnk_hist["close"]
            jnk_sma = float(jnk_close.iloc[-JNK_MA_WINDOW:].mean())
            jnk_price = float(jnk_close.iloc[-1])
            credit_ok = jnk_price > jnk_sma

        # --- Trend gate: SPY above 200d SMA ---
        spy_hist = ctx.history("SPY")
        trend_ok = False
        if len(spy_hist) >= TREND_WINDOW + 1:
            spy_close = spy_hist["close"]
            spy_sma = float(spy_close.iloc[-TREND_WINDOW:].mean())
            spy_price = float(spy_close.iloc[-1])
            trend_ok = spy_price > spy_sma

        closes = ctx.closes()
        if closes.empty:
            return []

        # --- Determine target ---
        if credit_ok and trend_ok:
            # Risk-on: rank SP500 stocks by 63d Sharpe ratio
            prices_window = ctx.closes_window(SHARPE_WINDOW + 5)
            if len(prices_window) < SHARPE_WINDOW:
                return []

            live = {s: float(closes[s]) for s in closes.index
                    if closes[s] > 0 and not s.startswith("^")
                    and s not in ("JNK", "TLT", "SPY")}

            sharpe_scores: dict[str, float] = {}
            for sym in live:
                if sym not in prices_window.columns:
                    continue
                col = prices_window[sym].dropna()
                if len(col) < SHARPE_WINDOW:
                    continue
                # Compute daily returns over the window
                daily_rets = col.pct_change().dropna()
                if len(daily_rets) < 20:
                    continue
                mean_ret = float(daily_rets.mean())
                std_ret = float(daily_rets.std())
                if std_ret <= 0 or not np.isfinite(std_ret):
                    continue
                # Annualized Sharpe (we rank on raw ratio, scale same for all)
                sharpe = mean_ret / std_ret
                if np.isfinite(sharpe):
                    sharpe_scores[sym] = sharpe

            if len(sharpe_scores) < TOP_K:
                # Fallback to TLT if not enough candidates
                target = {"TLT": EXPOSURE}
            else:
                ranked = sorted(sharpe_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [s for s, _ in ranked[:TOP_K]]
                target = {sym: EXPOSURE / len(selected) for sym in selected}
        else:
            # Risk-off: TLT
            target = {"TLT": EXPOSURE}

        # Compute portfolio equity
        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sp500_sharpe_ranked_credit_gated"
HYPOTHESIS = (
    "SP500 stocks ranked by 63d Sharpe ratio (return / realized vol) top-20 equal-weight; "
    "JNK 20d SMA credit gate AND SPY 200d SMA trend gate; TLT defensive; monthly rebalance; "
    "Sharpe-ranked selection orthogonal to raw-return momentum ranking."
)

STRATEGY = Sp500SharpeRankedCreditGated()
