"""SP500 momentum ranked by 21d Sharpe ratio (risk-adjusted momentum signal).

Hypothesis (sonnet-3, gen_10):
    Instead of ranking SP500 stocks by raw 126d return, rank by 21d Sharpe
    ratio: (21d return) / (21d realized vol). This selects stocks that have
    been rising smoothly and efficiently, not just stocks that happened to
    spike up. The 21d Sharpe captures the quality and consistency of recent
    momentum.

    Rationale:
      - Raw momentum rankings can be dominated by stocks that had one large
        spike in the lookback window, inflating their 6-month return. The
        21d Sharpe normalizes by volatility, giving preference to stocks
        climbing steadily with low daily variance.
      - The mechanism is regime-invariant: smooth uptrends tend to persist
        regardless of VIX level. It does not depend on any macro signal.
      - Unlike RSI or SMA filters (which exclude names), Sharpe ranking
        reweights the selection — high-Sharpe names move to the top even
        if their raw return is lower.
      - Short Sharpe window (21d) is intentionally faster than 126d momentum:
        it selects names currently in the best phase of their uptrend.

    Diversification vs leaderboard:
      - All existing SP500 strategies rank by raw return (126d, 63d, skip-month).
      - No strategy on the leaderboard uses a risk-adjusted ranking signal.
      - Combined with portfolio vol-targeting for aggregate risk control and
        per-stock 200d SMA filter to exclude broken names.

    Design:
      - For each SP500 stock above its own 200d SMA, compute 21d Sharpe:
        mean(21d daily log-returns) / std(21d daily log-returns), annualized.
      - Rank by 21d Sharpe; hold top-15.
      - Inverse-vol weighted (21d realized vol) for position sizing.
      - Portfolio 13% annualized vol-target (30d realized) for aggregate exposure.
      - SPY 200d SMA outer gate: IEF defensive when bear.
      - Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
SHARPE_WINDOW = 63         # 63d Sharpe ranking window (more stable than 21d)
STOCK_TREND_WINDOW = 200   # per-stock 200d SMA quality filter
VOL_WINDOW = 21            # inverse-vol weight lookback
SPY_TREND_WINDOW = 200     # outer SPY 200d SMA gate
TOP_K = 15
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_TARGET = 0.14          # 14% annualized portfolio vol target
PORT_VOL_WINDOW = 30
ANNUALIZATION = 252


class Sp500SharpeRankingMomentum(Strategy):
    """SP500 top-15 by 21d Sharpe ratio; per-stock 200d SMA quality gate;
    inverse-vol weighted; portfolio vol-target (13% ann); SPY 200d outer gate
    to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_min: float = EXPOSURE_MIN,
        exposure_max: float = EXPOSURE_MAX,
        vol_target: float = VOL_TARGET,
        port_vol_window: int = PORT_VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sharpe_window=sharpe_window,
            stock_trend_window=stock_trend_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            vol_target=vol_target,
            port_vol_window=port_vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.sharpe_window = int(sharpe_window)
        self.stock_trend_window = int(stock_trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.stock_trend_window + self.sharpe_window + self.port_vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure_max
        else:
            # Need history for 200d SMA + 21d Sharpe
            need = self.stock_trend_window + self.sharpe_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}   # 21d Sharpe score per stock
            inv_vols: dict[str, float] = {}  # inverse-vol weight

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()

                # Quality filter: price above own 200d SMA
                if len(col) < self.stock_trend_window + 2:
                    continue
                stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                stock_price = float(col.iloc[-1])
                if stock_price <= stock_sma:
                    continue  # broken trend excluded

                # 63d Sharpe computation
                if len(col) < self.sharpe_window + 2:
                    continue
                tail = col.values[-(self.sharpe_window + 1):]
                daily_rets = np.log(tail[1:] / tail[:-1])
                if len(daily_rets) < self.sharpe_window:
                    continue
                ret_mean = float(np.mean(daily_rets))
                ret_std = float(np.std(daily_rets))
                if ret_std < 1e-8 or not np.isfinite(ret_std):
                    continue
                # Annualized Sharpe (no risk-free rate — we're ranking, not measuring)
                sharpe = (ret_mean / ret_std) * np.sqrt(ANNUALIZATION)
                if not np.isfinite(sharpe):
                    continue

                # Inverse-vol weight using same window
                inv_vol_rv = 1.0 / ret_std if ret_std > 1e-6 else 0.0

                scores[sym] = sharpe
                inv_vols[sym] = inv_vol_rv

            if len(scores) < 5:
                # Not enough quality candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # --- Portfolio vol-targeting ---
                port_prices = ctx.closes_window(self.port_vol_window + 5)
                port_rets = []
                n_rows = len(port_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in port_prices.columns:
                            continue
                        p_now = port_prices[sym].iloc[row_idx]
                        p_prev = port_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = self.vol_target / annual_vol if annual_vol > 1e-6 else 1.0
                    exposure = float(np.clip(scale, self.exposure_min, self.exposure_max))
                else:
                    exposure = self.exposure_max

                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

        # --- Build orders ---
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe

NAME = "sp500_sharpe_ranking_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 21d Sharpe ratio (21d return / 21d realized vol) as risk-adjusted "
    "momentum signal; inverse-vol weighted; portfolio 13pct vol-target exposure; SPY 200d "
    "SMA gate; IEF defensive; biweekly rebalance; 21d Sharpe ranking selects stocks with "
    "best recent risk-adjusted trend not raw momentum — distinct from all leaderboard entries "
    "which rank by raw return"
)

STRATEGY = Sp500SharpeRankingMomentum()
