"""Popular-ETF momentum ranked by 63d Sharpe ratio with portfolio vol-targeting.

Hypothesis (sonnet-3, gen_10):
    Use the popular_etfs universe (broad mix of equity, bond, commodity, factor,
    sector ETFs) rather than SP500 stocks. Rank ETFs by 63d Sharpe ratio
    (63d return / 63d realized vol) as a risk-adjusted momentum signal. Hold
    top-5 ETFs with positive Sharpe above their own 200d SMA. Portfolio vol-
    targeting scales aggregate exposure to 12% annualized vol.

    Rationale:
      - ETF universe provides natural diversification across asset classes
        (equities, bonds, commodities, gold, sectors) vs SP500-only strategies.
      - Sharpe ranking (not raw return) prevents chasing high-volatility ETFs
        that have spiked — selects ETFs with smooth consistent uptrends.
      - The popular_etfs universe includes TLT, GLD, IEF, QQQ, SPY, sector
        ETFs, factor ETFs — momentum rotation across these provides very
        different signal timing than SP500 cross-sectional momentum.
      - Portfolio vol-targeting provides regime-invariant risk control without
        any macro signal (VIX, yield curve, credit spread) dependence.

    Diversification vs leaderboard:
      - All gen_10 strategies use SP500 stocks. This uses popular_etfs.
      - gen5_atr_momentum_etf (failed: IS 0.09), gen5_rsi_etf_meanrev (failed).
      - Sharpe-ranking on ETFs + vol-target is untested on the leaderboard.

    Design:
      - For each ETF in popular_etfs, compute 63d Sharpe: mean(daily returns) /
        std(daily returns), annualized. Require positive Sharpe + price above 200d SMA.
      - Hold top-5 ETFs by 63d Sharpe, inverse-vol weighted.
      - Portfolio vol-target: 12% annualized (21d realized), clipped 40-97%.
      - Rebalance every 5 bars (weekly) for higher trade count in diverse ETF universe.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SHARPE_WINDOW = 63         # 63d risk-adjusted momentum
STOCK_TREND_WINDOW = 200   # per-ETF 200d SMA quality gate
VOL_WINDOW = 21            # inverse-vol weight lookback
TOP_K = 5                  # top-5 ETFs
EXPOSURE_MIN = 0.40
EXPOSURE_MAX = 0.97
VOL_TARGET = 0.12          # 12% annualized portfolio vol target
PORT_VOL_WINDOW = 21       # realized vol lookback
ANNUALIZATION = 252


class EtfMomentumSharpeVoltarget(Strategy):
    """Popular-ETF top-5 by 63d Sharpe ratio; per-ETF 200d SMA quality gate;
    inverse-vol weighted; portfolio vol-target (12% ann); weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
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

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Need history for all windows
        need = self.stock_trend_window + self.sharpe_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < need - 10:
            return []

        scores: dict[str, float] = {}   # 63d Sharpe per ETF
        inv_vols: dict[str, float] = {}  # inverse-vol weight

        for sym in prices.columns:
            col = prices[sym].dropna()

            # Need enough data
            if len(col) < self.stock_trend_window + 2:
                continue

            # Quality gate: ETF above own 200d SMA
            etf_sma = float(col.iloc[-self.stock_trend_window:].mean())
            etf_price = float(col.iloc[-1])
            if etf_price <= etf_sma:
                continue

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
            sharpe = (ret_mean / ret_std) * np.sqrt(ANNUALIZATION)
            if not np.isfinite(sharpe) or sharpe <= 0:
                continue  # require positive trend

            # Inverse-vol weight
            inv_vol_rv = 1.0 / ret_std if ret_std > 1e-6 else 0.0

            scores[sym] = sharpe
            inv_vols[sym] = inv_vol_rv

        target: dict[str, float] = {}

        if len(scores) < 2:
            # No qualifying ETFs — sit in cash (enforce_cash handles this)
            pass
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

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


UNIVERSE = "popular_etfs"

NAME = "etf_momentum_sharpe_voltarget"
HYPOTHESIS = (
    "Popular-ETF top-5 by 63d Sharpe ratio (risk-adjusted momentum) with per-ETF 200d SMA "
    "quality gate; inverse-vol weighted; portfolio 12pct vol-target scales exposure 40-97%; "
    "weekly rebalance — Sharpe-ranked ETF rotation across equity/bond/commodity/sector/factor "
    "ETFs avoids SP500-only concentration, distinct from all gen_10 SP500 strategies"
)

STRATEGY = EtfMomentumSharpeVoltarget()
