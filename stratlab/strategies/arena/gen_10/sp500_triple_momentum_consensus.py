"""SP500 triple-horizon momentum consensus filter.

Hypothesis (sonnet-3, gen_10):
    Only buy SP500 stocks where momentum is POSITIVE across ALL THREE horizons:
      - 21d return > 0  (short-term: recent upward pressure)
      - 63d return > 0  (medium-term: trend continuation)
      - 126d return > 0 (long-term: intermediate momentum)

    Then rank the qualifying stocks by their 126d return and hold top-15
    inverse-vol weighted.

    Rationale:
      - Most momentum strategies rank by a single lookback (126d or 63d) and
        may select stocks that are up over 6 months but declining over 1-3
        months (momentum topping). The triple-horizon consensus filter requires
        all three periods to be positive, ensuring we only buy names in active
        multi-horizon uptrends.
      - This is a FILTERING mechanism (not a scoring change), distinct from
        Sharpe ranking or quality screens (RSI, BB, ADX, Stochastic).
      - The filter eliminates: (a) stocks recovering from a long-term drop but
        in a short-term bounce (false signal); (b) stocks with strong 6-month
        return but now fading; (c) momentum reversals.
      - Regime-invariant: the 21d/63d/126d consensus check works in any VIX
        regime — it only asks whether this specific stock has been rising across
        all horizons.

    Diversification vs leaderboard:
      - gen9_sp500_rsi_quality_momentum: filters by RSI >= 35 (single indicator).
      - gen6_nearhi_momentum_quality: filters by price > 80% of 52w high.
      - gen10_sp500_dual_quality_momentum: RSI + 50d SMA dual filter.
      - This strategy: multi-horizon MOMENTUM CONSENSUS (all three windows
        positive) — entirely different filter mechanism. No indicator overlap.

    Design:
      - Compute 21d, 63d, 126d returns for each SP500 stock.
      - Require all three to be positive AND stock above own 200d SMA.
      - Rank qualifying stocks by 126d return (standard momentum ranking).
      - Hold top-15 inverse-vol weighted (21d realized vol).
      - Portfolio vol-target: 13% annualized (30d realized), clipped 50-97%.
      - SPY 200d SMA outer gate: IEF defensive when bear.
      - Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_SHORT = 21             # 21d short momentum
MOM_MED = 63               # 63d medium momentum
MOM_LONG = 126             # 126d long momentum (ranking window)
STOCK_TREND_WINDOW = 200   # per-stock 200d SMA
VOL_WINDOW = 21            # inverse-vol weight lookback
SPY_TREND_WINDOW = 200     # outer SPY gate
TOP_K = 15
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_TARGET = 0.13          # 13% portfolio vol target
PORT_VOL_WINDOW = 30
ANNUALIZATION = 252


class Sp500TripleMomentumConsensus(Strategy):
    """SP500 top-15 by 126d momentum requiring positive 21d/63d/126d triple consensus;
    per-stock 200d SMA gate; inverse-vol weighted; portfolio vol-target; SPY 200d outer
    gate to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_short: int = MOM_SHORT,
        mom_med: int = MOM_MED,
        mom_long: int = MOM_LONG,
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
            mom_short=mom_short,
            mom_med=mom_med,
            mom_long=mom_long,
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
        self.mom_short = int(mom_short)
        self.mom_med = int(mom_med)
        self.mom_long = int(mom_long)
        self.stock_trend_window = int(stock_trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.stock_trend_window + self.mom_long + self.port_vol_window + 10
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
            # Need enough history for all windows
            need = self.stock_trend_window + self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()

                # Need enough data for all windows
                min_len = self.stock_trend_window + 2
                if len(col) < min_len:
                    continue

                # Quality filter: stock above own 200d SMA
                stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                stock_price = float(col.iloc[-1])
                if stock_price <= stock_sma:
                    continue

                # Triple momentum consensus: all three must be positive
                if len(col) < self.mom_long + 2:
                    continue

                p_now = float(col.iloc[-1])

                # Short: 21d
                if len(col) < self.mom_short + 2:
                    continue
                p_short = float(col.iloc[-self.mom_short - 1])
                if p_short <= 0:
                    continue
                ret_short = p_now / p_short - 1.0
                if ret_short <= 0:
                    continue  # 21d negative — exclude

                # Medium: 63d
                if len(col) < self.mom_med + 2:
                    continue
                p_med = float(col.iloc[-self.mom_med - 1])
                if p_med <= 0:
                    continue
                ret_med = p_now / p_med - 1.0
                if ret_med <= 0:
                    continue  # 63d negative — exclude

                # Long: 126d (also serves as ranking score)
                p_long = float(col.iloc[-self.mom_long - 1])
                if p_long <= 0:
                    continue
                ret_long = p_now / p_long - 1.0
                if ret_long <= 0:
                    continue  # 126d negative — exclude

                if not (np.isfinite(ret_short) and np.isfinite(ret_med) and np.isfinite(ret_long)):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret_long
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Not enough triple-consensus candidates — fall back to IEF
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
                        p_now_v = port_prices[sym].iloc[row_idx]
                        p_prev_v = port_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now_v) and np.isfinite(p_prev_v) and p_prev_v > 0:
                            row_ret += np.log(float(p_now_v) / float(p_prev_v))
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe

NAME = "sp500_triple_momentum_consensus"
HYPOTHESIS = (
    "SP500 top-15 by 126d return from stocks where 21d AND 63d AND 126d returns are ALL positive "
    "(triple-horizon consensus filter); per-stock 200d SMA gate; inverse-vol weighted; portfolio "
    "13pct vol-target exposure; SPY 200d SMA gate; IEF defensive; biweekly rebalance — "
    "triple-horizon consensus ensures only active multi-horizon uptrends are held, distinct from "
    "single-metric quality filters (RSI, BB, Stochastic) on leaderboard"
)

STRATEGY = Sp500TripleMomentumConsensus()
