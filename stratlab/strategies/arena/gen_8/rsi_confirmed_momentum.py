"""SP500 Momentum with RSI Confirmation Filter — gen_8 sonnet-3

Hypothesis: Rank top-20 SP500 stocks by 63d return; require each stock to have
RSI(14)>50 (momentum confirmed, not in overbought exhaustion near RSI>80);
SPY golden cross gate (50d>200d SMA); IEF defensive on death cross;
inverse-vol weighted; biweekly rebalance.

Rationale: Pure momentum ranking picks the highest recent returners but some
of those stocks are already in parabolic overextension. RSI>50 confirms the
stock is in a defined uptrend (buyers winning vs sellers) without being
overbought. This is different from the nearhi_momentum_quality (which uses
52w-high proximity) and existing idiosyncratic momentum (which uses
beta-adjusted returns). The golden cross gate (50d>200d) is a confirmed trend
regime vs the simple 200d SMA level used by most strategies.

IS window: 2010-2018 (9 years).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # Biweekly
MOMENTUM_WINDOW = 63        # ~3 months
RSI_WINDOW = 14             # Standard RSI
RSI_MIN = 50.0              # Must have RSI above 50 (uptrend confirmed)
RSI_MAX = 80.0              # Exclude overbought >80
FAST_MA = 50                # Golden cross fast MA
SLOW_MA = 200               # Golden cross slow MA
VOL_WINDOW = 20             # For inverse-vol weights
TOP_K = 20
EXPOSURE = 0.97


def _compute_rsi(prices: np.ndarray, window: int) -> float:
    """Compute RSI from a price array."""
    if len(prices) < window + 1:
        return 50.0
    deltas = np.diff(prices[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


class RsiConfirmedMomentum(Strategy):
    """SP500 momentum with RSI(14)>50 confirmation; golden cross gate; inv-vol sized."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rsi_window: int = RSI_WINDOW,
        rsi_min: float = RSI_MIN,
        rsi_max: float = RSI_MAX,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rsi_window=rsi_window,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rsi_window = int(rsi_window)
        self.rsi_min = float(rsi_min)
        self.rsi_max = float(rsi_max)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + self.momentum_window + self.rsi_window + 10
        if ctx.idx < warmup:
            return []

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items()
                if not s.startswith("^") and float(p) > 0}

        # Check SPY golden cross (50d SMA > 200d SMA)
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.slow_ma + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.slow_ma:
            return []
        spy_fast_sma = float(spy_close.iloc[-self.fast_ma:].mean())
        spy_slow_sma = float(spy_close.iloc[-self.slow_ma:].mean())
        golden_cross = spy_fast_sma > spy_slow_sma

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not golden_cross:
            # Death cross — defensive
            if "IEF" in live:
                target["IEF"] = self.exposure
            elif "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Golden cross regime — look for RSI-confirmed momentum stocks
            need = max(self.momentum_window, self.slow_ma) + self.rsi_window + 5
            prices_df = ctx.closes_window(need)

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in live:
                if sym in ("SPY", "IEF", "TLT", "SHY"):
                    continue
                if sym not in prices_df.columns:
                    continue
                col = prices_df[sym].dropna()
                need_len = max(self.momentum_window, self.rsi_window) + 5
                if len(col) < need_len:
                    continue

                # Compute RSI(14)
                rsi = _compute_rsi(col.values, self.rsi_window)
                # Filter: RSI must be in (RSI_MIN, RSI_MAX) — confirmed uptrend, not overextended
                if rsi < self.rsi_min or rsi > self.rsi_max:
                    continue

                # 63d momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret) or ret <= 0:
                    continue

                # Inverse-vol weight
                if len(col) < self.vol_window + 2:
                    continue
                tail = col.iloc[-(self.vol_window + 1):]
                log_rets = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(log_rets))
                if rv < 1e-8 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Too few RSI-confirmed stocks; fall back to simple momentum top-k
                scores_fb: dict[str, float] = {}
                inv_vols_fb: dict[str, float] = {}
                for sym in live:
                    if sym in ("SPY", "IEF", "TLT", "SHY"):
                        continue
                    if sym not in prices_df.columns:
                        continue
                    col = prices_df[sym].dropna()
                    if len(col) < self.momentum_window + 2:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.momentum_window])
                    if p_start <= 0:
                        continue
                    ret = p_end / p_start - 1.0
                    if not np.isfinite(ret):
                        continue
                    if len(col) < self.vol_window + 2:
                        continue
                    tail = col.iloc[-(self.vol_window + 1):]
                    log_rets = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(log_rets))
                    if rv < 1e-8:
                        continue
                    scores_fb[sym] = ret
                    inv_vols_fb[sym] = 1.0 / rv
                scores = scores_fb
                inv_vols = inv_vols_fb

            if not scores:
                if "IEF" in live:
                    target["IEF"] = self.exposure
                elif "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
                selected = [sym for sym, _ in ranked]

                iv_sum = sum(inv_vols[s] for s in selected if s in inv_vols)
                if iv_sum <= 0:
                    # Equal weight fallback
                    for sym in selected:
                        target[sym] = self.exposure / len(selected)
                else:
                    for sym in selected:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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
    return sp500_tickers() + ["SPY", "IEF", "TLT", "SHY"]


NAME = "rsi_confirmed_momentum"
HYPOTHESIS = (
    "SP500 momentum with RSI(14) confirmation filter: rank top-20 SP500 stocks by 63d return; "
    "require each stock RSI(14)>50 (momentum confirmed) AND RSI<80 (not overextended); "
    "SPY 50d>200d golden cross gate; IEF defensive on death cross; inverse-vol weighted; "
    "biweekly rebalance; RSI confirmation selects stocks mid-uptrend not at exhaustion"
)

UNIVERSE = _universe

STRATEGY = RsiConfirmedMomentum()
