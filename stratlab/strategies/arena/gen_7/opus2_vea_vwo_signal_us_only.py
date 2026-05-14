"""VEA/VWO global-risk signal driving US-only allocation — gen_7 opus-2.

Hypothesis: International ETFs failed as exposure vehicles in 2010-2018
(USD strength, EM/EFA dragged returns). But the relative strength of VWO
(emerging) vs VEA (developed-international) is still a useful global-risk
signal. EM is more sensitive to USD/global-growth than developed; when EM
outperforms developed, global risk-on. We use this signal to choose between
US equity factor sleeves but never hold the foreign ETFs themselves.

Logic:
  - 60d return of VWO and VEA.
  - VWO_ret > VEA_ret AND SPY > 200d -> SP500 top-10 by 63d momentum (full risk-on).
  - VEA_ret > VWO_ret AND SPY > 200d -> SPY 60% + TLT 37% (USD-strength, defensive blend).
  - SPY < 200d -> TLT 97% (bear regime).
  - Biweekly rebalance.

Distinction: existing intl strategies (gen_6) rotated INTO foreign ETFs and
all failed at MIN_CALMAR_IS=0.5 because intl was in a bear vs SPY. This
strategy uses intl ETFs as SIGNAL ONLY; exposure stays in SPY/TLT/SP500.
This is the gap left when intl-allocation strategies failed: keep the
diagnostic, drop the bad exposure.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
RATIO_WINDOW = 60
STOCK_MOM_WINDOW = 63
TOP_K = 10
TREND_WINDOW = 200
EXPOSURE = 0.97


class VeaVwoSignalUsOnly(Strategy):
    """VEA vs VWO 60d ratio signal -> SP500 top-K momentum or SPY/TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ratio_window: int = RATIO_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ratio_window=ratio_window,
            stock_mom_window=stock_mom_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.ratio_window = int(ratio_window)
        self.stock_mom_window = int(stock_mom_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_mom_window, self.ratio_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d gate
        bull_market = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    bull_market = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # VWO vs VEA 60d returns
        global_risk_on = True
        signal_ok = False
        try:
            vwo_hist = ctx.history("VWO")
            vea_hist = ctx.history("VEA")
            if (vwo_hist is not None and vea_hist is not None
                    and len(vwo_hist) >= self.ratio_window
                    and len(vea_hist) >= self.ratio_window):
                vwo_close = vwo_hist["close"].dropna()
                vea_close = vea_hist["close"].dropna()
                if (len(vwo_close) >= self.ratio_window
                        and len(vea_close) >= self.ratio_window):
                    vwo_ret = float(vwo_close.iloc[-1] / vwo_close.iloc[-self.ratio_window] - 1.0)
                    vea_ret = float(vea_close.iloc[-1] / vea_close.iloc[-self.ratio_window] - 1.0)
                    global_risk_on = (vwo_ret > vea_ret)
                    signal_ok = True
        except Exception:
            pass

        target: dict[str, float] = {}
        if not bull_market:
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif signal_ok and global_risk_on:
            # Full risk-on: SP500 top-K momentum
            need = self.stock_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.stock_mom_window:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.stock_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.stock_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret
                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
                        target[sym] = per_w
        else:
            # USD-strength regime: defensive blend
            for sym, w in [("SPY", 0.60), ("TLT", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure

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
    return sp500_tickers() + ["SPY", "TLT", "VWO", "VEA"]


NAME = "opus2_vea_vwo_signal_us_only"
HYPOTHESIS = (
    "VWO vs VEA 60d ratio as global-risk signal: VWO > VEA AND SPY > 200d hold SP500 top-10 63d "
    "momentum; VEA > VWO (USD-strength) hold SPY 60%+TLT 37%; bear hold TLT; intl ETFs SIGNAL ONLY."
)
UNIVERSE = _universe

STRATEGY = VeaVwoSignalUsOnly()
