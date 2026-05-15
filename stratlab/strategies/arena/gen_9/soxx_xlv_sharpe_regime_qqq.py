"""SOXX vs XLV Rolling Sharpe Competition as Growth Regime Gate — gen_9 sonnet-8

Hypothesis:
Use SOXX (semiconductor ETF) vs XLV (healthcare ETF) 63d rolling Sharpe ratio
(return / realized vol) competition as a growth-vs-defensive barometer:
  - When SOXX 63d Sharpe > XLV 63d Sharpe (growth regime, semis outperforming
    healthcare on risk-adjusted basis): hold QQQ 97%
  - When XLV 63d Sharpe >= SOXX 63d Sharpe (defensive preference) AND SPY
    above 200d SMA: hold SPY 60% + TLT 37%
  - SPY below 200d SMA (bear): hold TLT 97%
Rebalance every 5 bars (weekly).

Rationale:
Semiconductors (SOXX) are the most cyclically sensitive growth sector —
their risk-adjusted return relative to healthcare (XLV, a defensive sector)
captures the growth/defensive cycle more cleanly than a simple price comparison.
Healthcare is resilient in recessions; semis are fragile. When semis are
producing better risk-adjusted returns than healthcare, the economy is in
growth mode, and QQQ should capture the tech premium. When healthcare leads
on Sharpe, defensives are preferred. This signal is orthogonal to VIX levels,
credit spreads, yield curves, and consumer sentiment.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
SHARPE_WINDOW = 63        # 63d rolling Sharpe window
TREND_WINDOW = 200        # SPY 200d SMA bear gate
EXPOSURE = 0.97
ANNUALIZE = 252 ** 0.5    # annualize daily Sharpe


class SoxxXlvSharpeRegimeQQQ(Strategy):
    """SOXX/XLV 63d Sharpe ratio competition routing QQQ/SPY+TLT/TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sharpe_window=sharpe_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sharpe_window = int(sharpe_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def _rolling_sharpe(self, close_series: "pd.Series", window: int) -> float:  # type: ignore[name-defined]
        """Compute rolling Sharpe ratio over last `window` bars of daily log returns."""
        if len(close_series) < window + 1:
            return float("nan")
        tail = close_series.iloc[-(window + 1):]
        log_rets = np.log(tail.values[1:] / tail.values[:-1])
        if len(log_rets) == 0:
            return float("nan")
        mean_r = float(np.mean(log_rets))
        std_r = float(np.std(log_rets, ddof=1))
        if std_r <= 1e-10 or not np.isfinite(std_r):
            return float("nan")
        return mean_r / std_r * ANNUALIZE

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.sharpe_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(closes_now[s]) for s in closes_now.index
                if float(closes_now[s]) > 0}

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        if not spy_bull:
            target = {"TLT": self.exposure}
        else:
            # --- SOXX vs XLV Sharpe competition ---
            try:
                soxx_hist = ctx.history("SOXX")
                xlv_hist = ctx.history("XLV")
            except KeyError:
                return []
            if len(soxx_hist) < self.sharpe_window + 2 or len(xlv_hist) < self.sharpe_window + 2:
                return []
            soxx_close = soxx_hist["close"].dropna()
            xlv_close = xlv_hist["close"].dropna()

            soxx_sharpe = self._rolling_sharpe(soxx_close, self.sharpe_window)
            xlv_sharpe = self._rolling_sharpe(xlv_close, self.sharpe_window)

            if not np.isfinite(soxx_sharpe) or not np.isfinite(xlv_sharpe):
                # Fallback: SPY+TLT blend
                target = {}
                if "SPY" in live:
                    target["SPY"] = 0.60
                if "TLT" in live:
                    target["TLT"] = 0.37
            elif soxx_sharpe > xlv_sharpe:
                # Growth regime: QQQ
                target = {"QQQ": self.exposure}
            else:
                # Defensive regime: SPY + TLT blend
                target = {}
                if "SPY" in live:
                    target["SPY"] = 0.60
                if "TLT" in live:
                    target["TLT"] = 0.37

        # --- Build orders ---
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "gen9_soxx_xlv_sharpe_regime_qqq"
HYPOTHESIS = (
    "SOXX vs XLV 63d Sharpe competition as growth regime gate: SOXX Sharpe > "
    "XLV Sharpe -> QQQ 97%; XLV leads AND SPY bull -> SPY 60%+TLT 37%; "
    "SPY bear -> TLT 97%; weekly rebalance."
)

UNIVERSE = ["QQQ", "SPY", "TLT", "SOXX", "XLV"]

STRATEGY = SoxxXlvSharpeRegimeQQQ()
